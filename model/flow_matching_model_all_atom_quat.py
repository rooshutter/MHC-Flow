import torch
import numpy as np
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch.nn as nn
import torch.nn.functional as F
from openfold.utils.rigid_utils import Rotation
from torch_scatter import scatter_add, scatter_mean

from model.noise_schedule import Noise_Schedule
from utils import create_new_pdb_hdf5, create_new_pdb_hdf5_100k, create_new_pdb_hdf5_swift

from torchdiffeq import odeint_adjoint as odeint

from tools.rigid import Rigid 
from tools.quat import safe_rot_to_quat as safe_rot_to_quat

from openfold.model.primitives import Linear, LayerNorm
from openfold.model.structure_module import AngleResnet, StructureModuleTransition, BackboneUpdate
from openfold.utils.tensor_utils import (
    dict_multimap,
    masked_mean as openfold_masked_mean,
    batched_gather as openfold_batched_gather
)
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_ambiguous_atoms as openfold_restype_atom14_ambiguous_atoms,
    van_der_waals_radius as openfold_van_der_waals_radius,
    atom_types as openfold_atom_types,
    make_atom14_dists_bounds as openfold_make_atom14_dists_bounds,
)
import ml_collections
from openfold.utils.rigid_utils import rot_to_quat, quat_to_rot
from openfold.utils.loss import (
    compute_fape as openfold_compute_fape,
    sidechain_loss as openfold_compute_sidechain_loss,
    compute_renamed_ground_truth as openfold_compute_renamed_ground_truth,
    between_residue_clash_loss as openfold_between_residue_clash_loss,
    within_residue_violations as openfold_within_residue_violations,
    between_residue_bond_loss as openfold_between_residue_bond_loss,
)
from openfold.data.data_transforms import (
    atom37_to_frames as openfold_atom37_to_frames,
    make_atom14_masks as openfold_make_atom14_masks
)
from openfold.config import config as openfold_config


from rvf.manifolds.sphere import SphereManifold

from scipy.spatial.transform import Rotation as SciPy_Rotation
from ReQFlow.so3_utils import quaternion_slerp_exp, calc_quat_wt_qt_q1, calc_rot_vf
from ReQFlow.so3_utils import rotmat_to_rotvec, rotvec_to_rotmat, local_log, vector_to_skew_matrix


_regression_loss_function = torch.nn.HuberLoss(reduction="none", delta=1.0)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        SciPy_Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _compute_torsion_angle_loss( # from swiftmhc
    a: torch.Tensor,
    a_mask: torch.Tensor,
    a_gt: torch.Tensor,
    a_alt_gt: torch.Tensor,
) -> torch.Tensor:
    """
    Torsion angle loss, according to alphafold Algorithm 27
    This code was copied from openfold and modified.
    The original torsion_angle_loss function is at:
    https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/loss.py

    Args:
        a:          [*, N, 7, 2] predicted torsion angles sin, cos
        a_mask:     [*, N, 7] (bool) torsion angles mask
        a_gt:       [*, N, 7, 2] true torsion angles sin, cos
        a_alt_gt:   [*, N, 7, 2] true alternative torsion angles sin, cos

    Returns:
        [*] losses per case
    """

    # [*, N, 7]
    norm = torch.sqrt(torch.sum(a**2, dim=-1) + 1e-12)
    # norm = torch.clamp(norm, min=1e-6)

    # [*, N, 7, 2]
    a = a / norm.unsqueeze(-1)
    
    # [*, N, 7]
    diff_norm_gt = torch.sqrt(torch.sum((a - a_gt)**2, dim=-1) + 1e-12)
    diff_norm_alt_gt = torch.sqrt(torch.sum((a - a_alt_gt)**2, dim=-1) + 1e-12)
    min_diff = torch.minimum(diff_norm_gt ** 2, diff_norm_alt_gt ** 2)

    # [*]
    l_torsion = openfold_masked_mean(a_mask, min_diff, dim=(-2, -1))
    l_angle_norm = openfold_masked_mean(a_mask, torch.abs(norm - 1), dim=(-2, -1))

    an_weight = 0.02
    return l_torsion + an_weight * l_angle_norm

def _compute_fape_loss( # from swiftmhc
    v_hat_mol_quat, v_hat_mol_trans, T_peptide, molecule, protein_pocket, positions, sidechain_frames
): 
    """
    Compute FAPE loss as in openfold

    Returns:
        backbone:   [*] backbone FAPE
        sidechain:  [*] sidechain FAPE
        total:      [*] backbone FAPE + sidechain FAPE
    """

    # compute backbone FAPE
    protein_frames = Rigid.from_tensor_4x4(protein_pocket['backbone_rigid_tensor'])
    protein_mask = protein_pocket['cross_residues_mask'].view(protein_frames.shape[0], -1)
    peptide_output_frames = Rigid.from_tensor_7(torch.cat((v_hat_mol_quat, v_hat_mol_trans.view(v_hat_mol_quat.shape[0], v_hat_mol_quat.shape[1], -1)), dim=-1))
    
    peptide_true_frames = Rigid.from_tensor_7(T_peptide)
    peptide_mask = molecule['cross_residues_mask'].view(peptide_output_frames.shape[0], -1)

    bb_loss = torch.mean(
        openfold_compute_fape(
            pred_frames=protein_frames,
            target_frames=protein_frames,
            frames_mask=protein_mask,
            pred_positions=peptide_output_frames.get_trans(),
            target_positions=peptide_true_frames.get_trans(),
            positions_mask=peptide_mask,
            length_scale=10.0,
            l1_clamp_distance=10.0,
            eps=1e-4,
        )
    )

    # Find out which atoms are ambiguous.
    peptide_aatype = molecule['aatype'].reshape(-1, 9)
    atom14_atom_is_ambiguous = torch.tensor(openfold_restype_atom14_ambiguous_atoms[peptide_aatype.cpu().numpy()],
                                            device=peptide_aatype.device)
    molecule_atom14_gt_positions = molecule['x'].reshape(atom14_atom_is_ambiguous.shape[0], atom14_atom_is_ambiguous.shape[1], molecule['x'].shape[-2], molecule['x'].shape[-1])
    molecule_atom14_alt_gt_positions = molecule['atom14_alt_gt_positions'].reshape(atom14_atom_is_ambiguous.shape[0], atom14_atom_is_ambiguous.shape[1], molecule['atom14_alt_gt_positions'].shape[-2], molecule['atom14_alt_gt_positions'].shape[-1])
    molecule_atom14_gt_exists = molecule['atom14_gt_exists'].reshape(atom14_atom_is_ambiguous.shape[0], atom14_atom_is_ambiguous.shape[1], molecule['atom14_gt_exists'].shape[-1])

    positions = positions.reshape(atom14_atom_is_ambiguous.shape[0], atom14_atom_is_ambiguous.shape[1], positions.shape[-2], positions.shape[-1])

    renamed_truth = openfold_compute_renamed_ground_truth({
                                                            "atom14_gt_positions": molecule_atom14_gt_positions,
                                                            "atom14_alt_gt_positions": molecule_atom14_alt_gt_positions,
                                                            "atom14_gt_exists": molecule_atom14_gt_exists.float(),
                                                            "atom14_atom_is_ambiguous": atom14_atom_is_ambiguous,
                                                            "atom14_alt_gt_exists": molecule_atom14_gt_exists.float(),
                                                          },
                                                          positions)


    # Get the truth frames and alternative truth frames from the true atom positions,
    # This involves converting from 14-atoms to 37-atoms format.
    peptide_data = openfold_make_atom14_masks({"aatype": peptide_aatype})
    peptide_residx_atom37_to_atom14 = peptide_data["residx_atom37_to_atom14"]
    atom37_positions = openfold_batched_gather(
        molecule_atom14_gt_positions,
        peptide_residx_atom37_to_atom14,
        dim=-2,
        no_batch_dims=len(molecule_atom14_gt_positions.shape[:-2]),
    )
    atom37_mask = openfold_batched_gather(
        molecule_atom14_gt_exists,
        peptide_residx_atom37_to_atom14,
        dim=-1,
        no_batch_dims=len(molecule_atom14_gt_exists.shape[:-1]),
    )
    truth_frames = openfold_atom37_to_frames(
        {
            "aatype": peptide_aatype,
            "all_atom_positions": atom37_positions,
            "all_atom_mask": atom37_mask,
        }
    )

    # compute the actual sidechain FAPE
    sc_loss = openfold_compute_sidechain_loss(sidechain_frames=sidechain_frames[None, ...],
                                              sidechain_atom_pos=positions[None, ...],
                                              rigidgroups_gt_frames=truth_frames["rigidgroups_gt_frames"],
                                              rigidgroups_alt_gt_frames=truth_frames["rigidgroups_alt_gt_frames"],
                                              rigidgroups_gt_exists=truth_frames["rigidgroups_gt_exists"],

                                              renamed_atom14_gt_positions=renamed_truth["renamed_atom14_gt_positions"],
                                              renamed_atom14_gt_exists=renamed_truth["renamed_atom14_gt_exists"],
                                              alt_naming_is_better=renamed_truth["alt_naming_is_better"],
                                              **openfold_config.loss.fape.sidechain)
    total_loss = 0.5 * bb_loss + 0.5 * sc_loss

    return {
        "total": total_loss,
        "backbone": bb_loss,
        "sidechain": sc_loss,
    }

def _compute_cross_violation_loss( #from swiftmhc
    molecule, protein_pocket, positions
):
    """
    Compute violations in the predicted structure.
    Returns:
        bond:                       [*] bond length violations between residues within the peptide
        CA-C-N-angles:              [*] C-alpha-C-N angle violations in peptide
        C-N-CA-angles:              [*] C-N-C-alpha angle violations in peptide
        between-residues-clash:     [*] clashes between residues from protein and peptide
        within-residues-clash:      [*] clashes between atoms within peptide residues
    """

    # Reshape batched inputs to [Batch, N_res, ...]
    batch_size = molecule["aatype"].shape[0] // 9
    peptide_len = 9
    protein_len = 180

    peptide_aatype = molecule["aatype"].reshape(batch_size, peptide_len)
    protein_aatype = protein_pocket["aatype"].reshape(batch_size, protein_len)

    # Compute the between residue clash loss. (include both peptide and protein)
    # [*, peptide_maxlen + protein_maxlen, 14]
    peptide_data = openfold_make_atom14_masks({"aatype": peptide_aatype})
    protein_data = openfold_make_atom14_masks({"aatype": protein_aatype})

    residx_atom14_to_atom37 = torch.cat((peptide_data["residx_atom14_to_atom37"],
                                         protein_data["residx_atom14_to_atom37"]), dim=1)

    positions_reshaped = positions.reshape(batch_size, peptide_len, 14, 3)
    protein_x_reshaped = protein_pocket["x"].reshape(batch_size, protein_len, 14, 3)

    # [*, peptide_maxlen + protein_maxlen, 14, 3]
    atom14_pred_positions = torch.cat((positions_reshaped,
                                       protein_x_reshaped), dim=1)

    pep_exists = molecule["atom14_gt_exists"].reshape(batch_size, peptide_len, 14)
    restype_mask_tensor = torch.tensor(restype_atom14_mask, device=protein_aatype.device, dtype=pep_exists.dtype)
    prot_exists = restype_mask_tensor[protein_aatype]

    # [*, peptide_maxlen + protein_maxlen, 14]
    atom14_atom_exists = torch.cat((pep_exists, prot_exists), dim=1)

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # [37]
    atomtype_radius = [
        openfold_van_der_waals_radius[name[0]]
        for name in openfold_atom_types
    ]
    # [37]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)

    # [*, peptide_maxlen + protein_maxlen, 14]
    atom14_atom_radius = atom14_atom_exists * atomtype_radius[residx_atom14_to_atom37]

    # [*, peptide_maxlen]
    peptide_residue_index = molecule["residue_index"].reshape(batch_size, peptide_len)
    # [*, protein_maxlen]
    protein_residue_index = protein_pocket["residue_index"].reshape(batch_size, protein_len)

    # [*, peptide_maxlen + protein_maxlen]
    residue_index = torch.cat((peptide_residue_index, protein_residue_index), dim=1)

    between_residue_clashes = openfold_between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=residue_index,
        overlap_tolerance_soft=openfold_config.loss.violation.clash_overlap_tolerance,
        overlap_tolerance_hard=openfold_config.loss.violation.clash_overlap_tolerance,
    )

    # Compute all within-residue violations: clashes, bond length and angle violations.
    # (only within peptide)
    restype_atom14_bounds = openfold_make_atom14_dists_bounds(
        overlap_tolerance=openfold_config.loss.violation.clash_overlap_tolerance,
        bond_length_tolerance_factor=openfold_config.loss.violation.violation_tolerance_factor,
    )
    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["lower_bound"]
    )[peptide_aatype]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["upper_bound"]
    )[peptide_aatype]

    residue_violations = openfold_within_residue_violations(
        atom14_pred_positions=positions_reshaped,
        atom14_atom_exists=pep_exists,
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    # Compute between residue backbone violations of bonds and angles.
    connection_violations = openfold_between_residue_bond_loss(
        pred_atom_positions=positions_reshaped,
        pred_atom_mask=pep_exists,
        residue_index=peptide_residue_index,
        aatype=peptide_aatype,
        tolerance_factor_soft=openfold_config.loss.violation.violation_tolerance_factor,
        tolerance_factor_hard=openfold_config.loss.violation.violation_tolerance_factor,
    )

    # [*]
    violations_between_residues_bonds_c_n_loss_mean = connection_violations["c_n_loss_mean"]
    violations_between_residues_angles_ca_c_n_loss_mean = connection_violations["ca_c_n_loss_mean"]
    violations_between_residues_angles_c_n_ca_loss_mean = connection_violations["c_n_ca_loss_mean"]

    # [*, peptide_len + protein_len, 14]
    violations_between_residues_clashes_per_atom_loss_sum = between_residue_clashes["per_atom_loss_sum"]

    # [*, peptide_len, 14]
    violations_within_residues_per_atom_loss_sum = residue_violations["per_atom_loss_sum"]

    # Calculate loss, as in openfold
    peptide_num_atoms = torch.sum(pep_exists)

    # [*]
    between_residues_clash = torch.sum(violations_between_residues_clashes_per_atom_loss_sum) / (openfold_config.loss.violation.eps + peptide_num_atoms)
    within_residues_clash = torch.sum(violations_within_residues_per_atom_loss_sum) / (openfold_config.loss.violation.eps + peptide_num_atoms)

    # [*]
    loss = {
        "bond": violations_between_residues_bonds_c_n_loss_mean,
        "CA-C-N-angles": violations_between_residues_angles_ca_c_n_loss_mean,
        "C-N-CA-angles": violations_between_residues_angles_c_n_ca_loss_mean,
        "between-residues-clash": between_residues_clash,
        "within-residues-clash": within_residues_clash,
        "total": (violations_between_residues_bonds_c_n_loss_mean +
                  violations_between_residues_angles_ca_c_n_loss_mean +
                  violations_between_residues_angles_c_n_ca_loss_mean +
                  between_residues_clash +
                  within_residues_clash)
    }

    return loss



class Flow_Matching_Model_all_atom(nn.Module):
    def __init__(
        self,
        neural_net: nn.Module,
        features_fixed: bool,
        confidence_score: bool,
        timesteps: int,
        position_encoding: bool,
        com_handling: str,
        sampling_stepsize: int,
        noise_scaling: int,
        high_noise_training: bool,
        num_atoms: int,
        num_residues: int,
        norm_values: list,
        all_atom = False,
        variational: bool = True,
        solver: str = "euler",
        ba: bool = False,

    ):
        super().__init__()

        self.neural_net = neural_net
        self.T = timesteps
        self.features_fixed = features_fixed
        self.position_encoding = position_encoding
        self.com_handling = com_handling
        self.sampling_stepsize = sampling_stepsize

        # dataset info
        self.num_atoms = num_atoms
        self.num_residues = num_residues
        self.norm_values = norm_values
        self.x_dim = 3
        self.rot_dim = 4  # quaternion representation
        self.angle_dim = 14
        self.eps = 1e-7

        # Noise Schedule
        self.noise_schedule = Noise_Schedule(self.T)

        # Further model hyperparameters
        if noise_scaling == None:
            self.noise_scaling = 0
        else:
            self.noise_scaling = noise_scaling
        self.high_noise_training = high_noise_training

        self.confidence_score = confidence_score
        self.all_atom = all_atom
        self.variational = variational
        self.solver = solver
        self.ba = ba

    def _center_inputs(self, molecule, protein_pocket):
        """
        Centers both peptide and protein coordinates based on the peptide's 
        backbone Center of Mass (COM). This ensures translational invariance
        and consistent coordinate systems for all-atom reconstruction and metrics.
        """
        # Calculate mean from peptide backbone CA positions
        # backbone_rigid_tensor can be 4x4 [N, 4, 4] or 7-dim [N, 7]
        # In this model, it's typically [Total_N, 4, 4] after collation
        if molecule['backbone_rigid_tensor'].shape[-1] == 4:
            ca_pos = molecule['backbone_rigid_tensor'][:, :3, 3] # [Total_N, 3]
        else:
            ca_pos = molecule['backbone_rigid_tensor'][:, 4:7] # [Total_N, 3]
            
        mean = scatter_mean(ca_pos, molecule['idx'], dim=0) # [Batch, 3]
        
        # Shift all peptide coordinates
        molecule['x'] = molecule['x'] - mean[molecule['idx']].view(-1, 1, 3)
        if 'atom14_gt_exists' in molecule:
            molecule['x'] = molecule['x'] * molecule['atom14_gt_exists'].to(molecule['x'].dtype).unsqueeze(-1)
          
        if molecule['backbone_rigid_tensor'].shape[-1] == 4:
            molecule['backbone_rigid_tensor'][:, :3, 3] -= mean[molecule['idx']]
        else:
            molecule['backbone_rigid_tensor'][:, 4:7] -= mean[molecule['idx']]
            
        if 'atom14_alt_gt_positions' in molecule:
            molecule['atom14_alt_gt_positions'] = molecule['atom14_alt_gt_positions'] - mean[molecule['idx']].view(-1, 1, 3)
            if 'atom14_gt_exists' in molecule:
                molecule['atom14_alt_gt_positions'] = molecule['atom14_alt_gt_positions'] * molecule['atom14_gt_exists'].to(molecule['x'].dtype).unsqueeze(-1)

        #TODO check 0's in positions

        # Shift all protein coordinates by the SAME mean
        protein_pocket['x'] = protein_pocket['x'] - mean[protein_pocket['idx']].view(-1, 1, 3)
        if protein_pocket['backbone_rigid_tensor'].shape[-1] == 4:
            protein_pocket['backbone_rigid_tensor'][:, :3, 3] -= mean[protein_pocket['idx']]
        else:
            protein_pocket['backbone_rigid_tensor'][:, 4:7] -= mean[protein_pocket['idx']]

        return mean

    def forward(self, z_data, current_epoch=None, max_epochs=None, run_id=None, data_dir=None, save_pdb=False):

        molecule, protein_pocket = z_data
        
        if self.com_handling != 'no_COM':
            self._center_inputs(molecule, protein_pocket)

        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None

        z_t_mol, z_t_pro, v_mol, v_pro, t = self.compute_flow_match(z_data)
    
        if self.noise_scaling > 0:
            z_t_mol['rot'] = z_t_mol['rot'] + torch.randn_like(z_t_mol['rot']) * self.noise_scaling
            z_t_mol['rot'] = F.normalize(z_t_mol['rot'], dim=-1, eps=1e-6)
            z_t_mol['trans'] = z_t_mol['trans'] + torch.randn_like(z_t_mol['trans']) * self.noise_scaling
            z_t_mol['angles'] = z_t_mol['angles'] + torch.randn_like(z_t_mol['angles']) * self.noise_scaling

        mask = molecule['cross_residues_mask'].reshape(molecule['h'].shape[0], molecule['h'].shape[1])

        if self.ba:
            z_1_mol, z_1_pro, _, _, t_1 = self.compute_flow_match(z_data, t_is_1=True)
            z_1_mol_cat = torch.cat((z_1_mol['rot'], z_1_mol['trans'], z_1_mol['angles'], molecule['h']), dim=-1)
            _, _, _, ba_hat = self.neural_net(z_1_mol_cat, z_1_pro, t_1, molecule['idx'], protein_pocket['idx'], molecule_pos, molecule['torsion_angles_mask'], mask)
        else:
            ba_hat = None

        z_t_mol_cat = torch.cat((z_t_mol['rot'], z_t_mol['trans'], z_t_mol['angles'], molecule['h']), dim=-1)
        v_hat_mol, v_hat_pro, c_s, _ = self.neural_net(z_t_mol_cat, z_t_pro, t, molecule['idx'], protein_pocket['idx'], molecule_pos, molecule['torsion_angles_mask'], mask)

        if self.training:
            loss, info = self.train_loss(molecule, z_t_mol, v_mol, 
                                        v_hat_mol, protein_pocket, 
                                        z_t_pro, v_pro, v_hat_pro, t, c_s, ba_hat, current_epoch, max_epochs,
                                        run_id=run_id, data_dir=data_dir, save_pdb=save_pdb)
        else: 
            loss, info = self.validation_loss(z_data, molecule, z_t_mol, v_mol, 
                                        v_hat_mol, protein_pocket, 
                                        z_t_pro, v_pro, v_hat_pro, t, ba_hat, current_epoch, max_epochs,
                                        run_id=run_id, data_dir=data_dir, save_pdb=save_pdb)

        return loss.mean(0), info
    
    def compute_flow_match(self, z_data, t_is_1 = False):
        molecule, protein_pocket = z_data
        batch_size = molecule['size'].size(0)
        device = molecule['x'].device
                
        size_mol = molecule['size'][0]
        size_pro = protein_pocket['size'][0]

        if molecule['h'].shape[0] != batch_size:
            molecule['h'] = molecule['h'].view(batch_size, size_mol, *molecule['h'].shape[1:])
        if protein_pocket['h'].shape[0] != batch_size:
            protein_pocket['h'] = protein_pocket['h'].view(batch_size, size_pro, *protein_pocket['h'].shape[1:])
        if molecule['torsion_angles_sin_cos'].shape[0] != batch_size:
            molecule['torsion_angles_sin_cos'] = molecule['torsion_angles_sin_cos'].view(batch_size, size_mol, *molecule['torsion_angles_sin_cos'].shape[1:])
        if molecule['torsion_angles_mask'].shape[0] != batch_size:
            molecule['torsion_angles_mask'] = molecule['torsion_angles_mask'].view(batch_size, size_mol, *molecule['torsion_angles_mask'].shape[1:])
        if molecule['backbone_rigid_tensor'].shape[0] != batch_size:
            molecule['backbone_rigid_tensor'] = molecule['backbone_rigid_tensor'].view(batch_size, size_mol, *molecule['backbone_rigid_tensor'].shape[1:])
        if protein_pocket['backbone_rigid_tensor'].shape[0] != batch_size:
            protein_pocket['backbone_rigid_tensor'] = protein_pocket['backbone_rigid_tensor'].view(batch_size, size_pro, *protein_pocket['backbone_rigid_tensor'].shape[1:])
    
        # molecule['x'] = molecule['x'] / self.norm_values[0]
        molecule['h'] = molecule['h'] / self.norm_values[1]
        # protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]

        # sample t ~ U(0,...,T) for each graph individually
        t_low = 0 if self.training else 1
        t = torch.randint(t_low, self.T + 1, size=(batch_size, 1, 1), device=device)
        
        # normalize t
        t = t / self.T

        t = torch.ones((batch_size, 1, 1), device=device) if t_is_1 else t
        
        # Target angles
        angles = molecule['torsion_angles_sin_cos']
        angles = angles.view(batch_size, size_mol, -1)

        # Target rotation and translation of peptide
        peptide_backbone_rigid_tensor = molecule['backbone_rigid_tensor']
        T_peptide = Rigid.from_tensor_4x4(peptide_backbone_rigid_tensor)
        T_peptide = T_peptide.to_tensor_7()
        quat_peptide = T_peptide[:,:,:4]
        quat_peptide = quat_peptide.view(-1, quat_peptide.shape[-1])
        rot_peptide = quat_peptide.view(T_peptide.shape[0], T_peptide.shape[1], 4)
        trans_peptide = T_peptide[:,:,4:]
        T_peptide = torch.cat((rot_peptide, trans_peptide), dim=-1)
        
        xh_mol = torch.cat((T_peptide, angles, molecule['h']), dim=-1)

        # Rotation and translation of protein
        protein_backbone_rigid_tensor = protein_pocket['backbone_rigid_tensor']
        T_protein = Rigid.from_tensor_4x4(protein_backbone_rigid_tensor)
        T_protein = T_protein.to_tensor_7()
        quat_protein = T_protein[:,:,:4]
        quat_protein = quat_protein.view(-1, quat_protein.shape[-1])
        rot_protein = quat_protein.view(T_protein.shape[0], T_protein.shape[1], 4)
        trans_protein = T_protein[:,:,4:]
        T_protein = torch.cat((rot_protein, trans_protein), dim=-1)

        xh_pro = torch.cat((T_protein, protein_pocket['h']), dim=-1)

        # Get rotation and translation noise
        z_trans = torch.randn((*molecule['h'].shape[:-1], 3), device=device)
        z_trans = z_trans - scatter_mean(z_trans.view(-1, 3), molecule['idx'], dim=0)[molecule['idx']].view(z_trans.shape)
        rotmats_0 = _uniform_so3(molecule['h'].shape[0], molecule['h'].shape[1], device)
        rotquats_0 = safe_rot_to_quat(rotmats_0)
        rotquats_0_flat = rotquats_0.view(rotquats_0.shape[0], rotquats_0.shape[1], 4)
        T_peptide_z = torch.cat((rotquats_0_flat, z_trans), dim=-1)

        # Get torsion angle noise
        random_angles = torch.rand((molecule['h'].shape[0], molecule['h'].shape[1], 7), device=device) * 2 * math.pi
        sin_angles = torch.sin(random_angles)
        cos_angles = torch.cos(random_angles)
        angles_z = torch.stack((sin_angles, cos_angles), dim=-1)

        z_x_mol = torch.cat((T_peptide_z, angles_z.reshape(angles_z.shape[0], angles_z.shape[1], -1)), dim=-1)

        z_x_pro = torch.zeros(size=(len(xh_pro), xh_pro.shape[1], T_peptide.shape[-1]), device=device)

        if self.features_fixed:
            z_h_mol = torch.zeros(size=(len(xh_mol), xh_mol.shape[1], self.num_atoms), device=device)
            z_h_pro = torch.zeros(size=(len(xh_pro), xh_pro.shape[1], self.num_residues), device=device)
        else:
            z_h_mol = torch.randn(size=(len(xh_mol), xh_mol.shape[1], self.num_atoms), device=device)
            z_h_pro = torch.randn(size=(len(xh_pro), xh_pro.shape[1], self.num_residues), device=device)

        z_pro = torch.cat((z_x_pro, z_h_pro), dim=-1)

        # Interpolate rotation
        quat_peptide = quat_peptide.reshape(T_peptide.shape[0], T_peptide.shape[1], 4)
        dot = torch.sum(quat_peptide * rotquats_0, dim=-1, keepdim=True)
        rotquats_0 = torch.where(dot < 0, -rotquats_0, rotquats_0)
        rotquats_t = quaternion_slerp_exp(t.squeeze(-1).expand(-1, T_peptide.shape[1]), quat_peptide, rotquats_0) 
        rotquats_t = rotquats_t.view(-1, rotquats_t.shape[-1])
        rotquats_t_flat = rotquats_t.view(T_peptide.shape[0], T_peptide.shape[1], 4)

        # Interpolate translation
        T_peptide_t = (1 - t) * T_peptide_z + t * xh_mol[:, :, :T_peptide_z.shape[-1]] 

        T_peptide_t = torch.cat((rotquats_t_flat, T_peptide_t[:,:,rotquats_t_flat.shape[-1]:]), dim=-1)

        # Interpolate angles
        angle_mask = molecule['torsion_angles_mask']
        angles_z = angles_z * angle_mask.unsqueeze(-1)
        angles_z = angles_z.view(batch_size, size_mol, -1)
        angles_z = angles_z.reshape(batch_size, size_mol, 7, 2)
        angles = angles.view(batch_size, size_mol, 7, 2)
        alpha_0 = torch.atan2(angles_z[:, :, :, 0], angles_z[:, :, :, 1])
        alpha_1 = torch.atan2(angles[:, :, :, 0], angles[:, :, :, 1])
        diff = alpha_1 - alpha_0
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        alpha_t = alpha_0 + t * diff
        angles_t = torch.stack([torch.sin(alpha_t), torch.cos(alpha_t)], dim=-1)
        angles_t = angles_t * angle_mask.unsqueeze(-1)
        angles_t = angles_t.reshape(batch_size, size_mol, -1)
        
        z_t_mol = {
            "rot": rotquats_t_flat,
            "trans": T_peptide_t[:,:,self.rot_dim:],
            "angles": angles_t,
            "h": molecule['h']
        }
        z_t_pro = xh_pro.clone().detach()

        # Velocity
        one_minus_t = torch.clamp(1 - t, min=1.0 / self.T)

        xh_mol_rot = xh_mol[:,:,:self.rot_dim].reshape(-1, 4) 
        z_x_mol_rot = z_x_mol[:,:,:self.rot_dim].reshape(-1, 4)
        xh_mol_rotmat = quat_to_rot(xh_mol_rot)  
        z_t_mol_rot = z_t_mol["rot"].reshape(-1, 4)
        z_t_mol_rotmat = quat_to_rot(z_t_mol_rot) 
        rot_rel = torch.matmul(z_t_mol_rotmat.transpose(-2, -1), xh_mol_rotmat)
        v_rot_mol = rotmat_to_rotvec(rot_rel).reshape(batch_size, size_mol, -1)
        v_rot_mol_vec = v_rot_mol / one_minus_t
        v_mol_rot = vector_to_skew_matrix(v_rot_mol_vec)

        v_mol_quat = calc_quat_wt_qt_q1(rotquats_t.reshape(batch_size, size_mol, -1), quat_peptide)
        v_mol_quat = v_mol_quat / one_minus_t

        a_1 = xh_mol[:,:,self.rot_dim:self.rot_dim+self.angle_dim].view(batch_size, size_mol, 7, 2)
        a_t = z_t_mol['angles'].view(batch_size, size_mol, 7, 2)
        cos_1, sin_1 = a_1[..., 0], a_1[..., 1]
        cos_t, sin_t = a_t[..., 0], a_t[..., 1]
        sin_diff = sin_1 * cos_t - cos_1 * sin_t
        cos_diff = cos_1 * cos_t + sin_1 * sin_t
        geodesic_distance = torch.atan2(sin_diff, cos_diff)
        v_mol_angles = geodesic_distance / one_minus_t
        
        v_mol_trans = (xh_mol[:,:,self.rot_dim:self.rot_dim+self.x_dim] - z_t_mol['trans']) / one_minus_t
        
        v_mol = {
            "rot": v_mol_rot,
            "trans": v_mol_trans,
            "angles": v_mol_angles,
            "quat": v_mol_quat
        }
        v_pro = xh_pro - z_pro

        return z_t_mol, z_t_pro, v_mol, v_pro, t
    
    def predict_pos(self, molecule, T_peptide, angles): 
        """
        Predict the atom positions from the backbone frames and angles
        """

        if len(T_peptide.shape) == 2:
            T_peptide = T_peptide.view(-1, 9, *T_peptide.shape[1:])

        T_peptide = Rigid.from_tensor_7(T_peptide)
        
        backb_to_global = Rigid(
            Rotation(
                rot_mats=T_peptide.get_rots().get_rot_mats(),
                quats=None
            ),
            T_peptide.get_trans(),
        )

        angles = angles.view(-1, 9, 7, 2)
        peptide_aatype = molecule['aatype']
        peptide_aatype = peptide_aatype.view(-1, 9, *peptide_aatype.shape[1:])
            
        all_frames_to_global = self.torsion_angles_to_frames(
            backb_to_global,
            angles,
            peptide_aatype,
        )

        pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            peptide_aatype,
        )

        peptide_mask = molecule['cross_residues_mask'].view(pred_xyz.shape[0], -1, *molecule['cross_residues_mask'].shape[1:])
        post_omegas_from_xyz = self._calculate_omegas_from_positions(pred_xyz, peptide_mask)
        last_omega = post_omegas_from_xyz.new_tensor([0.0, -1.0])  # sine 0, cosine -1 : 180 degrees
        last_omega = last_omega.unsqueeze(0).expand(post_omegas_from_xyz.shape[0], -1).unsqueeze(1)
        omegas = torch.cat([post_omegas_from_xyz, last_omega], dim=-2)
        angles = torch.cat([omegas.unsqueeze(-2), angles[..., 1:, :]], dim=-2)

        return pred_xyz, angles, all_frames_to_global.to_tensor_4x4() 

    def train_loss(
            self, molecule, z_t_mol, v_mol, 
            v_hat_mol, protein_pocket, 
            z_t_pro, v_pro, v_hat_pro, t, c_s, ba_hat, current_epoch=None, max_epochs=None,
            run_id=None, data_dir=None, save_pdb=False
    ):
        print(' ')
        batch_size = len(molecule['affinity'])
        size_mol = molecule['size'][0]

        if self.ba:
            ba = molecule['affinity']
            ba_hat = ba_hat.squeeze(-1)
            error_ba = torch.mean((ba - ba_hat) ** 2, dim=-1)
            print(f"BA RMSE mean: {torch.sqrt(error_ba.mean() + 1e-8)}")

            print(f"{ba_hat[0]=}, {ba[0]=}")
            affinity_loss = _regression_loss_function(ba_hat.float(), ba.float())
            print(f"BA Loss: {affinity_loss.mean()}")
        else:
            error_ba = 0
            affinity_loss = 0

        if self.variational:
            T_peptide = Rigid.from_tensor_4x4(molecule['backbone_rigid_tensor']).to_tensor_7()
            v_mol_quat = T_peptide[:,:,:4]
            v_mol_trans = T_peptide[:,:,4:7]
            v_mol_angles = molecule['torsion_angles_sin_cos']
        else:
            v_mol_rot = v_mol['rot']
            v_mol_quat = v_mol['quat']
            v_mol_trans = v_mol['trans']
            v_mol_angles = v_mol['angles']

        v_hat_mol = torch.clamp(v_hat_mol, min=-1000.0, max=1000.0) 
        
        # Calculate peptide positions 
        v_hat_mol_quat = v_hat_mol[:,:self.rot_dim] 
        v_hat_mol_quat = F.normalize(v_hat_mol_quat, dim=-1, eps=1e-6)
        v_hat_mol_trans = v_hat_mol[:,self.rot_dim:self.rot_dim+self.x_dim]
        v_hat_mol_angles = v_hat_mol[:,self.rot_dim+self.x_dim:self.rot_dim+self.x_dim+self.angle_dim]

        x_hat_mol, v_hat_mol_angles, sidechain_frames = self.predict_pos(molecule, torch.concat([v_hat_mol_quat, v_hat_mol_trans], dim=-1), v_hat_mol_angles)
        
        if not self.variational:
            one_minus_t = torch.clamp(1 - t, min=1.0 / self.T)

            v_hat_mol_quat = v_hat_mol[:,:self.rot_dim] 
            v_hat_mol_quat = F.normalize(v_hat_mol_quat, dim=-1, eps=1e-6)
            quat_hat_mol = v_hat_mol_quat.reshape(batch_size, size_mol, 4)
            z_t_mol_quat = z_t_mol['rot'].reshape(-1, 4)  
            v_hat_mol_quat = calc_quat_wt_qt_q1(z_t_mol_quat.reshape(batch_size, size_mol, 4), v_hat_mol_quat.reshape(batch_size, size_mol, 4)) 
            v_hat_mol_quat = v_hat_mol_quat / one_minus_t
            v_hat_mol_quat = v_hat_mol_quat.reshape(-1, 3) 

            trans_hat_mol = v_hat_mol[:,self.rot_dim:self.rot_dim+self.x_dim].reshape(batch_size, size_mol, 3)
            v_hat_mol_trans = (trans_hat_mol - z_t_mol['trans']) / one_minus_t
            v_hat_mol_trans = v_hat_mol_trans.reshape(-1, 3)

            angles_hat_mol = v_hat_mol[:,self.rot_dim+self.x_dim:self.rot_dim+self.x_dim+self.angle_dim].view(batch_size, size_mol, 7, 2)
            a_t = z_t_mol['angles'].view(batch_size, size_mol, 7, 2)
            
            cos_1, sin_1 = angles_hat_mol[..., 0], angles_hat_mol[..., 1]
            cos_t, sin_t = a_t[..., 0], a_t[..., 1]
            sin_diff = sin_1 * cos_t - cos_1 * sin_t
            cos_diff = cos_1 * cos_t + sin_1 * sin_t

            geodesic_distance = torch.atan2(sin_diff, cos_diff)
            v_hat_mol_angles = geodesic_distance / one_minus_t
            v_hat_mol_angles = v_hat_mol_angles.reshape(-1, 7) 
        
        # Calculate rotation loss
        if self.variational:
            v_hat_mol_quat = v_hat_mol_quat.reshape(batch_size, size_mol, v_hat_mol_quat.shape[-1])
            v_hat_mol_rot_proj = quat_to_rot(v_hat_mol_quat).reshape(-1, 3, 3)
            gt_rot = quat_to_rot(v_mol_quat.reshape(-1, 4)).reshape(-1, 3, 3)
            rot_rel = torch.matmul(v_hat_mol_rot_proj.transpose(-2, -1), gt_rot)
            rot_vec = rotmat_to_rotvec(rot_rel)
            error_rot_geodesic = (rot_vec**2).sum(dim=-1)
            error_rot = scatter_mean(error_rot_geodesic, molecule["idx"], dim=0)
            rmse = torch.sqrt(error_rot)
            print(f"Rotation RMSE: {rmse.mean():.4f}")
        else:
            error_rot = scatter_add(torch.sum(((v_mol_quat.reshape(-1, 3) - v_hat_mol_quat))**2, dim=-1), molecule['idx'], dim=0)
            rmse = torch.sqrt(error_rot / molecule['size'])
            print(f"Rotation RMSE: {rmse.mean():.4f}")
        
        # Calculate position loss
        x_mol = molecule['x']
        x_hat_mol = x_hat_mol.reshape(-1, x_hat_mol.shape[-2], x_hat_mol.shape[-1])
        # print(f'{x_mol[0]=}')
        # print(f'{x_hat_mol[0]=}')

        atom14_exists = molecule["atom14_gt_exists"]
        x_alt_mol = molecule["atom14_alt_gt_positions"]

        sq_gt = torch.sum((x_hat_mol - x_mol) ** 2, dim=-1)        
        sq_alt = torch.sum((x_hat_mol - x_alt_mol) ** 2, dim=-1)    
        sq_min = torch.minimum(sq_gt, sq_alt)
        mask = atom14_exists.to(x_hat_mol.dtype)                   
        sse_per_res = torch.sum(sq_min * mask, dim=-1)          
        n_atoms_per_res = torch.sum(mask, dim=-1).clamp_min(1.0)    
        error_x = scatter_add(sse_per_res, molecule["idx"], dim=0)
        n_atoms_per_graph = scatter_add(n_atoms_per_res, molecule["idx"], dim=0)
        rmse_pos = torch.sqrt((error_x / n_atoms_per_graph) + 1e-6)
        print(f"Position RMSE: {rmse_pos.mean()}")

        # Calculate FAPE loss
        T_peptide = Rigid.from_tensor_4x4(molecule['backbone_rigid_tensor']).to_tensor_7()
        if self.variational:
            fape_loss = _compute_fape_loss(v_hat_mol_quat, v_hat_mol_trans, T_peptide, molecule, protein_pocket, x_hat_mol, sidechain_frames)
        else:
            fape_loss = _compute_fape_loss(quat_hat_mol, trans_hat_mol, T_peptide, molecule, protein_pocket, x_hat_mol, sidechain_frames)
        print(f"Backbone FAPE: {fape_loss['backbone'].mean()}")
        print(f"Sidechain FAPE: {fape_loss['sidechain'].mean()}")
        print(f"Total FAPE: {fape_loss['total'].mean()}")

        # Calculate translation loss
        x1_pos = v_mol_trans.reshape(-1, v_mol_trans.shape[-1])
        x1_pos = x1_pos.reshape(-1, x1_pos.shape[-1])
        # print(f'{x1_pos[0:20]=}')
        # print(f'{v_hat_mol_trans[0:20]=}')
        error_trans = scatter_add(torch.sum(((x1_pos - v_hat_mol_trans))**2, dim=-1), molecule['idx'], dim=0)
        rmse = torch.sqrt((error_trans / molecule['size']) + 1e-8)
        print(f"Translation RMSE mean: {rmse.mean()}")

        # Calculate angle loss
        angles = molecule['torsion_angles_sin_cos']
        angles = angles.reshape(angles.shape[0], angles.shape[1], 7, 2)
        # print(f'{angles[0,0]=}')
        if self.variational:
            angles_hat_mol = v_hat_mol_angles.reshape(angles.shape[0], angles.shape[1], angles.shape[2], angles.shape[3])
        # print(f'{angles_hat_mol[0,0]=}')
        molecule['torsion_angles_mask'] = molecule['torsion_angles_mask'].reshape(angles.shape[0], angles.shape[1], angles.shape[2])
        molecule['alt_torsion_angles_sin_cos'] = molecule['alt_torsion_angles_sin_cos'].reshape(angles.shape[0], angles.shape[1], angles.shape[2], angles.shape[3])
        error_angles = _compute_torsion_angle_loss(angles_hat_mol, molecule['torsion_angles_mask'], angles, molecule['alt_torsion_angles_sin_cos'])
        rmse = torch.sqrt((error_angles / molecule['size']) + 1e-8)
        print(f"Angles RMSE mean: {rmse.mean()}")

        # save pdbs for first sample in batch
        # if save_pdb and run_id is not None:
        #     # Save noisy and predicted peptides for the first element in the batch
        #     mol_single = molecule.copy()
        #     mol_single['size'] = molecule['size'][:1]
            
        #     # 1. Save Predicted Peptide
        #     self.safe_pdbs(x_hat_mol, mol_single, run_id, data_dir, time_step=f'epoch_{current_epoch}_predicted', atom_level=True, save_peptide_only=True)
            
        #     # 2. Save Noisy Peptide
        #     # Extract rigid parameters and angles from z_t_mol
        #     rotmats_t = z_t_mol[:, :, :9].reshape(-1, 3, 3)
        #     quat_t = safe_rot_to_quat(rotmats_t)
        #     quat_t = F.normalize(quat_t, dim=-1, eps=1e-6)
        #     trans_t = z_t_mol[:, :, 9:12].reshape(-1, 3)
        #     angles_t = z_t_mol[:, :, 12:26].reshape(-1, 14)
        #     T_t = torch.cat([quat_t, trans_t], dim=-1)
            
        #     x_t_mol, _, _ = self.predict_pos(molecule, T_t, angles_t)
        #     x_t_mol = x_t_mol.reshape(-1, x_t_mol.shape[-2], x_t_mol.shape[-1])
        #     self.safe_pdbs(x_t_mol, mol_single, run_id, data_dir, time_step=f'epoch_{current_epoch}_noisy', atom_level=True, save_peptide_only=True)
            
        #     # 3. Save Ground Truth Peptide
        #     self.safe_pdbs(molecule['x'], mol_single, run_id, data_dir, time_step=f'epoch_{current_epoch}_gt', atom_level=True, save_peptide_only=True)


        error_pro = torch.zeros(protein_pocket['size'].size(0), device=molecule['x'].device)

        # Loss part from MHC-Diff
        kl_prior = self.kl_prior(molecule)

        # SNR_t = (1 / self.SNR_t(t).squeeze(1))

        # t_0_mask = (t == 1).float().squeeze()
        t_0_mask = (t >= (1.0 - 1.0 / self.T)).float().squeeze()
        t_not_0_mask = 1 - t_0_mask

        loss_x_mol_t0, loss_x_protein_t0, loss_h_t0 = self.loss_t0(
            molecule, z_t_mol, x_mol, x_hat_mol,
            protein_pocket, z_t_pro, v_pro, v_hat_pro, t  
        )

        # seperate loss computation for t = 0 and t != 0
        loss_x_mol_t0 = - loss_x_mol_t0 * t_0_mask
        loss_x_protein_t0 = - loss_x_protein_t0 * t_0_mask
        loss_h_t0 = - loss_h_t0 * t_0_mask
        error_x = error_x * t_not_0_mask
        error_pro = error_pro * t_not_0_mask

        # Normalize loss_t to be mean squared error per coordinate
        # error_x is sum of (x - x_hat)^2 over [N, 14, 3] -> divide by 3 * 14 * N_res
        n_coords_mol = (self.x_dim * 14 * molecule['size']).float()
        n_coords_pro = (self.x_dim * 14 * protein_pocket['size']).float()
        
        error_x = error_x / n_coords_mol
        error_pro = error_pro / n_coords_pro
        loss_t = 0.5 * (error_x + error_pro).mean()

        # Normalize t=0 components similarly
        loss_x_mol_t0 = loss_x_mol_t0 / n_coords_mol
        loss_x_protein_t0 = loss_x_protein_t0 / n_coords_pro
        loss_0 = (loss_x_mol_t0 + loss_x_protein_t0 + loss_h_t0).mean()
        
        # Calculate violation loss
        violation_weight = 0.0
        if current_epoch is not None and max_epochs is not None:
            # Gradually introduce in the last 50% of training
            start_epoch = max_epochs * 0.5
            if current_epoch >= start_epoch:
                violation_weight = (current_epoch - start_epoch) / (max_epochs - start_epoch)

        if violation_weight > 0:
            x_hat_mol_violation = x_hat_mol.clone()
            if x_hat_mol_violation.requires_grad:
                x_hat_mol_violation.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))
            violation_losses = _compute_cross_violation_loss(molecule, protein_pocket, x_hat_mol_violation)
            print(f"Cross Violation Loss: {violation_losses['total'].mean()}, weight: {violation_weight:.4f}")
        else:
            violation_losses = {'total': torch.tensor(0.0, device=molecule['x'].device)}

        # Calculate total loss
        loss = fape_loss["total"] + affinity_loss + error_angles + error_rot + 0.01 * loss_t + 0.01 * loss_0 + kl_prior + 0.01 * violation_weight * violation_losses["total"]

        if self.confidence_score == True:
            c_s_peptide = scatter_add(c_s, molecule['idx'], dim=0).squeeze(1) / molecule['size']
            # confidence weighted loss
            loss_with_conf = 1/(c_s_peptide)**2 * loss + torch.log(c_s_peptide**2)
        else:
            c_s_peptide = torch.zeros_like(loss)
            loss_with_conf = torch.zeros_like(loss)
            

        info = {
            'loss_t': loss_t.mean(0),
            'loss_0': loss_0.mean(0),
            'error_x': error_x.mean(0),
            'loss_x_mol_t0': loss_x_mol_t0.mean(0),
            'kl_prior': kl_prior.mean(0),
            'confidence': c_s_peptide.mean(0),
            'loss_with_conf': loss_with_conf.mean(0),
            'error_trans': error_trans.mean(0),
            'error_rot': error_rot.mean(0),
            'error_angles': error_angles.mean(0),
            'error_ba': error_ba.mean(0),
            'affinity_loss': affinity_loss.mean(0),
            'fape_loss_total': fape_loss['total'].mean(0),
            'rmse_pos': rmse_pos.mean(0),
            'violation_losses': violation_losses['total'].mean(0),
        }

        if self.confidence_score == True:
            return loss_with_conf, info

        return loss, info
    
    def validation_loss(
            self, z_data, molecule, z_t_mol, v_mol, v_hat_mol,
            protein_pocket, z_t_pro, v_pro, v_hat_pro, 
            t, ba_hat, current_epoch=None, max_epochs=None,
            run_id=None, data_dir=None, save_pdb=False
    ):
        print(' ')
        batch_size = len(molecule['affinity'])
        size_mol = molecule['size'][0]

        if self.ba:
            ba = molecule['affinity']
            ba_hat = ba_hat.squeeze(-1)
            error_ba = torch.mean((ba - ba_hat) ** 2, dim=-1)
            print(f"BA RMSE mean: {torch.sqrt(error_ba.mean())}")

            affinity_loss = _regression_loss_function(ba_hat.float(), ba.float())
            print(f"BA Loss: {affinity_loss.mean()}")
        else:
            error_ba = 0
            affinity_loss = 0

        if self.variational:
            T_peptide = Rigid.from_tensor_4x4(molecule['backbone_rigid_tensor']).to_tensor_7()
            v_mol_quat = T_peptide[:,:,:4] # shape [B*N, 4]
            v_mol_trans = T_peptide[:,:,4:7] # shape [B*N, 3]
            v_mol_angles = molecule['torsion_angles_sin_cos'] # shape [B*N, 14]
        else:
            v_mol_rot = v_mol['rot'] # shape [B*N, 3, 3]
            v_mol_quat = v_mol['quat'] # shape [B*N, 3]
            v_mol_trans = v_mol['trans'] # shape [B*N, 3]
            v_mol_angles = v_mol['angles'] # shape [B*N, 7]
    
        # Calculate peptide positions 
        # EGNN quat outputs quaternions directly (4-dim)
        v_hat_mol_quat = v_hat_mol[:,:self.rot_dim]  # [B*N, 4] quaternions
        v_hat_mol_quat = F.normalize(v_hat_mol_quat, dim=-1, eps=1e-6)
        v_hat_mol_trans = v_hat_mol[:,self.rot_dim:self.rot_dim+self.x_dim] # shape [B*N, 3]
        v_hat_mol_angles = v_hat_mol[:,self.rot_dim+self.x_dim:self.rot_dim+self.x_dim+self.angle_dim] # shape [B*N, 14]

        x_hat_mol, v_hat_mol_angles, sidechain_frames = self.predict_pos(molecule, torch.concat([v_hat_mol_quat, v_hat_mol_trans], dim=-1), v_hat_mol_angles)
        
        if not self.variational:
            one_minus_t = torch.clamp(1 - t, min=1.0 / self.T)

            v_hat_mol_quat = v_hat_mol[:,:self.rot_dim]  # [B*N, 4]
            v_hat_mol_quat = F.normalize(v_hat_mol_quat, dim=-1, eps=1e-6)
            quat_hat_mol = v_hat_mol_quat.reshape(batch_size, size_mol, 4)
            z_t_mol_quat = z_t_mol['rot'].reshape(-1, 4)  # already quaternions
            v_hat_mol_quat = calc_quat_wt_qt_q1(z_t_mol_quat.reshape(batch_size, size_mol, 4), v_hat_mol_quat.reshape(batch_size, size_mol, 4)) 
            v_hat_mol_quat = v_hat_mol_quat / one_minus_t
            v_hat_mol_quat = v_hat_mol_quat.reshape(-1, 3) # shape [B*N, 3]

            trans_hat_mol = v_hat_mol[:,self.rot_dim:self.rot_dim+self.x_dim].reshape(batch_size, size_mol, 3)
            v_hat_mol_trans = (trans_hat_mol - z_t_mol['trans']) / one_minus_t
            v_hat_mol_trans = v_hat_mol_trans.reshape(-1, 3)

            angles_hat_mol = v_hat_mol[:,self.rot_dim+self.x_dim:self.rot_dim+self.x_dim+self.angle_dim].view(batch_size, size_mol, 7, 2)
            a_t = z_t_mol['angles'].view(batch_size, size_mol, 7, 2)
            cos_1, sin_1 = angles_hat_mol[..., 0], angles_hat_mol[..., 1]
            cos_t, sin_t = a_t[..., 0], a_t[..., 1]
            sin_diff = sin_1 * cos_t - cos_1 * sin_t
            cos_diff = cos_1 * cos_t + sin_1 * sin_t

            # sin_diff = a_t[..., 0]*a_1[..., 1] - a_t[..., 1]*a_1[..., 0]
            # cos_diff = a_t[..., 0]*a_1[..., 0] + a_t[..., 1]*a_1[..., 1]
            geodesic_distance = torch.atan2(sin_diff, cos_diff)
            v_hat_mol_angles = geodesic_distance / one_minus_t
            v_hat_mol_angles = v_hat_mol_angles.reshape(-1, 7) # shape [B*N, 7]

        # Calculate rotation loss
        if self.variational:
            v_hat_mol_quat = v_hat_mol_quat.reshape(batch_size, size_mol, v_hat_mol_quat.shape[-1])
            v_hat_mol_rot_proj = quat_to_rot(v_hat_mol_quat).reshape(-1, 3, 3)
            gt_rot = quat_to_rot(v_mol_quat.reshape(-1, 4)).reshape(-1, 3, 3)
            rot_rel = torch.matmul(v_hat_mol_rot_proj.transpose(-2, -1), gt_rot)
            rot_vec = rotmat_to_rotvec(rot_rel)
            error_rot_geodesic = (rot_vec**2).sum(dim=-1)
            error_rot = scatter_mean(error_rot_geodesic, molecule["idx"], dim=0)
            rmse = torch.sqrt(error_rot)
            print(f"Rotation RMSE: {rmse.mean():.4f}")
        else:
            error_rot = scatter_add(torch.sum(((v_mol_quat.reshape(-1, 3) - v_hat_mol_quat))**2, dim=-1), molecule['idx'], dim=0)
            rmse = torch.sqrt(error_rot / molecule['size'])
            print(f"Rotation RMSE: {rmse.mean():.4f}")
        
        
        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None

        # Calculate position loss
        x_mol = molecule['x']
        x_hat_mol = x_hat_mol.reshape(-1, x_hat_mol.shape[-2], x_hat_mol.shape[-1])
  
        atom14_exists = molecule["atom14_gt_exists"]
        x_alt_mol = molecule["atom14_alt_gt_positions"]

        sq_gt = torch.sum((x_hat_mol - x_mol) ** 2, dim=-1)        
        sq_alt = torch.sum((x_hat_mol - x_alt_mol) ** 2, dim=-1)    
        sq_min = torch.minimum(sq_gt, sq_alt)
        mask = atom14_exists.to(x_hat_mol.dtype)                   
        sse_per_res = torch.sum(sq_min * mask, dim=-1)          
        n_atoms_per_res = torch.sum(mask, dim=-1).clamp_min(1.0)    
        error_x = scatter_add(sse_per_res, molecule["idx"], dim=0)
        n_atoms_per_graph = scatter_add(n_atoms_per_res, molecule["idx"], dim=0)
        rmse_pos = torch.sqrt(error_x / n_atoms_per_graph)
        print(f"Position RMSE: {rmse_pos.mean()}")

        # Calculate FAPE loss
        T_peptide = Rigid.from_tensor_4x4(molecule['backbone_rigid_tensor']).to_tensor_7()
        if self.variational:
            fape_loss = _compute_fape_loss(v_hat_mol_quat, v_hat_mol_trans, T_peptide, molecule, protein_pocket, x_hat_mol, sidechain_frames)
        else:
            fape_loss = _compute_fape_loss(quat_hat_mol, trans_hat_mol, T_peptide, molecule, protein_pocket, x_hat_mol, sidechain_frames)
        print(f"Backbone FAPE: {fape_loss['backbone'].mean()}")
        print(f"Sidechain FAPE: {fape_loss['sidechain'].mean()}")
        print(f"Total FAPE: {fape_loss['total'].mean()}")

        # Calculate translation loss
        v_mol_trans = v_mol_trans.reshape(-1, v_mol_trans.shape[-1])
        v_hat_mol_trans = v_hat_mol_trans.reshape(-1, v_hat_mol_trans.shape[-1])
        # translation_loss_weight = 2.0
        # trans_scale = 0.1
        # error_trans = scatter_add(torch.sum(((x1_pos - v_hat_mol_pos) * trans_scale)**2, dim=-1), molecule['idx'], dim=0) * translation_loss_weight
        error_trans = scatter_add(torch.sum(((v_mol_trans - v_hat_mol_trans))**2, dim=-1), molecule['idx'], dim=0)
        rmse = torch.sqrt(error_trans / molecule['size'])
        print(f"Translation RMSE mean: {rmse.mean()}")

        # Calculate angles loss
        angles = molecule['torsion_angles_sin_cos']
        angles = angles.reshape(angles.shape[0], angles.shape[1], 7, 2)
        if self.variational:
            angles_hat_mol = v_hat_mol_angles.reshape(angles.shape[0], angles.shape[1], angles.shape[2], angles.shape[3])
        molecule['torsion_angles_mask'] = molecule['torsion_angles_mask'].reshape(angles.shape[0], angles.shape[1], angles.shape[2])
        molecule['alt_torsion_angles_sin_cos'] = molecule['alt_torsion_angles_sin_cos'].reshape(angles.shape[0], angles.shape[1], angles.shape[2], angles.shape[3])
        error_angles = _compute_torsion_angle_loss(angles_hat_mol, molecule['torsion_angles_mask'], angles, molecule['alt_torsion_angles_sin_cos'])
        rmse = torch.sqrt(error_angles / molecule['size'])
        print(f"Angles RMSE mean: {rmse.mean()}")

        error_pro = torch.zeros(protein_pocket['size'].size(0), device=molecule['x'].device)

        # Calculate loss like MHC-Diff
        kl_prior = self.kl_prior(molecule)

        # if pocket not fixed then molecule['size'] + protein_pocket['size']
        neg_log_const = self.neg_log_const(molecule['size'], molecule['size'].size(0), device=molecule['x'].device)
        delta_log_px = self.delta_log_px(molecule['size'])

        # SNR is computed between timestep s and t (with s = t-1)
        SNR_weight = (1 - self.SNR_s_t(t).squeeze(1))

        # TODO: add log_pN computation using the dataset histogram
        log_pN = self.log_pN(molecule['size'], protein_pocket['size'])

        # TODO optional: can add auxiliary loss / lennard-jones potential

        ## For evaluation we want to compute t = 1 losses for all z_data samples that we have
        # compute noised sample for t = 1
        z_0_mol, z_0_pro, v_target_0_mol, v_target_0_pro, t_0 = self.compute_flow_match(z_data, t_is_1 = True)
        z_0_mol = torch.cat((z_0_mol['rot'], z_0_mol['trans'], z_0_mol['angles'], z_0_mol['h']), dim=-1)
        if self.variational:
            v_target_0_quat = v_hat_mol_quat.reshape(-1, 4)
            v_target_0_trans = v_hat_mol_trans.reshape(-1, 3)
            v_target_0_angles = v_hat_mol_angles.reshape(-1, 7, 2)
        else:
            v_target_0_trans = trans_hat_mol.reshape(-1, 3)
            v_target_0_quat = quat_hat_mol.reshape(-1, 4)
            v_target_0_angles = angles_hat_mol.reshape(-1, 7, 2)
        x_target_0_mol, _, _ = self.predict_pos(molecule, torch.concat([v_target_0_quat, v_target_0_trans], dim=-1), v_target_0_angles)
        x_target_0_mol = x_target_0_mol.reshape(-1, x_target_0_mol.shape[-2], x_target_0_mol.shape[-1])

        mask = molecule['cross_residues_mask'].reshape(molecule['h'].shape[0], molecule['h'].shape[1])

        # use neural network to predict noise for t = 0
        v_hat_0_mol, v_hat_0_pro, _, _ = self.neural_net(z_0_mol, z_0_pro, t_0, molecule['idx'], protein_pocket['idx'], molecule_pos, molecule['torsion_angles_mask'], mask)
        v_hat_0_mol = v_hat_0_mol.reshape(-1, v_hat_0_mol.shape[-1])
        v_hat_0_mol_quat = F.normalize(v_hat_0_mol[:, :self.rot_dim], dim=-1, eps=1e-6)  # [B*N, 4]
        v_hat_0_mol_trans = v_hat_0_mol[:, self.rot_dim:self.rot_dim+self.x_dim]
        v_hat_0_mol_angles = v_hat_0_mol[:, self.rot_dim + self.x_dim:self.rot_dim + self.x_dim + self.angle_dim]
        x_hat_0_mol, _, _ = self.predict_pos(molecule, torch.concat([v_hat_0_mol_quat, v_hat_0_mol_trans], dim=-1), v_hat_0_mol_angles)

        x_hat_0_mol = x_hat_0_mol.reshape(-1, x_hat_0_mol.shape[-2], x_hat_0_mol.shape[-1])
        loss_x_mol_t0, loss_x_protein_t0, loss_h_t0 = self.loss_t0(
            molecule, z_0_mol, x_target_0_mol, x_hat_0_mol,
            protein_pocket, z_0_pro, v_pro, v_hat_pro, t_0
        )

        loss_x_mol_t0 = - loss_x_mol_t0
        loss_x_protein_t0 = - loss_x_protein_t0
        loss_h_t0 = - loss_h_t0

        # loss_t = - self.T * 0.5 * SNR_weight * (error_mol + error_pro)
        loss_t = self.T * 0.5 * (error_x + error_pro)
        loss_0 = loss_x_mol_t0 + loss_x_protein_t0 + loss_h_t0
        loss_0 = loss_0 + neg_log_const

        # Calculate violation loss
        violation_weight = 0.0
        if current_epoch is not None and max_epochs is not None:
            start_epoch = max_epochs * 0.5
            if current_epoch >= start_epoch:
                violation_weight = (current_epoch - start_epoch) / (max_epochs - start_epoch)
        
        if violation_weight > 0:
            x_hat_mol_violation = x_hat_mol.clone()
            if x_hat_mol_violation.requires_grad:
                x_hat_mol_violation.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))
            violation_losses = _compute_cross_violation_loss(molecule, protein_pocket, x_hat_mol_violation)
            print(f"Cross Violation Loss: {violation_losses['total'].mean()}, weight: {violation_weight:.4f}")
        else:
            violation_losses = {'total': torch.tensor(0.0, device=molecule['x'].device)}

        # loss = fape_loss["total"] + affinity_loss + error_angles + violation_weight * violation_losses["total"] #+ (error_trans / molecule['size']) + 10.0 * error_quat
        # Two added loss terms for vlb
        # loss = loss_t + loss_0 + kl_prior - delta_log_px - log_pN #+ error_quat + error_trans + error_angles + error_ba
        # loss += fape_loss["total"] + affinity_loss + error_angles + violation_weight * violation_losses["total"]
        # loss = loss_t + kl_prior
        loss = fape_loss["total"] + affinity_loss + error_angles + error_rot + 0.01 * loss_t + 0.01 * loss_0 + kl_prior + 0.01 * violation_weight * violation_losses["total"]


        info = {
            'loss_t': loss_t.mean(0),
            'loss_0': loss_0.mean(0),
            'error_x': error_x.mean(0),
            'loss_x_mol_t0': loss_x_mol_t0.mean(0),
            'kl_prior': kl_prior.mean(0),
            'neg_log_const': neg_log_const.mean(0),
            'SNR_weight': SNR_weight.mean(0),
            'error_trans': error_trans.mean(0),
            'error_rot': error_rot.mean(0),
            'error_angles': error_angles.mean(0),
            'error_ba': error_ba.mean(0),
            'affinity_loss': affinity_loss.mean(0),
            'fape_loss_total': fape_loss['total'].mean(0),
            'rmse_pos': rmse_pos.mean(0),
            'violation_losses': violation_losses['total'].mean(0),
        }

        return loss, info
        
    
    def loss_t0(
            self, molecule, z_t_mol, v_target_mol, v_hat_mol,
            protein_pocket, z_t_pro, v_target_pro, v_hat_pro, 
            t, epsilon=1e-10
    ):
        """
        This function calculate log(p(xh|z_0))
        """

        ## Normal computation of position error when sampling from fully denoised distribution

        a = torch.sum((v_target_mol - v_hat_mol)**2, dim=-1)
        a = torch.sum(a, dim=-1) / a.shape[-1]
        loss_x_mol_t0 = - 0.5 * scatter_add(a, molecule['idx'], dim=0)

        loss_x_protein_t0 = torch.zeros(protein_pocket['size'].size(0), device=molecule['x'].device)

        if self.features_fixed:

            loss_h_t0 = torch.zeros(molecule['size'].size(0), device=molecule['x'].device)

        else:
            ## Computation for changed features

            
            sigma_0 = self.noise_schedule(t, 'sigma')
            sigma_0_unnormalized = sigma_0 * self.norm_values[1]
            # unnormalize not necessary for molecule['h'] because molecule was only locally normalized (can change that if necessary later)
            mol_h_hat = z_t_mol[:, self.x_dim:] * self.norm_values[1]
            mol_h_hat_centered = mol_h_hat - 1

            # Compute integrals from 0.5 to 1.5 of the normal distribution
            # N(mean=z_h_cat, stdev=sigma_0_cat)
            # 0.5 * (1. + torch.erf(x / math.sqrt(2)))
            log_probabilities_mol_unnormalized = torch.log(
                0.5 * (1. + torch.erf((mol_h_hat_centered + 0.5) / sigma_0_unnormalized[molecule['idx']]) / math.sqrt(2)) \
                - 0.5 * (1. + torch.erf((mol_h_hat_centered - 0.5) / sigma_0_unnormalized[molecule['idx']]) / math.sqrt(2)) \
                + epsilon
            )

            # Normalize the distribution over the categories.
            log_Z = torch.logsumexp(log_probabilities_mol_unnormalized, dim=1,
                                    keepdim=True)
            
            log_probabilities_mol = log_probabilities_mol_unnormalized - log_Z

            loss_h_t0 = scatter_add(torch.sum(log_probabilities_mol * molecule['h'], dim=-1), molecule['idx'], dim=0)
        
        return loss_x_mol_t0, loss_x_protein_t0, loss_h_t0
    
    def kl_prior(self, molecule):

        device=molecule['x'].device

        peptide_backbone_rigid_tensor = molecule['backbone_rigid_tensor']
        T_peptide = Rigid.from_tensor_4x4(peptide_backbone_rigid_tensor)
        T_peptide = T_peptide.to_tensor_7()
        T_peptide = T_peptide.reshape(-1, T_peptide.shape[-1])
        T_peptide = T_peptide[:, 4:7]

        T_normalized = torch.ones((len(molecule['size']), 1), device=device)
        # alpha_T = self.noise_schedule(T_normalized, 'alpha')
        # sigma_T = self.noise_schedule(T_normalized, 'sigma')
        # sigma_T_value = sigma_T[0,0].item()
        alpha_T = 0.0  # At t=1, (1-t) is 0
        sigma_T_val = 1.0

        # mu_x_mol = molecule['x'] * alpha_T[molecule['idx']] # [:,3]
        # mu_x_mol = molecule['x'] * alpha_T # [:,3]
        mu_x_mol = T_peptide * alpha_T # [:,3]
        # mu_x_mol = T_peptide_pos * alpha_T # [:,3]
        # mu_h_mol = molecule['h'] * alpha_T[molecule['idx']] # [:,20]
        
        # sigma_T_x = torch.full(alpha_T.shape, fill_value=sigma_T_value, device=device).squeeze() # [64,1]
        # sigma_T_h = torch.full(alpha_T.shape, fill_value=sigma_T_value, device=device).squeeze() # [64,1]
        # sigma_T_x = torch.full((len(molecule['size']),), fill_value=sigma_T_val * self.noise_scaling, device=device)
        sigma_T_x = torch.full((len(molecule['size']),), fill_value=sigma_T_val, device=device)
        # sigma_T_h = torch.full((len(molecule['size']),), fill_value=sigma_T_val, device=device)


        # KL computation h (if features are diffused)
        kl_h = 0
        # zeros = torch.zeros_like(mu_h_mol)
        # ones = torch.ones_like(sigma_T_h)
        # mu_norm2 = scatter_add(torch.sum((mu_h_mol - zeros) ** 2, dim=1), molecule['idx'], dim=0)
        # kl_h = torch.log(ones / sigma_T_h) + 0.5 * (sigma_T_h**2 + mu_norm2) / (ones**2) - 0.5

        # KL computation x
        zeros = torch.zeros_like(mu_x_mol)
        ones = torch.ones_like(sigma_T_x) #* self.noise_scaling
        mu_norm2 = scatter_add(torch.sum((mu_x_mol - zeros) ** 2, dim=-1), molecule['idx'], dim=0)
        d = (molecule['size'] - 1) * self.x_dim
        kl_x = d * torch.log(ones / sigma_T_x) + 0.5 * (d * sigma_T_x**2 + mu_norm2) / (ones**2) - 0.5 * d

        kl_loss = kl_x + kl_h

        return kl_loss
    
    def delta_log_px(self, num_nodes):

        delta_log_px = - (num_nodes - 1) * self.x_dim * np.log(self.norm_values[0])

        return delta_log_px
    
    def log_pN(self, molecule_N, protein_pocket_N):

        # add log_pN computation using the dataset histogram
        # only matters for diverse molecule sizes, therefore we set it to 0
        log_pN = 0

        return log_pN
    
    def neg_log_const(self, num_nodes, batch_size, device):

        # t0 = torch.zeros((batch_size, 1), device=device)
        # log_sigma_0 = torch.log(self.noise_schedule(t0, 'sigma')).view(batch_size)

        # neg_log_const = - ((num_nodes - 1) * self.x_dim) * (- log_sigma_0 - 0.5 * np.log(2 * np.pi))

        # return neg_log_const
        return torch.zeros(batch_size, device=device)
    
    def SNR_s_t(self, t):

        # 1. Define previous timestep s
        s = torch.clamp(torch.round(t * self.T).long() - 1, min=0)
        s = s / self.T

        # 2. In Flow Matching: x_t = (1-t)x_0 + t*x_1
        # Effective alpha (signal weight) is (1-t)
        # Effective sigma (noise weight) is t
        # alpha2_t = (1.0 - t)**2
        # alpha2_s = (1.0 - s)**2
        
        # sigma2_t = t**2
        # sigma2_s = s**2

        alpha2_t = t**2
        alpha2_s = s**2

        sigma2_t = (1.0 - t)**2
        sigma2_s = (1.0 - s)**2

        # 3. Compute SNR as the ratio of signal-to-noise ratios
        # Adding epsilon to avoid division by zero at t=0
        eps = 1e-8
        snr_t = alpha2_t / (sigma2_t + eps)
        snr_s = alpha2_s / (sigma2_s + eps)

        # This represents the "step-wise" change in SNR
        SNR_s_t = snr_s / (snr_t + eps)

        return SNR_s_t

    def _init_residue_constants(self, float_dtype, device): 
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    dtype=torch.long,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, T, alpha, aatype):

        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)

        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(T, alpha, aatype, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, T, aatype  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        T_rots = T.get_rots()
        self._init_residue_constants(T_rots.dtype, T_rots.device)

        return frames_and_literature_positions_to_atom14_pos(
            T,
            aatype,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
    
    def _calculate_omegas_from_positions(self, positions: torch.Tensor, res_mask: torch.Tensor):
        """
        The amide's hydrogen is absent.
        So we calculate the omega from the Ca-C-N-Ca angle.

        Args:
            positions: [*, N_res, 14, 3]
            res_mask:  [*, N_res] (boolean)
        Returns:
            post omegas sin, cos:  [*, N_res - 1, 2] (normalized)
        """

        # positions in the array where the backbone atoms are stored:
        atom_index_N = 0
        atom_index_CA = 1
        atom_index_C = 2

        # find the backbone atoms

        # [*, N_res - 1, 3]
        positions_CA0 = positions[..., :-1, atom_index_CA, :]
        positions_C0 = positions[..., :-1:, atom_index_C, :]
        positions_N1 = positions[..., 1:, atom_index_N, :]
        positions_CA1 = positions[..., 1:, atom_index_CA, :]

        # [*, N_res - 1]
        mask = torch.logical_and(res_mask[..., :-1], res_mask[..., 1:])
        masked_out = torch.logical_not(mask)

        # make directional vectors for the 3 bonds: C-alpha---C---N---C-alpha

        # [*, N_res - 1, 3]
        vec_CCA0 = positions_CA0 - positions_C0

        # [*, N_res - 1, 3]
        vec_NCA1 = positions_CA1 - positions_N1

        # [*, N_res - 1, 3]
        vec_CN = positions_N1 - positions_C0

        # make the newmann projections of the C-alphas on the C---N bond

        # [*, N_res - 1, 3]
        plane_n = torch.nn.functional.normalize(vec_CN, dim=-1)

        # [*, N_res - 1, 3]
        newmann0 = torch.nn.functional.normalize(vec_CCA0 - (plane_n * vec_CCA0).sum(dim=-1).unsqueeze(-1) * plane_n, dim=-1)

        # [*, N_res - 1, 3]
        newmann1 = torch.nn.functional.normalize(vec_NCA1 - (plane_n * vec_NCA1).sum(dim=-1).unsqueeze(-1) * plane_n, dim=-1)

        # convert the projections to cosine and sine

        # [*, N_res - 1]
        omega_cos = (newmann0 * newmann1).sum(dim=-1)

        # [*, N_res - 1, 3]
        cross01 = torch.linalg.cross(newmann0, newmann1, dim=-1)

        # [*, N_res - 1]
        # Use gradient-safe norm to avoid NaN at zero cross product
        omega_sin = torch.sqrt(torch.sum(cross01**2, dim=-1) + 1e-12)
        omega_sin = torch.where((cross01 * plane_n).sum(dim=-1) < 0.0, -omega_sin, omega_sin)

        # masked areas get 180 degrees omega
        omega_cos = torch.where(mask, omega_cos,-1.0)
        omega_sin = torch.where(mask, omega_sin, 0.0)

        return torch.cat([omega_sin[..., None], omega_cos[..., None]], dim=-1)

    # def compute_ba(self, peptide_embd, peptide_mask):


    #     # [*, peptide_maxlen, output_size]
    #     peptide_scores = self.ba_module(peptide_embd)

    #     peptide_mask = peptide_mask.to(bool)

    #     # [*, peptide_maxlen, output_size]
    #     masked_scores = torch.where(peptide_mask.unsqueeze(-1), peptide_scores, 0.0)

    #     # [*, output_size]
    #     ba = masked_scores.sum(dim=-2)

    #     return ba
    

    @torch.no_grad()
    def sample_structure(self, num_samples, molecule, protein_pocket, sampling_without_noise, data_dir, run_id, save_trajectory=False, fold=None):
        
        device = molecule['x'].device
        num_graphs = molecule['size'].size(0)
        
        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None
        
        size_mol = molecule['size'][0]
        size_pro = protein_pocket['size'][0]

        if self.com_handling != 'no_COM':
            mean_shift = self._center_inputs(molecule, protein_pocket)

        if molecule['h'].shape[0] != num_graphs:
            molecule['h'] = molecule['h'].view(num_graphs, size_mol, *molecule['h'].shape[1:])
        if protein_pocket['h'].shape[0] != num_graphs:
            protein_pocket['h'] = protein_pocket['h'].view(num_graphs, size_pro, *protein_pocket['h'].shape[1:])
        if molecule['torsion_angles_sin_cos'].shape[0] != num_graphs:
            molecule['torsion_angles_sin_cos'] = molecule['torsion_angles_sin_cos'].view(num_graphs, size_mol, *molecule['torsion_angles_sin_cos'].shape[1:])
        if molecule['backbone_rigid_tensor'].shape[0] != num_graphs:
            molecule['backbone_rigid_tensor'] = molecule['backbone_rigid_tensor'].view(num_graphs, size_mol, *molecule['backbone_rigid_tensor'].shape[1:])
        if 'torsion_angles_mask' in molecule and molecule['torsion_angles_mask'].shape[0] != num_graphs:
            molecule['torsion_angles_mask'] = molecule['torsion_angles_mask'].view(num_graphs, size_mol, 7)
        if protein_pocket['backbone_rigid_tensor'].shape[0] != num_graphs:
            protein_pocket['backbone_rigid_tensor'] = protein_pocket['backbone_rigid_tensor'].view(num_graphs, size_pro, *protein_pocket['backbone_rigid_tensor'].shape[1:])

        # Generate random initial positions and orientations
        z_trans = torch.randn((*molecule['h'].shape[:-1], 3), device=device)
        z_trans = z_trans - scatter_mean(z_trans.view(-1, 3), molecule['idx'], dim=0)[molecule['idx']].view(z_trans.shape)
        rotmats_0 = _uniform_so3(molecule['h'].shape[0], molecule['h'].shape[1], device)
        # Store as quaternions (4-dim) for quat variant
        rotquats_0 = safe_rot_to_quat(rotmats_0)
        rotquats_0 = rotquats_0.view(rotquats_0.shape[0], rotquats_0.shape[1], 4)
        T_peptide_z = torch.cat((rotquats_0, z_trans), dim=-1)

        # mol_norm_x = molecule['x'] / self.norm_values[0]
        protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]
        
        # Get protein rotation and translation 
        protein_backbone_rigid_tensor = protein_pocket['backbone_rigid_tensor']
        T_protein = Rigid.from_tensor_4x4(protein_backbone_rigid_tensor)
        T_protein = T_protein.to_tensor_7()
        quat_protein = T_protein[:,:,:4]
        quat_protein = quat_protein.view(-1, quat_protein.shape[-1])
        # Store as quaternions (4-dim) for quat variant
        rot_protein = quat_protein.view(T_protein.shape[0], T_protein.shape[1], 4)
        trans_protein = T_protein[:,:,4:]
        T_protein = torch.cat((rot_protein, trans_protein), dim=-1)

        xh_pro = torch.cat((T_protein, protein_pocket['h']), dim=-1)

        if self.features_fixed:
            z_h_mol = (molecule['h'] / self.norm_values[1]).clone().detach()
        else:
            raise NotImplementedError

        # Get random torsion angles
        random_angles = torch.rand((molecule['h'].shape[0], molecule['h'].shape[1], 7), device=device) * 2 * math.pi
        sin_angles = torch.sin(random_angles)
        cos_angles = torch.cos(random_angles)
        angles_z = torch.stack((sin_angles, cos_angles), dim=-1)
        if 'torsion_angles_mask' in molecule:
            angles_z = angles_z * molecule['torsion_angles_mask'].unsqueeze(-1)
        angles_z = angles_z.view(num_graphs, size_mol, -1)

        current_xh_mol = torch.cat((T_peptide_z, angles_z, z_h_mol), dim=-1)
        current_xh_mol = current_xh_mol.reshape(-1, current_xh_mol.shape[-1])
        xh_pro = xh_pro.reshape(-1, xh_pro.shape[-1])

        # Get target peptide rotation and translation 
        peptide_backbone_rigid_tensor = molecule['backbone_rigid_tensor']
        T_peptide = Rigid.from_tensor_4x4(peptide_backbone_rigid_tensor)
        T_peptide = T_peptide.to_tensor_7()

        print('using solver', self.solver)
        if self.solver == "loop":

            steps = self.T // self.sampling_stepsize
            dt = 1.0 / steps

            for k in range(steps):
                t_val = k / steps
                t_array = torch.full((num_graphs, 1, 1), fill_value=t_val, device=device)

                current_rot_quat = current_xh_mol[:, :self.rot_dim]  # [N, 4] quaternions
                current_rot_quat = F.normalize(current_rot_quat, dim=-1, eps=1e-6)
                current_trans_mol = current_xh_mol[:, self.rot_dim:self.rot_dim+self.x_dim]
                current_angle_mol = current_xh_mol[:, self.rot_dim+self.x_dim:self.rot_dim+self.x_dim+self.angle_dim]
                current_h_mol = current_xh_mol[:, self.rot_dim+self.x_dim+self.angle_dim:]  

                # Predict velocity v_hat
                v_hat_mol, _, c_s, ba_hat = self.neural_net(
                    current_xh_mol, xh_pro, t_array, 
                    molecule['idx'], protein_pocket['idx'], molecule_pos, molecule['torsion_angles_mask']
                )

                # Euler Step: x_{t+dt} = x_t + v(x_t, t) * dt
                # EGNN quat outputs quaternions directly
                v_hat_mol_quat = F.normalize(v_hat_mol[:, :self.rot_dim], dim=-1, eps=1e-6)
                
                v_hat_mol_trans = v_hat_mol[:, self.rot_dim:self.rot_dim+self.x_dim]
                v_hat_mol_angle = v_hat_mol[:, self.rot_dim+self.x_dim:self.rot_dim+self.x_dim+self.angle_dim]
                v_hat_mol_h = v_hat_mol[:, self.rot_dim+self.x_dim+self.angle_dim:]  

                # Convert quats to rotmats for rotvec-based Euler step
                current_rot_mat = quat_to_rot(current_rot_quat)  # [N, 3, 3]
                v_hat_rot_mat = quat_to_rot(v_hat_mol_quat)  # [N, 3, 3]
                rot_mat = torch.matmul(current_rot_mat.transpose(-2, -1), v_hat_rot_mat)
                rot_vec = rotmat_to_rotvec(rot_mat) / max(1 - t_val, 1.0 / steps)
                current_rot_mat = torch.matmul(current_rot_mat, rotvec_to_rotmat(dt * rot_vec))
                # Convert back to quaternions
                current_rot_quat = safe_rot_to_quat(current_rot_mat)
                current_rot_quat = F.normalize(current_rot_quat, dim=-1, eps=1e-6)
                current_rot_quat = current_rot_quat.reshape(num_graphs * size_mol, 4)

                current_trans_mol += dt * (v_hat_mol_trans - current_trans_mol) / max(1 - t_val, 1.0 / steps)

                a_t_sincos = current_angle_mol.view(-1, 7, 2)
                a_hat_sincos = v_hat_mol_angle.view(-1, 7, 2)
                alpha_t = torch.atan2(a_t_sincos[..., 0], a_t_sincos[..., 1])
                alpha_hat = torch.atan2(a_hat_sincos[..., 0], a_hat_sincos[..., 1])
                diff = ((alpha_hat - alpha_t + math.pi) % (2 * math.pi)) - math.pi
                omega = diff / max(1 - t_val, 1.0 / steps)
                alpha_next = alpha_t + dt * omega
                current_angle_mol = torch.stack([torch.sin(alpha_next), torch.cos(alpha_next)], dim=-1).view(-1, self.angle_dim)

                if self.features_fixed:
                    current_h_mol = v_hat_mol_h
                else:
                    current_h_mol += dt * v_hat_mol_h / max(1 - t_val, 1.0 / steps)

                current_xh_mol = torch.cat((current_rot_quat, current_trans_mol, current_angle_mol, current_h_mol), dim=-1)

        elif self.solver in ("euler", "rk4"):

            ode_func = ODEWrapper(
                self, T_peptide, xh_pro, molecule, molecule_pos, protein_pocket, 
                step_size=(1.0 / self.T), rot_dim=self.rot_dim, x_dim=self.x_dim, angle_dim=self.angle_dim,
                angle_mask=molecule['torsion_angles_mask'], variational=self.variational
            )
            
            if save_trajectory:
                t_span = torch.linspace(0.0, 1.0, self.T + 1, device=device)
            else:
                t_span = torch.tensor([0.0, 1.0], device=device)

            trajectory = odeint(
                ode_func, 
                current_xh_mol, 
                t_span, 
                method=self.solver, 
                options={'step_size': 1.0 / self.T} 
            )

            if save_trajectory:
                print(f"Saving trajectory for 3 peptides, {num_samples} samples each...")
                
                sample_batch_size = molecule['size'].size(0) // num_samples
                
                for t_idx in range(len(t_span)):
                    xh_t = trajectory[t_idx]
                    
                    # Extract quaternions directly (4-dim)
                    quat_t = F.normalize(xh_t[:, :self.rot_dim], dim=-1, eps=1e-6)
                    trans_t = xh_t[:, self.rot_dim:self.rot_dim+self.x_dim]
                    angles_t = xh_t[:, self.rot_dim+self.x_dim:self.rot_dim+self.x_dim+self.angle_dim]
                    
                    angles_t = angles_t.view(-1, 7, 2)
                    angles_t = F.normalize(angles_t, dim=-1, eps=1e-6)
                    if 'torsion_angles_mask' in molecule:
                        mask = molecule['torsion_angles_mask'].view(-1, 7, 1)
                        angles_t = angles_t * mask
                    angles_t = angles_t.view(-1, 14)
                    
                    T_hat_t = torch.cat((quat_t, trans_t), dim=-1)
                    x_hat_t, _, _ = self.predict_pos(molecule, T_hat_t, angles_t)
                    x_hat_t = x_hat_t.reshape(-1, 3)
                    
                    # Only save for the first 3 peptides across all samples
                    save_mask = torch.zeros(molecule['size'].size(0), dtype=torch.bool, device=device)
                    for s in range(num_samples):
                        for j in range(min(3, sample_batch_size)):
                            save_mask[s * sample_batch_size + j] = True
                    
                    self.save_trajectory_step(x_hat_t, molecule, run_id, t_idx, num_samples, save_mask, data_dir)


            current_xh_mol = trajectory[-1]
            c_s = ode_func.last_c_s

        # Extract final quaternions directly
        quat_hat = F.normalize(current_xh_mol[:,:self.rot_dim], dim=-1, eps=1e-6)
        trans_hat = current_xh_mol[:,self.rot_dim:self.rot_dim+self.x_dim]
        angles_hat = current_xh_mol[:,self.rot_dim+self.x_dim:self.rot_dim+self.x_dim+self.angle_dim]
        
        angles_hat = angles_hat.view(-1, 7, 2)
        angles_hat = F.normalize(angles_hat, dim=-1, eps=1e-6)
        
        if 'torsion_angles_mask' in molecule:
            mask = molecule['torsion_angles_mask'].view(-1, 7, 1)
            angles_hat = angles_hat * mask
        angles_hat = angles_hat.view(-1, 14)

        h_mol_final = current_xh_mol[:,self.rot_dim+self.x_dim+self.angle_dim:] * self.norm_values[0]
        x_pro_final = protein_pocket['x']
        h_pro_final = xh_pro[:,self.rot_dim+self.x_dim:] * self.norm_values[0]
        
        T_peptide_hat = torch.cat((quat_hat, trans_hat), dim=-1)

        if not self.features_fixed:
            h_mol_final = F.one_hot(torch.argmax(current_xh_mol[:, self.x_dim:], dim=1), self.num_atoms)
        else:
            h_mol_final = molecule['h'] 

        h_pro_final = h_pro_final.unsqueeze(1).expand(-1, x_pro_final.shape[1], -1)
        xh_pro_final = torch.cat([x_pro_final, h_pro_final], dim=-1)

        print(f"quat_true: {T_peptide[0, 0, :4]}")
        print(f"quat_hat: {quat_hat[0]}")
        print(f"trans_true: {T_peptide[0, 0, 4:]}")
        print(f"trans_hat: {trans_hat[0]}")
        print(f"angles_true: {molecule['torsion_angles_sin_cos'][0][0]}")
        print(f"angles_hat: {angles_hat[0].view(7, 2)}")
        
        # Calculate final position of peptide from predicted rotaiton, translation and angles
        x_hat_mol, angles_hat, sidechain_frames = self.predict_pos(molecule, T_peptide_hat, angles_hat)
        h_mol_final = h_mol_final.unsqueeze(2).expand(-1, -1, x_hat_mol.shape[2], -1)
        x_hat_mol = x_hat_mol.reshape(-1, x_hat_mol.shape[-2], x_hat_mol.shape[-1])
        x_hat_mol = x_hat_mol.reshape(-1, x_hat_mol.shape[-1])
        
        h_mol_final = h_mol_final.reshape(-1, h_mol_final.shape[-1])

        if self.com_handling != 'no_COM':
            # Unshift coordinates back to the original frame
            idx_mol_14 = molecule['idx'].repeat_interleave(14)
            x_hat_mol = x_hat_mol + mean_shift[idx_mol_14]
            molecule['x'] = molecule['x'] + mean_shift[molecule['idx']].view(-1, 1, 3)
            
            mol_shift = mean_shift[molecule['idx']].view(num_graphs, size_mol, 3)
            if molecule['backbone_rigid_tensor'].shape[-1] == 4:
                molecule['backbone_rigid_tensor'][..., :3, 3] += mol_shift
            else:
                molecule['backbone_rigid_tensor'][..., 4:7] += mol_shift
                
            protein_pocket['x'] = protein_pocket['x'] + mean_shift[protein_pocket['idx']].view(-1, 1, 3)
            pro_shift = mean_shift[protein_pocket['idx']].view(num_graphs, size_pro, 3)
            if protein_pocket['backbone_rigid_tensor'].shape[-1] == 4:
                protein_pocket['backbone_rigid_tensor'][..., :3, 3] += pro_shift
            else:
                protein_pocket['backbone_rigid_tensor'][..., 4:7] += pro_shift

        # Mask out non-existent atoms
        if 'atom14_gt_exists' in molecule:
            mask = molecule['atom14_gt_exists'].view(-1, 1)
            x_hat_mol = x_hat_mol * mask
            molecule['x'] = molecule['x'].view(-1, 3) * mask
            molecule['x'] = molecule['x'].view(-1, 14, 3)

        xh_mol_final = torch.cat([x_hat_mol, h_mol_final], dim=-1)

        mask = molecule['cross_residues_mask'].reshape(molecule['h'].shape[0], molecule['h'].shape[1])
        
        # Time the BA prediction separately
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_before_ba = time.time()
        
        if self.ba:
            z_final_mol = torch.cat((
                quat_hat,
                trans_hat,
                angles_hat.reshape(-1, self.angle_dim),
                molecule['h'].reshape(-1, molecule['h'].shape[-1])
            ), dim=-1)
            z_final_pro = xh_pro.reshape(-1, xh_pro.shape[-1])
            t_final = torch.ones((num_graphs, 1, 1), device=device)
            
            quat_mol_true = T_peptide[:, :, :4].reshape(-1, 4)
            trans_mol_true = T_peptide[:, :, 4:].reshape(-1, 3)
            angles_mol_true = molecule['torsion_angles_sin_cos'].reshape(-1, 14)
            h_mol_true = molecule['h'].reshape(-1, molecule['h'].shape[-1])
            xh_mol_true = torch.cat([quat_mol_true, trans_mol_true, angles_mol_true, h_mol_true], dim=-1)

            _, _, _, ba_hat = self.neural_net(
                z_final_mol, z_final_pro, t_final, 
                molecule['idx'], protein_pocket['idx'], 
                molecule_pos, molecule['torsion_angles_mask'], mask
            )
            _, _, _, ba_true_struct = self.neural_net(
                xh_mol_true, z_final_pro, t_final, 
                molecule['idx'], protein_pocket['idx'], 
                molecule_pos, molecule['torsion_angles_mask'], mask
            )
        else:
            ba_hat = torch.zeros(num_graphs, device=device)
            ba_true_struct = torch.zeros(num_graphs, device=device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_after_ba = time.time()
        ba_time = t_after_ba - t_before_ba
        
        print(f"ba_hat: {ba_hat[0]}")
        ba_true = molecule['affinity']
        print(f"ba_true: {ba_true[0]}")
        ba = {"ba_hat": ba_hat, "ba_true": ba_true, "ba_true_struct": ba_true_struct, "ba_time": ba_time}

        self.safe_pdbs(xh_mol_final, molecule, run_id, data_dir, time_step='F', atom_level=True, save_peptide_only=True, fold=fold)
        self.safe_pdbs(molecule['x'].reshape(-1, 3), molecule, run_id, data_dir, time_step='GT', atom_level=True, save_peptide_only=True, fold=fold)
        
        if 'backbone_rigid_tensor' in molecule and molecule['backbone_rigid_tensor'] is not None:
            T_peptide_true = Rigid.from_tensor_4x4(molecule['backbone_rigid_tensor']).to_tensor_7()
            angles_true = molecule['torsion_angles_sin_cos']
            x_true_recon, _, _ = self.predict_pos(molecule, T_peptide_true, angles_true)
            self.safe_pdbs(x_true_recon.reshape(-1, 3), molecule, run_id, data_dir, time_step='RECON', atom_level=True, save_peptide_only=True, fold=fold)

        return (xh_mol_final, xh_pro_final, c_s, ba)
    
    def safe_pdbs(self, pos, molecule, run_id, data_dir, time_step, atom_level=False, save_peptide_only=False, fold=None):
        # Ensure we only have the first 3 dimensions (x, y, z)
        pos = pos[..., :3]
        
        for i in range(len(molecule['size'])):
            # (1) extract the peptide position
            if len(pos.shape) == 3: # [num_residues, 14, 3]
                peptide_pos = pos[molecule['idx'] == i]
            else: # [num_residues * 14, 3] or [num_residues, 3]
                if pos.shape[0] == molecule['idx'].shape[0] * 14:
                    idx_mol = molecule['idx'].repeat_interleave(14)
                    peptide_pos = pos[idx_mol == i]
                else:
                    peptide_pos = pos[molecule['idx'] == i]
            # (2) bring peptides into correct order
            peptide_idx = molecule['pos_in_seq'][molecule['idx'] == i]
            # peptide_pos_orderd = peptide_pos[peptide_idx-1] # pos starts at 1
            # (3) get graph name for elemnt in batch
            if isinstance(molecule['graph_name'], str):
                graph_name = molecule['graph_name']
            else:
                graph_name = molecule['graph_name'][i]

            if '100K' in data_dir:
                create_new_pdb_hdf5_100k(peptide_pos, peptide_idx, graph_name, run_id, data_dir, time_step=time_step, sample_id=i, atom_level=atom_level)
            else:
                create_new_pdb_hdf5_swift(peptide_pos, peptide_idx, graph_name, run_id, data_dir, time_step=time_step, sample_id=i, atom_level=atom_level, save_peptide_only=save_peptide_only, fold=fold)

    def save_trajectory_step(self, pos, molecule, run_id, t_idx, num_samples, mask, data_dir):
        # Format: sampleID_structureID_timestep.pdb
        # sampleID = graph_name (peptide ID)
        # structureID = s (0-9)
        # timestep = t_idx (0-50)
        
        pos = pos[..., :3]
        total_batch_size = len(molecule['size'])
        sample_batch_size = total_batch_size // num_samples
        
        # Target directory
        out_dir = Path('results') / 'trajectories' / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(total_batch_size):
            if not mask[i]: continue
            
            # Identify which peptide and which stochastic sample this is
            s_idx = i // sample_batch_size
            
            graph_name = molecule['graph_name'][i]
            if isinstance(graph_name, bytes):
                graph_name = graph_name.decode('utf-8')
            
            # Filename: sampleID_structureID_timestep
            pdb_filename = f"{graph_name}_{s_idx}_{t_idx:03d}.pdb"
            pdb_output_path = out_dir / pdb_filename
            
            # Extract peptide positions
            if pos.shape[0] == len(molecule['idx']) * 14: # All-atom flattened
                idx_mol = molecule['idx'].repeat_interleave(14)
                peptide_pos = pos[idx_mol == i]
            else: # Residue level
                peptide_pos = pos[molecule['idx'] == i]
                
            peptide_idx = molecule['pos_in_seq'][molecule['idx'] == i]
            
            # We need the reference data to build the full pMHC structure
            # Since create_new_pdb_hdf5_swift is already optimized for this, we'll reuse its internal logic
            # but override the path
            protein_data, peptide_data = self._get_ref_data(graph_name, data_dir)
            
            from utils import write_updated_peptide_coords_pdb_swiftmhc
            write_updated_peptide_coords_pdb_swiftmhc(
                peptide_coords=peptide_pos,
                peptide_data=peptide_data,
                protein_data=protein_data,
                pdb_output_path=str(pdb_output_path),
                atom_level=True,
                save_peptide_only=False  # Save full pMHC complex (protein + peptide)
            )

    def _get_ref_data(self, graph_name, data_dir):
        # Helper to get reference protein/peptide data from HDF5
        import h5py
        fold = "1" # Hardcoded as in utils
        possible_files = [
            Path(data_dir) / f"BA_cluster{fold}.hdf5",
            Path(data_dir) / f"train_fold{fold}.hdf5",
            Path(data_dir) / f"valid_fold{fold}.hdf5",
        ]
        for i in range(10):
            possible_files.append(Path(data_dir) / f"xray_cluster{i}.hdf5")
            
        for p in possible_files:
            if p.exists():
                with h5py.File(p, 'r') as f5:
                    if graph_name in f5:
                        group = f5[graph_name]
                        protein_data = {
                            'aatype': group['protein']['aatype'][:],
                            'atom_positions': group['protein']['all_atom_positions'][:],
                            'atom_mask': group['protein']['all_atom_mask'][:]
                        }
                        peptide_data = {
                            'aatype': group['peptide']['aatype'][:]
                        }
                        return protein_data, peptide_data
        raise KeyError(f"{graph_name} not found for reference data")


class ODEWrapper(nn.Module):
    def __init__(self, model, T_peptide, xh_pro, molecule, molecule_pos, protein_pocket, step_size, rot_dim=9, x_dim=3, angle_dim=14, angle_mask=None, variational=True):
        super().__init__()
        self.model = model
        self.T_peptide = T_peptide
        self.xh_pro = xh_pro
        self.molecule_idx = molecule['idx']
        self.protein_idx = protein_pocket['idx']
        self.molecule_pos = molecule_pos
        self.num_graphs = molecule['size'].size(0)
        self.last_c_s = None
        self.step_size = step_size
        self.molecule = molecule
        self.rot_dim = rot_dim
        self.x_dim = x_dim
        self.angle_dim = angle_dim
        self.angle_mask = angle_mask
        self.variational = variational

    def forward(self, t, xh_mol):
        t_vec = torch.full((self.num_graphs, 1, 1), fill_value=t.item(), device=xh_mol.device)

        v_hat_mol, _, c_s, _ = self.model.neural_net(
            xh_mol, self.xh_pro, t_vec, 
            self.molecule_idx, self.protein_idx, self.molecule_pos, self.angle_mask
        )
        self.last_c_s = c_s
        
        v_xh_final = torch.zeros_like(xh_mol)

        v_x = v_hat_mol[:, :self.rot_dim+self.x_dim+self.angle_dim]

        if self.variational:
            scale = torch.clip(torch.ones_like(t) / (1-t), 0, 20)
            v_x_trans = (v_x[:, self.rot_dim:self.rot_dim+self.x_dim] - xh_mol[:, self.rot_dim:self.rot_dim+self.x_dim]) * scale
            
            scale = torch.clip(torch.ones_like(t) / (1-t), 0, 10)
            scaling = 10
            v_x_rot = (v_x[:, :self.rot_dim] - xh_mol[:, :self.rot_dim]) * scale
            
            # Not correct
            v_x_angle = (v_x[:, self.rot_dim+self.x_dim:self.rot_dim+self.x_dim+self.angle_dim] - xh_mol[:, self.rot_dim+self.x_dim:self.rot_dim+self.x_dim+self.angle_dim]) * scale

            v_x = torch.cat([v_x_rot, v_x_trans, v_x_angle], dim=-1)

        # v_x = v_x - scatter_mean(v_x, self.molecule_idx, dim=0)[self.molecule_idx] # TODO: com handling testing?
        
        v_xh_final[:, :self.rot_dim+self.x_dim+self.angle_dim] = v_x
        
        return v_xh_final
    