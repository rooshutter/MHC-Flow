import torch
import numpy as np
import torch.nn as nn
from torch_scatter import scatter_mean

from torch_geometric.data import Data, Batch

from model.egnn import EGNN, GNN
from model.egnn_all_atom import EGNN_all_atom
from model.egnn_all_atom_quat import EGNN_all_atom as EGNN_all_atom_quat
from model.positional_encoding import sin_pE
from model.confidence_score import Confidence_Score

"""
This file sets up the neural network for the generative framework.
"""

class NN_Model(nn.Module):

    def __init__(
            self,
            architecture: str,
            features_fixed: bool,
            confidence_score: bool,
            position_encoding: bool,
            position_encoding_dim: int,
            network_params,
            num_atoms: int,
            num_residues: int,
            device: str,
            all_atom: bool = False,
            variational: bool = False,
            ba: bool = False,
            use_quat: bool = False,
    ):
        
        """
        Parameters:

        architecture: name of the neural network to use
        features_fixed: option to train to generate new peptides, instead of just structures (not implemented)
        position_encoding: switch for the positional encoding
        position_encoding_dim: size of the positional encoding
        network_params: parameters for the specific neural network architecture
        num_atoms: number of amino-acid types in peptide
        num_residues: number of amino-acid types in protein pocket
        device: hardware to run on (should be a gpu)
        
        """

        super().__init__()

        self.architecture = architecture
        self.features_fixed = features_fixed
        self.all_atom = all_atom
        self.variational = variational
        self.ba = ba
        self.use_quat = use_quat
        self.x_dim = 3
        self.rot_dim = 4 if use_quat else 9
        self.angle_dim = 14
        self.act_fn = nn.SiLU()

        self.joint_dim = network_params.joint_dim
        self.hidden_dim = network_params.hidden_dim
        self.num_layers = network_params.num_layers
        self.conditioned_on_time = network_params.conditioned_on_time

        # positional encoding
        self.position_encoding = position_encoding
        self.pE_dim = position_encoding_dim

        # edge parameters
        self.edge_embedding_dim = network_params.edge_embedding_dim
        self.edge_cutoff_l = network_params.edge_cutoff_ligand
        self.edge_cutoff_p = network_params.edge_cutoff_pocket
        self.edge_cutoff_i = network_params.edge_cutoff_interaction

        if architecture == 'egnn' or 'gnn':

            # edge embedding
            self.edge_embedding_dim = network_params.edge_embedding_dim
            if self.edge_embedding_dim is not None: 
                self.edge_embedding = nn.Embedding(self.x_dim, self.edge_embedding_dim)
            else: 
                self.edge_embedding = None
            self.edge_embedding_dim = 0 if self.edge_embedding_dim is None else self.edge_embedding_dim

            # node embedding
            self.atom_encoder = nn.Sequential(
                nn.Linear(num_atoms, 2 * num_atoms),
                self.act_fn,
                nn.Linear(2 * num_atoms, self.joint_dim)
            )

            self.atom_decoder = nn.Sequential(
                nn.Linear(self.joint_dim, 2 * num_atoms),
                self.act_fn,
                nn.Linear(2 * num_atoms, num_atoms)
            )

            self.residue_encoder = nn.Sequential(
                nn.Linear(num_residues, 2 * num_residues),
                self.act_fn,
                nn.Linear(2 * num_residues, self.joint_dim)
            )

            self.residue_decoder = nn.Sequential(
                nn.Linear(self.joint_dim, 2 * num_residues),
                self.act_fn,
                nn.Linear(2 * num_residues, num_residues)
            )

            if self.position_encoding:
                self.joint_dim += self.pE_dim

            if self.conditioned_on_time:
                self.joint_dim += 1

            self.confidence_score = confidence_score
            if confidence_score:
                self.confidence_scores = Confidence_Score(3 + self.hidden_dim)

            if architecture == 'egnn':
                if self.all_atom:
                    EGNNClass = EGNN_all_atom_quat if self.use_quat else EGNN_all_atom
                    self.egnn = EGNNClass(in_node_nf=self.joint_dim, in_edge_nf=self.edge_embedding_dim,
                                 hidden_nf=self.hidden_dim, device=device, act_fn=self.act_fn,
                                 n_layers=self.num_layers, attention=network_params.attention, tanh=network_params.tanh,
                                 norm_constant=network_params.norm_constant,
                                 inv_sublayers=network_params.inv_sublayers, sin_embedding=network_params.sin_embedding,
                                 normalization_factor=network_params.normalization_factor,
                                 aggregation_method=network_params.aggregation_method,
                                 reflection_equiv=network_params.reflection_equivariant, all_atom=self.all_atom, variational=self.variational, ba=self.ba) 
                else:
                    self.egnn = EGNN(in_node_nf=self.joint_dim, in_edge_nf=self.edge_embedding_dim,
                                    hidden_nf=self.hidden_dim, device=device, act_fn=self.act_fn,
                                    n_layers=self.num_layers, attention=network_params.attention, tanh=network_params.tanh,
                                    norm_constant=network_params.norm_constant,
                                    inv_sublayers=network_params.inv_sublayers, sin_embedding=network_params.sin_embedding,
                                    normalization_factor=network_params.normalization_factor,
                                    aggregation_method=network_params.aggregation_method,
                                    reflection_equiv=network_params.reflection_equivariant, all_atom=self.all_atom) # edge_sin_attr=self.edge_sin_attrs
            else:
                
                self.gnn = GNN(in_node_nf=self.joint_dim + self.x_dim, in_edge_nf=self.edge_embedding_dim,
                               hidden_nf=self.hidden_dim, out_node_nf=self.x_dim + self.joint_dim,
                               device=device, act_fn=self.act_fn, n_layers=self.num_layers,
                               attention=network_params.attention, normalization_factor=network_params.normalization_factor,
                               aggregation_method=network_params.aggregation_method)

        else:
            raise Exception(f"Wrong architecture {architecture}")


    def forward(self, z_t_mol, z_t_pro, t, molecule_idx, protein_pocket_idx, molecule_pos=None, angle_mask=None, mask=None):

        '''
        Takes in noised sample and outputs predicted added noise

        Inputs:
        z_t_mol: size = [batch_node_dim_mol, x_dim + num_atoms + position_encoding_dim]
        z_t_pro: size = [batch_node_dim_mol, x_dim + num_residues + position_encoding_dim]
        t: size = [batch_node_dim]
        molecule['idx']: size = [batch_node_dim]
        protein_pocket['idx']: size = [batch_node_dim]

        return epsilon_hat_mol: size = [batch_node_dim_mol, x + num_atoms], 
                epsilon_hat_pro: size = [batch_node_dim_pro, x + num_residues]
        '''
        if self.all_atom:
            if z_t_mol.dim() == 3:
                mol_dim = z_t_mol.shape[0] * z_t_mol.shape[1]
            else:
                mol_dim = z_t_mol.shape[0]
                
            z_t_mol_rot = z_t_mol.reshape(-1, z_t_mol.shape[-1])[:, :self.rot_dim]
            z_t_pro_rot = z_t_pro.reshape(-1, z_t_pro.shape[-1])[:, :self.rot_dim]
            rot = torch.cat((z_t_mol_rot, z_t_pro_rot), dim=0)
            z_t_mol_angles = z_t_mol.reshape(-1, z_t_mol.shape[-1])[:, self.rot_dim+self.x_dim:self.rot_dim+self.x_dim+self.angle_dim]
            z_t_pro_angles = torch.zeros_like(z_t_pro.reshape(-1, z_t_pro.shape[-1])[:, self.rot_dim+self.x_dim:self.rot_dim+self.x_dim+self.angle_dim])
            angles = torch.cat((z_t_mol_angles, z_t_pro_angles), dim=0)

        else:
            q = None
        if self.all_atom:
            z_t_mol = z_t_mol.reshape(-1, z_t_mol.shape[-1]) 
            z_t_pro = z_t_pro.reshape(-1, z_t_pro.shape[-1]) 
            t = t.squeeze(-1)
        

        idx_joint = torch.cat((molecule_idx, protein_pocket_idx), dim=0)
        if self.all_atom:
            x_mol = z_t_mol[:,self.rot_dim:self.x_dim+self.rot_dim].clone()
            x_pro = z_t_pro[:,self.rot_dim:self.x_dim+self.rot_dim].clone()
        else:
            x_mol = z_t_mol[:,:self.x_dim].clone()
            x_pro = z_t_pro[:,:self.x_dim].clone()

        # add edges to the graph (edges are determined by distance cutoffs)
        edges = self.get_edges(molecule_idx, protein_pocket_idx, x_mol, x_pro)
        assert torch.all(idx_joint[edges[0]] == idx_joint[edges[1]])

        if self.architecture == 'egnn' or self.architecture == 'gnn':

            # encode z_t_mol, z_t_pro (possible need to .clone() the inputs)
            if self.all_atom:                
                # check if there is nan in ztmol:
                if torch.any(torch.isnan(z_t_mol)):
                    raise ValueError("NaN detected in z_t_mol")
                h_mol = self.atom_encoder(z_t_mol[:,self.x_dim+self.rot_dim+self.angle_dim:]).clone()
            else:
                h_mol = self.atom_encoder(z_t_mol[:,self.x_dim:]).clone()
            if self.all_atom:                
                h_pro = self.residue_encoder(z_t_pro[:,self.x_dim+self.rot_dim:]).clone()
            else:
                h_pro = self.residue_encoder(z_t_pro[:,self.x_dim:]).clone()
  
            # position_encoding
            if self.position_encoding:
                # TODO: molecule_pos[molecule_idx] not correct !!!
                pE = sin_pE(molecule_pos, self.pE_dim)
                h_mol = torch.cat([h_mol, pE], dim=1)
                h_pro = torch.cat([h_pro, torch.zeros((h_pro.shape[0], self.pE_dim), device=h_pro.device)], dim=1)

            # combine molecule and protein in joint space
            if self.all_atom:
                x_joint = torch.cat((z_t_mol[:,self.rot_dim:self.rot_dim+self.x_dim], z_t_pro[:,self.rot_dim:self.rot_dim+self.x_dim]), dim=0) # [batch_node_dim_mol + batch_node_dim_pro, 3]
            else:
                x_joint = torch.cat((z_t_mol[:,:self.x_dim], z_t_pro[:,:self.x_dim]), dim=0) # [batch_node_dim_mol + batch_node_dim_pro, 3]
            h_joint = torch.cat((h_mol, h_pro), dim=0) # [batch_node_dim_mol + batch_node_dim_pro, joint_dim]
 
            # add time conditioning
            if self.conditioned_on_time:
                h_time = t[idx_joint]
                if self.all_atom:
                    h_joint = torch.cat([h_joint, h_time], dim=1)
                else:
                    h_joint = torch.cat([h_joint, h_time], dim=1)
                

            # add edge embedding and types
            if self.edge_embedding_dim > 0:
                # 0: ligand-pocket, 1: ligand-ligand, 2: pocket-pocket
                edge_types = torch.zeros(edges.size(1), dtype=int, device=edges.device)
                edge_types[(edges[0] < len(molecule_idx)) & (edges[1] < len(molecule_idx))] = 1
                edge_types[(edges[0] >= len(molecule_idx)) & (edges[1] >= len(molecule_idx))] = 2

                # Learnable embedding
                edge_types = self.edge_embedding(edge_types)
            else:
                edge_types = None
            
            ba = None

            if self.architecture == 'egnn':

                # choose whether to get protein_pocket corrdinates fixed
                protein_pocket_fixed = torch.cat((torch.ones_like(molecule_idx), torch.zeros_like(protein_pocket_idx))).unsqueeze(1)

                # neural net forward pass
                if self.all_atom:
                    h_new, x_new, h_last_layer, rot, angles, ba = self.egnn(h_joint, x_joint, edges,
                                                            update_coords_mask=protein_pocket_fixed,
                                                            batch_mask=idx_joint, edge_attr=edge_types, 
                                                            rot=rot, angles=angles, mol_dim=mol_dim,
                                                            angle_mask=angle_mask, mask=mask)
                else:
                    h_new, x_new, h_last_layer = self.egnn(h_joint, x_joint, edges,
                                            update_coords_mask=protein_pocket_fixed,
                                            batch_mask=idx_joint, edge_attr=edge_types)
                
                # calculate displacement vectors
                displacement_vec = (x_new - x_joint) # TODO is this needed?

                if self.all_atom:
                    rot_mol = rot[:len(molecule_idx)]
                    rot_pro = rot[len(molecule_idx):]
                    angle_mol = angles[:len(molecule_idx)]
                    angle_pro = angles[len(molecule_idx):]
                    displacement_vec = x_new

            elif self.architecture == 'gnn':

                # GNN
                x_h_joint = torch.cat([x_joint, h_joint], dim=1)
                out = self.gnn(x_h_joint, edges, node_mask=None, edge_attr=edge_types)
                displacement_vec = out[:, :self.x_dim]
                h_new = out[:, self.x_dim:]

            else:
                raise Exception(f"Wrong architecture {self.architecture}")

        else:
            raise Exception(f"Wrong architecture {self.architecture}")

        # TODO: confidence calculation on peptide nodes
        if self.confidence_score:
            c_s_input = torch.cat((displacement_vec[:len(molecule_idx)], h_last_layer[:len(molecule_idx)]), dim=1)
            c_s = self.confidence_scores(c_s_input, molecule_idx)
        else:
            c_s = 0

        # remove time dim
        if self.conditioned_on_time:
            # Slice off last dimension which represented time.
            h_new = h_new[:, :-1]

        # remove position information
        if self.position_encoding:
            # Slice off last dimension which represented postional encoding.
            h_new = h_new[:, :-self.pE_dim]

        # decode h_new
        h_new_mol = self.atom_decoder(h_new[:len(molecule_idx)])
        h_new_pro = self.residue_decoder(h_new[len(molecule_idx):])

        # might not be necessary but let's see
        if torch.any(torch.isnan(displacement_vec)):
            raise ValueError("NaN detected in translation vector")
        if self.all_atom:
            if torch.any(torch.isnan(rot)):
                raise ValueError("NaN detected in rotation")
            if torch.any(torch.isnan(angles)):
                raise ValueError("NaN detected in angles")
        
        # output
        if self.all_atom:
            epsilon_hat_mol = torch.cat((rot_mol, displacement_vec[:len(molecule_idx)], angle_mol, h_new_mol), dim=1)
            epsilon_hat_pro = torch.cat((rot_pro, displacement_vec[len(molecule_idx):], angle_pro, h_new_pro), dim=1)

        else:
            epsilon_hat_mol = torch.cat((displacement_vec[:len(molecule_idx)], h_new_mol), dim=1)
            epsilon_hat_pro = torch.cat((displacement_vec[len(molecule_idx):], h_new_pro), dim=1)

        # Predict BA from final peptide embeddings via dedicated head
        if self.ba:
            ba = ba[:len(molecule_idx)] 
            ba = scatter_mean(ba.squeeze(-1), molecule_idx, dim=0) 
            

        return epsilon_hat_mol, epsilon_hat_pro, c_s, ba
    
    def get_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket): 
        '''
        get edges based on edge cutoff
        ''' 
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)

        if self.edge_cutoff_p is not None:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)

        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)

        adj = torch.cat((torch.cat((adj_ligand, adj_cross), dim=1),
                         torch.cat((adj_cross.T, adj_pocket), dim=1)), dim=0)
        edges = torch.stack(torch.where(adj), dim=0)

        return edges