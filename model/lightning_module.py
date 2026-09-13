import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pathlib import Path

FLOAT_TYPE = torch.float32
INT_TYPE = torch.int64

from dataset_8k_xray import PDB_Dataset
from dataset_8k_xray import PDB_Dataset_all, PDB_Dataset_combine, PDB_Dataset_swift
from dataset_100k_xray import PDB_Dataset_Mixed

from model.diffusion_model import Conditional_Diffusion_Model
from model.architecture import NN_Model
from model.flow_matching_model import Flow_Matching_Model
from model.flow_matching_model_all_atom import Flow_Matching_Model_all_atom
from model.flow_matching_model_all_atom_quat import Flow_Matching_Model_all_atom as Flow_Matching_Model_all_atom_quat
from model.flow_matching_model_all_atom_var import Flow_Matching_Model_all_atom_var # not a correct implementation

import numpy as np
import os
import h5py

"""
This file implements the 3D-structure prediction for a moelcule 
and a protein with a conditional diffusion model.
"""

class Structure_Prediction_Model(pl.LightningModule):

    def __init__(
            self,
            dataset: str,
            data_dir: str,
            dataset_params: dict,
            task_params: dict,
            generative_model: str,
            generative_model_params: dict,
            architecture: str,
            network_params: dict,
            batch_size: int,
            lr: float,
            num_workers: int,
            device,
            all_atom: bool = False,
            run_name: str = "default",
            variational: bool = True,
            solver: str = "euler",
            ba: bool = False,
            use_quat: bool = False,
    ):
        """
        Parameters:

        All of these parameters are set in the config file

        dataset: name of the dataset
        data_dir: path to the data directory folder with train, valid and test hdf5 files
        dataset_params: dataset parameters (see diffusion_model for details)
        task_params: task parameters (see diffusion_model for details)
        generative_model: name of the generative framework (only conditional diffusion currently available)
        generative_model_params: generative model parameters (see diffusion_model for details)
        architecture: neural network model name
        network_params: neural network parameters (see architecture for details)
        batch_size: batch size 
            in training: size of stochastic min-batches for learning
            in sampling: size of parallel generation
        lr: learning rate
        num_workers: number of workers
        device: hardware to run on (should be a gpu)

        
        """
        
        super().__init__()

        # set a seed
        torch.manual_seed(42)

        # choose the generative framework
        frameworks = {'conditional_diffusion': Conditional_Diffusion_Model, 'flow_matching': Flow_Matching_Model, 'flow_matching_all_atom': Flow_Matching_Model_all_atom_quat if use_quat else Flow_Matching_Model_all_atom, 'flow_matching_all_atom_var': Flow_Matching_Model_all_atom_var}
        assert generative_model in frameworks

        # choose the neural net architecture
        self.neural_net = NN_Model(
            # model parameters
            architecture,
            task_params.features_fixed,
            task_params.confidence_score,
            generative_model_params.position_encoding,
            generative_model_params.position_encoding_dim,
            network_params,
            dataset_params.num_atoms,
            dataset_params.num_residues,
            device,
            all_atom,
            variational,
            ba=ba,
            use_quat=use_quat,
        )

        self.model = frameworks[generative_model](
            # framework parameters
            self.neural_net,
            task_params.features_fixed,
            task_params.confidence_score,
            generative_model_params.timesteps,
            generative_model_params.position_encoding,
            generative_model_params.com_handling,
            generative_model_params.sampling_stepsize,
            generative_model_params.noise_scaling,
            generative_model_params.high_noise_training,
            dataset_params.num_atoms,
            dataset_params.num_residues,
            dataset_params.norm_values,
            all_atom,
            variational,
            solver,
            ba=ba,
        )
        
        self.dataset = dataset
        self.data_dir = data_dir
        self.fold = getattr(dataset_params, 'fold', "1")
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.all_atom = all_atom
        self.run_name = run_name
        self.last_saved_epoch = -1

    # Data section

    def setup(self, stage):

        if self.dataset == 'pmhc_8K_xray':

            if not self.all_atom:
                if stage == 'fit':
                    self.train_dataset = PDB_Dataset(self.data_dir, 'train', fold=str(self.fold), all_atom=self.all_atom)
                    self.val_dataset = PDB_Dataset(self.data_dir, 'valid', fold=str(self.fold), all_atom=self.all_atom)
                elif stage == 'test':
                    self.test_dataset = PDB_Dataset(self.data_dir, 'test', fold=str(self.fold), all_atom=self.all_atom)

            if self.all_atom:
                if stage == 'fit':
                    self.train_dataset = PDB_Dataset_swift("/scratch-shared/roos/preprocessed/", 'train', fold=str(self.fold))
                    self.val_dataset = PDB_Dataset_swift("/scratch-shared/roos/preprocessed/", 'valid', fold=str(self.fold))
                elif stage == 'test':
                    self.test_dataset = PDB_Dataset_swift("/scratch-shared/roos/preprocessed/", 'BA', fold=str(self.fold))

        elif self.dataset == 'pmhc_100K_xray':

            if stage == 'fit':
                self.train_dataset = PDB_Dataset_Mixed(self.data_dir, '100k_train')
                self.val_dataset = PDB_Dataset_Mixed(self.data_dir, '100k_valid')
            elif stage == 'test':
                self.test_dataset = PDB_Dataset_Mixed(self.data_dir, '100k_test')

    def train_dataloader(self):
        # Need to pick a dataloader (geometric or normal)
        # what does pin_memory do?
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self.train_dataset.collate_fn,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.val_dataset.collate_fn,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.test_dataset.collate_fn,
                          pin_memory=True)
    
    def get_molecule_and_protein(self, data):
        '''
        function to unpack the molecule and it's protein
        '''
        molecule = {
            'x': data['peptide_positions'].to(self.device, FLOAT_TYPE),
            'h': data['peptide_features'].to(self.device, FLOAT_TYPE),
            'size': data['num_peptide_residues'].to(self.device, INT_TYPE),
            'idx': data['peptide_idx'].to(self.device, INT_TYPE),
            'pos_in_seq': data['pos_in_seq'].to(self.device, INT_TYPE),
            'graph_name': data['graph_name'],
            'backbone_rigid_tensor': data['peptide_backbone_rigid_tensor'].to(self.device, FLOAT_TYPE) if 'peptide_backbone_rigid_tensor' in data else None,
            'torsion_angles_sin_cos': data['peptide_torsion_angles_sin_cos'].to(self.device, FLOAT_TYPE) if 'peptide_torsion_angles_sin_cos' in data else None,
            'aatype': data['peptide_aatype'].to(self.device, INT_TYPE) if 'peptide_aatype' in data else None,
            "atom14_gt_exists": data["peptide_atom14_gt_exists"].to(self.device) if "peptide_atom14_gt_exists" in data else None,
            "atom14_alt_gt_positions": data["peptide_atom14_alt_gt_positions"].to(self.device, FLOAT_TYPE) if "peptide_atom14_alt_gt_positions" in data else None,
            "alt_torsion_angles_sin_cos": data["peptide_alt_torsion_angles_sin_cos"].to(self.device, FLOAT_TYPE) if "peptide_alt_torsion_angles_sin_cos" in data else None,
            "cross_residues_mask": data["peptide_cross_residues_mask"].to(self.device, INT_TYPE) if "peptide_cross_residues_mask" in data else None,
            "affinity": data["affinity"].to(self.device, FLOAT_TYPE) if "affinity" in data else None,
            "ba_mask": data["ba_mask"].to(self.device) if "ba_mask" in data else None,
            "torsion_angles_mask": data["peptide_torsion_angles_mask"].to(self.device, FLOAT_TYPE) if "peptide_torsion_angles_mask" in data else None,
            "residue_index": data["peptide_residue_index"].to(self.device, INT_TYPE) if "peptide_residue_index" in data else None,
        }

        protein_pocket = {
            'x': data['protein_pocket_positions'].to(self.device, FLOAT_TYPE),
            'h': data['protein_pocket_features'].to(self.device, FLOAT_TYPE),
            'size': data['num_protein_pocket_residues'].to(self.device, INT_TYPE),
            'idx': data['protein_pocket_idx'].to(self.device, INT_TYPE),
            'backbone_rigid_tensor': data['protein_backbone_rigid_tensor'].to(self.device, FLOAT_TYPE) if 'protein_backbone_rigid_tensor' in data else None,
            'aatype': data['protein_aatype'].to(self.device, INT_TYPE) if 'protein_aatype' in data else None,
            'cross_residues_mask': data['protein_cross_residues_mask'].to(self.device, INT_TYPE) if 'protein_cross_residues_mask' in data else None,
            "residue_index": data["protein_residue_index"].to(self.device, INT_TYPE) if "protein_residue_index" in data else None,
        }
                
        return (molecule, protein_pocket)
        

    # training section

    def training_step(self, data_batch):
        save_pdb = False
        if self.current_epoch > self.last_saved_epoch:
            save_pdb = True
            self.last_saved_epoch = self.current_epoch

        mol_pro_batch = self.get_molecule_and_protein(data_batch)
        # TODO: could add augment_noise and augment_rotation but excluded in DiffDock
        loss, info = self.model(mol_pro_batch, 
                                current_epoch=self.current_epoch, 
                                max_epochs=self.trainer.max_epochs,
                                run_id=self.run_name,
                                data_dir=self.data_dir,
                                save_pdb=save_pdb)
        self.log('train_loss', loss)

        for key, value in info.items():
            self.log(key, value)

        return loss

    def validation_step(self, data_batch, *args):
        mol_pro_batch = self.get_molecule_and_protein(data_batch)
        loss, info = self.model(mol_pro_batch, 
                                current_epoch=self.current_epoch, 
                                max_epochs=self.trainer.max_epochs,
                                run_id=self.run_name,
                                data_dir=self.data_dir,
                                save_pdb=False)
        self.log('val_loss', loss)

        for key, value in info.items():
            val_key = key + '_val'
            self.log(val_key, value)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-4)

        if self.all_atom:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss", 
                    "interval": "epoch", 
                    "frequency": 1,
                },
            }
        else:
            return {
                "optimizer": optimizer,
            }

    



    


