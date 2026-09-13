import torch
import numpy as np
import math
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
# from zmq import device

from model.noise_schedule import Noise_Schedule
from utils import create_new_pdb_hdf5, create_new_pdb_hdf5_100k

from torchdiffeq import odeint_adjoint as odeint

from tools.rigid import Rigid


class Flow_Matching_Model(nn.Module):
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

        # Noise Schedule
        self.noise_schedule = Noise_Schedule(self.T)

        # Further model hyperparameters
        if noise_scaling == None:
            self.noise_scaling = 1
        else:
            self.noise_scaling = noise_scaling
        self.high_noise_training = high_noise_training

        self.confidence_score = confidence_score

        self.all_atom = all_atom

    def forward(self, z_data, current_epoch=None, max_epochs=None, run_id=None, data_dir=None, save_pdb=False):

        molecule, protein_pocket = z_data

        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None

        if self.all_atom:
            z_t_mol, z_t_pro, v_x_mol, v_pro, t = self.compute_flow_match_all_atom(z_data)
        else:
            z_t_mol, z_t_pro, v_x_mol, v_pro, t = self.compute_flow_match(z_data)

        print(f"{t.shape=}")

        if self.noise_scaling > 0:
            z_t_mol = z_t_mol + torch.randn_like(z_t_mol) * self.noise_scaling

        
        v_hat_mol, v_hat_pro, c_s, _ = self.neural_net(z_t_mol, z_t_pro, t, molecule['idx'], protein_pocket['idx'], molecule_pos)

        
        if self.training:
            loss, info = self.train_loss(molecule, z_t_mol, v_x_mol, 
                                        v_hat_mol, protein_pocket, 
                                        z_t_pro, v_pro, v_hat_pro, t, c_s)
        else: 
            loss, info = self.validation_loss(z_data, molecule, z_t_mol, v_x_mol, 
                                        v_hat_mol, protein_pocket, 
                                        z_t_pro, v_pro, v_hat_pro, t)

        return loss.mean(0), info
    

    def compute_flow_match(self, z_data, t_is_0 = False):

        molecule, protein_pocket = z_data
        batch_size = molecule['size'].size(0)
        device = molecule['x'].device
    
        # normalisation with norm_values (dataset dependend) -> changes likelyhood (adjusted for in vlb)!
        molecule['x'] = molecule['x'] / self.norm_values[0]
        molecule['h'] = molecule['h'] / self.norm_values[1]
        protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]

        # sample t ~ U(0,...,T) for each graph individually
        t_low = 0 if self.train else 1
        t = torch.randint(t_low, self.T + 1, size=(batch_size, 1), device=device)

        # normalize t
        t = t / self.T

        # option for computing t = 0 representations
        t = torch.zeros((batch_size, 1), device=device) if t_is_0 else t
        
        xh_mol = torch.cat((molecule['x'], molecule['h']), dim=1)
        xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=1)

        # center of mass handling
        if self.com_handling == 'both':
            xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]
            xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
        elif self.com_handling == 'no_COM':
            dumy_variable = 0
        else:
            # data is translated to 0, COM noise added and again translated to 0
            mean = scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)
            xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - mean[molecule['idx']]
            xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

        
        z_x_mol = torch.randn(size=(len(xh_mol), self.x_dim), device=device) #* self.noise_scaling
        
        z_x_pro = torch.zeros(size=(len(xh_pro), self.x_dim), device=device)


        if self.features_fixed:
            z_h_mol = torch.zeros(size=(len(xh_mol), self.num_atoms), device=device)
            z_h_pro = torch.zeros(size=(len(xh_pro), self.num_residues), device=device)
        else:
            # for h we need standard normal noise (this would be sampling new peptides)
            z_h_mol = torch.randn(size=(len(xh_mol), self.num_atoms), device=device)
            z_h_pro = torch.randn(size=(len(xh_pro), self.num_residues), device=device)

        # z_mol = torch.cat((z_x_mol, z_h_mol), dim=1)
        z_pro = torch.cat((z_x_pro, z_h_pro), dim=1)


        z_t_mol_x = (1 - t[molecule['idx']]) * xh_mol[:,:self.x_dim] + (t[molecule['idx']]) * z_x_mol
        z_t_mol = torch.cat((z_t_mol_x, z_h_mol), dim=1)
        z_t_pro = xh_pro.clone().detach()


        if self.com_handling == 'both':
            dumy_variable = 0
        elif self.com_handling == 'no_COM':
            dumy_variable = 0
        else:
            # data is translated to 0, COM noise added and again translated to 0 (turn off for old centering approach)
            mean = scatter_mean(z_t_mol[:,:self.x_dim], molecule['idx'], dim=0)
            z_t_mol[:,:self.x_dim] = z_t_mol[:,:self.x_dim] - mean[molecule['idx']]
            z_t_pro[:,:self.x_dim] = z_t_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

        v_x_mol = xh_mol[:,:self.x_dim] - z_x_mol 

        v_pro = xh_pro - z_pro

        if self.com_handling == 'both':
            dumy_variable = 0
        elif self.com_handling == 'no_COM':
            dumy_variable = 0
        else:
            mean = scatter_mean(v_x_mol, molecule['idx'], dim=0)
            v_x_mol = v_x_mol - mean[molecule['idx']]
            v_pro[:,:self.x_dim] = v_pro[:,:self.x_dim] - mean[protein_pocket['idx']]


        return z_t_mol, z_t_pro, v_x_mol, v_pro, t

    def train_loss(
            self, molecule, z_t_mol, v_x_mol, 
            v_hat_mol, protein_pocket, 
            z_t_pro, v_pro, v_hat_pro, t, c_s
    ):
        
        # compute the sum squared error loss per graph # TODO: modified to not take the h_dims
        error_mol = scatter_add(torch.sum((v_x_mol[:,:3] - v_hat_mol[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        error_pro = torch.zeros(protein_pocket['size'].size(0), device=molecule['x'].device)

        kl_prior = self.kl_prior(molecule)

        # Add a SNR modulation term to upweight highly noised samples (default: turned off)
        # SNR_t = (1 / self.SNR_t(t).squeeze(1))

        # t = 0 and t != 0 masks for seperate computation of log p(x | z0)
        t_0_mask = (t == 0).float().squeeze()
        t_not_0_mask = 1 - t_0_mask

        # likelyhood of drawing our structure from our completley denoised distribution
        loss_x_mol_t0, loss_x_protein_t0, loss_h_t0 = self.loss_t0(
            molecule, z_t_mol, v_x_mol, v_hat_mol,
            protein_pocket, z_t_pro, v_pro, v_hat_pro, t
        )

        # seperate loss computation for t = 0 and t != 0
        loss_x_mol_t0 = - loss_x_mol_t0 * t_0_mask
        loss_x_protein_t0 = - loss_x_protein_t0 * t_0_mask
        loss_h_t0 = - loss_h_t0 * t_0_mask
        error_mol = error_mol * t_not_0_mask
        error_pro = error_pro * t_not_0_mask

        # Normalize loss_t by graph size
        error_mol = error_mol / ((self.x_dim) * molecule['size'])
        error_pro = error_pro / ((self.x_dim + self.num_residues * protein_pocket['size']))
        loss_t = 0.5 * (error_mol + error_pro) # * SNR_t

        # Normalize loss_0 by graph size
        loss_x_mol_t0 = loss_x_mol_t0 / (self.x_dim * molecule['size'])
        loss_x_protein_t0 = loss_x_protein_t0 / (self.x_dim * protein_pocket['size'])
        loss_0 = loss_x_mol_t0 + loss_x_protein_t0 + loss_h_t0

        loss = loss_t + loss_0 + kl_prior
        # loss = loss_t + kl_prior

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
            'error_mol': error_mol.mean(0),
            'loss_x_mol_t0': loss_x_mol_t0.mean(0),
            'kl_prior': kl_prior.mean(0),
            'confidence': c_s_peptide.mean(0),
            'loss_with_conf': loss_with_conf.mean(0)
        }

        if self.confidence_score == True:
            return loss_with_conf, info

        return loss, info
    
    def validation_loss(
            self, z_data, molecule, z_t_mol, v_x_mol, v_hat_mol,
            protein_pocket, z_t_pro, v_pro, v_hat_pro, 
            t,
    ):
        
        ### Additional evaluation (VLB) variables
        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None

        # compute the sum squared error loss per graph
        error_mol = scatter_add(torch.sum((v_x_mol[:,:3] - v_hat_mol[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        error_pro = torch.zeros(protein_pocket['size'].size(0), device=molecule['x'].device)

        kl_prior = self.kl_prior(molecule)

        # if pocket not fixed then molecule['size'] + protein_pocket['size']
        neg_log_const = self.neg_log_const(molecule['size'], molecule['size'].size(0), device=molecule['x'].device)
        delta_log_px = self.delta_log_px(molecule['size'])

        # SNR is computed between timestep s and t (with s = t-1)
        SNR_weight = (1 - self.SNR_s_t(t).squeeze(1))

        # TODO: add log_pN computation using the dataset histogram
        log_pN = self.log_pN(molecule['size'], protein_pocket['size'])

        # TODO optional: can add auxiliary loss / lennard-jones potential

        ## For evaluation we want to compute t = 0 losses for all z_data samples that we have

        # compute noised sample for t = 0
        # z_0_mol, z_0_pro, epsilon_0_mol, epsilon_0_pro, t_0 = self.noise_process(z_data, t_is_0 = True)
        z_0_mol, z_0_pro, v_target_0_mol, v_target_0_pro, t_0 = self.compute_flow_match(z_data, t_is_0 = True)

        # use neural network to predict noise for t = 0
        # v_hat_mol, v_hat_pro, c_s = self.neural_net(z_t_mol, z_t_pro, t, molecule['idx'], protein_pocket['idx'], molecule_pos)
        # epsilon_hat_0_mol, epsilon_hat_0_pro, _ = self.neural_net(z_0_mol, z_0_pro, t_0, molecule['idx'], protein_pocket['idx'], molecule_pos)
        v_hat_0_mol, v_hat_0_pro, _, _ = self.neural_net(z_0_mol, z_0_pro, t_0, molecule['idx'], protein_pocket['idx'], molecule_pos)

        loss_x_mol_t0, loss_x_protein_t0, loss_h_t0 = self.loss_t0(
            molecule, z_0_mol, v_target_0_mol, v_hat_0_mol,
            protein_pocket, z_0_pro, v_target_0_pro, v_hat_0_pro, t_0
        )

        loss_x_mol_t0 = - loss_x_mol_t0
        loss_x_protein_t0 = - loss_x_protein_t0
        loss_h_t0 = - loss_h_t0


        # loss_t = - self.T * 0.5 * SNR_weight * (error_mol + error_pro)
        loss_t = self.T * 0.5 * (error_mol + error_pro)
        loss_0 = loss_x_mol_t0 + loss_x_protein_t0 + loss_h_t0
        loss_0 = loss_0 + neg_log_const

        # Two added loss terms for vlb
        loss = loss_t + loss_0 + kl_prior - delta_log_px - log_pN
        # loss = loss_t + kl_prior

        info = {
            'loss_t': loss_t.mean(0),
            'loss_0': loss_0.mean(0),
            'error_mol': error_mol.mean(0),
            'loss_x_mol_t0': loss_x_mol_t0.mean(0),
            'kl_prior': kl_prior.mean(0),
            'neg_log_const': neg_log_const.mean(0),
            'SNR_weight': SNR_weight.mean(0)
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

        v_target_mol_x = v_target_mol[:,:self.x_dim]
        v_hat_mol_x = v_hat_mol[:,:self.x_dim]
        loss_x_mol_t0 = - 0.5 * scatter_add(torch.sum((v_target_mol_x - v_hat_mol_x)**2, dim=-1), molecule['idx'], dim=0)

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

        molecule['x'] = molecule['x'] - scatter_mean(molecule['x'], molecule['idx'], dim=0)[molecule['idx']]

        T_normalized = torch.ones((len(molecule['size']), 1), device=device)

        alpha_T = 0.0  # At t=1, (1-t) is 0
        sigma_T_val = 1.0

        mu_x_mol = molecule['x'] * alpha_T # [:,3]
        
        sigma_T_x = torch.full((len(molecule['size']),), fill_value=sigma_T_val, device=device)

        # KL computation h (if features are diffused)
        kl_h = 0
  
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
        alpha2_t = (1.0 - t)**2
        alpha2_s = (1.0 - s)**2
        
        sigma2_t = t**2
        sigma2_s = s**2

        # 3. Compute SNR as the ratio of signal-to-noise ratios
        # Adding epsilon to avoid division by zero at t=0
        eps = 1e-8
        snr_t = alpha2_t / (sigma2_t + eps)
        snr_s = alpha2_s / (sigma2_s + eps)

        # This represents the "step-wise" change in SNR
        SNR_s_t = snr_s / (snr_t + eps)

        return SNR_s_t
    

    @torch.no_grad()
    def sample_structure(self, num_samples, molecule, protein_pocket, sampling_without_noise, data_dir, run_id, save_trajectory=False, fold=None):
        
        device = molecule['x'].device
        num_graphs = molecule['size'].size(0)
        
        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None

        z_x_mol = torch.randn(size=(len(molecule['x']), self.x_dim), device=device) #* self.noise_scaling
        
        mol_norm_x = molecule['x'] / self.norm_values[0]
        protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]
        xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=1)

        if self.com_handling == 'both':
            z_x_mol = z_x_mol - scatter_mean(z_x_mol, molecule['idx'], dim=0)[molecule['idx']]
            xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
        elif self.com_handling == 'no_COM':
                dumy_variable = 0
        else:
            # data is translated to 0, COM noise added and again translated to 0
            mean = scatter_mean(z_x_mol, molecule['idx'], dim=0)
            z_x_mol = z_x_mol - mean[molecule['idx']]
            xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

        if self.features_fixed:
            z_h_mol = (molecule['h'] / self.norm_values[1]).clone().detach()
        else:
            raise NotImplementedError

        current_xh_mol = torch.cat((z_x_mol, z_h_mol), dim=1)
        
        steps = self.T // self.sampling_stepsize
        dt = 1.0 / steps

        solver = "euler"

        if solver == "loop":

            for i in reversed(range(1, steps + 1)):
                t_val = i / steps
                t_array = torch.full((num_graphs, 1), fill_value=t_val, device=device)

                # Predict velocity v_hat
                v_hat_mol, _, c_s, _ = self.neural_net(
                    current_xh_mol, xh_pro, t_array, 
                    molecule['idx'], protein_pocket['idx'], molecule_pos
                )

                # Euler Step: x_{t+dt} = x_t + v(x_t, t) * dt
                if self.features_fixed:
                    current_xh_mol[:, :self.x_dim] += v_hat_mol[:, :self.x_dim] * dt
                else:
                    current_xh_mol += v_hat_mol * dt

                if self.com_handling == 'both':
                    dumy_variable = 0
                elif self.com_handling == 'no_COM':
                    dumy_variable = 0
                else:
                    # project both pocket and peptide to 0 COM again (only mol mean changes)
                    mean = scatter_mean(current_xh_mol[:,:self.x_dim], molecule['idx'], dim=0)
                    current_xh_mol[:,:self.x_dim] = current_xh_mol[:,:self.x_dim] - mean[molecule['idx']]
                    xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

                if self.com_handling == 'both':
                    # old centering approach
                    current_xh_mol[:,:self.x_dim] = current_xh_mol[:,:self.x_dim] - scatter_mean(current_xh_mol[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]
                    xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
                else:
                    dumy_variable = 0

        elif solver == "euler" or "rk4":

            ode_func = ODEWrapper(
                self, xh_pro, molecule, molecule_pos, protein_pocket
            )
            
            t_span = torch.tensor([1.0, 0.0], device=device)
            
            trajectory = odeint(
                ode_func, 
                current_xh_mol, 
                t_span, 
                method=solver, 
                options={'step_size': 1.0 / self.T} 
            )

            current_xh_mol = trajectory[-1]
            c_s = ode_func.last_c_s

        x_mol_final = current_xh_mol[:,:self.x_dim] * self.norm_values[0]
        h_mol_final = current_xh_mol[:,self.x_dim:] * self.norm_values[0]
        x_pro_final = xh_pro[:,:self.x_dim] * self.norm_values[0]
        h_pro_final = xh_pro[:,self.x_dim:] * self.norm_values[0]

        if not self.features_fixed:
            h_mol_final = F.one_hot(torch.argmax(current_xh_mol[:, self.x_dim:], dim=1), self.num_atoms)
        else:
            h_mol_final = molecule['h'] 

        xh_mol_final = torch.cat([x_mol_final, h_mol_final], dim=1)
        xh_pro_final = torch.cat([x_pro_final, h_pro_final], dim=1)

        self.safe_pdbs(xh_mol_final, molecule, run_id, data_dir, time_step='F', fold=fold)

        ba = 0

        return (xh_mol_final, xh_pro_final, c_s, ba)
    
    def safe_pdbs(self, pos, molecule, run_id, data_dir, time_step, fold=None):

        for i in range(len(molecule['size'])):
            # (1) extract the peptide position
            pos = pos[:,:3]
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

                create_new_pdb_hdf5_100k(peptide_pos, peptide_idx, graph_name, run_id, data_dir, time_step=time_step, sample_id=i)

            else:

                create_new_pdb_hdf5(peptide_pos, peptide_idx, graph_name, run_id, data_dir, time_step=time_step, sample_id=i, fold=fold)

class ODEWrapper(nn.Module):
    def __init__(self, model, xh_pro, molecule, molecule_pos, protein_pocket):
        super().__init__()
        self.model = model
        self.xh_pro = xh_pro
        self.molecule_idx = molecule['idx']
        self.protein_idx = protein_pocket['idx']
        self.molecule_pos = molecule_pos
        self.num_graphs = molecule['size'].size(0)
        self.last_c_s = None

    def forward(self, t, xh_mol):
        t_vec = torch.full((self.num_graphs, 1), fill_value=t.item(), device=xh_mol.device)
        
        v_hat_mol, _, c_s, _ = self.model.neural_net(
            xh_mol, self.xh_pro, t_vec, 
            self.molecule_idx, self.protein_idx, self.molecule_pos
        )
        self.last_c_s = c_s
        
        v_xh_final = torch.zeros_like(xh_mol)
        
        v_x = -v_hat_mol[:, :3] 

        v_x = v_x - scatter_mean(v_x, self.molecule_idx, dim=0)[self.molecule_idx]
        
        v_xh_final[:, :3] = v_x
        
        return v_xh_final

