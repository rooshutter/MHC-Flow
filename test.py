import argparse
from argparse import Namespace
from pathlib import Path
import yaml
import os
import time
import pickle
import gzip

import os
import sys

import torch
import pytorch_lightning as pl
from torch_scatter import scatter_add

import wandb

def count_clashes(peptide_pos, protein_pos, peptide_mask, protein_mask):
    radii = torch.tensor([1.55, 1.7, 1.7, 1.52, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7], dtype=torch.float32, device=peptide_pos.device)
    pep_flat = peptide_pos.reshape(-1, 3)
    pro_flat = protein_pos.reshape(-1, 3)
    pep_ex_flat = peptide_mask.reshape(-1)
    pro_ex_flat = protein_mask.reshape(-1)
    
    pep_r = radii.unsqueeze(0).expand(peptide_pos.shape[0], -1).reshape(-1)
    pro_r = radii.unsqueeze(0).expand(protein_pos.shape[0], -1).reshape(-1)
    
    pep_flat = pep_flat[pep_ex_flat]
    pep_r = pep_r[pep_ex_flat]
    pro_flat = pro_flat[pro_ex_flat]
    pro_r = pro_r[pro_ex_flat]
    
    if pep_flat.shape[0] == 0 or pro_flat.shape[0] == 0:
        return 0
        
    dists = torch.cdist(pep_flat, pro_flat)
    radii_sum = pep_r.unsqueeze(1) + pro_r.unsqueeze(0)
    return (dists < radii_sum).sum().item()

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set working directory and import modules
    desired_directory = '/home/rhutter/MHC-Diff/'
    os.chdir(desired_directory)
    sys.path.insert(0, desired_directory)
    from model.lightning_module import Structure_Prediction_Model
    from dataset_8k_xray import PDB_Dataset, PDB_Dataset_swift

    # read in config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    args_dict = args.__dict__
    for key, value in config.items():
        if isinstance(value, dict):
            args_dict[key] = Namespace(**value)
        else:
            args_dict[key] = value

    if args.wandb_log:
        logger = pl.loggers.WandbLogger(
            save_dir=args.logdir,
            project=args.project,
            name=args.run_name,
            entity=args.entity,
            config=args_dict  
        )

    num_samples = args.num_samples
    sample_batch_size = args.sample_batch_size
    sample_savepath = args.sample_savepath

    lightning_model = Structure_Prediction_Model.load_from_checkpoint(
                    args.checkpoint,
                    dataset=args.dataset,
                    data_dir=args.data_dir,
                    dataset_params=args.dataset_params,
                    task_params=args.task_params,
                    generative_model=args.generative_model,
                    generative_model_params=args.generative_model_params,
                    architecture=args.architecture,
                    network_params=args.network_params,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    num_workers=args.num_workers,
                    device=args.device,
                    all_atom=args.all_atom,
                    variational=args.variational,
                    solver=args.solver,
                    ba=args.ba,
                    use_quat=getattr(args, 'use_quat', False),
    )   

    lightning_model = lightning_model.to(device)
    lightning_model.setup('test')
    test_dataset = lightning_model.test_dataset

    ## calculate the test_dataset variance
    var = []
    for data in test_dataset:
        pos = data['peptide_positions'].to(torch.float32)
        var += [torch.sum((pos - torch.mean(pos, dim=0))**2, dim=0) / len(pos)]
    dataset_variance = sum(var) / len(var)
    # print(f"{dataset_variance=}")

    results = []
    saved_samples = {}
    saved_samples['graph_name'] = [data['graph_name'] for data in test_dataset]
    saved_samples['x_target'] = {}
    saved_samples['x_predicted'] = {}
    saved_samples['h'] = {}
    saved_samples['rmse'] = []
    saved_samples['rmse_ca'] = []
    saved_samples['rmse_mean'] = []
    saved_samples['rmse_best'] = []
    saved_samples['rmse_ca_mean'] = []
    saved_samples['rmse_ca_best'] = []
    saved_samples['clashes_mean'] = []
    saved_samples['clashes_best'] = []
    if args.ba:
        saved_samples['ba_per_peptide'] = []   # list of [num_samples] tensors per peptide
        saved_samples['ba_true_per_peptide'] = []  # list of scalar tensors per peptide
        saved_samples['ba_true_struct_per_peptide'] = []  # list of scalar tensors per peptide
        saved_samples['ba_mask_per_peptide'] = []  # list of boolean tensors per peptide


    total_sampling_time = 0.0
    if args.ba:
        total_ba_time = 0.0
    total_structures_sampled = 0
    total_pmhc_cases = 0

    start_time_total = time.time()

    for i in range(0, len(test_dataset), sample_batch_size):

        current_batch_size = min(sample_batch_size, len(test_dataset) - i)

        start_time = time.time()

        # prepare peptide-MHC, grouped by structure so samples are contiguous
        mol_pro_list = [test_dataset[i+j] for j in range(current_batch_size) for _ in range(num_samples)]

        if args.all_atom:
            mol_pro_samples = PDB_Dataset_swift.collate_fn(mol_pro_list)
        else:
            mol_pro_samples = PDB_Dataset.collate_fn(mol_pro_list)

        # sample new peptide-MHC structures using trained model
        mol_pro_batch = lightning_model.get_molecule_and_protein(mol_pro_samples)
        molecule, protein_pocket = mol_pro_batch

        t_before_sampling = time.time()
        print(f'Prior to sampling time {t_before_sampling - start_time}')

        save_trajectory = False
        xh_mol_final, xh_pro_final, _, ba = lightning_model.model.sample_structure(num_samples, molecule, protein_pocket, args.sampling_without_noise, args.data_dir, args.run_name, save_trajectory=save_trajectory, fold=args.dataset_params.fold)
        if args.all_atom:
            x_mol_final = xh_mol_final[:,:3].reshape(-1, molecule['x'].shape[1], 3)
        else:
            x_mol_final = xh_mol_final[:,:3]

        t_after_sampling = time.time()
        sampling_time = t_after_sampling - t_before_sampling
        if args.ba:
            ba_batch_time = ba.get('ba_time', 0.0)
        else:
            ba_batch_time = 0.0
        structure_batch_time = sampling_time - ba_batch_time
        n_structures = num_samples * current_batch_size
        total_sampling_time += sampling_time
        if args.ba:
            total_ba_time += ba_batch_time
        total_structures_sampled += n_structures
        total_pmhc_cases += current_batch_size
        print(f'After sampling time {t_after_sampling - start_time}')
        print(f'Sampling time: {sampling_time:.3f}s for {n_structures} structures ({sampling_time / n_structures:.4f}s per structure)')
        print(f'  Structure prediction: {structure_batch_time:.3f}s ({structure_batch_time / current_batch_size:.4f}s per pMHC case)')
        if args.ba:
            print(f'  BA prediction:        {ba_batch_time:.3f}s ({ba_batch_time / current_batch_size:.4f}s per pMHC case)')
        
        if args.all_atom:
            molecule['h'] = molecule['h'].reshape(-1, molecule['h'].shape[-1])
        # Safe resulting structures
        size_tuple = tuple(molecule['size'].tolist())
        true_pos = [torch.split(molecule['x'], size_tuple, dim=0)[i*num_samples] for i in range(current_batch_size)] # [current_batch_size, num_nodes, 3]
        true_h = [torch.split(molecule['h'], size_tuple, dim=0)[i*num_samples] for i in range(current_batch_size)] # [current_batch_size, num_nodes, 3]

        for j in range(current_batch_size):
            key = i+j
            saved_samples['x_target'][key] = true_pos[j]
            saved_samples['x_predicted'][key] = torch.split(x_mol_final, size_tuple, dim=0)[j*num_samples:(j+1)*num_samples]
            saved_samples['h'][key] = true_h[j]

        # Calculate the RMSE error
        if args.all_atom:

            error_mol = scatter_add(torch.sum((molecule['x'] - x_mol_final)**2, dim=(-2, -1)), molecule['idx'], dim=0)
            rmse = torch.sqrt(error_mol / (molecule['size'] * molecule["x"].shape[1]))

            x_ca_pred = x_mol_final[:, 1, :]  
            x_ca_true = molecule['x'][:, 1, :] 
            error_ca = scatter_add(torch.sum((x_ca_true - x_ca_pred)**2, dim=-1), molecule['idx'], dim=0)
            rmse_ca = torch.sqrt(error_ca / molecule['size'])

            if "2GTW" in molecule['graph_name']:
                idx = molecule['graph_name'].index("2GTW")
                x_ca_pred = x_ca_pred.view(100, 9, 3)
                x_ca_true = x_ca_true.view(100, 9, 3)
                x_ca_pred_2gtw = x_ca_pred[idx:(idx+10)]
                x_ca_true_2gtw = x_ca_true[idx:(idx+10)]
                rmse_ca_2gtw = rmse_ca[idx:(idx+10)]

        else:
            error_mol = scatter_add(torch.sum((molecule['x'] - x_mol_final)**2, dim=-1), molecule['idx'], dim=0)
            rmse = torch.sqrt(error_mol / (molecule['size']))
            rmse_ca = rmse

        rmse_sample_mean = [rmse[j*num_samples:(j+1)*num_samples].mean(0) for j in range(current_batch_size)]
        rmse_sample_best = [rmse[j*num_samples:(j+1)*num_samples].min(0)[0] for j in range(current_batch_size)]
        rmse_ca_sample_mean = [rmse_ca[j*num_samples:(j+1)*num_samples].mean(0) for j in range(current_batch_size)]
        rmse_ca_sample_best = [rmse_ca[j*num_samples:(j+1)*num_samples].min(0)[0] for j in range(current_batch_size)]

        clashes = []
        size_tuple_pro = tuple(protein_pocket['size'].tolist())
        pro_pos_split = torch.split(protein_pocket['x'], size_tuple_pro, dim=0)
        pep_pos_split = torch.split(x_mol_final, size_tuple, dim=0)
        
        for k in range(num_samples * current_batch_size):
            pep_p = pep_pos_split[k]
            pro_p = pro_pos_split[k]
            pep_ex = (pep_p.abs().sum(-1) > 1e-6)
            pro_ex = (pro_p.abs().sum(-1) > 1e-6)
            clashes.append(count_clashes(pep_p, pro_p, pep_ex, pro_ex))
        
        clashes = torch.tensor(clashes, dtype=torch.float32, device=device)
        clashes_sample_mean = [clashes[j*num_samples:(j+1)*num_samples].mean(0) for j in range(current_batch_size)]
        
        clashes_sample_best = []
        for j in range(current_batch_size):
            best_idx = rmse[j*num_samples:(j+1)*num_samples].argmin(0)
            clashes_sample_best.append(clashes[j*num_samples + best_idx])

        if args.ba:
            ba_hat = ba['ba_hat'].squeeze(-1)  # [num_samples * sample_batch_size]
            ba_true = ba['ba_true'].squeeze(-1)  # [num_samples * sample_batch_size]
            ba_true_struct = ba.get('ba_true_struct', torch.zeros_like(ba_hat)).squeeze(-1)
            ba_mask = molecule.get('ba_mask', torch.ones_like(ba_true, dtype=torch.bool))

        end_time = time.time()

        saved_samples['rmse'] += [rmse[j*num_samples:(j+1)*num_samples] for j in range(current_batch_size)]
        saved_samples['rmse_ca'] += [rmse_ca[j*num_samples:(j+1)*num_samples] for j in range(current_batch_size)]
        if args.ba:
            saved_samples['ba_per_peptide'] += [ba_hat[j*num_samples:(j+1)*num_samples] for j in range(current_batch_size)]
            saved_samples['ba_true_per_peptide'] += [ba_true[j*num_samples] for j in range(current_batch_size)]
            saved_samples['ba_true_struct_per_peptide'] += [ba_true_struct[j*num_samples:(j+1)*num_samples].mean(0) for j in range(current_batch_size)]
            saved_samples['ba_mask_per_peptide'] += [ba_mask[j*num_samples] for j in range(current_batch_size)]

        print(len(saved_samples['rmse']), rmse.shape)
        print(f'RMSE (all-atom) sample mean: {rmse_sample_mean}')
        print(f'RMSE (all-atom) sample best: {rmse_sample_best}')
        print(f'RMSE (CA-only) sample mean: {rmse_ca_sample_mean}')
        print(f'RMSE (CA-only) sample best: {rmse_ca_sample_best}')
        # print(f'Clashes sample mean: {clashes_sample_mean}')
        # print(f'Clashes sample best: {clashes_sample_best}')
        if args.ba:
            print(f'BA hat: {ba_hat[:10].tolist()}')
            print(f'BA true: {ba_true[:10].tolist()}')

        saved_samples['rmse_mean'] += [rmse_sample_mean[j] for j in range(current_batch_size)]
        saved_samples['rmse_best'] += [rmse_sample_best[j] for j in range(current_batch_size)]
        saved_samples['rmse_ca_mean'] += [rmse_ca_sample_mean[j] for j in range(current_batch_size)]
        saved_samples['rmse_ca_best'] += [rmse_ca_sample_best[j] for j in range(current_batch_size)]
        saved_samples['clashes_mean'] += [clashes_sample_mean[j] for j in range(current_batch_size)]
        saved_samples['clashes_best'] += [clashes_sample_best[j] for j in range(current_batch_size)]

        print(f'Time: {end_time - start_time}')

    end_time_total = time.time()
    time_total = end_time_total - start_time_total

    print(f"{len(saved_samples['rmse_ca'])=}")
    print(f"{len(saved_samples['rmse_ca_mean'])=}")
    print(f"{len(saved_samples['rmse_ca_best'])=}")

    saved_samples['rmse_mean'] = torch.stack(saved_samples['rmse_mean'], dim=0)
    saved_samples['rmse_best'] = torch.stack(saved_samples['rmse_best'], dim=0)
    saved_samples['rmse_ca_mean'] = torch.stack(saved_samples['rmse_ca_mean'], dim=0)
    saved_samples['rmse_ca_best'] = torch.stack(saved_samples['rmse_ca_best'], dim=0)
    saved_samples['clashes_mean'] = torch.stack(saved_samples['clashes_mean'], dim=0)
    saved_samples['clashes_best'] = torch.stack(saved_samples['clashes_best'], dim=0)
    rmse_mean = saved_samples['rmse_mean'].mean(0)
    rmse_best = saved_samples['rmse_best'].mean(0)
    rmse_ca_mean = saved_samples['rmse_ca_mean'].mean(0)
    rmse_ca_best = saved_samples['rmse_ca_best'].mean(0)
    clashes_mean = saved_samples['clashes_mean'].mean(0)
    clashes_best = saved_samples['clashes_best'].mean(0)

    if args.ba:
        ba_per_peptide = torch.stack(saved_samples['ba_per_peptide']).cpu()      # [N, num_samples]
        ba_true_per_peptide = torch.stack(saved_samples['ba_true_per_peptide']).cpu()  # [N]
        ba_true_struct_per_peptide = torch.stack(saved_samples['ba_true_struct_per_peptide']).cpu()  # [N]
        ba_mask_per_peptide = torch.stack(saved_samples['ba_mask_per_peptide']).cpu()  # [N]
    rmse_per_peptide = torch.stack(saved_samples['rmse']).cpu()              # [N, num_samples]

    if args.ba:
        best_idx = rmse_per_peptide.argmin(dim=1)  # [N]
        ba_best_pred = ba_per_peptide[torch.arange(len(ba_per_peptide)), best_idx]  # [N]

    # Mask BA so we only evaluate BA metrics on Pandora (BA) samples, not X-ray samples
    if args.ba:
        ba_per_peptide = ba_per_peptide[ba_mask_per_peptide]
        ba_true_per_peptide = ba_true_per_peptide[ba_mask_per_peptide]
        ba_true_struct_per_peptide = ba_true_struct_per_peptide[ba_mask_per_peptide]
        ba_best_pred = ba_best_pred[ba_mask_per_peptide]

        print(f"Filtered BA metrics to {ba_mask_per_peptide.sum().item()}/{len(ba_mask_per_peptide)} samples with true BA scores.")

    print(f"{saved_samples['rmse_mean']=}")
    print(f"{saved_samples['rmse_best']=}")
    print(f"{saved_samples['rmse_ca_mean']=}")
    print(f"{saved_samples['rmse_ca_best']=}")

    print(f'Mean RMSE across all mean/best sample: mean {round(rmse_mean.item(),3)}, best {round(rmse_best.item(),3)}')
    print(f'This took {time_total} seconds for 1000*10 samples')
    print(f'Mean RMSE (all-atom): mean {round(rmse_mean.item(),3)}, best {round(rmse_best.item(),3)}')
    print(f'Mean RMSE (CA-only):  mean {round(rmse_ca_mean.item(),3)}, best {round(rmse_ca_best.item(),3)}')
    # print(f'Mean Clashes: mean {round(clashes_mean.item(),3)}, best {round(clashes_best.item(),3)}')

    avg_time_per_structure = total_sampling_time / total_structures_sampled if total_structures_sampled > 0 else 0.0
    if args.ba:
        total_structure_time = total_sampling_time - total_ba_time
    else:
        total_structure_time = total_sampling_time
    avg_structure_time_per_pmhc = total_structure_time / total_pmhc_cases if total_pmhc_cases > 0 else 0.0
    if args.ba:
        avg_ba_time_per_pmhc = total_ba_time / total_pmhc_cases if total_pmhc_cases > 0 else 0.0

    print(f'Total pMHC cases evaluated: {total_pmhc_cases}')
    print(f'Samples per pMHC case: {num_samples}')
    print(f'Average total sampling time per structure: {avg_time_per_structure:.4f}s ({total_sampling_time:.1f}s total for {total_structures_sampled} structures)')
    print(f'Structure prediction: {avg_structure_time_per_pmhc:.4f}s per pMHC case ({total_structure_time:.1f}s total)')
    if args.ba:
        print(f'BA prediction:        {avg_ba_time_per_pmhc:.4f}s per pMHC case ({total_ba_time:.1f}s total)')

    print(f'This took {time_total} seconds for {len(test_dataset)}*{num_samples} samples')


    def compute_ba_metrics(y_true, y_pred, threshold=0.4256):
        """Compute ROC AUC, AUPR, Pearson, and RMSE in pure PyTorch."""
        y_true_binary = (y_true >= threshold).long()
        n_pos = y_true_binary.sum().item()
        n_neg = len(y_true_binary) - n_pos

        if n_pos > 0 and n_neg > 0:
            sorted_idx = torch.argsort(y_pred, descending=True)
            sorted_labels = y_true_binary[sorted_idx].float()
            cum_tp = sorted_labels.cumsum(0)
            cum_fp = (1.0 - sorted_labels).cumsum(0)
            tpr = torch.cat([torch.tensor([0.0]), cum_tp / n_pos])
            fpr = torch.cat([torch.tensor([0.0]), cum_fp / n_neg])
            auc = torch.trapezoid(tpr, fpr).item()

            precision = cum_tp / torch.arange(1, len(sorted_labels) + 1, dtype=torch.float32)
            recall = cum_tp / n_pos
            mask = sorted_labels == 1.0
            pr = torch.cat([torch.tensor([1.0]), precision[mask]])
            rc = torch.cat([torch.tensor([0.0]), recall[mask]])
            aupr = torch.trapezoid(pr, rc).item()
        else:
            auc = float('nan')
            aupr = float('nan')

        if len(y_true) > 1:
            cov = ((y_true - y_true.mean()) * (y_pred - y_pred.mean())).mean()
            pearson = (cov / (y_true.std(unbiased=False) * y_pred.std(unbiased=False) + 1e-8)).item()
        else:
            pearson = float('nan')

        rmse = torch.sqrt(torch.mean((y_true - y_pred)**2)).item()
        return auc, aupr, pearson, rmse

    if args.ba:
        auc_per_sample = []
        aupr_per_sample = []
        pearson_per_sample = []
        rmse_ba_per_sample = []
    
        for s in range(num_samples):
            ba_pred_s = ba_per_peptide[:, s]  
            auc_s, aupr_s, pearson_s, rmse_s = compute_ba_metrics(ba_true_per_peptide, ba_pred_s)
            auc_per_sample.append(auc_s)
            aupr_per_sample.append(aupr_s)
            pearson_per_sample.append(pearson_s)
            rmse_ba_per_sample.append(rmse_s)

        import numpy as np
        auc_mean = float(np.nanmean(auc_per_sample))
        aupr_mean = float(np.nanmean(aupr_per_sample))
        pearson_mean = float(np.nanmean(pearson_per_sample))
        rmse_ba_mean = float(np.nanmean(rmse_ba_per_sample))

        auc_best, aupr_best, pearson_best, rmse_ba_best = compute_ba_metrics(ba_true_per_peptide, ba_best_pred)

        auc_true_struct, aupr_true_struct, pearson_true_struct, rmse_ba_true_struct = compute_ba_metrics(ba_true_per_peptide, ba_true_struct_per_peptide)

        print(f"BA (mean of {num_samples} samples):    AUC {round(auc_mean, 3)}, AUPR {round(aupr_mean, 3)}, Pearson {round(pearson_mean, 3)}, RMSE {round(rmse_ba_mean, 3)}")
        print(f"BA (best RMSE sample):  AUC {round(auc_best, 3)}, AUPR {round(aupr_best, 3)}, Pearson {round(pearson_best, 3)}, RMSE {round(rmse_ba_best, 3)}")
        print(f"BA (true structure):  AUC {round(auc_true_struct, 3)}, AUPR {round(aupr_true_struct, 3)}, Pearson {round(pearson_true_struct, 3)}, RMSE {round(rmse_ba_true_struct, 3)}")

    start_time_saving = time.time()

    # Serialize dictionary with pickle
    pickled_data = pickle.dumps(saved_samples)

    # Make file directory if it does not exist 
    directory = os.path.dirname(sample_savepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Compress pickled data
    with gzip.open(f'{sample_savepath}.pkl.gz', 'wb') as f:
        f.write(pickled_data)

    end_time_saving = time.time()
    time_saving = end_time_saving - start_time_saving
    print(f'Time to save data: {time_saving} s')

    time_saving_per_structure = time_saving / total_structures_sampled if total_structures_sampled > 0 else 0.0
    structure_time_per_structure = total_structure_time / total_structures_sampled if total_structures_sampled > 0 else 0.0

    time_1_sample = structure_time_per_structure + time_saving_per_structure
    time_10_samples = time_1_sample * 10

    print(f'Time for 1 sample + saving: {time_1_sample:.4f} s')
    print(f'Time for 10 samples + saving: {time_10_samples:.4f} s')

    if args.ba:
        final_metrics = {
            'rmse_mean': rmse_mean,
            'rmse_best': rmse_best,
            'rmse_ca_mean': rmse_ca_mean,
            'rmse_ca_best': rmse_ca_best,
            'clashes_mean': clashes_mean.item(),
            'clashes_best': clashes_best.item(),
            'ba_auc_mean': auc_mean,
            'ba_auc_best': auc_best,
            'ba_aupr_mean': aupr_mean,
            'ba_aupr_best': aupr_best,
            'ba_pearson_mean': pearson_mean,
            'ba_pearson_best': pearson_best,
            'ba_rmse_mean': rmse_ba_mean,
            'ba_rmse_best': rmse_ba_best,
            'ba_auc_true_struct': auc_true_struct,
            'ba_aupr_true_struct': aupr_true_struct,
            'ba_pearson_true_struct': pearson_true_struct,
            'ba_rmse_true_struct': rmse_ba_true_struct,
            'time_total': time_total,
            'time_per_pmhc': avg_structure_time_per_pmhc,
            'time_per_ba': avg_ba_time_per_pmhc,
            'time_1_sample': time_1_sample,
            'time_10_samples': time_10_samples,
        }
    else:
        final_metrics = {
            'rmse_mean': rmse_mean,
            'rmse_best': rmse_best,
            'rmse_ca_mean': rmse_ca_mean,
            'rmse_ca_best': rmse_ca_best,
            'clashes_mean': clashes_mean.item(),
            'clashes_best': clashes_best.item(),
            'time_total': time_total,
            'time_per_pmhc': avg_structure_time_per_pmhc,
            'time_1_sample': time_1_sample,
            'time_10_samples': time_10_samples,
        }

    if args.wandb_log:
        logger.log_metrics(final_metrics)
        logger.experiment.finish()
