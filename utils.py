from pathlib import Path

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

import re
import glob
import os
import pickle
import h5py
import numpy as np
from io import StringIO
from typing import Sequence, Optional
import openfold.np.protein as protein
from openfold.np import residue_constants


def create_new_pdb_hdf5(
        peptide, peptide_idx, graph_name, run_id, data_dir, time_step, sample_id, atom_level=False, fold=None
):
    if data_dir == "/scratch-shared/roos/preprocessed/":
        file_idx = int(fold) - 1 if fold is not None else 1
        hdf5_file = h5py.File(f'{data_dir}BA_cluster{file_idx}.hdf5', 'r')
    elif fold is not None:
        hdf5_file = h5py.File(os.path.join(data_dir, 'folds', f'fold_{fold}', 'test.hdf5'), 'r')
    else:
        hdf5_file = h5py.File(f'{data_dir}/test.hdf5', 'r')
        
    pdb_names = hdf5_file['pdb_names'][:]
    pdb_strings = hdf5_file['pdb_strings'][:]
    pdb_string = pdb_strings[pdb_names.tolist().index(graph_name)].decode('utf-8')

    # Create a temporary file or use StringIO to make the string readable by parser
    pdb_fh = StringIO(pdb_string)
    
    pdb_output_path = f'./results/structures/{run_id}/{graph_name}_{time_step}_{sample_id}.pdb'

    directory = os.path.dirname(pdb_output_path)
    if not os.path.exists(directory):
         os.makedirs(directory)

    write_updated_peptide_coords_pdb(peptide, peptide_idx, pdb_fh, pdb_output_path, atom_level=atom_level)

def create_new_pdb_hdf5_100k(
    peptide: np.ndarray,
    peptide_idx: Sequence[int],
    graph_name: str,
    run_id: str,
    data_dir: str,
    time_step: int,
    sample_id: int,
    atom_level: bool = False
):
    """
    Saves a new PDB for non-BA entries only.
    Loads the original PDB string via group[()] decoding,
    then overwrites the P-chain CA coords with `peptide`.
    """
    # 1) Only handle non-BA
    if graph_name.startswith("BA"):
        return

    # 2) Read the PDB string from the group
    hdf5_path = Path(data_dir) / '100k_test.hdf5'
    with h5py.File(hdf5_path, 'r') as f5:
        if graph_name not in f5:
            raise KeyError(f"{graph_name} not found in {hdf5_path}")
        group = f5[graph_name]
        pdb_string = group[()].decode('utf-8')

    # 3) Prepare in-memory file for parser
    pdb_fh = StringIO(pdb_string)

    # 4) Build output path
    out_dir = Path('results') / 'structures' / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_output_path = out_dir / f"{graph_name}_{time_step}_{sample_id}.pdb"

    # 5) Write updated PDB: replaces only P-chain CA coords
    write_updated_peptide_coords_pdb(
        peptide=peptide,
        peptide_idx=peptide_idx,
        pdb_reference_path_or_stream=pdb_fh,
        pdb_output_path=str(pdb_output_path),
        atom_level=atom_level
    )
    
def write_updated_peptide_coords_pdb(
    peptide, peptide_idx, pdb_reference_path_or_stream, pdb_output_path, atom_level=False
):
    """
    Function from https://github.com/steusink/DiffSBDD.git

    Takes an existing pdb file with peptide and mhc and creates a new one
    with the same mhc pocket and the peptide with updated atom/residue
    coordinates given by the model.
    :param peptide: peptide with updated coordinates
    :param decoder: decoder, from index to atom/residue
    :param pdb_reference_path: path to the reference pdb file
    :param pdb_output_path: path to the output pdb file
    :param atom_level: whether to use atoms or residues

    :return: None
    """
    # Read the reference pdb file
    parser = PDBParser(QUIET=True)
    pdb_models = parser.get_structure("", pdb_reference_path_or_stream)

    # Get the peptide chain
    peptide_chain = pdb_models[0]["P"]

    if not atom_level:
        peptide_chain_new = Chain("P")

    # Get the peptide atoms/residues
    if atom_level:
        peptide_elements = list(peptide_chain.get_atoms())
    else:
        peptide_elements = list(peptide_chain.get_residues())

    # Update the peptide coordinates
    for i, element in enumerate(peptide_elements):
        if i >= len(peptide): break
        if atom_level:
            element.set_coord(peptide[i])
        else:
            try:
                ca_atom = element["CA"] # might need to switch this to "CB"
                ca_atom.set_coord(peptide[i])
                id = element.get_id()
                id = (' ', int(peptide_idx[i]), ' ')
                new_residue = Residue(id, element.get_resname(), "")
                new_residue.add(ca_atom)
                peptide_chain_new.add(new_residue)
            except (KeyError, IndexError):
                continue

    # Write the new pdb file
    if not atom_level:
        pdb_models[0].detach_child("P")
        pdb_models[0].add(peptide_chain_new)

    io = PDBIO()
    io.set_structure(pdb_models)
    io.save(str(pdb_output_path))


def create_new_pdb_hdf5_swift(
    peptide: np.ndarray,
    peptide_idx: Sequence[int],
    graph_name: str,
    run_id: str,
    data_dir: str,
    time_step: int,
    sample_id: int,
    split: str = "test",
    fold: str = "1",
    atom_level=False,
    save_peptide_only=False
):
    """
    Saves a new PDB for SwiftMHC entries.
    Loads protein and peptide metadata from HDF5 and uses prediction for peptide coords.
    """
    # 1) Determine HDF5 path (following dataset_8k_xray.py logic)
    # Search for the file containing the graph_name

    # Select the source file directly instead of searching every HDF5 file.
    file_idx = int(fold) - 1

    if graph_name.startswith("BA-"):
        source_file = Path(data_dir) / f"BA_cluster{file_idx}.hdf5"
    else:
        source_file = Path(data_dir) / f"xray_cluster{file_idx}.hdf5"

    if not source_file.exists():
        raise FileNotFoundError(
            f"Expected source file does not exist: {source_file}"
        )

    with h5py.File(source_file, "r") as f5:
        if graph_name not in f5:
            raise KeyError(
                f"{graph_name} not found in expected source file {source_file}"
            )

        group = f5[graph_name]

        protein_data = {
            "aatype": group["protein"]["aatype"][:],
            "atom_positions": group["protein"]["all_atom_positions"][:],
            "atom_mask": group["protein"]["all_atom_mask"][:],
        }

        peptide_data = {
            "aatype": group["peptide"]["aatype"][:],
        }

    # file_idx = int(fold) - 1
    # possible_files = [
    #     Path(data_dir) / f"BA_cluster{file_idx}.hdf5",
    #     Path(data_dir) / f"train_fold{file_idx}.hdf5",
    #     Path(data_dir) / f"valid_fold{file_idx}.hdf5",
    # ]
    # # Add xray clusters if they exist
    # for i in range(10):
    #     possible_files.append(Path(data_dir) / f"xray_cluster{i}.hdf5")
    
    # hdf5_path = None
    # group = None
    # f5_handle = None

    # for p in possible_files:
    #     if p.exists():
    #         try:
    #             f5 = h5py.File(p, 'r')
    #             if graph_name in f5:
    #                 hdf5_path = p
    #                 group = f5[graph_name]
    #                 # We found it, but we need to keep the handle or copy data
    #                 protein_data = {
    #                     'aatype': group['protein']['aatype'][:],
    #                     'atom_positions': group['protein']['all_atom_positions'][:],
    #                     'atom_mask': group['protein']['all_atom_mask'][:]
    #                 }
    #                 peptide_data = {
    #                     'aatype': group['peptide']['aatype'][:]
    #                 }
    #                 f5.close()
    #                 break
    #             else:
    #                 # Try partial match fallback
    #                 found_key = None
    #                 for k in f5.keys():
    #                     if graph_name in k or k in graph_name:
    #                         found_key = k
    #                         break
    #                 if found_key:
    #                     print(f"Warning: {graph_name} not found directly in {p}. Using {found_key} instead.")
    #                     hdf5_path = p
    #                     group = f5[found_key]
    #                     protein_data = {
    #                         'aatype': group['protein']['aatype'][:],
    #                         'atom_positions': group['protein']['all_atom_positions'][:],
    #                         'atom_mask': group['protein']['all_atom_mask'][:]
    #                     }
    #                     peptide_data = {
    #                         'aatype': group['peptide']['aatype'][:]
    #                     }
    #                     f5.close()
    #                     break
    #                 f5.close()
    #         except Exception as e:
    #             print(f"Error checking {p}: {e}")
    #             continue

    # if hdf5_path is None:
    #     raise KeyError(f"{graph_name} not found in any expected HDF5 file in {data_dir}.")

    # 2) Build output path
    out_dir = Path('results') / 'structures' / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_output_path = out_dir / f"{graph_name}_{time_step}_{sample_id}.pdb"

    # 3) Write updated PDB
    write_updated_peptide_coords_pdb_swiftmhc(
        peptide_coords=peptide,
        peptide_data=peptide_data,
        protein_data=protein_data,
        pdb_output_path=str(pdb_output_path),
        atom_level=atom_level,
        save_peptide_only=save_peptide_only
    )


def write_updated_peptide_coords_pdb_swiftmhc(
    peptide_coords: np.ndarray,
    peptide_data: dict,
    protein_data: dict,
    pdb_output_path: str,
    atom_level=False,
    save_peptide_only=False
):
    """
    Writes a PDB file for SwiftMHC complex using OpenFold's protein tools.
    """
    # 1) Process Peptide (convert atom14 predicted to atom37)
    peptide_aatype = peptide_data['aatype']
    n_peptide_res = len(peptide_aatype)
    
    # Create atom14 to atom37 mapping
    restype_atom14_to_atom37 = np.zeros((21, 14), dtype=np.int32)
    for rt_idx, restype_char in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_char]
        atom14_names = residue_constants.restype_name_to_atom14_names[restype_name]
        for a14_idx, atom_name in enumerate(atom14_names):
            if atom_name:
                a37_idx = residue_constants.atom_order[atom_name]
                restype_atom14_to_atom37[rt_idx, a14_idx] = a37_idx

    # Expand peptide_coords [N, 14, 3] to [N, 37, 3]
    peptide_atom37_positions = np.zeros((n_peptide_res, 37, 3))
    peptide_atom37_mask = np.zeros((n_peptide_res, 37))
    
    # Ensure peptide_coords is a numpy array on CPU
    if hasattr(peptide_coords, 'detach'):
        peptide_coords = peptide_coords.detach().cpu().numpy()
    
    # If flattened [N*14, 3], reshape to [N, 14, 3]
    if len(peptide_coords.shape) == 2 and peptide_coords.shape[0] == n_peptide_res * 14:
        peptide_coords = peptide_coords.reshape(n_peptide_res, 14, 3)
    
    # Check if we have all-atom [N, 14, 3] or just residue-level [N, 3]
    is_all_atom = (len(peptide_coords.shape) == 3 and peptide_coords.shape[1] == 14)
    
    for i in range(n_peptide_res):
        rt = peptide_aatype[i]
        if is_all_atom:
            for a14_idx in range(14):
                # Check if this atom14 slot holds a real atom for this residue type
                if residue_constants.restype_atom14_mask[rt, a14_idx]:
                    a37_idx = restype_atom14_to_atom37[rt, a14_idx]
                    peptide_atom37_positions[i, a37_idx] = peptide_coords[i, a14_idx]
                    peptide_atom37_mask[i, a37_idx] = 1.0
        else:
            # Residue level: only CA is available (usually index 1 in atom37)
            ca_idx = residue_constants.atom_order['CA']
            peptide_atom37_positions[i, ca_idx] = peptide_coords[i]
            peptide_atom37_mask[i, ca_idx] = 1.0

    # 2) Combine Protein and Peptide
    if save_peptide_only:
        final_aatype = peptide_aatype
        final_positions = peptide_atom37_positions
        final_mask = peptide_atom37_mask
        final_residue_index = np.arange(1, len(peptide_aatype) + 1)
        final_chain_index = np.zeros(len(peptide_aatype))
    else:
        # Protein is chain A (index 0), Peptide is chain B (index 1)
        final_aatype = np.concatenate([protein_data['aatype'], peptide_aatype])
        final_positions = np.concatenate([protein_data['atom_positions'], peptide_atom37_positions])
        final_mask = np.concatenate([protein_data['atom_mask'], peptide_atom37_mask])
        
        n_prot = len(protein_data['aatype'])
        n_pep = len(peptide_aatype)
        final_residue_index = np.concatenate([np.arange(1, n_prot + 1), np.arange(1, n_pep + 1)])
        final_chain_index = np.concatenate([np.zeros(n_prot), np.ones(n_pep)])
    
    # 3) Create OpenFold Protein object
    prot_obj = protein.Protein(
        atom_positions=final_positions,
        aatype=final_aatype,
        atom_mask=final_mask,
        residue_index=final_residue_index,
        b_factors=np.zeros_like(final_mask),
        chain_index=final_chain_index
    )
    
    # 4) Convert to PDB and Save
    pdb_str = protein.to_pdb(prot_obj)
    with open(pdb_output_path, 'w') as f:
        f.write(pdb_str)