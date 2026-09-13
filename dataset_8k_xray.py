from itertools import chain
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from Bio import PDB  # Biopython's PDB parser
from typing import Dict
from tools.rigid import Rigid
# from domain.amino_acid import amino_acids_by_letter, amino_acids_by_one_hot_index, AMINO_ACID_DIMENSION, canonical_amino_acids

# Define amino acid one-hot encoding for 20 standard residues
AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_LIST)}

from openfold.np.residue_constants import restype_atom14_mask, chi_angles_mask


def one_hot_encode_sequence(sequence):
    """One-hot encode the amino acid sequence based on standard residues."""
    encoding = torch.zeros((len(sequence), len(AA_LIST)), dtype=torch.float32)
    for i, aa in enumerate(sequence):
        if aa in AA_TO_INDEX:
            encoding[i, AA_TO_INDEX[aa]] = 1.0
        else:
            print(f"Warning: Non-standard amino acid '{aa}' found and ignored.")
    return encoding

class PDB_Dataset(Dataset):
    
    def __init__(self, datadir, split='train', fold="1", all_atom=False): 
        """
        Args:
            datadir (str): Path to the directory where HDF5 files are located.
            split (str): Dataset split, one of 'train', 'valid', 'test'.
        """
        # Define the HDF5 file paths for the dataset split
        if fold:
            self.hdf5_path = os.path.join(datadir, f'folds/fold_{fold}', f'{split}.hdf5')
        else:
            self.hdf5_path = os.path.join(datadir, f'{split}.hdf5')

        print(f"Loading dataset from {self.hdf5_path}...")

        # Open the HDF5 file and load the pdb_strings dataset directly
        with h5py.File(self.hdf5_path, 'r') as f5:
            #####################################################################
            # self.entry_names = list(f5.keys())
            # print(f"Entries in the HDF5 file: {self.entry_names}")
            
            # group = f5['BA-55224']

            # for name, dataset in group.items():
            
            #     # Skip if it's a nested group, only print datasets
            #     if not isinstance(dataset, h5py.Dataset):
            #         print(f"Skipping: {name} is a nested group.")
            #         for sub_name, sub_dataset in dataset.items():
            #             print(f"  - {sub_name}: shape {sub_dataset.shape}, dtype {sub_dataset.dtype}")
            #             print(sub_dataset)
            #         continue

            #     print(f"\n[DATASET: {name}]")
            #     print(f"  Shape: {dataset.shape}")
            #     print(f"  Data Type: {dataset.dtype}")
            #     print(dataset)

            ####################################################################

            self.pdb_strings = f5['pdb_strings'][:]  # Load the pdb_strings array directly
            self.pdb_names = f5['pdb_names'][:]  # Load the pdb_names array
            print(f"Loaded {len(self.pdb_strings)} pdb strings and names from {split} split.")
            self.all_atom = all_atom  

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns a data entry from the HDF5 file for a given index.
        """
        # print(f"Loading entry at index: {index}")
        return self.get_entry(index)

    def __len__(self) -> int:
        """
        Returns the total number of entries in the dataset.
        """
        return len(self.pdb_strings)

    def get_entry(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves and processes a single entry from the HDF5 file.
        
        Args:
            entry_name (str): The name of the entry in the HDF5 file.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the processed data.
        """
        data = {}
        with h5py.File(self.hdf5_path, 'r') as f5:
            
            # Access the pdb_string directly using the index
            pdb_string = self.pdb_strings[index].decode('utf-8')
            # print(f"Loaded PDB string for entry {index}: {pdb_string[:100]}...")  # Print the first 100 characters to verify
            # pdb_string = entry[0].decode('utf-8')  # Access by index to ensure it's correct

            # Parse the PDB data using Biopython
            structure = self.parse_pdb_structure(pdb_string)
            # print(f"Parsed structure for entry {index}: {structure}")  # Verify parsing was successful

            # Extract peptide (P chain) and protein (M chain) data
            peptide_chain = structure[0]['P']
            protein_chain = structure[0]['M']
            # print(f"{peptide_chain=}, {protein_chain=}") 

            # Extract C-alpha atom coordinates and sequence for peptide and protein
            peptide_coords, peptide_seq = self.extract_ca_coords_and_sequence(peptide_chain)
            # print(f"Peptide sequence: {peptide_seq} (length: {len(peptide_seq)})")
            # print(f"Peptide coordinates: {peptide_coords}") 
            
            # Truncate the M chain (protein chain) to the first 178 residues
            protein_coords, protein_seq = self.extract_ca_coords_and_sequence(protein_chain, max_residues=178)
            # print(f"Protein sequence: {protein_seq} (length: {len(protein_seq)})")

            # One-hot encode sequences
            peptide_onehot = one_hot_encode_sequence(peptide_seq)
            protein_onehot = one_hot_encode_sequence(protein_seq)

            # Generate masks (assuming all residues are valid for now)
            peptide_len = peptide_coords.shape[0]
            protein_len = protein_coords.shape[0]
            peptide_mask = torch.ones(peptide_len, dtype=torch.bool)
            protein_mask = torch.ones(protein_len, dtype=torch.bool)

            # Prepare the output dictionary
            data['graph_name'] = self.pdb_names[index]
            data['peptide_idx'] = peptide_mask
            data['peptide_positions'] = peptide_coords  # 3D C-alpha coordinates for peptide
            data['peptide_features'] = peptide_onehot  # One-hot encoded peptide sequence
            data['num_peptide_residues'] = peptide_len
            data['protein_pocket_idx'] = protein_mask
            data['protein_pocket_positions'] = protein_coords  # 3D C-alpha coordinates for protein
            data['protein_pocket_features'] = protein_onehot  # One-hot encoded protein sequence
            data['num_protein_pocket_residues'] = protein_len
            data['pos_in_seq'] = torch.arange(peptide_len) + 1  # Position in the sequence



        return data

    def parse_pdb_structure(self, pdb_string: str):
        """Parses the PDB string using Biopython and returns a structure."""
        parser = PDB.PDBParser(QUIET=True)
        from io import StringIO
        pdb_io = StringIO(pdb_string)
        structure = parser.get_structure("structure", pdb_io)
        return structure

    def extract_ca_coords_and_sequence(self, chain, max_residues=None):
        """
        Extract C-alpha coordinates and amino acid sequence from a PDB chain.

        Args:
            chain (Bio.PDB.Chain): The chain object from which to extract data.
            max_residues (int, optional): Maximum number of residues to include (for truncation).

        Returns:
            coords (torch.Tensor): C-alpha coordinates (Nx3 tensor).
            sequence (str): Corresponding amino acid sequence.
        """
        if max_residues is None and self.all_atom:
            ca_coords = np.zeros((9, 14, 3))  
        elif not self.all_atom:
            ca_coords = []
        else:
            ca_coords = np.zeros((max_residues, 14, 3))  
        sequence = []

        # (9, 14, 3)
        # print(f"{len(chain)=}")
        for i, residue in enumerate(chain):
            if max_residues is not None and i >= max_residues:
                break  # Truncate if the number of residues exceeds the limit

            if 'CA' in residue:
                if not self.all_atom:
                    ca_coords.append(residue['CA'].coord)
                    sequence.append(PDB.Polypeptide.three_to_one(residue.resname))
                else:
                    # residue_atoms = [atom.coord for atom in residue]
                    # ca_coords.extend(residue_atoms)
                    # print(f'{PDB.Polypeptide.three_to_one(residue.resname)=}')
                    for j, atom in enumerate(residue):
                        if j >= 14:
                            break  # Limit to 14 atoms per residue  
                        # print(f"Processing residue {residue.resname} (index {i}), atom {atom.name} (index {j})")
                        ca_coords[i][j] = atom.coord
                    # residue_seq = [PDB.Polypeptide.three_to_one(residue.resname) for atom in residue]
                    # sequence.extend(residue_seq)
                    sequence.append(PDB.Polypeptide.three_to_one(residue.resname)) 
            else:
                print(f"Warning: Missing CA atom for residue {residue.resname} in chain {chain.id}")

        # print(f"Extracted sequence for chain {chain.id}: {''.join(sequence)}")
        # Convert to torch tensors
        coords_tensor = torch.tensor(np.array(ca_coords), dtype=torch.float32)  # Shape: (N, 3) 
        sequence_str = ''.join(sequence)

        
        return coords_tensor, sequence_str


    @staticmethod
    def collate_fn(batch):
        """
        Collation function to combine batch data into a single batch.
        
        Args:
            batch (list of Dict): A list of individual data entries.

        Returns:
            Dict: A dictionary containing batched data.
        """
        data_batch = {}

        for key in batch[0].keys():

            if key == 'graph_name':
                data_batch[key] = [x[key] for x in batch]
            elif key == 'num_peptide_residues' or key == 'num_protein_pocket_residues':
                data_batch[key] = torch.tensor([x[key] for x in batch])
            elif 'idx' in key:
                # Ensure that indices in the batch start at zero (needed for torch_scatter)
                data_batch[key] = torch.cat([i * torch.ones(len(x[key]), dtype=torch.long) for i, x in enumerate(batch)], dim=0)
            else:
                data_batch[key] = torch.cat([x[key] for x in batch], dim=0)

        return data_batch
    

class PDB_Dataset_combine(Dataset):

    def __init__(self, datadir_swift, datadir_david, split='train', fold="1", path=None): # Roos: fold hardcoded to 1 for now
        """
        Args:
            datadir (str): Path to the directory where HDF5 files are located.
            split (str): Dataset split, one of 'train', 'valid', 'test'.
        """

        self.hdf5_path_swift = {'train': os.path.join(datadir_swift, f'train_fold{fold}.hdf5')}
        self.hdf5_path_swift['valid'] = os.path.join(datadir_swift, f'valid_fold{fold}.hdf5')
        self.hdf5_path_swift['BA'] = os.path.join(datadir_swift, f'BA_cluster{fold}.hdf5')
        for i in range(10):
            self.hdf5_path_swift[f'xray_cluster{i}'] = os.path.join(datadir_swift, f'xray_cluster{i}.hdf5')

        self.data_swift = {}
        for key, path in self.hdf5_path_swift.items():
            with h5py.File(path, 'r') as f5:
                for name in f5.keys():
                    # self.data_swift[name] = f5[name]
                    self.data_swift[name] = path

        print(f"Total entries from SwiftMHC: {len(self.data_swift)}") 

        self.hdf5_path = os.path.join(datadir_david, f'{split}.hdf5')
        
        print(f"Loading dataset from {self.hdf5_path}...")

        # Open the HDF5 file and load the pdb_strings dataset directly
        with h5py.File(self.hdf5_path, 'r') as f5:
            self.pdb_strings = f5['pdb_strings'][:]  # Load the pdb_strings array directly
            self.pdb_names = f5['pdb_names'][:]  # Load the pdb_names array
            print(f"Loaded {len(self.pdb_strings)} pdb strings and names from {split} split.")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns a data entry from the HDF5 file for a given index.
        """
        # print(f"Loading entry at index: {index}")
        return self.get_entry(index)

    def __len__(self) -> int:
        """
        Returns the total number of entries in the dataset.
        """
        return len(self.pdb_strings)

    def get_entry(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves and processes a single entry from the HDF5 file.
        
        Args:
            entry_name (str): The name of the entry in the HDF5 file.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the processed data.
        """

        data = {}
        name = self.pdb_names[index].decode('utf-8')
        path = self.data_swift[name]

        with h5py.File(path, 'r') as f5:

            pdb_string = f5[name]
            
            peptide_coords = pdb_string['peptide']['atom14_gt_positions']
            peptide_coords = torch.tensor(np.array(peptide_coords), dtype=torch.float32)  
            # only alpha c:
            # peptide_coords = peptide_coords[:, 1, :]
            
            restypes = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
            peptide_aatype = pdb_string['peptide']['aatype']
            peptide_seq = [restypes[i] for i in peptide_aatype]
            peptide_aatype = torch.tensor(peptide_aatype, dtype=torch.long)
            peptide_onehot = one_hot_encode_sequence(peptide_seq)

            protein_coords = pdb_string['protein']['atom14_gt_positions']
            protein_coords = torch.tensor(np.array(protein_coords), dtype=torch.float32) 
            # only alpha c:
            # protein_coords = protein_coords[:, 1, :]
            
            protein_aatype = pdb_string['protein']['aatype']
            protein_aatype = torch.tensor(protein_aatype, dtype=torch.long)
            protein_seq = [restypes[i] for i in protein_aatype]
            protein_onehot = one_hot_encode_sequence(protein_seq)

            # Generate masks (assuming all residues are valid for now)
            peptide_len = peptide_coords.shape[0]
            protein_len = protein_coords.shape[0]
            peptide_mask = torch.ones(peptide_len, dtype=torch.bool)
            protein_mask = torch.ones(protein_len, dtype=torch.bool)

            peptide_torsion_angles = torch.tensor(np.array(pdb_string['peptide']['torsion_angles_sin_cos']), dtype=torch.float32)
            peptide_backbone_rigid_tensor = torch.tensor(np.array(pdb_string['peptide']["rigidgroups_gt_frames"][..., 0, :, :]), dtype=torch.float32)
            protein_backbone_rigid_tensor = torch.tensor(np.array(pdb_string['protein']["rigidgroups_gt_frames"][..., 0, :, :]), dtype=torch.float32)

            peptide_atom14_gt_exists = torch.tensor(
                np.array([restype_atom14_mask[i] for i in peptide_aatype.tolist()]),
                dtype=torch.bool,
            )
            

            peptide_atom14_alt_gt_positions = torch.tensor(
                np.array(pdb_string["peptide"]["atom14_alt_gt_positions"]), dtype=torch.float32,
            )

            peptide_alt_torsion_angles_sin_cos = torch.tensor(
                np.array(pdb_string["peptide"]["alt_torsion_angles_sin_cos"]), dtype=torch.float32,
            )

            # Prepare the output dictionary
            data['graph_name'] = self.pdb_names[index]
            data['peptide_idx'] = peptide_mask
            data['peptide_positions'] = peptide_coords  
            data['peptide_features'] = peptide_onehot  # One-hot encoded peptide sequence
            data['num_peptide_residues'] = peptide_len
            data['protein_pocket_idx'] = protein_mask
            data['protein_pocket_positions'] = protein_coords  # 3D C-alpha coordinates for protein
            data['protein_pocket_features'] = protein_onehot  # One-hot encoded protein sequence
            data['num_protein_pocket_residues'] = protein_len
            data['pos_in_seq'] = torch.arange(peptide_len) + 1  # Position in the sequence

            data['peptide_torsion_angles_sin_cos'] = peptide_torsion_angles
            data['peptide_backbone_rigid_tensor'] = peptide_backbone_rigid_tensor
            data['protein_backbone_rigid_tensor'] = protein_backbone_rigid_tensor

            # print(f"{data['graph_name']=}")
            # print(f"{data['peptide_idx'].shape=}")                     #[9]     
            # print(f"{data['peptide_positions'].shape=}")               #[9, 14, 3]
            # print(f"{data['peptide_features'].shape=}")               # [9, 20]
            # print(f"{data['num_peptide_residues']}")                  #  9
            # print(f"{data['protein_pocket_idx'].shape=}")             # [180]
            # print(f"{data['protein_pocket_positions'].shape=}")       # [180, 14, 3]
            # print(f"{data['protein_pocket_features'].shape=}")        # [180, 20]
            # print(f"{data['num_protein_pocket_residues']}")           # 180
            # print(f"{data['pos_in_seq'].shape=}")                      #[9]
            # print(f"{data['peptide_torsion_angles_sin_cos'].shape=}")  #[9, 7, 2]
            # print(f"{data['peptide_backbone_rigid_tensor'].shape=}")   #[9, 4, 4]
            # print(f"{data['protein_backbone_rigid_tensor'].shape=}")  #[180, 4, 4]

            data['peptide_aatype'] = peptide_aatype
            data['protein_aatype'] = protein_aatype

            data["peptide_atom14_gt_exists"] = peptide_atom14_gt_exists
            data["peptide_atom14_alt_gt_positions"] = peptide_atom14_alt_gt_positions
            data["peptide_alt_torsion_angles_sin_cos"] = peptide_alt_torsion_angles_sin_cos
            
            torsion_angles_mask = torch.ones((peptide_len, 7))
            torsion_angles_mask[:, 3:] = torch.tensor([chi_angles_mask[i] for i in peptide_aatype])
            data["peptide_torsion_angles_mask"] = torsion_angles_mask

            data['peptide_cross_residues_mask'] = torch.tensor([True] * peptide_len, dtype=torch.bool)
            data['protein_cross_residues_mask'] = torch.tensor(np.array(pdb_string["protein"]["cross_residues_mask"]))
            data['affinity'] = torch.tensor(np.array(pdb_string['affinity']))

            max_length = 14 #TODO: 14? 16?
            start_index = 0
            length = peptide_aatype.shape[0]

            index = torch.zeros(max_length, dtype=torch.bool)
            index[:length] = True
            
            data["peptide_residue_index"] = torch.zeros(max_length, dtype=torch.long)
            data["peptide_residue_index"][index] = torch.arange(start_index, start_index + length, 1)
            
            start_index = max_length + 3
            max_length = 180 #TODO: 180? 200?
            length = protein_aatype.shape[0]

            index = torch.zeros(max_length, dtype=torch.bool)
            index[:length] = True

            data["protein_residue_index"] = torch.zeros(max_length, dtype=torch.long)
            data["protein_residue_index"][index] = torch.arange(start_index, start_index + length, 1)



        return data

    @staticmethod
    def collate_fn(batch):
        """
        Collation function to combine batch data into a single batch.
        
        Args:
            batch (list of Dict): A list of individual data entries.

        Returns:
            Dict: A dictionary containing batched data.
        """
        data_batch = {}

        for key in batch[0].keys():

            if key == 'graph_name':
                data_batch[key] = [x[key] for x in batch]
            elif key == 'num_peptide_residues' or key == 'num_protein_pocket_residues':
                data_batch[key] = torch.tensor([x[key] for x in batch])
            elif 'idx' in key:
                # Ensure that indices in the batch start at zero (needed for torch_scatter)
                data_batch[key] = torch.cat([i * torch.ones(len(x[key]), dtype=torch.long) for i, x in enumerate(batch)], dim=0)
            elif key == 'affinity':
                data_batch[key] = torch.tensor([x[key] for x in batch])
            else:
                data_batch[key] = torch.cat([x[key] for x in batch], dim=0)

        return data_batch


class PDB_Dataset_all(Dataset):

    def __init__(self, datadir, split='train', fold="1", path=None):
        """
        Args:
            datadir (str): Path to the directory where HDF5 files are located.
            split (str): Dataset split, one of 'train', 'valid', 'test'.
        """


        # datadir = "/scratch-shared/roos/preprocessed/"

        if datadir == "/scratch-shared/roos/preprocessed/":

            if split == 'test': # roos
                # self.hdf5_path = os.path.join(datadir, f'BA_cluster{fold}.hdf5')
                # self.hdf5_path = os.path.join(datadir, f'xray_cluster{fold}.hdf5')
                self.hdf5_path = os.path.join(datadir, f'{path}.hdf5')
            else:
                self.hdf5_path = os.path.join(datadir, f'{split}_fold{fold}.hdf5')
        else:
            self.hdf5_path = os.path.join(datadir, f'{split}.hdf5')
        print(f"Loading dataset from {self.hdf5_path}...")

        # Open file to get the list of keys (Entry IDs like 'BA-55224')
        with h5py.File(self.hdf5_path, 'r') as f5:
            # Instead of looking for 'pdb_strings', we get the group keys
            self.entry_names = list(f5.keys())
            
            
        print(f"Loaded {len(self.entry_names)} entries from {split} split.")

    def __len__(self) -> int:
        return len(self.entry_names)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.get_entry(index)

    def get_entry(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves pre-processed tensors and forces them to 20 dimensions.
        """
        entry_name = self.entry_names[index]
        data = {}

        with h5py.File(self.hdf5_path, 'r') as f5:
            group = f5[entry_name]

            #print all keys in group and also all the keys in the keys
            # print(f"\n[ENTRY: {entry_name}]")
            # for name, dataset in group.items():
            #     # Skip if it's a nested group, only print datasets
            #     if isinstance(dataset, h5py.Dataset):
            #         print(f"  - {name}: shape {dataset.shape}, dtype {dataset.dtype}")
            #         print(dataset)
            #     else:
            #         print(f"  - {name}: is a nested group.")
            #         for sub_name, sub_dataset in dataset.items():
            #             print(f"    - {sub_name}: shape {sub_dataset.shape}, dtype {sub_dataset.dtype}")
            #             print(sub_dataset)
            
            # --- 1. Load Peptide Data ---
            # Position: Take C-alpha (index 1)
            pep_all_pos = group['peptide']['atom14_gt_positions']
           
            peptide_coords = torch.tensor(np.array(pep_all_pos), dtype=torch.float32)  # Use all atom positions
            # peptide_coords = torch.tensor(pep_all_pos[:, 1, :], dtype=torch.float32)

            restypes = [
                "A",
                "R",
                "N",
                "D",
                "C",
                "Q",
                "E",
                "G",
                "H",
                "I",
                "L",
                "K",
                "M",
                "F",
                "P",
                "S",
                "T",
                "W",
                "Y",
                "V",
            ]
            
            # Features: SLICE the first 20 columns only
            aatype = group['peptide']['aatype']
            peptide_seq = [restypes[i] for i in aatype]
            # print(f"Original peptide sequence: {''.join(peptide_seq)}")

            # Force shape (N, 20)
            # peptide_onehot = torch.tensor(full_pep_onehot[:, :20], dtype=torch.float32)
            peptide_onehot = one_hot_encode_sequence(peptide_seq)
            # print(f"One-hot encoded peptide features: {peptide_onehot}")
            
            # pep_pos_in_seq = torch.tensor(group['peptide']['residue_numbers'][:], dtype=torch.long)

            # --- 2. Load Protein Data ---
            pro_all_pos = group['protein']['atom14_gt_positions']
            protein_coords = torch.tensor(np.array(pro_all_pos), dtype=torch.float32)  # Use all atom positions
            # protein_coords = torch.tensor(pro_all_pos[:, 1, :], dtype=torch.float32)
            
            # Features: SLICE the first 20 columns only
            aatype = group['protein']['aatype']
            protein_seq = [restypes[i] for i in aatype]
            # print(f"Original protein sequence: {''.join(protein_seq)}")

            # Force shape (N, 20)
            # protein_onehot = torch.tensor(full_pro_onehot[:, :20], dtype=torch.float32)
            protein_onehot = one_hot_encode_sequence(protein_seq)

            # --- 3. Create Masks ---
            peptide_len = peptide_coords.shape[0]
            protein_len = protein_coords.shape[0]
            
            peptide_mask = torch.ones(peptide_len, dtype=torch.bool)
            protein_mask = torch.ones(protein_len, dtype=torch.bool)

            # peptide_onehot_expanded = peptide_onehot.unsqueeze(1)
            # peptide_onehot = peptide_onehot_expanded.expand(-1, peptide_coords.shape[1], -1)
            # protein_onehot_expanded = protein_onehot.unsqueeze(1)
            # protein_onehot = protein_onehot_expanded.expand(-1, protein_coords.shape[1], -1)

            peptide_torsion_angles = torch.tensor(np.array(group['peptide']['torsion_angles_sin_cos']), dtype=torch.float32)

            # --- 4. Package ---
            data['graph_name'] = entry_name
            
            # Peptide
            data['peptide_idx'] = peptide_mask
            data['peptide_positions'] = peptide_coords
            data['peptide_features'] = peptide_onehot  
            data['num_peptide_residues'] = peptide_len
            data['pos_in_seq'] = torch.arange(peptide_len) + 1 
            # print(f"{peptide_coords.shape=}, {peptide_onehot.shape=}")

            # Protein Pocket
            data['protein_pocket_idx'] = protein_mask
            data['protein_pocket_positions'] = protein_coords
            data['protein_pocket_features'] = protein_onehot  
            data['num_protein_pocket_residues'] = protein_len
            # print(f"{protein_coords.shape=}, {protein_onehot.shape=}")

            data['peptide_torsion_angles_sin_cos'] = peptide_torsion_angles

            protein_backbone_rigid_tensor = group['protein']["rigidgroups_gt_frames"][..., 0, :, :]
            protein_backbone_rigid_tensor = torch.tensor(np.array(protein_backbone_rigid_tensor))
            data['protein_backbone_rigid_tensor'] = protein_backbone_rigid_tensor
            # print(f"{protein_backbone_rigid_tensor.shape=}")
            # T_protein = Rigid.from_tensor_4x4(protein_backbone_rigid_tensor)
            # print(f"{T_protein.shape=}")
            # T_protein = T_protein.to_tensor_7()
            # print(f"{T_protein.shape=}")
        
            peptide_backbone_rigid_tensor = group['peptide']["rigidgroups_gt_frames"][..., 0, :, :]
            peptide_backbone_rigid_tensor = torch.tensor(np.array(peptide_backbone_rigid_tensor))
            data['peptide_backbone_rigid_tensor'] = peptide_backbone_rigid_tensor
            # print(f"{peptide_backbone_rigid_tensor.shape=}")
            # T_peptide = Rigid.from_tensor_4x4(peptide_backbone_rigid_tensor)
            # print(f"{T_peptide.shape=}")
            # T_peptide = T_peptide.to_tensor_7()
            # print(f"{T_peptide.shape=}")
            

        return data

    @staticmethod
    def collate_fn(batch):
        """
        Collation function to combine batch data into a single batch.
        
        Args:
            batch (list of Dict): A list of individual data entries.

        Returns:
            Dict: A dictionary containing batched data.
        """
        data_batch = {}

        for key in batch[0].keys():

            if key == 'graph_name':
                data_batch[key] = [x[key] for x in batch]
            elif key == 'num_peptide_residues' or key == 'num_protein_pocket_residues':
                data_batch[key] = torch.tensor([x[key] for x in batch])
            elif 'idx' in key:
                # Ensure that indices in the batch start at zero (needed for torch_scatter)
                data_batch[key] = torch.cat([i * torch.ones(len(x[key]), dtype=torch.long) for i, x in enumerate(batch)], dim=0)
            else:
                data_batch[key] = torch.cat([x[key] for x in batch], dim=0)

        return data_batch
    

class PDB_Dataset_swift(Dataset):

    # Mapping of which xray clusters go to test/valid for each fold.
    # The remaining clusters go to train.
    XRAY_FOLD_MAPPING = {
        "1":  {"test": [0], "valid": [6]},
        "2":  {"test": [1], "valid": [7]},
        "3":  {"test": [2], "valid": [5]},
        "4":  {"test": [3], "valid": [2]},
        "5":  {"test": [4], "valid": [6]},
        "6":  {"test": [5], "valid": [7]},
        "7":  {"test": [6], "valid": [5]},
        "8":  {"test": [7], "valid": [2]},
        "9":  {"test": [8], "valid": [7]},
        "10": {"test": [9], "valid": [7]},
    }

    def __init__(self, datadir="/scratch-shared/roos/preprocessed/", split='train', fold="1", path=None):
        """
        Args:
            datadir (str): Path to the directory where HDF5 files are located.
            split (str): Dataset split, one of 'train', 'valid', 'test', or 'BA'.
            fold (str): Fold number (1-indexed, matching CA-only fold_1 through fold_10).
            path (str): Specific file name for test clusters.
        """
        self.datadir = datadir
        
        # Preprocessed files are 0-indexed (train_fold0..9, BA_cluster0..9)
        # while fold numbers in config are 1-indexed (fold 1..10, matching CA fold_1..fold_10)
        file_idx = int(fold) - 1

        # entry_sources maps each index to (hdf5_path, entry_name)
        self.entry_sources = []
        self.entry_names = []

        if split == 'test' and path is not None:
            self.hdf5_path = os.path.join(datadir, f'{path}.hdf5')
            self._load_entries_from_file(self.hdf5_path)
        elif split == 'BA':
            self.hdf5_path = os.path.join(datadir, f'BA_cluster{file_idx}.hdf5')
            self._load_entries_from_file(self.hdf5_path)
            # Also load xray test cluster(s) for this fold
            fold_mapping = self.XRAY_FOLD_MAPPING.get(str(fold), None)
            if fold_mapping is not None:
                for cluster_id in fold_mapping["test"]:
                    xray_path = os.path.join(datadir, f'xray_cluster{cluster_id}.hdf5')
                    if os.path.exists(xray_path):
                        self._load_entries_from_file(xray_path)
                    else:
                        print(f"Warning: xray cluster file not found: {xray_path}")
        else:
            # Load BA (Pandora) data
            ba_path = os.path.join(datadir, f'{split}_fold{file_idx}.hdf5')
            self.hdf5_path = ba_path  # keep for backward compat
            self._load_entries_from_file(ba_path)

            # Load xray cluster data for this fold and split
            fold_mapping = self.XRAY_FOLD_MAPPING.get(str(fold), None)
            if fold_mapping is not None:
                test_clusters = fold_mapping["test"]
                valid_clusters = fold_mapping["valid"]

                if split == 'test':
                    xray_clusters = test_clusters
                elif split == 'valid':
                    xray_clusters = valid_clusters
                else:  # train
                    xray_clusters = [c for c in range(10) if c not in test_clusters and c not in valid_clusters]

                for cluster_id in xray_clusters:
                    xray_path = os.path.join(datadir, f'xray_cluster{cluster_id}.hdf5')
                    if os.path.exists(xray_path):
                        self._load_entries_from_file(xray_path)
                    else:
                        print(f"Warning: xray cluster file not found: {xray_path}")
            
        print(f"Loaded {len(self.entry_names)} entries from {split} split (fold {fold}).")

    def _load_entries_from_file(self, hdf5_path):
        """Load entry names from an HDF5 file and track their source."""
        print(f"Loading SwiftMHC dataset from {hdf5_path}...")
        with h5py.File(hdf5_path, 'r') as f5:
            names = list(f5.keys())
        for name in names:
            self.entry_sources.append((hdf5_path, name))
            self.entry_names.append(name)
        print(f"  -> {len(names)} entries from {os.path.basename(hdf5_path)}")

    def __len__(self) -> int:
        return len(self.entry_names)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.get_entry(index)

    def get_entry(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves pre-processed tensors for a single pMHC entry.
        """
        hdf5_path, entry_name = self.entry_sources[index]
        data = {}

        with h5py.File(hdf5_path, 'r') as f5:
            group = f5[entry_name]
            
            restypes = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
            
            # --- 1. Load Peptide Data ---
            pep_group = group['peptide']
            peptide_coords = torch.tensor(np.array(pep_group['atom14_gt_positions']), dtype=torch.float32)
            peptide_aatype = torch.tensor(pep_group['aatype'][:], dtype=torch.long)
            peptide_seq = [restypes[i] for i in peptide_aatype]
            peptide_onehot = one_hot_encode_sequence(peptide_seq)
            
            peptide_len = peptide_coords.shape[0]
            peptide_mask = torch.ones(peptide_len, dtype=torch.bool)
            
            peptide_torsion_angles = torch.tensor(np.array(pep_group['torsion_angles_sin_cos']), dtype=torch.float32)
            peptide_backbone_rigid_tensor = torch.tensor(np.array(pep_group["rigidgroups_gt_frames"][..., 0, :, :]), dtype=torch.float32)

            # Extra features for compatibility
            peptide_atom14_gt_exists = torch.tensor(
                np.array([restype_atom14_mask[i] for i in peptide_aatype.tolist()]),
                dtype=torch.bool,
            )
            peptide_atom14_alt_gt_positions = torch.tensor(
                np.array(pep_group.get("atom14_alt_gt_positions", np.zeros_like(pep_group['atom14_gt_positions']))), 
                dtype=torch.float32,
            )
            peptide_alt_torsion_angles_sin_cos = torch.tensor(
                np.array(pep_group.get("alt_torsion_angles_sin_cos", np.zeros_like(pep_group['torsion_angles_sin_cos']))), 
                dtype=torch.float32,
            )
            
            torsion_angles_mask = torch.ones((peptide_len, 7))
            torsion_angles_mask[:, 3:] = torch.tensor([chi_angles_mask[i] for i in peptide_aatype])
            
            # --- 2. Load Protein Data ---
            pro_group = group['protein']
            protein_coords = torch.tensor(np.array(pro_group['atom14_gt_positions']), dtype=torch.float32)
            protein_aatype = torch.tensor(pro_group['aatype'][:], dtype=torch.long)
            protein_seq = [restypes[i] for i in protein_aatype]
            protein_onehot = one_hot_encode_sequence(protein_seq)
            
            protein_len = protein_coords.shape[0]
            protein_mask = torch.ones(protein_len, dtype=torch.bool)
            
            protein_backbone_rigid_tensor = torch.tensor(np.array(pro_group["rigidgroups_gt_frames"][..., 0, :, :]), dtype=torch.float32)

            # --- 3. Package Data ---
            data['graph_name'] = entry_name
            data['peptide_idx'] = peptide_mask
            data['peptide_positions'] = peptide_coords
            data['peptide_features'] = peptide_onehot
            data['num_peptide_residues'] = peptide_len
            data['pos_in_seq'] = torch.arange(peptide_len) + 1
            
            data['protein_pocket_idx'] = protein_mask
            data['protein_pocket_positions'] = protein_coords
            data['protein_pocket_features'] = protein_onehot
            data['num_protein_pocket_residues'] = protein_len
            
            data['peptide_torsion_angles_sin_cos'] = peptide_torsion_angles
            data['peptide_backbone_rigid_tensor'] = peptide_backbone_rigid_tensor
            data['protein_backbone_rigid_tensor'] = protein_backbone_rigid_tensor
            
            data['peptide_aatype'] = peptide_aatype
            data['protein_aatype'] = protein_aatype
            
            data["peptide_atom14_gt_exists"] = peptide_atom14_gt_exists
            data["peptide_atom14_alt_gt_positions"] = peptide_atom14_alt_gt_positions
            data["peptide_alt_torsion_angles_sin_cos"] = peptide_alt_torsion_angles_sin_cos
            data["peptide_torsion_angles_mask"] = torsion_angles_mask
            
            data['peptide_cross_residues_mask'] = torch.tensor([True] * peptide_len, dtype=torch.bool)
            data['protein_cross_residues_mask'] = torch.tensor(np.array(pro_group.get("cross_residues_mask", np.ones(protein_len, dtype=bool))))
            
            if 'affinity' in group:
                data['affinity'] = torch.tensor(np.array(group['affinity']), dtype=torch.float32)

            # BA mask: True if this entry has a real binding affinity score (Pandora/BA- entries),
            # False for X-ray entries which have a dummy affinity of 1.0
            data['ba_mask'] = torch.tensor(entry_name.startswith('BA-'), dtype=torch.bool)

            # Residue Index (SwiftMHC style)
            pep_max_len = 14 # Default for peptide
            pro_max_len = 180 # Default for protein
            
            data["peptide_residue_index"] = torch.arange(0, peptide_len, 1, dtype=torch.long)
            data["protein_residue_index"] = torch.arange(pep_max_len + 3, pep_max_len + 3 + protein_len, 1, dtype=torch.long)

        return data

    @staticmethod
    def collate_fn(batch):
        """
        Collation function to combine batch data into a single batch.
        """
        data_batch = {}

        for key in batch[0].keys():
            if key == 'graph_name':
                data_batch[key] = [x[key] for x in batch]
            elif key in ['num_peptide_residues', 'num_protein_pocket_residues', 'affinity', 'ba_mask']:
                if key in batch[0]:
                    data_batch[key] = torch.tensor([x[key] for x in batch])
            elif 'idx' in key:
                data_batch[key] = torch.cat([i * torch.ones(len(x[key]), dtype=torch.long) for i, x in enumerate(batch)], dim=0)
            else:
                data_batch[key] = torch.cat([x[key] for x in batch], dim=0)

        return data_batch

    # @staticmethod
    # def collate_fn(batch):
    #     """
    #     Collation function to combine batch data into a single batch.
    #     """
    #     data_batch = {}
        
    #     # Keys that are lists of strings
    #     data_batch['graph_name'] = [x['graph_name'] for x in batch]
        
    #     # Keys that are single values per graph (1D tensor)
    #     for key in ['num_peptide_residues', 'num_protein_pocket_residues']:
    #          data_batch[key] = torch.stack([x[key] for x in batch])

    #     # Keys that need concatenation (Node features/positions)
    #     cat_keys = [
    #         'peptide_idx', 'peptide_positions', 'peptide_features', 'pos_in_seq',
    #         'protein_pocket_idx', 'protein_pocket_positions', 'protein_pocket_features'
    #     ]
        
    #     for key in cat_keys:
    #         data_batch[key] = torch.cat([x[key] for x in batch], dim=0)

    #     # Handle Batch Indices (needed for torch_scatter usually)
    #     # We create a new 'idx' key that maps every node to its graph index in the batch
    #     peptide_batch_indices = []
    #     protein_batch_indices = []
        
    #     for i, item in enumerate(batch):
    #         peptide_batch_indices.append(torch.full((item['num_peptide_residues'],), i, dtype=torch.long))
    #         protein_batch_indices.append(torch.full((item['num_protein_pocket_residues'],), i, dtype=torch.long))
            
    #     data_batch['idx_peptide'] = torch.cat(peptide_batch_indices)
    #     data_batch['idx_protein'] = torch.cat(protein_batch_indices)
        
    #     # Map back to generic 'idx' if your model uses that specific name
    #     # (Assuming your model looks for 'peptide_idx' as the scatter index, 
    #     # usually usually usually named 'batch' or 'idx' in PyG)
    #     data_batch['peptide_idx'] = data_batch['idx_peptide'] 
    #     data_batch['protein_pocket_idx'] = data_batch['idx_protein']
        

    #     return data_batch