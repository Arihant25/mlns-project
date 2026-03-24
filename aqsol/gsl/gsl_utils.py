"""
gsl_utils.py
============
Shared dataset and collate utilities for GSL-based models.
"""

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from torch.utils.data import Dataset


class GSLDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, targets: torch.Tensor, smiles: list):
        assert embeddings.shape[0] == targets.shape[0] == len(smiles)
        self.embeddings = embeddings
        self.targets = targets
        self.smiles = smiles

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.targets[idx], self.smiles[idx]


def _compute_ecfp_tanimoto(smiles_list: list) -> torch.Tensor:
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles("C"), 2, nBits=1024
            )
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fps.append(fp)

    n = len(fps)
    A = torch.zeros(n, n)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for j, s in enumerate(sims, start=i + 1):
            A[i, j] = s
            A[j, i] = s
    return A


def gsl_collate_fn(batch):
    embeddings, targets, smiles_list = zip(*batch)
    embeddings = torch.stack(embeddings)
    targets = torch.stack(targets)
    A_ecfp = _compute_ecfp_tanimoto(list(smiles_list))
    return embeddings, targets, A_ecfp
