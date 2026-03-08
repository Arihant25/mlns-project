"""
model.py
========
Phase 1A — Model definitions: a lightweight PyTorch Dataset for pre-computed
embeddings and the BaselineMLP architecture.

Architecture
------------
Input (768) → Linear(512) → GELU → Dropout(0.1)
           → Linear(256) → GELU → Dropout(0.1)
           → Linear(1)
"""

import torch
from torch import nn
from torch.utils.data import Dataset


# ── Dataset ──────────────────────────────────────────────────────────────────
class EmbeddingDataset(Dataset):
    """
    Wraps pre-computed embedding tensors and their regression targets.

    Parameters
    ----------
    embeddings : torch.Tensor, shape (N, 768)
    targets : torch.Tensor, shape (N,)
    """

    def __init__(self, embeddings: torch.Tensor, targets: torch.Tensor):
        assert embeddings.shape[0] == targets.shape[0], (
            f"Mismatch: {embeddings.shape[0]} embeddings vs {targets.shape[0]} targets"
        )
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.targets[idx]


# ── MLP ──────────────────────────────────────────────────────────────────────
class BaselineMLP(nn.Module):
    """
    Three-layer MLP for scalar regression from 768-d MolFormer embeddings.

    Architecture:
        768 → 512 → GELU → Dropout → 256 → GELU → Dropout → 1
    """

    def __init__(self,
                 input_dim: int = 768,
                 hidden1: int = 512,
                 hidden2: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 768)

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        return self.net(x)
