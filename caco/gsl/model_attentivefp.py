"""
model_attentivefp.py
====================
AttentiveFP-style model operating on MolFormer embeddings with a learned
similarity graph prior.

This is a lightweight attentive message-passing variant that follows the same
input protocol as the existing GSL models:
    forward(X, A_ecfp) -> predictions
"""

import math

import torch
from model_gsl import LearnedGraphMaker
from torch import nn


class AttentiveFPSimModel(nn.Module):
    """
    AttentiveFP-style regression model over a learned molecular graph.

    Steps:
    1) Build sparse adjacency with LearnedGraphMaker.
    2) Compute masked self-attention over neighbors.
    3) Residual fusion + feed-forward refinement.
    4) MLP regression head.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 768,
        top_k: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.graph_maker = LearnedGraphMaker(embed_dim=embed_dim, top_k=top_k)

        self.q_proj = nn.Linear(embed_dim, hidden_dim)
        self.k_proj = nn.Linear(embed_dim, hidden_dim)
        self.v_proj = nn.Linear(embed_dim, hidden_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, X: torch.Tensor, A_ecfp: torch.Tensor) -> torch.Tensor:
        # Learned sparse adjacency and self-loop augmentation.
        A = self.graph_maker(X, A_ecfp)
        A = A + torch.eye(A.size(0), device=A.device)

        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)

        scores = (q @ k.T) / math.sqrt(q.size(-1))

        # Mask non-neighbors in attention; every node has self-loop so valid rows.
        mask = A <= 0
        scores = scores.masked_fill(mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        m = attn @ v
        h = self.norm1(X + m)

        z = self.ffn(h)
        h = self.norm2(h + z)

        return self.head(h)
