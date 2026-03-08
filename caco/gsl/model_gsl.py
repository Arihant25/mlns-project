"""
model_gsl.py
============
Phase 1B — Graph Structure Learning model for molecular property prediction.

Implements metric-based graph construction (inspired by GSL-MPP / Zhao et al.,
2024) with an ECFP similarity prior, Top-K sparsification, and a single-layer
GCN message-passing step followed by the same MLP head used in Phase 1A.
"""

import torch
from torch import nn


# ── Graph Constructor ────────────────────────────────────────────────────────
class LearnedGraphMaker(nn.Module):
    """
    Construct a sparse, symmetric adjacency matrix by blending a learnable
    latent similarity with a pre-computed ECFP Tanimoto prior.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of node embeddings (768 for MolFormer-XL).
    top_k : int
        Number of nearest neighbours to retain per node after sparsification.
    """

    def __init__(self, embed_dim: int = 768, top_k: int = 5):
        super().__init__()
        # Learnable metric matrix for latent similarity: S = ReLU(X W_g X^T)
        self.W_g = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.W_g)

        # Learnable blending coefficient (raw logit; sigmoided at runtime)
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

        self.top_k = top_k

    def forward(self, X: torch.Tensor, A_ecfp: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X      : (B, D) node embeddings
        A_ecfp : (B, B) pre-computed ECFP Tanimoto similarity matrix

        Returns
        -------
        A : (B, B) sparse, symmetric adjacency matrix (diagonal zeroed)
        """
        # 1. Latent similarity via learnable metric
        S = torch.relu(X @ self.W_g @ X.T)                 # (B, B)

        # 2. Blend structural prior with learned similarity
        alpha = torch.sigmoid(self.raw_alpha)               # ∈ (0, 1)
        A = alpha * A_ecfp + (1.0 - alpha) * S

        # 3. Top-K sparsification (per row)
        k = min(self.top_k, A.size(0) - 1)                 # safety for tiny batches
        _, top_idx = A.topk(k, dim=-1)
        mask = torch.zeros_like(A)
        mask.scatter_(1, top_idx, 1.0)

        # 4. Symmetrise mask and apply
        mask = ((mask + mask.T) > 0).float()
        A = A * mask

        # 5. Zero the diagonal (self-loops added later in the model)
        A.fill_diagonal_(0.0)

        return A


# ── Full GSL Model ───────────────────────────────────────────────────────────
class SimpleGSLModel(nn.Module):
    """
    One-layer GCN on a learned molecular similarity graph, followed by the
    same MLP regression head as the Phase 1A BaselineMLP.

    Architecture
    ------------
    LearnedGraphMaker → A
    A ← A + I  (self-loops)
    H = GELU( D⁻¹ A X W )          single GCN layer  (768 → 768)
    ŷ = MLP(H)                      768 → 512 → 256 → 1
    """

    def __init__(self,
                 embed_dim: int = 768,
                 top_k: int = 5,
                 dropout: float = 0.1):
        super().__init__()

        # Graph constructor
        self.graph_maker = LearnedGraphMaker(embed_dim, top_k=top_k)

        # Single GCN message-passing layer
        self.gcn_linear = nn.Linear(embed_dim, embed_dim)
        self.gcn_act = nn.GELU()

        # Regression head (identical to Phase 1A BaselineMLP)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self,
                X: torch.Tensor,
                A_ecfp: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X      : (B, 768) frozen MolFormer embeddings
        A_ecfp : (B, B)   ECFP Tanimoto similarity matrix

        Returns
        -------
        preds : (B, 1) scalar property predictions
        """
        # 1. Learn the adjacency matrix
        A = self.graph_maker(X, A_ecfp)                     # (B, B)

        # 2. Add self-loops
        A = A + torch.eye(A.size(0), device=A.device)

        # 3. Row-normalise (D⁻¹ A) for stable message passing
        D_inv = 1.0 / A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        A_norm = A * D_inv                                   # (B, B)

        # 4. Single GCN layer: H = GELU( A_norm @ X @ W + b )
        H = self.gcn_act(self.gcn_linear(A_norm @ X))        # (B, 768)

        # 5. Per-node regression
        return self.mlp(H)                                    # (B, 1)
