"""
model_evidential_gsl.py
=======================
Phase 2B — Evidential GSL model with sender-indexed aleatoric uncertainty
gating and an explicit skip connection.

The model first estimates per-node aleatoric uncertainty via an initial
evidential head, then uses it to gate neighbor contributions during
message passing, preventing noise propagation at activity cliffs.
"""

import torch
from torch import nn

from model_gsl import LearnedGraphMaker


# ── Evidential Head ──────────────────────────────────────────────────────────
class EvidentialHead(nn.Module):
    """
    MLP head that outputs NIG parameters (μ, ν, α, β) with appropriate
    softplus constraints.

    Architecture:
        input_dim → 512 → GELU → Dropout → 256 → GELU → Dropout → 4
    """

    def __init__(self,
                 input_dim: int = 768,
                 hidden1: int = 512,
                 hidden2: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden2, 4)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        mu    : (B,) predicted mean — identity
        v     : (B,) virtual evidence  (> 0)
        alpha : (B,) IG shape          (> 1)
        beta  : (B,) IG scale          (> 0)
        """
        h = self.backbone(x)
        raw = self.head(h)                                  # (B, 4)

        mu    = raw[:, 0]
        v     = self.softplus(raw[:, 1]) + 1e-6
        alpha = self.softplus(raw[:, 2]) + 1.0 + 1e-6
        beta  = self.softplus(raw[:, 3]) + 1e-6

        return mu, v, alpha, beta


# ── Evidential GSL Model ────────────────────────────────────────────────────
class EvidentialGSLModel(nn.Module):
    """
    Uncertainty-gated Graph Structure Learning model.

    Forward pass:
        1. Initial evidential head  →  per-node (μ₀, ν₀, α₀, β₀)
        2. Aleatoric uncertainty    →  u₀ = β₀ / (α₀ − 1)
        3. Sender gate              →  G = 1 − γ · σ(u₀)
        4. Graph construction       →  A = LearnedGraphMaker(X, A_ecfp)
        5. Gated aggregation        →  M = GELU( W · (A_norm @ (X ⊙ G)) )
        6. Skip connection          →  H = X + M
        7. Final evidential head    →  (μ, ν, α, β)
    """

    def __init__(self,
                 embed_dim: int = 768,
                 top_k: int = 5,
                 dropout: float = 0.1):
        super().__init__()

        # Graph constructor (shared with Phase 1B)
        self.graph_maker = LearnedGraphMaker(embed_dim, top_k=top_k)

        # Initial evidential head (pre-aggregation uncertainty estimate)
        self.initial_head = EvidentialHead(embed_dim, dropout=dropout)

        # Single GCN message-passing layer
        self.gcn_linear = nn.Linear(embed_dim, embed_dim)
        self.gcn_act = nn.GELU()

        # Final evidential head (post-aggregation prediction)
        self.final_head = EvidentialHead(embed_dim, dropout=dropout)

        # Learnable gate scaling hyperparameter
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, X: torch.Tensor, A_ecfp: torch.Tensor):
        """
        Parameters
        ----------
        X      : (B, 768) frozen MolFormer embeddings
        A_ecfp : (B, B)   ECFP Tanimoto similarity matrix

        Returns
        -------
        initial : tuple (mu_0, v_0, alpha_0, beta_0)  — pre-aggregation
        final   : tuple (mu, v, alpha, beta)           — post-aggregation
        """
        # ── Step 1: Initial evidential estimate ──
        mu_0, v_0, alpha_0, beta_0 = self.initial_head(X)

        # ── Step 2: Per-node aleatoric uncertainty ──
        u_0 = beta_0 / (alpha_0 - 1.0)                     # (B,)

        # ── Step 3: Sender-indexed gate ──
        # High uncertainty → sigmoid(u) → 1 → G → small → suppress sender
        G = 1.0 - self.gamma * torch.sigmoid(u_0)           # (B,)
        G = G.unsqueeze(-1)                                  # (B, 1)

        # ── Step 4: Learn adjacency (diagonal zeroed) ──
        A = self.graph_maker(X, A_ecfp)                      # (B, B)

        # ── Step 5: Gated GCN aggregation ──
        X_gated = X * G                                      # (B, 768)

        # Row-normalise A
        D_inv = 1.0 / A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        A_norm = A * D_inv                                    # (B, B)

        M = self.gcn_act(self.gcn_linear(A_norm @ X_gated))   # (B, 768)

        # ── Step 6: Skip connection (isolate self-update) ──
        H = X + M                                             # (B, 768)

        # ── Step 7: Final evidential estimate ──
        mu, v, alpha, beta = self.final_head(H)

        return (mu_0, v_0, alpha_0, beta_0), (mu, v, alpha, beta)
