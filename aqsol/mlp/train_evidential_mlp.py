"""
train_evidential_mlp.py
=======================
Phase 2A — Train an Evidential MLP that outputs Normal-Inverse-Gamma (NIG)
parameters and is trained with the error-scaled evidential loss.

Usage:
    python train_evidential_mlp.py
"""

import copy
import os
import time

import numpy as np
import torch
from loss_evidential import ErrorScaledEvidentialLoss
from model import EmbeddingDataset
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
LR_PATIENCE = 5
LOSS_COEFF = 0.1  # error-scaled KL coefficient


# ── Evidential MLP ───────────────────────────────────────────────────────────
class EvidentialMLP(nn.Module):
    """
    Same hidden architecture as the Phase 1A BaselineMLP (768→512→256), but
    the final layer outputs 4 values per sample — the NIG parameters
    (μ, ν, α, β) — with appropriate activations to enforce constraints.

    Architecture:
        768 → 512 → GELU → Dropout(0.1)
            → 256 → GELU → Dropout(0.1)
            → 4   → split → (μ, softplus(ν), softplus(α)+1, softplus(β))
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden1: int = 512,
        hidden2: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden2, 4)  # μ, ν, α, β
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        mu    : (B,) predicted mean — identity activation
        v     : (B,) virtual evidence   (> 0)
        alpha : (B,) IG shape           (> 1)
        beta  : (B,) IG scale           (> 0)
        """
        h = self.backbone(x)
        raw = self.head(h)  # (B, 4)

        mu = raw[:, 0]  # identity
        v = self.softplus(raw[:, 1]) + 1e-6  # > 0
        alpha = self.softplus(raw[:, 2]) + 1.0 + 1e-6  # > 1
        beta = self.softplus(raw[:, 3]) + 1e-6  # > 0

        return mu, v, alpha, beta


# ── Helpers (same as Phase 1A) ───────────────────────────────────────────────
def load_tensors(split: str):
    emb = torch.load(
        os.path.join(DATA_DIR, f"{split}_embeddings.pt"), weights_only=True
    )
    tgt = torch.load(os.path.join(DATA_DIR, f"{split}_targets.pt"), weights_only=True)
    return emb, tgt


def fit_scaler(train_targets: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_targets.reshape(-1, 1))
    return scaler


def scale_targets(targets: np.ndarray, scaler: StandardScaler) -> torch.Tensor:
    scaled = scaler.transform(targets.reshape(-1, 1)).flatten()
    return torch.tensor(scaled, dtype=torch.float32)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Training / Evaluation ───────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        mu, v, alpha, beta = model(x)
        loss = criterion(y, mu, v, alpha, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        n += x.size(0)
    return running_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        mu, v, alpha, beta = model(x)
        loss = criterion(y, mu, v, alpha, beta)
        running_loss += loss.item() * x.size(0)
        n += x.size(0)
    return running_loss / n


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load pre-computed tensors
    print("[1/5] Loading pre-computed embeddings …")
    train_emb, train_tgt = load_tensors("train")
    val_emb, val_tgt = load_tensors("val")
    test_emb, test_tgt = load_tensors("test")
    print(
        f"       Train: {train_emb.shape[0]}, Val: {val_emb.shape[0]}, "
        f"Test: {test_emb.shape[0]}"
    )

    # 2. Fit scaler on training targets only
    print("[2/5] Fitting StandardScaler on training targets …")
    scaler = fit_scaler(train_tgt.numpy())
    train_tgt_s = scale_targets(train_tgt.numpy(), scaler)
    val_tgt_s = scale_targets(val_tgt.numpy(), scaler)
    test_tgt_s = scale_targets(test_tgt.numpy(), scaler)

    # 3. Build DataLoaders
    train_loader = DataLoader(
        EmbeddingDataset(train_emb, train_tgt_s), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(EmbeddingDataset(val_emb, val_tgt_s), batch_size=BATCH_SIZE)
    test_loader = DataLoader(
        EmbeddingDataset(test_emb, test_tgt_s), batch_size=BATCH_SIZE
    )

    # 4. Initialise model, optimiser, scheduler, loss
    print("[3/5] Initialising EvidentialMLP …")
    model = EvidentialMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )
    criterion = ErrorScaledEvidentialLoss(coeff=LOSS_COEFF)
    print(f"       Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"       Loss coeff: {LOSS_COEFF}")

    # 5. Training loop with early stopping
    print("[4/5] Training …")
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    epoch_times = []

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        epoch_times.append(time.perf_counter() - epoch_start)
        avg_epoch_s = sum(epoch_times) / len(epoch_times)
        eta_s = avg_epoch_s * (MAX_EPOCHS - epoch)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{MAX_EPOCHS}  |  "
            f"Train Loss: {train_loss:.6f}  |  "
            f"Val Loss: {val_loss:.6f}  |  "
            f"LR: {lr:.2e}  |  "
            f"ETA: {format_eta(eta_s)}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"  ⇢ Early stopping triggered at epoch {epoch}")
            break

    model_path = os.path.join(RESULTS_DIR, "evidential_mlp.pt")
    torch.save(best_state, model_path)
    print(f"  Best model saved → {model_path}")

    # 6. Test evaluation
    print("[5/5] Evaluating on test set …")
    model.load_state_dict(best_state)
    model.eval()

    all_mu, all_u, all_targets = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            mu, v, alpha, beta = model(x)

            # Aleatoric uncertainty: u = β / (α − 1)
            u = beta / (alpha - 1.0)

            all_mu.append(mu.cpu().numpy())
            all_u.append(u.cpu().numpy())
            all_targets.append(y.numpy())

    mu_scaled = np.concatenate(all_mu)
    u_scaled = np.concatenate(all_u)
    targets_scaled = np.concatenate(all_targets)

    # Inverse-transform predictions and targets to original scale
    mu_orig = scaler.inverse_transform(mu_scaled.reshape(-1, 1)).flatten()
    targets_orig = scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    # ── Prediction metrics ──
    rmse = float(np.sqrt(np.mean((mu_orig - targets_orig) ** 2)))
    mae = float(np.mean(np.abs(mu_orig - targets_orig)))
    pearson = float(pearsonr(mu_orig, targets_orig)[0])
    spearman_pred = float(spearmanr(mu_orig, targets_orig)[0])

    # ── Uncertainty metrics ──
    abs_error = np.abs(mu_orig - targets_orig)
    # Scale uncertainty back to original scale (variance scales by s^2)
    u_orig = u_scaled * (scaler.scale_[0] ** 2)
    mean_u = float(np.mean(u_orig))
    spearman_unc = float(spearmanr(u_orig, abs_error)[0])

    metrics_text = (
        "Phase 2A — Evidential MLP Test Metrics\n"
        "=======================================\n"
        f"RMSE                          : {rmse:.4f}\n"
        f"MAE                           : {mae:.4f}\n"
        f"Pearson  (r)                  : {pearson:.4f}\n"
        f"Spearman (ρ)                  : {spearman_pred:.4f}\n"
        f"\n"
        f"Mean Aleatoric Uncertainty    : {mean_u:.4f}\n"
        f"Spearman(u, |error|)          : {spearman_unc:.4f}\n"
        f"\nModel weights                 : {model_path}\n"
    )
    print("\n" + metrics_text)

    metrics_path = os.path.join(RESULTS_DIR, "phase2a_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(f"Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
