"""
train_minimol_evidential_gsl.py
================================
MiniMol with our evidential GSL architecture: train EvidentialGSLModel on
512-d MiniMol embeddings.

Produces evidential uncertainty estimates per molecule, saved to
results/evidential_gsl_minimol.pt (separate from the MolFormer-based
evidential_gsl.pt used by the Chemprop pipeline).

Usage:
    python train_minimol_evidential_gsl.py
"""

import copy
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

# ── Cross-directory imports ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GSL_DIR = os.path.join(PROJECT_ROOT, "gsl")
MLP_DIR = os.path.join(PROJECT_ROOT, "mlp")
sys.path.insert(0, GSL_DIR)
sys.path.insert(0, MLP_DIR)

from loss_evidential import ErrorScaledEvidentialLoss  # noqa: E402
from model_evidential_gsl import EvidentialGSLModel  # noqa: E402
from train_gsl import GSLDataset, gsl_collate_fn  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
LR_PATIENCE = 5
LOSS_COEFF = 0.1
AUX_WEIGHT = 0.5
EMBED_DIM = 512  # MiniMol output dimension


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_tensors(split: str):
    emb = torch.load(
        os.path.join(DATA_DIR, f"{split}_embeddings_minimol.pt"), weights_only=True
    )
    tgt = torch.load(os.path.join(DATA_DIR, f"{split}_targets.pt"), weights_only=True)
    return emb, tgt


def load_smiles():
    return {
        s: torch.load(os.path.join(DATA_DIR, f"{s}_smiles.pt"))
        for s in ("train", "val", "test")
    }


def fit_scaler(targets):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(targets.reshape(-1, 1))
    return scaler


def scale(targets, scaler):
    return torch.tensor(
        scaler.transform(targets.reshape(-1, 1)).flatten(), dtype=torch.float32
    )


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Training / Evaluation ────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total, n = 0.0, 0
    for X, y, A_ecfp in loader:
        X, y, A_ecfp = X.to(DEVICE), y.to(DEVICE), A_ecfp.to(DEVICE)
        (mu_0, v_0, a_0, b_0), (mu, v, a, b) = model(X, A_ecfp)
        loss = criterion(y, mu, v, a, b) + AUX_WEIGHT * criterion(y, mu_0, v_0, a_0, b_0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, n = 0.0, 0
    for X, y, A_ecfp in loader:
        X, y, A_ecfp = X.to(DEVICE), y.to(DEVICE), A_ecfp.to(DEVICE)
        (mu_0, v_0, a_0, b_0), (mu, v, a, b) = model(X, A_ecfp)
        loss = criterion(y, mu, v, a, b) + AUX_WEIGHT * criterion(y, mu_0, v_0, a_0, b_0)
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / n


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[1/5] Loading MiniMol embeddings and SMILES …")
    train_emb, train_tgt = load_tensors("train")
    val_emb, val_tgt = load_tensors("val")
    test_emb, test_tgt = load_tensors("test")
    smiles = load_smiles()
    print(
        f"       Train: {train_emb.shape[0]}, Val: {val_emb.shape[0]}, "
        f"Test: {test_emb.shape[0]}  |  embed_dim={train_emb.shape[1]}"
    )

    print("[2/5] Fitting StandardScaler …")
    scaler = fit_scaler(train_tgt.numpy())
    train_tgt_s = scale(train_tgt.numpy(), scaler)
    val_tgt_s = scale(val_tgt.numpy(), scaler)
    test_tgt_s = scale(test_tgt.numpy(), scaler)

    train_loader = DataLoader(
        GSLDataset(train_emb, train_tgt_s, smiles["train"]),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=gsl_collate_fn,
    )
    val_loader = DataLoader(
        GSLDataset(val_emb, val_tgt_s, smiles["val"]),
        batch_size=BATCH_SIZE, collate_fn=gsl_collate_fn,
    )
    test_loader = DataLoader(
        GSLDataset(test_emb, test_tgt_s, smiles["test"]),
        batch_size=BATCH_SIZE, collate_fn=gsl_collate_fn,
    )

    print("[3/5] Initialising EvidentialGSLModel(embed_dim=512) …")
    model = EvidentialGSLModel(embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )
    criterion = ErrorScaledEvidentialLoss(coeff=LOSS_COEFF)
    print(f"       Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("[4/5] Training …")
    best_val, best_state, no_improve = float("inf"), None, 0
    epoch_times = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        epoch_times.append(time.perf_counter() - t0)
        eta_s = (sum(epoch_times) / len(epoch_times)) * (MAX_EPOCHS - epoch)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{MAX_EPOCHS}  |  "
            f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  "
            f"LR: {lr:.2e}  ETA: {format_eta(eta_s)}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"  ⇢ Early stopping at epoch {epoch}")
            break

    # Save as minimol-specific model to avoid overwriting the 768-d MolFormer version
    model_path = os.path.join(RESULTS_DIR, "evidential_gsl_minimol.pt")
    torch.save(best_state, model_path)
    print(f"  Best model saved → {model_path}")

    print("[5/5] Evaluating on test set …")
    model.load_state_dict(best_state)
    model.eval()

    all_mu, all_u, all_tgt = [], [], []
    with torch.no_grad():
        for X, y, A_ecfp in test_loader:
            X, A_ecfp = X.to(DEVICE), A_ecfp.to(DEVICE)
            _, (mu, v, alpha, beta) = model(X, A_ecfp)
            u = beta / (alpha - 1.0)
            all_mu.append(mu.cpu().numpy())
            all_u.append(u.cpu().numpy())
            all_tgt.append(y.numpy())

    mu_s = np.concatenate(all_mu)
    u_s = np.concatenate(all_u)
    t_s = np.concatenate(all_tgt)

    mu_orig = scaler.inverse_transform(mu_s.reshape(-1, 1)).flatten()
    t_orig = scaler.inverse_transform(t_s.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(np.mean((mu_orig - t_orig) ** 2)))
    mae = float(np.mean(np.abs(mu_orig - t_orig)))
    pearson = float(pearsonr(mu_orig, t_orig)[0])
    spearman_pred = float(spearmanr(mu_orig, t_orig)[0])

    u_orig = u_s * (scaler.scale_[0] ** 2)
    mean_u = float(np.mean(u_orig))
    spearman_unc = float(spearmanr(u_orig, np.abs(mu_orig - t_orig))[0])

    metrics_text = (
        "MiniMol + Our Architecture — Evidential GSL Test Metrics (AqSolDB)\n"
        "=====================================================================\n"
        "Model: MiniMol (512-d) → EvidentialGSLModel(embed_dim=512)\n"
        "\n"
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

    metrics_path = os.path.join(RESULTS_DIR, "minimol_evidential_gsl_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(f"Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
