"""
train_evidential_gsl.py
=======================
Phase 2B — Train the EvidentialGSLModel with a dual-loss objective:
    Loss = loss_final + 0.5 * loss_initial

Uses the same ECFP collation, StandardScaler, and early-stopping logic as
Phase 1B, combined with the ErrorScaledEvidentialLoss from Phase 2A.

Usage:
    python train_evidential_gsl.py
"""

import copy
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

# ── Cross-directory import: ErrorScaledEvidentialLoss lives in mlp/ ──────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MLP_DIR = os.path.join(PROJECT_ROOT, "mlp")
sys.path.insert(0, MLP_DIR)
from loss_evidential import ErrorScaledEvidentialLoss  # noqa: E402

# ── Same-directory imports ───────────────────────────────────────────────────
from model_evidential_gsl import EvidentialGSLModel  # noqa: E402
from train_gsl import (  # noqa: E402
    GSLDataset,
    fit_scaler,
    gsl_collate_fn,
    load_smiles,
    load_tensors,
    scale_targets,
)

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
LR_PATIENCE = 5
LOSS_COEFF = 0.1  # error-scaled KL coefficient
AUX_WEIGHT = 0.5  # weight on the initial-head auxiliary loss


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
    for X, y, A_ecfp in loader:
        X, y, A_ecfp = X.to(DEVICE), y.to(DEVICE), A_ecfp.to(DEVICE)

        (mu_0, v_0, a_0, b_0), (mu, v, a, b) = model(X, A_ecfp)

        loss_initial = criterion(y, mu_0, v_0, a_0, b_0)
        loss_final = criterion(y, mu, v, a, b)
        loss = loss_final + AUX_WEIGHT * loss_initial

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        n += X.size(0)
    return running_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    n = 0
    for X, y, A_ecfp in loader:
        X, y, A_ecfp = X.to(DEVICE), y.to(DEVICE), A_ecfp.to(DEVICE)

        (mu_0, v_0, a_0, b_0), (mu, v, a, b) = model(X, A_ecfp)

        loss_initial = criterion(y, mu_0, v_0, a_0, b_0)
        loss_final = criterion(y, mu, v, a, b)
        loss = loss_final + AUX_WEIGHT * loss_initial

        running_loss += loss.item() * X.size(0)
        n += X.size(0)
    return running_loss / n


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load embeddings, targets, and SMILES
    print("[1/5] Loading pre-computed embeddings and SMILES …")
    train_emb, train_tgt = load_tensors("train")
    val_emb, val_tgt = load_tensors("val")
    test_emb, test_tgt = load_tensors("test")
    smiles = load_smiles()
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

    # 3. Build DataLoaders with ECFP collate
    train_loader = DataLoader(
        GSLDataset(train_emb, train_tgt_s, smiles["train"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=gsl_collate_fn,
    )
    val_loader = DataLoader(
        GSLDataset(val_emb, val_tgt_s, smiles["val"]),
        batch_size=BATCH_SIZE,
        collate_fn=gsl_collate_fn,
    )
    test_loader = DataLoader(
        GSLDataset(test_emb, test_tgt_s, smiles["test"]),
        batch_size=BATCH_SIZE,
        collate_fn=gsl_collate_fn,
    )

    # 4. Initialise model, optimiser, scheduler, loss
    print("[3/5] Initialising EvidentialGSLModel …")
    model = EvidentialGSLModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )
    criterion = ErrorScaledEvidentialLoss(coeff=LOSS_COEFF)
    print(f"       Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"       Loss coeff: {LOSS_COEFF}, Aux weight: {AUX_WEIGHT}")

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

    model_path = os.path.join(RESULTS_DIR, "evidential_gsl.pt")
    torch.save(best_state, model_path)
    print(f"  Best model saved → {model_path}")

    # 6. Test evaluation (use FINAL head predictions)
    print("[5/5] Evaluating on test set …")
    model.load_state_dict(best_state)
    model.eval()

    all_mu, all_u, all_targets = [], [], []
    with torch.no_grad():
        for X, y, A_ecfp in test_loader:
            X, A_ecfp = X.to(DEVICE), A_ecfp.to(DEVICE)
            _, (mu, v, alpha, beta) = model(X, A_ecfp)

            # Final aleatoric uncertainty: u = β / (α − 1)
            u = beta / (alpha - 1.0)

            all_mu.append(mu.cpu().numpy())
            all_u.append(u.cpu().numpy())
            all_targets.append(y.numpy())

    mu_scaled = np.concatenate(all_mu)
    u_scaled = np.concatenate(all_u)
    targets_scaled = np.concatenate(all_targets)

    # Inverse-transform to original scale
    mu_orig = scaler.inverse_transform(mu_scaled.reshape(-1, 1)).flatten()
    targets_orig = scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    # ── Prediction metrics ──
    rmse = float(np.sqrt(np.mean((mu_orig - targets_orig) ** 2)))
    mae = float(np.mean(np.abs(mu_orig - targets_orig)))
    pearson = float(pearsonr(mu_orig, targets_orig)[0])
    spearman_pred = float(spearmanr(mu_orig, targets_orig)[0])

    # ── Uncertainty metrics ──
    abs_error = np.abs(mu_orig - targets_orig)
    u_orig = u_scaled * (scaler.scale_[0] ** 2)
    mean_u = float(np.mean(u_orig))
    spearman_unc = float(spearmanr(u_orig, abs_error)[0])

    metrics_text = (
        "Phase 2B — Evidential GSL Test Metrics\n"
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

    metrics_path = os.path.join(RESULTS_DIR, "phase2b_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(f"Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
