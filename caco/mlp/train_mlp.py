"""
train_mlp.py
============
Phase 1A — Step 3: Train the BaselineMLP on frozen MolFormer-XL embeddings,
evaluate on the held-out test set, and save metrics to disk.

Usage:
    python train_mlp.py
"""

import copy
import os
import time

import numpy as np
import torch
from model import BaselineMLP, EmbeddingDataset
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
PATIENCE = 10  # early-stopping patience (epochs)
LR_PATIENCE = 5  # ReduceLROnPlateau patience


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_tensors(split: str):
    """Load pre-computed embedding and target tensors for *split*."""
    emb = torch.load(
        os.path.join(DATA_DIR, f"{split}_embeddings.pt"), weights_only=True
    )
    tgt = torch.load(os.path.join(DATA_DIR, f"{split}_targets.pt"), weights_only=True)
    return emb, tgt


def fit_scaler(train_targets: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on training targets only (prevents data leakage)."""
    scaler = StandardScaler()
    scaler.fit(train_targets.reshape(-1, 1))
    return scaler


def scale_targets(targets: np.ndarray, scaler: StandardScaler) -> torch.Tensor:
    """Transform targets using a pre-fitted scaler and return a tensor."""
    scaled = scaler.transform(targets.reshape(-1, 1)).flatten()
    return torch.tensor(scaled, dtype=torch.float32)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Training loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).squeeze(-1)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    return running_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).squeeze(-1)
        loss = criterion(preds, y)
        running_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    return running_loss / n_samples


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

    # 4. Initialise model, optimizer, scheduler, loss
    print("[3/5] Initialising model …")
    model = BaselineMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )
    criterion = nn.MSELoss()

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

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{MAX_EPOCHS}  |  "
            f"Train Loss: {train_loss:.6f}  |  "
            f"Val Loss: {val_loss:.6f}  |  "
            f"LR: {current_lr:.2e}  |  "
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

    # Save best model
    model_path = os.path.join(RESULTS_DIR, "baseline_mlp.pt")
    torch.save(best_state, model_path)
    print(f"  Best model saved → {model_path}")

    # 6. Test evaluation
    print("[5/5] Evaluating on test set …")
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            preds = model(x).squeeze(-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())

    preds_scaled = np.concatenate(all_preds)
    targets_scaled = np.concatenate(all_targets)

    # Inverse-transform to original scale
    preds_orig = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    targets_orig = scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    # Compute metrics
    rmse = float(np.sqrt(np.mean((preds_orig - targets_orig) ** 2)))
    mae = float(np.mean(np.abs(preds_orig - targets_orig)))
    pearson = float(pearsonr(preds_orig, targets_orig)[0])
    spearman = float(spearmanr(preds_orig, targets_orig)[0])

    # Print & save
    metrics_text = (
        "Phase 1A — Baseline MLP Test Metrics\n"
        "=====================================\n"
        f"RMSE              : {rmse:.4f}\n"
        f"MAE               : {mae:.4f}\n"
        f"Pearson  (r)      : {pearson:.4f}\n"
        f"Spearman (ρ)      : {spearman:.4f}\n"
        f"\nModel weights     : {model_path}\n"
    )
    print("\n" + metrics_text)

    metrics_path = os.path.join(RESULTS_DIR, "phase1a_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(f"Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()

