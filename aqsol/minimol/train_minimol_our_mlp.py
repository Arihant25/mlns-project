"""
train_minimol_our_mlp.py
========================
MiniMol with our architecture: train BaselineMLP on 512-d MiniMol embeddings.

Uses our standard BaselineMLP (768→512→256→1 adapted to 512→512→256→1)
with the same training loop, early stopping, and evaluation as the MolFormer
pipeline. This isolates the effect of MiniMol vs MolFormer embeddings under
a consistent architecture.

Usage:
    python train_minimol_our_mlp.py
"""

import copy
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

# ── Cross-directory import ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MLP_DIR = os.path.join(PROJECT_ROOT, "mlp")
sys.path.insert(0, MLP_DIR)

from model import BaselineMLP, EmbeddingDataset  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
LR_PATIENCE = 5
EMBED_DIM = 512  # MiniMol output dimension


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_tensors(split: str):
    emb = torch.load(
        os.path.join(DATA_DIR, f"{split}_embeddings_minimol.pt"), weights_only=True
    )
    tgt = torch.load(os.path.join(DATA_DIR, f"{split}_targets.pt"), weights_only=True)
    return emb, tgt


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Training / Evaluation ────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).squeeze(-1)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).squeeze(-1)
        loss = criterion(preds, y)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[1/5] Loading MiniMol embeddings …")
    train_emb, train_tgt = load_tensors("train")
    val_emb, val_tgt = load_tensors("val")
    test_emb, test_tgt = load_tensors("test")
    print(
        f"       Train: {train_emb.shape[0]}, Val: {val_emb.shape[0]}, "
        f"Test: {test_emb.shape[0]}  |  embed_dim={train_emb.shape[1]}"
    )

    print("[2/5] Fitting StandardScaler …")
    scaler = StandardScaler()
    scaler.fit(train_tgt.numpy().reshape(-1, 1))

    def scale(t):
        return torch.tensor(
            scaler.transform(t.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32
        )

    train_tgt_s = scale(train_tgt)
    val_tgt_s = scale(val_tgt)
    test_tgt_s = scale(test_tgt)

    train_loader = DataLoader(
        EmbeddingDataset(train_emb, train_tgt_s), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(EmbeddingDataset(val_emb, val_tgt_s), batch_size=BATCH_SIZE)
    test_loader = DataLoader(
        EmbeddingDataset(test_emb, test_tgt_s), batch_size=BATCH_SIZE
    )

    print("[3/5] Initialising BaselineMLP(input_dim=512) …")
    model = BaselineMLP(input_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )
    criterion = nn.MSELoss()
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

    model_path = os.path.join(RESULTS_DIR, "minimol_our_mlp.pt")
    torch.save(best_state, model_path)
    print(f"  Best model saved → {model_path}")

    print("[5/5] Evaluating on test set …")
    model.load_state_dict(best_state)
    model.eval()

    preds_all, tgts_all = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds_all.append(model(x.to(DEVICE)).squeeze(-1).cpu().numpy())
            tgts_all.append(y.numpy())

    p_s = np.concatenate(preds_all)
    t_s = np.concatenate(tgts_all)
    p = scaler.inverse_transform(p_s.reshape(-1, 1)).flatten()
    t = scaler.inverse_transform(t_s.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(np.abs(p - t)))
    pearson = float(pearsonr(p, t)[0])
    spearman = float(spearmanr(p, t)[0])

    metrics_text = (
        "MiniMol + Our Architecture — BaselineMLP Test Metrics (AqSolDB)\n"
        "=================================================================\n"
        "Model: MiniMol (512-d) → BaselineMLP(512→512→256→1)\n"
        "\n"
        f"RMSE              : {rmse:.4f}\n"
        f"MAE               : {mae:.4f}\n"
        f"Pearson  (r)      : {pearson:.4f}\n"
        f"Spearman (ρ)      : {spearman:.4f}\n"
        f"\nModel weights     : {model_path}\n"
    )
    print("\n" + metrics_text)

    metrics_path = os.path.join(RESULTS_DIR, "minimol_our_mlp_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(f"Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
