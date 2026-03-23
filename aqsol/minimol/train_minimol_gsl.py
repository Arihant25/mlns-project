"""
train_minimol_gsl.py
====================
MiniMol with our GSL architecture: train SimpleGSLModel on 512-d MiniMol
embeddings.

The only change from the standard GSL pipeline is the embedding dimension:
    SimpleGSLModel(embed_dim=512) instead of the default 768.

Usage:
    python train_minimol_gsl.py
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

# ── Cross-directory imports ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GSL_DIR = os.path.join(PROJECT_ROOT, "gsl")
sys.path.insert(0, GSL_DIR)

from model_gsl import SimpleGSLModel  # noqa: E402
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
        preds = model(X, A_ecfp).squeeze(-1)
        loss = criterion(preds, y)
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
        preds = model(X, A_ecfp).squeeze(-1)
        loss = criterion(preds, y)
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

    print("[3/5] Initialising SimpleGSLModel(embed_dim=512) …")
    model = SimpleGSLModel(embed_dim=EMBED_DIM).to(DEVICE)
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

    model_path = os.path.join(RESULTS_DIR, "minimol_gsl.pt")
    torch.save(best_state, model_path)
    print(f"  Best model saved → {model_path}")

    print("[5/5] Evaluating on test set …")
    model.load_state_dict(best_state)
    model.eval()

    preds_all, tgts_all = [], []
    with torch.no_grad():
        for X, y, A_ecfp in test_loader:
            X, A_ecfp = X.to(DEVICE), A_ecfp.to(DEVICE)
            preds_all.append(model(X, A_ecfp).squeeze(-1).cpu().numpy())
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
        "MiniMol + Our Architecture — GSL Test Metrics (AqSolDB)\n"
        "=========================================================\n"
        "Model: MiniMol (512-d) → SimpleGSLModel(embed_dim=512)\n"
        "\n"
        f"RMSE              : {rmse:.4f}\n"
        f"MAE               : {mae:.4f}\n"
        f"Pearson  (r)      : {pearson:.4f}\n"
        f"Spearman (ρ)      : {spearman:.4f}\n"
        f"\nModel weights     : {model_path}\n"
    )
    print("\n" + metrics_text)

    metrics_path = os.path.join(RESULTS_DIR, "minimol_gsl_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(f"Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
