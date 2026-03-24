"""
train_minimol_evidential_gsl.py
================================
Train EvidentialGSLModel on 512-d MiniMol embeddings to obtain per-sample
aleatoric uncertainty estimates used by the soft-weighting pipeline.

Uses the TDC ADMET benchmark group protocol: seed=1 train/valid split for
training, fixed benchmark test set for evaluation.

Saves results/evidential_gsl_minimol.pt — consumed by:
    - train_minimol_weighted_mlp.py
    - chemprop/export_sample_weights.py

Usage:
    python train_minimol_evidential_gsl.py

Prerequisites:
    python generate_embeddings_minimol.py
"""

import copy
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tdc.benchmark_group import admet_group

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GSL_DIR = os.path.join(PROJECT_ROOT, "gsl")
sys.path.insert(0, GSL_DIR)

from gsl_utils import GSLDataset, gsl_collate_fn  # noqa: E402
from loss_evidential import ErrorScaledEvidentialLoss  # noqa: E402
from model_evidential_gsl import EvidentialGSLModel  # noqa: E402

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
EMBED_DIM = 512

# Uncertainty model trained on seed=1 split (internal; not directly benchmarked)
UNCERTAINTY_SEED = 1


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total, n = 0.0, 0
    for X, y, A in loader:
        X, y, A = X.to(DEVICE), y.to(DEVICE), A.to(DEVICE)
        (mu0, v0, a0, b0), (mu, v, a, b) = model(X, A)
        loss = criterion(y, mu, v, a, b) + AUX_WEIGHT * criterion(y, mu0, v0, a0, b0)
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
    for X, y, A in loader:
        X, y, A = X.to(DEVICE), y.to(DEVICE), A.to(DEVICE)
        (mu0, v0, a0, b0), (mu, v, a, b) = model(X, A)
        loss = criterion(y, mu, v, a, b) + AUX_WEIGHT * criterion(y, mu0, v0, a0, b0)
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / n


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[1/5] Loading TDC ADMET benchmark group ...")
    group = admet_group(path=DATA_DIR)
    benchmark = group.get("Solubility_AqSolDB")
    name = benchmark["name"]

    print(f"[2/5] Loading MiniMol embeddings and getting seed={UNCERTAINTY_SEED} split ...")
    all_tv_emb = torch.load(
        os.path.join(DATA_DIR, "trainval_embeddings_minimol.pt"), weights_only=True
    )
    all_tv_tgt = torch.load(
        os.path.join(DATA_DIR, "trainval_targets.pt"), weights_only=True
    )
    all_tv_smi = torch.load(os.path.join(DATA_DIR, "trainval_smiles.pt"))
    test_emb = torch.load(
        os.path.join(DATA_DIR, "test_embeddings_minimol.pt"), weights_only=True
    )
    test_tgt = torch.load(
        os.path.join(DATA_DIR, "test_targets.pt"), weights_only=True
    )

    smi_to_idx = {s: i for i, s in enumerate(all_tv_smi)}

    train_df, val_df = group.get_train_valid_split(
        benchmark=name, split_type="default", seed=UNCERTAINTY_SEED
    )
    t_idx = [smi_to_idx[s] for s in train_df["Drug"].tolist()]
    v_idx = [smi_to_idx[s] for s in val_df["Drug"].tolist()]

    train_emb = all_tv_emb[t_idx]
    val_emb = all_tv_emb[v_idx]
    train_smi = train_df["Drug"].tolist()
    val_smi = val_df["Drug"].tolist()
    test_smi = torch.load(os.path.join(DATA_DIR, "test_smiles.pt"))

    scaler = StandardScaler()
    scaler.fit(all_tv_tgt.numpy().reshape(-1, 1))

    def scale(t):
        return torch.tensor(
            scaler.transform(t.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32
        )

    train_tgt_s = scale(all_tv_tgt[t_idx])
    val_tgt_s = scale(all_tv_tgt[v_idx])
    test_tgt_s = scale(test_tgt)

    print("[3/5] Building data loaders ...")
    train_loader = DataLoader(
        GSLDataset(train_emb, train_tgt_s, train_smi),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=gsl_collate_fn,
    )
    val_loader = DataLoader(
        GSLDataset(val_emb, val_tgt_s, val_smi),
        batch_size=BATCH_SIZE, collate_fn=gsl_collate_fn,
    )
    test_loader = DataLoader(
        GSLDataset(test_emb, test_tgt_s, test_smi),
        batch_size=BATCH_SIZE, collate_fn=gsl_collate_fn,
    )

    print(f"[4/5] Initialising EvidentialGSLModel(embed_dim={EMBED_DIM}) ...")
    model = EvidentialGSLModel(embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )
    criterion = ErrorScaledEvidentialLoss(coeff=LOSS_COEFF)
    print(f"       Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("[5/5] Training ...")
    best_val, best_state, no_improve = float("inf"), None, 0
    epoch_times = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        epoch_times.append(time.perf_counter() - t0)
        eta_s = (sum(epoch_times) / len(epoch_times)) * (MAX_EPOCHS - epoch)
        print(
            f"  Epoch {epoch:3d}/{MAX_EPOCHS}  |  "
            f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}  ETA: {format_eta(eta_s)}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    model_path = os.path.join(RESULTS_DIR, "evidential_gsl_minimol.pt")
    torch.save(best_state, model_path)
    print(f"  Saved -> {model_path}")

    print("\nEvaluating on test set ...")
    model.load_state_dict(best_state)
    model.eval()

    all_mu, all_u, all_tgt = [], [], []
    with torch.no_grad():
        for X, y, A in test_loader:
            X, A = X.to(DEVICE), A.to(DEVICE)
            _, (mu, v, alpha, beta) = model(X, A)
            u = beta / (alpha - 1.0)
            all_mu.append(mu.cpu().numpy())
            all_u.append(u.cpu().numpy())
            all_tgt.append(y.numpy())

    mu_s = np.concatenate(all_mu)
    u_s = np.concatenate(all_u)
    t_s = np.concatenate(all_tgt)
    mu_orig = scaler.inverse_transform(mu_s.reshape(-1, 1)).flatten()
    t_orig = scaler.inverse_transform(t_s.reshape(-1, 1)).flatten()
    u_orig = u_s * (scaler.scale_[0] ** 2)

    rmse = float(np.sqrt(np.mean((mu_orig - t_orig) ** 2)))
    mae = float(np.mean(np.abs(mu_orig - t_orig)))
    spearman_unc = float(spearmanr(u_orig, np.abs(mu_orig - t_orig))[0])

    metrics_text = (
        "MiniMol Evidential GSL -- Uncertainty Model (AqSolDB)\n"
        "========================================================\n"
        f"Model: MiniMol (512-d) -> EvidentialGSLModel(embed_dim={EMBED_DIM})\n"
        f"[Internal uncertainty source for soft weighting, trained on seed={UNCERTAINTY_SEED} split]\n"
        "\n"
        f"RMSE                    : {rmse:.4f}\n"
        f"MAE                     : {mae:.4f}\n"
        f"Pearson  (r)            : {float(pearsonr(mu_orig, t_orig)[0]):.4f}\n"
        f"Spearman (rho)          : {float(spearmanr(mu_orig, t_orig)[0]):.4f}\n"
        f"Mean Aleatoric Unc.     : {float(np.mean(u_orig)):.4f}\n"
        f"Spearman(u, |error|)    : {spearman_unc:.4f}\n"
        f"\nModel weights           : {model_path}\n"
    )
    print("\n" + metrics_text)

    with open(os.path.join(RESULTS_DIR, "evidential_gsl_minimol_metrics.txt"), "w") as f:
        f.write(metrics_text)


if __name__ == "__main__":
    main()
