"""
train_weighted_mlp.py
=====================
Phase 3C — Soft-Weighting Protocol.

Trains the BaselineMLP on 100 % of the training data but down-weights the
per-sample MSE loss using the aleatoric uncertainty from the trained
Evidential GSL model:

    loss = mean( exp(−γ · u) · (ŷ − y)² )

Sweeps over multiple γ values and random seeds, reporting mean ± std.

Usage:
    python train_weighted_mlp.py
"""

import copy
import os
import random
import sys
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ── Cross-directory imports ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GSL_DIR = os.path.join(PROJECT_ROOT, "gsl")
sys.path.insert(0, GSL_DIR)

from model import BaselineMLP  # noqa: E402
from model_evidential_gsl import EvidentialGSLModel  # noqa: E402
from train_gsl import (  # noqa: E402
    GSLDataset,
    gsl_collate_fn,
    load_smiles,
)

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "evidential_gsl.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
LR_PATIENCE = 5

GAMMAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
SEEDS = [42, 123, 456, 789, 1024]


# ── Helpers ──────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_tensors(split: str):
    emb = torch.load(
        os.path.join(DATA_DIR, f"{split}_embeddings.pt"), weights_only=True
    )
    tgt = torch.load(os.path.join(DATA_DIR, f"{split}_targets.pt"), weights_only=True)
    return emb, tgt


def fit_scaler(targets: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(targets.reshape(-1, 1))
    return scaler


def scale(targets: np.ndarray, scaler: StandardScaler) -> torch.Tensor:
    return torch.tensor(
        scaler.transform(targets.reshape(-1, 1)).flatten(),
        dtype=torch.float32,
    )


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Weighted Dataset ─────────────────────────────────────────────────────────
class WeightedDataset(Dataset):
    """Returns (embedding, scaled_target, uncertainty_value)."""

    def __init__(self, embeddings, targets, u_values):
        self.embeddings = embeddings
        self.targets = targets
        self.u_values = u_values

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx], self.u_values[idx]


# ── Extract training-set uncertainties ───────────────────────────────────────
def extract_uncertainties(train_emb, train_tgt_s, train_smiles):
    """
    Run full-batch inference through the trained Evidential GSL model and
    return the per-molecule aleatoric uncertainty u = β / (α − 1).
    """
    print("  Loading EvidentialGSLModel for uncertainty extraction …")
    model = EvidentialGSLModel()
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    n = train_emb.shape[0]
    loader = DataLoader(
        GSLDataset(train_emb, train_tgt_s, train_smiles),
        batch_size=n,
        shuffle=False,
        collate_fn=gsl_collate_fn,
    )

    with torch.no_grad():
        for X, _, A_ecfp in loader:
            X, A_ecfp = X.to(DEVICE), A_ecfp.to(DEVICE)
            _, (mu, v, alpha, beta) = model(X, A_ecfp)

    u = (beta / (alpha - 1.0)).detach().cpu()
    print(f"  Uncertainty: min={u.min():.4f}, max={u.max():.4f}, mean={u.mean():.4f}")
    return u


# ── Train / Eval ─────────────────────────────────────────────────────────────
def train_one_epoch_weighted(model, loader, optimizer, gamma):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, u in loader:
        x, y, u = x.to(DEVICE), y.to(DEVICE), u.to(DEVICE)
        preds = model(x).squeeze(-1)
        errors = (preds - y) ** 2
        weights = torch.exp(-gamma * u)
        loss = torch.mean(weights * errors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate_mse(model, loader):
    """Standard (unweighted) MSE for validation early stopping."""
    model.eval()
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).squeeze(-1)
        loss = torch.mean((preds - y) ** 2)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def test_metrics(model, loader, scaler):
    model.eval()
    preds_all, tgts_all = [], []
    for x, y, _ in loader:
        x = x.to(DEVICE)
        preds_all.append(model(x).squeeze(-1).cpu().numpy())
        tgts_all.append(y.numpy())

    p = scaler.inverse_transform(np.concatenate(preds_all).reshape(-1, 1)).flatten()
    t = scaler.inverse_transform(np.concatenate(tgts_all).reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(np.abs(p - t)))
    r = float(pearsonr(p, t)[0])
    rho = float(spearmanr(p, t)[0])
    return rmse, mae, r, rho


def run_single_seed(seed, gamma, train_ds, val_ds, test_ds, scaler):
    set_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = BaselineMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )

    best_val, best_state, no_improve = float("inf"), None, 0
    epoch_times = []
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()
        train_one_epoch_weighted(model, train_loader, optimizer, gamma)
        v_loss = evaluate_mse(model, val_loader)
        scheduler.step(v_loss)

        epoch_times.append(time.perf_counter() - epoch_start)
        avg_epoch_s = sum(epoch_times) / len(epoch_times)
        eta_s = avg_epoch_s * (MAX_EPOCHS - epoch)
        print(
            f"      [seed={seed}] epoch {epoch:3d}/{MAX_EPOCHS}  "
            f"val={v_loss:.6f}  ETA: {format_eta(eta_s)}"
        )

        if v_loss < best_val:
            best_val = v_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return test_metrics(model, test_loader, scaler)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load data
    print("[1/3] Loading data …")
    train_emb, train_tgt = load_tensors("train")
    val_emb, val_tgt = load_tensors("val")
    test_emb, test_tgt = load_tensors("test")
    smiles = load_smiles()

    scaler = fit_scaler(train_tgt.numpy())
    train_tgt_s = scale(train_tgt.numpy(), scaler)
    val_tgt_s = scale(val_tgt.numpy(), scaler)
    test_tgt_s = scale(test_tgt.numpy(), scaler)

    # 2. Extract per-molecule uncertainty via Evidential GSL
    print("[2/3] Extracting training-set uncertainties …")
    u_train = extract_uncertainties(train_emb, train_tgt_s, smiles["train"])
    u_dummy_val = torch.zeros(val_emb.shape[0])
    u_dummy_test = torch.zeros(test_emb.shape[0])

    train_ds = WeightedDataset(train_emb, train_tgt_s, u_train)
    val_ds = WeightedDataset(val_emb, val_tgt_s, u_dummy_val)
    test_ds = WeightedDataset(test_emb, test_tgt_s, u_dummy_test)

    # 3. Sweep gammas × seeds
    print(f"[3/3] Training sweep: {len(GAMMAS)} γ × {len(SEEDS)} seeds …\n")
    results = []
    total_runs = len(GAMMAS) * len(SEEDS)
    completed_runs = 0
    sweep_start = time.perf_counter()
    for gamma in GAMMAS:
        print(f"  ── γ = {gamma} ──")
        seed_metrics = []
        for seed in SEEDS:
            m = run_single_seed(seed, gamma, train_ds, val_ds, test_ds, scaler)
            print(
                f"    seed={seed:>4d}  RMSE={m[0]:.4f}  MAE={m[1]:.4f}  "
                f"r={m[2]:.4f}  ρ={m[3]:.4f}"
            )
            seed_metrics.append(m)
            completed_runs += 1
            avg_run_s = (time.perf_counter() - sweep_start) / completed_runs
            eta_s = avg_run_s * (total_runs - completed_runs)
            print(f"      [sweep ETA] remaining: {format_eta(eta_s)}")

        arr = np.array(seed_metrics)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        print(
            f"    → mean  RMSE={means[0]:.4f}±{stds[0]:.4f}  "
            f"MAE={means[1]:.4f}±{stds[1]:.4f}  "
            f"r={means[2]:.4f}±{stds[2]:.4f}  "
            f"ρ={means[3]:.4f}±{stds[3]:.4f}\n"
        )

        results.append(
            {
                "gamma": gamma,
                "rmse_mean": means[0],
                "rmse_std": stds[0],
                "mae_mean": means[1],
                "mae_std": stds[1],
                "r_mean": means[2],
                "r_std": stds[2],
                "rho_mean": means[3],
                "rho_std": stds[3],
            }
        )

    # ── Summary report ───────────────────────────────────────────────────────
    seeds_str = ", ".join(str(s) for s in SEEDS)
    header = (
        f"{'γ':>5s}  {'RMSE':>14s}  {'MAE':>14s}  {'Pearson':>14s}  {'Spearman':>14s}"
    )
    sep = "─" * len(header)
    lines = [
        "Phase 3C — Soft-Weighting Report (Multi-Seed)",
        "===============================================",
        f"Seeds: [{seeds_str}]  ({len(SEEDS)} runs per γ)",
        "Training set: 637 molecules (full, no removal)",
        "",
        header,
        sep,
    ]
    for r in results:
        lines.append(
            f"{r['gamma']:>5.1f}  "
            f"{r['rmse_mean']:.4f}±{r['rmse_std']:.4f}  "
            f"{r['mae_mean']:.4f}±{r['mae_std']:.4f}  "
            f"{r['r_mean']:.4f}±{r['r_std']:.4f}  "
            f"{r['rho_mean']:.4f}±{r['rho_std']:.4f}"
        )
    lines += ["", sep]

    best = min(results, key=lambda x: x["mae_mean"])
    lines.append(
        f"Best mean MAE at γ={best['gamma']:.1f}: "
        f"{best['mae_mean']:.4f} ± {best['mae_std']:.4f}"
    )

    report = "\n".join(lines)
    print(f"\n{report}")

    report_path = os.path.join(RESULTS_DIR, "phase3c_soft_weighting_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()

