"""
golden_mlp.py
=============
Train the BaselineMLP on both the full (100%) and Golden-curated training
sets across 10 random seeds, reporting mean ± std test metrics.

Usage:
    python golden_mlp.py
"""

import os
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

from model import BaselineMLP, EmbeddingDataset


# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE   = 64
LR           = 1e-3
MAX_EPOCHS   = 100
PATIENCE     = 10
LR_PATIENCE  = 5

SEEDS = list(range(10))

VARIANTS = [
    ("", "100 % (original)"),
    ("_golden", "Golden (GCI-curated)"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_tensors(split: str, suffix: str = ""):
    emb = torch.load(os.path.join(DATA_DIR, f"{split}_embeddings{suffix}.pt"),
                     weights_only=True)
    tgt = torch.load(os.path.join(DATA_DIR, f"{split}_targets{suffix}.pt"),
                     weights_only=True)
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


# ── Train / Eval ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = criterion(model(x).squeeze(-1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def evaluate_loss(model, loader, criterion):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = criterion(model(x).squeeze(-1), y)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def test_metrics(model, loader, scaler):
    model.eval()
    preds_all, tgts_all = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        preds_all.append(model(x).squeeze(-1).cpu().numpy())
        tgts_all.append(y.numpy())

    p = scaler.inverse_transform(np.concatenate(preds_all).reshape(-1, 1)).flatten()
    t = scaler.inverse_transform(np.concatenate(tgts_all).reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae  = float(np.mean(np.abs(p - t)))
    r    = float(pearsonr(p, t)[0])
    rho  = float(spearmanr(p, t)[0])
    return rmse, mae, r, rho


def run_single_seed(seed, train_emb, train_tgt, val_emb, val_tgt,
                    test_emb, test_tgt):
    set_seed(seed)

    scaler = fit_scaler(train_tgt.numpy())
    train_tgt_s = scale(train_tgt.numpy(), scaler)
    val_tgt_s   = scale(val_tgt.numpy(),   scaler)
    test_tgt_s  = scale(test_tgt.numpy(),  scaler)

    train_loader = DataLoader(EmbeddingDataset(train_emb, train_tgt_s),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(EmbeddingDataset(val_emb, val_tgt_s),
                              batch_size=BATCH_SIZE)
    test_loader  = DataLoader(EmbeddingDataset(test_emb, test_tgt_s),
                              batch_size=BATCH_SIZE)

    model = BaselineMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )
    criterion = nn.MSELoss()

    best_val, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss = evaluate_loss(model, val_loader, criterion)
        scheduler.step(v_loss)

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

    val_emb,  val_tgt  = load_tensors("val")
    test_emb, test_tgt = load_tensors("test")

    results = []

    for suffix, label in VARIANTS:
        print(f"\n{'═' * 60}")
        print(f"  {label}  (suffix='{suffix}')")
        print(f"{'═' * 60}")

        train_emb, train_tgt = load_tensors("train", suffix)
        n_train = train_emb.shape[0]
        print(f"  Train size: {n_train}")
        print(f"  Running {len(SEEDS)} seeds: {SEEDS}\n")

        seed_metrics = []
        for seed in SEEDS:
            m = run_single_seed(seed, train_emb, train_tgt,
                                val_emb, val_tgt, test_emb, test_tgt)
            print(f"    seed={seed:>2d}  RMSE={m[0]:.4f}  MAE={m[1]:.4f}  "
                  f"r={m[2]:.4f}  ρ={m[3]:.4f}")
            seed_metrics.append(m)

        arr = np.array(seed_metrics)
        means = arr.mean(axis=0)
        stds  = arr.std(axis=0)

        print(f"\n  ── Averaged over {len(SEEDS)} seeds ──")
        print(f"  RMSE    = {means[0]:.4f} ± {stds[0]:.4f}")
        print(f"  MAE     = {means[1]:.4f} ± {stds[1]:.4f}")
        print(f"  Pearson = {means[2]:.4f} ± {stds[2]:.4f}")
        print(f"  Spearman= {means[3]:.4f} ± {stds[3]:.4f}")

        results.append({
            "label": label, "n_train": n_train,
            "rmse_mean": means[0], "rmse_std": stds[0],
            "mae_mean":  means[1], "mae_std":  stds[1],
            "r_mean":    means[2], "r_std":    stds[2],
            "rho_mean":  means[3], "rho_std":  stds[3],
        })

    # ── Summary report ───────────────────────────────────────────────────────
    seeds_str = ", ".join(str(s) for s in SEEDS)
    header = (
        f"{'Dataset':<24s} {'N':>5s}  {'RMSE':>14s}  {'MAE':>14s}  "
        f"{'Pearson':>14s}  {'Spearman':>14s}"
    )
    sep = "─" * len(header)
    lines = [
        "Golden Dataset — MLP Comparison (Multi-Seed)",
        "=============================================",
        f"Seeds: [{seeds_str}]  ({len(SEEDS)} runs per dataset)",
        "",
        header, sep,
    ]
    for r in results:
        lines.append(
            f"{r['label']:<24s} {r['n_train']:>5d}  "
            f"{r['rmse_mean']:.4f}±{r['rmse_std']:.4f}  "
            f"{r['mae_mean']:.4f}±{r['mae_std']:.4f}  "
            f"{r['r_mean']:.4f}±{r['r_std']:.4f}  "
            f"{r['rho_mean']:.4f}±{r['rho_std']:.4f}"
        )
    lines += ["", sep]

    best = min(results, key=lambda x: x["mae_mean"])
    lines.append(
        f"Best mean MAE: {best['label']} "
        f"({best['mae_mean']:.4f} ± {best['mae_std']:.4f})"
    )

    report = "\n".join(lines)
    print(f"\n{report}")

    report_path = os.path.join(RESULTS_DIR, "golden_mlp_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()

