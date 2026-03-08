"""
train_retrain_mlp.py
====================
Phase 3B — Retrain the BaselineMLP on four training-set variations
(100 %, 95 %, 90 %, 85 %) and compare test-set generalization to
demonstrate that removing uncertainty-flagged noise improves performance.

Usage:
    python train_retrain_mlp.py
"""

import os
import copy
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

# Dataset suffixes: "" = original 100 %, then curated variants
SUFFIXES = ["", "_curated_05", "_curated_10", "_curated_15"]
LABELS   = ["100 % (original)", "95 % (−5 %)", "90 % (−10 %)", "85 % (−15 %)"]

# Reproducibility seeds
SEEDS = [0, 1, 2, 3, 4]


# ── Helpers ──────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    """Return (RMSE, MAE, Pearson, Spearman) on original scale."""
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


def run_single_seed(seed, suffix, label, train_emb, train_tgt,
                    val_emb, val_tgt, test_emb, test_tgt):
    """Train one BaselineMLP with a given seed. Returns (rmse, mae, r, rho)."""
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
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss = evaluate_loss(model, val_loader, criterion)
        scheduler.step(v_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"    [seed={seed}] Epoch {epoch:3d}/{MAX_EPOCHS}  |  "
              f"Train: {t_loss:.6f}  |  Val: {v_loss:.6f}  |  "
              f"LR: {lr:.2e}")

        if v_loss < best_val:
            best_val = v_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"    [seed={seed}] ⇢ Early stopping at epoch {epoch}")
            break

    # Save best model for this seed
    tag = suffix if suffix else "_original"
    model_path = os.path.join(RESULTS_DIR, f"mlp{tag}_seed{seed}.pt")
    torch.save(best_state, model_path)

    model.load_state_dict(best_state)
    rmse, mae, r, rho = test_metrics(model, test_loader, scaler)
    print(f"    [seed={seed}] Test → RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"Pearson={r:.4f}  Spearman={rho:.4f}\n")
    return rmse, mae, r, rho


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Fixed validation / test sets (constant across all experiments)
    val_emb,  val_tgt  = load_tensors("val")
    test_emb, test_tgt = load_tensors("test")

    results = []   # list of dicts for the summary table

    for suffix, label in zip(SUFFIXES, LABELS):
        print(f"\n{'═' * 60}")
        print(f"  Experiment: {label}  (suffix='{suffix}')")
        print(f"{'═' * 60}")

        train_emb, train_tgt = load_tensors("train", suffix)
        n_train = train_emb.shape[0]
        print(f"  Train size: {n_train}")
        print(f"  Running {len(SEEDS)} seeds: {SEEDS}\n")

        seed_metrics = []  # list of (rmse, mae, r, rho) per seed
        for seed in SEEDS:
            m = run_single_seed(seed, suffix, label,
                                train_emb, train_tgt,
                                val_emb, val_tgt,
                                test_emb, test_tgt)
            seed_metrics.append(m)

        arr = np.array(seed_metrics)                        # (n_seeds, 4)
        mean_rmse, mean_mae, mean_r, mean_rho = arr.mean(axis=0)
        std_rmse,  std_mae,  std_r,  std_rho  = arr.std(axis=0)

        print(f"  ── Averaged over {len(SEEDS)} seeds ──")
        print(f"  RMSE    = {mean_rmse:.4f} ± {std_rmse:.4f}")
        print(f"  MAE     = {mean_mae:.4f} ± {std_mae:.4f}")
        print(f"  Pearson = {mean_r:.4f} ± {std_r:.4f}")
        print(f"  Spearman= {mean_rho:.4f} ± {std_rho:.4f}")

        results.append({
            "label": label, "n_train": n_train,
            "rmse_mean": mean_rmse, "rmse_std": std_rmse,
            "mae_mean":  mean_mae,  "mae_std":  std_mae,
            "r_mean":    mean_r,    "r_std":    std_r,
            "rho_mean":  mean_rho,  "rho_std":  std_rho,
        })

    # ── Summary report ───────────────────────────────────────────────────────
    seeds_str = ", ".join(str(s) for s in SEEDS)
    header = (
        f"{'Dataset':<22s} {'N':>5s}  {'RMSE':>14s}  {'MAE':>14s}  "
        f"{'Pearson':>14s}  {'Spearman':>14s}"
    )
    sep = "─" * len(header)
    lines = [
        "Phase 3B — Retrain Report (Multi-Seed)",
        "=======================================",
        f"Seeds: [{seeds_str}]  ({len(SEEDS)} runs per dataset)",
        "",
        header, sep,
    ]
    for r in results:
        lines.append(
            f"{r['label']:<22s} {r['n_train']:>5d}  "
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

    report_path = os.path.join(RESULTS_DIR, "phase3b_retrain_report.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
