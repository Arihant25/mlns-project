"""
train_xgboost_molformer.py
==========================
Evaluate an XGBoost regressor on 768-d MolFormer embeddings for AqSolDB
across 10 random seeds.

Usage:
    python train_xgboost_molformer.py
"""

import os
import time

import numpy as np
import torch
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

SEEDS = list(range(10))


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Evaluation ──────────────────────────────────────────────────────────────
def evaluate_xgb(X_train, y_train, X_test, y_test, seed):
    """
    Scale y → fit XGBRegressor → predict → inverse-transform → metrics.
    Returns (RMSE, MAE, Pearson, Spearman).
    """
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        colsample_bytree=0.3,
        subsample=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_s)

    y_pred_s = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test_s.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    r = float(pearsonr(y_pred, y_true)[0])
    rho = float(spearmanr(y_pred, y_true)[0])
    return rmse, mae, r, rho


# ── Data loading helper ─────────────────────────────────────────────────────
def load_tensors(split):
    emb = torch.load(
        os.path.join(DATA_DIR, f"{split}_embeddings.pt"), weights_only=True
    )
    tgt = torch.load(
        os.path.join(DATA_DIR, f"{split}_targets.pt"), weights_only=True
    )
    return emb.numpy(), tgt.numpy()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[1/3] Loading MolFormer embeddings …")
    X_train, y_train = load_tensors("train")
    X_test, y_test = load_tensors("test")
    print(f"       Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    print(f"\n[2/3] Evaluating across {len(SEEDS)} seeds …\n")

    seed_metrics = []
    sweep_start = time.perf_counter()
    for i, seed in enumerate(SEEDS):
        m = evaluate_xgb(X_train, y_train, X_test, y_test, seed)
        print(
            f"    seed={seed:>2d}  RMSE={m[0]:.4f}  MAE={m[1]:.4f}  "
            f"r={m[2]:.4f}  ρ={m[3]:.4f}"
        )
        seed_metrics.append(m)
        avg_run_s = (time.perf_counter() - sweep_start) / (i + 1)
        eta_s = avg_run_s * (len(SEEDS) - i - 1)
        print(f"      [sweep ETA] remaining: {format_eta(eta_s)}")

    arr = np.array(seed_metrics)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    print(
        f"\n  → mean  RMSE={means[0]:.4f}±{stds[0]:.4f}  "
        f"MAE={means[1]:.4f}±{stds[1]:.4f}  "
        f"r={means[2]:.4f}±{stds[2]:.4f}  "
        f"ρ={means[3]:.4f}±{stds[3]:.4f}\n"
    )

    print("[3/3] Writing report …\n")
    seeds_str = ", ".join(str(s) for s in SEEDS)
    header = (
        f"{'Dataset':<24s} {'N':>5s}  {'RMSE':>14s}  {'MAE':>14s}  "
        f"{'Pearson':>14s}  {'Spearman':>14s}"
    )
    sep = "─" * len(header)
    lines = [
        "XGBoost (MolFormer Embeddings) — AqSolDB (Multi-Seed)",
        "======================================================",
        f"Seeds: [{seeds_str}]  ({len(SEEDS)} runs)",
        "Architecture: XGBRegressor(n_estimators=300, lr=0.05, max_depth=3, "
        "colsample_bytree=0.3, subsample=0.8, reg_alpha=1.0, reg_lambda=1.0)",
        "",
        header,
        sep,
        f"{'Original (100%)':<24s} {X_train.shape[0]:>5d}  "
        f"{means[0]:.4f}±{stds[0]:.4f}  "
        f"{means[1]:.4f}±{stds[1]:.4f}  "
        f"{means[2]:.4f}±{stds[2]:.4f}  "
        f"{means[3]:.4f}±{stds[3]:.4f}",
        "",
        sep,
        f"Mean MAE: {means[1]:.4f} ± {stds[1]:.4f}",
    ]

    report = "\n".join(lines)
    print(report)

    report_path = os.path.join(RESULTS_DIR, "xgboost_molformer_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
