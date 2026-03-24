"""
train_chemprop_weighted.py
===========================
Chemprop-RDKit with our soft weighting: trains Chemprop on AqSolDB with
per-sample loss weights derived from EvidentialGSLModel aleatoric uncertainty.

    w_i = exp(-gamma * u_i)   where u_i = beta_i / (alpha_i - 1)

Uses the TDC ADMET benchmark group evaluation protocol:
  - Per-seed train/valid splits (5 seeds [1,2,3,4,5])
  - Fixed benchmark test set evaluated via group.evaluate_many
  - Sweeps over GAMMAS x SEEDS

Usage:
    python train_chemprop_weighted.py

Prerequisites:
    python export_chemprop_data.py       -> data/chemprop_*_seed_*.csv
    python export_sample_weights.py      -> data/chemprop_train_gamma_*_seed_*.csv
"""

import glob
import os
import re
import sys
import time
from csv import DictReader

import numpy as np
import pandas as pd
from tdc.benchmark_group import admet_group

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TEST_CSV = os.path.join(DATA_DIR, "chemprop_test.csv")

GAMMAS = [0.0, 7.0, 10.0, 12.5, 15.0, 20.0]
SEEDS = [1, 2, 3, 4, 5]

_STANDALONE_H_PATTERN = re.compile(r"(^|\.)\[H[+-]?\](\.|$)")


def _count_standalone_h_fragments(csv_path: str) -> tuple[int, int]:
    total, flagged = 0, 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in DictReader(f):
            total += 1
            if _STANDALONE_H_PATTERN.search(row.get("smiles", "")):
                flagged += 1
    return flagged, total


def run_chemprop_seed_gamma(seed, gamma):
    from chemprop.cli.main import main as chemprop_main

    train_csv = os.path.join(DATA_DIR, f"chemprop_train_gamma_{gamma}_seed_{seed}.csv")
    val_csv = os.path.join(DATA_DIR, f"chemprop_val_seed_{seed}.csv")
    save_dir = os.path.join(RESULTS_DIR, f"chemprop_weighted_gamma_{gamma}", f"seed_{seed}")

    for path in [train_csv, val_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run export_sample_weights.py first."
            )

    args = [
        "-i", train_csv, val_csv, TEST_CSV,
        "--task-type", "regression",
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--smiles-columns", "smiles",
        "--target-columns", "y",
        "-w", "weight",
        "-o", save_dir,
        "--epochs", "50",
        "--num-workers", "0",
        "--pytorch-seed", str(seed),
    ]
    sys.argv = ["chemprop", "train"] + args
    chemprop_main()
    return save_dir


def load_test_predictions(save_dir):
    pred_files = glob.glob(
        os.path.join(save_dir, "**", "test_predictions.csv"), recursive=True
    )
    if not pred_files:
        return None
    return pd.read_csv(pred_files[0])["y"].to_numpy()


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(
            f"Missing {TEST_CSV}. Run export_chemprop_data.py first."
        )

    import chemprop
    print(f"Chemprop version: {chemprop.__version__}")

    flagged, total = _count_standalone_h_fragments(TEST_CSV)
    print(
        f"Note: RDKit warnings for standalone hydrogen fragments are expected. "
        f"test={flagged}/{total} affected rows."
    )

    group = admet_group(path=DATA_DIR)
    benchmark = group.get("Solubility_AqSolDB")
    name = benchmark["name"]

    print(f"Sweep: {len(GAMMAS)} gammas x {len(SEEDS)} seeds ({len(GAMMAS)*len(SEEDS)} total runs)\n")

    sweep_results = []
    total_runs = len(GAMMAS) * len(SEEDS)
    completed = 0
    sweep_start = time.perf_counter()

    for gamma in GAMMAS:
        print(f"\n-- gamma={gamma} --")
        gamma_predictions = []

        for seed in SEEDS:
            print(f"  seed={seed}  gamma={gamma}")
            save_dir = run_chemprop_seed_gamma(seed, gamma)
            y_pred = load_test_predictions(save_dir)
            if y_pred is not None:
                gamma_predictions.append({name: y_pred})

            completed += 1
            eta_s = (
                (time.perf_counter() - sweep_start) / completed
            ) * (total_runs - completed)
            print(f"    [sweep ETA] {format_eta(eta_s)}")

        results = group.evaluate_many(gamma_predictions)
        mean_mae, std_mae = list(results.values())[0]
        sweep_results.append({
            "gamma": gamma,
            "mae_mean": mean_mae,
            "mae_std": std_mae,
        })
        print(f"  gamma={gamma:.2f}  MAE={mean_mae:.4f} +/- {std_mae:.4f}")

    best = min(sweep_results, key=lambda x: x["mae_mean"])
    lines = [
        "Chemprop-RDKit + Soft Weighting -- AqSolDB",
        "============================================",
        "Model: Chemprop MPNN + v1_rdkit_2d_normalized + uncertainty-weighted loss",
        "Uncertainty source: EvidentialGSLModel (MiniMol 512-d)",
        "w_i = exp(-gamma * u_i),  u_i = beta_i / (alpha_i - 1)",
        f"Seeds: {SEEDS}",
        "TDC Rank-2 reference: MAE 0.761 +/- 0.025",
        "",
        f"{'gamma':>7s}  {'MAE':>20s}",
        "-" * 35,
    ]
    for r in sweep_results:
        lines.append(
            f"{r['gamma']:>7.2f}  "
            f"{r['mae_mean']:.4f} +/- {r['mae_std']:.4f}"
        )
    lines += [
        "", "-" * 35,
        f"Best MAE at gamma={best['gamma']:.2f}: "
        f"{best['mae_mean']:.4f} +/- {best['mae_std']:.4f}",
        "",
        "[Evaluated via group.evaluate_many -- TDC official metric]",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    report_path = os.path.join(RESULTS_DIR, "chemprop_weighted_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
