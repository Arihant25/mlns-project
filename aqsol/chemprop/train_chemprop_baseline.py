"""
train_chemprop_baseline.py
===========================
Chemprop-RDKit "as-is" baseline for AqSolDB using the TDC ADMET benchmark
group evaluation protocol.

Reproduces the TDC ADMET leaderboard Rank-2 approach (MAE 0.761 +/- 0.025):
    Chemprop MPNN + v1_rdkit_2d_normalized molecular features.

Runs 5 independent seeds [1,2,3,4,5] with per-seed train/valid splits and
a fixed benchmark test set. Final metrics via group.evaluate_many.

Usage:
    python train_chemprop_baseline.py

Prerequisites:
    pip install chemprop
    python export_chemprop_data.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from tdc.benchmark_group import admet_group

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SAVE_DIR = os.path.join(RESULTS_DIR, "chemprop_baseline")
TEST_CSV = os.path.join(DATA_DIR, "chemprop_test.csv")

SEEDS = [1, 2, 3, 4, 5]


def run_chemprop_seed(seed):
    from chemprop.cli.main import main as chemprop_main

    train_csv = os.path.join(DATA_DIR, f"chemprop_train_seed_{seed}.csv")
    val_csv = os.path.join(DATA_DIR, f"chemprop_val_seed_{seed}.csv")
    seed_save_dir = os.path.join(SAVE_DIR, f"seed_{seed}")

    for path in [train_csv, val_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run export_chemprop_data.py first."
            )

    args = [
        "-i", train_csv, val_csv, TEST_CSV,
        "--task-type", "regression",
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--smiles-columns", "smiles",
        "--target-columns", "y",
        "-o", seed_save_dir,
        "--epochs", "50",
        "--num-workers", "0",
        "--pytorch-seed", str(seed),
    ]
    sys.argv = ["chemprop", "train"] + args
    chemprop_main()
    return seed_save_dir


def load_test_predictions(seed_save_dir):
    pred_files = glob.glob(
        os.path.join(seed_save_dir, "**", "test_predictions.csv"), recursive=True
    )
    if not pred_files:
        return None
    return pd.read_csv(pred_files[0])["y"].to_numpy()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(
            f"Missing {TEST_CSV}. Run export_chemprop_data.py first."
        )

    import chemprop
    print(f"Chemprop version: {chemprop.__version__}")

    group = admet_group(path=DATA_DIR)
    benchmark = group.get("Solubility_AqSolDB")
    name = benchmark["name"]

    print(f"[1/2] Training Chemprop-RDKit baseline over {len(SEEDS)} seeds: {SEEDS} ...")
    predictions_list = []
    for seed in SEEDS:
        print(f"  -- seed={seed} --")
        seed_save_dir = run_chemprop_seed(seed)
        y_pred = load_test_predictions(seed_save_dir)
        if y_pred is not None:
            predictions_list.append({name: y_pred})

    print("[2/2] Evaluating ...")
    results = group.evaluate_many(predictions_list)
    mean_mae, std_mae = list(results.values())[0]

    lines = [
        "Chemprop-RDKit Baseline -- AqSolDB Test Metrics",
        "==================================================",
        "Model: Chemprop MPNN + v1_rdkit_2d_normalized (= Chemprop-RDKit)",
        f"Seeds: {SEEDS}",
        "TDC Rank-2 reference: MAE 0.761 +/- 0.025",
        "",
        f"MAE      : {mean_mae:.4f} +/- {std_mae:.4f}",
        "",
        "[Evaluated via group.evaluate_many -- TDC official metric]",
        f"\nResults directory: {SAVE_DIR}",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    report_path = os.path.join(RESULTS_DIR, "chemprop_baseline_metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"Report saved -> {report_path}")


if __name__ == "__main__":
    main()
