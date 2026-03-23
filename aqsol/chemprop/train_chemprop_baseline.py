"""
train_chemprop_baseline.py
===========================
Chemprop-RDKit "as-is" baseline for AqSolDB.

Reproduces the TDC ADMET leaderboard Rank-2 approach (MAE 0.761 ± 0.025):
    Chemprop MPNN + v1_rdkit_2d_normalized molecular features.

Uses Chemprop v2 Python API.

Usage:
    python train_chemprop_baseline.py

Prerequisites:
    pip install chemprop
    python export_chemprop_data.py  (writes data/chemprop_{train,val,test}.csv)
"""

import os

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

TRAIN_CSV = os.path.join(DATA_DIR, "chemprop_train.csv")
VAL_CSV = os.path.join(DATA_DIR, "chemprop_val.csv")
TEST_CSV = os.path.join(DATA_DIR, "chemprop_test.csv")
SAVE_DIR = os.path.join(RESULTS_DIR, "chemprop_baseline")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run export_chemprop_data.py first."
            )

    import chemprop

    print(f"Chemprop version: {chemprop.__version__}")

    # Chemprop v2 API: -i accepts train, val, test as positional sequence
    # --molecule-featurizers (or --features-generators) for RDKit features
    # -o for output directory
    args = [
        "-i",
        TRAIN_CSV,
        VAL_CSV,
        TEST_CSV,
        "--task-type",
        "regression",
        "--molecule-featurizers",
        "v1_rdkit_2d_normalized",
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "y",
        "-o",
        SAVE_DIR,
        "--epochs",
        "50",
        "--num-workers",
        "0",
    ]

    print("[1/2] Training Chemprop-RDKit baseline …")
    print(f"       Args: {' '.join(args)}\n")

    import sys
    from chemprop.cli.main import main as chemprop_main

    sys.argv = ["chemprop", "train"] + args
    chemprop_main()

    # Parse results
    import glob
    import json

    result_files = glob.glob(
        os.path.join(SAVE_DIR, "**", "test_scores.json"), recursive=True
    )

    if result_files:
        with open(result_files[0]) as f:
            scores = json.load(f)
        score_lines = "\n".join(f"  {k}: {v}" for k, v in scores.items())
        report = (
            "Chemprop-RDKit Baseline — AqSolDB Test Metrics\n"
            "================================================\n"
            "Model: Chemprop MPNN + v1_rdkit_2d_normalized (= Chemprop-RDKit)\n"
            "TDC Rank-2 reference MAE: 0.761 ± 0.025\n"
            "\n"
            f"Test scores:\n{score_lines}\n"
            f"\nResults directory: {SAVE_DIR}\n"
        )
    else:
        report = (
            "Chemprop-RDKit Baseline — AqSolDB\n"
            "===================================\n"
            f"Results saved in: {SAVE_DIR}\n"
            "Check test_scores.json in the save directory for metrics.\n"
        )

    print("\n" + report)
    report_path = os.path.join(RESULTS_DIR, "chemprop_baseline_metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[2/2] Report saved → {report_path}")


if __name__ == "__main__":
    main()
