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
    # TDC standard: 5 seeds [1,2,3,4,5]
    SEEDS = [1, 2, 3, 4, 5]

    import sys
    from chemprop.cli.main import main as chemprop_main

    print(f"[1/2] Training Chemprop-RDKit baseline over {len(SEEDS)} seeds: {SEEDS} ...")
    for seed in SEEDS:
        seed_save_dir = os.path.join(SAVE_DIR, f"seed_{seed}")
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
            seed_save_dir,
            "--epochs",
            "50",
            "--num-workers",
            "0",
            "--pytorch-seed",
            str(seed),
        ]
        print(f"  -- seed={seed} --")
        sys.argv = ["chemprop", "train"] + args
        chemprop_main()

    # Aggregate results across seeds
    import glob
    import json
    import numpy as np

    all_scores = {}
    for seed in SEEDS:
        seed_save_dir = os.path.join(SAVE_DIR, f"seed_{seed}")
        result_files = glob.glob(
            os.path.join(seed_save_dir, "**", "test_scores.json"), recursive=True
        )
        if result_files:
            with open(result_files[0]) as f:
                scores = json.load(f)
            for k, v in scores.items():
                all_scores.setdefault(k, []).append(v)

    lines = [
        "Chemprop-RDKit Baseline -- AqSolDB Test Metrics",
        "==================================================",
        "Model: Chemprop MPNN + v1_rdkit_2d_normalized (= Chemprop-RDKit)",
        f"Seeds: {SEEDS}",
        "TDC Rank-2 reference: MAE 0.761 +/- 0.025",
        "",
    ]
    if all_scores:
        for k, vals in all_scores.items():
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            lines.append(f"  {k}: {mean:.4f} +/- {std:.4f}")
            for seed, v in zip(SEEDS, vals):
                lines.append(f"    seed={seed}: {v:.4f}")
    else:
        lines.append(f"Results saved in: {SAVE_DIR}")
        lines.append("Check test_scores.json in each seed subdirectory.")

    lines.append(f"\nResults directory: {SAVE_DIR}")
    report = "\n".join(lines)

    print("\n" + report)
    report_path = os.path.join(RESULTS_DIR, "chemprop_baseline_metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"[2/2] Report saved -> {report_path}")


if __name__ == "__main__":
    main()
