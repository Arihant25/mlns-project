"""
train_chemprop_weighted.py
===========================
Chemprop-RDKit with our soft weighting: trains Chemprop on AqSolDB with
per-sample loss weights derived from EvidentialGSLModel aleatoric uncertainty.

    w_i = exp(-gamma * u_i)   where u_i = beta_i / (alpha_i - 1)

The weight is embedded as a 'weight' column in the training CSV and passed
to Chemprop v2 via the -w / --weight-column flag.

Sweeps over GAMMAS and reports test metrics for each.

Usage:
    python train_chemprop_weighted.py

Prerequisites:
    python export_chemprop_data.py       → data/chemprop_{train,val,test}.csv
    python export_sample_weights.py      → data/chemprop_train_gamma_*.csv
"""

import glob
import json
import os

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

VAL_CSV = os.path.join(DATA_DIR, "chemprop_val.csv")
TEST_CSV = os.path.join(DATA_DIR, "chemprop_test.csv")

GAMMAS = [0.0, 7.0, 10.0, 12.5, 15.0, 20.0]


def run_chemprop_for_gamma(gamma: float) -> dict:
    """Train Chemprop with soft weights for one gamma value."""
    train_csv = os.path.join(DATA_DIR, f"chemprop_train_gamma_{gamma}.csv")
    save_dir = os.path.join(RESULTS_DIR, f"chemprop_weighted_gamma_{gamma}")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"Missing {train_csv}. Run export_sample_weights.py first."
        )

    from chemprop.cli.train import TrainSubcommand

    args = [
        "-i",
        train_csv,
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
        "-w",
        "weight",  # per-sample weight column in training CSV
        "-o",
        save_dir,
        "--epochs",
        "50",
        "--num-workers",
        "0",
    ]

    print(f"  gamma={gamma}  train_csv={train_csv}")
    subcommand = TrainSubcommand()
    parser = subcommand.parser
    train_args = parser.parse_args(args)
    subcommand.func(train_args)

    result_files = glob.glob(
        os.path.join(save_dir, "**", "test_scores.json"), recursive=True
    )
    if result_files:
        with open(result_files[0]) as f:
            return json.load(f)
    return {}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for path in [VAL_CSV, TEST_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run export_chemprop_data.py first."
            )

    import chemprop

    print(f"Chemprop version: {chemprop.__version__}")
    print(f"Sweeping {len(GAMMAS)} gamma values: {GAMMAS}\n")

    results = []
    for gamma in GAMMAS:
        print(f"\n── gamma={gamma} ──────────────────────────────────────────")
        scores = run_chemprop_for_gamma(gamma)
        results.append({"gamma": gamma, "scores": scores})
        print(f"  Scores: {scores}")

    # Summary report
    lines = [
        "Chemprop-RDKit + Soft Weighting — AqSolDB",
        "==========================================",
        "Model: Chemprop MPNN + v1_rdkit_2d_normalized + uncertainty-weighted loss",
        "Uncertainty source: EvidentialGSLModel (MolFormer 768-d)",
        "w_i = exp(-gamma * u_i),  u_i = beta_i / (alpha_i - 1)",
        "",
        f"{'gamma':>7s}  {'scores'}",
        "-" * 60,
    ]
    for r in results:
        lines.append(f"{r['gamma']:>7.2f}  {r['scores']}")

    # Find best gamma
    try:
        scored = [
            (r["gamma"], list(r["scores"].values())[0]) for r in results if r["scores"]
        ]
        if scored:
            best_gamma, best_score = min(scored, key=lambda x: x[1])
            lines += [
                "",
                "-" * 60,
                f"Best gamma: {best_gamma}  score: {best_score:.4f}",
            ]
    except Exception:
        pass

    report = "\n".join(lines)
    print("\n\n" + report)

    report_path = os.path.join(RESULTS_DIR, "chemprop_weighted_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
