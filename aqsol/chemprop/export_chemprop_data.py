"""
export_chemprop_data.py
=======================
Export AqSolDB splits as CSV files for Chemprop training using the
TDC ADMET benchmark group protocol.

Outputs:
  data/chemprop_test.csv                 -- fixed benchmark test set (shared)
  data/chemprop_train_seed_{s}.csv       -- per-seed training set  (s in 1..5)
  data/chemprop_val_seed_{s}.csv         -- per-seed validation set (s in 1..5)

Each CSV has columns: smiles, y

Usage:
    python export_chemprop_data.py
"""

import os

import pandas as pd
from tdc.benchmark_group import admet_group

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SEEDS = [1, 2, 3, 4, 5]


def df_to_csv(df, path):
    out = pd.DataFrame({"smiles": df["Drug"].tolist(), "y": df["Y"].tolist()})
    out.to_csv(path, index=False)
    return len(out)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[1/2] Loading Solubility_AqSolDB from TDC ADMET benchmark group ...")
    group = admet_group(path=DATA_DIR)
    benchmark = group.get("Solubility_AqSolDB")
    name = benchmark["name"]
    test_df = benchmark["test"]

    # Fixed test set -- same across all seeds
    test_path = os.path.join(DATA_DIR, "chemprop_test.csv")
    n_test = df_to_csv(test_df, test_path)
    print(f"       Saved chemprop_test.csv  ({n_test} rows)")

    print(f"[2/2] Writing per-seed train/val CSVs for seeds {SEEDS} ...")
    for seed in SEEDS:
        train_df, val_df = group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed
        )

        train_path = os.path.join(DATA_DIR, f"chemprop_train_seed_{seed}.csv")
        val_path = os.path.join(DATA_DIR, f"chemprop_val_seed_{seed}.csv")

        n_train = df_to_csv(train_df, train_path)
        n_val = df_to_csv(val_df, val_path)
        print(
            f"       seed={seed}  train={n_train} rows -> chemprop_train_seed_{seed}.csv"
            f"  |  val={n_val} rows -> chemprop_val_seed_{seed}.csv"
        )

    print("\nDone — Chemprop CSV files written to data/")


if __name__ == "__main__":
    main()
