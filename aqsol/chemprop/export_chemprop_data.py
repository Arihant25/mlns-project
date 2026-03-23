"""
export_chemprop_data.py
=======================
Export AqSolDB train/val/test splits as CSV files for Chemprop training.

Each CSV has two columns: smiles,y

Chemprop expects the target column to be named 'y' by default, or the
column name must be passed via --target-columns. We use 'y' here.

Usage:
    python export_chemprop_data.py
"""

import os

import pandas as pd
import torch
from tdc.single_pred import ADME

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[1/2] Loading Solubility_AqSolDB from TDC …")
    data = ADME(name="Solubility_AqSolDB")
    split = data.get_split(method="scaffold")
    train_df = split["train"]
    val_df = split["valid"]
    test_df = split["test"]
    print(f"       Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("[2/2] Writing Chemprop CSV files …")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = pd.DataFrame({"smiles": df["Drug"].tolist(), "y": df["Y"].tolist()})
        path = os.path.join(DATA_DIR, f"chemprop_{name}.csv")
        out.to_csv(path, index=False)
        print(f"       Saved {path}  ({len(out)} rows)")

    print("\nDone — Chemprop CSV files written to data/")


if __name__ == "__main__":
    main()
