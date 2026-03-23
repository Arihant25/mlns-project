"""
generate_embeddings_minimol.py
==============================
Generate 512-d MiniMol embeddings for the Solubility_AqSolDB dataset
and save them to disk alongside SMILES and targets.

MiniMol is the TDC ADMET leaderboard Rank-1 model for AqSolDB (MAE 0.741).
It is a 10M-parameter molecular featurizer pre-trained on 3,300+ biological
and quantum-mechanical datasets.

Usage:
    python generate_embeddings_minimol.py

Prerequisites:
    pip install minimol torch-sparse torch-scatter torch-cluster
"""

import os

import numpy as np
import torch
from minimol import Minimol
from tdc.single_pred import ADME

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
BATCH_SIZE = 64


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Load dataset via PyTDC (same scaffold split as MolFormer pipeline)
    print("[1/3] Loading Solubility (AqSolDB) dataset from TDC …")
    data = ADME(name="Solubility_AqSolDB")
    split = data.get_split(method="scaffold")
    train_df = split["train"]
    val_df = split["valid"]
    test_df = split["test"]
    print(f"       Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Load MiniMol featurizer
    print("[2/3] Loading MiniMol featurizer …")
    model = Minimol()
    print("       MiniMol loaded (512-d output)")

    # 3. Generate embeddings for each split
    print("[3/3] Generating embeddings …")
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        smiles = df["Drug"].tolist()
        targets = df["Y"].values.astype(np.float32)

        # MiniMol returns a list of tensors; process in batches for progress reporting
        all_emb = []
        for start in range(0, len(smiles), BATCH_SIZE):
            batch = smiles[start : start + BATCH_SIZE]
            batch_emb = model(batch)  # list of (512,) tensors
            if isinstance(batch_emb, list):
                batch_emb = torch.stack([
                    e if isinstance(e, torch.Tensor) else torch.tensor(e)
                    for e in batch_emb
                ])
            else:
                batch_emb = torch.tensor(np.array(batch_emb), dtype=torch.float32)
            all_emb.append(batch_emb)
            print(
                f"  [{split_name}] {min(start + BATCH_SIZE, len(smiles)):>5d}"
                f" / {len(smiles)} molecules"
            )

        embeddings = torch.cat(all_emb, dim=0).float()
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        emb_path = os.path.join(DATA_DIR, f"{split_name}_embeddings_minimol.pt")
        tgt_path = os.path.join(DATA_DIR, f"{split_name}_targets.pt")
        smi_path = os.path.join(DATA_DIR, f"{split_name}_smiles.pt")

        torch.save(embeddings, emb_path)
        # Targets and SMILES are shared with MolFormer pipeline; only overwrite if
        # not already present to avoid conflicts when running pipelines together.
        if not os.path.exists(tgt_path):
            torch.save(targets_tensor, tgt_path)
        if not os.path.exists(smi_path):
            torch.save(smiles, smi_path)

        print(f"       Saved {emb_path}  shape={embeddings.shape}")

    print("\nDone — MiniMol embeddings saved to data/")


if __name__ == "__main__":
    main()
