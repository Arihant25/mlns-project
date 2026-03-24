"""
generate_embeddings_minimol.py
==============================
Generate 512-d MiniMol embeddings for the Solubility_AqSolDB dataset
using the TDC ADMET benchmark group protocol.

Saves:
  data/trainval_embeddings_minimol.pt  -- shape [N_trainval, 512]
  data/trainval_targets.pt             -- shape [N_trainval]
  data/trainval_smiles.pt              -- list of N_trainval SMILES
  data/test_embeddings_minimol.pt      -- shape [N_test, 512]
  data/test_targets.pt                 -- shape [N_test]
  data/test_smiles.pt                  -- list of N_test SMILES

Per-seed train/valid splits are derived at training time via:
    group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)

Usage:
    python generate_embeddings_minimol.py

Prerequisites:
    pip install minimol torch-sparse torch-scatter torch-cluster
"""

import os

import numpy as np
import torch
from minimol import Minimol
from tdc.benchmark_group import admet_group

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
BATCH_SIZE = 64


EMBED_DIM = 512


def _embed_one(model, smi):
    """Embed a single SMILES, returning a zero vector if it fails."""
    try:
        result = model([smi])
        if isinstance(result, list):
            e = result[0]
            return e.float() if isinstance(e, torch.Tensor) else torch.zeros(EMBED_DIM)
        return torch.tensor(np.array(result)[0], dtype=torch.float32)
    except Exception:
        return torch.zeros(EMBED_DIM)


def embed_smiles(model, smiles_list, label=""):
    all_emb = []
    for start in range(0, len(smiles_list), BATCH_SIZE):
        batch = smiles_list[start : start + BATCH_SIZE]
        try:
            batch_emb = model(batch)
            if isinstance(batch_emb, list):
                # MiniMol returns SMILES strings for molecules it can't featurize;
                # fall back per-molecule for any string in the list.
                if any(isinstance(e, str) for e in batch_emb):
                    raise ValueError("bad molecules in batch")
                batch_emb = torch.stack([
                    e if isinstance(e, torch.Tensor) else torch.tensor(e)
                    for e in batch_emb
                ])
            else:
                batch_emb = torch.tensor(np.array(batch_emb), dtype=torch.float32)
        except Exception:
            batch_emb = torch.stack([_embed_one(model, smi) for smi in batch])
        all_emb.append(batch_emb.float())
        print(
            f"  [{label}] {min(start + BATCH_SIZE, len(smiles_list)):>5d}"
            f" / {len(smiles_list)} molecules"
        )
    return torch.cat(all_emb, dim=0).float()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[1/3] Loading Solubility_AqSolDB from TDC ADMET benchmark group ...")
    group = admet_group(path=DATA_DIR)
    benchmark = group.get("Solubility_AqSolDB")
    train_val_df = benchmark["train_val"]
    test_df = benchmark["test"]
    print(f"       TrainVal: {len(train_val_df)}, Test: {len(test_df)}")

    print("[2/3] Loading MiniMol featurizer ...")
    model = Minimol()
    print("       MiniMol loaded (512-d output)")

    print("[3/3] Generating embeddings ...")
    for split_name, df in [("trainval", train_val_df), ("test", test_df)]:
        smiles = df["Drug"].tolist()
        targets = df["Y"].values.astype(np.float32)

        embeddings = embed_smiles(model, smiles, label=split_name)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        torch.save(embeddings, os.path.join(DATA_DIR, f"{split_name}_embeddings_minimol.pt"))
        torch.save(targets_tensor, os.path.join(DATA_DIR, f"{split_name}_targets.pt"))
        torch.save(smiles, os.path.join(DATA_DIR, f"{split_name}_smiles.pt"))
        print(f"       Saved {split_name}_embeddings_minimol.pt  shape={embeddings.shape}")

    print("\nDone — MiniMol embeddings saved to data/")


if __name__ == "__main__":
    main()
