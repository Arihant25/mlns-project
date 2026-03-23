"""
export_sample_weights.py
========================
Export per-sample soft weights for Chemprop training using aleatoric
uncertainty from the trained MolFormer-based EvidentialGSLModel.

For each gamma in GAMMAS:
    w_i = exp(-gamma * u_i)

Writes the weight as an additional column in the training CSV so that
Chemprop v2's -w / --weight-column flag can reference it.

Outputs (one per gamma):
    data/chemprop_train_gamma_{gamma}.csv  — train CSV with 'weight' column added

Also saves:
    data/train_uncertainties.pt  — raw u_i tensor for reuse

Usage:
    python export_sample_weights.py

Prerequisites:
    Run aqsol/gsl/train_evidential_gsl.py  → results/evidential_gsl.pt
    Run export_chemprop_data.py            → data/chemprop_train.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── Cross-directory imports ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GSL_DIR = os.path.join(PROJECT_ROOT, "gsl")
sys.path.insert(0, GSL_DIR)

from model_evidential_gsl import EvidentialGSLModel  # noqa: E402
from train_gsl import GSLDataset, gsl_collate_fn  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "evidential_gsl.pt")  # MolFormer 768-d model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMAS = [0.0, 7.0, 10.0, 12.5, 15.0, 20.0]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run aqsol/gsl/train_evidential_gsl.py first."
        )

    # 1. Load training embeddings and SMILES (MolFormer 768-d)
    print("[1/3] Loading MolFormer training embeddings and SMILES …")
    train_emb = torch.load(
        os.path.join(DATA_DIR, "train_embeddings.pt"), weights_only=True
    )
    train_tgt = torch.load(
        os.path.join(DATA_DIR, "train_targets.pt"), weights_only=True
    )
    train_smiles = torch.load(os.path.join(DATA_DIR, "train_smiles.pt"))
    print(f"       Train: {train_emb.shape[0]} molecules, embed_dim={train_emb.shape[1]}")

    # Scale targets (same as during training — fit on train only)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_tgt_s = torch.tensor(
        scaler.fit_transform(train_tgt.numpy().reshape(-1, 1)).flatten(),
        dtype=torch.float32,
    )

    # 2. Run EvidentialGSLModel inference to get u_i
    print("[2/3] Running EvidentialGSLModel inference …")
    model = EvidentialGSLModel()  # default embed_dim=768
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    n = train_emb.shape[0]
    loader = DataLoader(
        GSLDataset(train_emb, train_tgt_s, train_smiles),
        batch_size=n,
        shuffle=False,
        collate_fn=gsl_collate_fn,
    )

    with torch.no_grad():
        for X, _, A_ecfp in loader:
            X, A_ecfp = X.to(DEVICE), A_ecfp.to(DEVICE)
            _, (_, _, alpha, beta) = model(X, A_ecfp)

    u = (beta / (alpha - 1.0)).detach().cpu()
    print(f"       Uncertainty: min={u.min():.4f} max={u.max():.4f} mean={u.mean():.4f}")

    # Save raw uncertainties for reuse
    u_path = os.path.join(DATA_DIR, "train_uncertainties.pt")
    torch.save(u, u_path)
    print(f"       Saved uncertainties → {u_path}")

    # 3. Load base training CSV and embed weight column per gamma
    print(f"[3/3] Writing weighted training CSVs for {len(GAMMAS)} gamma values …")
    chemprop_train_path = os.path.join(DATA_DIR, "chemprop_train.csv")
    if not os.path.exists(chemprop_train_path):
        raise FileNotFoundError(
            f"Missing {chemprop_train_path}. Run export_chemprop_data.py first."
        )
    base_df = pd.read_csv(chemprop_train_path)

    if len(base_df) != len(train_smiles):
        raise ValueError(
            f"Row count mismatch: chemprop_train.csv has {len(base_df)} rows "
            f"but train_smiles has {len(train_smiles)} entries. "
            "Ensure both use the same scaffold split."
        )

    u_np = u.numpy()
    for gamma in GAMMAS:
        weights = np.exp(-gamma * u_np)
        df_w = base_df.copy()
        df_w["weight"] = weights
        out_path = os.path.join(DATA_DIR, f"chemprop_train_gamma_{gamma}.csv")
        df_w.to_csv(out_path, index=False)
        print(
            f"       gamma={gamma:5.1f}  weight range [{weights.min():.4f}, "
            f"{weights.max():.4f}]  → {out_path}"
        )

    print("\nDone — weighted CSVs ready for train_chemprop_weighted.py")


if __name__ == "__main__":
    main()
