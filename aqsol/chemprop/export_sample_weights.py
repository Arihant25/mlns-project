"""
export_sample_weights.py
========================
Export per-sample soft weights for Chemprop training using aleatoric
uncertainty from the trained MiniMol EvidentialGSLModel (512-d).

For each seed and gamma:
    w_i = exp(-gamma * u_i)

Outputs:
    data/chemprop_train_gamma_{gamma}_seed_{s}.csv  -- train CSV with 'weight' column

Usage:
    python export_sample_weights.py

Prerequisites:
    python minimol/train_minimol_evidential_gsl.py  -> results/evidential_gsl_minimol.pt
    python export_chemprop_data.py                  -> data/chemprop_train_seed_*.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tdc.benchmark_group import admet_group

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GSL_DIR = os.path.join(PROJECT_ROOT, "gsl")
sys.path.insert(0, GSL_DIR)

from gsl_utils import GSLDataset, gsl_collate_fn  # noqa: E402
from model_evidential_gsl import EvidentialGSLModel  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "evidential_gsl_minimol.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMAS = [0.0, 7.0, 10.0, 12.5, 15.0, 20.0]
EMBED_DIM = 512
SEEDS = [1, 2, 3, 4, 5]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run minimol/train_minimol_evidential_gsl.py first."
        )

    print("[1/3] Loading MiniMol trainval embeddings and benchmark group ...")
    all_tv_emb = torch.load(
        os.path.join(DATA_DIR, "trainval_embeddings_minimol.pt"), weights_only=True
    )
    all_tv_tgt = torch.load(
        os.path.join(DATA_DIR, "trainval_targets.pt"), weights_only=True
    )
    all_tv_smi = torch.load(os.path.join(DATA_DIR, "trainval_smiles.pt"))
    smi_to_idx = {s: i for i, s in enumerate(all_tv_smi)}
    print(f"       TrainVal: {all_tv_emb.shape[0]} molecules, embed_dim={all_tv_emb.shape[1]}")

    # Scaler consistent with training scripts
    scaler = StandardScaler()
    scaler.fit(all_tv_tgt.numpy().reshape(-1, 1))

    def scale(t):
        return torch.tensor(
            scaler.transform(t.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32
        )

    group = admet_group(path=DATA_DIR)
    benchmark = group.get("Solubility_AqSolDB")
    name = benchmark["name"]

    print("[2/3] Loading EvidentialGSLModel ...")
    unc_model = EvidentialGSLModel(embed_dim=EMBED_DIM)
    unc_model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    unc_model.to(DEVICE)
    unc_model.eval()

    print(f"[3/3] Writing weighted CSVs for {len(SEEDS)} seeds x {len(GAMMAS)} gammas ...")
    for seed in SEEDS:
        train_df, _ = group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed
        )
        t_idx = [smi_to_idx[s] for s in train_df["Drug"].tolist()]
        train_emb = all_tv_emb[t_idx]
        train_smi = train_df["Drug"].tolist()
        train_tgt_s = scale(all_tv_tgt[t_idx])

        # Run uncertainty inference
        loader = DataLoader(
            GSLDataset(train_emb, train_tgt_s, train_smi),
            batch_size=train_emb.shape[0], shuffle=False, collate_fn=gsl_collate_fn,
        )
        with torch.no_grad():
            for X, _, A in loader:
                X, A = X.to(DEVICE), A.to(DEVICE)
                _, (_, _, alpha, beta) = unc_model(X, A)
        u = (beta / (alpha - 1.0)).detach().cpu().numpy()
        print(
            f"  seed={seed}  uncertainty: "
            f"min={u.min():.4f}  max={u.max():.4f}  mean={u.mean():.4f}"
        )

        base_csv = os.path.join(DATA_DIR, f"chemprop_train_seed_{seed}.csv")
        if not os.path.exists(base_csv):
            raise FileNotFoundError(
                f"Missing {base_csv}. Run export_chemprop_data.py first."
            )
        base_df = pd.read_csv(base_csv)

        for gamma in GAMMAS:
            weights = np.exp(-gamma * u)
            df_w = base_df.copy()
            df_w["weight"] = weights
            out_path = os.path.join(DATA_DIR, f"chemprop_train_gamma_{gamma}_seed_{seed}.csv")
            df_w.to_csv(out_path, index=False)
            print(
                f"    gamma={gamma:5.1f}  seed={seed}  "
                f"weight range [{weights.min():.4f}, {weights.max():.4f}]"
            )

    print("\nDone -- weighted CSVs ready for train_chemprop_weighted.py")


if __name__ == "__main__":
    main()
