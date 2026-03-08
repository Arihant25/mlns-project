"""
curate_dataset.py
=================
Phase 3A — Filter-and-Retrain protocol.

Uses the trained Evidential GSL model to estimate per-molecule aleatoric
uncertainty on the full training set, then creates curated subsets by removing
the top 5 %, 10 %, and 15 % most uncertain molecules.

Usage:
    python curate_dataset.py
"""

import os
import sys
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# ── Cross-directory import for loss (not used here, but keeps path consistent)
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MLP_DIR = os.path.join(PROJECT_ROOT, "mlp")
sys.path.insert(0, MLP_DIR)

# ── Same-directory imports ───────────────────────────────────────────────────
from model_evidential_gsl import EvidentialGSLModel          # noqa: E402
from train_gsl import (                                      # noqa: E402
    GSLDataset,
    gsl_collate_fn,
    load_tensors,
    load_smiles,
    fit_scaler,
    scale_targets,
)
from torch.utils.data import DataLoader


# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH  = os.path.join(RESULTS_DIR, "evidential_gsl.pt")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Percentages to remove (top-k most uncertain)
REMOVE_PCTS = [5, 10, 15]


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────────
    print("[1/6] Loading trained EvidentialGSLModel …")
    model = EvidentialGSLModel()
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"       Loaded weights from {MODEL_PATH}")

    # ── 2. Load full training data (un-shuffled) ─────────────────────────────
    print("[2/6] Loading full training data …")
    train_emb, train_tgt = load_tensors("train")
    smiles = load_smiles()
    train_smiles = smiles["train"]
    n_train = train_emb.shape[0]
    print(f"       {n_train} training molecules")

    # Re-fit scaler (same deterministic fit as during training)
    scaler = fit_scaler(train_tgt.numpy())
    train_tgt_s = scale_targets(train_tgt.numpy(), scaler)

    # Full-batch DataLoader (single pass — global adjacency matrix)
    loader = DataLoader(
        GSLDataset(train_emb, train_tgt_s, train_smiles),
        batch_size=n_train,
        shuffle=False,
        collate_fn=gsl_collate_fn,
    )

    # ── 3. Inference ─────────────────────────────────────────────────────────
    print("[3/6] Running full-batch inference …")
    with torch.no_grad():
        for X, y_s, A_ecfp in loader:
            X, y_s, A_ecfp = X.to(DEVICE), y_s.to(DEVICE), A_ecfp.to(DEVICE)
            _, (mu, v, alpha, beta) = model(X, A_ecfp)

    # ── 4. Uncertainty & error calculations ──────────────────────────────────
    print("[4/6] Computing uncertainty and prediction errors …")
    u = (beta / (alpha - 1.0)).cpu().numpy()                  # aleatoric unc.
    mu_np = mu.cpu().numpy()
    y_s_np = y_s.cpu().numpy()

    # Inverse-transform to original Lipophilicity units
    mu_orig  = scaler.inverse_transform(mu_np.reshape(-1, 1)).flatten()
    tgt_orig = scaler.inverse_transform(y_s_np.reshape(-1, 1)).flatten()
    abs_err  = np.abs(tgt_orig - mu_orig)

    # ── 5. Thresholding & filtering ──────────────────────────────────────────
    print("[5/6] Creating curated subsets …")

    report_lines = [
        "Phase 3A — Dataset Curation Report",
        "====================================",
        f"Total training molecules : {n_train}",
        "",
    ]

    for pct in REMOVE_PCTS:
        keep_pct = 100 - pct
        threshold = float(np.percentile(u, keep_pct))

        mask_keep = u <= threshold
        mask_remove = ~mask_keep

        n_keep   = int(mask_keep.sum())
        n_remove = int(mask_remove.sum())

        mae_kept    = float(abs_err[mask_keep].mean()) if n_keep > 0 else float("nan")
        mae_removed = float(abs_err[mask_remove].mean()) if n_remove > 0 else float("nan")

        # Save curated tensors (using ORIGINAL un-scaled embeddings & targets)
        suffix = f"curated_{pct:02d}"
        emb_path = os.path.join(DATA_DIR, f"train_embeddings_{suffix}.pt")
        tgt_path = os.path.join(DATA_DIR, f"train_targets_{suffix}.pt")
        torch.save(train_emb[mask_keep], emb_path)
        torch.save(train_tgt[mask_keep], tgt_path)

        block = (
            f"--- Remove top {pct}% (threshold u > {threshold:.6f}) ---\n"
            f"  Kept     : {n_keep:4d} molecules  |  MAE = {mae_kept:.4f}\n"
            f"  Removed  : {n_remove:4d} molecules  |  MAE = {mae_removed:.4f}\n"
            f"  MAE ratio (removed / kept) : {mae_removed / mae_kept:.2f}x\n"
            f"  Saved    : {os.path.basename(emb_path)}, "
            f"{os.path.basename(tgt_path)}\n"
        )
        print(block)
        report_lines.append(block)

    # ── 6. Save report ───────────────────────────────────────────────────────
    print("[6/6] Saving curation report …")
    report_text = "\n".join(report_lines)
    report_path = os.path.join(RESULTS_DIR, "phase3a_curation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
