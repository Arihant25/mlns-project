"""
create_golden_dataset.py
========================
Ensemble Golden Curation Index (GCI) — train 10 Evidential GSL seeds,
compute per-molecule aleatoric uncertainty and neighbourhood-label
discrepancy from each, then use the consensus product to surgically
remove only true assay artifacts (3-sigma outliers).

No existing scripts are modified.

Usage:
    python create_golden_dataset.py
"""

import os
import sys
import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ── Cross-directory import: ErrorScaledEvidentialLoss lives in mlp/ ──────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MLP_DIR = os.path.join(PROJECT_ROOT, "mlp")
sys.path.insert(0, MLP_DIR)
from loss_evidential import ErrorScaledEvidentialLoss        # noqa: E402

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


# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyper-params (same as train_evidential_gsl.py)
BATCH_SIZE   = 128
LR           = 1e-3
MAX_EPOCHS   = 100
PATIENCE     = 10
LR_PATIENCE  = 5
LOSS_COEFF   = 0.1
AUX_WEIGHT   = 0.5

SEEDS = list(range(10))   # 10 ensemble members


# ── Helpers ──────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Train / Eval (replicated from train_evidential_gsl.py) ──────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total, n = 0.0, 0
    for X, y, A_ecfp in loader:
        X, y, A_ecfp = X.to(DEVICE), y.to(DEVICE), A_ecfp.to(DEVICE)
        (mu_0, v_0, a_0, b_0), (mu, v, a, b) = model(X, A_ecfp)
        loss = criterion(y, mu, v, a, b) + AUX_WEIGHT * criterion(y, mu_0, v_0, a_0, b_0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, n = 0.0, 0
    for X, y, A_ecfp in loader:
        X, y, A_ecfp = X.to(DEVICE), y.to(DEVICE), A_ecfp.to(DEVICE)
        (mu_0, v_0, a_0, b_0), (mu, v, a, b) = model(X, A_ecfp)
        loss = criterion(y, mu, v, a, b) + AUX_WEIGHT * criterion(y, mu_0, v_0, a_0, b_0)
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / n


def train_single_seed(seed, train_loader, val_loader, criterion):
    """Train one EvidentialGSLModel; return best state_dict."""
    set_seed(seed)
    model = EvidentialGSLModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )

    best_val, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss = evaluate(model, val_loader, criterion)
        scheduler.step(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"    [seed={seed}] stopped at epoch {epoch:3d}  "
                  f"(best val={best_val:.4f})")
            break
    else:
        print(f"    [seed={seed}] finished {MAX_EPOCHS} epochs  "
              f"(best val={best_val:.4f})")

    return best_state


# ── Extract u and delta from one trained model ──────────────────────────────
@torch.no_grad()
def extract_u_delta(state_dict, full_loader, y_scaled):
    """
    Full-batch inference → aleatoric uncertainty u and neighbourhood
    label discrepancy delta.
    """
    model = EvidentialGSLModel().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    for X, _, A_ecfp in full_loader:
        X, A_ecfp = X.to(DEVICE), A_ecfp.to(DEVICE)

        # Forward pass → final NIG params
        _, (mu, v, alpha, beta) = model(X, A_ecfp)
        u = (beta / (alpha - 1.0)).cpu()

        # Extract learned adjacency WITHOUT modifying the model
        A_raw = model.graph_maker(X, A_ecfp)  # (N, N)

        # Row-normalise
        row_sum = A_raw.sum(dim=1, keepdim=True) + 1e-8
        A_norm = A_raw / row_sum

        # Expected local label from neighbours
        y_dev = y_scaled.to(DEVICE)
        y_bar = torch.matmul(A_norm, y_dev.unsqueeze(1)).squeeze()  # (N,)
        delta = torch.abs(y_dev - y_bar).cpu()

    return u.numpy(), delta.numpy()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load data
    print("[1/5] Loading data …")
    train_emb, train_tgt = load_tensors("train")
    val_emb, val_tgt     = load_tensors("val")
    smiles = load_smiles()

    scaler = fit_scaler(train_tgt.numpy())
    train_tgt_s = scale_targets(train_tgt.numpy(), scaler)
    val_tgt_s   = scale_targets(val_tgt.numpy(),   scaler)

    n_train = train_emb.shape[0]
    print(f"       Train: {n_train} molecules")

    # Loaders for training
    train_loader = DataLoader(
        GSLDataset(train_emb, train_tgt_s, smiles["train"]),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=gsl_collate_fn,
    )
    val_loader = DataLoader(
        GSLDataset(val_emb, val_tgt_s, smiles["val"]),
        batch_size=BATCH_SIZE, collate_fn=gsl_collate_fn,
    )

    # Full-batch loader for inference (no shuffle)
    full_loader = DataLoader(
        GSLDataset(train_emb, train_tgt_s, smiles["train"]),
        batch_size=n_train, shuffle=False, collate_fn=gsl_collate_fn,
    )

    criterion = ErrorScaledEvidentialLoss(coeff=LOSS_COEFF)

    # 2. Train 10 seeds and extract u, delta from each
    print(f"[2/5] Training {len(SEEDS)} Evidential GSL seeds …")
    all_u, all_delta = [], []

    for seed in SEEDS:
        print(f"  ── Seed {seed} ──")
        best_state = train_single_seed(seed, train_loader, val_loader, criterion)

        u, delta = extract_u_delta(best_state, full_loader, train_tgt_s)
        all_u.append(u)
        all_delta.append(delta)
        print(f"    u: [{u.min():.4f}, {u.max():.4f}]  "
              f"delta: [{delta.min():.4f}, {delta.max():.4f}]")

    # 3. Consensus & GCI
    print("[3/5] Computing Ensemble GCI …")
    u_consensus     = np.mean(all_u, axis=0)        # (N,)
    delta_consensus = np.mean(all_delta, axis=0)     # (N,)

    u_norm     = MinMaxScaler().fit_transform(u_consensus.reshape(-1, 1)).flatten()
    delta_norm = MinMaxScaler().fit_transform(delta_consensus.reshape(-1, 1)).flatten()

    gci = u_norm * delta_norm

    # 4. Statistical thresholding (3-sigma)
    mean_gci = np.mean(gci)
    std_gci  = np.std(gci)
    tau      = mean_gci + 3.0 * std_gci

    keep_mask = gci <= tau
    n_kept    = int(keep_mask.sum())
    n_removed = n_train - n_kept

    print(f"       GCI:  mean={mean_gci:.6f}  std={std_gci:.6f}  τ={tau:.6f}")
    print(f"       Kept: {n_kept}  |  Removed: {n_removed}")

    # 5. Filter & save (original unscaled tensors)
    print("[4/5] Saving golden dataset …")
    mask_t = torch.tensor(keep_mask)
    golden_emb = train_emb[mask_t]
    golden_tgt = train_tgt[mask_t]

    emb_path = os.path.join(DATA_DIR, "train_embeddings_golden.pt")
    tgt_path = os.path.join(DATA_DIR, "train_targets_golden.pt")
    torch.save(golden_emb, emb_path)
    torch.save(golden_tgt, tgt_path)
    print(f"       {emb_path}  ({golden_emb.shape})")
    print(f"       {tgt_path}  ({golden_tgt.shape})")

    # 6. Report
    print("[5/5] Writing report …")
    lines = [
        "Golden Dataset Curation Report",
        "==============================",
        f"Ensemble seeds : {SEEDS}",
        f"Total training : {n_train}",
        f"",
        f"GCI statistics:",
        f"  mean = {mean_gci:.6f}",
        f"  std  = {std_gci:.6f}",
        f"  τ    = mean + 3σ = {tau:.6f}",
        f"",
        f"Molecules kept    : {n_kept}  ({100*n_kept/n_train:.1f}%)",
        f"Molecules removed : {n_removed}  ({100*n_removed/n_train:.1f}%)",
        f"",
        f"Saved: train_embeddings_golden.pt, train_targets_golden.pt",
    ]
    report = "\n".join(lines)
    print(f"\n{report}")

    report_path = os.path.join(RESULTS_DIR, "golden_dataset_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()

