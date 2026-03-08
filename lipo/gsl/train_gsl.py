"""
train_gsl.py
============
Phase 1B — Train the SimpleGSLModel on frozen MolFormer-XL embeddings
with dynamically computed ECFP Tanimoto similarity graphs.

Usage:
    python train_gsl.py
"""

import os
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from tdc.single_pred import ADME

from model_gsl import SimpleGSLModel


# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE   = 128
LR           = 1e-3
MAX_EPOCHS   = 100
PATIENCE     = 10        # early-stopping patience
LR_PATIENCE  = 5         # ReduceLROnPlateau patience


# ── Dataset ──────────────────────────────────────────────────────────────────
class GSLDataset(Dataset):
    """
    Wraps pre-computed embeddings + targets and also returns the raw SMILES
    string (needed by the custom collate function for ECFP computation).
    """

    def __init__(self,
                 embeddings: torch.Tensor,
                 targets: torch.Tensor,
                 smiles: list[str]):
        assert embeddings.shape[0] == targets.shape[0] == len(smiles)
        self.embeddings = embeddings
        self.targets = targets
        self.smiles = smiles

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.targets[idx], self.smiles[idx]


# ── ECFP Collate Function ───────────────────────────────────────────────────
def _compute_ecfp_tanimoto(smiles_list: list[str]) -> torch.Tensor:
    """
    Compute the pairwise Tanimoto similarity matrix from Morgan (ECFP)
    fingerprints (radius=2, 1024 bits).

    Returns
    -------
    A_ecfp : torch.Tensor, shape (B, B)
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Fallback: zero fingerprint for unparseable SMILES
            fp = AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles("C"), 2, nBits=1024
            )
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fps.append(fp)

    n = len(fps)
    A = torch.zeros(n, n)
    for i in range(n):
        # BulkTanimotoSimilarity is vectorised in C++ — much faster than loop
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for j, s in enumerate(sims, start=i + 1):
            A[i, j] = s
            A[j, i] = s
    return A


def gsl_collate_fn(batch):
    """
    Custom collate that additionally computes the ECFP Tanimoto similarity
    matrix for the batch.

    Returns
    -------
    embeddings : (B, 768)
    targets    : (B,)
    A_ecfp     : (B, B)
    """
    embeddings, targets, smiles_list = zip(*batch)
    embeddings = torch.stack(embeddings)
    targets = torch.stack(targets)
    A_ecfp = _compute_ecfp_tanimoto(list(smiles_list))
    return embeddings, targets, A_ecfp


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_tensors(split: str):
    """Load pre-computed embedding and target tensors."""
    emb = torch.load(os.path.join(DATA_DIR, f"{split}_embeddings.pt"),
                     weights_only=True)
    tgt = torch.load(os.path.join(DATA_DIR, f"{split}_targets.pt"),
                     weights_only=True)
    return emb, tgt


def load_smiles() -> dict[str, list[str]]:
    """
    Reload SMILES from the cached TDC dataset and return the same scaffold
    split used during embedding generation.
    """
    data = ADME(name="Lipophilicity_AstraZeneca")
    split = data.get_split(method="scaffold")
    return {
        "train": split["train"]["Drug"].tolist(),
        "val":   split["valid"]["Drug"].tolist(),
        "test":  split["test"]["Drug"].tolist(),
    }


def fit_scaler(train_targets: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_targets.reshape(-1, 1))
    return scaler


def scale_targets(targets: np.ndarray, scaler: StandardScaler) -> torch.Tensor:
    scaled = scaler.transform(targets.reshape(-1, 1)).flatten()
    return torch.tensor(scaled, dtype=torch.float32)


# ── Training / Evaluation ───────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    n_samples = 0
    for X, y, A_ecfp in loader:
        X, y, A_ecfp = X.to(DEVICE), y.to(DEVICE), A_ecfp.to(DEVICE)
        preds = model(X, A_ecfp).squeeze(-1)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        n_samples += X.size(0)
    return running_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    for X, y, A_ecfp in loader:
        X, y, A_ecfp = X.to(DEVICE), y.to(DEVICE), A_ecfp.to(DEVICE)
        preds = model(X, A_ecfp).squeeze(-1)
        loss = criterion(preds, y)
        running_loss += loss.item() * X.size(0)
        n_samples += X.size(0)
    return running_loss / n_samples


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load embeddings, targets, and SMILES
    print("[1/5] Loading pre-computed embeddings and SMILES …")
    train_emb, train_tgt = load_tensors("train")
    val_emb,   val_tgt   = load_tensors("val")
    test_emb,  test_tgt  = load_tensors("test")
    smiles = load_smiles()
    print(f"       Train: {train_emb.shape[0]}, Val: {val_emb.shape[0]}, "
          f"Test: {test_emb.shape[0]}")

    # 2. Fit scaler on training targets only
    print("[2/5] Fitting StandardScaler on training targets …")
    scaler = fit_scaler(train_tgt.numpy())
    train_tgt_s = scale_targets(train_tgt.numpy(), scaler)
    val_tgt_s   = scale_targets(val_tgt.numpy(),   scaler)
    test_tgt_s  = scale_targets(test_tgt.numpy(),  scaler)

    # 3. Build DataLoaders with custom collate
    train_loader = DataLoader(
        GSLDataset(train_emb, train_tgt_s, smiles["train"]),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=gsl_collate_fn,
    )
    val_loader = DataLoader(
        GSLDataset(val_emb, val_tgt_s, smiles["val"]),
        batch_size=BATCH_SIZE, collate_fn=gsl_collate_fn,
    )
    test_loader = DataLoader(
        GSLDataset(test_emb, test_tgt_s, smiles["test"]),
        batch_size=BATCH_SIZE, collate_fn=gsl_collate_fn,
    )

    # 4. Initialise model, optimiser, scheduler, loss
    print("[3/5] Initialising SimpleGSLModel …")
    model = SimpleGSLModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )
    criterion = nn.MSELoss()
    print(f"       Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Training loop with early stopping
    print("[4/5] Training …")
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss   = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{MAX_EPOCHS}  |  "
              f"Train Loss: {train_loss:.6f}  |  "
              f"Val Loss: {val_loss:.6f}  |  "
              f"LR: {lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"  ⇢ Early stopping triggered at epoch {epoch}")
            break

    # Save best model
    model_path = os.path.join(RESULTS_DIR, "gsl_model.pt")
    torch.save(best_state, model_path)
    print(f"  Best model saved → {model_path}")

    # 6. Test evaluation
    print("[5/5] Evaluating on test set …")
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y, A_ecfp in test_loader:
            X, A_ecfp = X.to(DEVICE), A_ecfp.to(DEVICE)
            preds = model(X, A_ecfp).squeeze(-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())

    preds_scaled   = np.concatenate(all_preds)
    targets_scaled = np.concatenate(all_targets)

    # Inverse-transform to original scale
    preds_orig   = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    targets_orig = scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    # Compute metrics
    rmse     = float(np.sqrt(np.mean((preds_orig - targets_orig) ** 2)))
    mae      = float(np.mean(np.abs(preds_orig - targets_orig)))
    pearson  = float(pearsonr(preds_orig, targets_orig)[0])
    spearman_r = float(spearmanr(preds_orig, targets_orig)[0])

    metrics_text = (
        "Phase 1B — GSL Model Test Metrics\n"
        "==================================\n"
        f"RMSE              : {rmse:.4f}\n"
        f"MAE               : {mae:.4f}\n"
        f"Pearson  (r)      : {pearson:.4f}\n"
        f"Spearman (ρ)      : {spearman_r:.4f}\n"
        f"\nModel weights     : {model_path}\n"
    )
    print("\n" + metrics_text)

    metrics_path = os.path.join(RESULTS_DIR, "phase1b_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(metrics_text)
    print(f"Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
