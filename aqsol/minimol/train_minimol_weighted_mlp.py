"""
train_minimol_weighted_mlp.py
==============================
MiniMol + soft-weighted MLP: our contribution on top of the TDC #1 model.

Trains the same MiniMolMLP as the as-is baseline but down-weights
high-uncertainty training samples using aleatoric uncertainty from the
EvidentialGSLModel:

    loss = mean(exp(-gamma * u_i) * (y_hat_i - y_i)^2)

Sweeps over GAMMAS x SEEDS [1,2,3,4,5] (TDC standard).

Usage:
    python train_minimol_weighted_mlp.py

Prerequisites:
    python train_minimol_evidential_gsl.py  -> results/evidential_gsl_minimol.pt
"""

import copy
import os
import random
import sys
import time

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GSL_DIR = os.path.join(PROJECT_ROOT, "gsl")
sys.path.insert(0, GSL_DIR)

from gsl_utils import GSLDataset, gsl_collate_fn  # noqa: E402
from model_evidential_gsl import EvidentialGSLModel  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "evidential_gsl_minimol.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
LR_PATIENCE = 5
EMBED_DIM = 512

GAMMAS = [0.0, 7.0, 10.0, 12.5, 15.0, 20.0]
SEEDS = [1, 2, 3, 4, 5]  # TDC standard

RDLogger.DisableLog("rdApp.warning")


# ── Dataset ──────────────────────────────────────────────────────────────────
class WeightedEmbeddingDataset(Dataset):
    def __init__(self, embeddings, targets, uncertainties):
        self.embeddings = embeddings
        self.targets = targets
        self.uncertainties = uncertainties

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx], self.uncertainties[idx]


# ── Model ─────────────────────────────────────────────────────────────────────
class MiniMolMLP(nn.Module):
    def __init__(self, input_dim: int = EMBED_DIM, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


# ── Helpers ──────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_tensors(split: str):
    emb = torch.load(
        os.path.join(DATA_DIR, f"{split}_embeddings_minimol.pt"), weights_only=True
    )
    tgt = torch.load(os.path.join(DATA_DIR, f"{split}_targets.pt"), weights_only=True)
    return emb, tgt


def load_smiles():
    return {
        s: torch.load(os.path.join(DATA_DIR, f"{s}_smiles.pt"))
        for s in ("train", "val", "test")
    }


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_uncertainties(train_emb, train_tgt_s, train_smiles):
    print("  Loading EvidentialGSLModel(embed_dim=512) for uncertainty extraction ...")
    model = EvidentialGSLModel(embed_dim=EMBED_DIM)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    loader = DataLoader(
        GSLDataset(train_emb, train_tgt_s, train_smiles),
        batch_size=train_emb.shape[0], shuffle=False, collate_fn=gsl_collate_fn,
    )
    with torch.no_grad():
        for X, _, A in loader:
            X, A = X.to(DEVICE), A.to(DEVICE)
            _, (_, _, alpha, beta) = model(X, A)

    u = (beta / (alpha - 1.0)).detach().cpu()
    print(f"  Uncertainty: min={u.min():.4f}  max={u.max():.4f}  mean={u.mean():.4f}")
    return u


# ── Train / Eval ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, gamma):
    model.train()
    total, n = 0.0, 0
    for x, y, u in loader:
        x, y, u = x.to(DEVICE), y.to(DEVICE), u.to(DEVICE)
        preds = model(x).squeeze(-1)
        loss = torch.mean(torch.exp(-gamma * u) * (preds - y) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def evaluate_mse(model, loader):
    model.eval()
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).squeeze(-1)
        total += torch.mean((preds - y) ** 2).item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def test_metrics(model, loader, scaler):
    model.eval()
    preds_all, tgts_all = [], []
    for x, y, _ in loader:
        preds_all.append(model(x.to(DEVICE)).squeeze(-1).cpu().numpy())
        tgts_all.append(y.numpy())

    p = scaler.inverse_transform(np.concatenate(preds_all).reshape(-1, 1)).flatten()
    t = scaler.inverse_transform(np.concatenate(tgts_all).reshape(-1, 1)).flatten()

    return (
        float(np.sqrt(np.mean((p - t) ** 2))),
        float(np.mean(np.abs(p - t))),
        float(pearsonr(p, t)[0]),
        float(spearmanr(p, t)[0]),
    )


def run_single_seed(seed, gamma, train_ds, val_ds, test_ds, scaler):
    set_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = MiniMolMLP(input_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )

    best_val, best_state, no_improve = float("inf"), None, 0
    epoch_times = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        train_one_epoch(model, train_loader, optimizer, gamma)
        val_loss = evaluate_mse(model, val_loader)
        scheduler.step(val_loss)

        epoch_times.append(time.perf_counter() - t0)
        eta_s = (sum(epoch_times) / len(epoch_times)) * (MAX_EPOCHS - epoch)
        print(
            f"      [seed={seed}] epoch {epoch:3d}/{MAX_EPOCHS}  "
            f"val={val_loss:.6f}  ETA: {format_eta(eta_s)}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return test_metrics(model, test_loader, scaler)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run train_minimol_evidential_gsl.py first."
        )

    print("[1/3] Loading data ...")
    train_emb, train_tgt = load_tensors("train")
    val_emb, val_tgt = load_tensors("val")
    test_emb, test_tgt = load_tensors("test")
    smiles = load_smiles()

    scaler = StandardScaler()
    scaler.fit(train_tgt.numpy().reshape(-1, 1))

    def scale(t):
        return torch.tensor(
            scaler.transform(t.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32
        )

    train_tgt_s = scale(train_tgt)
    val_tgt_s = scale(val_tgt)
    test_tgt_s = scale(test_tgt)

    print("[2/3] Extracting uncertainties from EvidentialGSL ...")
    u_train = extract_uncertainties(train_emb, train_tgt_s, smiles["train"])
    u_zero = torch.zeros

    train_ds = WeightedEmbeddingDataset(train_emb, train_tgt_s, u_train)
    val_ds = WeightedEmbeddingDataset(val_emb, val_tgt_s, torch.zeros(val_emb.shape[0]))
    test_ds = WeightedEmbeddingDataset(test_emb, test_tgt_s, torch.zeros(test_emb.shape[0]))

    print(f"[3/3] Sweep: {len(GAMMAS)} gammas x {len(SEEDS)} seeds ...")
    results = []
    total_runs = len(GAMMAS) * len(SEEDS)
    completed = 0
    sweep_start = time.perf_counter()

    for gamma in GAMMAS:
        print(f"  -- gamma={gamma} --")
        fold = []
        for seed in SEEDS:
            m = run_single_seed(seed, gamma, train_ds, val_ds, test_ds, scaler)
            print(
                f"    seed={seed}  RMSE={m[0]:.4f}  MAE={m[1]:.4f}  "
                f"r={m[2]:.4f}  rho={m[3]:.4f}"
            )
            fold.append(m)
            completed += 1
            eta_s = ((time.perf_counter() - sweep_start) / completed) * (total_runs - completed)
            print(f"      [sweep ETA] {format_eta(eta_s)}")

        arr = np.array(fold)
        means, stds = arr.mean(0), arr.std(0)
        results.append({
            "gamma": gamma,
            "rmse_mean": means[0], "rmse_std": stds[0],
            "mae_mean": means[1], "mae_std": stds[1],
            "r_mean": means[2], "r_std": stds[2],
            "rho_mean": means[3], "rho_std": stds[3],
        })
        print(
            f"    mean  RMSE={means[0]:.4f}+/-{stds[0]:.4f}  "
            f"MAE={means[1]:.4f}+/-{stds[1]:.4f}"
        )

    best = min(results, key=lambda x: x["mae_mean"])
    lines = [
        "MiniMol + Soft-Weighted MLP -- AqSolDB",
        "=========================================",
        "Model: MiniMolMLP  |  Uncertainty: EvidentialGSLModel(512-d)",
        f"Seeds: {SEEDS}",
        "",
        f"{'gamma':>7s}  {'RMSE':>15s}  {'MAE':>15s}  {'Pearson':>15s}  {'Spearman':>15s}",
        "-" * 80,
    ]
    for r in results:
        lines.append(
            f"{r['gamma']:>7.2f}  "
            f"{r['rmse_mean']:.4f}+/-{r['rmse_std']:.4f}  "
            f"{r['mae_mean']:.4f}+/-{r['mae_std']:.4f}  "
            f"{r['r_mean']:.4f}+/-{r['r_std']:.4f}  "
            f"{r['rho_mean']:.4f}+/-{r['rho_std']:.4f}"
        )
    lines += [
        "", "-" * 80,
        f"Best MAE at gamma={best['gamma']:.2f}: "
        f"{best['mae_mean']:.4f} +/- {best['mae_std']:.4f}",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    report_path = os.path.join(RESULTS_DIR, "minimol_weighted_mlp_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
