"""
train_minimol_weighted_gsl.py
==============================
MiniMol with our architecture + soft weighting: trains SimpleGSLModel on
512-d MiniMol embeddings, down-weighting high-uncertainty samples using
aleatoric uncertainty from the trained MiniMol EvidentialGSLModel.

    loss = mean(exp(-gamma * u) * (y_hat - y)^2)

Sweeps over GAMMAS × SEEDS.

Usage:
    python train_minimol_weighted_gsl.py

Prerequisites:
    Run train_minimol_evidential_gsl.py first to produce evidential_gsl_minimol.pt
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
from torch.utils.data import DataLoader, Dataset

# ── Cross-directory imports ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GSL_DIR = os.path.join(PROJECT_ROOT, "gsl")
sys.path.insert(0, GSL_DIR)

from model_evidential_gsl import EvidentialGSLModel  # noqa: E402
from model_gsl import SimpleGSLModel  # noqa: E402
from train_gsl import GSLDataset, gsl_collate_fn  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "evidential_gsl_minimol.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
LR_PATIENCE = 5
EMBED_DIM = 512

GAMMAS = [0.0, 7.0, 10.0, 12.5, 15.0, 20.0]
SEEDS = list(range(10))

RDLogger.DisableLog("rdApp.warning")


# ── Dataset ──────────────────────────────────────────────────────────────────
class WeightedGSLDataset(Dataset):
    def __init__(self, embeddings, targets, smiles, uncertainties):
        assert embeddings.shape[0] == targets.shape[0] == len(smiles) == len(uncertainties)
        self.embeddings = embeddings
        self.targets = targets
        self.smiles = smiles
        self.uncertainties = uncertainties

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx], self.smiles[idx], self.uncertainties[idx]


def weighted_gsl_collate_fn(batch):
    embeddings, targets, smiles_list, u_vals = zip(*batch)
    embeddings = torch.stack(embeddings)
    targets = torch.stack(targets)
    u_vals = torch.stack(u_vals)

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles("C"), 2, nBits=1024)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fps.append(fp)

    n = len(fps)
    A = torch.zeros(n, n)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for j, s in enumerate(sims, start=i + 1):
            A[i, j] = s
            A[j, i] = s

    return embeddings, targets, A, u_vals


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
    print("  Loading EvidentialGSLModel(embed_dim=512) for uncertainty extraction …")
    model = EvidentialGSLModel(embed_dim=EMBED_DIM)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    n = train_emb.shape[0]
    loader = DataLoader(
        GSLDataset(train_emb, train_tgt_s, train_smiles),
        batch_size=n, shuffle=False, collate_fn=gsl_collate_fn,
    )

    with torch.no_grad():
        for X, _, A_ecfp in loader:
            X, A_ecfp = X.to(DEVICE), A_ecfp.to(DEVICE)
            _, (_, _, alpha, beta) = model(X, A_ecfp)

    u = (beta / (alpha - 1.0)).detach().cpu()
    print(f"  Uncertainty: min={u.min():.4f} max={u.max():.4f} mean={u.mean():.4f}")
    return u


# ── Train / Eval ─────────────────────────────────────────────────────────────
def train_one_epoch_weighted(model, loader, optimizer, gamma):
    model.train()
    total, n = 0.0, 0
    for X, y, A_ecfp, u in loader:
        X, y, A_ecfp, u = X.to(DEVICE), y.to(DEVICE), A_ecfp.to(DEVICE), u.to(DEVICE)
        preds = model(X, A_ecfp).squeeze(-1)
        loss = torch.mean(torch.exp(-gamma * u) * (preds - y) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / n


@torch.no_grad()
def evaluate_mse(model, loader):
    model.eval()
    total, n = 0.0, 0
    for X, y, A_ecfp, _ in loader:
        X, y, A_ecfp = X.to(DEVICE), y.to(DEVICE), A_ecfp.to(DEVICE)
        preds = model(X, A_ecfp).squeeze(-1)
        total += torch.mean((preds - y) ** 2).item() * X.size(0)
        n += X.size(0)
    return total / n


@torch.no_grad()
def test_metrics(model, loader, scaler):
    model.eval()
    preds_all, tgts_all = [], []
    for X, y, A_ecfp, _ in loader:
        X, A_ecfp = X.to(DEVICE), A_ecfp.to(DEVICE)
        preds_all.append(model(X, A_ecfp).squeeze(-1).cpu().numpy())
        tgts_all.append(y.numpy())

    p = scaler.inverse_transform(np.concatenate(preds_all).reshape(-1, 1)).flatten()
    t = scaler.inverse_transform(np.concatenate(tgts_all).reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(np.abs(p - t)))
    r = float(pearsonr(p, t)[0])
    rho = float(spearmanr(p, t)[0])
    return rmse, mae, r, rho


def run_single_seed(seed, gamma, train_ds, val_ds, test_ds, scaler):
    set_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=weighted_gsl_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=weighted_gsl_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=weighted_gsl_collate_fn)

    model = SimpleGSLModel(embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )

    best_val, best_state, no_improve = float("inf"), None, 0
    epoch_times = []
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        train_one_epoch_weighted(model, train_loader, optimizer, gamma)
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

    print("[1/3] Loading data …")
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

    print("[2/3] Extracting uncertainties …")
    u_train = extract_uncertainties(train_emb, train_tgt_s, smiles["train"])
    u_val = torch.zeros(val_emb.shape[0])
    u_test = torch.zeros(test_emb.shape[0])

    train_ds = WeightedGSLDataset(train_emb, train_tgt_s, smiles["train"], u_train)
    val_ds = WeightedGSLDataset(val_emb, val_tgt_s, smiles["val"], u_val)
    test_ds = WeightedGSLDataset(test_emb, test_tgt_s, smiles["test"], u_test)

    print(f"[3/3] Training sweep: {len(GAMMAS)} gammas × {len(SEEDS)} seeds …")
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
                f"    seed={seed:>4d} RMSE={m[0]:.4f} MAE={m[1]:.4f} "
                f"r={m[2]:.4f} rho={m[3]:.4f}"
            )
            fold.append(m)
            completed += 1
            avg_run_s = (time.perf_counter() - sweep_start) / completed
            eta_s = avg_run_s * (total_runs - completed)
            print(f"      [sweep ETA] remaining: {format_eta(eta_s)}")

        arr = np.array(fold)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        results.append({
            "gamma": gamma,
            "rmse_mean": means[0], "rmse_std": stds[0],
            "mae_mean": means[1], "mae_std": stds[1],
            "r_mean": means[2], "r_std": stds[2],
            "rho_mean": means[3], "rho_std": stds[3],
        })
        print(
            f"    mean RMSE={means[0]:.4f}±{stds[0]:.4f} "
            f"MAE={means[1]:.4f}±{stds[1]:.4f}"
        )

    best = min(results, key=lambda x: x["mae_mean"])
    lines = [
        "MiniMol + Soft-Weighting Report (GSL, Multi-Seed) — AqSolDB",
        "=============================================================",
        f"Seeds: {SEEDS} ({len(SEEDS)} runs per gamma)",
        f"Training set: {train_emb.shape[0]} molecules (full)",
        "",
        f"{'gamma':>7s}  {'RMSE':>15s}  {'MAE':>15s}  {'Pearson':>15s}  {'Spearman':>15s}",
        "-" * 80,
    ]
    for r in results:
        lines.append(
            f"{r['gamma']:>7.2f}  "
            f"{r['rmse_mean']:.4f}±{r['rmse_std']:.4f}  "
            f"{r['mae_mean']:.4f}±{r['mae_std']:.4f}  "
            f"{r['r_mean']:.4f}±{r['r_std']:.4f}  "
            f"{r['rho_mean']:.4f}±{r['rho_std']:.4f}"
        )
    lines += [
        "", "-" * 80,
        f"Best mean MAE at gamma={best['gamma']:.2f}: "
        f"{best['mae_mean']:.4f} ± {best['mae_std']:.4f}",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    report_path = os.path.join(RESULTS_DIR, "minimol_soft_weighting_gsl_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
