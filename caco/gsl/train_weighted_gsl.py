"""
train_weighted_gsl.py
=====================
Phase 3C extension - Soft-weighted training for the baseline GCN/GSL model.

Trains SimpleGSLModel on 100% of training data while down-weighting per-sample
squared error using aleatoric uncertainty from EvidentialGSLModel:

    loss = mean(exp(-gamma * u) * (y_hat - y)^2)

Usage:
    python train_weighted_gsl.py
"""

import copy
import os
import random
import time

import numpy as np
import torch
from model_evidential_gsl import EvidentialGSLModel
from model_gsl import SimpleGSLModel
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from train_gsl import GSLDataset, gsl_collate_fn, load_smiles

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "evidential_gsl.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
LR_PATIENCE = 5

GAMMAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
SEEDS = [42, 123, 456, 789, 1024]


# Keep logs concise during repeated fingerprint generation.
RDLogger.DisableLog("rdApp.warning")


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class WeightedGSLDataset(Dataset):
    """Returns (embedding, scaled_target, smiles, uncertainty_value)."""

    def __init__(self, embeddings, targets, smiles, uncertainties):
        assert (
            embeddings.shape[0] == targets.shape[0] == len(smiles) == len(uncertainties)
        )
        self.embeddings = embeddings
        self.targets = targets
        self.smiles = smiles
        self.uncertainties = uncertainties

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return (
            self.embeddings[idx],
            self.targets[idx],
            self.smiles[idx],
            self.uncertainties[idx],
        )


def weighted_gsl_collate_fn(batch):
    embeddings, targets, smiles_list, u_vals = zip(*batch)
    embeddings = torch.stack(embeddings)
    targets = torch.stack(targets)
    u_vals = torch.stack(u_vals)

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles("C"), 2, nBits=1024
            )
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fps.append(fp)

    n = len(fps)
    A = torch.zeros(n, n)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        for j, s in enumerate(sims, start=i + 1):
            A[i, j] = s
            A[j, i] = s

    return embeddings, targets, A, u_vals


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_tensors(split: str):
    emb = torch.load(
        os.path.join(DATA_DIR, f"{split}_embeddings.pt"), weights_only=True
    )
    tgt = torch.load(os.path.join(DATA_DIR, f"{split}_targets.pt"), weights_only=True)
    return emb, tgt


def fit_scaler(targets: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(targets.reshape(-1, 1))
    return scaler


def scale_targets(targets: np.ndarray, scaler: StandardScaler) -> torch.Tensor:
    return torch.tensor(
        scaler.transform(targets.reshape(-1, 1)).flatten(), dtype=torch.float32
    )


def extract_uncertainties(train_emb, train_tgt_s, train_smiles):
    model = EvidentialGSLModel()
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    n = train_emb.shape[0]
    full_loader = DataLoader(
        GSLDataset(train_emb, train_tgt_s, train_smiles),
        batch_size=n,
        shuffle=False,
        collate_fn=gsl_collate_fn,
    )

    with torch.no_grad():
        for X, _, A_ecfp in full_loader:
            X, A_ecfp = X.to(DEVICE), A_ecfp.to(DEVICE)
            _, (_, _, alpha, beta) = model(X, A_ecfp)

    u = (beta / (alpha - 1.0)).detach().cpu()
    print(
        f"  Uncertainty stats -> min={u.min():.4f}, max={u.max():.4f}, mean={u.mean():.4f}"
    )
    return u


def train_one_epoch_weighted(model, loader, optimizer, gamma):
    model.train()
    total_loss, n = 0.0, 0

    for X, y, A_ecfp, u in loader:
        X, y, A_ecfp, u = X.to(DEVICE), y.to(DEVICE), A_ecfp.to(DEVICE), u.to(DEVICE)
        preds = model(X, A_ecfp).squeeze(-1)

        errors = (preds - y) ** 2
        weights = torch.exp(-gamma * u)
        loss = torch.mean(weights * errors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        n += X.size(0)

    return total_loss / n


@torch.no_grad()
def evaluate_mse(model, loader):
    model.eval()
    total_loss, n = 0.0, 0

    for X, y, A_ecfp, _ in loader:
        X, y, A_ecfp = X.to(DEVICE), y.to(DEVICE), A_ecfp.to(DEVICE)
        preds = model(X, A_ecfp).squeeze(-1)
        loss = torch.mean((preds - y) ** 2)
        total_loss += loss.item() * X.size(0)
        n += X.size(0)

    return total_loss / n


@torch.no_grad()
def test_metrics(model, loader, scaler):
    model.eval()
    preds_all, tgts_all = [], []

    for X, y, A_ecfp, _ in loader:
        X, A_ecfp = X.to(DEVICE), A_ecfp.to(DEVICE)
        preds = model(X, A_ecfp).squeeze(-1).cpu().numpy()
        preds_all.append(preds)
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

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=weighted_gsl_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=weighted_gsl_collate_fn
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=weighted_gsl_collate_fn,
    )

    model = SimpleGSLModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )

    best_val, best_state, no_improve = float("inf"), None, 0
    epoch_times = []

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()
        train_one_epoch_weighted(model, train_loader, optimizer, gamma)
        val_loss = evaluate_mse(model, val_loader)
        scheduler.step(val_loss)

        epoch_times.append(time.perf_counter() - epoch_start)
        avg_epoch_s = sum(epoch_times) / len(epoch_times)
        eta_s = avg_epoch_s * (MAX_EPOCHS - epoch)
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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[1/3] Loading data ...")
    train_emb, train_tgt = load_tensors("train")
    val_emb, val_tgt = load_tensors("val")
    test_emb, test_tgt = load_tensors("test")
    smiles = load_smiles()

    scaler = fit_scaler(train_tgt.numpy())
    train_tgt_s = scale_targets(train_tgt.numpy(), scaler)
    val_tgt_s = scale_targets(val_tgt.numpy(), scaler)
    test_tgt_s = scale_targets(test_tgt.numpy(), scaler)

    print("[2/3] Extracting uncertainties from Evidential GSL ...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing evidential model at {MODEL_PATH}. Run train_evidential_gsl.py first."
        )

    u_train = extract_uncertainties(train_emb, train_tgt_s, smiles["train"])
    u_val = torch.zeros(val_emb.shape[0])
    u_test = torch.zeros(test_emb.shape[0])

    train_ds = WeightedGSLDataset(train_emb, train_tgt_s, smiles["train"], u_train)
    val_ds = WeightedGSLDataset(val_emb, val_tgt_s, smiles["val"], u_val)
    test_ds = WeightedGSLDataset(test_emb, test_tgt_s, smiles["test"], u_test)

    print(f"[3/3] Training sweep: {len(GAMMAS)} gammas x {len(SEEDS)} seeds")
    results = []
    total_runs = len(GAMMAS) * len(SEEDS)
    completed_runs = 0
    sweep_start = time.perf_counter()

    for gamma in GAMMAS:
        print(f"  -- gamma={gamma} --")
        fold = []
        for seed in SEEDS:
            m = run_single_seed(seed, gamma, train_ds, val_ds, test_ds, scaler)
            print(
                f"    seed={seed:>4d} RMSE={m[0]:.4f} MAE={m[1]:.4f} r={m[2]:.4f} rho={m[3]:.4f}"
            )
            fold.append(m)
            completed_runs += 1
            avg_run_s = (time.perf_counter() - sweep_start) / completed_runs
            eta_s = avg_run_s * (total_runs - completed_runs)
            print(f"      [sweep ETA] remaining: {format_eta(eta_s)}")

        arr = np.array(fold)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)

        results.append(
            {
                "gamma": gamma,
                "rmse_mean": means[0],
                "rmse_std": stds[0],
                "mae_mean": means[1],
                "mae_std": stds[1],
                "r_mean": means[2],
                "r_std": stds[2],
                "rho_mean": means[3],
                "rho_std": stds[3],
            }
        )

        print(
            f"    mean RMSE={means[0]:.4f}+/-{stds[0]:.4f} "
            f"MAE={means[1]:.4f}+/-{stds[1]:.4f} "
            f"r={means[2]:.4f}+/-{stds[2]:.4f} "
            f"rho={means[3]:.4f}+/-{stds[3]:.4f}"
        )

    best = min(results, key=lambda x: x["mae_mean"])

    lines = [
        "Phase 3C - Soft-Weighting Report (GCN/GSL, Multi-Seed)",
        "=====================================================",
        f"Seeds: {SEEDS} ({len(SEEDS)} runs per gamma)",
        f"Training set: {train_emb.shape[0]} molecules (full, no removal)",
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

    lines.extend(
        [
            "",
            "-" * 80,
            f"Best mean MAE at gamma={best['gamma']:.2f}: {best['mae_mean']:.4f} +/- {best['mae_std']:.4f}",
        ]
    )

    report = "\n".join(lines)
    print("\n" + report)

    report_path = os.path.join(RESULTS_DIR, "phase3c_soft_weighting_gcn_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
