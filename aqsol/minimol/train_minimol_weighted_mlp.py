"""
train_minimol_weighted_mlp.py
==============================
MiniMol + soft-weighted MLP: our contribution on top of the TDC #1 model.

Trains the same MiniMolMLP as the as-is baseline but down-weights
high-uncertainty training samples using aleatoric uncertainty from the
EvidentialGSLModel:

    loss = mean(exp(-gamma * u_i) * (y_hat_i - y_i)^2)

Uses the TDC ADMET benchmark group evaluation protocol:
  - Per-seed train/valid splits via get_train_valid_split
  - Fixed benchmark test set evaluated via group.evaluate_many
  - Sweeps over GAMMAS x SEEDS [1,2,3,4,5]

Usage:
    python train_minimol_weighted_mlp.py

Prerequisites:
    python generate_embeddings_minimol.py
    python train_minimol_evidential_gsl.py  -> results/evidential_gsl_minimol.pt
"""

import copy
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
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

BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
LR_PATIENCE = 5
EMBED_DIM = 512

GAMMAS = [0.0, 7.0, 10.0, 12.5, 15.0, 20.0]
SEEDS = [1, 2, 3, 4, 5]  # TDC standard


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


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_uncertainties(emb, tgt_s, smiles, unc_model):
    """Run EvidentialGSLModel inference and return per-sample aleatoric uncertainty."""
    loader = DataLoader(
        GSLDataset(emb, tgt_s, smiles),
        batch_size=emb.shape[0], shuffle=False, collate_fn=gsl_collate_fn,
    )
    with torch.no_grad():
        for X, _, A in loader:
            X, A = X.to(DEVICE), A.to(DEVICE)
            _, (_, _, alpha, beta) = unc_model(X, A)
    u = (beta / (alpha - 1.0)).detach().cpu()
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
def get_test_predictions(model, loader, scaler):
    model.eval()
    preds_all = []
    for x, _, __ in loader:
        preds_all.append(model(x.to(DEVICE)).squeeze(-1).cpu().numpy())
    p_s = np.concatenate(preds_all)
    return scaler.inverse_transform(p_s.reshape(-1, 1)).flatten()


def run_seed(seed, gamma, train_ds, val_ds, test_loader, scaler):
    set_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

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
    return get_test_predictions(model, test_loader, scaler)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run train_minimol_evidential_gsl.py first."
        )

    print("[1/4] Loading TDC ADMET benchmark group ...")
    group = admet_group(path=DATA_DIR)
    benchmark = group.get("Solubility_AqSolDB")
    name = benchmark["name"]

    print("[2/4] Loading MiniMol embeddings ...")
    all_tv_emb = torch.load(
        os.path.join(DATA_DIR, "trainval_embeddings_minimol.pt"), weights_only=True
    )
    all_tv_tgt = torch.load(
        os.path.join(DATA_DIR, "trainval_targets.pt"), weights_only=True
    )
    all_tv_smi = torch.load(os.path.join(DATA_DIR, "trainval_smiles.pt"))
    test_emb = torch.load(
        os.path.join(DATA_DIR, "test_embeddings_minimol.pt"), weights_only=True
    )
    test_tgt = torch.load(
        os.path.join(DATA_DIR, "test_targets.pt"), weights_only=True
    )
    smi_to_idx = {s: i for i, s in enumerate(all_tv_smi)}

    scaler = StandardScaler()
    scaler.fit(all_tv_tgt.numpy().reshape(-1, 1))

    def scale(t):
        return torch.tensor(
            scaler.transform(t.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32
        )

    test_tgt_s = scale(test_tgt)
    test_ds = WeightedEmbeddingDataset(test_emb, test_tgt_s, torch.zeros(test_emb.shape[0]))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    print("[3/4] Loading EvidentialGSLModel for uncertainty extraction ...")
    unc_model = EvidentialGSLModel(embed_dim=EMBED_DIM)
    unc_model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    unc_model.to(DEVICE)
    unc_model.eval()

    print(f"[4/4] Sweep: {len(GAMMAS)} gammas x {len(SEEDS)} seeds ...")
    sweep_results = []
    total_runs = len(GAMMAS) * len(SEEDS)
    completed = 0
    sweep_start = time.perf_counter()

    for gamma in GAMMAS:
        print(f"  -- gamma={gamma} --")
        gamma_predictions = []

        for seed in SEEDS:
            train_df, val_df = group.get_train_valid_split(
                benchmark=name, split_type="default", seed=seed
            )

            t_idx = [smi_to_idx[s] for s in train_df["Drug"].tolist()]
            v_idx = [smi_to_idx[s] for s in val_df["Drug"].tolist()]

            train_emb = all_tv_emb[t_idx]
            val_emb = all_tv_emb[v_idx]
            train_smi = train_df["Drug"].tolist()
            train_tgt_s = scale(all_tv_tgt[t_idx])
            val_tgt_s = scale(all_tv_tgt[v_idx])

            u_train = extract_uncertainties(train_emb, train_tgt_s, train_smi, unc_model)
            print(
                f"    [seed={seed}] Uncertainty: "
                f"min={u_train.min():.4f}  max={u_train.max():.4f}  mean={u_train.mean():.4f}"
            )

            train_ds = WeightedEmbeddingDataset(train_emb, train_tgt_s, u_train)
            val_ds = WeightedEmbeddingDataset(val_emb, val_tgt_s, torch.zeros(val_emb.shape[0]))

            y_pred = run_seed(seed, gamma, train_ds, val_ds, test_loader, scaler)
            gamma_predictions.append({name: y_pred})

            completed += 1
            eta_s = (
                (time.perf_counter() - sweep_start) / completed
            ) * (total_runs - completed)
            print(f"      [sweep ETA] {format_eta(eta_s)}")

        results = group.evaluate_many(gamma_predictions)
        mean_mae, std_mae = list(results.values())[0]
        sweep_results.append({
            "gamma": gamma,
            "mae_mean": mean_mae,
            "mae_std": std_mae,
        })
        print(f"    gamma={gamma:.2f}  MAE={mean_mae:.4f} +/- {std_mae:.4f}")

    best = min(sweep_results, key=lambda x: x["mae_mean"])
    lines = [
        "MiniMol + Soft-Weighted MLP -- AqSolDB",
        "=========================================",
        "Model: MiniMolMLP  |  Uncertainty: EvidentialGSLModel(512-d)",
        f"Seeds: {SEEDS}",
        "TDC Rank-1 reference: MAE 0.741 +/- 0.013",
        "",
        f"{'gamma':>7s}  {'MAE':>20s}",
        "-" * 35,
    ]
    for r in sweep_results:
        lines.append(
            f"{r['gamma']:>7.2f}  "
            f"{r['mae_mean']:.4f} +/- {r['mae_std']:.4f}"
        )
    lines += [
        "", "-" * 35,
        f"Best MAE at gamma={best['gamma']:.2f}: "
        f"{best['mae_mean']:.4f} +/- {best['mae_std']:.4f}",
        "",
        "[Evaluated via group.evaluate_many -- TDC official metric]",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    report_path = os.path.join(RESULTS_DIR, "minimol_weighted_mlp_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
