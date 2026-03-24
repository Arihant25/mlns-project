"""
train_minimol_mlp.py
====================
MiniMol "as-is" baseline: train a simple MLP on 512-d MiniMol embeddings
using the TDC ADMET benchmark group evaluation protocol.

Reproduces the TDC ADMET leaderboard Rank-1 approach for AqSolDB
(MAE 0.741 +/- 0.013). Runs 5 independent seeds [1,2,3,4,5] per TDC
evaluation protocol with per-seed train/valid splits and a fixed test set.

Architecture: 512 -> 256 -> GELU -> Dropout -> 128 -> GELU -> Dropout -> 1

Usage:
    python train_minimol_mlp.py

Prerequisites:
    python generate_embeddings_minimol.py
"""

import copy
import os
import random
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tdc.benchmark_group import admet_group

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
LR_PATIENCE = 5
EMBED_DIM = 512

SEEDS = [1, 2, 3, 4, 5]  # TDC standard


# ── Dataset ──────────────────────────────────────────────────────────────────
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


# ── MLP ──────────────────────────────────────────────────────────────────────
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


# ── Training / Evaluation ────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).squeeze(-1)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).squeeze(-1)
        loss = criterion(preds, y)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


def run_seed(seed, train_emb, train_tgt_s, val_emb, val_tgt_s,
             test_emb, test_tgt_s, scaler):
    set_seed(seed)

    train_loader = DataLoader(
        EmbeddingDataset(train_emb, train_tgt_s), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(EmbeddingDataset(val_emb, val_tgt_s), batch_size=BATCH_SIZE)
    test_loader = DataLoader(EmbeddingDataset(test_emb, test_tgt_s), batch_size=BATCH_SIZE)

    model = MiniMolMLP(input_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
    )
    criterion = nn.MSELoss()

    best_val, best_state, no_improve = float("inf"), None, 0
    epoch_times = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        epoch_times.append(time.perf_counter() - t0)
        eta_s = (sum(epoch_times) / len(epoch_times)) * (MAX_EPOCHS - epoch)
        print(
            f"    [seed={seed}] epoch {epoch:3d}/{MAX_EPOCHS}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  ETA: {format_eta(eta_s)}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"    [seed={seed}] Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    preds_all = []
    with torch.no_grad():
        for x, _ in test_loader:
            preds_all.append(model(x.to(DEVICE)).squeeze(-1).cpu().numpy())

    p_s = np.concatenate(preds_all)
    # Inverse-transform to original scale for TDC evaluation
    return scaler.inverse_transform(p_s.reshape(-1, 1)).flatten()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[1/3] Loading TDC ADMET benchmark group ...")
    group = admet_group(path=DATA_DIR)
    benchmark = group.get("Solubility_AqSolDB")
    name = benchmark["name"]

    print("[2/3] Loading MiniMol embeddings ...")
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
    print(
        f"       TrainVal: {all_tv_emb.shape[0]}, Test: {test_emb.shape[0]}"
        f"  |  embed_dim={all_tv_emb.shape[1]}"
    )

    smi_to_idx = {s: i for i, s in enumerate(all_tv_smi)}

    # Scaler fitted on full train_val targets (superset of any seed's training set)
    scaler = StandardScaler()
    scaler.fit(all_tv_tgt.numpy().reshape(-1, 1))

    def scale(t):
        return torch.tensor(
            scaler.transform(t.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32
        )

    test_tgt_s = scale(test_tgt)

    print(f"[3/3] Running {len(SEEDS)} seeds: {SEEDS} ...")
    predictions_list = []

    for seed in SEEDS:
        print(f"  -- seed={seed} --")
        train_df, val_df = group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed
        )

        t_idx = [smi_to_idx[s] for s in train_df["Drug"].tolist()]
        v_idx = [smi_to_idx[s] for s in val_df["Drug"].tolist()]

        train_emb = all_tv_emb[t_idx]
        val_emb = all_tv_emb[v_idx]
        train_tgt_s = scale(all_tv_tgt[t_idx])
        val_tgt_s = scale(all_tv_tgt[v_idx])

        y_pred = run_seed(
            seed, train_emb, train_tgt_s, val_emb, val_tgt_s,
            test_emb, test_tgt_s, scaler,
        )
        predictions_list.append({name: y_pred})

    results = group.evaluate_many(predictions_list)
    mean_mae, std_mae = list(results.values())[0]

    metrics_text = (
        "MiniMol As-Is Baseline -- MLP Test Metrics (AqSolDB)\n"
        "=======================================================\n"
        "Model: MiniMol (512-d) -> 512->256->128->1 MLP\n"
        f"Seeds: {SEEDS}\n"
        "TDC Rank-1 reference: MAE 0.741 +/- 0.013\n"
        "\n"
        f"MAE      : {mean_mae:.4f} +/- {std_mae:.4f}\n"
        "\n[Evaluated via group.evaluate_many -- TDC official metric]\n"
    )
    print("\n" + metrics_text)

    metrics_path = os.path.join(RESULTS_DIR, "minimol_baseline_mlp_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(f"Metrics saved -> {metrics_path}")


if __name__ == "__main__":
    main()
