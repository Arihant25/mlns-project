"""
train_minimol_mlp.py
====================
MiniMol "as-is" baseline: train a simple MLP on 512-d MiniMol embeddings.

Reproduces the TDC ADMET leaderboard Rank-1 approach for AqSolDB
(MAE 0.741 +/- 0.013). Runs 5 independent seeds [1,2,3,4,5] per TDC
evaluation protocol and reports mean +/- std.

Architecture: 512 -> 256 -> GELU -> Dropout -> 128 -> GELU -> Dropout -> 1

Usage:
    python train_minimol_mlp.py
"""

import copy
import os
import random
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
EMBED_DIM = 512  # MiniMol output dimension

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


# ── MLP (MiniMol standard approach) ─────────────────────────────────────────
class MiniMolMLP(nn.Module):
    """
    Lightweight MLP matching the standard MiniMol downstream head:
        512 -> 256 -> GELU -> Dropout(0.1) -> 128 -> GELU -> Dropout(0.1) -> 1
    """

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


def run_single_seed(seed, train_emb, val_emb, test_emb,
                    train_tgt_s, val_tgt_s, test_tgt_s, scaler):
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

    preds_all, tgts_all = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds_all.append(model(x.to(DEVICE)).squeeze(-1).cpu().numpy())
            tgts_all.append(y.numpy())

    p_s = np.concatenate(preds_all)
    t_s = np.concatenate(tgts_all)
    p = scaler.inverse_transform(p_s.reshape(-1, 1)).flatten()
    t = scaler.inverse_transform(t_s.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(np.abs(p - t)))
    r = float(pearsonr(p, t)[0])
    rho = float(spearmanr(p, t)[0])
    return rmse, mae, r, rho


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[1/3] Loading MiniMol embeddings ...")
    train_emb, train_tgt = load_tensors("train")
    val_emb, val_tgt = load_tensors("val")
    test_emb, test_tgt = load_tensors("test")
    print(
        f"       Train: {train_emb.shape[0]}, Val: {val_emb.shape[0]}, "
        f"Test: {test_emb.shape[0]}  |  embed_dim={train_emb.shape[1]}"
    )

    print("[2/3] Fitting StandardScaler ...")
    scaler = StandardScaler()
    scaler.fit(train_tgt.numpy().reshape(-1, 1))

    def scale(t):
        return torch.tensor(
            scaler.transform(t.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32
        )

    train_tgt_s = scale(train_tgt)
    val_tgt_s = scale(val_tgt)
    test_tgt_s = scale(test_tgt)

    print(f"[3/3] Running {len(SEEDS)} seeds: {SEEDS} ...")
    results = []
    for seed in SEEDS:
        print(f"  -- seed={seed} --")
        m = run_single_seed(
            seed, train_emb, val_emb, test_emb,
            train_tgt_s, val_tgt_s, test_tgt_s, scaler
        )
        print(f"  seed={seed}  RMSE={m[0]:.4f}  MAE={m[1]:.4f}  r={m[2]:.4f}  rho={m[3]:.4f}")
        results.append(m)

    arr = np.array(results)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)

    metrics_text = (
        "MiniMol As-Is Baseline -- MLP Test Metrics (AqSolDB)\n"
        "=======================================================\n"
        "Model: MiniMol (512-d) -> 512->256->128->1 MLP\n"
        f"Seeds: {SEEDS}\n"
        f"TDC Rank-1 reference: MAE 0.741 +/- 0.013\n"
        "\n"
        f"RMSE     : {means[0]:.4f} +/- {stds[0]:.4f}\n"
        f"MAE      : {means[1]:.4f} +/- {stds[1]:.4f}\n"
        f"Pearson  : {means[2]:.4f} +/- {stds[2]:.4f}\n"
        f"Spearman : {means[3]:.4f} +/- {stds[3]:.4f}\n"
        "\nPer-seed results:\n"
    )
    for seed, m in zip(SEEDS, results):
        metrics_text += f"  seed={seed}  RMSE={m[0]:.4f}  MAE={m[1]:.4f}  r={m[2]:.4f}  rho={m[3]:.4f}\n"

    print("\n" + metrics_text)

    metrics_path = os.path.join(RESULTS_DIR, "minimol_baseline_mlp_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(f"Metrics saved -> {metrics_path}")


if __name__ == "__main__":
    main()
