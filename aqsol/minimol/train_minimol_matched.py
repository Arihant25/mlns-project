"""
train_minimol_matched.py
========================
Matches the MiniMol TDC leaderboard submission as closely as possible,
in two backbone modes:

  FROZEN  -- pre-computed 512-d embeddings, TaskHead only (matches paper exactly)
  FINETUNE -- full GNN fine-tuned end-to-end via _readout_cache bypass (our extension)

Head architecture (AqSolDB config from tdc_leaderboard_submission.py):
  depth=4, hidden_dim=1024, combine=True, dropout=0.1
  3 x (Linear -> BatchNorm1d -> ReLU -> Dropout)
  Final: cat(hidden_1024, original_512) -> Linear(1536, 1)

Training (frozen, matching paper):
  Optimizer : Adam, weight_decay=1e-4
  Scheduler : cosine annealing with 5-epoch linear warmup
  Epochs    : 25  (no patience-based stopping; keep best val checkpoint)
  Batch     : train=32, eval=128
  Seeds     : [1, 2, 3, 4, 5]  (TDC standard)
  Eval      : group.evaluate_many  (TDC official)

Known remaining difference vs leaderboard:
  - Paper uses 5 reps x 5 folds = 25 models (Cantor-paired seeds);
    we use 5 seeds with one model each.
  - Fine-tune run is our own extension; paper uses frozen embeddings only.

Usage:
    python train_minimol_matched.py

Prerequisites:
    python generate_embeddings_minimol.py
    python train_minimol_evidential_gsl.py  -> results/evidential_gsl_minimol.pt
"""

import copy
import math
import os
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.nn import global_max_pool
from tdc.benchmark_group import admet_group

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GSL_DIR = os.path.join(PROJECT_ROOT, "gsl")
sys.path.insert(0, GSL_DIR)

from gsl_utils import GSLDataset, gsl_collate_fn      # noqa: E402
from model_evidential_gsl import EvidentialGSLModel   # noqa: E402

DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH  = os.path.join(RESULTS_DIR, "evidential_gsl_minimol.pt")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paper-matched hyperparameters (tdc_leaderboard_submission.py, AqSolDB) ───
HIDDEN_DIM       = 1024
DEPTH            = 4        # 3 hidden layers + output (depth-1 hidden layers)
LR               = 5e-4
WEIGHT_DECAY     = 1e-4
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL  = 128
MAX_EPOCHS       = 25       # run all 25; keep best val checkpoint
WARMUP_EPOCHS    = 5        # linear warmup then cosine decay
EMBED_DIM        = 512
DROPOUT          = 0.1

# ── Fine-tune specific (our extension, not in paper) ──────────────────────────
LR_BACKBONE   = 1e-5   # GNN backbone LR (~50x lower)
MAX_EPOCHS_FT = 50     # more epochs for GNN adaptation
WARMUP_FT     = 10
PATIENCE_FT   = 15     # early stopping patience

SEEDS          = [1, 2, 3, 4, 5]
GAMMA_WEIGHTED = 20.0


# ── Datasets ──────────────────────────────────────────────────────────────────
class EmbeddingDataset(Dataset):
    def __init__(self, emb, tgt):
        self.emb, self.tgt = emb, tgt

    def __len__(self):
        return self.emb.shape[0]

    def __getitem__(self, idx):
        return self.emb[idx], self.tgt[idx]


class WeightedEmbeddingDataset(Dataset):
    def __init__(self, emb, tgt, weights):
        self.emb, self.tgt, self.weights = emb, tgt, weights

    def __len__(self):
        return self.emb.shape[0]

    def __getitem__(self, idx):
        return self.emb[idx], self.tgt[idx], self.weights[idx]


class GraphDataset(Dataset):
    """PyG Data objects + scaled targets; silently drops featurization failures (str entries)."""
    def __init__(self, data_list, targets):
        valid = [(d, t) for d, t in zip(data_list, targets) if not isinstance(d, str)]
        self.data    = [x[0] for x in valid]
        self.targets = torch.stack([x[1] for x in valid]) if valid else torch.empty(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def graph_collate(batch):
    data_list, targets = zip(*batch)
    return Batch.from_data_list(list(data_list)), torch.stack(list(targets))


# ── Head (exact match to paper) ───────────────────────────────────────────────
class TaskHead(nn.Module):
    """
    Matches tdc_leaderboard_submission.py TaskHead for AqSolDB:
      depth=4  -> 3 hidden layers (Linear -> BN1d -> ReLU -> Dropout)
      combine=True -> cat(last_hidden_1024, original_512) -> Linear(1536, 1)
    """
    def __init__(self, input_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 depth=DEPTH, dropout=DROPOUT):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth - 1):          # 3 hidden layers
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.final  = nn.Linear(hidden_dim + input_dim, 1)  # 1024+512=1536 -> 1

    def forward(self, x):
        return self.final(torch.cat([self.hidden(x), x], dim=-1))


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def format_eta(sec: float) -> str:
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_cosine_warmup_scheduler(optimizer, warmup, total):
    def lr_lambda(epoch):
        if epoch < warmup:
            return float(epoch) / float(max(1, warmup))
        progress = float(epoch - warmup) / float(max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _val_mse_emb(model, loader):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            total += torch.mean((model(x).squeeze(-1) - y) ** 2).item() * x.size(0)
            n += x.size(0)
    return total / n


def _test_preds_emb(model, loader, scaler):
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            out.append(model(x).squeeze(-1).cpu().numpy())
    p_s = np.concatenate(out)
    return scaler.inverse_transform(p_s.reshape(-1, 1)).flatten()


# ── Frozen: train + predict ───────────────────────────────────────────────────
def train_frozen(train_emb, train_tgt_s, val_emb, val_tgt_s,
                 test_emb, test_tgt_s, scaler, seed):
    set_seed(seed)
    tr_loader = DataLoader(EmbeddingDataset(train_emb, train_tgt_s),
                           BATCH_SIZE_TRAIN, shuffle=True)
    vl_loader = DataLoader(EmbeddingDataset(val_emb,   val_tgt_s),   BATCH_SIZE_EVAL)
    te_loader = DataLoader(EmbeddingDataset(test_emb,  test_tgt_s),  BATCH_SIZE_EVAL)

    model     = TaskHead().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = make_cosine_warmup_scheduler(optimizer, WARMUP_EPOCHS, MAX_EPOCHS)
    criterion = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for _ in range(MAX_EPOCHS):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(x).squeeze(-1), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
        val_loss = _val_mse_emb(model, vl_loader)
        if val_loss < best_val:
            best_val, best_state = val_loss, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return _test_preds_emb(model, te_loader, scaler)


def train_weighted(train_emb, train_tgt_s, val_emb, val_tgt_s,
                   test_emb, test_tgt_s, scaler, seed, gamma, u_train):
    set_seed(seed)
    tr_loader = DataLoader(
        WeightedEmbeddingDataset(train_emb, train_tgt_s, u_train),
        BATCH_SIZE_TRAIN, shuffle=True,
    )
    vl_loader = DataLoader(
        WeightedEmbeddingDataset(val_emb, val_tgt_s, torch.zeros(val_emb.shape[0])),
        BATCH_SIZE_EVAL,
    )
    te_loader = DataLoader(
        WeightedEmbeddingDataset(test_emb, test_tgt_s, torch.zeros(test_emb.shape[0])),
        BATCH_SIZE_EVAL,
    )

    model     = TaskHead().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = make_cosine_warmup_scheduler(optimizer, WARMUP_EPOCHS, MAX_EPOCHS)

    best_val, best_state = float("inf"), None
    for _ in range(MAX_EPOCHS):
        model.train()
        for x, y, u in tr_loader:
            x, y, u = x.to(DEVICE), y.to(DEVICE), u.to(DEVICE)
            loss = torch.mean(torch.exp(-gamma * u) * (model(x).squeeze(-1) - y) ** 2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
        val_loss = _val_mse_emb(model, vl_loader)
        if val_loss < best_val:
            best_val, best_state = val_loss, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return _test_preds_emb(model, te_loader, scaler)


def extract_uncertainties(emb, tgt_s, smiles, unc_model):
    loader = DataLoader(
        GSLDataset(emb, tgt_s, smiles),
        batch_size=emb.shape[0], shuffle=False, collate_fn=gsl_collate_fn,
    )
    with torch.no_grad():
        for X, _, A in loader:
            X, A = X.to(DEVICE), A.to(DEVICE)
            _, (_, _, alpha, beta) = unc_model(X, A)
    return (beta / (alpha - 1.0)).detach().cpu()


# ── Fine-tune: train + predict ────────────────────────────────────────────────
def _featurize_all(mm, smiles_list):
    """SMILES -> list of PyG Data objects (failed molecules stay as str)."""
    with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
        data_list, _ = mm.datamodule._featurize_molecules(smiles_list)
        data_list = mm.to_fp32(data_list)
    return data_list


def train_finetune(network, initial_net_state,
                   train_data, train_tgt_s, val_data, val_tgt_s,
                   test_data, test_tgt_s, scaler, seed):
    set_seed(seed)
    network.load_state_dict(copy.deepcopy(initial_net_state))
    network.to(DEVICE)

    tr_loader = DataLoader(GraphDataset(train_data, train_tgt_s),
                           BATCH_SIZE_TRAIN, shuffle=True, collate_fn=graph_collate)
    vl_loader = DataLoader(GraphDataset(val_data, val_tgt_s),
                           BATCH_SIZE_EVAL, collate_fn=graph_collate)
    te_loader = DataLoader(GraphDataset(test_data, test_tgt_s),
                           BATCH_SIZE_EVAL, collate_fn=graph_collate)

    head = TaskHead().to(DEVICE)
    optimizer = torch.optim.Adam([
        {"params": network.parameters(), "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": head.parameters(),    "lr": LR,          "weight_decay": WEIGHT_DECAY},
    ])
    scheduler = make_cosine_warmup_scheduler(optimizer, WARMUP_FT, MAX_EPOCHS_FT)
    criterion = nn.MSELoss()

    best_val, best_head_state, best_net_state, no_improve = float("inf"), None, None, 0

    for _ in range(MAX_EPOCHS_FT):
        network.train()
        head.train()
        for batch_graph, y in tr_loader:
            batch_graph, y = batch_graph.to(DEVICE), y.to(DEVICE)
            network(batch_graph)
            node_emb = network._module_map["gnn"]._readout_cache[15]
            mol_emb  = global_max_pool(node_emb, batch_graph.batch)
            loss = criterion(head(mol_emb).squeeze(-1), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        network.eval()
        head.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for batch_graph, y in vl_loader:
                batch_graph, y = batch_graph.to(DEVICE), y.to(DEVICE)
                network(batch_graph)
                node_emb = network._module_map["gnn"]._readout_cache[15]
                mol_emb  = global_max_pool(node_emb, batch_graph.batch)
                pred = head(mol_emb).squeeze(-1)
                val_total += torch.mean((pred - y) ** 2).item() * batch_graph.num_graphs
                val_n += batch_graph.num_graphs
        val_loss = val_total / val_n

        if val_loss < best_val:
            best_val = val_loss
            best_head_state = copy.deepcopy(head.state_dict())
            best_net_state  = copy.deepcopy(network.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE_FT:
            break

    network.load_state_dict(best_net_state)
    head.load_state_dict(best_head_state)

    network.eval()
    head.eval()
    out = []
    with torch.no_grad():
        for batch_graph, _ in te_loader:
            batch_graph = batch_graph.to(DEVICE)
            network(batch_graph)
            node_emb = network._module_map["gnn"]._readout_cache[15]
            mol_emb  = global_max_pool(node_emb, batch_graph.batch)
            out.append(head(mol_emb).squeeze(-1).cpu().numpy())
    p_s = np.concatenate(out)
    return scaler.inverse_transform(p_s.reshape(-1, 1)).flatten()


# ── Seed loops ────────────────────────────────────────────────────────────────
def run_seeds_frozen(group, name, all_tv_emb, all_tv_tgt, all_tv_smi, smi_to_idx,
                     test_emb, test_tgt, scaler, scale, gamma, unc_model, label):
    predictions_list = []
    t0 = time.perf_counter()

    for i, seed in enumerate(SEEDS):
        train_df, val_df = group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed
        )
        t_idx = [smi_to_idx[s] for s in train_df["Drug"].tolist()]
        v_idx = [smi_to_idx[s] for s in val_df["Drug"].tolist()]

        train_emb   = all_tv_emb[t_idx]
        val_emb     = all_tv_emb[v_idx]
        train_tgt_s = scale(all_tv_tgt[t_idx])
        val_tgt_s   = scale(all_tv_tgt[v_idx])
        test_tgt_s  = scale(test_tgt)

        if gamma > 0:
            u_train = extract_uncertainties(
                train_emb, train_tgt_s, train_df["Drug"].tolist(), unc_model
            )
            y_pred = train_weighted(
                train_emb, train_tgt_s, val_emb, val_tgt_s,
                test_emb, test_tgt_s, scaler, seed, gamma, u_train,
            )
        else:
            y_pred = train_frozen(
                train_emb, train_tgt_s, val_emb, val_tgt_s,
                test_emb, test_tgt_s, scaler, seed,
            )

        predictions_list.append({name: y_pred})
        elapsed = time.perf_counter() - t0
        eta = (elapsed / (i + 1)) * (len(SEEDS) - i - 1)
        print(f"  [{label}] seed={seed}  ETA: {format_eta(eta)}")

    results = group.evaluate_many(predictions_list)
    return list(results.values())[0]


def run_seeds_finetune(group, name, mm, all_tv_data, all_tv_tgt, all_tv_smi, smi_to_idx,
                       test_data, test_tgt, scaler, scale):
    network = mm.predictor.network
    initial_state = copy.deepcopy(network.state_dict())

    test_valid_idx  = [i for i, d in enumerate(test_data) if not isinstance(d, str)]
    test_valid_data = [test_data[i] for i in test_valid_idx]
    train_mean      = float(scaler.mean_[0])
    n_test          = len(test_data)

    predictions_list = []
    t0 = time.perf_counter()

    for i, seed in enumerate(SEEDS):
        train_df, val_df = group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed
        )
        t_idx = [smi_to_idx[s] for s in train_df["Drug"].tolist()]
        v_idx = [smi_to_idx[s] for s in val_df["Drug"].tolist()]

        train_data_s   = [all_tv_data[j] for j in t_idx]
        val_data_s     = [all_tv_data[j] for j in v_idx]
        train_tgt_s    = scale(all_tv_tgt[t_idx])
        val_tgt_s      = scale(all_tv_tgt[v_idx])
        test_tgt_s     = scale(test_tgt)
        test_tgt_valid = test_tgt_s[test_valid_idx]

        y_pred_valid = train_finetune(
            network, initial_state,
            train_data_s, train_tgt_s,
            val_data_s,   val_tgt_s,
            test_valid_data, test_tgt_valid,
            scaler, seed,
        )

        y_pred_full = np.full(n_test, train_mean, dtype=np.float32)
        for j, pred in zip(test_valid_idx, y_pred_valid):
            y_pred_full[j] = pred

        predictions_list.append({name: y_pred_full})
        elapsed = time.perf_counter() - t0
        eta = (elapsed / (i + 1)) * (len(SEEDS) - i - 1)
        print(f"  [finetune] seed={seed}  ETA: {format_eta(eta)}")

    results = group.evaluate_many(predictions_list)
    return list(results.values())[0]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[1/5] Loading TDC ADMET benchmark group ...")
    group     = admet_group(path=DATA_DIR)
    benchmark = group.get("Solubility_AqSolDB")
    name      = benchmark["name"]

    print("[2/5] Loading frozen MiniMol embeddings ...")
    all_tv_emb = torch.load(os.path.join(DATA_DIR, "trainval_embeddings_minimol.pt"), weights_only=True)
    all_tv_tgt = torch.load(os.path.join(DATA_DIR, "trainval_targets.pt"),             weights_only=True)
    all_tv_smi = torch.load(os.path.join(DATA_DIR, "trainval_smiles.pt"))
    test_emb   = torch.load(os.path.join(DATA_DIR, "test_embeddings_minimol.pt"),      weights_only=True)
    test_tgt   = torch.load(os.path.join(DATA_DIR, "test_targets.pt"),                 weights_only=True)
    smi_to_idx = {s: i for i, s in enumerate(all_tv_smi)}
    print(f"       TrainVal: {all_tv_emb.shape[0]}  Test: {test_emb.shape[0]}  embed_dim={EMBED_DIM}")

    scaler = StandardScaler()
    scaler.fit(all_tv_tgt.numpy().reshape(-1, 1))

    def scale(t):
        return torch.tensor(
            scaler.transform(t.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32
        )

    print("[3/5] Loading EvidentialGSLModel for uncertainty ...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run train_minimol_evidential_gsl.py first.")
    unc_model = EvidentialGSLModel(embed_dim=EMBED_DIM)
    unc_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    unc_model.to(DEVICE).eval()

    print(f"[4/4] Frozen-embedding runs ({len(SEEDS)} seeds, {MAX_EPOCHS} epochs each) ...")
    print(f"      Head: {EMBED_DIM}->{HIDDEN_DIM}x{DEPTH-1}->1536->1  combine=True  "
          f"Adam lr={LR}  cosine warmup={WARMUP_EPOCHS}ep\n")

    print("=== Paper baseline (gamma=0) ===")
    base_mae, base_std = run_seeds_frozen(
        group, name, all_tv_emb, all_tv_tgt, all_tv_smi, smi_to_idx,
        test_emb, test_tgt, scaler, scale,
        gamma=0.0, unc_model=None, label="baseline",
    )
    print(f"\nBaseline MAE: {base_mae:.4f} +/- {base_std:.4f}\n")

    print(f"=== Soft-weighted (gamma={GAMMA_WEIGHTED}) ===")
    w_mae, w_std = run_seeds_frozen(
        group, name, all_tv_emb, all_tv_tgt, all_tv_smi, smi_to_idx,
        test_emb, test_tgt, scaler, scale,
        gamma=GAMMA_WEIGHTED, unc_model=unc_model, label="weighted",
    )
    print(f"\nWeighted MAE: {w_mae:.4f} +/- {w_std:.4f}\n")

    lines = [
        "MiniMol Matched Protocol -- AqSolDB",
        "=====================================",
        f"Head:     {EMBED_DIM} -> {HIDDEN_DIM}x{DEPTH-1} -> cat(1024,512) -> 1  [BN+ReLU+skip]",
        f"Optimizer: Adam lr={LR} wd={WEIGHT_DECAY}  |  Cosine warmup {WARMUP_EPOCHS}/{MAX_EPOCHS}ep",
        f"Seeds:    {SEEDS}",
        "",
        f"{'Run':<32}  {'MAE':>22}",
        "-" * 58,
        f"{'Paper baseline (gamma=0)':<32}  {base_mae:.4f} +/- {base_std:.4f}",
        f"{'Soft-weighted (gamma=20)':<32}  {w_mae:.4f} +/- {w_std:.4f}",
        "",
        "-" * 58,
        "TDC Rank-1 (MiniMol, 5x5 ensemble): 0.7410 +/- 0.0130",
        "",
        "[Evaluated via group.evaluate_many -- TDC official metric]",
    ]
    report = "\n".join(lines)
    print(report)

    path = os.path.join(RESULTS_DIR, "minimol_matched_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved -> {path}")


if __name__ == "__main__":
    main()
