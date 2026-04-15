#!/usr/bin/env python3
"""
run_loto.py  —  meta/
========================
Leave-One-Task-Out (LOTO) cross-validation for the Two-Stage Evidential
Meta-Learner.  Imports all model infrastructure from run_hopfield_poc.py
without modifying it.

Task pool (8 TDC ADMET regression tasks):
    Caco2_Wang, VDss_Lombardo, PPBR_AZ, Lipophilicity_AstraZeneca,
    Solubility_AqSolDB, Clearance_Microsome_AZ, Clearance_Hepatocyte_AZ, LD50_Zhu
Half_Life_Obach excluded: N=532, too small for K=50 support sets.

Usage:
    python run_loto.py                                    # full run, ChEMBL context
    python run_loto.py --context tdc                      # TDC-derived context (fallback)
    python run_loto.py --folds Caco2_Wang,VDss_Lombardo  # specific folds only
    python run_loto.py --conditions maml_hopfield_two_stage,maml_mlp
    python run_loto.py --seeds 0,1                        # subset of seeds
    QUICK=1 python run_loto.py --folds Caco2_Wang --conditions maml_mlp --seeds 0

Results → meta/results_loto/loto_results.csv
"""

import argparse, copy, csv, importlib.util, math, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

SCRIPT_DIR    = Path(__file__).parent.resolve()
PROJECT_ROOT  = SCRIPT_DIR.parent
RESULTS_DIR   = SCRIPT_DIR / "results_loto"
LOTO_CSV      = RESULTS_DIR / "loto_results.csv"
LOTO_CTX_CACHE = PROJECT_ROOT / "data" / "loto_context"
CHEMBL_DIR    = PROJECT_ROOT / "data" / "chembl_context"
CONTEXT_CACHE = PROJECT_ROOT / "data" / "context_set"

QUICK = os.environ.get("QUICK", "").lower() in ("1", "true", "yes")

# ── LOTO Configuration ────────────────────────────────────────────────────────

LOTO_TASKS = [
    "Caco2_Wang",
    "VDss_Lombardo",
    "PPBR_AZ",
    "Lipophilicity_AstraZeneca",
    "Solubility_AqSolDB",
    "Clearance_Microsome_AZ",
    "Clearance_Hepatocyte_AZ",
    # LD50_Zhu removed: it's in TDC's Tox module, not ADME; _load_tdc only wraps ADME
]

CONDITIONS_LOTO = [
    "maml_hopfield_two_stage",   # our model (primary)
    "maml_hopfield_evid",        # ablation: context gate only
    "maml_hopfield_nogating",    # MHNfs re-implementation
    "maml_dense_gsl",            # no Hopfield
    "maml_static_gnn",           # graph, fixed topology
    "maml_mlp",                  # floor baseline
]

K_SHOTS_LOTO  = [10] if QUICK else [10, 20, 50]
SEEDS_DEFAULT = [0]  if QUICK else [0, 1, 2, 3, 4]
D_INNER       = 256  # must match run_hopfield_poc.py


# ── Load run_hopfield_poc.py (read-only, no modification) ────────────────────

def _load_poc():
    poc_path = SCRIPT_DIR / "run_hopfield_poc.py"
    spec = importlib.util.spec_from_file_location("run_hopfield_poc", poc_path)
    poc  = importlib.util.module_from_spec(spec)
    sys.modules["run_hopfield_poc"] = poc
    spec.loader.exec_module(poc)
    return poc


# ── Dataset loading (with fallback for missing pre-computed embeddings) ───────

def get_dataset_embeddings_loto(poc, name: str) -> dict:
    """
    Wrapper around poc.get_dataset_embeddings that handles the case where
    Caco2_Wang and Lipophilicity_AstraZeneca have no pre-computed .pt files
    in caco/data or lipo/data.  Falls back to the generic TDC+EMBED_CACHE path.
    """
    KNOWN_PATHS = {
        "Lipophilicity_AstraZeneca": Path(poc.LIPO_DATA),
        "Caco2_Wang":               Path(poc.CACO_DATA),
    }
    if name in KNOWN_PATHS:
        base = KNOWN_PATHS[name]
        files_exist = all(
            (base / f"{sn}_embeddings.pt").exists()
            for sn in ("train", "val", "test")
        )
        if not files_exist:
            # Fall through to TDC download + generic EMBED_CACHE
            print(f"  [{name}] Pre-computed embeddings not found in {base}; "
                  f"using TDC + EMBED_CACHE …")
            cache = Path(poc.EMBED_CACHE) / name
            cache.mkdir(parents=True, exist_ok=True)
            raw = poc._load_tdc(name)
            out = {}
            for sn in ("train", "val", "test"):
                ep = cache / f"{sn}_embeddings.pt"
                tp = cache / f"{sn}_targets.pt"
                sp = cache / f"{sn}_smiles.pt"
                if ep.exists() and tp.exists() and sp.exists():
                    out[sn] = {
                        "emb": torch.load(ep, weights_only=True),
                        "tgt": torch.load(tp, weights_only=True),
                        "smi": torch.load(sp, weights_only=False),
                    }
                else:
                    ch  = raw[sn]
                    print(f"    Embedding {sn} ({len(ch['smi'])} mols) …")
                    emb = poc.embed_smiles(ch["smi"])
                    torch.save(emb,       ep)
                    torch.save(ch["tgt"], tp)
                    torch.save(ch["smi"], sp)
                    out[sn] = {"emb": emb, "tgt": ch["tgt"], "smi": ch["smi"]}
            return out
    return poc.get_dataset_embeddings(name)


# ── Context set loading ───────────────────────────────────────────────────────

def _get_wk():
    """Load or derive the fixed Wk projection matrix (always seed=42)."""
    wk_path = CONTEXT_CACHE / "Wk_fixed.pt"
    if wk_path.exists():
        return torch.load(wk_path, weights_only=True)
    # Re-derive deterministically — identical to run_hopfield_poc.py
    torch.manual_seed(42)
    R    = torch.randn(768, D_INNER)
    R, _ = torch.linalg.qr(R)
    return R


def load_chembl_context(device) -> tuple:
    """Load pre-built ChEMBL context tensors (from build_chembl_context.py)."""
    v_path = CHEMBL_DIR / "chembl_ctx_v.pt"
    k_path = CHEMBL_DIR / "chembl_ctx_k.pt"
    if not (v_path.exists() and k_path.exists()):
        sys.exit(
            f"[ERROR] ChEMBL context not found at {CHEMBL_DIR}.\n"
            "Run build_chembl_context.py first, or use --context tdc."
        )
    ctx_v = torch.load(v_path, weights_only=True).to(device)
    ctx_k = torch.load(k_path, weights_only=True).to(device)
    print(f"  [Context] ChEMBL: {ctx_v.shape[0]:,} molecules ({ctx_v.shape})")
    return ctx_v, ctx_k


def build_loto_tdc_context(train_datasets: dict, held_out: str,
                            ctx_size: int, device) -> tuple:
    """
    Build fold-specific context from training task embeddings.
    Uses a per-fold cache key so different folds don't share files.
    """
    LOTO_CTX_CACHE.mkdir(parents=True, exist_ok=True)
    fold_tag = held_out.replace("/", "_").replace(" ", "_")
    v_path   = LOTO_CTX_CACHE / f"ctx_v_{fold_tag}_{ctx_size}.pt"
    k_path   = LOTO_CTX_CACHE / f"ctx_k_{fold_tag}_{ctx_size}.pt"

    if v_path.exists() and k_path.exists():
        ctx_v = torch.load(v_path, weights_only=True).to(device)
        ctx_k = torch.load(k_path, weights_only=True).to(device)
        print(f"  [Context] TDC fold={fold_tag}: {ctx_v.shape[0]:,} mols (cached)")
        return ctx_v, ctx_k

    embs = []
    for ds in train_datasets.values():
        embs.append(ds["train"]["emb"])
        embs.append(ds["val"]["emb"])
    pool = torch.cat(embs, dim=0)   # [M_all, 768]

    g    = torch.Generator(); g.manual_seed(42)
    perm = torch.randperm(pool.size(0), generator=g)[:ctx_size]
    ctx_v = pool[perm].contiguous()

    R     = _get_wk()
    ctx_k = (ctx_v @ R).contiguous()

    torch.save(ctx_v, v_path)
    torch.save(ctx_k, k_path)
    ctx_v, ctx_k = ctx_v.to(device), ctx_k.to(device)
    print(f"  [Context] TDC fold={fold_tag}: {ctx_v.shape[0]:,} mols (built)")
    return ctx_v, ctx_k


# ── CSV I/O (resumable) ───────────────────────────────────────────────────────

CSV_FIELDS = ["held_out", "condition", "seed", "task_std"] + [f"K{k}" for k in [10, 20, 50]]


def load_existing_loto(path: Path) -> set:
    """Return set of 'held_out|condition|seed' keys already completed."""
    done = set()
    if not path.exists():
        return done
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            done.add(f"{row['held_out']}|{row['condition']}|{row['seed']}")
    return done


def append_loto_row(row: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        # Fill missing K columns with empty string
        out = {k: row.get(k, "") for k in CSV_FIELDS}
        w.writerow(out)


# ── Main LOTO loop ────────────────────────────────────────────────────────────

def run_loto(args):
    poc    = _load_poc()
    device = poc.DEVICE
    ctx_size = poc.CTX_SIZE
    meta_episodes = 20 if QUICK else poc.META_EPISODES

    # Override META_EPISODES for QUICK mode
    poc.META_EPISODES = meta_episodes

    folds      = args.folds      or LOTO_TASKS
    conditions = args.conditions or CONDITIONS_LOTO
    seeds      = args.seeds      or SEEDS_DEFAULT
    use_chembl = (args.context == "chembl")

    print("=" * 68)
    print("LOTO Cross-Validation: Two-Stage Evidential Meta-Learner")
    print(f"  Device:     {device}")
    print(f"  Context:    {'ChEMBL (independent)' if use_chembl else 'TDC (fold-specific)'}")
    print(f"  Folds:      {folds}")
    print(f"  Conditions: {conditions}")
    print(f"  Seeds:      {seeds}")
    print(f"  K-shots:    {K_SHOTS_LOTO}")
    print(f"  Episodes:   {meta_episodes}")
    print(f"  Quick mode: {QUICK}")
    print("=" * 68)

    # Pre-load ChEMBL context once if using it (it's fold-independent)
    chembl_ctx = None
    if use_chembl:
        print("\n[Step 0] Loading ChEMBL context …")
        chembl_ctx = load_chembl_context(device)

    # Load existing results to enable resumption
    done_set = load_existing_loto(LOTO_CSV)
    if done_set:
        print(f"\n[Resume] Skipping {len(done_set)} already-completed runs.")

    # Load all task embeddings upfront (cached to disk by get_dataset_embeddings)
    print("\n[Step 1] Loading / caching all task embeddings …")
    datasets = {}
    for name in LOTO_TASKS:
        datasets[name] = get_dataset_embeddings_loto(poc, name)
    print("  All datasets ready.")

    total_runs = len(folds) * len(conditions) * len(seeds)
    completed  = 0

    for fold_idx, held_out in enumerate(folds):
        train_tasks = [t for t in LOTO_TASKS if t != held_out]

        print(f"\n{'═'*68}")
        print(f"Fold {fold_idx+1}/{len(folds)}: held-out = {held_out}")
        print(f"  Training on: {train_tasks}")
        print(f"{'═'*68}")

        # Context set for this fold
        if use_chembl:
            ctx_v, ctx_k = chembl_ctx
        else:
            train_ds = {n: datasets[n] for n in train_tasks}
            ctx_v, ctx_k = build_loto_tdc_context(
                train_ds, held_out, ctx_size, device)

        # Prepare training data (train split only — val is in query for generalization)
        train_data = {n: datasets[n]["train"] for n in train_tasks}

        # Prepare test data: full dataset (train+val+test pooled as query)
        full_ds   = poc.get_full_dataset(datasets[held_out])
        test_full = {held_out: full_ds}

        # Task target std for normalization in the results table
        task_std  = float(full_ds["tgt"].numpy().std())

        for cond in conditions:
            for seed in seeds:
                run_key = f"{held_out}|{cond}|{seed}"
                if run_key in done_set:
                    print(f"  [Skip] {cond} seed={seed}  (already done)")
                    completed += 1
                    continue

                print(f"\n  [{completed+1}/{total_runs}] cond={cond}  seed={seed}")
                t0 = time.time()

                # Train
                model = poc.make_model(cond, "reg")
                trained_model, _ = poc.meta_train(
                    model, train_data, "reg", seed, ctx_k, ctx_v)

                # Test at each K
                row = {"held_out": held_out, "condition": cond,
                       "seed": seed, "task_std": f"{task_std:.6f}"}
                for K in K_SHOTS_LOTO:
                    results = poc.meta_test_reg(
                        trained_model, test_full, K, seed, ctx_k, ctx_v)
                    rmse = results[held_out]
                    row[f"K{K}"] = f"{rmse:.6f}"
                    print(f"    K={K:2d}: RMSE={rmse:.4f}")

                append_loto_row(row, LOTO_CSV)
                done_set.add(run_key)
                completed += 1
                elapsed = time.time() - t0
                print(f"  → saved  ({elapsed:.0f}s)")

    print(f"\n{'='*68}")
    print(f"LOTO complete. Results: {LOTO_CSV}")
    print_summary(LOTO_CSV, conditions, seeds)


# ── Summary printing ──────────────────────────────────────────────────────────

def print_summary(csv_path: Path, conditions: list, seeds: list):
    if not csv_path.exists():
        return

    # Load all rows
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            row = dict(r)
            for k in [f"K{k}" for k in [10, 20, 50]]:
                try:    row[k] = float(row[k])
                except: row[k] = float("nan")
            rows.append(row)

    print("\n" + "=" * 68)
    print("LOTO Summary  (mean RMSE ± std across folds)")
    print("  Aggregation: per-fold mean over seeds, then std across folds")
    print("=" * 68)

    folds_seen = sorted(set(r["held_out"] for r in rows))

    for cond in conditions:
        cond_rows = [r for r in rows if r["condition"] == cond]
        if not cond_rows:
            continue
        print(f"\n  {cond}:")
        for K in K_SHOTS_LOTO:
            key = f"K{K}"
            # Per-fold mean across seeds
            fold_means = []
            for fold in folds_seen:
                vals = [r[key] for r in cond_rows
                        if r["held_out"] == fold and not math.isnan(r[key])]
                if vals:
                    fold_means.append(np.mean(vals))
            if fold_means:
                m = np.mean(fold_means)
                s = np.std(fold_means)
                print(f"    K={K:2d}: {m:.4f} ± {s:.4f}  (n_folds={len(fold_means)})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LOTO cross-validation for Hopfield meta-learner")
    p.add_argument(
        "--context", choices=["chembl", "tdc"], default="chembl",
        help="Context set source (default: chembl — requires build_chembl_context.py)")
    p.add_argument(
        "--folds", type=lambda s: s.split(","), default=None,
        metavar="TASK1,TASK2",
        help="Comma-separated list of held-out tasks (default: all 8)")
    p.add_argument(
        "--conditions", type=lambda s: s.split(","), default=None,
        metavar="COND1,COND2",
        help="Comma-separated conditions (default: all 6)")
    p.add_argument(
        "--seeds", type=lambda s: [int(x) for x in s.split(",")], default=None,
        metavar="0,1,2",
        help="Comma-separated seeds (default: 0-4)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_loto(args)
