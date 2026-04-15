"""
Run missing classification seeds (1-4) for maml_hopfield_two_stage.
Appends to meta/results_hopfield/classification_results.csv.
"""
import importlib.util, os, sys, time
import numpy as np

# ── locate poc ───────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
poc_path     = os.path.join(SCRIPT_DIR, "run_hopfield_poc.py")

spec = importlib.util.spec_from_file_location("run_hopfield_poc", poc_path)
poc  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(poc)

import torch
DEVICE = poc.DEVICE
K_SHOTS = poc.K_SHOTS        # [5, 10, 20, 50]
CTX_SIZE = poc.CTX_SIZE      # 50000 on GPU

COND    = "maml_hopfield_two_stage"
SEEDS   = [1, 2, 3, 4]      # seed 0 already done
CLS_CSV = os.path.join(SCRIPT_DIR, "results_hopfield", "classification_results.csv")

print("=" * 68)
print("Classification completion: maml_hopfield_two_stage seeds 1-4")
print(f"Device: {DEVICE}  |  K: {K_SHOTS}  |  Seeds: {SEEDS}")
print(f"CSV: {CLS_CSV}")
print("=" * 68)

# ── load existing done set ───────────────────────────────────────────────────
_, cls_done = poc.load_existing_csv(CLS_CSV)
print(f"\n[Cache] {len(cls_done)} rows already done in CSV.")

# ── load/cache datasets ───────────────────────────────────────────────────────
print("\n[Step 0] Loading datasets …")
all_names = list(dict.fromkeys(
    poc.CLS_TRAIN_TASKS + poc.CLS_TEST_TASKS))
datasets = {n: poc.get_dataset_embeddings(n) for n in all_names}
print("  All datasets ready.")

# ── build context set ─────────────────────────────────────────────────────────
print("\n[Step 1] Building context set …")
all_train_ds = {n: datasets[n] for n in poc.CLS_TRAIN_TASKS}
ctx_v, ctx_k = poc.build_context_set(all_train_ds, CTX_SIZE)
ctx_v = ctx_v.to(DEVICE)
ctx_k = ctx_k.to(DEVICE)
print(f"  ctx_v: {ctx_v.shape}  ctx_k: {ctx_k.shape}")

# ── prepare splits ─────────────────────────────────────────────────────────────
cls_train_data = {n: datasets[n]["train"] for n in poc.CLS_TRAIN_TASKS}
cls_test_full  = {n: poc.get_full_dataset(datasets[n]) for n in poc.CLS_TEST_TASKS}

# ── run missing seeds ─────────────────────────────────────────────────────────
for seed in SEEDS:
    key = poc._row_key(seed, COND)
    if key in cls_done:
        print(f"\n  [Seed {seed}] already done — skipping")
        continue

    print(f"\n{'═'*68}")
    print(f"  SEED {seed}  —  {COND}")
    print(f"{'═'*68}")
    poc.set_seed(seed)

    t_train = time.perf_counter()
    cls_model, _ = poc.meta_train(
        poc.make_model(COND, "cls"), cls_train_data, "cls", seed, ctx_k, ctx_v)
    print(f"  Meta-training done  ({time.perf_counter()-t_train:.0f}s)")

    cls_row = {"seed": seed, "condition": COND}
    for K in K_SHOTS:
        t0  = time.perf_counter()
        res = poc.meta_test_cls(cls_model, cls_test_full, K, seed, ctx_k, ctx_v)
        avg = float(np.nanmean(list(res.values())))
        cls_row[f"K{K}"] = avg
        for tname, val in res.items():
            cls_row[f"{tname}_K{K}"] = val
        detail = "  ".join(f"{n}={v:.3f}" for n, v in res.items())
        print(f"    K={K:2d}: AUROC={avg:.4f}  [{detail}]  ({time.perf_counter()-t0:.0f}s)")

    poc.append_csv_row(cls_row, CLS_CSV)
    cls_done.add(key)
    print(f"  Appended seed {seed} to {CLS_CSV}")

print(f"\n{'═'*68}")
print("Done — maml_hopfield_two_stage classification seeds 1-4 complete.")
