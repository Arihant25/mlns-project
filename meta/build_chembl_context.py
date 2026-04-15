#!/usr/bin/env python3
"""
build_chembl_context.py  —  meta/
===================================
One-time job: sample 75k diverse ChEMBL molecules, embed with MolFormer,
save context tensors for use in run_loto.py.

Prerequisites:
    pip install chembl-webresource-client

Output (in PROJECT_ROOT/data/chembl_context/):
    chembl_ctx_v.pt      [N, 768]  raw MolFormer values (context V)
    chembl_ctx_k.pt      [N, 256]  pre-projected keys   (context K)
    chembl_smiles.txt    one SMILES per line
    Wk_fixed.pt          [768, 256] same fixed projection used by all models

CRITICAL: The Wk projection matrix MUST match what was used during meta-training.
It is deterministically derived from torch.manual_seed(42) and is identical to
the one saved by run_hopfield_poc.py at data/context_set/Wk_fixed.pt.
"""

import importlib.util, os, sys, math, time
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.warning")

SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DIR      = PROJECT_ROOT / "data" / "chembl_context"
CONTEXT_CACHE = PROJECT_ROOT / "data" / "context_set"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_N     = 75_000        # desired diversity-sampled molecules
D_INNER      = 256           # must match run_hopfield_poc.py
EMBED_BATCH  = 32
CHEMBL_LIMIT = 150_000       # cap on raw API pull; 150k raw → ~75k scaffold-diverse
MW_MAX       = 600
HBD_MAX      = 5
HBA_MAX      = 10


# ── Import embed_smiles from run_hopfield_poc without modifying it ─────────────

def _load_poc():
    poc_path = SCRIPT_DIR / "run_hopfield_poc.py"
    spec = importlib.util.spec_from_file_location("run_hopfield_poc", poc_path)
    poc  = importlib.util.module_from_spec(spec)
    # Suppress main() from running
    sys.modules["run_hopfield_poc"] = poc
    spec.loader.exec_module(poc)
    return poc


# ── Step A: Download raw SMILES from ChEMBL ───────────────────────────────────

def fetch_chembl_smiles(limit: int = CHEMBL_LIMIT) -> list:
    """
    Pull SMILES from ChEMBL REST API with basic Lipinski filter.
    Returns list of canonical SMILES strings (deduplicated, valid).
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        sys.exit("[ERROR] chembl-webresource-client not installed.\n"
                 "Run: pip install chembl-webresource-client")

    print(f"[ChEMBL] Querying API (limit={limit:,}) …")
    client = new_client.molecule
    query  = client.filter(
        molecule_properties__mw_freebase__lte=MW_MAX,
        molecule_properties__hbd__lte=HBD_MAX,
        molecule_properties__hba__lte=HBA_MAX,
        structure_type="MOL",
    ).only(["molecule_structures"])

    smiles_raw = []
    t0 = time.time()
    for i, mol in enumerate(query):
        if i >= limit:
            break
        structs = mol.get("molecule_structures") or {}
        smi = structs.get("canonical_smiles") or structs.get("molfile")
        if smi and isinstance(smi, str) and len(smi) < 512:
            smiles_raw.append(smi)
        if (i + 1) % 10_000 == 0:
            print(f"  Fetched {i+1:,} records in {time.time()-t0:.0f}s …")

    print(f"[ChEMBL] Raw records fetched: {len(smiles_raw):,}")

    # Canonicalize and deduplicate
    seen, valid = set(), []
    for smi in smiles_raw:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can not in seen:
            seen.add(can)
            valid.append(can)

    print(f"[ChEMBL] Valid canonical SMILES: {len(valid):,}")
    return valid


# ── Step B: Murcko scaffold diversity sampling ────────────────────────────────

def scaffold_diversity_sample(smiles: list, target_n: int, seed: int = 42) -> list:
    """
    Group molecules by Murcko scaffold, then greedily sample one per scaffold
    until target_n is reached.  Ties broken by lexicographic scaffold order
    (deterministic).
    """
    print(f"[Scaffold] Computing Murcko scaffolds for {len(smiles):,} molecules …")
    scaffold_to_mols = defaultdict(list)
    failed = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            sc = ""   # molecules with no ring system → empty scaffold bucket
        scaffold_to_mols[sc].append(smi)

    print(f"  {len(scaffold_to_mols):,} unique scaffolds  ({failed} parse failures)")

    rng = np.random.RandomState(seed)
    scaffolds = sorted(scaffold_to_mols.keys())   # deterministic order

    selected = []
    # Pass 1: one molecule per scaffold
    for sc in scaffolds:
        mols = scaffold_to_mols[sc]
        selected.append(rng.choice(mols))
        if len(selected) >= target_n:
            break

    # Pass 2: if still under budget, sample more from the largest scaffold groups
    if len(selected) < target_n:
        remaining = target_n - len(selected)
        already   = set(selected)
        pool      = [m for sc in scaffolds for m in scaffold_to_mols[sc]
                     if m not in already]
        if pool:
            extra = rng.choice(pool, min(remaining, len(pool)), replace=False)
            selected.extend(extra.tolist())

    selected = selected[:target_n]
    print(f"[Scaffold] Selected {len(selected):,} molecules")
    return selected


# ── Step C+D: Embed with MolFormer + project to keys ─────────────────────────

def build_context_tensors(smiles: list, poc) -> tuple:
    """
    Embed with MolFormer (via poc.embed_smiles) and project to D_INNER-d keys.
    Returns (ctx_v [N,768], ctx_k [N,256], R [768,256])
    """
    print(f"[Embed] Embedding {len(smiles):,} molecules with MolFormer …")
    print(f"  Device: {DEVICE}")

    parts = []
    for i in range(0, len(smiles), EMBED_BATCH * 10):   # process in 10-batch chunks
        chunk = smiles[i : i + EMBED_BATCH * 10]
        emb   = poc.embed_smiles(chunk, bs=EMBED_BATCH)  # [chunk, 768]
        parts.append(emb.cpu())
        done  = min(i + len(chunk), len(smiles))
        if done % 5_000 == 0 or done == len(smiles):
            print(f"  {done:,} / {len(smiles):,} embedded")

    ctx_v = torch.cat(parts, dim=0).contiguous()   # [N, 768]
    print(f"[Embed] ctx_v shape: {ctx_v.shape}")

    # Fixed orthogonal projection — MUST match run_hopfield_poc.py (seed 42)
    # Check if Wk_fixed.pt already exists from a previous poc run
    wk_path = CONTEXT_CACHE / "Wk_fixed.pt"
    if wk_path.exists():
        R = torch.load(wk_path, weights_only=True)
        print(f"[Keys] Loaded Wk_fixed.pt from {wk_path}")
    else:
        torch.manual_seed(42)
        R    = torch.randn(768, D_INNER)
        R, _ = torch.linalg.qr(R)
        print(f"[Keys] Derived Wk_fixed from seed=42  (same as run_hopfield_poc.py)")

    ctx_k = (ctx_v @ R).contiguous()   # [N, 256]
    print(f"[Keys] ctx_k shape: {ctx_k.shape}")
    return ctx_v, ctx_k, R


# ── Save ─────────────────────────────────────────────────────────────────────

def save_artifacts(ctx_v, ctx_k, R, smiles, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    v_path   = out_dir / "chembl_ctx_v.pt"
    k_path   = out_dir / "chembl_ctx_k.pt"
    wk_path  = out_dir / "Wk_fixed.pt"
    smi_path = out_dir / "chembl_smiles.txt"

    torch.save(ctx_v, v_path)
    torch.save(ctx_k, k_path)
    torch.save(R,     wk_path)
    smi_path.write_text("\n".join(smiles))

    print(f"\n[Save] Artifacts written to {out_dir}")
    print(f"  chembl_ctx_v.pt   {tuple(ctx_v.shape)}  ({v_path.stat().st_size/1e6:.1f} MB)")
    print(f"  chembl_ctx_k.pt   {tuple(ctx_k.shape)}  ({k_path.stat().st_size/1e6:.1f} MB)")
    print(f"  chembl_smiles.txt  {len(smiles):,} molecules")
    print(f"  Wk_fixed.pt        {tuple(R.shape)}")


# ── Verification ──────────────────────────────────────────────────────────────

def verify(ctx_v, ctx_k, R):
    assert ctx_v.shape[1] == 768,    f"ctx_v dim mismatch: {ctx_v.shape}"
    assert ctx_k.shape[1] == D_INNER, f"ctx_k dim mismatch: {ctx_k.shape}"
    assert ctx_v.shape[0] == ctx_k.shape[0], "ctx_v/ctx_k row count mismatch"
    assert R.shape == (768, D_INNER), f"R shape mismatch: {R.shape}"

    # Spot-check: re-derive keys from values and compare
    ctx_k_check = (ctx_v[:5] @ R)
    assert torch.allclose(ctx_k[:5], ctx_k_check, atol=1e-5), \
        "Key re-derivation mismatch — Wk inconsistency!"

    # Check non-trivial dot products
    dots = (ctx_v[:100] @ ctx_v[:100].T)
    assert dots.std() > 0.01, "ctx_v appears degenerate (zero variance)"

    print(f"\n[Verify] All checks passed.")
    print(f"  ctx_v norm (mean): {ctx_v.norm(dim=1).mean():.3f}")
    print(f"  ctx_k norm (mean): {ctx_k.norm(dim=1).mean():.3f}")
    print(f"  ctx_v pairwise dot std (first 100): {dots.std():.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # Check if already built
    v_path = OUT_DIR / "chembl_ctx_v.pt"
    k_path = OUT_DIR / "chembl_ctx_k.pt"
    if v_path.exists() and k_path.exists():
        ctx_v = torch.load(v_path, weights_only=True)
        ctx_k = torch.load(k_path, weights_only=True)
        print(f"[Cache] ChEMBL context already exists: {ctx_v.shape[0]:,} molecules")
        R_path = OUT_DIR / "Wk_fixed.pt"
        R = torch.load(R_path, weights_only=True) if R_path.exists() else None
        if R is not None:
            verify(ctx_v, ctx_k, R)
        return

    print("=" * 68)
    print("Building ChEMBL Independent Context Set")
    print(f"Target: {TARGET_N:,} diverse molecules | Device: {DEVICE}")
    print("=" * 68)

    # Load poc for embed_smiles
    print("\n[Step 0] Loading run_hopfield_poc.py for MolFormer …")
    poc = _load_poc()

    # Step A
    print("\n[Step 1] Fetching ChEMBL SMILES …")
    raw_smiles = fetch_chembl_smiles(limit=CHEMBL_LIMIT)

    # Step B
    print("\n[Step 2] Scaffold diversity sampling …")
    selected   = scaffold_diversity_sample(raw_smiles, TARGET_N)

    # Steps C+D
    print("\n[Step 3] Embedding + projecting …")
    ctx_v, ctx_k, R = build_context_tensors(selected, poc)

    # Save
    print("\n[Step 4] Saving artifacts …")
    save_artifacts(ctx_v, ctx_k, R, selected, OUT_DIR)

    # Verify
    verify(ctx_v, ctx_k, R)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/3600:.1f} hrs  ({elapsed:.0f} s)")


if __name__ == "__main__":
    main()
