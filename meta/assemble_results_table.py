#!/usr/bin/env python3
"""
assemble_results_table.py  —  meta/
======================================
Read all result CSVs, aggregate mean ± std across folds and seeds,
print LaTeX tables for the paper.

Sources:
    meta/results_hopfield/regression_results.csv   → ablation table (2 fixed test tasks)
    meta/results_hopfield/classification_results.csv → secondary CLS table
    meta/results_loto/loto_results.csv             → main LOTO table (8-task LOTO)

Aggregation for LOTO:
    1. Per fold: mean RMSE across seeds → 1 scalar per fold
    2. Across folds: mean ± std         → generalization estimate

Usage:
    python assemble_results_table.py
    python assemble_results_table.py --format latex    # LaTeX tabular
    python assemble_results_table.py --format markdown # GitHub markdown
    python assemble_results_table.py --format text     # plain text (default)
"""

import argparse, csv, math, os
from pathlib import Path
from collections import defaultdict

import numpy as np

SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR  = SCRIPT_DIR / "results_hopfield"
LOTO_DIR     = SCRIPT_DIR / "results_loto"

REG_CSV  = RESULTS_DIR / "regression_results.csv"
CLS_CSV  = RESULTS_DIR / "classification_results.csv"
LOTO_CSV = LOTO_DIR    / "loto_results.csv"

# Display names for conditions
DISPLAY_NAMES = {
    "maml_mlp":                  "MAML-MLP",
    "maml_static_gnn":           "MAML-StaticGNN",
    "maml_dense_gsl":            "MAML-DenseGSL",
    "maml_hopfield_nogating":    "MHNfs (re-impl.)",
    "maml_hopfield_evid":        "Ours (entropy gate)",
    "maml_hopfield_two_stage":   "Ours (two-stage)$^\\dagger$",
}

DISPLAY_NAMES_MD = {
    "maml_mlp":                  "MAML-MLP",
    "maml_static_gnn":           "MAML-StaticGNN",
    "maml_dense_gsl":            "MAML-DenseGSL",
    "maml_hopfield_nogating":    "MHNfs (re-impl.)",
    "maml_hopfield_evid":        "Ours (entropy gate)",
    "maml_hopfield_two_stage":   "**Ours (two-stage)**",
}

CATEGORY_1 = ["maml_mlp", "maml_static_gnn", "maml_dense_gsl"]
CATEGORY_2 = ["maml_hopfield_nogating", "maml_hopfield_evid", "maml_hopfield_two_stage"]
ALL_CONDS  = CATEGORY_1 + CATEGORY_2

K_SHOTS_REG  = [5, 10, 20, 50]
K_SHOTS_LOTO = [10, 20, 50]
K_SHOTS_CLS  = [5, 10, 20, 50]

LOTO_TASKS = [
    "Caco2_Wang", "VDss_Lombardo", "PPBR_AZ", "Lipophilicity_AstraZeneca",
    "Solubility_AqSolDB", "Clearance_Microsome_AZ", "Clearance_Hepatocyte_AZ",
]


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = dict(r)
            for k, v in row.items():
                try:    row[k] = float(v)
                except: pass
            rows.append(row)
    return rows


# ── Aggregation helpers ───────────────────────────────────────────────────────

def agg_poc_reg(rows: list, cond: str, K: int) -> tuple:
    """Mean ± std across seeds for a fixed-test-task ablation (poc runs)."""
    key  = f"K{K}"
    vals = [r[key] for r in rows
            if r.get("condition") == cond
            and not isinstance(r.get(key), str)
            and not math.isnan(float(r.get(key, float("nan"))))]
    if not vals:
        return None, None
    return np.mean(vals), np.std(vals)


def agg_loto(rows: list, cond: str, K: int, normalize: bool = True) -> tuple:
    """
    LOTO aggregation:
        1. Per fold: mean RMSE across seeds → optionally divide by task_std
        2. Across folds: mean ± std of (normalized) RMSE

    normalize=True  → nRMSE = RMSE / task_std  (dimensionless, cross-task comparable)
    normalize=False → raw RMSE (different units per fold, not directly averaged)
    """
    key = f"K{K}"
    fold_means = []
    for fold in LOTO_TASKS:
        fold_rows = [r for r in rows
                     if r.get("condition") == cond
                     and r.get("held_out") == fold
                     and isinstance(r.get(key), (int, float))
                     and not math.isnan(float(r.get(key, float("nan"))))]
        if not fold_rows:
            continue
        rmse_vals = [float(r[key]) for r in fold_rows]
        fold_rmse = np.mean(rmse_vals)
        if normalize:
            # task_std stored in CSV; fall back to 1.0 if missing
            task_std = float(fold_rows[0].get("task_std", 1.0) or 1.0)
            fold_means.append(fold_rmse / task_std if task_std > 0 else fold_rmse)
        else:
            fold_means.append(fold_rmse)
    if not fold_means:
        return None, None
    return np.mean(fold_means), np.std(fold_means)


def agg_cls(rows: list, cond: str, K: int) -> tuple:
    key  = f"K{K}"
    vals = [float(r[key]) for r in rows
            if r.get("condition") == cond
            and isinstance(r.get(key), (int, float))
            and not math.isnan(float(r.get(key, float("nan"))))]
    if not vals:
        return None, None
    return np.mean(vals), np.std(vals)


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_cell(mu, std, fmt="text", bold=False) -> str:
    if mu is None:
        return "—"
    s = f"{mu:.4f}±{std:.4f}"
    if fmt == "latex":
        inner = f"{mu:.4f} $\\pm$ {std:.4f}"
        return f"\\textbf{{{inner}}}" if bold else inner
    if fmt == "markdown":
        return f"**{s}**" if bold else s
    return ("*" + s + "*") if bold else s


def is_best_reg(cond: str, cond_vals: dict, K: int) -> bool:
    """True if this condition has the lowest mean RMSE at this K."""
    mu_self = cond_vals.get((cond, K), (None, None))[0]
    if mu_self is None:
        return False
    return all(
        cond_vals.get((c, K), (None, None))[0] is None
        or mu_self <= cond_vals.get((c, K), (None, None))[0]
        for c in ALL_CONDS
    )


def is_best_cls(cond: str, cond_vals: dict, K: int) -> bool:
    """True if this condition has the highest mean AUROC at this K."""
    mu_self = cond_vals.get((cond, K), (None, None))[0]
    if mu_self is None:
        return False
    return all(
        cond_vals.get((c, K), (None, None))[0] is None
        or mu_self >= cond_vals.get((c, K), (None, None))[0]
        for c in ALL_CONDS
    )


# ── Table printers ────────────────────────────────────────────────────────────

def print_section_header(title: str, fmt: str):
    if fmt == "latex":
        print(f"\n% ── {title} ──")
    elif fmt == "markdown":
        print(f"\n### {title}\n")
    else:
        print(f"\n{'─'*68}")
        print(f"  {title}")
        print(f"{'─'*68}")


def print_loto_table(loto_rows: list, fmt: str = "text"):
    if not loto_rows:
        print("  [No LOTO results found]")
        return

    # Pre-compute all means for bolding (normalized RMSE)
    cond_vals = {}
    for cond in ALL_CONDS:
        for K in K_SHOTS_LOTO:
            cond_vals[(cond, K)] = agg_loto(loto_rows, cond, K, normalize=True)

    Ks = K_SHOTS_LOTO

    if fmt == "latex":
        cols = " & ".join([f"K={K}" for K in Ks])
        print(r"\begin{table}[t]")
        print(r"\caption{Few-Shot ADMET Regression (nRMSE $\downarrow$), LOTO Cross-Validation (7 tasks). nRMSE = RMSE / $\sigma_\text{task}$, averaged across folds.}")
        print(r"\label{tab:loto}")
        print(r"\centering")
        print(r"\begin{tabular}{l" + "c" * len(Ks) + r"}")
        print(r"\toprule")
        print(f"Model & {cols} \\\\ \\midrule")
        print(r"\multicolumn{" + str(len(Ks)+1) + r"}{l}{\textit{Category 1: Strict Few-Shot (No Context Set)}} \\")
        for cond in CATEGORY_1:
            cells = []
            for K in Ks:
                mu, std = cond_vals[(cond, K)]
                cells.append(fmt_cell(mu, std, "latex", is_best_reg(cond, cond_vals, K)))
            print(f"{DISPLAY_NAMES[cond]} & {' & '.join(cells)} \\\\")
        print(r"\midrule")
        print(r"\multicolumn{" + str(len(Ks)+1) + r"}{l}{\textit{Category 2: Context-Augmented Few-Shot}} \\")
        for cond in CATEGORY_2:
            cells = []
            for K in Ks:
                mu, std = cond_vals[(cond, K)]
                cells.append(fmt_cell(mu, std, "latex", is_best_reg(cond, cond_vals, K)))
            print(f"{DISPLAY_NAMES[cond]} & {' & '.join(cells)} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\begin{tablenotes}")
        print(r"\item $^\dagger$ Two-stage gating: entropy gate (context) + evidential gate (episode).")
        print(r"\item MHNfs re-impl.: Hopfield cross-attention without gating, within our FOMAML framework.")
        print(r"\end{tablenotes}")
        print(r"\end{table}")

    elif fmt == "markdown":
        header = "| Model | " + " | ".join(f"K={K}" for K in Ks) + " |"
        sep    = "|-------|" + "|".join(["--------"] * len(Ks)) + "|"
        print(header)
        print(sep)
        print("| **Category 1: No Context** | | | |")
        for cond in CATEGORY_1:
            cells = [fmt_cell(*cond_vals[(cond, K)], "markdown",
                               is_best_reg(cond, cond_vals, K)) for K in Ks]
            print(f"| {DISPLAY_NAMES_MD[cond]} | " + " | ".join(cells) + " |")
        print("| **Category 2: Context-Augmented** | | | |")
        for cond in CATEGORY_2:
            cells = [fmt_cell(*cond_vals[(cond, K)], "markdown",
                               is_best_reg(cond, cond_vals, K)) for K in Ks]
            print(f"| {DISPLAY_NAMES_MD[cond]} | " + " | ".join(cells) + " |")

    else:  # text
        col_w = 26
        print(f"{'Model':<{col_w}}" + "".join(f"{'K='+str(K):>18}" for K in Ks))
        print("─" * (col_w + 18 * len(Ks)))
        print("Category 1: Strict Few-Shot (No Context Set)")
        for cond in CATEGORY_1:
            cells = [fmt_cell(*cond_vals[(cond, K)], "text",
                               is_best_reg(cond, cond_vals, K)) for K in Ks]
            print(f"  {DISPLAY_NAMES[cond]:<{col_w-2}}" + "".join(f"{c:>18}" for c in cells))
        print("\nCategory 2: Context-Augmented Few-Shot")
        for cond in CATEGORY_2:
            cells = [fmt_cell(*cond_vals[(cond, K)], "text",
                               is_best_reg(cond, cond_vals, K)) for K in Ks]
            print(f"  {DISPLAY_NAMES[cond]:<{col_w-2}}" + "".join(f"{c:>18}" for c in cells))


def print_ablation_table(reg_rows: list, fmt: str = "text"):
    """Ablation table from the original 2-test-task PoC runs."""
    if not reg_rows:
        print("  [No ablation regression results found]")
        return

    cond_vals = {}
    for cond in ALL_CONDS:
        for K in K_SHOTS_REG:
            cond_vals[(cond, K)] = agg_poc_reg(reg_rows, cond, K)

    Ks = K_SHOTS_REG

    if fmt == "text":
        col_w = 26
        print(f"{'Model':<{col_w}}" + "".join(f"{'K='+str(K):>18}" for K in Ks))
        print("─" * (col_w + 18 * len(Ks)))
        print("Category 1: Strict Few-Shot (No Context Set)")
        for cond in CATEGORY_1:
            cells = [fmt_cell(*cond_vals[(cond, K)], "text",
                               is_best_reg(cond, cond_vals, K)) for K in Ks]
            print(f"  {DISPLAY_NAMES[cond]:<{col_w-2}}" + "".join(f"{c:>18}" for c in cells))
        print("\nCategory 2: Context-Augmented Few-Shot")
        for cond in CATEGORY_2:
            cells = [fmt_cell(*cond_vals[(cond, K)], "text",
                               is_best_reg(cond, cond_vals, K)) for K in Ks]
            print(f"  {DISPLAY_NAMES[cond]:<{col_w-2}}" + "".join(f"{c:>18}" for c in cells))

    elif fmt == "latex":
        cols = " & ".join([f"K={K}" for K in Ks])
        print(r"\begin{table}[h]")
        print(r"\caption{Ablation: Few-Shot ADMET Regression (RMSE $\downarrow$), 2 Test Tasks (Caco2, VDss)}")
        print(r"\label{tab:ablation}")
        print(r"\centering")
        print(r"\begin{tabular}{l" + "c" * len(Ks) + r"}")
        print(r"\toprule")
        print(f"Model & {cols} \\\\ \\midrule")
        for cat, label in [(CATEGORY_1, "Category 1: No Context"), (CATEGORY_2, "Category 2: Context-Augmented")]:
            print(r"\multicolumn{" + str(len(Ks)+1) + r"}{l}{\textit{" + label + r"}} \\")
            for cond in cat:
                cells = [fmt_cell(*cond_vals[(cond, K)], "latex",
                                   is_best_reg(cond, cond_vals, K)) for K in Ks]
                print(f"{DISPLAY_NAMES[cond]} & {' & '.join(cells)} \\\\")
            if cat == CATEGORY_1:
                print(r"\midrule")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

    elif fmt == "markdown":
        header = "| Model | " + " | ".join(f"K={K}" for K in Ks) + " |"
        sep    = "|-------|" + "|".join(["--------"] * len(Ks)) + "|"
        print(header); print(sep)
        for cat, label in [(CATEGORY_1, "**Category 1: No Context**"), (CATEGORY_2, "**Context-Augmented**")]:
            print(f"| {label} | " + " | ".join([""] * len(Ks)) + " |")
            for cond in cat:
                cells = [fmt_cell(*cond_vals[(cond, K)], "markdown",
                                   is_best_reg(cond, cond_vals, K)) for K in Ks]
                print(f"| {DISPLAY_NAMES_MD[cond]} | " + " | ".join(cells) + " |")


def print_cls_table(cls_rows: list, fmt: str = "text"):
    if not cls_rows:
        print("  [No classification results found]")
        return

    cond_vals = {}
    for cond in ALL_CONDS:
        for K in K_SHOTS_CLS:
            cond_vals[(cond, K)] = agg_cls(cls_rows, cond, K)

    Ks = K_SHOTS_CLS
    col_w = 26

    if fmt == "text":
        print(f"{'Model':<{col_w}}" + "".join(f"{'K='+str(K):>18}" for K in Ks))
        print("─" * (col_w + 18 * len(Ks)))
        print("Category 1: Strict Few-Shot (No Context Set)")
        for cond in CATEGORY_1:
            cells = [fmt_cell(*cond_vals[(cond, K)], "text",
                               is_best_cls(cond, cond_vals, K)) for K in Ks]
            print(f"  {DISPLAY_NAMES[cond]:<{col_w-2}}" + "".join(f"{c:>18}" for c in cells))
        print("\nCategory 2: Context-Augmented Few-Shot")
        for cond in CATEGORY_2:
            cells = [fmt_cell(*cond_vals[(cond, K)], "text",
                               is_best_cls(cond, cond_vals, K)) for K in Ks]
            print(f"  {DISPLAY_NAMES[cond]:<{col_w-2}}" + "".join(f"{c:>18}" for c in cells))

    elif fmt == "latex":
        cols = " & ".join([f"K={K}" for K in Ks])
        print(r"\begin{table}[h]")
        print(r"\caption{Few-Shot ADMET Classification (AUROC $\uparrow$), 2 Test Tasks (HIA, CYP2C9-sub)}")
        print(r"\label{tab:cls}")
        print(r"\centering")
        print(r"\begin{tabular}{l" + "c" * len(Ks) + r"}")
        print(r"\toprule")
        print(f"Model & {cols} \\\\ \\midrule")
        for cat, label in [(CATEGORY_1, "Category 1: No Context"), (CATEGORY_2, "Category 2: Context-Augmented")]:
            print(r"\multicolumn{" + str(len(Ks)+1) + r"}{l}{\textit{" + label + r"}} \\")
            for cond in cat:
                cells = [fmt_cell(*cond_vals[(cond, K)], "latex",
                                   is_best_cls(cond, cond_vals, K)) for K in Ks]
                print(f"{DISPLAY_NAMES[cond]} & {' & '.join(cells)} \\\\")
            if cat == CATEGORY_1:
                print(r"\midrule")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

    elif fmt == "markdown":
        header = "| Model | " + " | ".join(f"K={K}" for K in Ks) + " |"
        sep    = "|-------|" + "|".join(["--------"] * len(Ks)) + "|"
        print(header); print(sep)
        for cat, label in [(CATEGORY_1, "**Category 1: No Context**"), (CATEGORY_2, "**Context-Augmented**")]:
            print(f"| {label} | " + " | ".join([""] * len(Ks)) + " |")
            for cond in cat:
                cells = [fmt_cell(*cond_vals[(cond, K)], "markdown",
                                   is_best_cls(cond, cond_vals, K)) for K in Ks]
                print(f"| {DISPLAY_NAMES_MD[cond]} | " + " | ".join(cells) + " |")


def print_per_task_table(loto_rows: list, cond: str, fmt: str = "text"):
    """Appendix table: per-task RMSE breakdown for a given condition."""
    if not loto_rows:
        return
    Ks = K_SHOTS_LOTO
    print(f"\n  Per-task breakdown: {DISPLAY_NAMES.get(cond, cond)}")

    cond_rows = [r for r in loto_rows if r.get("condition") == cond]
    if fmt in ("text", "markdown"):
        header = f"| {'Task':<30} | " + " | ".join(f"K={K}" for K in Ks) + " |"
        sep    = "|" + "-" * 32 + "|" + "|".join(["-" * 10] * len(Ks)) + "|"
        print(header); print(sep)
        for fold in LOTO_TASKS:
            cells = []
            for K in Ks:
                vals = [float(r[f"K{K}"]) for r in cond_rows
                        if r.get("held_out") == fold
                        and not math.isnan(float(r.get(f"K{K}", float("nan"))))]
                if vals:
                    cells.append(f"{np.mean(vals):.4f}±{np.std(vals):.4f}")
                else:
                    cells.append("—")
            print(f"| {fold:<30} | " + " | ".join(f"{c:<10}" for c in cells) + " |")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Assemble paper results tables")
    p.add_argument("--format", choices=["text", "latex", "markdown"],
                   default="text", help="Output format (default: text)")
    p.add_argument("--per-task", action="store_true",
                   help="Also print per-task breakdown (appendix)")
    args = p.parse_args()
    fmt  = args.format

    reg_rows  = load_csv(REG_CSV)
    cls_rows  = load_csv(CLS_CSV)
    loto_rows = load_csv(LOTO_CSV)

    print("=" * 68)
    print("Paper Results Tables — Entropy-Gated Hopfield Meta-Learning")
    print("=" * 68)
    print(f"\nLoaded: {len(reg_rows)} ablation-reg rows | "
          f"{len(cls_rows)} ablation-cls rows | "
          f"{len(loto_rows)} LOTO rows")

    # ── Table 1: Main LOTO results ────────────────────────────────────────────
    print_section_header(
        "TABLE 1 (MAIN): LOTO Regression  nRMSE↓ = RMSE/σ_task  (7 tasks × 5 seeds)", fmt)
    if loto_rows:
        folds_seen = set(r["held_out"] for r in loto_rows)
        conds_seen = set(r["condition"] for r in loto_rows)
        print(f"  Folds present: {sorted(folds_seen)}")
        print(f"  Conditions:    {sorted(conds_seen)}\n")
    print_loto_table(loto_rows, fmt)

    # ── Table 2: Ablation (original 2-task PoC) ───────────────────────────────
    print_section_header(
        "TABLE 2 (ABLATION): PoC Regression  RMSE↓  (Caco2 + VDss, 5 seeds)", fmt)
    print_ablation_table(reg_rows, fmt)

    # ── Table 3: Classification ───────────────────────────────────────────────
    print_section_header(
        "TABLE 3 (SUPPLEMENTARY): PoC Classification  AUROC↑  (HIA + CYP2C9-sub, 5 seeds)", fmt)
    print_cls_table(cls_rows, fmt)

    # ── Appendix: per-task breakdown ──────────────────────────────────────────
    if args.per_task and loto_rows:
        print_section_header("APPENDIX: Per-Task LOTO Breakdown", fmt)
        for cond in ["maml_hopfield_two_stage", "maml_hopfield_evid",
                     "maml_hopfield_nogating"]:
            print_per_task_table(loto_rows, cond, fmt)

    print()


if __name__ == "__main__":
    main()
