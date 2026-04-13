# Project Context: Evidential Graph Meta-Learning for Low-Data ADMET

**Last updated:** April 13, 2026

---

## Project Arc

### Chapter 1 — Single-Task Evidential GSL (Done)
Within a fixed dataset (e.g. Lipophilicity → Caco2), build a graph over K labeled
molecules and gate message passing by each node's evidential uncertainty:
- If node i is uncertain about its own label → suppress its messages to neighbours
- Gate: `G_i = 1 - γ * sigmoid(u_epistemic_i)`
- Showed: MiniMol + EvidentialGSL > non-gated GNN (SOTA on single-task)
- **Problem:** GSL and evidential heads both exist in literature. Not novel enough alone.

### Chapter 2 — Cross-Dataset Augmentation PoC (Done, inconclusive)
Use an evidential oracle trained on a source dataset to filter and weight pseudo-labelled
molecules for augmentation into a smaller target dataset.
- CYP2C9 → CYP2C9_Substrate (cls), Lipophilicity → Caco2 (reg)
- Results: similarity-based filtering helped regression but augmentation PoC was
  statistically noisy; K-shot regime showed bottleneck was data volume not architecture.

### Chapter 3 — Meta-Learning + Hopfield Context (Current, running)
Extend evidential GSL into FOMAML meta-learning, adding cross-dataset molecular
context enrichment (Hopfield mechanism) and entropy-based OOD gating.

---

## Current Architecture: `run_hopfield_poc.py`

**File:** `meta/run_hopfield_poc.py`
**Results:** `meta/results_hopfield/`

### Pipeline (condition 5 — hopfield_evid)
```
raw_emb [N, 768]  (MolFormer embeddings, N = K support molecules)
  → HopfieldContext.Wq (768→256, trained, OUTER LOOP ONLY)
  → cross-attn to frozen context set [M, 256 keys | M, 768 values]
    A_hop = softmax(Wq(X) @ ctx_k.T / sqrt(256))   [N, M]
    enriched = A_hop @ ctx_v                         [N, 768]
    combined = sigmoid(α) * enriched + (1-α) * X    [N, 768]
  → Entropy gate (no parameters):
    G_i = 1 - H(A_hop_i) / log(M)                  [N]  ∈ (0,1)
    (high entropy = diffuse = OOD → G ≈ 0)
  → MolAttention (within-episode dense GSL):
    A[i,j] = softmax(Wq_attn(X2)_i · Wk_attn(X2)_j / sqrt(768))  [N,N]
  → Edge-wise gate: A_gated[i,j] = A[i,j] * G_i * G_j
  → GCN: H = X2 + norm(A_gated) @ X2 @ W_gcn
  → NIGHead(H) → (μ, ν, α, β) → RMSE + calibrated uncertainty
     DirHead(H) → α_class            → AUROC
```

**What IS graph structure learning:** MolAttention computes a dynamic adjacency over
the K episode molecules from their enriched embeddings. Different episodes → different
graph topology. This IS learned GSL.

**What is NOT evidential (currently missing):** The within-episode gate that was in the
original Evidential GSL (gating GCN edges by the node's OWN label uncertainty) has been
removed. It was causing numerical instability (gam=1 init → G≈0 → 0×∞ NaN). The
gate that EXISTS now (entropy gate) is about CONTEXT RETRIEVAL quality, not label
uncertainty.

### 5 Ablation Conditions
| # | Name | What it has |
|---|---|---|
| 1 | `maml_mlp` | FOMAML + MolFormer + NIG/Dir head (no graph, no context) |
| 2 | `maml_static_gnn` | + ECFP fixed graph + GCN |
| 3 | `maml_dense_gsl` | + MolAttention dense GSL + evidential gate (no Hopfield) |
| 4 | `maml_hopfield_nogating` | + Hopfield context enrichment, gate=1 (MHNfs analog) |
| 5 | `maml_hopfield_evid` | + Hopfield + entropy gate (our model) |

### Context Set
- Source: all meta-training task molecules (train+val splits pooled), subsampled to CTX_SIZE
- CTX_SIZE: 1k (QUICK), 5k (CPU), 50k (GPU)
- Keys: fixed orthogonal projection 768→256 (seed 42, never trained)
- Values: raw MolFormer embeddings [M, 768]
- Limitation: NOT a real independent database (ChEMBL). TDC training pool only.

### FOMAML Parameter Split
- **Inner loop adapts:** gcn, fh, mol_attn (task-specific heads)
- **Outer loop only:** hopfield.Wq, hopfield.alpha (shared cross-task retrieval)
- Hopfield grads zeroed after each inner backward() to prevent support loss reaching Wq

---

## Results So Far (2-3 seeds of 5)

### Classification (AUROC ↑)
| Condition | K=5 | K=10 | K=20 | K=50 |
|---|---|---|---|---|
| maml_mlp | **0.6448** | **0.6501** | **0.6551** | **0.6601** |
| maml_static_gnn | 0.6330 | 0.6359 | 0.6413 | 0.6442 |
| maml_dense_gsl | 0.6264 | 0.6285 | 0.6355 | 0.6379 |
| maml_hopfield_nogating | 0.6278 | 0.6300 | 0.6336 | 0.6352 |
| **maml_hopfield_evid** | 0.6438 | 0.6471 | 0.6524 | 0.6529 |

### Regression (RMSE ↓)
| Condition | K=5 | K=10 | K=20 | K=50 |
|---|---|---|---|---|
| maml_mlp | 14.14 | 14.16 | 14.04 | 14.08 |
| maml_static_gnn | 14.23 | 14.28 | 14.11 | 14.18 |
| maml_dense_gsl | 14.54 | 14.54 | 14.54 | 14.40 |
| maml_hopfield_nogating | 13.17 | 13.20 | 13.17 | 13.24 |
| **maml_hopfield_evid** | **13.07** | **12.93** | **12.90** | **12.90** |

**Key findings:**
- Regression: Hopfield enrichment gives ~8% RMSE improvement over MLP/StaticGNN.
  Entropy gate gives additional ~0.27 RMSE improvement over nogating (marginal at 2-3 seeds).
- Classification: MLP is strongest. hopfield_evid ≈ MLP. hopfield_nogating is worst
  (blind enrichment hurts HIA_Hou specifically).
- Dense GSL without context (condition 3) is WORSE than MLP on regression → GSL alone
  with K<50 is unstable (K-shot bottleneck confirmed).
- NaN regression divergence: fixed via gam=0.01 init, a≥1.01 clamp, inner grad clip,
  additive epsilon in row norm.

---

## What Evidential GSL and Hopfield Gate Each Guard Against

| Gate | Guards against | Mechanism |
|---|---|---|
| Chapter 1: within-episode evidential gate | Noisy labeled molecules in support set corrupting GCN message passing | G_i = 1 - γ*sigmoid(u_ep) from NIG/Dir head prediction |
| Chapter 2: entropy gate (current) | OOD molecules whose Hopfield retrieval is unreliable (diffuse attention) | G_i = 1 - H(A_hop_i)/log(M) |

These are two different failure modes. They are NOT redundant.

---

## Open Questions / Next Steps

### Immediate (blocking paper narrative)
1. **Run hopfield_two_stage (condition 6):** Add within-episode evidential gate BACK
   on top of the entropy gate. Do NOT backprop through it (detach gradient from Hopfield).
   This directly tests: does gating within-episode label noise add to entropy OOD gating?
2. **Run real MHNfs** on Caco2_Wang and VDss_Lombardo with same scaffold splits.
   This makes `hopfield_nogating` vs MHNfs comparison concrete.
3. **Expand to 10+ TDC ADMET tasks.** 4 test tasks is too narrow for a paper.

### Medium-term (paper quality)
4. **ChEMBL independent context set** (50k, GPU): removes data proximity concern.
5. **G_i vs. prediction error correlation study:** prove entropy gate IS an OOD detector,
   not just a regulariser.
6. **Calibration experiments:** reliability diagrams, ECE for evid vs. nogating.
7. **Context size ablation:** 100 → 1k → 5k → 50k; show monotonic RMSE improvement.

---

## Paper Narrative (Target: NeurIPS/ICML or JCIM)

**Working title:** "Entropy-Gated Hopfield Meta-Learning for Few-Shot ADMET Prediction"

**Arc:**
1. K-shot ADMET prediction is bottlenecked by two failure modes:
   (a) Within-episode label noise (noisy support molecules corrupt GCN)
   (b) OOD context retrieval (Hopfield blindly enriches from irrelevant molecules)
2. Chapter 1 solves (a): evidential GSL gates within-episode message passing
3. This work solves (b): entropy-gated Hopfield retrieval detects when context is unreliable
4. Two-level system addresses both: entropy gate (context level) + evidential gate (episode level)

**Core claim:** "We show that knowing WHEN to trust retrieved context is as important as
having it — blind Hopfield retrieval (MHNfs mechanism) is suboptimal, and an
information-theoretic entropy gate delivers consistent improvement with zero additional
parameters."

**Evidence chain:**
- Within-episode gating: Chapter 1 results (EvidentialGSL single task)
- Context entropy gating: condition 5 > condition 4 (current results, regression)
- Two-level combination: PENDING (condition 6 = hopfield_two_stage)
- vs. actual MHNfs: PENDING (need to run MHNfs on same benchmarks)