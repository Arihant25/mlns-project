#!/usr/bin/env python3
"""
run_hopfield_poc.py  —  meta/
================================
PoC 3: Hopfield Context Enrichment + Evidential GSL (FOMAML)

Full architecture (condition 5):
  raw_emb [N, 768]
    → HopfieldContext (cross-attn to frozen context set)   [outer loop only]
    → combined = alpha * enriched + (1-alpha) * raw
    → NIG_init(combined) → u_ep → G_i = 1 - gamma*sigmoid(u_ep)
    → MolAttention(combined) → A [N,N]   dense learned adjacency
    → A_gated[i,j] = A[i,j] * G_i * G_j   edge-wise epistemic gate
    → GCN: H = combined + norm(A_gated) @ combined @ W_gcn
    → NIG_final / Dir_final → prediction + calibrated uncertainty

Context set: all meta-training task molecules (train+val) subsampled to CTX_SIZE,
             pre-projected to D_INNER=256d via fixed orthogonal matrix.
             Keys [M, 256] and Values [M, 768] stored in data/context_set/.

5 Ablation conditions (clean ladder):
  1  maml_mlp               FOMAML + MolFormer + head  (floor, no graph)
  2  maml_static_gnn        + ECFP sparse GCN           (graph, fixed topology)
  3  maml_dense_gsl         + MolAttention + gate        (learned topology, no Hopfield)
  4  maml_hopfield_nogating + Hopfield enrichment        (MHNfs analog, no gate)
  5  maml_hopfield_evid     + Hopfield + entropy gate    (our model, context-level gate)
  6  maml_hopfield_two_stage + both gates                (context entropy + episode evidential)

Parameter group separation:
  Inner loop adapts: gcn, ih, fh, gam, mol_attn (task-specific)
  Outer loop only:   hopfield.Wq, hopfield.alpha  (shared context retrieval)

Results → meta/results_hopfield/
Set QUICK=1 for smoke-test (1 seed, K=10, 20 meta-episodes, 1k context molecules).
"""

import copy, csv, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.warning")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR)
LIPO_DATA     = os.path.join(PROJECT_ROOT, "lipo", "data")
CACO_DATA     = os.path.join(PROJECT_ROOT, "caco", "data")
EMBED_CACHE   = os.path.join(PROJECT_ROOT, "data", "embeddings")
CONTEXT_CACHE = os.path.join(PROJECT_ROOT, "data", "context_set")
RESULTS_DIR   = os.path.join(SCRIPT_DIR, "results_hopfield")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ────────────────────────────────────────────────────────────────────
QUICK = os.environ.get("QUICK", "").lower() in ("1", "true", "yes")

SEEDS            = [0]         if QUICK else [0, 1, 2, 3, 4]
K_SHOTS          = [10]        if QUICK else [5, 10, 20, 50]
META_EPISODES    = 20          if QUICK else 500
META_BATCH_TASKS = 4
INNER_STEPS      = 5
LR_INNER         = 0.01
LR_META          = 1e-3
MAX_QUERY        = 128
NIG_COEFF        = 0.1
DIR_COEFF        = 0.1
AUX_W            = 0.5
D_INNER          = 256                          # context attention hidden dim
CTX_SIZE         = 1000 if QUICK else (5000 if DEVICE.type == "cpu" else 50000)
MOLFORMER        = "ibm/MoLFormer-XL-both-10pct"

# Inner-loop adapts everything EXCEPT Hopfield query projection
HOPFIELD_KEYS = ["hopfield"]

CLS_TRAIN_TASKS = ["CYP2C9_Veith", "CYP2D6_Veith", "CYP3A4_Veith",
                   "BBB_Martins", "Pgp_Broccatelli"]
CLS_TEST_TASKS  = ["HIA_Hou", "CYP2C9_Substrate_CarbonMangels"]
REG_TRAIN_TASKS = ["Solubility_AqSolDB", "Lipophilicity_AstraZeneca", "PPBR_AZ"]
REG_TEST_TASKS  = ["Caco2_Wang", "VDss_Lombardo"]
CONDITIONS      = ["maml_mlp", "maml_static_gnn", "maml_dense_gsl",
                   "maml_hopfield_nogating", "maml_hopfield_evid",
                   "maml_hopfield_two_stage"]

_mf_cache: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# §1  EMBEDDING INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def _get_molformer():
    if "model" not in _mf_cache:
        from transformers import AutoTokenizer, AutoModel
        print(f"  [MolFormer] Loading {MOLFORMER} …")
        tok = AutoTokenizer.from_pretrained(MOLFORMER, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(MOLFORMER, trust_remote_code=True).to(DEVICE)
        mdl.eval()
        _mf_cache["tok"], _mf_cache["model"] = tok, mdl
    return _mf_cache["tok"], _mf_cache["model"]


def embed_smiles(smiles: list, bs: int = 32) -> torch.Tensor:
    tok, mdl = _get_molformer()
    parts = []
    for s in range(0, len(smiles), bs):
        b   = smiles[s:s + bs]
        inp = tok(b, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = mdl(**inp)
        hs   = out.last_hidden_state
        mask = inp["attention_mask"].unsqueeze(-1).float()
        parts.append(((hs * mask).sum(1) / mask.sum(1)).cpu())
    return torch.cat(parts)


def _load_tdc(name: str) -> dict:
    from tdc.single_pred import ADME
    data  = ADME(name=name)
    split = data.get_split(method="scaffold")
    return {sn: {"smi": split[key]["Drug"].tolist(),
                 "tgt": torch.tensor(split[key]["Y"].values.astype(np.float32))}
            for sn, key in [("train","train"), ("val","valid"), ("test","test")]}


def get_dataset_embeddings(name: str) -> dict:
    KNOWN  = {"Lipophilicity_AstraZeneca": LIPO_DATA, "Caco2_Wang": CACO_DATA}
    splits = ("train", "val", "test")

    if name in KNOWN:
        base, out = KNOWN[name], {}
        for sn in splits:
            emb = torch.load(os.path.join(base, f"{sn}_embeddings.pt"), weights_only=True)
            tgt = torch.load(os.path.join(base, f"{sn}_targets.pt"),    weights_only=True)
            sp  = os.path.join(base, f"{sn}_smiles.pt")
            if os.path.exists(sp):
                smi = torch.load(sp, weights_only=False)
            else:
                smi = _load_tdc(name)[sn]["smi"]
                torch.save(smi, sp)
            out[sn] = {"emb": emb, "tgt": tgt, "smi": smi}
        print(f"  [{name}] ← {base}"); return out

    cache = os.path.join(EMBED_CACHE, name)
    if all(os.path.exists(os.path.join(cache, f"{sn}_{k}.pt"))
           for sn in splits for k in ("embeddings", "targets", "smiles")):
        out = {}
        for sn in splits:
            out[sn] = {
                "emb": torch.load(os.path.join(cache, f"{sn}_embeddings.pt"), weights_only=True),
                "tgt": torch.load(os.path.join(cache, f"{sn}_targets.pt"),    weights_only=True),
                "smi": torch.load(os.path.join(cache, f"{sn}_smiles.pt"),     weights_only=False),
            }
        print(f"  [{name}] ← cache"); return out

    print(f"  [{name}] Generating (TDC + MolFormer) …")
    os.makedirs(cache, exist_ok=True)
    raw, out = _load_tdc(name), {}
    for sn in splits:
        ch = raw[sn]
        print(f"    Embedding {sn} ({len(ch['smi'])} mols) …")
        emb = embed_smiles(ch["smi"])
        torch.save(emb,       os.path.join(cache, f"{sn}_embeddings.pt"))
        torch.save(ch["tgt"], os.path.join(cache, f"{sn}_targets.pt"))
        torch.save(ch["smi"], os.path.join(cache, f"{sn}_smiles.pt"))
        out[sn] = {"emb": emb, "tgt": ch["tgt"], "smi": ch["smi"]}
        print(f"    → {emb.shape}")
    return out


def get_full_dataset(ds: dict) -> dict:
    emb = torch.cat([ds["train"]["emb"], ds["val"]["emb"], ds["test"]["emb"]])
    tgt = torch.cat([ds["train"]["tgt"], ds["val"]["tgt"], ds["test"]["tgt"]])
    smi = ds["train"]["smi"] + ds["val"]["smi"] + ds["test"]["smi"]
    return {"emb": emb, "tgt": tgt, "smi": smi}


# ══════════════════════════════════════════════════════════════════════════════
# §2  CONTEXT SET CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_context_set(all_train_datasets: dict, ctx_size: int):
    """
    Pool all meta-training task embeddings (train+val splits), subsample to
    ctx_size molecules, pre-project to D_INNER=256d via a fixed orthogonal
    matrix (seed 42, stored alongside for reproducibility).

    Returns:
        ctx_v  [M, 768]    — raw MolFormer values (frozen, used as V in attn)
        ctx_k  [M, D_INNER] — pre-projected keys  (frozen, used as K in attn)
    Both on CPU.  Move to DEVICE once in main.
    """
    ctx_v_path = os.path.join(CONTEXT_CACHE, f"ctx_values_{ctx_size}.pt")
    ctx_k_path = os.path.join(CONTEXT_CACHE, f"ctx_keys_{ctx_size}_{D_INNER}.pt")

    if os.path.exists(ctx_v_path) and os.path.exists(ctx_k_path):
        ctx_v = torch.load(ctx_v_path, weights_only=True)
        ctx_k = torch.load(ctx_k_path, weights_only=True)
        print(f"  [Context Set] Loaded ({ctx_v.shape[0]} mols, "
              f"{ctx_k.shape[1]}d keys) from cache")
        return ctx_v, ctx_k

    print(f"  [Context Set] Building from training task pool …")
    embs = []
    for name, ds in all_train_datasets.items():
        embs.append(ds["train"]["emb"])
        embs.append(ds["val"]["emb"])
    pool = torch.cat(embs, dim=0)                       # [M_all, 768]

    # Random diverse subsample (fixed seed for reproducibility)
    g    = torch.Generator(); g.manual_seed(42)
    perm = torch.randperm(pool.size(0), generator=g)[:ctx_size]
    ctx_v = pool[perm].contiguous()                     # [ctx_size, 768]

    # Fixed orthogonal projection 768 → D_INNER (produced once, saved alongside)
    torch.manual_seed(42)
    R     = torch.randn(768, D_INNER)
    R, _  = torch.linalg.qr(R)                         # [768, D_INNER]
    ctx_k = (ctx_v @ R).contiguous()                   # [ctx_size, D_INNER]

    os.makedirs(CONTEXT_CACHE, exist_ok=True)
    torch.save(ctx_v, ctx_v_path)
    torch.save(ctx_k, ctx_k_path)
    torch.save(R, os.path.join(CONTEXT_CACHE, "Wk_fixed.pt"))
    print(f"  [Context Set] Built {ctx_v.shape[0]} mols → {CONTEXT_CACHE}")
    return ctx_v, ctx_k


# ══════════════════════════════════════════════════════════════════════════════
# §3  ECFP + LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def ecfp_matrix(smiles: list) -> torch.Tensor:
    fps = [AllChem.GetMorganFingerprintAsBitVect(
               Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C"), 2, nBits=1024)
           for s in smiles]
    from rdkit.Chem import DataStructs
    n, A = len(fps), torch.zeros(len(fps), len(fps))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for j, v in enumerate(sims, i + 1):
            A[i, j] = A[j, i] = v
    return A


def nig_nll(y, mu, v, a, b):
    r = (y - mu) ** 2
    return (0.5 * torch.log(math.pi / v)
            - a * torch.log(b)
            + (a + 0.5) * torch.log(b + 0.5 * v * r)
            + torch.lgamma(a) - torch.lgamma(a + 0.5))

def nig_reg(y, mu, v, a, _b):
    return torch.abs(y - mu) * (2.0 * v + a)

def nig_loss(y, mu, v, a, b, coeff=NIG_COEFF):
    nll = nig_nll(y, mu, v, a, b)
    reg = nig_reg(y, mu, v, a, b)
    return (nll + coeff * torch.abs(y - mu).detach() * reg).mean()

def dirichlet_loss(y_oh, alpha, coeff=DIR_COEFF):
    S   = alpha.sum(-1, keepdim=True).clamp(1e-8)
    p   = alpha / S
    mse = ((y_oh - p) ** 2).sum(-1)
    var = (p * (1 - p) / (S + 1)).sum(-1)
    K   = alpha.size(-1)
    a_t = y_oh + (1 - y_oh) * alpha
    KL  = (torch.lgamma(a_t.sum(-1)) - math.lgamma(K)
           - torch.lgamma(a_t).sum(-1)
           + ((a_t - 1) * (torch.digamma(a_t)
              - torch.digamma(a_t.sum(-1, keepdim=True)))).sum(-1))
    return (mse + var + coeff * KL).mean()


# ══════════════════════════════════════════════════════════════════════════════
# §4  BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class HopfieldContext(nn.Module):
    """
    Context enrichment via cross-attention to a frozen background database.

    Q = Wq(episode_emb)  [N, D_INNER]   ← TRAINED (outer loop only)
    K = ctx_k            [M, D_INNER]   ← FIXED (pre-projected at startup)
    V = ctx_v            [M, 768]       ← FIXED (raw MolFormer embeddings)

    enriched = softmax(Q @ K.T / sqrt(D_INNER)) @ V   [N, 768]
    combined = sigmoid(alpha) * enriched + (1-sigmoid(alpha)) * raw

    alpha is a learnable scalar that controls trust in the context set.
    Initialized to 0 → sigmoid(0) = 0.5 → equal blend at start.
    """
    def __init__(self, d_in: int = 768, d_attn: int = D_INNER):
        super().__init__()
        self.Wq    = nn.Linear(d_in, d_attn, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))   # logit; sigmoid → blend
        self.scale = math.sqrt(d_attn)
        nn.init.xavier_uniform_(self.Wq.weight)

    def forward(self, X: torch.Tensor,
                ctx_k: torch.Tensor,
                ctx_v: torch.Tensor):
        """
        X:     [N, 768]
        ctx_k: [M, D_INNER]  (on same device as X)
        ctx_v: [M, 768]      (on same device as X)
        Returns:
          combined  [N, 768]  context-enriched embeddings
          A_hop     [N, M]    attention weights (used downstream for entropy gate)
        """
        Q        = self.Wq(X)                                       # [N, D_INNER]
        A_hop    = torch.softmax(Q @ ctx_k.T / self.scale, dim=-1)  # [N, M]
        enriched = A_hop @ ctx_v                                     # [N, 768]
        blend    = torch.sigmoid(self.alpha)
        combined = blend * enriched + (1.0 - blend) * X
        return combined, A_hop


class MolAttention(nn.Module):
    """
    Dense self-attention adjacency: every episode node attends to every other.
    No topk, no ECFP, no discrete edge decisions — much more stable at K=20.
    """
    def __init__(self, d: int = 768):
        super().__init__()
        self.Wq    = nn.Linear(d, d, bias=False)
        self.Wk    = nn.Linear(d, d, bias=False)
        self.scale = math.sqrt(d)
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(X); K = self.Wk(X)
        return torch.softmax(Q @ K.T / self.scale, dim=-1)   # [N, N]


class NIGHead(nn.Module):
    def __init__(self, d: int = 768, h1: int = 512, h2: int = 256, drop: float = 0.1):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(d, h1), nn.GELU(), nn.Dropout(drop),
                                   nn.Linear(h1, h2), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(h2, 4)
        self.sp   = nn.Softplus()

    def forward(self, x):
        r = self.head(self.net(x))
        return (r[:, 0],
                self.sp(r[:, 1]) + 1e-6,
                # clamp alpha: prevents a→1 (u→∞) which causes 0×∞ gate NaN
                self.sp(r[:, 2]).clamp(max=28.0) + 1.01,
                self.sp(r[:, 3]) + 1e-6)


class DirHead(nn.Module):
    def __init__(self, d: int = 768, h1: int = 512, h2: int = 256,
                 K: int = 2, drop: float = 0.1):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(d, h1), nn.GELU(), nn.Dropout(drop),
                                   nn.Linear(h1, h2), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(h2, K)
        self.sp   = nn.Softplus()

    def forward(self, x):
        return self.sp(self.head(self.net(x))) + 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# §5  MODEL DEFINITIONS  (5 ablation conditions)
# ══════════════════════════════════════════════════════════════════════════════

# Uniform API:
#   compute_loss(emb, tgt, smi=None, ctx_k=None, ctx_v=None) → scalar loss
#   predict(emb, smi=None, ctx_k=None, ctx_v=None)           → [N] predictions


# ── Condition 1: MAML-MLP ────────────────────────────────────────────────────
class MetaMLP(nn.Module):
    """Baseline: FOMAML + MolFormer + evidential head. No graph, no context."""
    def __init__(self, task_type: str = "reg"):
        super().__init__()
        self.task_type = task_type
        if task_type == "reg":
            self.head = NIGHead()
        else:
            self.head = nn.Sequential(
                nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 1))

    def _fwd(self, emb):
        return self.head(emb.to(DEVICE))

    def compute_loss(self, emb, tgt, smi=None, ctx_k=None, ctx_v=None):
        emb, tgt = emb.to(DEVICE), tgt.to(DEVICE)
        if self.task_type == "reg":
            mu, v, a, b = self._fwd(emb)
            return nig_loss(tgt, mu, v, a, b)
        return F.binary_cross_entropy_with_logits(self._fwd(emb).squeeze(-1), tgt)

    def predict(self, emb, smi=None, ctx_k=None, ctx_v=None):
        with torch.no_grad():
            out = self._fwd(emb)
        return out[0].cpu() if self.task_type == "reg" \
               else torch.sigmoid(out.squeeze(-1)).cpu()


# ── Condition 2: MAML-StaticGNN ──────────────────────────────────────────────
class MetaStaticGNN(nn.Module):
    """ECFP graph is FIXED; only GNN weights adapt in inner loop. No context."""
    def __init__(self, task_type: str = "reg", k: int = 5, d: int = 768):
        super().__init__()
        self.task_type = task_type
        self.k   = k
        self.gcn = nn.Linear(d, d)
        self.act = nn.GELU()
        self.head = NIGHead(d) if task_type == "reg" else nn.Sequential(
            nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 1))

    def _adj(self, smi):
        A  = ecfp_matrix(smi).to(DEVICE)
        k  = min(self.k, A.size(0) - 1)
        _, ti = A.topk(k, -1)
        M  = torch.zeros_like(A); M.scatter_(1, ti, 1.0)
        M  = ((M + M.T) > 0).float()
        A  = (A * M); A.fill_diagonal_(0.0)
        Ai = A + torch.eye(A.size(0), device=A.device)
        return Ai / Ai.sum(-1, keepdim=True).clamp(1e-8)

    def _gnn(self, emb, smi):
        return self.act(self.gcn(self._adj(smi) @ emb.to(DEVICE))) + emb.to(DEVICE)

    def compute_loss(self, emb, tgt, smi=None, ctx_k=None, ctx_v=None):
        H, tgt = self._gnn(emb, smi), tgt.to(DEVICE)
        if self.task_type == "reg":
            mu, v, a, b = self.head(H)
            return nig_loss(tgt, mu, v, a, b)
        return F.binary_cross_entropy_with_logits(self.head(H).squeeze(-1), tgt)

    def predict(self, emb, smi=None, ctx_k=None, ctx_v=None):
        with torch.no_grad():
            H = self._gnn(emb, smi)
            if self.task_type == "reg":
                return self.head(H)[0].cpu()
            return torch.sigmoid(self.head(H).squeeze(-1)).cpu()


# ── Condition 3: MAML-DenseGSL ───────────────────────────────────────────────
class MetaDenseGSL(nn.Module):
    """
    Dense MolAttention + edge-wise evidential gate.
    No ECFP, no Hopfield context — isolates the GSL contribution.
    """
    def __init__(self, task_type: str = "reg", d: int = 768):
        super().__init__()
        self.task_type  = task_type
        self.mol_attn   = MolAttention(d)
        self.gcn        = nn.Linear(d, d)
        self.act        = nn.GELU()
        # Small init: gate barely open at start — prevents 0×∞ instability
        self.gam        = nn.Parameter(torch.tensor(0.01))
        if task_type == "reg":
            self.ih = NIGHead(d); self.fh = NIGHead(d)
        else:
            self.ih = DirHead(d, K=2); self.fh = DirHead(d, K=2)

    def _fwd_reg(self, X):
        m0, v0, a0, b0 = self.ih(X)
        u0 = b0 / (a0 - 1.0).clamp(min=0.01)          # bounded: a0 ≥ 1.01 now
        G  = 1.0 - self.gam * torch.sigmoid(u0)
        A  = self.mol_attn(X)
        Ag = A * G.unsqueeze(0) * G.unsqueeze(1)
        # +1e-3 prevents 0×∞ when gate kills all edges
        D  = 1.0 / (Ag.sum(-1, keepdim=True) + 1e-3)
        M  = self.act(self.gcn((Ag * D) @ X))
        mu, v, al, be = self.fh(X + M)
        return (m0, v0, a0, b0), (mu, v, al, be)

    def _fwd_cls(self, X):
        a0 = self.ih(X)
        u0 = 2.0 / a0.sum(-1).clamp(1e-8)
        G  = 1.0 - self.gam * torch.sigmoid(u0)
        A  = self.mol_attn(X)
        Ag = A * G.unsqueeze(0) * G.unsqueeze(1)
        D  = 1.0 / (Ag.sum(-1, keepdim=True) + 1e-3)
        M  = self.act(self.gcn((Ag * D) @ X))
        return a0, self.fh(X + M)

    def compute_loss(self, emb, tgt, smi=None, ctx_k=None, ctx_v=None):
        emb, tgt = emb.to(DEVICE), tgt.to(DEVICE)
        if self.task_type == "reg":
            (m0,v0,a0,b0), (mu,v,al,be) = self._fwd_reg(emb)
            return nig_loss(tgt,mu,v,al,be) + AUX_W*nig_loss(tgt,m0,v0,a0,b0)
        # Classification: no aux loss — it biases the Hopfield/attention projections
        # toward encoding uncertainty rather than class-discriminative structure
        _, af = self._fwd_cls(emb)
        oh = F.one_hot(tgt.long(), 2).float()
        return dirichlet_loss(oh, af)

    def predict(self, emb, smi=None, ctx_k=None, ctx_v=None):
        emb = emb.to(DEVICE)
        with torch.no_grad():
            if self.task_type == "reg":
                _, (mu, *_) = self._fwd_reg(emb)
                return mu.cpu()
            _, af = self._fwd_cls(emb)
            return (af[:, 1] / af.sum(-1)).cpu()


# ── Condition 4: MAML-Hopfield-NoGating  (MHNfs analog) ─────────────────────
class MetaHopfieldNoGate(nn.Module):
    """
    Hopfield context enrichment + dense MolAttention.
    Gate is REMOVED (G=1 everywhere) — this is the MHNfs analog.
    Direct comparison: condition 5 minus the gate = this model.
    """
    def __init__(self, task_type: str = "reg", d: int = 768):
        super().__init__()
        self.task_type = task_type
        self.hopfield  = HopfieldContext(d, D_INNER)
        self.mol_attn  = MolAttention(d)
        self.gcn       = nn.Linear(d, d)
        self.act       = nn.GELU()
        if task_type == "reg":
            self.head = NIGHead(d)
        else:
            self.head = DirHead(d, K=2)

    def _fwd(self, X, ctx_k, ctx_v):
        X2, _ = self.hopfield(X, ctx_k, ctx_v)        # discard attn weights
        A  = self.mol_attn(X2)                         # dense adjacency, no gate
        D  = 1.0 / A.sum(-1, keepdim=True).clamp(1e-8)
        M  = self.act(self.gcn((A * D) @ X2))
        return self.head(X2 + M)

    def compute_loss(self, emb, tgt, smi=None, ctx_k=None, ctx_v=None):
        emb, tgt = emb.to(DEVICE), tgt.to(DEVICE)
        ctx_k    = ctx_k.to(DEVICE); ctx_v = ctx_v.to(DEVICE)
        out      = self._fwd(emb, ctx_k, ctx_v)
        if self.task_type == "reg":
            mu, v, a, b = out
            return nig_loss(tgt, mu, v, a, b)
        oh = F.one_hot(tgt.long(), 2).float()
        return dirichlet_loss(oh, out)

    def predict(self, emb, smi=None, ctx_k=None, ctx_v=None):
        emb   = emb.to(DEVICE)
        ctx_k = ctx_k.to(DEVICE); ctx_v = ctx_v.to(DEVICE)
        with torch.no_grad():
            out = self._fwd(emb, ctx_k, ctx_v)
        if self.task_type == "reg":
            return out[0].cpu()
        return (out[:, 1] / out.sum(-1)).cpu()


# ── Condition 5: MAML-Hopfield-Evidential  (our full model) ──────────────────
class MetaHopfieldEvidential(nn.Module):
    """
    Hopfield enrichment → Hopfield attention entropy gate →
    dense MolAttention + edge-wise gate → evidential prediction.

    Gate: G_i = 1 - H(A_hop[i]) / log(M)
      H = Shannon entropy of molecule i's attention over the context set.
      Low entropy  (sharp, focused retrieval) → G_i ≈ 1 → trust this molecule
      High entropy (diffuse, no neighbours)   → G_i ≈ 0 → OOD, sever its edges

    vs. MHNfs (hopfield_nogating): same enrichment, gate = 1 everywhere.
    vs. old NIGHead gate: no extra head, no extra parameters, no separate loss.
      The gate self-calibrates as Wq learns to focus attention on ADMET-relevant
      context molecules across meta-training tasks.
    """
    def __init__(self, task_type: str = "reg", d: int = 768):
        super().__init__()
        self.task_type = task_type
        self.hopfield  = HopfieldContext(d, D_INNER)
        self.mol_attn  = MolAttention(d)
        self.gcn       = nn.Linear(d, d)
        self.act       = nn.GELU()
        # No ih head, no gam — gate comes from Hopfield attention entropy only
        if task_type == "reg":
            self.fh = NIGHead(d)
        else:
            self.fh = DirHead(d, K=2)

    @staticmethod
    def _entropy_gate(A_hop: torch.Tensor) -> torch.Tensor:
        """
        A_hop: [N, M] softmax attention weights.
        Returns G: [N] in (0, 1).
        G_i = 1 - normalised_entropy(A_hop[i])
        """
        H     = -(A_hop * (A_hop + 1e-9).log()).sum(-1)   # [N] Shannon entropy
        H_max = math.log(A_hop.size(1))                   # log(M) = max entropy
        return (1.0 - H / H_max).clamp(0.0, 1.0)          # [N]

    def _fwd_reg(self, X, ctx_k, ctx_v):
        X2, A_hop = self.hopfield(X, ctx_k, ctx_v)
        G = self._entropy_gate(A_hop)                      # [N]
        A       = self.mol_attn(X2)
        A_gated = A * G.unsqueeze(0) * G.unsqueeze(1)     # sever OOD edges
        D       = 1.0 / (A_gated.sum(-1, keepdim=True) + 1e-3)
        M       = self.act(self.gcn((A_gated * D) @ X2))
        return self.fh(X2 + M)                            # (mu, v, al, be)

    def _fwd_cls(self, X, ctx_k, ctx_v):
        X2, A_hop = self.hopfield(X, ctx_k, ctx_v)
        G = self._entropy_gate(A_hop)
        A       = self.mol_attn(X2)
        A_gated = A * G.unsqueeze(0) * G.unsqueeze(1)
        D       = 1.0 / (A_gated.sum(-1, keepdim=True) + 1e-3)
        M       = self.act(self.gcn((A_gated * D) @ X2))
        return self.fh(X2 + M)                            # alpha [N, 2]

    def compute_loss(self, emb, tgt, smi=None, ctx_k=None, ctx_v=None):
        emb, tgt = emb.to(DEVICE), tgt.to(DEVICE)
        ctx_k    = ctx_k.to(DEVICE); ctx_v = ctx_v.to(DEVICE)
        if self.task_type == "reg":
            mu, v, al, be = self._fwd_reg(emb, ctx_k, ctx_v)
            return nig_loss(tgt, mu, v, al, be)
        af = self._fwd_cls(emb, ctx_k, ctx_v)
        oh = F.one_hot(tgt.long(), 2).float()
        return dirichlet_loss(oh, af)

    def predict(self, emb, smi=None, ctx_k=None, ctx_v=None):
        emb   = emb.to(DEVICE)
        ctx_k = ctx_k.to(DEVICE); ctx_v = ctx_v.to(DEVICE)
        with torch.no_grad():
            if self.task_type == "reg":
                mu, *_ = self._fwd_reg(emb, ctx_k, ctx_v)
                return mu.cpu()
            af = self._fwd_cls(emb, ctx_k, ctx_v)
            return (af[:, 1] / af.sum(-1)).cpu()


# ── Condition 6: MAML-Hopfield-TwoStage  (full two-level evidential gate) ──────
class MetaHopfieldTwoStage(nn.Module):
    """
    Two-level uncertainty gate:
      Gate 1 (context level, no params) — entropy gate from Hopfield attention:
        G_ctx_i = 1 - H(A_hop_i) / log(M)
        High entropy = diffuse retrieval = OOD = sever edges

      Gate 2 (episode level, learnable) — within-episode evidential gate:
        G_ep_i = 1 - gam * sigmoid(u_ep_i)
        u_ep from a NIG/Dir head run on X2.detach() — gradient is blocked
        from reaching Hopfield.Wq, so Wq learns context retrieval, not labeling

      Combined: G_i = G_ctx_i * G_ep_i   (both must pass for edge to survive)

    Chapter 1 showed G_ep alone helps within a single task.
    Chapter 2 showed G_ctx alone helps across tasks (hopfield_evid > nogating).
    This condition tests if they are ADDITIVE or REDUNDANT.
    """
    def __init__(self, task_type: str = "reg", d: int = 768):
        super().__init__()
        self.task_type = task_type
        self.hopfield  = HopfieldContext(d, D_INNER)
        self.mol_attn  = MolAttention(d)
        self.gcn       = nn.Linear(d, d)
        self.act       = nn.GELU()
        # gam=0.1: episode gate is meaningful from the start (not 0.01 like before)
        self.gam       = nn.Parameter(torch.tensor(0.1))
        # ih: episode-level uncertainty head (uses X2.detach — safe from Hopfield)
        # fh: final prediction head
        if task_type == "reg":
            self.ih = NIGHead(d); self.fh = NIGHead(d)
        else:
            self.ih = DirHead(d, K=2); self.fh = DirHead(d, K=2)

    @staticmethod
    def _entropy_gate(A_hop: torch.Tensor) -> torch.Tensor:
        H     = -(A_hop * (A_hop + 1e-9).log()).sum(-1)
        H_max = math.log(A_hop.size(1))
        return (1.0 - H / H_max).clamp(0.0, 1.0)

    def _episode_gate_reg(self, X2_detached: torch.Tensor) -> torch.Tensor:
        """Within-episode gate from NIG uncertainty. Input is X2.detach()."""
        _, _, a_ep, b_ep = self.ih(X2_detached)
        u_ep = b_ep / (a_ep - 1.0).clamp(min=0.01)
        return (1.0 - self.gam * torch.sigmoid(u_ep)).clamp(0.0, 1.0)

    def _episode_gate_cls(self, X2_detached: torch.Tensor) -> torch.Tensor:
        a_ep = self.ih(X2_detached)
        u_ep = 2.0 / a_ep.sum(-1).clamp(1e-8)
        return (1.0 - self.gam * torch.sigmoid(u_ep)).clamp(0.0, 1.0)

    def _fwd_reg(self, X, ctx_k, ctx_v):
        X2, A_hop = self.hopfield(X, ctx_k, ctx_v)

        G_ctx = self._entropy_gate(A_hop)                # [N] context quality
        G_ep  = self._episode_gate_reg(X2.detach())      # [N] episode uncertainty
        G     = G_ctx * G_ep                             # combined gate

        A       = self.mol_attn(X2)
        A_gated = A * G.unsqueeze(0) * G.unsqueeze(1)
        D       = 1.0 / (A_gated.sum(-1, keepdim=True) + 1e-3)
        M       = self.act(self.gcn((A_gated * D) @ X2))
        return self.fh(X2 + M)                           # (mu, v, al, be)

    def _fwd_cls(self, X, ctx_k, ctx_v):
        X2, A_hop = self.hopfield(X, ctx_k, ctx_v)

        G_ctx = self._entropy_gate(A_hop)
        G_ep  = self._episode_gate_cls(X2.detach())
        G     = G_ctx * G_ep

        A       = self.mol_attn(X2)
        A_gated = A * G.unsqueeze(0) * G.unsqueeze(1)
        D       = 1.0 / (A_gated.sum(-1, keepdim=True) + 1e-3)
        M       = self.act(self.gcn((A_gated * D) @ X2))
        return self.fh(X2 + M)                           # alpha [N, 2]

    def compute_loss(self, emb, tgt, smi=None, ctx_k=None, ctx_v=None):
        emb, tgt = emb.to(DEVICE), tgt.to(DEVICE)
        ctx_k    = ctx_k.to(DEVICE); ctx_v = ctx_v.to(DEVICE)
        if self.task_type == "reg":
            mu, v, al, be = self._fwd_reg(emb, ctx_k, ctx_v)
            return nig_loss(tgt, mu, v, al, be)
        af = self._fwd_cls(emb, ctx_k, ctx_v)
        oh = F.one_hot(tgt.long(), 2).float()
        return dirichlet_loss(oh, af)

    def predict(self, emb, smi=None, ctx_k=None, ctx_v=None):
        emb   = emb.to(DEVICE)
        ctx_k = ctx_k.to(DEVICE); ctx_v = ctx_v.to(DEVICE)
        with torch.no_grad():
            if self.task_type == "reg":
                mu, *_ = self._fwd_reg(emb, ctx_k, ctx_v)
                return mu.cpu()
            af = self._fwd_cls(emb, ctx_k, ctx_v)
            return (af[:, 1] / af.sum(-1)).cpu()


def make_model(cond: str, task_type: str) -> nn.Module:
    return {
        "maml_mlp":                 MetaMLP(task_type),
        "maml_static_gnn":          MetaStaticGNN(task_type),
        "maml_dense_gsl":           MetaDenseGSL(task_type),
        "maml_hopfield_nogating":   MetaHopfieldNoGate(task_type),
        "maml_hopfield_evid":       MetaHopfieldEvidential(task_type),
        "maml_hopfield_two_stage":  MetaHopfieldTwoStage(task_type),
    }[cond]


# ══════════════════════════════════════════════════════════════════════════════
# §6  FOMAML UTILITIES  (with Hopfield outer-loop separation)
# ══════════════════════════════════════════════════════════════════════════════

def fomaml_adapt(model, sup_emb, sup_tgt, sup_smi, ctx_k, ctx_v,
                 inner_steps, lr_inner):
    """
    Clone model, run inner_steps SGD steps on support set.
    Hopfield params are EXCLUDED from the inner optimizer →
    they only receive meta-gradients from the query loss (outer loop).
    Returns: adapted clone.
    """
    adapted   = copy.deepcopy(model).to(DEVICE)
    # Inner loop params: everything except Hopfield
    inner_ps  = [p for n, p in adapted.named_parameters()
                 if not any(k in n for k in HOPFIELD_KEYS)]
    inner_opt = torch.optim.SGD(inner_ps, lr=lr_inner)

    for _ in range(inner_steps):
        inner_opt.zero_grad()
        loss = adapted.compute_loss(sup_emb, sup_tgt,
                                    smi=sup_smi, ctx_k=ctx_k, ctx_v=ctx_v)
        if torch.isnan(loss) or torch.isinf(loss):
            break   # abort inner loop on NaN — keep last good weights
        loss.backward()
        nn.utils.clip_grad_norm_(inner_ps, max_norm=1.0)  # critical for NIG stability
        # Zero Hopfield grads: prevent support loss from reaching them
        for n, p in adapted.named_parameters():
            if any(k in n for k in HOPFIELD_KEYS) and p.grad is not None:
                p.grad.zero_()
        inner_opt.step()

    return adapted


def fomaml_outer_step(model, episodes, meta_opt, ctx_k, ctx_v):
    """
    One outer FOMAML step over a batch of episodes.
    Query-loss gradients from adapted clones are copied → original model params.
    All params (including Hopfield) receive meta-gradients from the QUERY loss.
    """
    meta_opt.zero_grad()
    total_loss = 0.0
    n_valid = 0
    n = len(episodes)

    for sup_emb, sup_tgt, sup_smi, qry_emb, qry_tgt, qry_smi in episodes:
        adapted = fomaml_adapt(model, sup_emb, sup_tgt, sup_smi,
                               ctx_k, ctx_v, INNER_STEPS, LR_INNER)

        q_loss  = adapted.compute_loss(qry_emb, qry_tgt,
                                        smi=qry_smi, ctx_k=ctx_k, ctx_v=ctx_v)
        if torch.isnan(q_loss) or torch.isinf(q_loss):
            continue   # skip degenerate episode — do not propagate NaN grads
        q_loss.backward()
        total_loss += q_loss.item()
        n_valid    += 1

        # FOMAML: copy adapted clone grads → original model
        for p_o, p_a in zip(model.parameters(), adapted.parameters()):
            if p_a.grad is not None:
                if p_o.grad is None:
                    p_o.grad = p_a.grad.clone() / n
                else:
                    p_o.grad += p_a.grad.clone() / n

    if n_valid == 0:
        return float("nan")   # entire batch was degenerate
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    meta_opt.step()
    return total_loss / n_valid


# ══════════════════════════════════════════════════════════════════════════════
# §7  EPISODE SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def sample_cls_episode(emb, tgt, smi, K_total, max_query=MAX_QUERY, rng=None):
    if rng is None: rng = np.random.RandomState()
    tgt_np = tgt.numpy()
    pos, neg = np.where(tgt_np == 1)[0], np.where(tgt_np == 0)[0]
    Kh = max(1, min(K_total // 2, len(pos) // 2, len(neg) // 2))
    sup_idx  = np.concatenate([rng.choice(pos, Kh, replace=False),
                                rng.choice(neg, Kh, replace=False)])
    qry_pool = np.setdiff1d(np.arange(len(emb)), sup_idx)
    if len(qry_pool) > max_query:
        qry_pool = rng.choice(qry_pool, max_query, replace=False)
    sup_smi = [smi[i] for i in sup_idx]
    qry_smi = [smi[i] for i in qry_pool]
    return (emb[sup_idx], tgt[sup_idx].float(), sup_smi,
            emb[qry_pool], tgt[qry_pool].float(), qry_smi)


def sample_reg_episode(emb, tgt_s, smi, K, max_query=MAX_QUERY, rng=None):
    if rng is None: rng = np.random.RandomState()
    n = len(emb); K = min(K, n // 2)
    sup_idx  = rng.choice(n, K, replace=False)
    qry_pool = np.setdiff1d(np.arange(n), sup_idx)
    if len(qry_pool) > max_query:
        qry_pool = rng.choice(qry_pool, max_query, replace=False)
    sup_smi = [smi[i] for i in sup_idx]
    qry_smi = [smi[i] for i in qry_pool]
    return (emb[sup_idx], tgt_s[sup_idx], sup_smi,
            emb[qry_pool], tgt_s[qry_pool], qry_smi)


# ══════════════════════════════════════════════════════════════════════════════
# §8  META-TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def scale_task(full_tgt):
    sc = StandardScaler()
    sc.fit(full_tgt.numpy().reshape(-1, 1))
    def t(x): return torch.tensor(
        sc.transform(x.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32)
    return sc, t


def meta_train(model, train_data: dict, task_type: str, seed: int,
               ctx_k: torch.Tensor, ctx_v: torch.Tensor):
    set_seed(seed)
    scaled, task_names = {}, list(train_data.keys())
    scalers = {}
    for name, d in train_data.items():
        if task_type == "reg":
            sc, tf = scale_task(d["tgt"])
            scalers[name] = sc; scaled[name] = tf(d["tgt"])
        else:
            scaled[name] = d["tgt"]

    model    = model.to(DEVICE)
    meta_opt = torch.optim.Adam(model.parameters(), lr=LR_META)
    rng      = np.random.RandomState(seed)

    for ep in range(1, META_EPISODES + 1):
        chosen = rng.choice(task_names, min(META_BATCH_TASKS, len(task_names)),
                            replace=len(task_names) < META_BATCH_TASKS)
        episodes = []
        for name in chosen:
            d = train_data[name]; tgt = scaled[name]
            K_ep = 10
            if task_type == "cls":
                ep_data = sample_cls_episode(d["emb"], tgt, d["smi"], K_ep, rng=rng)
            else:
                ep_data = sample_reg_episode(d["emb"], tgt, d["smi"], K_ep, rng=rng)
            episodes.append(ep_data)

        loss = fomaml_outer_step(model, episodes, meta_opt, ctx_k, ctx_v)
        if ep % max(1, META_EPISODES // 5) == 0:
            print(f"    Meta-ep {ep:4d}/{META_EPISODES}  query_loss={loss:.4f}")

    return model, scalers


# ══════════════════════════════════════════════════════════════════════════════
# §9  META-TESTING
# ══════════════════════════════════════════════════════════════════════════════

def meta_test_cls(model, test_full: dict, K, seed, ctx_k, ctx_v):
    rng = np.random.RandomState(seed)
    results = {}
    for name, d in test_full.items():
        ep = sample_cls_episode(d["emb"], d["tgt"], d["smi"], K,
                                max_query=9999, rng=rng)
        sup_emb, sup_tgt, sup_smi, qry_emb, qry_tgt, qry_smi = ep

        adapted = fomaml_adapt(model, sup_emb, sup_tgt, sup_smi,
                               ctx_k, ctx_v, INNER_STEPS, LR_INNER)
        prob    = adapted.predict(qry_emb, smi=qry_smi,
                                  ctx_k=ctx_k, ctx_v=ctx_v).numpy()
        tgt_np  = qry_tgt.numpy()
        results[name] = float(roc_auc_score(tgt_np, prob)) \
                        if len(np.unique(tgt_np)) >= 2 else float("nan")
    return results


def meta_test_reg(model, test_full: dict, K, seed, ctx_k, ctx_v):
    rng = np.random.RandomState(seed)
    results = {}
    for name, d in test_full.items():
        sc, tf  = scale_task(d["tgt"])
        tgt_s   = tf(d["tgt"])
        ep      = sample_reg_episode(d["emb"], tgt_s, d["smi"], K,
                                     max_query=9999, rng=rng)
        sup_emb, sup_tgt, sup_smi, qry_emb, qry_tgt, qry_smi = ep

        adapted = fomaml_adapt(model, sup_emb, sup_tgt, sup_smi,
                               ctx_k, ctx_v, INNER_STEPS, LR_INNER)
        mu_s    = adapted.predict(qry_emb, smi=qry_smi,
                                  ctx_k=ctx_k, ctx_v=ctx_v).numpy()
        mu_o    = sc.inverse_transform(mu_s.reshape(-1, 1)).flatten()
        tgt_o   = sc.inverse_transform(qry_tgt.numpy().reshape(-1, 1)).flatten()
        results[name] = float(np.sqrt(np.mean((mu_o - tgt_o) ** 2)))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# §10  HELPERS + RESULT I/O
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _row_key(seed: int, cond: str) -> str:
    return f"{seed}_{cond}"


def load_existing_csv(path: str):
    """Load existing results CSV. Returns (rows, done_set of 'seed_cond' keys)."""
    if not os.path.exists(path):
        return [], set()
    rows, done = [], set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            # Convert numeric fields back to float
            converted = {}
            for k, v in row.items():
                try:    converted[k] = float(v)
                except: converted[k] = v
            rows.append(converted)
            done.add(_row_key(int(float(row["seed"])), row["condition"]))
    return rows, done


def append_csv_row(row: dict, path: str):
    """Append one row to CSV, writing header if file doesn't exist yet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def write_summary(cls_rows, reg_rows, path):
    lines = ["=" * 68,
             "PoC 3 — Hopfield Context + Evidential GSL + FOMAML",
             "=" * 68]
    for label, rows, metric in [
        ("Classification (AUROC ↑)", cls_rows, "AUROC"),
        ("Regression     (RMSE  ↓)", reg_rows, "RMSE"),
    ]:
        lines += ["", label]
        for cond in CONDITIONS:
            lines.append(f"  {cond}:")
            cond_rows = [r for r in rows if r.get("condition") == cond]
            for K in K_SHOTS:
                key  = f"K{K}"
                vals = [r[key] for r in cond_rows
                        if key in r and not (isinstance(r[key], float) and math.isnan(r[key]))]
                if vals:
                    lines.append(f"    K={K:2d}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  [{metric}]")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# §11  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    CLS_CSV = os.path.join(RESULTS_DIR, "classification_results.csv")
    REG_CSV = os.path.join(RESULTS_DIR, "regression_results.csv")
    SUM_TXT = os.path.join(RESULTS_DIR, "summary.txt")

    print("=" * 68)
    print("PoC 3: Hopfield Context Enrichment + Evidential GSL + FOMAML")
    print(f"Device: {DEVICE}  |  K: {K_SHOTS}  |  Seeds: {SEEDS}"
          f"  |  Ctx: {CTX_SIZE}  |  Quick: {QUICK}")
    print("=" * 68)

    # ── §0  Load / cache embeddings ───────────────────────────────────────────
    print("\n[Step 0] Loading / caching embeddings …")
    all_names = list(dict.fromkeys(
        CLS_TRAIN_TASKS + CLS_TEST_TASKS + REG_TRAIN_TASKS + REG_TEST_TASKS))
    datasets  = {n: get_dataset_embeddings(n) for n in all_names}
    print("  All datasets ready.")

    # ── §1  Build context set ─────────────────────────────────────────────────
    print("\n[Step 1] Building context set …")
    all_train_ds = {n: datasets[n] for n in CLS_TRAIN_TASKS + REG_TRAIN_TASKS}
    ctx_v, ctx_k = build_context_set(all_train_ds, CTX_SIZE)
    ctx_v = ctx_v.to(DEVICE)
    ctx_k = ctx_k.to(DEVICE)

    # ── §2  Prepare splits ────────────────────────────────────────────────────
    cls_train_data = {n: datasets[n]["train"] for n in CLS_TRAIN_TASKS}
    reg_train_data = {n: datasets[n]["train"] for n in REG_TRAIN_TASKS}
    cls_test_full  = {n: get_full_dataset(datasets[n]) for n in CLS_TEST_TASKS}
    reg_test_full  = {n: get_full_dataset(datasets[n]) for n in REG_TEST_TASKS}

    # ── §3  Load existing results (skip already-done pairs) ───────────────────
    cls_rows, cls_done = load_existing_csv(CLS_CSV)
    reg_rows, reg_done = load_existing_csv(REG_CSV)
    if cls_done or reg_done:
        print(f"\n  [Cache] Skipping {len(cls_done)} CLS and {len(reg_done)} REG "
              f"already-completed (seed, condition) pairs.")

    # ── §4  Main loop ─────────────────────────────────────────────────────────
    for seed in SEEDS:
        print(f"\n{'═'*68}")
        print(f"SEED {seed}")
        print(f"{'═'*68}")
        set_seed(seed)

        for cond in CONDITIONS:
            key = _row_key(seed, cond)
            print(f"\n  [{cond}]")

            # ── Classification ────────────────────────────────────────────────
            # if key in cls_done:
            #     print(f"    [CLS] already done — skipping")
            # else:
            #     print(f"    [CLS] Meta-training …")
            #     cls_model, _ = meta_train(
            #         make_model(cond, "cls"), cls_train_data, "cls", seed,
            #         ctx_k, ctx_v)
            #     cls_row = {"seed": seed, "condition": cond}
            #     for K in K_SHOTS:
            #         t0  = time.perf_counter()
            #         res = meta_test_cls(cls_model, cls_test_full, K, seed,
            #                             ctx_k, ctx_v)
            #         avg = float(np.nanmean(list(res.values())))
            #         cls_row[f"K{K}"] = avg
            #         for tname, val in res.items():
            #             cls_row[f"{tname}_K{K}"] = val
            #         detail = "  ".join(f"{n}={v:.3f}" for n, v in res.items())
            #         print(f"    K={K:2d}: AUROC={avg:.4f}  [{detail}]  "
            #               f"({time.perf_counter()-t0:.0f}s)")
            #     append_csv_row(cls_row, CLS_CSV)
            #     cls_rows.append(cls_row)
            #     cls_done.add(key)
            #     write_summary(cls_rows, reg_rows, SUM_TXT)  # live update

            # ── Regression ────────────────────────────────────────────────────
            if key in reg_done:
                print(f"    [REG] already done — skipping")
            else:
                print(f"    [REG] Meta-training …")
                reg_model, _ = meta_train(
                    make_model(cond, "reg"), reg_train_data, "reg", seed,
                    ctx_k, ctx_v)
                reg_row = {"seed": seed, "condition": cond}
                for K in K_SHOTS:
                    t0  = time.perf_counter()
                    res = meta_test_reg(reg_model, reg_test_full, K, seed,
                                        ctx_k, ctx_v)
                    avg = float(np.nanmean(list(res.values())))
                    reg_row[f"K{K}"] = avg
                    for tname, val in res.items():
                        reg_row[f"{tname}_K{K}"] = val
                    detail = "  ".join(f"{n}={v:.3f}" for n, v in res.items())
                    print(f"    K={K:2d}: RMSE ={avg:.4f}  [{detail}]  "
                          f"({time.perf_counter()-t0:.0f}s)")
                append_csv_row(reg_row, REG_CSV)
                reg_rows.append(reg_row)
                reg_done.add(key)
                write_summary(cls_rows, reg_rows, SUM_TXT)  # live update

    # ── §5  Final summary ─────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    write_summary(cls_rows, reg_rows, SUM_TXT)
    print(f"\nAll results saved → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
