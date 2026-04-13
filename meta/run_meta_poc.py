#!/usr/bin/env python3
"""
run_meta_poc.py  —  meta/
==========================
PoC 2: Evidential Graph Meta-Learning (FOMAML)

Classification Meta-Learner
  Train tasks: CYP2C9_Veith, CYP2D6_Veith, CYP3A4_Veith, BBB_Martins, Pgp_Broccatelli
  Test  tasks: HIA_Hou, CYP2C9_Substrate

Regression Meta-Learner
  Train tasks: AqSol_DB, Lipophilicity_AstraZeneca, PPBR_AZ
  Test  tasks: Caco2_Wang, VDss_Lombardo

3 ablation conditions × K∈{10,20,50} × 5 seeds.

Results → meta/results/
  classification_meta_results.csv
  regression_meta_results.csv
  summary.txt

Set QUICK=1 for a fast smoke-test.
"""

import copy, csv, math, os, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.warning")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LIPO_DATA    = os.path.join(PROJECT_ROOT, "lipo", "data")
CACO_DATA    = os.path.join(PROJECT_ROOT, "caco", "data")
EMBED_CACHE  = os.path.join(PROJECT_ROOT, "data", "embeddings")
RESULTS_DIR  = os.path.join(SCRIPT_DIR,   "results")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ───────────────────────────────────────────────────────────────────
QUICK = os.environ.get("QUICK", "").lower() in ("1", "true", "yes")

SEEDS            = [0] if QUICK else [0, 1, 2, 3, 4]
K_SHOTS          = [10] if QUICK else [10, 20, 50]
META_EPISODES    = 10   if QUICK else 200     # outer FOMAML steps
META_BATCH_TASKS = 4                          # tasks per outer step
INNER_STEPS      = 5
LR_INNER         = 0.05
LR_META          = 1e-3
MAX_QUERY        = 128    # cap query set size per episode
NIG_COEFF        = 0.1
DIR_COEFF        = 0.1
AUX_W            = 0.5
MOLFORMER        = "ibm/MoLFormer-XL-both-10pct"

CLS_TRAIN_TASKS  = ["CYP2C9_Veith", "CYP2D6_Veith", "CYP3A4_Veith",
                     "BBB_Martins", "Pgp_Broccatelli"]
CLS_TEST_TASKS   = ["HIA_Hou", "CYP2C9_Substrate_CarbonMangels"]
REG_TRAIN_TASKS  = ["Solubility_AqSolDB", "Lipophilicity_AstraZeneca", "PPBR_AZ"]
REG_TEST_TASKS   = ["Caco2_Wang", "VDss_Lombardo"]
CONDITIONS       = ["maml_mlp", "maml_static_gnn", "maml_evidential_gsl"]

_mf_cache: dict = {}

# ══════════════════════════════════════════════════════════════════════════════
# §1  EMBEDDING INFRASTRUCTURE  (identical to augmentation script)
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
    """Priority: known pre-computed → cache → generate."""
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
                print(f"  [{name}/{sn}] fetching smiles from TDC …")
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
    """Concatenate train+val+test for meta-test tasks."""
    emb = torch.cat([ds["train"]["emb"], ds["val"]["emb"], ds["test"]["emb"]])
    tgt = torch.cat([ds["train"]["tgt"], ds["val"]["tgt"], ds["test"]["tgt"]])
    smi = ds["train"]["smi"] + ds["val"]["smi"] + ds["test"]["smi"]
    return {"emb": emb, "tgt": tgt, "smi": smi}


# ══════════════════════════════════════════════════════════════════════════════
# §2  ECFP UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def ecfp_matrix(smiles: list) -> torch.Tensor:
    fps = [AllChem.GetMorganFingerprintAsBitVect(
               Chem.MolFromSmiles(s) or Chem.MolFromSmiles("C"), 2, nBits=1024)
           for s in smiles]
    n, A = len(fps), torch.zeros(len(fps), len(fps))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for j, v in enumerate(sims, i + 1):
            A[i, j] = A[j, i] = v
    return A


# ══════════════════════════════════════════════════════════════════════════════
# §3  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

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
# §4  META-LEARNER MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class MolAttention(nn.Module):
    """
    Dense self-attention adjacency — the MHNfs insight.
    Every node pair is connected; weights are continuous softmax scores.
    No topk, no ECFP blend, no discrete topology decisions.
    At K=20 this is vastly more stable than sparse edge learning.
    """
    def __init__(self, d=768):
        super().__init__()
        self.Wq    = nn.Linear(d, d, bias=False)
        self.Wk    = nn.Linear(d, d, bias=False)
        self.scale = math.sqrt(d)
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)

    def forward(self, X):
        # X: [N, d]  →  A: [N, N] fully dense, row-softmax
        Q = self.Wq(X)
        K = self.Wk(X)
        return torch.softmax(Q @ K.T / self.scale, dim=-1)


class NIGHead(nn.Module):
    def __init__(self, d=768, h1=512, h2=256, drop=0.1):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(d, h1), nn.GELU(), nn.Dropout(drop),
                                   nn.Linear(h1, h2), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(h2, 4)
        self.sp   = nn.Softplus()

    def forward(self, x):
        r = self.head(self.net(x))
        return (r[:,0], self.sp(r[:,1])+1e-6, self.sp(r[:,2])+1.0+1e-6, self.sp(r[:,3])+1e-6)


class DirHead(nn.Module):
    def __init__(self, d=768, h1=512, h2=256, K=2, drop=0.1):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(d, h1), nn.GELU(), nn.Dropout(drop),
                                   nn.Linear(h1, h2), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(h2, K)
        self.sp   = nn.Softplus()

    def forward(self, x):
        return self.sp(self.head(self.net(x))) + 1e-6


# ── Baseline 1: MAML + MLP ───────────────────────────────────────────────────
class MetaMLP(nn.Module):
    """No graph structure; adapts MLP head on MolFormer embeddings."""
    def __init__(self, task_type="reg"):
        super().__init__()
        self.task_type = task_type
        if task_type == "reg":
            self.head = NIGHead()
        else:
            self.head = nn.Sequential(
                nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 1))

    def forward(self, emb):
        return self.head(emb)

    def compute_loss(self, emb, tgt, smi=None):
        emb, tgt = emb.to(DEVICE), tgt.to(DEVICE)
        if self.task_type == "reg":
            mu, v, a, b = self.forward(emb)
            return nig_loss(tgt, mu, v, a, b)
        else:
            return F.binary_cross_entropy_with_logits(
                self.forward(emb).squeeze(-1), tgt)

    def predict(self, emb, smi=None):
        emb = emb.to(DEVICE)
        with torch.no_grad():
            out = self.forward(emb)
        if self.task_type == "reg":
            return out[0].cpu()
        else:
            return torch.sigmoid(out.squeeze(-1)).cpu()


# ── Baseline 2: MAML + Static GNN ────────────────────────────────────────────
class MetaStaticGNN(nn.Module):
    """ECFP topology is FIXED; only GNN weights adapt in inner loop."""
    def __init__(self, task_type="reg", k=5, d=768):
        super().__init__()
        self.task_type = task_type
        self.k   = k
        self.gcn = nn.Linear(d, d)
        self.act = nn.GELU()
        if task_type == "reg":
            self.head = NIGHead(d)
        else:
            self.head = nn.Sequential(
                nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 1))

    def _adj(self, smi):
        A = ecfp_matrix(smi).to(DEVICE)
        k = min(self.k, A.size(0) - 1)
        _, ti = A.topk(k, -1)
        M = torch.zeros_like(A); M.scatter_(1, ti, 1.0)
        M = ((M + M.T) > 0).float()
        A = A * M; A.fill_diagonal_(0.0)
        A_id = A + torch.eye(A.size(0), device=A.device)
        return A_id / A_id.sum(-1, keepdim=True).clamp(1e-8)

    def _gnn(self, emb, smi):
        A_norm = self._adj(smi)
        return self.act(self.gcn(A_norm @ emb.to(DEVICE))) + emb.to(DEVICE)

    def compute_loss(self, emb, tgt, smi):
        H, tgt = self._gnn(emb, smi), tgt.to(DEVICE)
        if self.task_type == "reg":
            mu, v, a, b = self.head(H)
            return nig_loss(tgt, mu, v, a, b)
        else:
            return F.binary_cross_entropy_with_logits(
                self.head(H).squeeze(-1), tgt)

    def predict(self, emb, smi):
        with torch.no_grad():
            H = self._gnn(emb, smi)
            if self.task_type == "reg":
                return self.head(H)[0].cpu()
            else:
                return torch.sigmoid(self.head(H).squeeze(-1)).cpu()


# ── Ours: MAML + Evidential GSL (Dense Attention + Uncertainty Gate) ─────────
class MetaEvidentialGSL(nn.Module):
    """
    Dense attention adjacency (MHNfs insight) + epistemic uncertainty gate.

    Key differences from the sparse GSL baseline:
      - MolAttention: fully-connected soft graph, continuous weights, no topk.
        Avoids the discrete edge-selection instability at K=20.
      - Edge-wise gate: A_gated[i,j] = A[i,j] * G_i * G_j
        Severs edges where EITHER endpoint is uncertain (OOD).
        MHNfs retrieves blindly; we use uncertainty as the scalpel.
      - No ECFP matrix: topology is computed from embeddings alone.
        This makes the inner loop 5-10x faster (no RDKit fingerprint calls).

    During FOMAML inner-loop, the warm-start NIG/Dir head gives meaningful
    u_ep from step 0, even at K=20 — the gate is informed, not random.
    """
    def __init__(self, task_type="reg", d=768):
        super().__init__()
        self.task_type = task_type
        self.gm  = MolAttention(d)
        self.gcn = nn.Linear(d, d)
        self.act = nn.GELU()
        self.gam = nn.Parameter(torch.tensor(1.0))
        if task_type == "reg":
            self.ih = NIGHead(d)
            self.fh = NIGHead(d)
        else:
            self.ih = DirHead(d, K=2)
            self.fh = DirHead(d, K=2)

    def _forward_reg(self, X):
        m0, v0, a0, b0 = self.ih(X)
        u0 = b0 / (a0 - 1.0).clamp(1e-8)              # [N] aleatoric uncertainty
        G  = 1.0 - self.gam * torch.sigmoid(u0)         # [N] per-node gate ∈ (0,1)
        A  = self.gm(X)                                  # [N, N] dense attention
        # Edge gate: product severs if EITHER node is uncertain
        A_gated = A * G.unsqueeze(0) * G.unsqueeze(1)   # [N, N]
        D  = 1.0 / A_gated.sum(-1, keepdim=True).clamp(1e-8)
        M  = self.act(self.gcn((A_gated * D) @ X))
        mu, v, al, be = self.fh(X + M)
        return (m0, v0, a0, b0), (mu, v, al, be)

    def _forward_cls(self, X):
        a0 = self.ih(X)
        u0 = 2.0 / a0.sum(-1).clamp(1e-8)              # [N] vacancy = uncertainty
        G  = 1.0 - self.gam * torch.sigmoid(u0)
        A  = self.gm(X)
        A_gated = A * G.unsqueeze(0) * G.unsqueeze(1)
        D  = 1.0 / A_gated.sum(-1, keepdim=True).clamp(1e-8)
        M  = self.act(self.gcn((A_gated * D) @ X))
        return a0, self.fh(X + M)

    def compute_loss(self, emb, tgt, smi=None):    # smi unused: no ECFP needed
        emb, tgt = emb.to(DEVICE), tgt.to(DEVICE)
        if self.task_type == "reg":
            (m0,v0,a0,b0), (mu,v,al,be) = self._forward_reg(emb)
            return nig_loss(tgt,mu,v,al,be) + AUX_W*nig_loss(tgt,m0,v0,a0,b0)
        else:
            a0, af = self._forward_cls(emb)
            oh = F.one_hot(tgt.long(), 2).float()
            return dirichlet_loss(oh, af) + AUX_W * dirichlet_loss(oh, a0)

    def predict(self, emb, smi=None):              # smi unused: no ECFP needed
        emb = emb.to(DEVICE)
        with torch.no_grad():
            if self.task_type == "reg":
                _, (mu, *_) = self._forward_reg(emb)
                return mu.cpu()
            else:
                _, af = self._forward_cls(emb)
                return (af[:, 1] / af.sum(-1)).cpu()


# ══════════════════════════════════════════════════════════════════════════════
# §5  FOMAML UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def fomaml_adapt(model, sup_emb, sup_tgt, sup_smi, inner_steps, lr_inner):
    """
    Return adapted CLONE after `inner_steps` SGD steps on support.
    FOMAML: first-order approximation (no second-order grads through inner loop).
    """
    adapted   = copy.deepcopy(model).to(DEVICE)
    inner_opt = torch.optim.SGD(adapted.parameters(), lr=lr_inner)
    for _ in range(inner_steps):
        inner_opt.zero_grad()
        loss = adapted.compute_loss(sup_emb, sup_tgt, sup_smi)
        loss.backward()
        inner_opt.step()
    return adapted


def fomaml_outer_step(model, episodes, meta_opt):
    """
    One outer FOMAML step.
    episodes: list of (sup_emb, sup_tgt, sup_smi, qry_emb, qry_tgt, qry_smi)
    Copies query gradients from adapted clones back to original params.
    """
    meta_opt.zero_grad()
    total_loss = 0.0
    for sup_emb, sup_tgt, sup_smi, qry_emb, qry_tgt, qry_smi in episodes:
        adapted = fomaml_adapt(model, sup_emb, sup_tgt, sup_smi,
                               INNER_STEPS, LR_INNER)
        q_loss  = adapted.compute_loss(qry_emb, qry_tgt, qry_smi)
        q_loss.backward()
        total_loss += q_loss.item()
        # FOMAML: copy grads from adapted clone → original model
        for p_o, p_a in zip(model.parameters(), adapted.parameters()):
            if p_a.grad is not None:
                if p_o.grad is None:
                    p_o.grad = p_a.grad.clone() / len(episodes)
                else:
                    p_o.grad += p_a.grad.clone() / len(episodes)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    meta_opt.step()
    return total_loss / len(episodes)


# ══════════════════════════════════════════════════════════════════════════════
# §6  EPISODE SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def sample_cls_episode(emb, tgt, smi, K_total, max_query=MAX_QUERY, rng=None):
    """
    Binary classification episode.
    K_total support molecules (K//2 per class).
    Returns: (sup_emb, sup_tgt, sup_smi, qry_emb, qry_tgt, qry_smi)
    """
    if rng is None: rng = np.random.RandomState()
    tgt_np = tgt.numpy()
    pos    = np.where(tgt_np == 1)[0]
    neg    = np.where(tgt_np == 0)[0]
    Kh     = max(1, min(K_total // 2, len(pos) // 2, len(neg) // 2))
    sup_idx = np.concatenate([rng.choice(pos, Kh, replace=False),
                               rng.choice(neg, Kh, replace=False)])
    qry_pool = np.setdiff1d(np.arange(len(emb)), sup_idx)
    if len(qry_pool) > max_query:
        qry_pool = rng.choice(qry_pool, max_query, replace=False)
    sup_smi = [smi[i] for i in sup_idx]
    qry_smi = [smi[i] for i in qry_pool]
    return (emb[sup_idx], tgt[sup_idx].float(), sup_smi,
            emb[qry_pool], tgt[qry_pool].float(), qry_smi)


def sample_reg_episode(emb, tgt_s, smi, K, max_query=MAX_QUERY, rng=None):
    """
    Regression episode: K support molecules, up to max_query query molecules.
    """
    if rng is None: rng = np.random.RandomState()
    n       = len(emb)
    K       = min(K, n // 2)
    sup_idx = rng.choice(n, K, replace=False)
    qry_pool = np.setdiff1d(np.arange(n), sup_idx)
    if len(qry_pool) > max_query:
        qry_pool = rng.choice(qry_pool, max_query, replace=False)
    sup_smi = [smi[i] for i in sup_idx]
    qry_smi = [smi[i] for i in qry_pool]
    return (emb[sup_idx], tgt_s[sup_idx], sup_smi,
            emb[qry_pool], tgt_s[qry_pool], qry_smi)


# ══════════════════════════════════════════════════════════════════════════════
# §7  META-TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def scale_task(full_tgt):
    sc = StandardScaler()
    sc.fit(full_tgt.numpy().reshape(-1, 1))
    def t(x): return torch.tensor(
        sc.transform(x.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32)
    return sc, t


def meta_train(model, train_data: dict, task_type: str, seed: int):
    """
    FOMAML meta-training.
    train_data: {name: {'emb', 'tgt', 'smi'}} — training splits only.
    Returns meta-trained model (in-place modified, also returned).
    """
    set_seed(seed)
    # Pre-scale regression targets per task
    scalers = {}
    scaled  = {}
    task_names = list(train_data.keys())
    for name, d in train_data.items():
        if task_type == "reg":
            sc, tf = scale_task(d["tgt"])
            scalers[name] = sc
            scaled[name]  = tf(d["tgt"])
        else:
            scaled[name] = d["tgt"]   # binary, no scaling

    model   = model.to(DEVICE)
    meta_opt = torch.optim.Adam(model.parameters(), lr=LR_META)
    rng      = np.random.RandomState(seed)

    for ep in range(1, META_EPISODES + 1):
        # Sample META_BATCH_TASKS tasks (with replacement if fewer tasks)
        chosen = rng.choice(task_names, min(META_BATCH_TASKS, len(task_names)),
                            replace=len(task_names) < META_BATCH_TASKS)
        episodes = []
        for name in chosen:
            d   = train_data[name]
            tgt = scaled[name]
            K_ep = 10   # support size per meta-training episode
            if task_type == "cls":
                ep_data = sample_cls_episode(d["emb"], tgt, d["smi"], K_ep, rng=rng)
            else:
                ep_data = sample_reg_episode(d["emb"], tgt, d["smi"], K_ep, rng=rng)
            episodes.append(ep_data)

        loss = fomaml_outer_step(model, episodes, meta_opt)
        if ep % max(1, META_EPISODES // 5) == 0:
            print(f"    Meta-ep {ep:4d}/{META_EPISODES}  query_loss={loss:.4f}")

    return model, scalers


# ══════════════════════════════════════════════════════════════════════════════
# §8  META-TESTING
# ══════════════════════════════════════════════════════════════════════════════

def meta_test_cls(model, test_full: dict, K, seed):
    """
    K-shot adaptation + AUROC on remaining query.
    Returns {task_name: auroc}.
    """
    rng     = np.random.RandomState(seed)
    results = {}
    for name, d in test_full.items():
        ep = sample_cls_episode(d["emb"], d["tgt"], d["smi"], K, max_query=9999, rng=rng)
        sup_emb, sup_tgt, sup_smi, qry_emb, qry_tgt, qry_smi = ep

        adapted = fomaml_adapt(model, sup_emb, sup_tgt, sup_smi, INNER_STEPS, LR_INNER)
        prob    = adapted.predict(qry_emb, qry_smi).numpy()
        tgt_np  = qry_tgt.numpy()

        if len(np.unique(tgt_np)) < 2:
            results[name] = float("nan")
        else:
            results[name] = float(roc_auc_score(tgt_np, prob))
    return results


def meta_test_reg(model, test_full: dict, K, seed):
    """
    K-shot adaptation + RMSE on remaining query (original scale).
    Returns {task_name: rmse}.
    """
    rng     = np.random.RandomState(seed)
    results = {}
    for name, d in test_full.items():
        sc, tf  = scale_task(d["tgt"])
        tgt_s   = tf(d["tgt"])
        ep      = sample_reg_episode(d["emb"], tgt_s, d["smi"], K, max_query=9999, rng=rng)
        sup_emb, sup_tgt, sup_smi, qry_emb, qry_tgt, qry_smi = ep

        adapted = fomaml_adapt(model, sup_emb, sup_tgt, sup_smi, INNER_STEPS, LR_INNER)
        mu_s    = adapted.predict(qry_emb, qry_smi).numpy()
        mu_o    = sc.inverse_transform(mu_s.reshape(-1, 1)).flatten()
        tgt_o   = sc.inverse_transform(qry_tgt.numpy().reshape(-1, 1)).flatten()
        results[name] = float(np.sqrt(np.mean((mu_o - tgt_o) ** 2)))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# §9  HELPERS + RESULT I/O
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(cond, task_type):
    if cond == "maml_mlp":
        return MetaMLP(task_type)
    elif cond == "maml_static_gnn":
        return MetaStaticGNN(task_type)
    else:
        return MetaEvidentialGSL(task_type)


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved → {path}")


def write_summary(cls_rows, reg_rows, path):
    lines = ["=" * 65,
             "PoC 2 — Evidential Graph Meta-Learning: Summary",
             "=" * 65]
    for label, rows, metric, up in [
        ("Classification (AUROC ↑)", cls_rows, "AUROC", True),
        ("Regression     (RMSE  ↓)", reg_rows, "RMSE",  False),
    ]:
        lines += ["", label]
        for cond in CONDITIONS:
            for K in K_SHOTS:
                key  = f"{cond}_K{K}"
                vals = [r[key] for r in rows if key in r and not np.isnan(r[key])]
                if vals:
                    tag = f"{cond} K={K}"
                    lines.append(f"  {tag:35s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# §10  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 65)
    print("PoC 2: Evidential Graph Meta-Learning (FOMAML)")
    print(f"Device: {DEVICE}  |  Seeds: {SEEDS}  |  K-shots: {K_SHOTS}  |  Quick: {QUICK}")
    print("=" * 65)

    # ── §0  Cache / load all embeddings ─────────────────────────────────────
    print("\n[Step 0] Loading / caching embeddings …")
    all_names = (CLS_TRAIN_TASKS + CLS_TEST_TASKS +
                 REG_TRAIN_TASKS + REG_TEST_TASKS)
    datasets  = {}
    for name in dict.fromkeys(all_names):   # deduplicate while preserving order
        datasets[name] = get_dataset_embeddings(name)
    print("  All datasets ready.\n")

    # Prepare training splits (train-only for meta-training)
    cls_train_data = {n: datasets[n]["train"] for n in CLS_TRAIN_TASKS}
    reg_train_data = {n: datasets[n]["train"] for n in REG_TRAIN_TASKS}

    # Prepare full test datasets (train+val+test concatenated)
    cls_test_full  = {n: get_full_dataset(datasets[n]) for n in CLS_TEST_TASKS}
    reg_test_full  = {n: get_full_dataset(datasets[n]) for n in REG_TEST_TASKS}

    cls_rows, reg_rows = [], []

    for seed in SEEDS:
        print(f"\n{'═'*65}")
        print(f"SEED {seed}")
        print(f"{'═'*65}")
        set_seed(seed)

        cls_row = {"seed": seed}
        reg_row = {"seed": seed}

        for cond in CONDITIONS:
            print(f"\n[{cond}]")

            # ── Classification ───────────────────────────────────────────────
            print(f"  [CLS] Meta-training on {CLS_TRAIN_TASKS} …")
            cls_model, _ = meta_train(
                make_model(cond, "cls"), cls_train_data, "cls", seed)

            for K in K_SHOTS:
                t0 = time.perf_counter()
                res = meta_test_cls(cls_model, cls_test_full, K, seed)
                avg = np.nanmean(list(res.values()))
                key = f"{cond}_K{K}"
                cls_row[key] = avg
                detail = "  ".join(f"{n}={v:.3f}" for n, v in res.items())
                print(f"    K={K:2d}: AUROC={avg:.4f}  [{detail}]  "
                      f"({time.perf_counter()-t0:.0f}s)")

            # ── Regression ───────────────────────────────────────────────────
            print(f"  [REG] Meta-training on {REG_TRAIN_TASKS} …")
            reg_model, _ = meta_train(
                make_model(cond, "reg"), reg_train_data, "reg", seed)

            for K in K_SHOTS:
                t0 = time.perf_counter()
                res = meta_test_reg(reg_model, reg_test_full, K, seed)
                avg = np.nanmean(list(res.values()))
                key = f"{cond}_K{K}"
                reg_row[key] = avg
                detail = "  ".join(f"{n}={v:.3f}" for n, v in res.items())
                print(f"    K={K:2d}: RMSE ={avg:.4f}  [{detail}]  "
                      f"({time.perf_counter()-t0:.0f}s)")

        cls_rows.append(cls_row)
        reg_rows.append(reg_row)

    # ── §11  Save results ────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    save_csv(cls_rows, os.path.join(RESULTS_DIR, "classification_meta_results.csv"))
    save_csv(reg_rows, os.path.join(RESULTS_DIR, "regression_meta_results.csv"))
    write_summary(cls_rows, reg_rows, os.path.join(RESULTS_DIR, "summary.txt"))
    print(f"\nAll results saved → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
