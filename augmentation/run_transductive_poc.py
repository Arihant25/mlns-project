#!/usr/bin/env python3
"""
run_transductive_poc.py  —  augmentation/
==========================================
PoC 1 v3: Transductive Semi-Supervised GSL (no pseudo-labels)

Key idea vs v2:
  Instead of pseudo-labeling source molecules, we treat oracle-selected
  source molecules as UNLABELED ANCHOR NODES in the graph.

  Transductive training:
    - Full graph = labeled target nodes + unlabeled anchor nodes
    - ECFP matrix built ONCE over all nodes (fixed topology)
    - EvidentialGSL / DirGSL forward pass over the FULL graph
    - Loss computed ONLY on the labeled target node slice [:n_tgt]
    - Anchors contribute ONLY via message passing — no pseudo-label required

4 ablation conditions × 5 seeds.
Results → augmentation/results_transductive/
Set QUICK=1 for smoke-test.
"""

import copy, csv, math, os, time
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
ORACLE_CACHE = os.path.join(PROJECT_ROOT, "data", "oracle_cache")
RESULTS_DIR  = os.path.join(SCRIPT_DIR,   "results_transductive")   # ← separate
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ───────────────────────────────────────────────────────────────────
QUICK = os.environ.get("QUICK", "").lower() in ("1", "true", "yes")

ORACLE_SEEDS      = [40] if QUICK else [40, 41, 42]
EXPERIMENT_SEEDS  = [0]  if QUICK else [0, 1, 2, 3, 4]
MAX_EP_ORACLE     = 3    if QUICK else 60
MAX_EP_FINAL      = 3    if QUICK else 80
PATIENCE          = 10
LR_PATIENCE       = 5
LR                = 1e-3
BATCH_ORACLE      = 256
NIG_COEFF         = 0.1
DIR_COEFF         = 0.1
AUX_W             = 0.5
SIM_THRESH        = 0.75
EPIST_PCTILE      = 20
MAX_ANCHOR_RATIO  = 0.5   # anchors ≤ 50 % of labeled target set
MOLFORMER         = "ibm/MoLFormer-XL-both-10pct"
CONDITIONS        = ["target_only", "random_aug", "similarity_only", "evidential_oracle"]

_mf_cache: dict = {}

# ══════════════════════════════════════════════════════════════════════════════
# §1  EMBEDDING INFRASTRUCTURE  — unchanged from v2
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


# ══════════════════════════════════════════════════════════════════════════════
# §2  ECFP + DATASET UTILITIES
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


class GSLDS(Dataset):
    def __init__(self, e, t, s): self.e, self.t, self.s = e, t, s
    def __len__(self):           return len(self.e)
    def __getitem__(self, i):    return self.e[i], self.t[i], self.s[i]


def gsl_collate(batch):
    embs, tgts, smis = zip(*batch)
    return torch.stack(embs), torch.stack(tgts), ecfp_matrix(list(smis))


def gsl_loader(emb, tgt, smi, bs, shuffle=False):
    return DataLoader(GSLDS(emb, tgt, smi), batch_size=bs,
                      shuffle=shuffle, collate_fn=gsl_collate)


# ══════════════════════════════════════════════════════════════════════════════
# §3  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def nig_nll(y, mu, v, a, b):
    r = (y - mu) ** 2
    return (0.5 * torch.log(math.pi / v)
            - a * torch.log(b)
            + (a + 0.5) * torch.log(b + 0.5 * v * r)
            + torch.lgamma(a) - torch.lgamma(a + 0.5))

def nig_reg_term(y, mu, v, a, _b):
    return torch.abs(y - mu) * (2.0 * v + a)

def nig_loss(y, mu, v, a, b, coeff=NIG_COEFF):
    nll = nig_nll(y, mu, v, a, b)
    reg = nig_reg_term(y, mu, v, a, b)
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
# §4  MODEL DEFINITIONS  — unchanged from v2
# ══════════════════════════════════════════════════════════════════════════════

class LearnedGraphMaker(nn.Module):
    def __init__(self, d=768, k=5):
        super().__init__()
        self.W  = nn.Parameter(torch.empty(d, d))
        self.ra = nn.Parameter(torch.tensor(0.0))
        self.k  = k
        nn.init.xavier_uniform_(self.W)

    def forward(self, X, A_ecfp):
        S  = torch.relu(X @ self.W @ X.T)
        al = torch.sigmoid(self.ra)
        A  = al * A_ecfp + (1 - al) * S
        k  = min(self.k, A.size(0) - 1)
        _, top = A.topk(k, dim=-1)
        M  = torch.zeros_like(A); M.scatter_(1, top, 1.0)
        M  = ((M + M.T) > 0).float()
        A  = A * M; A.fill_diagonal_(0.0)
        return A


class NIGHead(nn.Module):
    def __init__(self, d=768, h1=512, h2=256, drop=0.1):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(d, h1), nn.GELU(), nn.Dropout(drop),
                                   nn.Linear(h1, h2), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(h2, 4)
        self.sp   = nn.Softplus()

    def forward(self, x):
        r = self.head(self.net(x))
        return (r[:, 0],
                self.sp(r[:, 1]) + 1e-6,
                self.sp(r[:, 2]) + 1.0 + 1e-6,
                self.sp(r[:, 3]) + 1e-6)


class DirHead(nn.Module):
    def __init__(self, d=768, h1=512, h2=256, K=2, drop=0.1):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(d, h1), nn.GELU(), nn.Dropout(drop),
                                   nn.Linear(h1, h2), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(h2, K)
        self.sp   = nn.Softplus()

    def forward(self, x):
        return self.sp(self.head(self.net(x))) + 1e-6


class EvidentialGSL(nn.Module):
    def __init__(self, d=768, k=5, drop=0.1):
        super().__init__()
        self.gm  = LearnedGraphMaker(d, k)
        self.ih  = NIGHead(d, drop=drop)
        self.gcn = nn.Linear(d, d)
        self.act = nn.GELU()
        self.fh  = NIGHead(d, drop=drop)
        self.gam = nn.Parameter(torch.tensor(1.0))

    def forward(self, X, A):
        m0, v0, a0, b0 = self.ih(X)
        u0 = b0 / (a0 - 1.0).clamp(1e-8)
        G  = (1.0 - self.gam * torch.sigmoid(u0)).unsqueeze(-1)
        Ag = self.gm(X, A)
        D  = 1.0 / Ag.sum(-1, keepdim=True).clamp(1e-8)
        M  = self.act(self.gcn((Ag * D) @ (X * G)))
        mu, v, al, be = self.fh(X + M)
        return (m0, v0, a0, b0), (mu, v, al, be)


class DirGSL(nn.Module):
    def __init__(self, d=768, k=5, K=2, drop=0.1):
        super().__init__()
        self.gm  = LearnedGraphMaker(d, k)
        self.ih  = DirHead(d, K=K, drop=drop)
        self.gcn = nn.Linear(d, d)
        self.act = nn.GELU()
        self.fh  = DirHead(d, K=K, drop=drop)
        self.gam = nn.Parameter(torch.tensor(1.0))
        self.K   = K

    def forward(self, X, A):
        a0 = self.ih(X)
        u0 = self.K / a0.sum(-1).clamp(1e-8)
        G  = (1.0 - self.gam * torch.sigmoid(u0)).unsqueeze(-1)
        Ag = self.gm(X, A)
        D  = 1.0 / Ag.sum(-1, keepdim=True).clamp(1e-8)
        M  = self.act(self.gcn((Ag * D) @ (X * G)))
        return a0, self.fh(X + M)


# ══════════════════════════════════════════════════════════════════════════════
# §5  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def scale_tgt(tr, val, te):
    sc = StandardScaler()
    sc.fit(tr.numpy().reshape(-1, 1))
    def t(x): return torch.tensor(sc.transform(x.numpy().reshape(-1, 1)).flatten(),
                                   dtype=torch.float32)
    return sc, t(tr), t(val), t(te)


# ══════════════════════════════════════════════════════════════════════════════
# §5b  ORACLE SELECTION CACHE  — unchanged, reuses same cache as v2
# ══════════════════════════════════════════════════════════════════════════════

def _oracle_cache_paths(src_name: str, tgt_name: str, seeds: list):
    tag = f"{src_name}__{tgt_name}__seeds_{'_'.join(str(s) for s in seeds)}"
    return (os.path.join(ORACLE_CACHE, f"{tag}_u_ep.pt"),
            os.path.join(ORACLE_CACHE, f"{tag}_sel_mask.pt"))

def load_oracle_cache(src_name: str, tgt_name: str, seeds: list):
    u_path, m_path = _oracle_cache_paths(src_name, tgt_name, seeds)
    if os.path.exists(u_path) and os.path.exists(m_path):
        u_ep     = torch.load(u_path, weights_only=True)
        sel_mask = torch.load(m_path, weights_only=True)
        print(f"  [Oracle cache] Loaded {src_name}→{tgt_name} (seeds={seeds})")
        return u_ep, sel_mask
    return None

def save_oracle_cache(src_name: str, tgt_name: str, seeds: list,
                      u_ep: torch.Tensor, sel_mask: torch.Tensor):
    os.makedirs(ORACLE_CACHE, exist_ok=True)
    u_path, m_path = _oracle_cache_paths(src_name, tgt_name, seeds)
    torch.save(u_ep,     u_path)
    torch.save(sel_mask, m_path)
    print(f"  [Oracle cache] Saved {src_name}→{tgt_name} → {ORACLE_CACHE}")


# ══════════════════════════════════════════════════════════════════════════════
# §6  ORACLE TRAINING  — identical to v2
# ══════════════════════════════════════════════════════════════════════════════

def _early_stop_loop(mdl, tr_ldr, val_ldr, opt, sch, max_ep, loss_fn, val_loss_fn=None):
    if val_loss_fn is None: val_loss_fn = loss_fn
    best_vl, best_st, no_imp = float("inf"), None, 0
    for _ in range(max_ep):
        mdl.train()
        for batch in tr_ldr:
            loss = loss_fn(mdl, batch)
            opt.zero_grad(); loss.backward(); opt.step()
        mdl.eval(); vl = n = 0
        with torch.no_grad():
            for batch in val_ldr:
                l = val_loss_fn(mdl, batch)
                b = batch[0].size(0)
                vl += l.item() * b; n += b
        vl /= n; sch.step(vl)
        if vl < best_vl: best_vl = vl; best_st = copy.deepcopy(mdl.state_dict()); no_imp = 0
        else:            no_imp += 1
        if no_imp >= PATIENCE: break
    mdl.load_state_dict(best_st)
    return mdl


def train_reg_oracle(src: dict, seed: int) -> EvidentialGSL:
    set_seed(seed)
    sc, tr_s, val_s, _ = scale_tgt(src["train"]["tgt"], src["val"]["tgt"], src["train"]["tgt"])
    tr_ldr  = gsl_loader(src["train"]["emb"], tr_s,  src["train"]["smi"], BATCH_ORACLE, True)
    val_ldr = gsl_loader(src["val"]["emb"],   val_s, src["val"]["smi"],   BATCH_ORACLE)
    mdl = EvidentialGSL().to(DEVICE)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.5, LR_PATIENCE)

    def lfn(m, batch):
        X, y, A = [x.to(DEVICE) for x in batch]
        (m0,v0,a0,b0),(mu,v,al,be) = m(X, A)
        return nig_loss(y,mu,v,al,be) + AUX_W * nig_loss(y,m0,v0,a0,b0)

    _early_stop_loop(mdl, tr_ldr, val_ldr, opt, sch, MAX_EP_ORACLE, lfn)
    for p in mdl.parameters(): p.requires_grad_(False)
    return mdl


def train_cls_oracle(src: dict, seed: int) -> DirGSL:
    set_seed(seed)
    tr_ldr  = gsl_loader(src["train"]["emb"], src["train"]["tgt"].long(),
                          src["train"]["smi"], BATCH_ORACLE, True)
    val_ldr = gsl_loader(src["val"]["emb"],   src["val"]["tgt"].long(),
                          src["val"]["smi"],   BATCH_ORACLE)
    mdl = DirGSL(K=2).to(DEVICE)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.5, LR_PATIENCE)

    def lfn(m, batch):
        X, y, A = [x.to(DEVICE) for x in batch]
        a0, af  = m(X, A)
        oh      = F.one_hot(y, 2).float()
        return dirichlet_loss(oh, af) + AUX_W * dirichlet_loss(oh, a0)

    _early_stop_loop(mdl, tr_ldr, val_ldr, opt, sch, MAX_EP_ORACLE, lfn)
    for p in mdl.parameters(): p.requires_grad_(False)
    return mdl


@torch.no_grad()
def reg_uncertainty(oracle, emb, smi) -> torch.Tensor:
    oracle.eval()
    u = []
    for X, _, A in gsl_loader(emb, torch.zeros(len(emb)), smi, BATCH_ORACLE):
        X, A = X.to(DEVICE), A.to(DEVICE)
        _, (_, v, _, _) = oracle(X, A)
        u.append((1.0 / v.clamp(1e-8)).cpu())
    return torch.cat(u)


@torch.no_grad()
def cls_uncertainty(oracle, emb, smi) -> torch.Tensor:
    oracle.eval()
    u = []
    for X, _, A in gsl_loader(emb, torch.zeros(len(emb)), smi, BATCH_ORACLE):
        X, A = X.to(DEVICE), A.to(DEVICE)
        _, af = oracle(X, A)
        u.append((af.size(-1) / af.sum(-1).clamp(1e-8)).cpu())
    return torch.cat(u)


def ensemble_reg_uncertainty(oracles: list, emb, smi) -> torch.Tensor:
    return torch.stack([reg_uncertainty(o, emb, smi) for o in oracles]).mean(0)

def ensemble_cls_uncertainty(oracles: list, emb, smi) -> torch.Tensor:
    return torch.stack([cls_uncertainty(o, emb, smi) for o in oracles]).mean(0)


# ══════════════════════════════════════════════════════════════════════════════
# §7  ORACLE SELECTION  — identical to v2
# ══════════════════════════════════════════════════════════════════════════════

def select_oracle(src_emb, tgt_emb, u_ep, max_n) -> torch.Tensor:
    centroid = F.normalize(tgt_emb.mean(0, keepdim=True), dim=-1)
    sims     = (F.normalize(src_emb, dim=-1) @ centroid.T).squeeze(-1)
    s_mask   = sims > SIM_THRESH
    thr      = float(np.percentile(u_ep.numpy(), EPIST_PCTILE))
    e_mask   = u_ep < thr
    sel      = s_mask & e_mask
    if sel.sum() < 5: sel = s_mask
    if sel.sum() < 5:
        k = max(5, int(0.05 * len(src_emb)))
        _, ti = sims.topk(k)
        sel = torch.zeros(len(src_emb), dtype=torch.bool); sel[ti] = True
    if sel.sum() > max_n:
        u_sel = u_ep[sel]; si = u_sel.argsort()[:max_n]
        fi    = sel.nonzero(as_tuple=True)[0]
        sel   = torch.zeros(len(src_emb), dtype=torch.bool); sel[fi[si]] = True
    return sel


# ══════════════════════════════════════════════════════════════════════════════
# §8  ANCHOR BUILDERS  — replaces pseudo-labeling build_*_aug
#     Returns (anc_emb, anc_smi) — NO labels, used only for message passing
# ══════════════════════════════════════════════════════════════════════════════

def build_anchors_target_only():
    """No anchors: pure supervised on target only."""
    return torch.zeros(0, 768), []


def build_anchors_random(src_emb, src_smi, max_n: int, seed: int):
    """Random source molecules as unlabeled anchors."""
    g    = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(src_emb), generator=g)[:max_n].tolist()
    return src_emb[perm], [src_smi[i] for i in perm]


def build_anchors_similarity(src_emb, src_smi, tgt_emb, max_n: int):
    """Similarity-filtered source molecules as unlabeled anchors."""
    centroid = F.normalize(tgt_emb.mean(0, keepdim=True), dim=-1)
    sims     = (F.normalize(src_emb, dim=-1) @ centroid.T).squeeze(-1)
    mask     = sims > SIM_THRESH
    if mask.sum() == 0:
        _, ti = sims.topk(max_n)
        mask = torch.zeros(len(sims), dtype=torch.bool); mask[ti] = True
    idx = mask.nonzero(as_tuple=True)[0][:max_n].tolist()
    return src_emb[idx], [src_smi[i] for i in idx]


def build_anchors_oracle(src_emb, src_smi, sel_mask: torch.Tensor):
    """Oracle-selected molecules as unlabeled anchors."""
    idx = sel_mask.nonzero(as_tuple=True)[0].tolist()
    return src_emb[idx], [src_smi[i] for i in idx]


# ══════════════════════════════════════════════════════════════════════════════
# §9  TRANSDUCTIVE TRAINING
#     Full graph: labeled target nodes + unlabeled anchor nodes.
#     ECFP computed ONCE (fixed topology). Loss masked to [:n_tgt].
# ══════════════════════════════════════════════════════════════════════════════

def train_reg_transductive(tgt_emb, tgt_tgt_s, tgt_smi,
                            anc_emb, anc_smi,
                            val_emb, val_tgt_s, val_smi, seed) -> EvidentialGSL:
    """
    Transductive EvidentialGSL for regression.
    anc_emb / anc_smi can be empty (target_only condition).
    """
    set_seed(seed)
    n_tgt = len(tgt_emb)

    if len(anc_emb) > 0:
        all_emb = torch.cat([tgt_emb, anc_emb])   # labeled first, then anchors
        all_smi = tgt_smi + anc_smi
    else:
        all_emb = tgt_emb
        all_smi = tgt_smi

    print(f"      Graph: {n_tgt} labeled + {len(all_emb) - n_tgt} anchors"
          f" = {len(all_emb)} nodes")

    mdl  = EvidentialGSL().to(DEVICE)
    opt  = torch.optim.AdamW(mdl.parameters(), lr=LR)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.5, LR_PATIENCE)

    # Pre-compute ECFP once (fixed topology for the run)
    A_full = ecfp_matrix(all_smi).to(DEVICE)
    A_val  = ecfp_matrix(val_smi).to(DEVICE)
    X_all  = all_emb.to(DEVICE)
    y_lbl  = tgt_tgt_s.to(DEVICE)
    X_val  = val_emb.to(DEVICE)
    y_val  = val_tgt_s.to(DEVICE)

    best_vl, best_st, no_imp = float("inf"), None, 0

    for _ in range(MAX_EP_FINAL):
        mdl.train()
        (m0, v0, a0, b0), (mu, v, al, be) = mdl(X_all, A_full)
        # Loss ONLY on labeled slice [:n_tgt]
        loss = (nig_loss(y_lbl, mu[:n_tgt], v[:n_tgt], al[:n_tgt], be[:n_tgt])
                + AUX_W * nig_loss(y_lbl, m0[:n_tgt], v0[:n_tgt], a0[:n_tgt], b0[:n_tgt]))
        opt.zero_grad(); loss.backward(); opt.step()

        # Validate on val set (no anchors)
        mdl.eval()
        with torch.no_grad():
            _, (mu_v, v_v, al_v, be_v) = mdl(X_val, A_val)
            vl = nig_loss(y_val, mu_v, v_v, al_v, be_v).item()
        sch.step(vl)

        if vl < best_vl: best_vl = vl; best_st = copy.deepcopy(mdl.state_dict()); no_imp = 0
        else:            no_imp += 1
        if no_imp >= PATIENCE: break

    mdl.load_state_dict(best_st)
    return mdl


def train_cls_transductive(tgt_emb, tgt_tgt, tgt_smi,
                            anc_emb, anc_smi,
                            val_emb, val_tgt, val_smi, seed) -> DirGSL:
    """
    Transductive DirGSL for classification.
    """
    set_seed(seed)
    n_tgt = len(tgt_emb)

    if len(anc_emb) > 0:
        all_emb = torch.cat([tgt_emb, anc_emb])
        all_smi = tgt_smi + anc_smi
    else:
        all_emb = tgt_emb
        all_smi = tgt_smi

    print(f"      Graph: {n_tgt} labeled + {len(all_emb) - n_tgt} anchors"
          f" = {len(all_emb)} nodes")

    mdl  = DirGSL(K=2).to(DEVICE)
    opt  = torch.optim.AdamW(mdl.parameters(), lr=LR)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.5, LR_PATIENCE)

    A_full  = ecfp_matrix(all_smi).to(DEVICE)
    A_val   = ecfp_matrix(val_smi).to(DEVICE)
    X_all   = all_emb.to(DEVICE)
    y_lbl   = tgt_tgt.long().to(DEVICE)
    oh_lbl  = F.one_hot(y_lbl, 2).float()
    X_val   = val_emb.to(DEVICE)
    oh_val  = F.one_hot(val_tgt.long().to(DEVICE), 2).float()

    best_vl, best_st, no_imp = float("inf"), None, 0

    for _ in range(MAX_EP_FINAL):
        mdl.train()
        a0, af = mdl(X_all, A_full)
        # Loss ONLY on labeled slice [:n_tgt]
        loss = (dirichlet_loss(oh_lbl, af[:n_tgt])
                + AUX_W * dirichlet_loss(oh_lbl, a0[:n_tgt]))
        opt.zero_grad(); loss.backward(); opt.step()

        mdl.eval()
        with torch.no_grad():
            a0_v, af_v = mdl(X_val, A_val)
            vl = (dirichlet_loss(oh_val, af_v)
                  + AUX_W * dirichlet_loss(oh_val, a0_v)).item()
        sch.step(vl)

        if vl < best_vl: best_vl = vl; best_st = copy.deepcopy(mdl.state_dict()); no_imp = 0
        else:            no_imp += 1
        if no_imp >= PATIENCE: break

    mdl.load_state_dict(best_st)
    return mdl


# ══════════════════════════════════════════════════════════════════════════════
# §10  EVALUATION  — identical to v2
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_reg(mdl, test_emb, test_tgt_s, test_smi, scaler) -> float:
    mdl.eval()
    mus = []
    for X, _, A in gsl_loader(test_emb, test_tgt_s, test_smi, BATCH_ORACLE):
        X, A = X.to(DEVICE), A.to(DEVICE)
        _, (mu, *_) = mdl(X, A)
        mus.append(mu.cpu())
    mu_s  = torch.cat(mus).numpy()
    tgt_s = test_tgt_s.numpy()
    mu_o  = scaler.inverse_transform(mu_s.reshape(-1, 1)).flatten()
    tgt_o = scaler.inverse_transform(tgt_s.reshape(-1, 1)).flatten()
    return float(np.sqrt(np.mean((mu_o - tgt_o) ** 2)))


@torch.no_grad()
def eval_cls(mdl, test_emb, test_tgt, test_smi) -> float:
    mdl.eval()
    probs = []
    for X, _, A in gsl_loader(test_emb, torch.zeros(len(test_emb)), test_smi, BATCH_ORACLE):
        X, A = X.to(DEVICE), A.to(DEVICE)
        _, af = mdl(X, A)
        probs.append((af[:, 1] / af.sum(-1)).cpu())
    return float(roc_auc_score(test_tgt.numpy(), torch.cat(probs).numpy()))


# ══════════════════════════════════════════════════════════════════════════════
# §11  RESULT I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved → {path}")


def write_summary(res_A, res_B, path):
    lines = ["=" * 62,
             "PoC 1 v3 — Transductive Semi-Supervised GSL (no pseudo-labels)",
             "=" * 62]
    for label, res, metric in [
        ("Exp A  Classification (AUROC ↑)", res_A, "AUROC"),
        ("Exp B  Regression     (RMSE  ↓)", res_B, "RMSE"),
    ]:
        lines += ["", label]
        for c in CONDITIONS:
            vals = [r[c] for r in res]
            lines.append(f"  {c:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  [{metric}]")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# §12  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 62)
    print("PoC 1 v3: Transductive Semi-Supervised GSL (no pseudo-labels)")
    print(f"Device: {DEVICE}  |  Seeds: {EXPERIMENT_SEEDS}"
          f"  |  Oracle seeds: {ORACLE_SEEDS}  |  Quick: {QUICK}")
    print("=" * 62)

    # ── §0  Embeddings ────────────────────────────────────────────────────────
    print("\n[Step 0] Loading / caching embeddings …")
    lipo    = get_dataset_embeddings("Lipophilicity_AstraZeneca")
    caco    = get_dataset_embeddings("Caco2_Wang")
    cyp_src = get_dataset_embeddings("CYP2C9_Veith")
    cyp_tgt = get_dataset_embeddings("CYP2C9_Substrate_CarbonMangels")
    print("  All datasets ready.\n")

    max_anc_reg = len(caco["train"]["emb"]) // 2
    max_anc_cls = len(cyp_tgt["train"]["emb"]) // 2

    # ── §1  Oracle ensemble (reuses shared cache with v2) ────────────────────
    cached = load_oracle_cache("Lipophilicity_AstraZeneca", "Caco2_Wang", ORACLE_SEEDS)
    if cached is not None:
        u_ep_reg, sel_reg = cached
        print(f"  Skipping oracle — using cached selection "
              f"({sel_reg.sum()}/{len(sel_reg)} mols).\n")
    else:
        print(f"[Step 1] Training regression oracle ensemble (seeds={ORACLE_SEEDS}) …")
        oracles_reg = []
        for s in ORACLE_SEEDS:
            print(f"  oracle seed {s} …")
            oracles_reg.append(train_reg_oracle(lipo, s))
        u_ep_reg = ensemble_reg_uncertainty(oracles_reg, lipo["train"]["emb"],
                                             lipo["train"]["smi"])
        sel_reg  = select_oracle(lipo["train"]["emb"], caco["train"]["emb"],
                                  u_ep_reg, max_anc_reg)
        print(f"  Ensemble selected {sel_reg.sum()}/{len(sel_reg)} source molecules.")
        save_oracle_cache("Lipophilicity_AstraZeneca", "Caco2_Wang", ORACLE_SEEDS,
                          u_ep_reg, sel_reg)
        print()

    cached = load_oracle_cache("CYP2C9_Veith", "CYP2C9_Substrate_CarbonMangels", ORACLE_SEEDS)
    if cached is not None:
        u_ep_cls, sel_cls = cached
        print(f"  Skipping oracle — using cached selection "
              f"({sel_cls.sum()}/{len(sel_cls)} mols).\n")
    else:
        print(f"[Step 2] Training classification oracle ensemble (seeds={ORACLE_SEEDS}) …")
        oracles_cls = []
        for s in ORACLE_SEEDS:
            print(f"  oracle seed {s} …")
            oracles_cls.append(train_cls_oracle(cyp_src, s))
        u_ep_cls = ensemble_cls_uncertainty(oracles_cls, cyp_src["train"]["emb"],
                                             cyp_src["train"]["smi"])
        sel_cls  = select_oracle(cyp_src["train"]["emb"], cyp_tgt["train"]["emb"],
                                  u_ep_cls, max_anc_cls)
        print(f"  Ensemble selected {sel_cls.sum()}/{len(sel_cls)} source molecules.")
        save_oracle_cache("CYP2C9_Veith", "CYP2C9_Substrate_CarbonMangels", ORACLE_SEEDS,
                          u_ep_cls, sel_cls)
        print()

    # ── §2  Experiment loops ──────────────────────────────────────────────────
    results_A, results_B = [], []

    for seed in EXPERIMENT_SEEDS:
        sc_caco, caco_tr_s, caco_val_s, caco_te_s = scale_tgt(
            caco["train"]["tgt"], caco["val"]["tgt"], caco["test"]["tgt"])

        print(f"\n{'─'*60}")
        print(f"[Exp B / Seed {seed}] Lipophilicity → Caco2  (Regression, RMSE↓)")
        row_B = {"seed": seed}

        for cond in CONDITIONS:
            t0 = time.perf_counter()
            if cond == "target_only":
                anc_emb, anc_smi = build_anchors_target_only()
            elif cond == "random_aug":
                anc_emb, anc_smi = build_anchors_random(
                    lipo["train"]["emb"], lipo["train"]["smi"], max_anc_reg, seed)
            elif cond == "similarity_only":
                anc_emb, anc_smi = build_anchors_similarity(
                    lipo["train"]["emb"], lipo["train"]["smi"],
                    caco["train"]["emb"], max_anc_reg)
            else:   # evidential_oracle
                anc_emb, anc_smi = build_anchors_oracle(
                    lipo["train"]["emb"], lipo["train"]["smi"], sel_reg)

            mdl  = train_reg_transductive(
                caco["train"]["emb"], caco_tr_s, caco["train"]["smi"],
                anc_emb, anc_smi,
                caco["val"]["emb"],   caco_val_s, caco["val"]["smi"], seed)
            rmse = eval_reg(mdl, caco["test"]["emb"], caco_te_s,
                            caco["test"]["smi"], sc_caco)
            row_B[cond] = rmse
            print(f"  {cond:25s}: RMSE={rmse:.4f}  ({time.perf_counter()-t0:.0f}s)")

        results_B.append(row_B)

        print(f"\n[Exp A / Seed {seed}] CYP2C9_Veith → CYP2C9_Substrate  (Classification, AUROC↑)")
        row_A = {"seed": seed}

        for cond in CONDITIONS:
            t0 = time.perf_counter()
            if cond == "target_only":
                anc_emb, anc_smi = build_anchors_target_only()
            elif cond == "random_aug":
                anc_emb, anc_smi = build_anchors_random(
                    cyp_src["train"]["emb"], cyp_src["train"]["smi"], max_anc_cls, seed)
            elif cond == "similarity_only":
                anc_emb, anc_smi = build_anchors_similarity(
                    cyp_src["train"]["emb"], cyp_src["train"]["smi"],
                    cyp_tgt["train"]["emb"], max_anc_cls)
            else:   # evidential_oracle
                anc_emb, anc_smi = build_anchors_oracle(
                    cyp_src["train"]["emb"], cyp_src["train"]["smi"], sel_cls)

            mdl   = train_cls_transductive(
                cyp_tgt["train"]["emb"], cyp_tgt["train"]["tgt"], cyp_tgt["train"]["smi"],
                anc_emb, anc_smi,
                cyp_tgt["val"]["emb"], cyp_tgt["val"]["tgt"], cyp_tgt["val"]["smi"], seed)
            auroc = eval_cls(mdl, cyp_tgt["test"]["emb"],
                             cyp_tgt["test"]["tgt"], cyp_tgt["test"]["smi"])
            row_A[cond] = auroc
            print(f"  {cond:25s}: AUROC={auroc:.4f}  ({time.perf_counter()-t0:.0f}s)")

        results_A.append(row_A)

    # ── §3  Save ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    save_csv(results_A, os.path.join(RESULTS_DIR, "exp_A_classification_results.csv"))
    save_csv(results_B, os.path.join(RESULTS_DIR, "exp_B_regression_results.csv"))
    write_summary(results_A, results_B, os.path.join(RESULTS_DIR, "summary.txt"))
    print(f"\nAll results saved → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
