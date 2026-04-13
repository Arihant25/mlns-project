#!/usr/bin/env python3
"""
run_kshot_poc.py  —  augmentation/
====================================
PoC 1 v4: K-Shot Transductive Semi-Supervised GSL

The correct evaluation regime for a low-data paper.

Protocol per (K, seed, episode):
  1. Pool all target molecules (train + val + test combined).
  2. Sample K molecules as SUPPORT (labeled).
  3. Remaining molecules = QUERY (evaluation + validation).
  4. Build full graph: support + oracle-anchors; loss masked to support nodes.
  5. Evaluate on query in original label space.
  6. Average over N_EPISODES × SEEDS.

K ∈ {10, 20, 50}  |  N_EPISODES=5  |  SEEDS=5
Results → augmentation/results_kshot/
Set QUICK=1 for smoke-test (K=10 only, 1 seed, 1 episode, 3 epochs).
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
RESULTS_DIR  = os.path.join(SCRIPT_DIR,   "results_kshot")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ───────────────────────────────────────────────────────────────────
QUICK = os.environ.get("QUICK", "").lower() in ("1", "true", "yes")

ORACLE_SEEDS      = [40] if QUICK else [40, 41, 42]
EXPERIMENT_SEEDS  = [0]  if QUICK else [0, 1, 2, 3, 4]
K_SHOTS           = [10] if QUICK else [10, 20, 50]
N_EPISODES        = 1    if QUICK else 5     # episode draws per (K, seed)
MAX_EP_ORACLE     = 3    if QUICK else 60
MAX_EP_FINAL      = 3    if QUICK else 100   # more epochs — tiny support needs longer
PATIENCE          = 15                        # more patience for K-shot
LR_PATIENCE       = 7
LR                = 5e-4                      # slightly lower lr for stability
BATCH_ORACLE      = 256
NIG_COEFF         = 0.1
DIR_COEFF         = 0.1
AUX_W             = 0.5
SIM_THRESH        = 0.75
EPIST_PCTILE      = 20
MOLFORMER         = "ibm/MoLFormer-XL-both-10pct"
CONDITIONS        = ["target_only", "random_aug", "similarity_only", "evidential_oracle"]

_mf_cache: dict = {}

# ══════════════════════════════════════════════════════════════════════════════
# §1  EMBEDDING INFRASTRUCTURE  — identical to v3
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
    """Concatenate train+val+test into a single pool for K-shot sampling."""
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
# §4  MODEL DEFINITIONS  — identical to v3
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


# ══════════════════════════════════════════════════════════════════════════════
# §5b  ORACLE CACHE  — identical to v3, shares same cache files
# ══════════════════════════════════════════════════════════════════════════════

def _oracle_cache_paths(src_name, tgt_name, seeds):
    tag = f"{src_name}__{tgt_name}__seeds_{'_'.join(str(s) for s in seeds)}"
    return (os.path.join(ORACLE_CACHE, f"{tag}_u_ep.pt"),
            os.path.join(ORACLE_CACHE, f"{tag}_sel_mask.pt"))

def load_oracle_cache(src_name, tgt_name, seeds):
    u_path, m_path = _oracle_cache_paths(src_name, tgt_name, seeds)
    if os.path.exists(u_path) and os.path.exists(m_path):
        u_ep     = torch.load(u_path, weights_only=True)
        sel_mask = torch.load(m_path, weights_only=True)
        print(f"  [Oracle cache] Loaded {src_name}→{tgt_name} (seeds={seeds})")
        return u_ep, sel_mask
    return None

def save_oracle_cache(src_name, tgt_name, seeds, u_ep, sel_mask):
    os.makedirs(ORACLE_CACHE, exist_ok=True)
    u_path, m_path = _oracle_cache_paths(src_name, tgt_name, seeds)
    torch.save(u_ep, u_path); torch.save(sel_mask, m_path)
    print(f"  [Oracle cache] Saved {src_name}→{tgt_name}")


# ══════════════════════════════════════════════════════════════════════════════
# §6  ORACLE TRAINING  — identical to v3
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
                l = val_loss_fn(mdl, batch); b = batch[0].size(0)
                vl += l.item() * b; n += b
        vl /= n; sch.step(vl)
        if vl < best_vl: best_vl = vl; best_st = copy.deepcopy(mdl.state_dict()); no_imp = 0
        else:             no_imp += 1
        if no_imp >= PATIENCE: break
    mdl.load_state_dict(best_st); return mdl


def train_reg_oracle(src, seed):
    set_seed(seed)
    sc     = StandardScaler()
    tr_np  = src["train"]["tgt"].numpy().reshape(-1, 1)
    val_np = src["val"]["tgt"].numpy().reshape(-1, 1)
    sc.fit(tr_np)
    tr_s   = torch.tensor(sc.transform(tr_np).flatten(),  dtype=torch.float32)
    val_s  = torch.tensor(sc.transform(val_np).flatten(), dtype=torch.float32)
    tr_ldr  = gsl_loader(src["train"]["emb"], tr_s,   src["train"]["smi"], BATCH_ORACLE, True)
    val_ldr = gsl_loader(src["val"]["emb"],   val_s,  src["val"]["smi"],   BATCH_ORACLE)
    mdl = EvidentialGSL().to(DEVICE)
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.5, 5)
    def lfn(m, batch):
        X, y, A = [x.to(DEVICE) for x in batch]
        (m0,v0,a0,b0),(mu,v,al,be) = m(X, A)
        return nig_loss(y,mu,v,al,be) + AUX_W * nig_loss(y,m0,v0,a0,b0)
    _early_stop_loop(mdl, tr_ldr, val_ldr, opt, sch, MAX_EP_ORACLE, lfn)
    for p in mdl.parameters(): p.requires_grad_(False)
    return mdl


def train_cls_oracle(src, seed):
    set_seed(seed)
    tr_ldr  = gsl_loader(src["train"]["emb"], src["train"]["tgt"].long(),
                          src["train"]["smi"], BATCH_ORACLE, True)
    val_ldr = gsl_loader(src["val"]["emb"],   src["val"]["tgt"].long(),
                          src["val"]["smi"],   BATCH_ORACLE)
    mdl = DirGSL(K=2).to(DEVICE)
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.5, 5)
    def lfn(m, batch):
        X, y, A = [x.to(DEVICE) for x in batch]
        a0, af  = m(X, A)
        oh = F.one_hot(y, 2).float()
        return dirichlet_loss(oh, af) + AUX_W * dirichlet_loss(oh, a0)
    _early_stop_loop(mdl, tr_ldr, val_ldr, opt, sch, MAX_EP_ORACLE, lfn)
    for p in mdl.parameters(): p.requires_grad_(False)
    return mdl


@torch.no_grad()
def reg_uncertainty(oracle, emb, smi):
    oracle.eval(); u = []
    for X, _, A in gsl_loader(emb, torch.zeros(len(emb)), smi, BATCH_ORACLE):
        X, A = X.to(DEVICE), A.to(DEVICE)
        _, (_, v, _, _) = oracle(X, A)
        u.append((1.0 / v.clamp(1e-8)).cpu())
    return torch.cat(u)


@torch.no_grad()
def cls_uncertainty(oracle, emb, smi):
    oracle.eval(); u = []
    for X, _, A in gsl_loader(emb, torch.zeros(len(emb)), smi, BATCH_ORACLE):
        X, A = X.to(DEVICE), A.to(DEVICE)
        _, af = oracle(X, A)
        u.append((af.size(-1) / af.sum(-1).clamp(1e-8)).cpu())
    return torch.cat(u)


def ensemble_reg_uncertainty(oracles, emb, smi):
    return torch.stack([reg_uncertainty(o, emb, smi) for o in oracles]).mean(0)

def ensemble_cls_uncertainty(oracles, emb, smi):
    return torch.stack([cls_uncertainty(o, emb, smi) for o in oracles]).mean(0)


# ══════════════════════════════════════════════════════════════════════════════
# §7  ORACLE SELECTION  — identical to v3
# ══════════════════════════════════════════════════════════════════════════════

def select_oracle(src_emb, tgt_emb, u_ep, max_n):
    centroid = F.normalize(tgt_emb.mean(0, keepdim=True), dim=-1)
    sims     = (F.normalize(src_emb, dim=-1) @ centroid.T).squeeze(-1)
    s_mask   = sims > SIM_THRESH
    thr      = float(np.percentile(u_ep.numpy(), EPIST_PCTILE))
    e_mask   = u_ep < thr
    sel      = s_mask & e_mask
    if sel.sum() < 5: sel = s_mask
    if sel.sum() < 5:
        k  = max(5, int(0.05 * len(src_emb)))
        _, ti = sims.topk(k)
        sel = torch.zeros(len(src_emb), dtype=torch.bool); sel[ti] = True
    if sel.sum() > max_n:
        u_sel = u_ep[sel]; si = u_sel.argsort()[:max_n]
        fi    = sel.nonzero(as_tuple=True)[0]
        sel   = torch.zeros(len(src_emb), dtype=torch.bool); sel[fi[si]] = True
    return sel


# ══════════════════════════════════════════════════════════════════════════════
# §8  ANCHOR BUILDERS  — identical to v3
# ══════════════════════════════════════════════════════════════════════════════

def build_anchors_target_only():
    return torch.zeros(0, 768), []

def build_anchors_random(src_emb, src_smi, max_n, seed):
    g    = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(src_emb), generator=g)[:max_n].tolist()
    return src_emb[perm], [src_smi[i] for i in perm]

def build_anchors_similarity(src_emb, src_smi, tgt_emb, max_n):
    centroid = F.normalize(tgt_emb.mean(0, keepdim=True), dim=-1)
    sims     = (F.normalize(src_emb, dim=-1) @ centroid.T).squeeze(-1)
    mask     = sims > SIM_THRESH
    if mask.sum() == 0:
        _, ti = sims.topk(max_n)
        mask = torch.zeros(len(sims), dtype=torch.bool); mask[ti] = True
    idx = mask.nonzero(as_tuple=True)[0][:max_n].tolist()
    return src_emb[idx], [src_smi[i] for i in idx]

def build_anchors_oracle(src_emb, src_smi, sel_mask):
    idx = sel_mask.nonzero(as_tuple=True)[0].tolist()
    return src_emb[idx], [src_smi[i] for i in idx]


# ══════════════════════════════════════════════════════════════════════════════
# §9  K-SHOT EPISODE SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def sample_reg_episode(all_emb, all_tgt, all_smi, K, rng):
    """
    Sample K support molecules from the full target pool.
    Returns: (sup_emb, sup_tgt, sup_smi, qry_emb, qry_tgt, qry_smi, scaler)
    Scaler is fit on K support targets only.
    """
    n       = len(all_emb)
    sup_idx = rng.choice(n, K, replace=False)
    qry_idx = np.setdiff1d(np.arange(n), sup_idx)

    sup_emb = all_emb[sup_idx]; sup_tgt = all_tgt[sup_idx]
    sup_smi = [all_smi[i] for i in sup_idx]
    qry_emb = all_emb[qry_idx]; qry_tgt = all_tgt[qry_idx]
    qry_smi = [all_smi[i] for i in qry_idx]

    # Scaler fit on K support targets
    sc = StandardScaler()
    sc.fit(sup_tgt.numpy().reshape(-1, 1))
    def scale(t): return torch.tensor(
        sc.transform(t.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32)

    return sup_emb, scale(sup_tgt), sup_smi, qry_emb, scale(qry_tgt), qry_smi, sc


def sample_cls_episode(all_emb, all_tgt, all_smi, K, rng):
    """
    Sample K support molecules with stratification (K//2 per class).
    Returns: (sup_emb, sup_tgt, sup_smi, qry_emb, qry_tgt, qry_smi)
    """
    tgt_np = all_tgt.numpy()
    pos    = np.where(tgt_np == 1)[0]
    neg    = np.where(tgt_np == 0)[0]
    Kh     = max(1, min(K // 2, len(pos) // 2, len(neg) // 2))

    sup_idx = np.concatenate([rng.choice(pos, Kh, replace=False),
                               rng.choice(neg, Kh, replace=False)])
    qry_idx = np.setdiff1d(np.arange(len(all_emb)), sup_idx)

    sup_emb = all_emb[sup_idx]; sup_tgt = all_tgt[sup_idx]
    sup_smi = [all_smi[i] for i in sup_idx]
    qry_emb = all_emb[qry_idx]; qry_tgt = all_tgt[qry_idx]
    qry_smi = [all_smi[i] for i in qry_idx]

    return sup_emb, sup_tgt, sup_smi, qry_emb, qry_tgt, qry_smi


# ══════════════════════════════════════════════════════════════════════════════
# §10  TRANSDUCTIVE TRAINING  (K-shot version — query used for early stopping)
# ══════════════════════════════════════════════════════════════════════════════

def train_reg_transductive_kshot(sup_emb, sup_tgt_s, sup_smi,
                                  anc_emb, anc_smi,
                                  qry_emb, qry_tgt_s, qry_smi, seed):
    """
    Transductive EvidentialGSL trained on K support molecules.
    Query molecules used for early stopping (standard few-shot protocol).
    """
    set_seed(seed)
    n_sup = len(sup_emb)

    if len(anc_emb) > 0:
        all_emb = torch.cat([sup_emb, anc_emb])
        all_smi = sup_smi + anc_smi
    else:
        all_emb = sup_emb; all_smi = sup_smi

    mdl = EvidentialGSL().to(DEVICE)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.5, LR_PATIENCE)

    # Precompute ECFP once
    A_full = ecfp_matrix(all_smi).to(DEVICE)
    A_qry  = ecfp_matrix(qry_smi).to(DEVICE)
    X_all  = all_emb.to(DEVICE)
    y_sup  = sup_tgt_s.to(DEVICE)
    X_qry  = qry_emb.to(DEVICE)
    y_qry  = qry_tgt_s.to(DEVICE)

    best_vl, best_st, no_imp = float("inf"), None, 0

    for _ in range(MAX_EP_FINAL):
        mdl.train()
        (m0,v0,a0,b0), (mu,v,al,be) = mdl(X_all, A_full)
        # Loss ONLY on support (labeled) slice [:n_sup]
        loss = (nig_loss(y_sup, mu[:n_sup], v[:n_sup], al[:n_sup], be[:n_sup])
                + AUX_W * nig_loss(y_sup, m0[:n_sup], v0[:n_sup], a0[:n_sup], b0[:n_sup]))
        opt.zero_grad(); loss.backward(); opt.step()

        # Validate on query (early stopping only — labels not used in training)
        mdl.eval()
        with torch.no_grad():
            _, (mu_q, v_q, al_q, be_q) = mdl(X_qry, A_qry)
            vl = nig_loss(y_qry, mu_q, v_q, al_q, be_q).item()
        sch.step(vl)

        if vl < best_vl: best_vl = vl; best_st = copy.deepcopy(mdl.state_dict()); no_imp = 0
        else:             no_imp += 1
        if no_imp >= PATIENCE: break

    mdl.load_state_dict(best_st)
    return mdl


def train_cls_transductive_kshot(sup_emb, sup_tgt, sup_smi,
                                  anc_emb, anc_smi,
                                  qry_emb, qry_tgt, qry_smi, seed):
    """
    Transductive DirGSL trained on K support molecules.
    """
    set_seed(seed)
    n_sup = len(sup_emb)

    if len(anc_emb) > 0:
        all_emb = torch.cat([sup_emb, anc_emb])
        all_smi = sup_smi + anc_smi
    else:
        all_emb = sup_emb; all_smi = sup_smi

    mdl = DirGSL(K=2).to(DEVICE)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.5, LR_PATIENCE)

    A_full  = ecfp_matrix(all_smi).to(DEVICE)
    A_qry   = ecfp_matrix(qry_smi).to(DEVICE)
    X_all   = all_emb.to(DEVICE)
    y_sup   = sup_tgt.long().to(DEVICE)
    oh_sup  = F.one_hot(y_sup, 2).float()
    X_qry   = qry_emb.to(DEVICE)
    oh_qry  = F.one_hot(qry_tgt.long().to(DEVICE), 2).float()

    best_vl, best_st, no_imp = float("inf"), None, 0

    for _ in range(MAX_EP_FINAL):
        mdl.train()
        a0, af = mdl(X_all, A_full)
        loss = (dirichlet_loss(oh_sup, af[:n_sup])
                + AUX_W * dirichlet_loss(oh_sup, a0[:n_sup]))
        opt.zero_grad(); loss.backward(); opt.step()

        mdl.eval()
        with torch.no_grad():
            a0_q, af_q = mdl(X_qry, A_qry)
            vl = (dirichlet_loss(oh_qry, af_q)
                  + AUX_W * dirichlet_loss(oh_qry, a0_q)).item()
        sch.step(vl)

        if vl < best_vl: best_vl = vl; best_st = copy.deepcopy(mdl.state_dict()); no_imp = 0
        else:             no_imp += 1
        if no_imp >= PATIENCE: break

    mdl.load_state_dict(best_st)
    return mdl


# ══════════════════════════════════════════════════════════════════════════════
# §11  EVALUATION  — identical to v3
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_reg(mdl, qry_emb, qry_tgt_s, qry_smi, scaler) -> float:
    mdl.eval(); mus = []
    for X, _, A in gsl_loader(qry_emb, qry_tgt_s, qry_smi, BATCH_ORACLE):
        X, A = X.to(DEVICE), A.to(DEVICE)
        _, (mu, *_) = mdl(X, A); mus.append(mu.cpu())
    mu_s  = torch.cat(mus).numpy()
    tgt_s = qry_tgt_s.numpy()
    mu_o  = scaler.inverse_transform(mu_s.reshape(-1, 1)).flatten()
    tgt_o = scaler.inverse_transform(tgt_s.reshape(-1, 1)).flatten()
    return float(np.sqrt(np.mean((mu_o - tgt_o) ** 2)))


@torch.no_grad()
def eval_cls(mdl, qry_emb, qry_tgt, qry_smi) -> float:
    mdl.eval(); probs = []
    for X, _, A in gsl_loader(qry_emb, torch.zeros(len(qry_emb)), qry_smi, BATCH_ORACLE):
        X, A = X.to(DEVICE), A.to(DEVICE)
        _, af = mdl(X, A)
        probs.append((af[:, 1] / af.sum(-1)).cpu())
    tgt_np = qry_tgt.numpy()
    if len(np.unique(tgt_np)) < 2:
        return float("nan")
    return float(roc_auc_score(tgt_np, torch.cat(probs).numpy()))


# ══════════════════════════════════════════════════════════════════════════════
# §12  RESULT I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved → {path}")


def write_summary(all_rows_A, all_rows_B, path):
    lines = ["=" * 65,
             "PoC 1 v4 — K-Shot Transductive Semi-Supervised GSL",
             "=" * 65]
    for label, all_rows, metric, up in [
        ("Exp A  Classification (AUROC ↑)", all_rows_A, "AUROC", True),
        ("Exp B  Regression     (RMSE  ↓)", all_rows_B, "RMSE",  False),
    ]:
        lines += ["", label]
        for K in K_SHOTS:
            lines.append(f"  K={K}:")
            rows = [r for r in all_rows if r["K"] == K]
            for cond in CONDITIONS:
                vals = [r[cond] for r in rows if not np.isnan(r[cond])]
                if vals:
                    lines.append(f"    {cond:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# §13  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 65)
    print("PoC 1 v4: K-Shot Transductive Semi-Supervised GSL")
    print(f"Device: {DEVICE}  |  K: {K_SHOTS}  |  Seeds: {EXPERIMENT_SEEDS}"
          f"  |  Episodes: {N_EPISODES}  |  Quick: {QUICK}")
    print("=" * 65)

    # ── §0  Embeddings ────────────────────────────────────────────────────────
    print("\n[Step 0] Loading / caching embeddings …")
    lipo    = get_dataset_embeddings("Lipophilicity_AstraZeneca")
    caco    = get_dataset_embeddings("Caco2_Wang")
    cyp_src = get_dataset_embeddings("CYP2C9_Veith")
    cyp_tgt = get_dataset_embeddings("CYP2C9_Substrate_CarbonMangels")
    print("  All datasets ready.\n")

    # Full target pools for K-shot sampling
    caco_all    = get_full_dataset(caco)
    cyp_tgt_all = get_full_dataset(cyp_tgt)

    # Oracle anchors: use full source train for selection (same as v3)
    max_anc_reg = 100    # fixed cap — K is tiny, 100 anchors gives high ratio at K=10
    max_anc_cls = 100

    # ── §1  Oracle (shared cache with v3) ────────────────────────────────────
    cached = load_oracle_cache("Lipophilicity_AstraZeneca", "Caco2_Wang", ORACLE_SEEDS)
    if cached:
        u_ep_reg, sel_reg = cached
        # Re-cap to max_anc_reg (cache may have more)
        if sel_reg.sum() > max_anc_reg:
            u_sel = u_ep_reg[sel_reg]
            si    = u_sel.argsort()[:max_anc_reg]
            fi    = sel_reg.nonzero(as_tuple=True)[0]
            sel_reg = torch.zeros(len(sel_reg), dtype=torch.bool); sel_reg[fi[si]] = True
        print(f"  Skipping oracle — cache ({sel_reg.sum()} anchors after cap).\n")
    else:
        print(f"[Step 1] Training regression oracle ensemble …")
        oracles_reg = [train_reg_oracle(lipo, s) for s in ORACLE_SEEDS]
        u_ep_reg    = ensemble_reg_uncertainty(oracles_reg,
                                               lipo["train"]["emb"], lipo["train"]["smi"])
        sel_reg     = select_oracle(lipo["train"]["emb"], caco["train"]["emb"],
                                    u_ep_reg, max_anc_reg)
        save_oracle_cache("Lipophilicity_AstraZeneca","Caco2_Wang", ORACLE_SEEDS,
                          u_ep_reg, sel_reg)

    cached = load_oracle_cache("CYP2C9_Veith","CYP2C9_Substrate_CarbonMangels", ORACLE_SEEDS)
    if cached:
        u_ep_cls, sel_cls = cached
        if sel_cls.sum() > max_anc_cls:
            u_sel = u_ep_cls[sel_cls]; si = u_sel.argsort()[:max_anc_cls]
            fi    = sel_cls.nonzero(as_tuple=True)[0]
            sel_cls = torch.zeros(len(sel_cls), dtype=torch.bool); sel_cls[fi[si]] = True
        print(f"  Skipping oracle — cache ({sel_cls.sum()} anchors after cap).\n")
    else:
        print(f"[Step 2] Training classification oracle ensemble …")
        oracles_cls = [train_cls_oracle(cyp_src, s) for s in ORACLE_SEEDS]
        u_ep_cls    = ensemble_cls_uncertainty(oracles_cls,
                                               cyp_src["train"]["emb"], cyp_src["train"]["smi"])
        sel_cls     = select_oracle(cyp_src["train"]["emb"], cyp_tgt["train"]["emb"],
                                    u_ep_cls, max_anc_cls)
        save_oracle_cache("CYP2C9_Veith","CYP2C9_Substrate_CarbonMangels", ORACLE_SEEDS,
                          u_ep_cls, sel_cls)

    # Pre-build anchors (fixed across all K and seeds — only selection changes)
    anc_reg_oracle  = build_anchors_oracle(lipo["train"]["emb"], lipo["train"]["smi"], sel_reg)
    anc_cls_oracle  = build_anchors_oracle(cyp_src["train"]["emb"], cyp_src["train"]["smi"], sel_cls)
    anc_reg_sim     = build_anchors_similarity(lipo["train"]["emb"], lipo["train"]["smi"],
                                               caco["train"]["emb"], max_anc_reg)
    anc_cls_sim     = build_anchors_similarity(cyp_src["train"]["emb"], cyp_src["train"]["smi"],
                                               cyp_tgt["train"]["emb"], max_anc_cls)

    print(f"  Reg anchors: oracle={sel_reg.sum()}, sim={len(anc_reg_sim[1])}")
    print(f"  Cls anchors: oracle={sel_cls.sum()}, sim={len(anc_cls_sim[1])}\n")

    # ── §2  K-shot experiment loops ───────────────────────────────────────────
    all_rows_A, all_rows_B = [], []

    for K in K_SHOTS:
        print(f"\n{'═'*65}")
        print(f"K = {K}")
        print(f"{'═'*65}")

        for seed in EXPERIMENT_SEEDS:
            for ep in range(N_EPISODES):
                rng   = np.random.RandomState(seed * 1000 + ep)
                run_id = f"K{K}_seed{seed}_ep{ep}"

                # ── Regression episode ────────────────────────────────────────
                (sup_emb, sup_tgt_s, sup_smi,
                 qry_emb, qry_tgt_s, qry_smi, sc) = sample_reg_episode(
                    caco_all["emb"], caco_all["tgt"], caco_all["smi"], K, rng)

                print(f"\n  [Exp B | {run_id}]  sup={len(sup_emb)} qry={len(qry_emb)}")
                row_B = {"K": K, "seed": seed, "episode": ep}

                anc_reg_rand = build_anchors_random(
                    lipo["train"]["emb"], lipo["train"]["smi"], max_anc_reg,
                    seed * 1000 + ep)

                for cond in CONDITIONS:
                    if cond == "target_only":   anc = build_anchors_target_only()
                    elif cond == "random_aug":  anc = anc_reg_rand
                    elif cond == "similarity_only": anc = anc_reg_sim
                    else:                           anc = anc_reg_oracle

                    mdl  = train_reg_transductive_kshot(
                        sup_emb, sup_tgt_s, sup_smi, anc[0], anc[1],
                        qry_emb, qry_tgt_s, qry_smi, seed + ep)
                    rmse = eval_reg(mdl, qry_emb, qry_tgt_s, qry_smi, sc)
                    row_B[cond] = rmse
                    print(f"    {cond:25s}: RMSE={rmse:.4f}")

                all_rows_B.append(row_B)

                # ── Classification episode ────────────────────────────────────
                rng2 = np.random.RandomState(seed * 1000 + ep + 500)
                (sup_emb, sup_tgt, sup_smi,
                 qry_emb, qry_tgt, qry_smi) = sample_cls_episode(
                    cyp_tgt_all["emb"], cyp_tgt_all["tgt"], cyp_tgt_all["smi"], K, rng2)

                print(f"\n  [Exp A | {run_id}]  sup={len(sup_emb)} qry={len(qry_emb)}")
                row_A = {"K": K, "seed": seed, "episode": ep}

                anc_cls_rand = build_anchors_random(
                    cyp_src["train"]["emb"], cyp_src["train"]["smi"], max_anc_cls,
                    seed * 1000 + ep)

                for cond in CONDITIONS:
                    if cond == "target_only":       anc = build_anchors_target_only()
                    elif cond == "random_aug":      anc = anc_cls_rand
                    elif cond == "similarity_only": anc = anc_cls_sim
                    else:                           anc = anc_cls_oracle

                    mdl   = train_cls_transductive_kshot(
                        sup_emb, sup_tgt, sup_smi, anc[0], anc[1],
                        qry_emb, qry_tgt, qry_smi, seed + ep)
                    auroc = eval_cls(mdl, qry_emb, qry_tgt, qry_smi)
                    row_A[cond] = auroc if not np.isnan(auroc) else float("nan")
                    print(f"    {cond:25s}: AUROC={auroc:.4f}" if not np.isnan(auroc)
                          else f"    {cond:25s}: AUROC=  nan (single class in query)")

                all_rows_A.append(row_A)

    # ── §3  Save ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    save_csv(all_rows_A, os.path.join(RESULTS_DIR, "exp_A_classification_kshot.csv"))
    save_csv(all_rows_B, os.path.join(RESULTS_DIR, "exp_B_regression_kshot.csv"))
    write_summary(all_rows_A, all_rows_B, os.path.join(RESULTS_DIR, "summary.txt"))
    print(f"\nAll results saved → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
