"""
generate_embeddings.py
======================
Phase 1A — Step 1: Load the Caco-2 dataset from TDC, generate frozen
MolFormer-XL embeddings, and save the resulting tensors to disk.

Usage:
    python generate_embeddings.py
"""

import os
import time

import numpy as np
import torch
from tdc.single_pred import ADME
from transformers import AutoModel, AutoTokenizer

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
BATCH_SIZE = 32  # keeps peak RAM reasonable on CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Helper: batched embedding extraction ─────────────────────────────────────
def generate_embeddings(
    smiles_list: list[str], tokenizer, model, batch_size: int = BATCH_SIZE
) -> torch.Tensor:
    """
    Tokenise a list of SMILES, pass them through the frozen MolFormer
    encoder, and return the mean-pooled hidden states (768-d per molecule).

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to embed.
    tokenizer : transformers.PreTrainedTokenizer
        MolFormer tokenizer.
    model : transformers.PreTrainedModel
        Frozen MolFormer model.
    batch_size : int
        Number of SMILES to process at once.

    Returns
    -------
    embeddings : torch.Tensor, shape (N, 768)
    """
    all_embeddings = []

    batch_times = []
    total_batches = (len(smiles_list) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(smiles_list), batch_size), start=1):
        batch_start = time.perf_counter()
        batch = smiles_list[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean-pool over the sequence-length dimension → (batch, 768)
        hidden_states = outputs.last_hidden_state  # (B, seq_len, 768)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (B, seq_len, 1)
        # Mask padded positions before averaging
        masked_hidden = hidden_states * attention_mask
        embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)

        all_embeddings.append(embeddings.cpu())
        batch_times.append(time.perf_counter() - batch_start)
        avg_batch_s = sum(batch_times) / len(batch_times)
        eta_s = avg_batch_s * (total_batches - batch_idx)
        print(
            f"  Processed {min(start + batch_size, len(smiles_list)):>5d}"
            f" / {len(smiles_list)} molecules"
            f"  |  ETA: {format_eta(eta_s)}"
        )

    return torch.cat(all_embeddings, dim=0)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Load the Caco-2 Wang dataset via PyTDC
    print("[1/4] Loading Caco-2 Wang dataset from TDC …")
    data = ADME(name="Caco2_Wang")
    split = data.get_split(method="scaffold")

    train_df = split["train"]
    val_df = split["valid"]
    test_df = split["test"]
    print(f"       Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Load the frozen MolFormer model
    print(f"[2/4] Loading model '{MODEL_NAME}' …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()
    print(f"       Model loaded on {DEVICE}")

    # 3. Generate embeddings for each split
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"[3/4] Generating {split_name} embeddings …")
        smiles = df["Drug"].tolist()
        targets = df["Y"].values.astype(np.float32)

        embeddings = generate_embeddings(smiles, tokenizer, model)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        # 4. Save to disk
        emb_path = os.path.join(DATA_DIR, f"{split_name}_embeddings.pt")
        tgt_path = os.path.join(DATA_DIR, f"{split_name}_targets.pt")
        smi_path = os.path.join(DATA_DIR, f"{split_name}_smiles.pt")
        torch.save(embeddings, emb_path)
        torch.save(targets_tensor, tgt_path)
        torch.save(smiles, smi_path)
        print(f"       Saved {emb_path}  ({embeddings.shape})")
        print(f"       Saved {tgt_path}  ({targets_tensor.shape})")
        print(f"       Saved {smi_path}  ({len(smiles)} smiles)")

    print("[4/4] Done — all embeddings saved to data/")


if __name__ == "__main__":
    main()
