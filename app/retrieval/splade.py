"""Local SPLADE sparse encoder — learned sparse lexical retrieval (torch/CPU).

A SPLADE model is a Masked-Language-Model (DistilBERT) whose MLM head projects
every input token onto the whole vocabulary, giving an importance logit for each
vocab term. Those are turned into one sparse vector over the vocabulary:

    w_ij = MLM_head(DistilBERT(text))      # token i → vocab term j importance
    a_ij = log(1 + relu(w_ij))             # non-negative, saturating
    s_j  = max_i  a_ij                      # max-pool over input tokens

The result is sparse (most of the ~30k dims are 0) and includes *expansion*
terms the model infers but that are not literally present — e.g. a "Taurus is
immovable" passage activates the term "stubborn". That closes the vocabulary gap
which exact-match BM25 (no expansion) and single-vector dense retrieval (opaque
1024-d compression) both miss. Query and document are encoded identically and
matched by sparse dot product.

Encoded once per chunk at ingest (offline, batched) and once per query at search
time. Memory-conscious: small DistilBERT (~66M params), batched inference under
`torch.no_grad`, in-place activations to avoid extra [B,T,V] copies, and each
vector pruned to its top-K terms (`config.splade_top_k`).
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings

# A sparse vector is represented as {vocab_term_id: weight}; tiny (≈top_k entries),
# pickle-friendly for the ingest cache, and converts directly to pgvector sparsevec.
SparseVec = dict[int, float]


@lru_cache(maxsize=1)
def _load():
    """Lazy singleton: load tokenizer + MLM model once, on first use only."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    s = get_settings()
    tok = AutoTokenizer.from_pretrained(s.splade_model)
    model = AutoModelForMaskedLM.from_pretrained(s.splade_model)
    model.eval()
    torch.set_grad_enabled(False)
    return tok, model


def _prune(vec, top_k: int) -> SparseVec:
    """Keep only the top_k highest-weight non-zero terms of a [V] tensor."""
    import torch

    nz = torch.nonzero(vec, as_tuple=False).squeeze(-1)
    if nz.numel() > top_k:
        vals = vec[nz]
        _, idx = torch.topk(vals, top_k)
        nz = nz[idx]
    return {int(j): float(vec[j]) for j in nz.tolist()}


def encode(texts: list[str], batch_size: int | None = None, top_k: int | None = None
           ) -> list[SparseVec]:
    """Encode texts into pruned SPLADE sparse vectors (list of {term_id: weight})."""
    import torch

    s = get_settings()
    batch_size = batch_size or s.splade_batch_size
    top_k = top_k or s.splade_top_k
    tok, model = _load()

    out: list[SparseVec] = []
    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i:i + batch_size],
            padding=True, truncation=True,
            max_length=s.splade_max_length, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**enc).logits                  # [B, T, V]
            logits.relu_().log1p_()                        # a_ij, in place
            mask = enc["attention_mask"].unsqueeze(-1)     # [B, T, 1]
            logits.mul_(mask)                              # zero out padding tokens
            vecs = logits.max(dim=1).values                # [B, V] — max-pool
            del logits
        out.extend(_prune(v, top_k) for v in vecs)
    return out


def encode_query(text: str) -> SparseVec:
    """Encode a single query string into a SPLADE sparse vector."""
    return encode([text], batch_size=1)[0]


def to_pgvector(vec: SparseVec):
    """Convert a {term_id: weight} dict to a pgvector SparseVector for SQL binding."""
    from pgvector import SparseVector

    return SparseVector(vec, get_settings().splade_vocab_dim)
