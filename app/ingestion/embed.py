"""Dense embeddings via OpenAI (text-embedding-3-small) through LiteLLM.

We request `embedding_dim` dimensions (Matryoshka) so the vectors stay 1024-d and
the existing pgvector schema/index is unchanged. Documents and queries are encoded
identically.
"""
from __future__ import annotations

import os

from app.config import get_settings


def _ensure_key() -> None:
    s = get_settings()
    if s.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", s.openai_api_key)


def embed_documents(texts: list[str], batch_size: int = 256) -> list[list[float]]:
    import litellm

    _ensure_key()
    s = get_settings()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        resp = litellm.embedding(
            model=s.embedding_model,
            input=texts[i:i + batch_size],
            dimensions=s.embedding_dim,
        )
        out.extend(item["embedding"] for item in resp.data)
    return out


def embed_query(text: str) -> list[float]:
    import litellm

    _ensure_key()
    s = get_settings()
    resp = litellm.embedding(
        model=s.embedding_model, input=[text], dimensions=s.embedding_dim
    )
    return resp.data[0]["embedding"]
