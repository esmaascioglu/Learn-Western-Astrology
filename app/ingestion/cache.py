"""On-disk cache of (chunks + embeddings) per book.

Embedding ~760 chunks on CPU takes tens of minutes, so we persist results and
keep them separate from the upsert. A DB hiccup then costs seconds to retry,
not a full re-embed.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from app.config import PROJECT_ROOT
from app.ingestion.chunk import Chunk

ARTIFACTS = PROJECT_ROOT / "artifacts"


def _path(slug: str) -> Path:
    return ARTIFACTS / f"{slug}.pkl"


def exists(slug: str) -> bool:
    return _path(slug).exists()


def save(slug: str, chunks: list[Chunk], embeddings: list[list[float]],
         sparse: list[dict[int, float]] | None = None) -> Path:
    ARTIFACTS.mkdir(exist_ok=True)
    p = _path(slug)
    with p.open("wb") as f:
        pickle.dump(
            {"chunks": chunks, "embeddings": embeddings, "sparse": sparse}, f
        )
    return p


def load(slug: str) -> tuple[list[Chunk], list[list[float]], list[dict[int, float]] | None]:
    with _path(slug).open("rb") as f:
        data = pickle.load(f)
    return data["chunks"], data["embeddings"], data.get("sparse")
