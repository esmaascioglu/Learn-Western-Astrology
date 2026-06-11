"""Settings: cached singleton + the defaults the schema/pipeline rely on."""
from __future__ import annotations

from app.config import get_settings


def test_settings_is_a_cached_singleton():
    assert get_settings() is get_settings()


def test_vector_dimensions_match_the_db_schema():
    s = get_settings()
    # These must stay in lockstep with db/schema.sql (vector(1024), sparsevec(30522)).
    assert s.embedding_dim == 1024
    assert s.splade_vocab_dim == 30522


def test_retrieval_defaults_present():
    s = get_settings()
    assert s.splade_top_k == 256
    assert s.top_n > 0
