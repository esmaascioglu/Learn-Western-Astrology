"""SPLADE vector pruning — bounds each sparse vector to its top-K terms.

Tests the pruning math against a hand-built logits tensor, so no model download
or inference is needed.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.retrieval.splade import _prune  # noqa: E402


def test_prune_keeps_only_the_top_k_highest_weights():
    vec = torch.tensor([0.0, 5.0, 0.0, 3.0, 1.0])
    pruned = _prune(vec, top_k=2)
    assert pruned == {1: 5.0, 3: 3.0}


def test_prune_returns_all_nonzero_when_fewer_than_k():
    vec = torch.tensor([0.0, 5.0, 0.0, 3.0, 1.0])
    assert _prune(vec, top_k=10) == {1: 5.0, 3: 3.0, 4: 1.0}


def test_prune_drops_zero_weights():
    vec = torch.zeros(8)
    vec[2] = 0.7
    assert _prune(vec, top_k=4) == {2: pytest.approx(0.7)}
