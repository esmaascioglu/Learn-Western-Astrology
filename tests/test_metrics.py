"""Ranking metrics: hand-computed expectations for recall@k, MRR, nDCG.

Pure math (app/retrieval/metrics.py) — no DB, no network. Gold labels are
section-level: a section counts as found when ANY of its chunks is retrieved.
"""
from __future__ import annotations

import math

import pytest

from app.retrieval.metrics import mrr, ndcg_at_k, section_recall_at_k

# Two gold sections: one with chunks {1, 2}, one with chunk {7}.
GOLD_SECTIONS = [{1, 2}, {7}]
GOLD_IDS = {1, 2, 7}


# --- section_recall_at_k ---------------------------------------------------
def test_recall_counts_a_section_found_via_any_of_its_chunks():
    # Chunk 2 belongs to the first section, so that section is "found" even
    # though chunk 1 never shows up. The {7} section is missed → 1 of 2.
    assert section_recall_at_k([2, 9, 8], GOLD_SECTIONS, k=3) == 0.5


def test_recall_is_perfect_when_every_section_is_represented():
    assert section_recall_at_k([7, 1], GOLD_SECTIONS, k=2) == 1.0


def test_recall_respects_the_k_cutoff():
    # Chunk 7 sits at rank 3 — invisible to recall@2, visible to recall@3.
    ranked = [9, 1, 7]
    assert section_recall_at_k(ranked, GOLD_SECTIONS, k=2) == 0.5
    assert section_recall_at_k(ranked, GOLD_SECTIONS, k=3) == 1.0


def test_recall_with_no_gold_or_no_hits_is_zero():
    assert section_recall_at_k([1, 2], [], k=5) == 0.0
    assert section_recall_at_k([8, 9], GOLD_SECTIONS, k=2) == 0.0


# --- mrr --------------------------------------------------------------------
def test_mrr_is_reciprocal_rank_of_first_relevant_chunk():
    assert mrr([1, 9, 9], GOLD_IDS) == 1.0          # hit at rank 1
    assert mrr([9, 9, 7], GOLD_IDS) == pytest.approx(1 / 3)  # first hit at rank 3


def test_mrr_ignores_later_hits():
    # Only the FIRST relevant position matters — extra hits don't add anything.
    assert mrr([9, 2, 7, 1], GOLD_IDS) == 0.5


def test_mrr_is_zero_when_nothing_relevant_was_retrieved():
    assert mrr([8, 9], GOLD_IDS) == 0.0
    assert mrr([], GOLD_IDS) == 0.0


# --- ndcg_at_k ----------------------------------------------------------------
def test_ndcg_is_one_for_a_perfect_ranking():
    # All 3 relevant chunks first, in the top k → nothing could rank better.
    assert ndcg_at_k([1, 2, 7, 9], GOLD_IDS, k=4) == pytest.approx(1.0)


def test_ndcg_penalizes_late_hits():
    # One relevant chunk at rank 3 vs. the ideal (rank 1):
    # DCG = 1/log2(4), IDCG = 1/log2(2) → ratio = 0.5.
    got = ndcg_at_k([8, 9, 7], {7}, k=3)
    assert got == pytest.approx((1 / math.log2(4)) / (1 / math.log2(2)))
    assert got == pytest.approx(0.5)


def test_ndcg_ideal_is_capped_by_total_relevant():
    # Only 1 relevant chunk exists in the corpus; finding it at rank 1 is
    # perfect even though k=6 leaves five "non-relevant" slots below it.
    assert ndcg_at_k([7, 8, 9], {7}, k=6, total_relevant=1) == pytest.approx(1.0)


def test_ndcg_is_zero_with_no_relevant_or_no_hits():
    assert ndcg_at_k([8, 9], GOLD_IDS, k=2) == 0.0
    assert ndcg_at_k([1, 2], set(), k=2) == 0.0
