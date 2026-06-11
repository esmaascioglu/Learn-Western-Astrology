"""Ranking metrics for retrieval evaluation — pure math, no I/O.

Used by eval/retrieval_metrics.py against the gold-labeled query set
(eval/retrieval_gold.jsonl). Gold labels are *sections* — stable (book, section)
pairs — because chunk ids are regenerated on every re-ingest. Each gold section
resolves to the set of chunk ids that belong to it, and a retrieved chunk counts
as relevant when it belongs to any gold section.

All functions take a ranked list of chunk ids (best first) and return a value
in [0, 1], where 1 is a perfect result.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def section_recall_at_k(
    ranked_ids: Sequence[int], gold_sections: Sequence[set[int]], k: int
) -> float:
    """Of the gold sections, what fraction shows up in the top k?

    A section "shows up" when at least one of its chunks is retrieved — we don't
    need every chunk of a long section, one is enough to ground the answer.
    """
    if not gold_sections:
        return 0.0
    top = set(ranked_ids[:k])
    found = sum(1 for chunk_ids in gold_sections if chunk_ids & top)
    return found / len(gold_sections)


def mrr(ranked_ids: Sequence[int], relevant_ids: set[int]) -> float:
    """Reciprocal rank of the FIRST relevant chunk (1 = top hit, 1/2 = second, …).

    Measures "how far does the user/LLM have to read before the right passage
    appears". 0 if no relevant chunk was retrieved at all.
    """
    for pos, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / pos
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[int],
    relevant_ids: set[int],
    k: int,
    total_relevant: int | None = None,
) -> float:
    """Normalized Discounted Cumulative Gain at k, with binary relevance.

    Rewards putting relevant chunks EARLY: a hit at rank 1 is worth more than a
    hit at rank 6 (gain is discounted by log2 of the position). The score is
    normalized by the best ordering possible, so 1.0 means "all the relevant
    chunks that could fit in the top k are there, ranked first".

    total_relevant: how many relevant chunks exist in the whole corpus (used to
    build the ideal ranking). Defaults to len(relevant_ids).
    """
    if not relevant_ids:
        return 0.0
    total = total_relevant if total_relevant is not None else len(relevant_ids)

    dcg = sum(
        1.0 / math.log2(pos + 1)
        for pos, chunk_id in enumerate(ranked_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    # Ideal ranking: every top position holds a relevant chunk, until we run out
    # of relevant chunks or out of the top-k window.
    ideal = sum(1.0 / math.log2(pos + 1) for pos in range(1, min(k, total) + 1))
    return dcg / ideal if ideal > 0 else 0.0
