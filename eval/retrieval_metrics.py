"""Gold-labeled rank metrics for the retriever: Recall@k, MRR, nDCG — per leg + fused.

This is the measuring stick for retrieval experiments. Unlike retrieval_eval.py
(book routing / term presence — coarse pass/fail proxies), this scores the actual
RANKING against hand-labeled gold sections, so a change that drops the right chunk
from #1 to #5 is visible here and invisible there.

Gold labels live in eval/retrieval_gold.jsonl as (book, section) pairs — section
names are stable across re-ingests, chunk ids are not. Each gold section resolves
to its chunk ids at run time; if a section no longer exists the script fails
loudly (the labels drifted from the corpus and must be re-checked, not ignored).

Each question is scored four ways through the SAME production SQL (hybrid_search):
fused (production weights), and each leg alone by zeroing the other legs' RRF
weights. That shows which leg earns its place and which one drags.

Run:
  .venv/bin/python eval/retrieval_metrics.py
  .venv/bin/python eval/retrieval_metrics.py --depth 20 --ndcg-k 6
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app.config import PROJECT_ROOT
from app.ingestion.embed import embed_query
from app.ingestion.store import connect
from app.retrieval.metrics import mrr, ndcg_at_k, section_recall_at_k
from app.retrieval.query import focus_query
from app.retrieval.splade import encode_query, to_pgvector

CASES = Path(__file__).parent / "retrieval_gold.jsonl"
RECALL_KS = (1, 3, 6, 10)  # 6 = production top_n (what the LLM actually sees)

# Each leg in isolation = production fusion with the other legs' weights zeroed.
# Order: (full_text_weight, semantic_weight, sparse_weight).
LEGS: dict[str, tuple[float, float, float]] = {
    "fused": (0.7, 1.0, 1.0),  # production weights (db/schema.sql defaults)
    "dense": (0.0, 1.0, 0.0),
    "fts": (1.0, 0.0, 0.0),
    "splade": (0.0, 0.0, 1.0),
}

console = Console()


def _resolve_gold(conn, cases: list[dict]) -> None:
    """Turn each case's (book, section) labels into chunk-id sets, in place.

    Adds to each case:  gold_sections = [set(ids), ...]  and  gold_ids = union.
    """
    for case in cases:
        sections: list[set[int]] = []
        for label in case["gold"]:
            ids = {
                r[0] for r in conn.execute(
                    "select c.id from chunks c"
                    " join documents d on d.id = c.document_id"
                    " where d.slug = %s and c.metadata->>'section' = %s",
                    (label["book"], label["section"]),
                ).fetchall()
            }
            if not ids:
                sys.exit(
                    f"Gold label not found in DB: {label['book']} :: "
                    f"{label['section']!r} — re-check eval/retrieval_gold.jsonl "
                    "against the current chunking."
                )
            sections.append(ids)
        case["gold_sections"] = sections
        case["gold_ids"] = set().union(*sections)


def _rank_legs(conn, question: str, depth: int) -> dict[str, list[int]]:
    """Return the ranked chunk-id list per leg, reusing one embed + one SPLADE
    encode across all four hybrid_search calls (the expensive parts)."""
    focused = focus_query(question)
    emb = "[" + ",".join(f"{x:.8f}" for x in embed_query(question)) + "]"
    sparse = to_pgvector(encode_query(focused))

    ranked: dict[str, list[int]] = {}
    for leg, (ftw, semw, spw) in LEGS.items():
        rows = conn.execute(
            "select id, score from hybrid_search(%s, %s::vector, %s::sparsevec,"
            " %s::int, full_text_weight => %s, semantic_weight => %s,"
            " sparse_weight => %s)",
            (focused, emb, sparse, depth, ftw, semw, spw),
        ).fetchall()
        # With a single live leg, chunks found only by the zeroed legs score 0 —
        # they are not part of this leg's ranking, so drop them.
        ranked[leg] = [r[0] for r in rows if r[1] > 0]
    return ranked


def _score(case: dict, ranked: list[int], ndcg_k: int) -> dict[str, float]:
    """All metrics for one (case, leg) ranking."""
    out = {
        f"recall@{k}": section_recall_at_k(ranked, case["gold_sections"], k)
        for k in RECALL_KS
    }
    out["mrr"] = mrr(ranked, case["gold_ids"])
    out[f"ndcg@{ndcg_k}"] = ndcg_at_k(
        ranked, case["gold_ids"], ndcg_k, total_relevant=len(case["gold_ids"])
    )
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(depth: int, ndcg_k: int) -> None:
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    metric_names = [f"recall@{k}" for k in RECALL_KS] + ["mrr", f"ndcg@{ndcg_k}"]

    # rows[leg][metric] = list of per-case values;  fused_by_class for breakdown
    rows: dict[str, dict[str, list[float]]] = {
        leg: {m: [] for m in metric_names} for leg in LEGS
    }
    by_class: dict[str, dict[str, list[float]]] = {}
    csv_rows: list[dict] = []

    with connect() as conn:
        _resolve_gold(conn, cases)
        for i, case in enumerate(cases, 1):
            console.print(f"[dim]{i:>2}/{len(cases)}  {case['question']}[/]")
            ranked = _rank_legs(conn, case["question"], depth)
            for leg in LEGS:
                scores = _score(case, ranked[leg], ndcg_k)
                for m, v in scores.items():
                    rows[leg][m].append(v)
                csv_rows.append(
                    {"question": case["question"], "class": case["class"],
                     "leg": leg, **{m: round(v, 4) for m, v in scores.items()}}
                )
                if leg == "fused":
                    cls = by_class.setdefault(
                        case["class"], {m: [] for m in metric_names}
                    )
                    for m, v in scores.items():
                        cls[m].append(v)

    # ---- summary: per leg ------------------------------------------------
    table = Table(title=f"Rank metrics — {len(cases)} gold-labeled questions")
    table.add_column("leg")
    for m in metric_names:
        table.add_column(m, justify="right")
    for leg in LEGS:
        style = "bold" if leg == "fused" else ""
        table.add_row(
            leg, *(f"{_mean(rows[leg][m]):.3f}" for m in metric_names), style=style
        )
    console.print(table)

    # ---- breakdown: fused, by query class --------------------------------
    cls_table = Table(title="Fused ranking by query class")
    cls_table.add_column("class")
    cls_table.add_column("n", justify="right")
    for m in metric_names:
        cls_table.add_column(m, justify="right")
    for cls, vals in sorted(by_class.items()):
        n = len(vals[metric_names[0]])
        cls_table.add_row(cls, str(n), *(f"{_mean(vals[m]):.3f}" for m in metric_names))
    console.print(cls_table)

    # ---- worst fused cases: where to look next ---------------------------
    fused_rows = [r for r in csv_rows if r["leg"] == "fused"]
    misses = sorted(fused_rows, key=lambda r: r["mrr"])[:5]
    console.rule("Lowest-MRR questions (fused)")
    for r in misses:
        # markup=False: rich would otherwise eat "[sun-sign]" as a style tag
        console.print(f"  mrr={r['mrr']:.3f}  recall@6={r['recall@6']:.2f}  "
                      f"[{r['class']}] {r['question']}", markup=False)

    out = PROJECT_ROOT / "artifacts" / "retrieval_metrics.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    console.print(f"\nSaved per-case scores to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=20,
                    help="how many candidates to rank per leg (match_count)")
    ap.add_argument("--ndcg-k", type=int, default=6,
                    help="nDCG cutoff; 6 = production top_n sent to the LLM")
    args = ap.parse_args()
    evaluate(args.depth, args.ndcg_k)
