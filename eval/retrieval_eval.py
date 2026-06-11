"""Retrieval-quality test: does the pipeline retrieve ONLY relevant passages?

Unlike the Ragas run (which scores the generated answer), this isolates the
retriever (hybrid search, RRF order) and checks, per labeled case:

  1. Book routing      - top hits come from the expected book.
  2. Term coverage     - retrieved passages contain the expected concepts.
  3. OOS rejection     - for out-of-scope questions, the best RRF score stays
                         below --oos-threshold (the system stays unconfident).
  4. Precision@k       - (optional, --judge) an LLM labels each retrieved passage
                         relevant/not, the most direct "only relevant info" check.

For rank quality (Recall@k / MRR / nDCG against gold-labeled sections, per leg),
see eval/retrieval_metrics.py — that is the measuring stick for experiments;
this script is the coarse routing/scope smoke test.

Run:
  .venv/bin/python eval/retrieval_eval.py
  .venv/bin/python eval/retrieval_eval.py --top-n 5 --judge
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app.config import PROJECT_ROOT
from app.ingestion.books import BOOKS_BY_SLUG
from app.retrieval.search import Retrieved, retrieve

CASES = Path(__file__).parent / "retrieval_cases.jsonl"
console = Console()


def _expected_title(slug: str | None) -> str | None:
    return BOOKS_BY_SLUG[slug].title if slug else None


def _judge_relevance(question: str, results: list[Retrieved]) -> list[bool]:
    """Ask the LLM which retrieved passages are relevant. Returns one bool each."""
    from app.llm import gateway

    listing = "\n\n".join(
        f"[{i + 1}] {r.content[:500]}" for i, r in enumerate(results)
    )
    prompt = (
        "You are judging retrieval quality. For the QUESTION below, decide which "
        "PASSAGES are relevant to answering it (contain on-topic information).\n\n"
        f"QUESTION: {question}\n\nPASSAGES:\n{listing}\n\n"
        "Reply ONLY with a JSON array of the relevant passage numbers, e.g. [1,3]. "
        "If none are relevant, reply []."
    )
    raw = gateway.complete([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=100)
    match = re.search(r"\[[\d,\s]*\]", raw)
    relevant_idx = set(json.loads(match.group(0))) if match else set()
    return [(i + 1) in relevant_idx for i in range(len(results))]


def evaluate(top_n: int, oos_threshold: float, judge: bool) -> None:
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]

    table = Table(title="Retrieval quality", show_lines=True)
    table.add_column("question", max_width=42)
    table.add_column("scope")
    table.add_column("top book / score")
    table.add_column("book✓")
    table.add_column("terms✓")
    table.add_column("max\nscore")
    if judge:
        table.add_column("prec@k")
    table.add_column("PASS")

    agg = {"in_book_top1": [], "in_book_any": [], "in_terms": [], "oos_reject": [],
           "precision": [], "passed": []}
    csv_rows: list[dict] = []

    for case in cases:
        q = case["question"]
        in_scope = case["in_scope"]
        results = retrieve(q, top_n=top_n)
        scores = [r.score for r in results]
        max_score = max(scores) if scores else 0.0
        books = [r.book_title for r in results]
        joined = " ".join(r.content.lower() for r in results)

        exp_title = _expected_title(case.get("expected_book"))
        book_top1 = (exp_title is not None and books and books[0] == exp_title)
        book_any = (exp_title is not None and exp_title in books)
        terms = case.get("must_include_any", [])
        term_hit = (not terms) or any(t.lower() in joined for t in terms)

        # pass/fail logic
        if in_scope:
            book_ok = (exp_title is None) or book_any
            passed = bool(book_ok and term_hit)
            agg["in_terms"].append(term_hit)
            if exp_title is not None:
                agg["in_book_top1"].append(book_top1)
                agg["in_book_any"].append(book_any)
        else:
            passed = max_score < oos_threshold
            agg["oos_reject"].append(passed)

        prec = None
        if judge and results:
            labels = _judge_relevance(q, results)
            prec = sum(labels) / len(labels)
            if in_scope:
                agg["precision"].append(prec)

        agg["passed"].append(passed)
        csv_rows.append({"question": q, "in_scope": in_scope, "passed": passed,
                         "max_score": round(max_score, 4), "book_any": book_any,
                         "term_hit": term_hit})

        top_cell = f"{books[0].split(':')[0] if books else '—'}\n{max_score:.2f}"
        book_cell = "—" if exp_title is None else ("✓" if book_any else "✗")
        row = [
            q, "in" if in_scope else "OOS", top_cell, book_cell,
            "✓" if term_hit else "✗", f"{max_score:.2f}",
        ]
        if judge:
            row.append("—" if prec is None else f"{prec:.2f}")
        row.append("[green]PASS[/]" if passed else "[red]FAIL[/]")
        table.add_row(*row)

    console.print(table)

    def pct(xs):
        return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "n/a"

    console.rule("Summary")
    console.print(f"Book routing — top-1 correct : {pct(agg['in_book_top1'])}")
    console.print(f"Book routing — in top-k      : {pct(agg['in_book_any'])}")
    console.print(f"Term coverage (in-scope)     : {pct(agg['in_terms'])}")
    console.print(f"OOS rejection (<{oos_threshold})        : {pct(agg['oos_reject'])}")
    if judge and agg["precision"]:
        console.print(f"Mean precision@{top_n} (in-scope) : "
                      f"{sum(agg['precision']) / len(agg['precision']):.2f}")
    console.print(f"[bold]Overall pass rate            : {pct(agg['passed'])}[/]")

    out = PROJECT_ROOT / "artifacts" / "retrieval_eval.csv"
    out.parent.mkdir(exist_ok=True)
    import csv
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    console.print(f"\nSaved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=5, help="passages to retrieve per question")
    # RRF scores are tiny by construction: a chunk ranked #1 by ALL three legs
    # scores (1.0 + 0.7 + 1.0) / (50 + 1) ≈ 0.053 — that's the theoretical MAX.
    # The old default (0.5) could never be exceeded, so every OOS case passed
    # vacuously. 0.04 ≈ "ranked near the top of at least two legs": an OOS query
    # should not get that much agreement. NOTE: rank-based scores carry no
    # absolute relevance signal — real OOS refusal needs a calibrated measure
    # (e.g. dense cosine similarity); this check is a weak guardrail until then.
    ap.add_argument("--oos-threshold", type=float, default=0.04,
                    help="max RRF score an out-of-scope question may have and still pass")
    ap.add_argument("--judge", action="store_true",
                    help="LLM-judge precision@k (costs a few cents)")
    args = ap.parse_args()
    evaluate(args.top_n, args.oos_threshold, args.judge)
