# Learn Western Astrology 🔭

An open-source assistant that **teaches Western (tropical) astrology** and answers
questions grounded in cited source books — natal technique, planetary transits, and
chart interpretation — with citations back to the page.

> Educational tool. Answers are grounded in cited texts, not fortune-telling.

## Why two grounding layers

Astrology answers split into two kinds, and the architecture keeps them separate:

| Layer | Example question | Source of truth | Mechanism |
|-------|------------------|-----------------|-----------|
| **Computational** | "When is my Saturn return?" | Deterministic ephemeris | Tools / MCP *(Phase 1)* |
| **Interpretive** | "What does a Saturn return *mean*?" | The books | **RAG** *(Phase 0)* |

This stops the LLM from inventing planetary positions (it's bad at that) while letting
it synthesize and teach from sourced material (it's good at that).

## Retrieval design

Interpretive answers come from a **three-leg hybrid retriever**, fused with Reciprocal
Rank Fusion (RRF) in a single Postgres function — no cross-encoder reranker:

| Leg | What it catches | Implementation |
|-----|-----------------|----------------|
| **Dense** | Paraphrase / semantic similarity | OpenAI `text-embedding-3-small` (1024-d), pgvector cosine |
| **Full-text** | Exact terminology, heading matches | Postgres FTS, heading-weighted + OR-ified `tsquery` |
| **SPLADE** | Vocabulary gap (term expansion) | `naver/splade-v3-distilbert`, learned-sparse `sparsevec(30522)` |

Two design notes that materially improved results:

- **Query focusing** ([`app/retrieval/query.py`](app/retrieval/query.py)) strips generic
  *framing* words ("which / make / natal / chart / combination") from the lexical legs
  so the discriminating term drives ranking. A question like *"which combinations in a
  natal chart make you stubborn?"* reduces to `stubborn`, surfacing Saturn-in-Taurus and
  the Taurus trait sections instead of unrelated long passages.
- **Structure-aware chunking** ([`app/ingestion/chunk.py`](app/ingestion/chunk.py)): each
  book is split on its own natural units (transit headings, table-of-contents, native
  font hierarchy) and tagged with a typed `indicator_type` / `context` metadata schema.

## Architecture

```
CLI (astro chat) ─► LangGraph agent (gpt-4o-mini, ReAct loop)
                          │
                          └─► search_knowledge_base ─► RAG retrieval ─► Supabase pgvector
                                                          dense + FTS + SPLADE, fused by RRF
        every node/tool/LLM call traced in Langfuse · answer quality measured with Ragas
```

A future Phase 2 may add a computational layer (ephemeris/chart-data tools), but Phase 1
deliberately keeps the agent to a **single tool**: it's a tutor that teaches *concepts* from
the books, not a personalized chart-reading service (which would require user data + auth).

## Stack

| Concern | Choice |
|---------|--------|
| Backend | Python 3.10–3.12 · FastAPI (SSE streaming) |
| PDF extraction | PyMuPDF |
| Dense embeddings | **OpenAI** `text-embedding-3-small` (1024-d via Matryoshka) |
| Sparse retrieval | **SPLADE** `naver/splade-v3-distilbert` (local, torch/CPU) |
| Vector DB | **Supabase** Postgres + pgvector (`vector` + `sparsevec`) + FTS |
| Generation (RAG eval) | **`gpt-4o-mini`|
| Agent (Phase 1) | **LangGraph** `create_react_agent` + **`langchain-openai`** (`gpt-4o-mini`) |
| Observability | **Langfuse** (optional) + **Ragas** eval |

## Status

- [x] Architecture & scope
- [x] Project scaffold + DB schema
- [x] Phase 0 — ingestion pipeline (parse → chunk → embed → upsert) · 1,960 chunks / 3 books for now
- [x] Phase 0 — three-leg hybrid retrieval (dense + FTS + SPLADE) + query focusing
- [x] Phase 0 — `/query` (+ SSE) endpoint, grounded answers with citations
- [x] Phase 0 — Langfuse tracing + Ragas eval
- [x] Phase 0 — gold-labeled retrieval eval (Recall@k / MRR / nDCG, per leg + fused)
- [x] Phase 1 (M1) — LangGraph tutor agent (single RAG tool) + Langfuse tracing + `astro chat`
- [ ] Phase 1 (M2+) — hand-built `StateGraph`, Postgres-backed conversation memory, FastAPI agent endpoint
- [ ] Phase 2 — computational layer (ephemeris/chart tools) + user data + auth + web UI

## Setup

```bash
# 1. Environment
make setup            # python -m venv .venv && pip install -e ".[dev,eval]"
source .venv/bin/activate

# 2. Configure
cp .env.example .env  # set DATABASE_URL + OPENAI_API_KEY (LANGFUSE_* optional)

# 3. Apply the DB schema to your Supabase / Postgres (needs the pgvector extension)
psql "$DATABASE_URL" -f db/schema.sql

# 4. Add source PDFs (see "Source corpus" below) under Data/, then ingest
make ingest           # chunk → embed (OpenAI dense + local SPLADE) → upsert

# 5. Run
make serve            # uvicorn app.api.main:app --reload
```

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"What does Venus in Gemini mean in a natal chart?"}' | jq
```

## Tutor agent (Phase 1)

```bash
astro chat   # interactive CLI; the agent decides when to search the books, with citations
```

The agent is a [LangGraph](https://langchain-ai.github.io/langgraph/) `create_react_agent`
(`app/agent/graph.py`) with **one tool**, `search_knowledge_base`
([`app/agent/tools.py`](app/agent/tools.py)) — a thin wrapper around the Phase-0 hybrid
retriever. Conversation memory is in-process for now (`MemorySaver`, one thread per `chat`
session). If `LANGFUSE_*` keys are set, every node/tool/LLM call in the graph is traced
automatically via `langfuse.langchain.CallbackHandler`.

## Source corpus & attribution

The interpretive corpus is built from the books below. **None of the source PDFs are
redistributed with this repository** (`Data/` is git-ignored) — only short excerpts
(a few hundred tokens), retrieved on demand and always presented to the user with an
inline citation `[n]` plus a full "Sources" reference, are surfaced through the chat
agent. This is intended as fair-use, transformative, educational use (a study aid that
teaches *from* and *points back to* these works), not a substitute for owning them.

| Title | Author | Original publisher / year | Status |
|-------|--------|---------------------------|--------|
| *Planets in Transit: Life Cycles for Living* | Robert Hand | Whitford Press / Schiffer Publishing, 1976 (ISBN 0-914918-24-9) | © — in print, used under fair use (excerpts only) |
| *The Only Astrology Book You'll Ever Need* | Joanna Martine Woolfolk | Originally Stein and Day, 1982; current editions via Taylor Trade Publishing | © — in print, used under fair use (excerpts only) |
| *Astrology: Its Technics and Ethics* | C. Aq. Libra (pen name of Roelf Takens) | P. Dz. Veen, Amersfoort (Netherlands), 1917 | Public domain (pre-1929); full text on [archive.org](https://archive.org/details/astrologyitstech00libr) |

If you are a rights holder and have a concern about how an excerpt is used, please open an
issue and it will be addressed promptly.

To reproduce the knowledge base, place your own legally obtained PDFs under `Data/` and
register them in [`app/ingestion/books.py`](app/ingestion/books.py); chunking strategies
are per-document, so a new book needs a strategy mapping there.

## Development

```bash
make lint     # ruff
make test     # pytest (hermetic: no DB / network / keys)
make eval     # retrieval smoke test: book routing + out-of-scope rejection
make metrics  # gold-labeled rank metrics: Recall@k / MRR / nDCG, per leg + fused
```

### Measuring retrieval

Retrieval changes are judged against a **gold-labeled query set**
([`eval/retrieval_gold.jsonl`](eval/retrieval_gold.jsonl)): 35 questions across 7
query classes (transit aspects, natal placements, techniques, trait queries, …),
each labeled with the book section(s) that should be retrieved. Labels are
`(book, section)` pairs — stable across re-ingests, unlike chunk ids.

[`eval/retrieval_metrics.py`](eval/retrieval_metrics.py) scores every question
four ways through the same production SQL — the fused ranking plus each leg in
isolation (zeroing the other legs' RRF weights) — and reports **Recall@k, MRR,
and nDCG@6** (6 = the passages actually sent to the LLM), with a per-class
breakdown and the worst-ranked questions to investigate next. Per-case scores
land in `artifacts/retrieval_metrics.csv`.

Baseline (2026-06): fused MRR **0.86** / recall@6 **0.94**, beating every
individual leg — dense 0.84, SPLADE 0.76, FTS 0.51 — which is the hybrid design
working as intended. Known limitation kept visible on purpose: RRF scores are
rank-based and carry no absolute relevance signal, so out-of-scope rejection
(`make eval`) is weak until a calibrated relevance gate (e.g. dense cosine
floor) is added.

## License

MIT — see [LICENSE](LICENSE). Source book content remains under its respective copyright.
