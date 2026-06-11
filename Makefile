# Developer shortcuts. Override the interpreter with `make PY=python test` etc.
PY  ?= .venv/bin/python
PIP ?= $(PY) -m pip

.PHONY: setup lint fmt test ingest serve eval metrics ragas clean

setup:  ## Create a venv and install the package with dev + eval extras
	python -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -e ".[dev,eval]"

lint:  ## Static checks (ruff)
	$(PY) -m ruff check app eval scripts tests

fmt:  ## Auto-fix lint issues
	$(PY) -m ruff check --fix app eval scripts tests

test:  ## Run the unit test suite
	$(PY) -m pytest

ingest:  ## Chunk -> embed (dense + SPLADE) -> upsert to Supabase
	$(PY) -m app.cli ingest

serve:  ## Run the HTTP API with autoreload
	$(PY) -m uvicorn app.api.main:app --reload

eval:  ## Retrieval smoke test: book routing + out-of-scope check (no LLM judge)
	$(PY) eval/retrieval_eval.py

metrics:  ## Gold-labeled rank metrics: Recall@k / MRR / nDCG, fused + per leg
	$(PY) eval/retrieval_metrics.py

ragas:  ## Reference-free answer-quality eval (LLM judge, costs tokens)
	$(PY) eval/run_ragas.py

clean:  ## Remove caches and build artifacts
	rm -rf .ruff_cache .pytest_cache **/__pycache__ *.egg-info
