"""Reference-free RAG evaluation with Ragas.

Runs the real pipeline (retrieve → generate) on a small question set,
then scores each answer on:
  - Faithfulness            : is the answer grounded in the retrieved passages?
  - ResponseRelevancy       : does the answer address the question?
  - ContextPrecision        : are the retrieved passages relevant/ranked well?

The judge LLM is EVAL_LLM_MODEL (via ChatLiteLLM, provider-agnostic) and the eval
embeddings reuse the app's OpenAI embedder. Faithfulness matters most here: it
catches whether the generator drifted from the retrieved sources.

Run:  .venv/bin/python eval/run_ragas.py
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

# Silence import-time DeprecationWarnings from the langchain/ragas 0.x stack
# before those modules are imported below (hence the deliberate import order).
warnings.filterwarnings("ignore", category=DeprecationWarning)

from app import rag  # noqa: E402
from app.config import PROJECT_ROOT, get_settings  # noqa: E402

DATASET = Path(__file__).parent / "dataset.jsonl"


def main() -> None:
    s = get_settings()
    # The default judge (EVAL_LLM_MODEL=gpt-4o-mini) and the eval embeddings both
    # call OpenAI. Point a different EVAL_LLM_MODEL at another provider if desired.
    if not s.openai_api_key:
        raise SystemExit("OPENAI_API_KEY missing in .env — needed for the judge + eval embeddings.")
    os.environ.setdefault("OPENAI_API_KEY", s.openai_api_key)

    from langchain_community.chat_models import ChatLiteLLM
    from langchain_core.embeddings import Embeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    from app.ingestion.embed import embed_documents, embed_query

    class AppEmbeddings(Embeddings):
        """Adapter so Ragas reuses the exact embedder the pipeline uses."""

        def embed_documents(self, texts):
            return embed_documents(list(texts))

        def embed_query(self, text):
            return embed_query(text)

    judge = LangchainLLMWrapper(ChatLiteLLM(model=s.eval_llm_model, temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(AppEmbeddings())

    lines = [line for line in DATASET.read_text().splitlines() if line.strip()]
    questions = [json.loads(line)["question"] for line in lines]

    samples = []
    for q in questions:
        print(f"  answering: {q}")
        a = rag.answer(q, top_n=4)  # fewer contexts = fewer judge tokens
        samples.append(
            SingleTurnSample(
                user_input=q,
                response=a.text,
                retrieved_contexts=[c.content for c in a.contexts],
            )
        )

    dataset = EvaluationDataset(samples=samples)
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()]

    # Keep concurrency low to stay comfortably under provider rate limits.
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=2, timeout=180),
    )

    print("\n=== Ragas scores (mean) ===")
    print(result)

    out = PROJECT_ROOT / "artifacts" / "eval_results.csv"
    out.parent.mkdir(exist_ok=True)
    result.to_pandas().to_csv(out, index=False)
    print(f"\nPer-question results saved to {out}")


if __name__ == "__main__":
    main()
