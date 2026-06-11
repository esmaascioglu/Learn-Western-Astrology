"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Database (Supabase / pgvector) ---
    database_url: str = ""
    supabase_url: str = ""
    supabase_anon_key: str = ""

    # --- Embeddings (OpenAI) ---
    # 3-small supports Matryoshka dimension reduction → keep the vector(1024)
    # schema by requesting embedding_dim dimensions.
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1024

    # --- Sparse retrieval (SPLADE, local/torch) ---
    # Learned sparse lexical encoder: a DistilBERT MLM head projects each token
    # onto the vocabulary, log(1+relu) max-pooled into one sparse vector. Closes
    # the vocabulary gap (term expansion) that BM25 and dense both miss.
    splade_model: str = "naver/splade-v3-distilbert"
    splade_vocab_dim: int = 30522   # DistilBERT vocab → sparsevec(30522)
    splade_top_k: int = 256         # keep top-K terms per vector (storage/speed bound)
    splade_batch_size: int = 8      # doc-encode batch (memory-bounded one-time pass)
    splade_max_length: int = 512    # token truncation for encoding

    # --- Generation LLM (via LiteLLM) ---
    llm_model: str = "gpt-4o-mini"
    # --- Agent (Phase 1, via langchain-openai — OpenAI only) ---
    agent_model: str = "gpt-4o-mini"
    # Judge model for Ragas eval (kept separate so it can differ from the app's).
    eval_llm_model: str = "gpt-4o-mini"
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    # Additional LiteLLM providers (open-model hosts with free tiers)
    openrouter_api_key: str = ""
    cerebras_api_key: str = ""
    gemini_api_key: str = ""
    together_api_key: str = ""

    # --- Observability ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Retrieval ---
    retrieval_top_k: int = 20  # candidates shown by the inspector
    top_n: int = 6             # passages sent to the LLM
    # Minimum hybrid (RRF) score to keep a chunk. RRF scores are small and NOT
    # 0–1 calibrated, so this is 0 (off) for now; out-of-scope refusal will be
    # reworked once the keyword/full-text leg is improved.
    relevance_floor: float = 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
