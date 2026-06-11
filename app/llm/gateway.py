"""Thin wrapper over LiteLLM so the rest of the app is provider-agnostic.

Swap providers by changing LLM_MODEL in .env (e.g. groq/llama-3.3-70b-versatile,
anthropic/claude-sonnet-4-6). Provider API keys are read from the environment.
"""
from __future__ import annotations

import os
from collections.abc import Iterator

import litellm

from app import obs
from app.config import get_settings

litellm.drop_params = True  # ignore params a given provider doesn't support


def _log_usage(resp) -> None:
    if not obs.ENABLED:
        return
    u = getattr(resp, "usage", None)
    if u is None:
        return
    obs.client().update_current_generation(
        model=get_settings().llm_model,
        usage_details={
            "input": u.prompt_tokens,
            "output": u.completion_tokens,
            "total": u.total_tokens,
        },
    )


# Map our settings fields → the env var names LiteLLM expects per provider.
_PROVIDER_KEYS = {
    "groq_api_key": "GROQ_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "cerebras_api_key": "CEREBRAS_API_KEY",
    "gemini_api_key": "GEMINI_API_KEY",
    "together_api_key": "TOGETHERAI_API_KEY",
}


def _ensure_keys() -> None:
    s = get_settings()
    for field, env_name in _PROVIDER_KEYS.items():
        value = getattr(s, field, "")
        if value:
            os.environ.setdefault(env_name, value)


@obs.observe(as_type="generation", name="llm-generate")
def complete(messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024) -> str:
    _ensure_keys()
    resp = litellm.completion(
        model=get_settings().llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    _log_usage(resp)
    return resp.choices[0].message.content or ""


@obs.observe(as_type="generation", name="llm-generate-stream")
def stream(messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024) -> Iterator[str]:
    _ensure_keys()
    for chunk in litellm.completion(
        model=get_settings().llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    ):
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
