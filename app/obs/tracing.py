"""Langfuse v4 tracing, gated on whether keys are configured.

"""
from __future__ import annotations

import os

from app.config import get_settings

_s = get_settings()
ENABLED = bool(_s.langfuse_public_key and _s.langfuse_secret_key)

if ENABLED:
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", _s.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", _s.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", _s.langfuse_host)

    from langfuse import get_client, observe  # noqa: F401

    _client = get_client()

    def client():
        return _client

    def flush() -> None:
        _client.flush()

else:
    def observe(func=None, **_kwargs):
        """No-op stand-in supporting both @observe and @observe(...) forms."""
        if callable(func):
            return func

        def deco(f):
            return f

        return deco

    def client():
        return None

    def flush() -> None:
        pass
