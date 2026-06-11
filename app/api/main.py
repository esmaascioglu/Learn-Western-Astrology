"""HTTP API: grounded astrology Q&A over the knowledge base.

    uvicorn app.api.main:app --reload
"""
from __future__ import annotations

import json

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app import rag

app = FastAPI(title="Learn Western Astrology", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int | None = None
    top_n: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    ans = rag.answer(req.question, top_k=req.top_k, top_n=req.top_n)
    return {"answer": ans.text, "sources": ans.sources}


@app.post("/query/stream")
def query_stream(req: QueryRequest) -> EventSourceResponse:
    contexts, tokens = rag.stream_answer(req.question, top_k=req.top_k, top_n=req.top_n)
    sources = [
        {"n": i + 1, "citation": c.citation, "score": round(c.score, 4)}
        for i, c in enumerate(contexts)
    ]

    def events():
        yield {"event": "sources", "data": json.dumps(sources)}
        for tok in tokens:
            yield {"event": "token", "data": tok}
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(events())
