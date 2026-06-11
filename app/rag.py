"""RAG orchestration: retrieve → build grounded prompt → generate cited answer."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from app import obs
from app.llm import gateway
from app.retrieval.search import Retrieved, retrieve

SYSTEM_PROMPT = (
    "You are a knowledgeable, careful Western-astrology tutor. "
    "Answer the user's question using ONLY the numbered SOURCES provided. "
    "Cite the sources you use inline with bracketed numbers like [1], [2]. "
    "If the sources do not contain enough information to answer, say so plainly "
    "instead of inventing facts. Be clear and educational, and prefer the wording "
    "and concepts found in the sources. Do not make predictions or fortune-telling "
    "claims; explain what the texts say."
)


@dataclass
class Answer:
    text: str
    contexts: list[Retrieved]

    @property
    def sources(self) -> list[dict]:
        return [
            {"n": i + 1, "citation": c.citation, "chunk_id": c.chunk_id,
             "score": round(c.score, 4)}
            for i, c in enumerate(self.contexts)
        ]


def _format_sources(contexts: list[Retrieved]) -> str:
    blocks = []
    for i, c in enumerate(contexts):
        blocks.append(f"[{i + 1}] {c.citation}\n{c.content}")
    return "\n\n".join(blocks)


def _messages(question: str, contexts: list[Retrieved]) -> list[dict]:
    user = (
        f"QUESTION:\n{question}\n\n"
        f"SOURCES:\n{_format_sources(contexts)}\n\n"
        "Answer the question using only these sources, with inline [n] citations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_NO_CONTEXT = (
    "I couldn't find anything in the knowledge base relevant to that question. "
    "The current corpus focuses on natal technique and planetary transits."
)


@obs.observe(name="rag-answer")
def answer(question: str, **retrieve_kwargs) -> Answer:
    if obs.ENABLED:
        obs.client().set_current_trace_io(input=question)
    contexts = retrieve(question, **retrieve_kwargs)
    if not contexts:
        return Answer(text=_NO_CONTEXT, contexts=[])
    text = gateway.complete(_messages(question, contexts))
    if obs.ENABLED:
        obs.client().set_current_trace_io(output=text)
    return Answer(text=text, contexts=contexts)


def stream_answer(question: str, **retrieve_kwargs) -> tuple[list[Retrieved], Iterator[str]]:
    """Returns (contexts, token_iterator). Contexts are known before generation."""
    contexts = retrieve(question, **retrieve_kwargs)
    if not contexts:
        return [], iter([_NO_CONTEXT])
    return contexts, gateway.stream(_messages(question, contexts))
