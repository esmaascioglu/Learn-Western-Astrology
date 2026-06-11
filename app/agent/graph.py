"""M1: the tutor agent, built with LangGraph's prebuilt ReAct agent.

`create_react_agent` wires up the standard agent loop as a small graph:

    ┌────────┐   tool calls    ┌──────┐
    │ agent  │ ───────────────►│ tools│
    │ (LLM)  │◄─────────────────┤      │
    └───┬────┘   tool results  └──────┘
        │ no tool calls
        ▼
       END

Each turn: the LLM sees the system prompt + conversation history and either
calls search_knowledge_base or returns a final answer. Tool results are appended
to the message history and fed back to the LLM. M2 will rebuild this same shape
as an explicit StateGraph so you can see what's happening under the hood.
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import search_knowledge_base
from app.config import get_settings


def _ensure_openai_key() -> None:
    """langchain-openai reads OPENAI_API_KEY from the environment; our settings
    load it from .env into a Settings object, not into os.environ directly."""
    s = get_settings()
    if s.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", s.openai_api_key)


def build_agent():
    """Build the agent graph. Each call gets its own MemorySaver, i.e. its own
    in-process conversation memory — fine for a single CLI session (M1).
    M3 swaps this for a Postgres checkpointer shared across requests/threads."""
    _ensure_openai_key()
    s = get_settings()
    llm = ChatOpenAI(model=s.agent_model, temperature=0.2)

    from langgraph.prebuilt import create_react_agent

    return create_react_agent(
        llm,
        tools=[search_knowledge_base],
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )


def get_callbacks() -> list:
    """Langfuse callback handler for the LangChain/LangGraph run, if configured.

    This is the ONE line that makes every node, tool call, and LLM call in the
    graph show up as a trace in Langfuse — wired from the first agent, not
    bolted on later, per the project's observability goal.
    """
    from app import obs

    if not obs.ENABLED:
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]
