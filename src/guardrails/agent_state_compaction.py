"""Agent-state compaction middleware for oversized ReAct histories."""

from __future__ import annotations

from typing import Any

from src.guardrails.context_budget_guard import (
    ContextBudgetGuard,
    DEFAULT_MAX_FRACTION,
)


class AgentStateCompactionMiddleware(ContextBudgetGuard):
    """Compact oversized ReAct state while preserving the active user goal."""


def build_agent_state_compaction_middleware(
    *,
    context_window_tokens: int,
    summary_model: Any | None = None,
    semantic_enabled: bool = True,
    max_fraction: float = DEFAULT_MAX_FRACTION,
) -> AgentStateCompactionMiddleware:
    """Build agent-state compaction middleware."""
    return AgentStateCompactionMiddleware(
        context_window_tokens=context_window_tokens,
        summary_model=summary_model,
        semantic_enabled=semantic_enabled,
        max_fraction=max_fraction,
    )


__all__ = [
    "AgentStateCompactionMiddleware",
    "build_agent_state_compaction_middleware",
]
