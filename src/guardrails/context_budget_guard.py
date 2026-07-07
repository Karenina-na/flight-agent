"""Guard model calls from continuing ReAct loops near the context limit."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from hashlib import sha256
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import HumanMessage, SystemMessage

from src.observability import log_event
from src.observability.model_trace import model_request_trace_chars
from src.prompt import (
    build_context_compaction_system_prompt,
    build_context_compaction_user_prompt,
)
from src.guardrails.tool_observation import (
    build_tool_observations,
    compact_tool_observations,
)
from src.runtime import Context


DEFAULT_MAX_FRACTION = 0.85
DEFAULT_CHARS_PER_TOKEN = 4
DEFAULT_LEDGER_FRACTION = 0.25
DEFAULT_MIN_LEDGER_BUDGET_CHARS = 12000


class ContextBudgetGuard(AgentMiddleware):
    """Compact large ReAct contexts into a final-answer request."""

    tools: list[Any] = []

    def __init__(
        self,
        *,
        context_window_tokens: int,
        max_fraction: float = DEFAULT_MAX_FRACTION,
        chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
        ledger_fraction: float = DEFAULT_LEDGER_FRACTION,
        min_ledger_budget_chars: int = DEFAULT_MIN_LEDGER_BUDGET_CHARS,
        max_tool_facts: int | None = None,
    ) -> None:
        self.context_window_tokens = context_window_tokens
        self.max_fraction = max_fraction
        self.chars_per_token = chars_per_token
        self.ledger_fraction = ledger_fraction
        self.min_ledger_budget_chars = min_ledger_budget_chars
        self.max_tool_facts = max_tool_facts

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Compact oversized ReAct requests before synchronous model calls."""
        return handler(self._guarded_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Compact oversized ReAct requests before asynchronous model calls."""
        return await handler(self._guarded_request(request))

    def _guarded_request(self, request: ModelRequest) -> ModelRequest:
        estimate = _request_size_estimate(request)
        threshold = self.context_window_tokens * self.chars_per_token * self.max_fraction
        observations = build_tool_observations(request.messages)
        if estimate <= threshold or not observations:
            return request
        ledger = compact_tool_observations(
            observations,
            budget_chars=max(
                round(threshold * self.ledger_fraction),
                self.min_ledger_budget_chars,
            ),
        )

        compact_request = request.override(
            system_message=SystemMessage(content=build_context_compaction_system_prompt()),
            messages=[
                HumanMessage(
                    content=build_context_compaction_user_prompt(
                        original_user_message=_latest_human_text(request.messages),
                        ledger=ledger,
                        estimate_chars=estimate,
                        threshold_chars=round(threshold),
                    )
                )
            ],
            tools=[],
            tool_choice="none",
        )
        _log_context_budget_compacted(
            request,
            estimate_chars=estimate,
            threshold_chars=round(threshold),
            ledger=ledger,
        )
        return compact_request


def build_context_budget_guard(
    *,
    context_window_tokens: int,
    max_fraction: float = DEFAULT_MAX_FRACTION,
) -> ContextBudgetGuard:
    """Build context budget guard middleware."""
    return ContextBudgetGuard(
        context_window_tokens=context_window_tokens,
        max_fraction=max_fraction,
    )


def _request_size_estimate(request: ModelRequest) -> int:
    return model_request_trace_chars(request)


def _latest_human_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if str(getattr(message, "type", "")) == "human":
            content = getattr(message, "content", "")
            return str(content)
    return ""


def _log_context_budget_compacted(
    request: ModelRequest,
    *,
    estimate_chars: int,
    threshold_chars: int,
    ledger: CompactObservationLedger,
) -> None:
    log_event(
        "react_context_budget_compacted",
        context=_request_context(request),
        redact=False,
        estimate_chars=estimate_chars,
        threshold_chars=threshold_chars,
        tool_fact_count=ledger.preserved_observation_count,
        observation_count=ledger.observation_count,
        preserved_observation_count=ledger.preserved_observation_count,
        dropped_observation_count=ledger.dropped_observation_count,
        preview_truncated_count=ledger.preview_truncated_count,
        compacted_request_chars=len(ledger.to_prompt_text()),
        original_message_count=len(request.messages),
        original_tool_count=len(request.tools),
        compacted_prompt_sha256=sha256(
            _latest_human_text(request.messages).encode("utf-8")
        ).hexdigest(),
    )


def _request_context(request: ModelRequest) -> Context | None:
    runtime = getattr(request, "runtime", None)
    context = getattr(runtime, "context", None)
    return context if isinstance(context, Context) else None
