"""Guard model calls from oversized ReAct contexts near the context limit."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from hashlib import sha256
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from src.observability import log_event
from src.observability.model_trace import model_request_trace_chars
from src.prompt import (
    CONTEXT_LEDGER_TOOL_NAME,
    build_context_ledger_tool_call_args,
    build_context_ledger_tool_observation,
)
from src.guardrails.layered_context import (
    CompactLayeredContextState,
    build_layered_context_state,
    has_compressible_history,
    partition_messages_for_compaction,
)
from src.runtime import Context


DEFAULT_MAX_FRACTION = 0.85
DEFAULT_CHARS_PER_TOKEN = 4
DEFAULT_LEDGER_FRACTION = 0.25
DEFAULT_MIN_LEDGER_BUDGET_CHARS = 12000
DEFAULT_RAW_RECENT_TURNS = 2


class ContextBudgetGuard(AgentMiddleware):
    """Compact old ReAct context while preserving working state."""

    tools: list[Any] = []

    def __init__(
        self,
        *,
        context_window_tokens: int,
        max_fraction: float = DEFAULT_MAX_FRACTION,
        chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
        ledger_fraction: float = DEFAULT_LEDGER_FRACTION,
        min_ledger_budget_chars: int = DEFAULT_MIN_LEDGER_BUDGET_CHARS,
        raw_recent_turns: int = DEFAULT_RAW_RECENT_TURNS,
        max_tool_facts: int | None = None,
    ) -> None:
        self.context_window_tokens = context_window_tokens
        self.max_fraction = max_fraction
        self.chars_per_token = chars_per_token
        self.ledger_fraction = ledger_fraction
        self.min_ledger_budget_chars = min_ledger_budget_chars
        self.raw_recent_turns = raw_recent_turns
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
        if estimate <= threshold or not has_compressible_history(request.messages):
            return request
        compressed_messages, raw_messages = partition_messages_for_compaction(
            request.messages,
            raw_recent_turns=self.raw_recent_turns,
        )
        if not compressed_messages:
            return request
        layered_state = build_layered_context_state(
            compressed_messages,
            budget_chars=max(
                round(threshold * self.ledger_fraction),
                self.min_ledger_budget_chars,
            ),
            preserve_latest_user_message=False,
        )

        latest_human_text = _latest_human_text(request.messages)
        ledger_messages = _synthetic_ledger_messages(
            latest_human_text=latest_human_text,
            ledger=layered_state,
            estimate_chars=estimate,
            threshold_chars=round(threshold),
        )
        recent_messages = raw_messages or [HumanMessage(content=latest_human_text)]
        compact_request = request.override(
            messages=[*ledger_messages, *recent_messages],
        )
        _log_context_budget_compacted(
            request,
            estimate_chars=estimate,
            threshold_chars=round(threshold),
            ledger=layered_state,
            compacted_message_count=len(compact_request.messages),
            raw_message_count=len(recent_messages),
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


def _has_human_message(messages: list[Any]) -> bool:
    return any(str(getattr(message, "type", "")) == "human" for message in messages)


def _synthetic_ledger_messages(
    *,
    latest_human_text: str,
    ledger: CompactLayeredContextState,
    estimate_chars: int,
    threshold_chars: int,
) -> list[Any]:
    """Represent compacted historical state as a protocol-valid tool observation."""
    tool_call_id = _synthetic_ledger_tool_call_id(ledger)
    tool_args = build_context_ledger_tool_call_args(
        original_user_message=latest_human_text,
        estimate_chars=estimate_chars,
        threshold_chars=threshold_chars,
    )
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": tool_call_id,
                    "name": CONTEXT_LEDGER_TOOL_NAME,
                    "args": tool_args,
                }
            ],
        ),
        ToolMessage(
            content=build_context_ledger_tool_observation(
                original_user_message=latest_human_text,
                ledger=ledger,
                estimate_chars=estimate_chars,
                threshold_chars=threshold_chars,
            ),
            name=CONTEXT_LEDGER_TOOL_NAME,
            tool_call_id=tool_call_id,
        ),
    ]


def _synthetic_ledger_tool_call_id(ledger: CompactLayeredContextState) -> str:
    digest = sha256(ledger.to_prompt_text().encode("utf-8")).hexdigest()[:16]
    return f"context_ledger_{digest}"


def _log_context_budget_compacted(
    request: ModelRequest,
    *,
    estimate_chars: int,
    threshold_chars: int,
    ledger: CompactLayeredContextState,
    compacted_message_count: int,
    raw_message_count: int,
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
        old_user_message_count=ledger.old_user_message_count,
        preserved_old_user_message_count=ledger.preserved_old_user_message_count,
        dropped_old_user_message_count=ledger.dropped_old_user_message_count,
        assistant_message_count=ledger.assistant_message_count,
        preserved_assistant_message_count=ledger.preserved_assistant_message_count,
        dropped_assistant_message_count=ledger.dropped_assistant_message_count,
        compacted_request_chars=len(ledger.to_prompt_text()),
        original_message_count=len(request.messages),
        compacted_message_count=compacted_message_count,
        raw_message_count=raw_message_count,
        original_tool_count=len(request.tools),
        compacted_tool_count=len(request.tools),
        compaction_mode=ledger.strategy,
        compacted_prompt_sha256=sha256(
            _latest_human_text(request.messages).encode("utf-8")
        ).hexdigest(),
    )


def _request_context(request: ModelRequest) -> Context | None:
    runtime = getattr(request, "runtime", None)
    context = getattr(runtime, "context", None)
    return context if isinstance(context, Context) else None
