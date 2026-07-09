"""Guard model calls from oversized ReAct contexts near the context limit."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from hashlib import sha256
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from src.observability import log_event
from src.observability.model_trace import model_request_trace_chars
from src.runtime import Context
from src.summarization.context_compaction import (
    ContextCompactionResult,
    build_context_compaction_request,
)
from src.summarization.layered_context import has_compressible_history


DEFAULT_MAX_FRACTION = 0.85
DEFAULT_CHARS_PER_TOKEN = 4
DEFAULT_LEDGER_FRACTION = 0.25
DEFAULT_MIN_LEDGER_BUDGET_CHARS = 12000
DEFAULT_RAW_RECENT_TURNS = 2
DEFAULT_COMPACTED_STATE_PREVIEW_CHARS = 4000


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
        return self._compaction_pipeline_request(
            request,
            estimate_chars=estimate,
            threshold_chars=round(threshold),
        )

    def _compaction_pipeline_request(
        self,
        request: ModelRequest,
        *,
        estimate_chars: int,
        threshold_chars: int,
    ) -> ModelRequest:
        """Build a transient compacted model request after budget pressure triggers."""
        compaction_result = build_context_compaction_request(
            request,
            latest_human_text=_current_user_goal(request),
            estimate_chars=estimate_chars,
            threshold_chars=threshold_chars,
            ledger_fraction=self.ledger_fraction,
            min_ledger_budget_chars=self.min_ledger_budget_chars,
            raw_recent_turns=self.raw_recent_turns,
        )
        if compaction_result is None:
            return request
        _log_context_budget_compacted(
            request,
            estimate_chars=estimate_chars,
            threshold_chars=threshold_chars,
            compaction_result=compaction_result,
        )
        return compaction_result.request


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


def _current_user_goal(request: ModelRequest) -> str:
    context = _request_context(request)
    if context is not None and context.current_user_input:
        return context.current_user_input
    return _latest_human_text(request.messages)


def _preview_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    return text if len(text) <= limit else f"{text[:limit]}..."


def _log_context_budget_compacted(
    request: ModelRequest,
    *,
    estimate_chars: int,
    threshold_chars: int,
    compaction_result: ContextCompactionResult,
) -> None:
    ledger = compaction_result.ledger
    projection = compaction_result.layer_one_projection
    compacted_state_text = ledger.to_prompt_text()
    compacted_state_preview = _preview_text(
        compacted_state_text,
        DEFAULT_COMPACTED_STATE_PREVIEW_CHARS,
    )
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
        compacted_request_chars=len(compacted_state_text),
        original_message_count=len(request.messages),
        compacted_message_count=len(compaction_result.request.messages),
        raw_message_count=compaction_result.raw_message_count,
        todo_snapshot_item_count=len(
            (compaction_result.todo_snapshot or {}).get("items", [])
        ),
        original_tool_count=len(request.tools),
        compacted_tool_count=len(request.tools),
        layer1_reasoning_block_removed_count=projection.reasoning_block_removed_count,
        layer1_tool_message_removed_count=projection.tool_message_removed_count,
        layer1_tool_call_removed_count=projection.tool_call_removed_count,
        layer1_adjacent_human_merged_count=projection.adjacent_human_merged_count,
        layer1_duplicate_tool_output_count=projection.duplicate_tool_output_count,
        layer1_empty_tool_output_count=projection.empty_tool_output_count,
        compaction_mode=ledger.strategy,
        compacted_state_preview=compacted_state_preview,
        compacted_state_preview_chars=len(compacted_state_preview),
        compacted_state_chars=len(compacted_state_text),
        compacted_state_sha256=sha256(compacted_state_text.encode("utf-8")).hexdigest(),
        compacted_prompt_sha256=sha256(
            _current_user_goal(request).encode("utf-8")
        ).hexdigest(),
    )


def _request_context(request: ModelRequest) -> Context | None:
    runtime = getattr(request, "runtime", None)
    context = getattr(runtime, "context", None)
    return context if isinstance(context, Context) else None
