"""Guard against repeated identical ReAct tool calls in one turn."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain.messages import ToolMessage

from src.observability import log_event
from src.runtime import Context


DEFAULT_LOOP_STOP_AFTER = 3


@dataclass
class _ScopeState:
    """In-memory duplicate-call state for one request scope."""

    seen_keys: set[str] = field(default_factory=set)
    duplicate_count: int = 0


class ParamAwareDuplicateToolCallGuard(AgentMiddleware):
    """Block repeated exact tool+args calls while allowing varied arguments."""

    tools: list[Any] = []

    def __init__(self, *, loop_stop_after: int = DEFAULT_LOOP_STOP_AFTER) -> None:
        self.loop_stop_after = loop_stop_after
        self._state_by_scope: dict[str, _ScopeState] = {}

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Run the first unique tool call and block exact duplicates."""
        duplicate_state = self._record_or_detect_duplicate(request)
        if duplicate_state:
            return self._duplicate_response(request, duplicate_state)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        """Run the first unique async tool call and block exact duplicates."""
        duplicate_state = self._record_or_detect_duplicate(request)
        if duplicate_state:
            return self._duplicate_response(request, duplicate_state)
        return await handler(request)

    def _record_or_detect_duplicate(self, request: ToolCallRequest) -> _ScopeState | None:
        scope = _request_scope(request)
        key = _tool_call_key(request)
        state = self._state_by_scope.setdefault(scope, _ScopeState())
        if key in state.seen_keys:
            state.duplicate_count += 1
            _log_duplicate_blocked(
                request,
                scope=scope,
                key=key,
                state=state,
                loop_stop_after=self.loop_stop_after,
            )
            return state
        state.seen_keys.add(key)
        return None

    def _duplicate_response(
        self,
        request: ToolCallRequest,
        state: _ScopeState,
    ) -> ToolMessage:
        stop_requested = state.duplicate_count >= self.loop_stop_after
        payload = {
            "status": "react_loop_stop_requested" if stop_requested else "duplicate_blocked",
            "message": _duplicate_message(stop_requested=stop_requested),
            "tool_name": _tool_name(request),
            "argument_keys": _argument_keys(request),
            "duplicate_count": state.duplicate_count,
            "loop_stop_after": self.loop_stop_after,
        }
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            name=_tool_name(request),
            tool_call_id=_tool_call_id(request) or "",
            status="success",
        )


def build_param_aware_duplicate_tool_call_guard(
    *,
    loop_stop_after: int = DEFAULT_LOOP_STOP_AFTER,
) -> ParamAwareDuplicateToolCallGuard:
    """Build parameter-aware duplicate tool-call guard middleware."""
    return ParamAwareDuplicateToolCallGuard(loop_stop_after=loop_stop_after)


def _duplicate_message(*, stop_requested: bool) -> str:
    if stop_requested:
        return (
            "This run has repeated already-executed tool calls several times. "
            "Stop calling tools and produce a concise interim answer from the "
            "existing observations and todos."
        )
    return (
        "This exact tool call has already been executed in this turn. "
        "Use the previous tool result and continue from the next distinct todo "
        "or produce an answer instead of calling the same tool again."
    )


def _request_scope(request: ToolCallRequest) -> str:
    context = _request_context(request)
    if context and context.request_id:
        return str(context.request_id)
    if context and context.thread_id:
        return str(context.thread_id)
    return "unknown-request"


def _tool_call_key(request: ToolCallRequest) -> str:
    payload = {
        "tool_name": _tool_name(request),
        "args": _canonical_json(request.tool_call.get("args")),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False, sort_keys=True)


def _request_context(request: ToolCallRequest) -> Context | None:
    runtime = getattr(request, "runtime", None)
    context = getattr(runtime, "context", None)
    return context if isinstance(context, Context) else None


def _tool_name(request: ToolCallRequest) -> str:
    if request.tool is not None:
        return str(request.tool.name)
    return str(request.tool_call.get("name") or "unknown_tool")


def _tool_call_id(request: ToolCallRequest) -> str | None:
    tool_call_id = request.tool_call.get("id")
    if tool_call_id:
        return str(tool_call_id)
    runtime_tool_call_id = getattr(request.runtime, "tool_call_id", None)
    return str(runtime_tool_call_id) if runtime_tool_call_id else None


def _argument_keys(request: ToolCallRequest) -> list[str]:
    args = request.tool_call.get("args")
    if not isinstance(args, dict):
        return []
    return sorted(str(key) for key in args)


def _log_duplicate_blocked(
    request: ToolCallRequest,
    *,
    scope: str,
    key: str,
    state: _ScopeState,
    loop_stop_after: int,
) -> None:
    log_event(
        "react_duplicate_tool_call_blocked",
        context=_request_context(request),
        redact=False,
        scope=scope,
        duplicate_key=key,
        duplicate_count=state.duplicate_count,
        loop_stop_after=loop_stop_after,
        stop_requested=state.duplicate_count >= loop_stop_after,
        tool_call_id=_tool_call_id(request),
        tool_name=_tool_name(request),
        argument_keys=_argument_keys(request),
        tool_call={
            "id": _tool_call_id(request) or "",
            "name": _tool_name(request),
            "args": request.tool_call.get("args"),
            "argument_keys": _argument_keys(request),
        },
    )


__all__ = [
    "ParamAwareDuplicateToolCallGuard",
    "build_param_aware_duplicate_tool_call_guard",
]
