"""Guard against repeated identical ReAct tool calls in one turn."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain.agents.middleware.types import hook_config
from langchain.messages import AIMessage, ToolMessage
from src.observability import log_event
from src.runtime import Context


DEFAULT_LOOP_STOP_AFTER = 3


@dataclass
class _ScopeState:
    """In-memory duplicate-call state for one request scope."""

    seen_keys: set[str] = field(default_factory=set)
    duplicate_counts: dict[str, int] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock, repr=False)


class ParamAwareDuplicateToolCallGuard(AgentMiddleware):
    """Block repeated exact tool+args calls while allowing varied arguments."""

    tools: list[Any] = []

    def __init__(self, *, loop_stop_after: int = DEFAULT_LOOP_STOP_AFTER) -> None:
        self.loop_stop_after = loop_stop_after
        self._state_by_scope: dict[str, _ScopeState] = {}

    @hook_config(can_jump_to=["end"])
    def after_model(
        self,
        state: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any] | None:
        """End before executing a duplicate call that reaches the stop threshold."""
        messages = state.get("messages")
        if not isinstance(messages, list) or not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return None

        context = _runtime_context(runtime)
        scope = _context_scope(context)
        duplicate_state = self._state_by_scope.setdefault(scope, _ScopeState())
        with duplicate_state.lock:
            for tool_call in last_message.tool_calls:
                key = _tool_call_payload_key(
                    str(tool_call.get("name") or "unknown_tool"),
                    tool_call.get("args"),
                )
                if key not in duplicate_state.seen_keys:
                    continue
                next_duplicate_count = duplicate_state.duplicate_counts.get(key, 0) + 1
                if next_duplicate_count < self.loop_stop_after:
                    continue

                duplicate_state.duplicate_counts[key] = next_duplicate_count
                tool_name = str(tool_call.get("name") or "unknown_tool")
                tool_call_id = str(tool_call.get("id") or "")
                argument_keys = _argument_keys_from_args(tool_call.get("args"))
                _log_duplicate_event(
                    context=context,
                    scope=scope,
                    key=key,
                    duplicate_count=next_duplicate_count,
                    loop_stop_after=self.loop_stop_after,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    argument_keys=argument_keys,
                    args=tool_call.get("args"),
                )
                stop_messages = [
                    _loop_stop_tool_message(
                        pending_call,
                        duplicate_count=next_duplicate_count,
                        loop_stop_after=self.loop_stop_after,
                    )
                    for pending_call in last_message.tool_calls
                ]
                return {
                    "messages": [
                        *stop_messages,
                        _loop_stop_ai_message(tool_name),
                    ],
                    "jump_to": "end",
                }
        return None

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self,
        state: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any] | None:
        """Async equivalent of the duplicate-call model gate."""
        return self.after_model(state, runtime)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Run the first unique tool call and block exact duplicates."""
        duplicate_count = self._record_or_detect_duplicate(request)
        if duplicate_count is not None:
            return self._duplicate_response(request, duplicate_count)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        """Run the first unique async tool call and block exact duplicates."""
        duplicate_count = self._record_or_detect_duplicate(request)
        if duplicate_count is not None:
            return self._duplicate_response(request, duplicate_count)
        return await handler(request)

    def _record_or_detect_duplicate(self, request: ToolCallRequest) -> int | None:
        scope = _request_scope(request)
        key = _tool_call_key(request)
        state = self._state_by_scope.setdefault(scope, _ScopeState())
        with state.lock:
            if key in state.seen_keys:
                duplicate_count = state.duplicate_counts.get(key, 0) + 1
                state.duplicate_counts[key] = duplicate_count
                _log_duplicate_blocked(
                    request,
                    scope=scope,
                    key=key,
                    duplicate_count=duplicate_count,
                    loop_stop_after=self.loop_stop_after,
                )
                return duplicate_count
            state.seen_keys.add(key)
            return None

    def _duplicate_response(
        self,
        request: ToolCallRequest,
        duplicate_count: int,
    ) -> ToolMessage:
        stop_requested = duplicate_count >= self.loop_stop_after
        payload = _duplicate_payload(
            tool_name=_tool_name(request),
            argument_keys=_argument_keys(request),
            duplicate_count=duplicate_count,
            loop_stop_after=self.loop_stop_after,
            stop_requested=stop_requested,
        )
        tool_message = ToolMessage(
            content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            name=_tool_name(request),
            tool_call_id=_tool_call_id(request) or "",
            status="error",
        )
        return tool_message


def build_param_aware_duplicate_tool_call_guard(
    *,
    loop_stop_after: int = DEFAULT_LOOP_STOP_AFTER,
) -> ParamAwareDuplicateToolCallGuard:
    """Build parameter-aware duplicate tool-call guard middleware."""
    return ParamAwareDuplicateToolCallGuard(loop_stop_after=loop_stop_after)


def _duplicate_message(*, stop_requested: bool) -> str:
    if stop_requested:
        return (
            "This run has repeated the same tool with the same arguments several "
            "times. Stop calling tools now and produce an answer from the "
            "existing observations, previous tool results, and todos."
        )
    return (
        "This exact same tool with the same arguments has already been executed "
        "in this turn. Do not call it again. Use the previous tool result and "
        "produce an answer, or continue only with a distinct todo or distinct "
        "tool arguments."
    )


def _request_scope(request: ToolCallRequest) -> str:
    return _context_scope(_request_context(request))


def _context_scope(context: Context | None) -> str:
    if context and context.request_id:
        return str(context.request_id)
    if context and context.thread_id:
        return str(context.thread_id)
    return "unknown-request"


def _tool_call_key(request: ToolCallRequest) -> str:
    return _tool_call_payload_key(
        _tool_name(request),
        request.tool_call.get("args"),
    )


def _tool_call_payload_key(tool_name: str, args: Any) -> str:
    payload = {
        "tool_name": tool_name,
        "args": _canonical_json(args),
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


def _runtime_context(runtime: Any) -> Context | None:
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
    return _argument_keys_from_args(request.tool_call.get("args"))


def _argument_keys_from_args(args: Any) -> list[str]:
    if not isinstance(args, dict):
        return []
    return sorted(str(key) for key in args)


def _duplicate_payload(
    *,
    tool_name: str,
    argument_keys: list[str],
    duplicate_count: int,
    loop_stop_after: int,
    stop_requested: bool,
) -> dict[str, Any]:
    return {
        "status": "react_loop_stop_requested" if stop_requested else "duplicate_blocked",
        "message": _duplicate_message(stop_requested=stop_requested),
        "tool_name": tool_name,
        "argument_keys": argument_keys,
        "duplicate_count": duplicate_count,
        "loop_stop_after": loop_stop_after,
        "stop_requested": stop_requested,
    }


def _loop_stop_ai_message(tool_name: str) -> AIMessage:
    return AIMessage(
        content="",
        additional_kwargs={
            "skypilot_react_loop_stop_requested": True,
            "skypilot_loop_stop_tool_name": tool_name,
        },
    )


def _loop_stop_tool_message(
    tool_call: dict[str, Any],
    *,
    duplicate_count: int,
    loop_stop_after: int,
) -> ToolMessage:
    tool_name = str(tool_call.get("name") or "unknown_tool")
    payload = _duplicate_payload(
        tool_name=tool_name,
        argument_keys=_argument_keys_from_args(tool_call.get("args")),
        duplicate_count=duplicate_count,
        loop_stop_after=loop_stop_after,
        stop_requested=True,
    )
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        name=tool_name,
        tool_call_id=str(tool_call.get("id") or ""),
        status="error",
    )


def _log_duplicate_blocked(
    request: ToolCallRequest,
    *,
    scope: str,
    key: str,
    duplicate_count: int,
    loop_stop_after: int,
) -> None:
    _log_duplicate_event(
        context=_request_context(request),
        scope=scope,
        key=key,
        duplicate_count=duplicate_count,
        loop_stop_after=loop_stop_after,
        tool_call_id=_tool_call_id(request),
        tool_name=_tool_name(request),
        argument_keys=_argument_keys(request),
        args=request.tool_call.get("args"),
    )


def _log_duplicate_event(
    *,
    context: Context | None,
    scope: str,
    key: str,
    duplicate_count: int,
    loop_stop_after: int,
    tool_call_id: str,
    tool_name: str,
    argument_keys: list[str],
    args: Any,
) -> None:
    log_event(
        "react_duplicate_tool_call_blocked",
        context=context,
        redact=False,
        scope=scope,
        duplicate_key=key,
        duplicate_count=duplicate_count,
        loop_stop_after=loop_stop_after,
        stop_requested=duplicate_count >= loop_stop_after,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        argument_keys=argument_keys,
        tool_call={
            "id": tool_call_id,
            "name": tool_name,
            "args": args,
            "argument_keys": argument_keys,
        },
    )


__all__ = [
    "ParamAwareDuplicateToolCallGuard",
    "build_param_aware_duplicate_tool_call_guard",
]
