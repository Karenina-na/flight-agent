"""Guard against repeated identical ReAct tool calls in one turn."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain.messages import ToolMessage

from src.observability import log_event
from src.runtime import Context


class ReactDuplicateToolCallGuard(AgentMiddleware):
    """Block identical tool calls repeated within the same request turn."""

    tools: list[Any] = []

    def __init__(self) -> None:
        self._seen_by_scope: dict[str, set[str]] = {}

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Run the first unique tool call and block exact duplicates."""
        duplicate = self._record_or_detect_duplicate(request)
        if duplicate:
            return self._duplicate_response(request)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        """Run the first unique async tool call and block exact duplicates."""
        duplicate = self._record_or_detect_duplicate(request)
        if duplicate:
            return self._duplicate_response(request)
        return await handler(request)

    def _record_or_detect_duplicate(self, request: ToolCallRequest) -> bool:
        scope = _request_scope(request)
        key = _tool_call_key(request)
        seen = self._seen_by_scope.setdefault(scope, set())
        if key in seen:
            _log_duplicate_blocked(request, scope=scope, key=key)
            return True
        seen.add(key)
        return False

    def _duplicate_response(self, request: ToolCallRequest) -> ToolMessage:
        payload = {
            "status": "duplicate_blocked",
            "message": (
                "This exact tool call has already been executed in this turn. "
                "Use the previous tool result and produce a final answer "
                "instead of calling the same tool again."
            ),
            "tool_name": _tool_name(request),
            "argument_keys": _argument_keys(request),
        }
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            name=_tool_name(request),
            tool_call_id=_tool_call_id(request) or "",
            status="success",
        )


def build_react_duplicate_tool_call_guard() -> ReactDuplicateToolCallGuard:
    """Build duplicate tool-call guard middleware."""
    return ReactDuplicateToolCallGuard()


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
) -> None:
    log_event(
        "react_duplicate_tool_call_blocked",
        context=_request_context(request),
        redact=False,
        scope=scope,
        duplicate_key=key,
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
