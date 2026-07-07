"""Agent middleware for structured observability events."""

from __future__ import annotations

import logging
import json
from collections.abc import Callable
from hashlib import sha256
from time import perf_counter
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)

from src.observability.events import log_event
from src.runtime import Context


class ObservabilityMiddleware(AgentMiddleware):
    """Record model-call lifecycle events with complete debug traces."""

    tools: list[Any] = []

    def __init__(self, *, redact: bool = True) -> None:
        self.redact = redact

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Record synchronous model-call lifecycle events."""
        context = _request_context(request)
        started_at = perf_counter()
        log_event(
            "model_call_start",
            context=context,
            redact=False,
            message_count=len(request.messages),
            request_trace=_model_request_trace(request),
            tool_count=len(request.tools),
        )

        try:
            response = handler(request)
        except Exception as exc:
            log_event(
                "model_call_error",
                context=context,
                level=logging.ERROR,
                redact=self.redact,
                duration_ms=_duration_ms(started_at),
                error_type=type(exc).__name__,
            )
            raise

        log_event(
            "model_call_end",
            context=context,
            redact=False,
            duration_ms=_duration_ms(started_at),
            message_count=len(response.result),
            response_trace=_message_trace(response.result),
        )
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        """Record asynchronous model-call lifecycle events."""
        context = _request_context(request)
        started_at = perf_counter()
        log_event(
            "model_call_start",
            context=context,
            redact=False,
            message_count=len(request.messages),
            request_trace=_model_request_trace(request),
            tool_count=len(request.tools),
        )

        try:
            response = await handler(request)
        except Exception as exc:
            log_event(
                "model_call_error",
                context=context,
                level=logging.ERROR,
                redact=self.redact,
                duration_ms=_duration_ms(started_at),
                error_type=type(exc).__name__,
            )
            raise

        log_event(
            "model_call_end",
            context=context,
            redact=False,
            duration_ms=_duration_ms(started_at),
            message_count=len(response.result),
            response_trace=_message_trace(response.result),
        )
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Record synchronous tool-call lifecycle events."""
        context = _tool_request_context(request)
        started_at = perf_counter()
        tool_name = _tool_name(request)
        argument_keys = _tool_argument_keys(request)
        log_event(
            "tool_call_start",
            context=context,
            redact=False,
            tool_call_id=_tool_call_id(request),
            tool_name=tool_name,
            argument_keys=argument_keys,
            tool_call=_tool_call_trace(request.tool_call),
        )

        try:
            response = handler(request)
        except Exception as exc:
            log_event(
                "tool_call_error",
                context=context,
                level=logging.ERROR,
                redact=False,
                tool_call_id=_tool_call_id(request),
                tool_name=tool_name,
                argument_keys=argument_keys,
                tool_call=_tool_call_trace(request.tool_call),
                duration_ms=_duration_ms(started_at),
                error_type=type(exc).__name__,
            )
            raise

        log_event(
            "tool_call_end",
            context=context,
            redact=False,
            tool_call_id=_tool_call_id(request),
            tool_name=tool_name,
            argument_keys=argument_keys,
            tool_call=_tool_call_trace(request.tool_call),
            response_trace=_tool_response_trace(response),
            duration_ms=_duration_ms(started_at),
            status=getattr(response, "status", "success"),
        )
        return response

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Record asynchronous tool-call lifecycle events."""
        context = _tool_request_context(request)
        started_at = perf_counter()
        tool_name = _tool_name(request)
        argument_keys = _tool_argument_keys(request)
        log_event(
            "tool_call_start",
            context=context,
            redact=False,
            tool_call_id=_tool_call_id(request),
            tool_name=tool_name,
            argument_keys=argument_keys,
            tool_call=_tool_call_trace(request.tool_call),
        )

        try:
            response = await handler(request)
        except Exception as exc:
            log_event(
                "tool_call_error",
                context=context,
                level=logging.ERROR,
                redact=False,
                tool_call_id=_tool_call_id(request),
                tool_name=tool_name,
                argument_keys=argument_keys,
                tool_call=_tool_call_trace(request.tool_call),
                duration_ms=_duration_ms(started_at),
                error_type=type(exc).__name__,
            )
            raise

        log_event(
            "tool_call_end",
            context=context,
            redact=False,
            tool_call_id=_tool_call_id(request),
            tool_name=tool_name,
            argument_keys=argument_keys,
            tool_call=_tool_call_trace(request.tool_call),
            response_trace=_tool_response_trace(response),
            duration_ms=_duration_ms(started_at),
            status=getattr(response, "status", "success"),
        )
        return response


def build_observability_middleware(*, redact: bool = True) -> ObservabilityMiddleware:
    """Build middleware that emits structured model lifecycle events."""
    middleware = ObservabilityMiddleware(redact=redact)
    log_event("agent_middleware_attached", redact=redact, middleware="observability")
    return middleware


def _request_context(request: ModelRequest) -> Context | None:
    runtime = getattr(request, "runtime", None)
    context = getattr(runtime, "context", None)
    return context if isinstance(context, Context) else None


def _tool_request_context(request: ToolCallRequest) -> Context | None:
    context = getattr(request.runtime, "context", None)
    return context if isinstance(context, Context) else None


def _tool_name(request: ToolCallRequest) -> str:
    if request.tool is not None:
        return request.tool.name

    name = request.tool_call.get("name")
    return str(name or "unknown_tool")


def _tool_call_id(request: ToolCallRequest) -> str | None:
    tool_call_id = request.tool_call.get("id")
    if tool_call_id:
        return str(tool_call_id)
    runtime_tool_call_id = getattr(request.runtime, "tool_call_id", None)
    return str(runtime_tool_call_id) if runtime_tool_call_id else None


def _tool_argument_keys(request: ToolCallRequest) -> list[str]:
    args = request.tool_call.get("args")
    if not isinstance(args, dict):
        return []
    return sorted(str(key) for key in args)


def _message_trace(messages: Any) -> list[dict[str, Any]]:
    return [_single_message_trace(index, message) for index, message in enumerate(messages)]


def _model_request_trace(request: ModelRequest) -> dict[str, Any]:
    return {
        "system_prompt": request.system_prompt,
        "messages": _message_trace(request.messages),
        "tools": [_tool_definition_trace(tool) for tool in request.tools],
    }


def _single_message_trace(index: int, message: Any) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "index": index,
        "message_type": type(message).__name__,
        "role": str(getattr(message, "type", "unknown")),
        "content": getattr(message, "content", None),
    }
    trace.update(_content_trace(getattr(message, "content", None)))

    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        trace["tool_calls"] = [_tool_call_trace(tool_call) for tool_call in tool_calls]

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        trace["tool_call_id"] = str(tool_call_id)

    name = getattr(message, "name", None)
    if name:
        trace["name"] = str(name)

    return trace


def _tool_call_trace(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        return {"tool_call_type": type(tool_call).__name__}

    args = tool_call.get("args")
    return {
        "id": str(tool_call.get("id") or ""),
        "name": str(tool_call.get("name") or ""),
        "args": args,
        "argument_keys": sorted(str(key) for key in args) if isinstance(args, dict) else [],
    }


def _tool_response_trace(response: Any) -> dict[str, Any]:
    trace = {
        "message_type": type(response).__name__,
        "status": str(getattr(response, "status", "success")),
        "content": getattr(response, "content", None),
    }
    trace.update(_content_trace(getattr(response, "content", None)))
    return trace


def _tool_definition_trace(tool: Any) -> dict[str, Any]:
    trace = {
        "tool_type": type(tool).__name__,
        "name": str(getattr(tool, "name", "")),
        "description": str(getattr(tool, "description", "")),
    }
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        trace["args_schema"] = _schema_trace(args_schema)
    args = getattr(tool, "args", None)
    if args is not None:
        trace["args"] = args
    return trace


def _schema_trace(schema: Any) -> Any:
    try:
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        if hasattr(schema, "schema"):
            return schema.schema()
        return str(schema)
    except Exception as exc:
        return {
            "schema_type": type(schema).__name__,
            "schema_repr": str(schema),
            "schema_error_type": type(exc).__name__,
            "schema_error": str(exc),
        }


def _content_trace(content: Any) -> dict[str, Any]:
    text = _content_text(content)
    encoded = text.encode("utf-8")
    trace: dict[str, Any] = {
        "content_type": type(content).__name__,
        "content_chars": len(text),
        "content_bytes": len(encoded),
        "content_sha256": sha256(encoded).hexdigest(),
    }
    if isinstance(content, list):
        trace["content_block_count"] = len(content)
        trace["content_block_types"] = [
            str(block.get("type", "unknown")) if isinstance(block, dict) else type(block).__name__
            for block in content
        ]
    return trace


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                for key in ("text", "content", "reasoning", "summary"):
                    parts.extend(_text_parts(block.get(key)))
            else:
                parts.extend(_text_parts(block))
        return "".join(parts)
    return "" if content is None else str(content)


def _text_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_text_parts(item))
        return parts
    if isinstance(value, dict):
        try:
            return [json.dumps(value, ensure_ascii=False, sort_keys=True)]
        except TypeError:
            return [str(value)]
    return [] if value is None else [str(value)]


def _duration_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


__all__ = ["ObservabilityMiddleware", "build_observability_middleware"]
