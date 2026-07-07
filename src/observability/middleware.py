"""Agent middleware for structured observability events."""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)

from src.observability.events import log_event
from src.observability.model_trace import (
    message_trace as _message_trace,
    model_request_trace as _model_request_trace,
    tool_call_trace as _tool_call_trace,
    tool_definition_trace as _tool_definition_trace,
    tool_response_trace as _tool_response_trace,
)
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


def _duration_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


__all__ = ["ObservabilityMiddleware", "build_observability_middleware"]
