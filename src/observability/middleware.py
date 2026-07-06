"""Agent middleware for structured observability events."""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from src.observability.events import log_event
from src.runtime import Context


class ObservabilityMiddleware(AgentMiddleware):
    """Record model-call lifecycle events without exposing message content."""

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
            redact=self.redact,
            message_count=len(request.messages),
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
            redact=self.redact,
            duration_ms=_duration_ms(started_at),
            message_count=len(response.result),
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
            redact=self.redact,
            message_count=len(request.messages),
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
            redact=self.redact,
            duration_ms=_duration_ms(started_at),
            message_count=len(response.result),
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


def _duration_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


__all__ = ["ObservabilityMiddleware", "build_observability_middleware"]
