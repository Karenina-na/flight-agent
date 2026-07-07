"""Event helpers for structured agent observability."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from threading import RLock
from time import perf_counter
from typing import Any

from src.observability.logging import get_logger, sanitize_fields
from src.runtime import Context

_TRACE_EVENTS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "skypilot_trace_events",
    default=None,
)
_TRACE_EVENT_REGISTRY: dict[str, list[dict[str, Any]]] = {}
_TRACE_EVENT_LOCK = RLock()


def log_event(
    event: str,
    *,
    context: Context | None = None,
    level: int = logging.INFO,
    redact: bool = True,
    **fields: Any,
) -> None:
    """Log a structured observability event."""
    event_fields = _context_fields(context)
    event_fields.update(fields)
    event_fields = sanitize_fields(event_fields, redact=redact)
    _append_trace_event(event, level=level, fields=event_fields)

    get_logger().log(
        level,
        event,
        extra={
            "event": event,
            "fields": event_fields,
        },
    )


@contextmanager
def collect_trace_events(trace_id: str | None = None) -> Iterator[list[dict[str, Any]]]:
    """Collect structured log events emitted within this context."""
    events: list[dict[str, Any]] = []
    token = _TRACE_EVENTS.set(events)
    if trace_id:
        with _TRACE_EVENT_LOCK:
            _TRACE_EVENT_REGISTRY[trace_id] = events
    try:
        yield events
    finally:
        if trace_id:
            with _TRACE_EVENT_LOCK:
                if _TRACE_EVENT_REGISTRY.get(trace_id) is events:
                    del _TRACE_EVENT_REGISTRY[trace_id]
        _TRACE_EVENTS.reset(token)


def text_trace_fields(prefix: str, text: str) -> dict[str, Any]:
    """Return safe text trace fields without logging raw content."""
    encoded = text.encode("utf-8")
    return {
        f"{prefix}_chars": len(text),
        f"{prefix}_bytes": len(encoded),
        f"{prefix}_sha256": sha256(encoded).hexdigest(),
    }


def full_text_trace_fields(prefix: str, text: str) -> dict[str, Any]:
    """Return full debug text trace fields, including raw text."""
    return {
        prefix: text,
        **text_trace_fields(prefix, text),
    }


@contextmanager
def observe_agent_run(
    context: Context,
    *,
    entrypoint: str,
    stream_mode: str,
    redact: bool = True,
) -> Iterator[None]:
    """Log start/end/error events around one agent run."""
    started_at = perf_counter()
    log_event(
        "agent_run_start",
        context=context,
        redact=redact,
        entrypoint=entrypoint,
        stream_mode=stream_mode,
    )

    try:
        yield
    except Exception as exc:
        log_event(
            "agent_run_error",
            context=context,
            level=logging.ERROR,
            redact=redact,
            entrypoint=entrypoint,
            stream_mode=stream_mode,
            duration_ms=_duration_ms(started_at),
            error_type=type(exc).__name__,
        )
        raise

    log_event(
        "agent_run_end",
        context=context,
        redact=redact,
        entrypoint=entrypoint,
        stream_mode=stream_mode,
        duration_ms=_duration_ms(started_at),
    )


def observe_agent_stream(
    stream: Iterable[Any],
    context: Context,
    *,
    entrypoint: str,
    stream_mode: str,
    redact: bool = True,
) -> Iterator[Any]:
    """Yield an agent stream while logging run lifecycle events."""
    with observe_agent_run(
        context,
        entrypoint=entrypoint,
        stream_mode=stream_mode,
        redact=redact,
    ):
        yield from stream


def _context_fields(context: Context | None) -> dict[str, Any]:
    if context is None:
        return {}

    return {
        "user_id": context.user_id,
        "thread_id": context.thread_id,
        "trace_id": context.thread_id or context.run_id or context.request_id,
        "turn_id": context.request_id,
        "tenant_id": context.tenant_id,
        "workspace_id": context.workspace_id,
        "request_id": context.request_id,
        "run_id": context.run_id,
        "environment": context.environment,
    }


def _duration_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _append_trace_event(event: str, *, level: int, fields: dict[str, Any]) -> None:
    trace_event = {
        "event": event,
        "level": logging.getLevelName(level),
        "fields": fields,
    }
    current_events = _TRACE_EVENTS.get()
    if current_events is not None:
        current_events.append(trace_event)

    trace_id = fields.get("trace_id")
    if not trace_id:
        return

    with _TRACE_EVENT_LOCK:
        registered_events = _TRACE_EVENT_REGISTRY.get(str(trace_id))

    if registered_events is not None and registered_events is not current_events:
        registered_events.append(trace_event)


def _clear_current_context_for_test() -> Any:
    return _TRACE_EVENTS.set(None)


def _reset_current_context_for_test(token: Any) -> None:
    _TRACE_EVENTS.reset(token)


collect_trace_events.clear_current_context_for_test = _clear_current_context_for_test  # type: ignore[attr-defined]
collect_trace_events.reset_current_context_for_test = _reset_current_context_for_test  # type: ignore[attr-defined]


__all__ = [
    "collect_trace_events",
    "full_text_trace_fields",
    "log_event",
    "observe_agent_run",
    "observe_agent_stream",
    "text_trace_fields",
]
