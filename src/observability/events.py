"""Event helpers for structured agent observability."""

from __future__ import annotations

import logging
from typing import Any

from src.observability.logging import get_logger, sanitize_fields
from src.runtime import Context


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

    get_logger().log(
        level,
        event,
        extra={
            "event": event,
            "fields": event_fields,
        },
    )


def _context_fields(context: Context | None) -> dict[str, Any]:
    if context is None:
        return {}

    return {
        "user_id": context.user_id,
        "thread_id": context.thread_id,
        "tenant_id": context.tenant_id,
        "workspace_id": context.workspace_id,
        "request_id": context.request_id,
        "run_id": context.run_id,
        "environment": context.environment,
    }


__all__ = ["log_event"]
