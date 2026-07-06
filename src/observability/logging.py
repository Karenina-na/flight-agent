"""Structured logging setup for agent observability."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import LoggingSettings

LOGGER_NAME = "skypilot"
SENSITIVE_KEYWORDS = ("api_key", "authorization", "password", "secret", "token")


class TextFormatter(logging.Formatter):
    """Human-readable formatter for structured event records."""

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event", record.getMessage())
        fields = getattr(record, "fields", {})
        field_text = _format_fields(fields)
        message = f"{record.levelname} event={event}"
        return f"{message} {field_text}" if field_text else message


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured event records."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging(settings: LoggingSettings) -> logging.Logger:
    """Configure and return the project logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False

    if not settings.enabled:
        logger.addHandler(logging.NullHandler())
        logger.disabled = True
        return logger

    logger.disabled = False
    logger.setLevel(getattr(logging, settings.level))

    level = getattr(logging, settings.level)
    formatter = JsonFormatter() if settings.format == "json" else TextFormatter()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if settings.output_path:
        log_path = Path(settings.output_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the project logger without mutating global logging state."""
    return logging.getLogger(LOGGER_NAME)


def sanitize_fields(fields: dict[str, Any], *, redact: bool = True) -> dict[str, Any]:
    """Return fields safe for logs."""
    if not redact:
        return dict(fields)

    return {key: _sanitize_value(key, value) for key, value in fields.items()}


def _sanitize_value(key: str, value: Any) -> Any:
    if _is_sensitive_key(key):
        return "[REDACTED]"

    if isinstance(value, dict):
        return sanitize_fields(value, redact=True)
    if isinstance(value, list):
        return [_sanitize_value(key, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(key, item) for item in value)
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized_key = key.lower()
    return any(keyword in normalized_key for keyword in SENSITIVE_KEYWORDS)


def _format_fields(fields: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in fields.items())


__all__ = [
    "JsonFormatter",
    "TextFormatter",
    "configure_logging",
    "get_logger",
    "sanitize_fields",
]
