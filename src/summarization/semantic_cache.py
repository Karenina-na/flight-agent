"""Process-local cache for semantic summary model calls."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock
from typing import Any

from langchain_core.messages import BaseMessage


DEFAULT_SEMANTIC_SUMMARY_CACHE_MAX_ITEMS = 256


@dataclass(frozen=True)
class SemanticSummaryCacheResult:
    """Cached summary text and cache status for one semantic request."""

    text: str
    cached: bool
    cache_key: str


class SemanticSummaryCache:
    """Small thread-safe LRU cache keyed by normalized summary inputs."""

    def __init__(
        self,
        *,
        max_items: int = DEFAULT_SEMANTIC_SUMMARY_CACHE_MAX_ITEMS,
    ) -> None:
        self.max_items = max(max_items, 0)
        self._items: OrderedDict[str, str] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> str | None:
        """Return cached text and mark the entry as recently used."""
        if self.max_items <= 0:
            return None
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: str) -> None:
        """Store summary text, evicting the least recently used entry if needed."""
        if self.max_items <= 0:
            return
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached summaries."""
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def semantic_summary_cache_key(
    *,
    summary_model: Any,
    messages: Any,
) -> str:
    """Build a deterministic cache key for a semantic summary request."""
    payload = {
        "model": _model_identity(summary_model),
        "messages": _normalize_value(messages),
    }
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(rendered.encode("utf-8")).hexdigest()


def _model_identity(summary_model: Any) -> dict[str, Any]:
    """Return stable public identity fields without serializing the client object."""
    fields: dict[str, Any] = {}
    for name in (
        "model_name",
        "model",
        "deployment_name",
        "openai_api_base",
        "base_url",
        "temperature",
    ):
        value = getattr(summary_model, name, None)
        if value is not None:
            fields[name] = str(value)
    if fields:
        return fields
    return {"class": f"{summary_model.__class__.__module__}.{summary_model.__class__.__name__}"}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return {
            "type": getattr(value, "type", value.__class__.__name__),
            "content": _normalize_value(getattr(value, "content", "")),
            "name": getattr(value, "name", None),
            "additional_kwargs": _normalize_value(
                getattr(value, "additional_kwargs", {})
            ),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_value(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _normalize_value(model_dump())
    return str(value)


__all__ = [
    "DEFAULT_SEMANTIC_SUMMARY_CACHE_MAX_ITEMS",
    "SemanticSummaryCache",
    "SemanticSummaryCacheResult",
    "semantic_summary_cache_key",
]
