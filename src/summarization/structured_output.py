"""Model invocation helpers for semantic summary stages."""

from __future__ import annotations

from collections.abc import Mapping
from threading import Lock
from typing import Any

from langchain_core.messages import BaseMessage

from src.summarization.response_content import visible_response_text
from src.summarization.semantic_cache import (
    SemanticSummaryCache,
    SemanticSummaryCacheResult,
    semantic_summary_cache_key,
)


class SemanticSummaryUnavailableError(RuntimeError):
    """Raised when a model cannot produce a usable semantic summary."""

    def __init__(self, reason: str, *, cached: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.cached = cached


class SemanticSummaryCapability:
    """Process-local capability state shared across summary attempts."""

    def __init__(self) -> None:
        self._available = True
        self._reason: str | None = None
        self._lock = Lock()

    @property
    def available(self) -> bool:
        with self._lock:
            return self._available

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def mark_unavailable(self, reason: str) -> None:
        """Disable semantic summaries after a confirmed incompatibility."""
        with self._lock:
            self._available = False
            self._reason = reason

    def ensure_available(self) -> None:
        """Fail fast when a previous call confirmed incompatibility."""
        with self._lock:
            if not self._available:
                raise SemanticSummaryUnavailableError(
                    self._reason or "semantic_summary_unavailable",
                    cached=True,
                )


def invoke_semantic_summary_text(
    summary_model: Any,
    messages: Any,
    *,
    capability: SemanticSummaryCapability | None = None,
) -> str:
    """Invoke a summary model and return its visible content as the summary."""
    return invoke_semantic_summary_text_with_cache(
        summary_model,
        messages,
        capability=capability,
        cache=None,
    ).text


def invoke_semantic_summary_text_with_cache(
    summary_model: Any,
    messages: Any,
    *,
    capability: SemanticSummaryCapability | None = None,
    cache: SemanticSummaryCache | None = None,
) -> SemanticSummaryCacheResult:
    """Invoke a summary model with optional deterministic process-local caching."""
    if capability is not None:
        capability.ensure_available()
    cache_key = semantic_summary_cache_key(
        summary_model=summary_model,
        messages=messages,
    )
    if cache is not None:
        cached_text = cache.get(cache_key)
        if cached_text is not None:
            return SemanticSummaryCacheResult(
                text=cached_text,
                cached=True,
                cache_key=cache_key,
            )
    response = summary_model.invoke(messages)
    try:
        summary_text = _response_text(response)
    except SemanticSummaryUnavailableError as exc:
        if capability is not None:
            capability.mark_unavailable(exc.reason)
        raise
    if cache is not None:
        cache.set(cache_key, summary_text)
    return SemanticSummaryCacheResult(
        text=summary_text,
        cached=False,
        cache_key=cache_key,
    )


def _response_text(response: Any) -> str:
    if isinstance(response, BaseMessage):
        return _visible_summary_text(response)
    if isinstance(response, Mapping):
        if any(key in response for key in ("raw", "parsed")):
            return _envelope_summary_text(response)
        content = response.get("content") or response.get("summary")
        if isinstance(content, str):
            return _non_empty_text(content, fallback=response)
        return _non_empty_text(visible_response_text(response), fallback=response)
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return _response_text(dumped)
    return _non_empty_text(visible_response_text(response), fallback=response)


def _envelope_summary_text(response: Mapping[str, Any]) -> str:
    parsed = response.get("parsed")
    if isinstance(parsed, str):
        return _non_empty_text(parsed, fallback=response)
    if isinstance(parsed, Mapping):
        content = parsed.get("content") or parsed.get("summary")
        if isinstance(content, str):
            return _non_empty_text(content, fallback=response)

    raw = response.get("raw")
    visible_text = visible_response_text(raw).strip() if raw is not None else ""
    return _non_empty_text(visible_text, fallback=raw)


def _visible_summary_text(response: Any) -> str:
    return _non_empty_text(visible_response_text(response), fallback=response)


def _non_empty_text(text: str, *, fallback: Any) -> str:
    normalized = str(text or "").strip()
    if normalized:
        return normalized
    reason = (
        "reasoning_only_output"
        if _contains_reasoning_content(fallback)
        else "empty_visible_output"
    )
    raise SemanticSummaryUnavailableError(reason)


def _contains_reasoning_content(response: Any) -> bool:
    if response is None:
        return False
    content = getattr(response, "content", response)
    if _value_contains_reasoning(content):
        return True
    additional_kwargs = getattr(response, "additional_kwargs", None)
    if isinstance(additional_kwargs, Mapping):
        if additional_kwargs.get("reasoning_content"):
            return True
    usage_metadata = getattr(response, "usage_metadata", None)
    return _value_contains_reasoning(usage_metadata)


def _value_contains_reasoning(value: Any) -> bool:
    if isinstance(value, Mapping):
        block_type = str(value.get("type") or "").lower()
        if "reasoning" in block_type:
            return True
        for key, nested in value.items():
            if "reasoning" in str(key).lower() and nested:
                return True
            if _value_contains_reasoning(nested):
                return True
        return False
    if isinstance(value, list):
        return any(_value_contains_reasoning(item) for item in value)
    return False


__all__ = [
    "SemanticSummaryCapability",
    "SemanticSummaryUnavailableError",
    "invoke_semantic_summary_text",
    "invoke_semantic_summary_text_with_cache",
]
