"""L3 semantic compression for individual oversized tool results."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from src.prompt.tool_summary import build_tool_result_summary_messages
from src.summarization.semantic_cache import SemanticSummaryCache
from src.summarization.structured_output import (
    SemanticSummaryCapability,
    SemanticSummaryUnavailableError,
    invoke_semantic_summary_text_with_cache,
)
from src.summarization.tool_observation import json_stats_summary


DEFAULT_TOOL_SUMMARY_MIN_CHARS = 1200
DEFAULT_TOOL_SUMMARY_CHUNK_CHARS = 12000
SummaryEventCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class ToolSummaryCandidate:
    """One complete tool result eligible for L3 semantic compression."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    content: str
    result_stats: dict[str, Any]


def build_tool_summary_candidates(
    messages: list[Any],
    *,
    min_chars: int = DEFAULT_TOOL_SUMMARY_MIN_CHARS,
) -> list[ToolSummaryCandidate]:
    """Collect oversized ToolMessages with their original call arguments."""
    args_by_id: dict[str, dict[str, Any]] = {}
    names_by_id: dict[str, str] = {}
    candidates: list[ToolSummaryCandidate] = []
    for message in messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = str(tool_call.get("id") or "")
            if not tool_call_id:
                continue
            args = tool_call.get("args")
            args_by_id[tool_call_id] = args if isinstance(args, dict) else {}
            names_by_id[tool_call_id] = str(tool_call.get("name") or "")

        if str(getattr(message, "type", "")) != "tool":
            continue
        content = _content_text(getattr(message, "content", ""))
        if len(content) < min_chars:
            continue
        tool_call_id = str(getattr(message, "tool_call_id", "") or "")
        parsed = _parse_json(content)
        candidates.append(
            ToolSummaryCandidate(
                tool_call_id=tool_call_id,
                tool_name=str(
                    getattr(message, "name", "")
                    or names_by_id.get(tool_call_id)
                    or "tool"
                ),
                args=args_by_id.get(tool_call_id, {}),
                content=content,
                result_stats=json_stats_summary(
                    parsed if parsed is not None else content
                ),
            )
        )
    return candidates


def chunk_tool_result(
    content: str,
    *,
    max_chars: int = DEFAULT_TOOL_SUMMARY_CHUNK_CHARS,
) -> list[str]:
    """Split JSON or text on complete semantic boundaries, never substrings."""
    if max_chars <= 0 or len(content) <= max_chars:
        return [content]
    parsed = _parse_json(content)
    if isinstance(parsed, dict):
        return _chunk_json_object(parsed, max_chars=max_chars)
    if isinstance(parsed, list):
        return _pack_json_items(parsed, max_chars=max_chars)
    return _pack_text_lines(content.splitlines(keepends=True), max_chars=max_chars)


def summarize_tool_candidates(
    summary_model: Any,
    candidates: list[ToolSummaryCandidate],
    *,
    max_chunk_chars: int = DEFAULT_TOOL_SUMMARY_CHUNK_CHARS,
    summary_capability: SemanticSummaryCapability | None = None,
    summary_cache: SemanticSummaryCache | None = None,
    event_callback: SummaryEventCallback | None = None,
) -> dict[str, dict[str, Any]]:
    """Summarize complete tool-result chunks as plain model-readable content."""
    summaries: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not candidate.tool_call_id:
            continue
        chunks = chunk_tool_result(candidate.content, max_chars=max_chunk_chars)
        chunk_summaries: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            started_at = perf_counter()
            _emit_summary_event(
                event_callback,
                "context_summary_start",
                stage="l3_tool_semantic",
                tool_name=candidate.tool_name,
                tool_call_id=candidate.tool_call_id,
                chunk_index=index,
                chunk_count=len(chunks),
                input_chars=len(chunk),
            )
            try:
                summary_result = invoke_semantic_summary_text_with_cache(
                    summary_model,
                    build_tool_result_summary_messages(
                        tool_name=candidate.tool_name,
                        args=candidate.args,
                        result_chunk=chunk,
                        result_stats=candidate.result_stats,
                        chunk_index=index,
                        chunk_count=len(chunks),
                    ),
                    capability=summary_capability,
                    cache=summary_cache,
                )
                summary_text = summary_result.text
            except SemanticSummaryUnavailableError as exc:
                _emit_summary_event(
                    event_callback,
                    "context_summary_unavailable",
                    stage="l3_tool_semantic",
                    tool_name=candidate.tool_name,
                    tool_call_id=candidate.tool_call_id,
                    chunk_index=index,
                    chunk_count=len(chunks),
                    status="unavailable",
                    duration_ms=_duration_ms(started_at),
                    reason=exc.reason,
                    cached=exc.cached,
                    fallback="deterministic_compaction",
                )
                raise
            except Exception as exc:
                _emit_summary_event(
                    event_callback,
                    "context_summary_error",
                    stage="l3_tool_semantic",
                    tool_name=candidate.tool_name,
                    tool_call_id=candidate.tool_call_id,
                    chunk_index=index,
                    chunk_count=len(chunks),
                    status="error",
                    duration_ms=_duration_ms(started_at),
                    error_type=type(exc).__name__,
                )
                raise
            _emit_summary_event(
                event_callback,
                "context_summary_end",
                stage="l3_tool_semantic",
                tool_name=candidate.tool_name,
                tool_call_id=candidate.tool_call_id,
                chunk_index=index,
                chunk_count=len(chunks),
                status="success",
                duration_ms=_duration_ms(started_at),
                output_chars=len(summary_text),
                summary_content=summary_text,
                cached=summary_result.cached,
                cache_key=summary_result.cache_key,
            )
            chunk_summaries.append(summary_text)
        summaries[candidate.tool_call_id] = {
            "content": "\n\n".join(_unique_strings(chunk_summaries)),
        }
    return summaries


def _chunk_json_object(value: dict[str, Any], *, max_chars: int) -> list[str]:
    list_fields = [
        (key, item)
        for key, item in value.items()
        if isinstance(item, list) and item
    ]
    if not list_fields:
        return _pack_json_items(
            [{key: item} for key, item in value.items()],
            max_chars=max_chars,
            merge_dict_items=True,
        )

    context = {
        key: item
        for key, item in value.items()
        if not isinstance(item, list)
    }
    chunks: list[str] = []
    for key, items in list_fields:
        groups = _pack_values_with_context(
            items,
            context=context,
            list_key=key,
            max_chars=max_chars,
        )
        chunks.extend(
            json.dumps(group, ensure_ascii=False, default=str)
            for group in groups
        )
    return chunks


def _pack_values_with_context(
    values: list[Any],
    *,
    context: dict[str, Any],
    list_key: str,
    max_chars: int,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current: list[Any] = []
    for value in values:
        candidate = {**context, list_key: [*current, value]}
        if current and len(json.dumps(candidate, ensure_ascii=False, default=str)) > max_chars:
            groups.append({**context, list_key: current})
            current = [value]
        else:
            current.append(value)
    if current:
        groups.append({**context, list_key: current})
    return groups


def _pack_json_items(
    values: list[Any],
    *,
    max_chars: int,
    merge_dict_items: bool = False,
) -> list[str]:
    chunks: list[str] = []
    current: list[Any] = []
    for value in values:
        candidate = [*current, value]
        rendered = _render_json_group(candidate, merge_dict_items=merge_dict_items)
        if current and len(rendered) > max_chars:
            chunks.append(_render_json_group(current, merge_dict_items=merge_dict_items))
            current = [value]
        else:
            current.append(value)
    if current:
        chunks.append(_render_json_group(current, merge_dict_items=merge_dict_items))
    return chunks


def _render_json_group(values: list[Any], *, merge_dict_items: bool) -> str:
    if merge_dict_items:
        merged: dict[str, Any] = {}
        for value in values:
            if isinstance(value, dict):
                merged.update(value)
        return json.dumps(merged, ensure_ascii=False, default=str)
    return json.dumps(values, ensure_ascii=False, default=str)


def _pack_text_lines(lines: list[str], *, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for line in lines:
        if current and len(current) + len(line) > max_chars:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks or [""]


def _parse_json(content: str) -> Any | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _emit_summary_event(
    callback: SummaryEventCallback | None,
    event: str,
    **fields: Any,
) -> None:
    if callback is not None:
        callback(event, fields)


def _duration_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


__all__ = [
    "DEFAULT_TOOL_SUMMARY_CHUNK_CHARS",
    "DEFAULT_TOOL_SUMMARY_MIN_CHARS",
    "ToolSummaryCandidate",
    "build_tool_summary_candidates",
    "chunk_tool_result",
    "summarize_tool_candidates",
]
