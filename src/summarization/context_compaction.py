"""Context compaction pipeline for oversized agent model requests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from langchain.agents.middleware import ModelRequest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from src.summarization.layered_context import (
    CompactLayeredContextState,
    build_layered_context_state,
    partition_messages_for_compaction,
)
from src.summarization.tool_semantic import (
    ToolSummaryCandidate,
    build_tool_summary_candidates,
)
from src.prompt import (
    CONTEXT_LEDGER_TOOL_NAME,
    build_context_ledger_tool_call_args,
    build_context_ledger_tool_observation,
)


DEFAULT_TODO_SNAPSHOT_MAX_ITEMS = 20
DEFAULT_TODO_SNAPSHOT_MAX_CONTENT_CHARS = 300


@dataclass(frozen=True)
class LayerOneProjection:
    """Projected compressed-prefix messages plus deterministic trim counters."""

    messages: list[Any]
    reasoning_block_removed_count: int = 0
    tool_message_removed_count: int = 0
    tool_call_removed_count: int = 0
    adjacent_human_merged_count: int = 0
    duplicate_tool_output_count: int = 0
    empty_tool_output_count: int = 0


@dataclass(frozen=True)
class ContextCompactionResult:
    """A transient compacted model request and metadata for observability."""

    request: ModelRequest
    ledger: CompactLayeredContextState
    raw_message_count: int
    layer_one_projection: LayerOneProjection
    todo_snapshot: dict[str, Any] | None = None
    raw_messages: list[Any] | None = None
    synthetic_message_builder: Any | None = None
    compaction_level: str = "l1_l3"
    local_semantic_summaries: list[dict[str, Any]] | None = None
    global_fallback_summary: dict[str, Any] | None = None
    semantic_summary_count: int = 0
    semantic_summary_failed: bool = False
    tool_semantic_candidates: list[ToolSummaryCandidate] | None = None
    tool_semantic_summary_count: int = 0
    tool_semantic_summary_failed: bool = False
    global_fallback_used: bool = False
    deterministic_ledger_included: bool = True
    post_compaction_chars: int = 0
    still_over_budget: bool = False
    semantic_skip_reason: str | None = None
    semantic_error_stage: str | None = None
    semantic_error_type: str | None = None
    semantic_summary_unavailable: bool = False
    semantic_unavailable_reason: str | None = None
    semantic_fallback_used: bool = False
    active_tool_call_ids: set[str] | None = None


def build_context_compaction_request(
    request: ModelRequest,
    *,
    latest_human_text: str,
    estimate_chars: int,
    threshold_chars: int,
    ledger_fraction: float,
    min_ledger_budget_chars: int,
    raw_recent_turns: int,
) -> ContextCompactionResult | None:
    """Build a transient compacted request after budget pressure triggers."""
    compressed_messages, raw_messages = partition_messages_for_compaction(
        request.messages,
        raw_recent_turns=raw_recent_turns,
    )
    active_tool_call_ids = _active_tool_call_ids(raw_messages)
    active_tool_messages = _active_tool_messages(raw_messages)
    if not compressed_messages and not active_tool_messages:
        return None

    projection = project_layer_one_messages(compressed_messages)
    tool_call_messages = [
        message
        for message in [*compressed_messages, *raw_messages]
        if getattr(message, "tool_calls", None)
    ]
    tool_semantic_candidates = build_tool_summary_candidates(
        [*tool_call_messages, *projection.messages, *active_tool_messages]
    )
    layered_state = build_layered_context_state(
        [*projection.messages, *active_tool_messages],
        budget_chars=max(
            round(threshold_chars * ledger_fraction),
            min_ledger_budget_chars,
        ),
        preserve_latest_user_message=False,
    )
    recent_messages = recent_messages_with_current_goal(
        raw_messages,
        current_user_goal=latest_human_text,
    )
    todo_snapshot = build_todo_snapshot_from_request(request)
    compact_request = request.override(
        messages=[
            *recent_messages,
            *synthetic_ledger_messages(
                latest_human_text=latest_human_text,
                ledger=layered_state,
                estimate_chars=estimate_chars,
                threshold_chars=threshold_chars,
                todo_snapshot=todo_snapshot,
            ),
        ],
    )
    return ContextCompactionResult(
        request=compact_request,
        ledger=layered_state,
        raw_message_count=len(recent_messages),
        layer_one_projection=projection,
        todo_snapshot=todo_snapshot,
        tool_semantic_candidates=tool_semantic_candidates,
        active_tool_call_ids=active_tool_call_ids,
        raw_messages=recent_messages,
        synthetic_message_builder=lambda *,
        ledger_override=None,
        local_semantic_summaries=None,
        global_fallback_summary=None,
        include_deterministic_ledger=True: synthetic_ledger_messages(
            latest_human_text=latest_human_text,
            ledger=ledger_override or layered_state,
            estimate_chars=estimate_chars,
            threshold_chars=threshold_chars,
            todo_snapshot=todo_snapshot,
            local_semantic_summaries=local_semantic_summaries or [],
            global_fallback_summary=global_fallback_summary,
            include_deterministic_ledger=include_deterministic_ledger,
        ),
    )


def build_todo_snapshot_from_request(
    request: ModelRequest,
    *,
    max_items: int = DEFAULT_TODO_SNAPSHOT_MAX_ITEMS,
    max_content_chars: int = DEFAULT_TODO_SNAPSHOT_MAX_CONTENT_CHARS,
) -> dict[str, Any] | None:
    """Extract compact protected todo state from a model request, if available."""
    state = getattr(request, "state", None)
    if not isinstance(state, dict):
        return None
    todos = state.get("todos")
    if not isinstance(todos, list):
        return None

    items: list[dict[str, str | int]] = []
    truncated_count = 0
    for index, todo in enumerate(todos):
        if len(items) >= max_items:
            break
        if not isinstance(todo, dict):
            continue
        content = todo.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        compact_content, was_truncated = _truncate_todo_content(
            content.strip(),
            max_content_chars=max_content_chars,
        )
        if was_truncated:
            truncated_count += 1
        status = todo.get("status")
        if not isinstance(status, str) or not status.strip():
            status = "unknown"
        items.append(
            {
                "index": index,
                "content": compact_content,
                "status": status.strip(),
            }
        )

    if not items:
        return None
    total_count = len(todos)
    return {
        "type": "todo_snapshot",
        "total_count": total_count,
        "preserved_count": len(items),
        "dropped_count": max(total_count - len(items), 0),
        "truncated_count": truncated_count,
        "items": items,
        "instruction": (
            "Continue from pending/in_progress items. If task state changes, "
            "update todos with the todo tool."
        ),
    }


def _truncate_todo_content(content: str, *, max_content_chars: int) -> tuple[str, bool]:
    if max_content_chars <= 0:
        return "", bool(content)
    if len(content) <= max_content_chars:
        return content, False
    return f"{content[:max_content_chars]}...", True


def recent_messages_with_current_goal(
    raw_messages: list[Any],
    *,
    current_user_goal: str,
) -> list[Any]:
    """Ensure the transient compacted view keeps the latest user goal visible."""
    if not current_user_goal:
        return raw_messages
    replaced: list[Any] = []
    replaced_human = False
    for message in raw_messages:
        if str(getattr(message, "type", "")) != "human":
            replaced.append(message)
            continue
        replaced.append(HumanMessage(content=current_user_goal))
        replaced_human = True
    if replaced_human:
        return replaced
    return [HumanMessage(content=current_user_goal), *raw_messages]


def replace_active_tool_messages(
    raw_messages: list[Any],
    replacements_by_tool_call_id: dict[str, str],
) -> list[Any]:
    """Replace active ToolMessage content while preserving tool-call protocol."""
    if not replacements_by_tool_call_id:
        return raw_messages
    replaced: list[Any] = []
    for message in raw_messages:
        if str(getattr(message, "type", "")) != "tool":
            replaced.append(message)
            continue
        tool_call_id = str(getattr(message, "tool_call_id", "") or "")
        replacement = replacements_by_tool_call_id.get(tool_call_id)
        if replacement is None:
            replaced.append(message)
            continue
        replaced.append(
            ToolMessage(
                content=replacement,
                name=str(getattr(message, "name", "") or "tool"),
                tool_call_id=tool_call_id,
            )
        )
    return replaced


def project_layer_one_messages(messages: list[Any]) -> LayerOneProjection:
    """Apply deterministic ReAct trimming to a compressed history prefix."""
    reasoning_removed = 0
    tool_calls_removed = 0
    adjacent_human_merged = 0
    duplicate_tool_outputs = 0
    empty_tool_outputs = 0
    historical_has_noise = False
    projected_messages: list[Any] = []
    seen_tool_outputs: dict[str, str] = {}

    for message in messages:
        message_type = str(getattr(message, "type", ""))
        if message_type == "human":
            human_content = str(getattr(message, "content", ""))
            if projected_messages and str(getattr(projected_messages[-1], "type", "")) == "human":
                previous = projected_messages[-1]
                projected_messages[-1] = HumanMessage(
                    content=f"{getattr(previous, 'content', '')}\n\n{human_content}"
                )
                adjacent_human_merged += 1
                historical_has_noise = True
            else:
                projected_messages.append(HumanMessage(content=human_content))
            continue
        if message_type == "tool":
            projected_tool, marker_kind = _project_tool_message(
                message,
                seen_tool_outputs=seen_tool_outputs,
            )
            if marker_kind == "duplicate":
                duplicate_tool_outputs += 1
                historical_has_noise = True
            elif marker_kind == "empty":
                empty_tool_outputs += 1
                historical_has_noise = True
            projected_messages.append(projected_tool)
            continue
        if message_type != "ai":
            projected_messages.append(message)
            continue

        visible_text, removed_reasoning = _visible_ai_text_and_reasoning_count(
            getattr(message, "content", "")
        )
        reasoning_removed += removed_reasoning
        message_tool_call_count = len(getattr(message, "tool_calls", None) or [])
        block_tool_call_count = _content_block_tool_call_count(getattr(message, "content", ""))
        if message_tool_call_count or block_tool_call_count or removed_reasoning:
            historical_has_noise = True
        tool_calls_removed += message_tool_call_count + block_tool_call_count
        if visible_text:
            projected_messages.append(AIMessage(content=visible_text))

    if not historical_has_noise:
        return LayerOneProjection(messages=messages)

    return LayerOneProjection(
        messages=projected_messages,
        reasoning_block_removed_count=reasoning_removed,
        tool_message_removed_count=0,
        tool_call_removed_count=tool_calls_removed,
        adjacent_human_merged_count=adjacent_human_merged,
        duplicate_tool_output_count=duplicate_tool_outputs,
        empty_tool_output_count=empty_tool_outputs,
    )


def _project_tool_message(
    message: Any,
    *,
    seen_tool_outputs: dict[str, str],
) -> tuple[Any, str | None]:
    content_text = _content_text(getattr(message, "content", ""))
    content_sha = sha256(content_text.encode("utf-8")).hexdigest()
    tool_call_id = str(getattr(message, "tool_call_id", "") or "")
    tool_name = str(getattr(message, "name", "") or "tool")

    if _is_empty_tool_output(content_text):
        return (
            _tool_marker_message(
                source=message,
                status="compacted_empty_tool_output",
                payload={
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "content_sha256": content_sha,
                },
            ),
            "empty",
        )

    duplicate_of = seen_tool_outputs.get(content_sha)
    if duplicate_of is not None:
        return (
            _tool_marker_message(
                source=message,
                status="compacted_duplicate_tool_output",
                payload={
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "duplicate_of_tool_call_id": duplicate_of,
                    "content_sha256": content_sha,
                },
            ),
            "duplicate",
        )

    seen_tool_outputs[content_sha] = tool_call_id
    return message, None


def _tool_marker_message(
    *,
    source: Any,
    status: str,
    payload: dict[str, Any],
) -> ToolMessage:
    marker = {"status": status, **payload}
    return ToolMessage(
        content=json.dumps(marker, ensure_ascii=False, separators=(",", ":")),
        name=str(getattr(source, "name", "") or "tool"),
        tool_call_id=str(getattr(source, "tool_call_id", "") or ""),
    )


def _is_empty_tool_output(content_text: str) -> bool:
    if not content_text.strip():
        return True
    try:
        value = json.loads(content_text)
    except json.JSONDecodeError:
        return False
    return _is_empty_json_value(value)


def _is_empty_json_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, list):
        return all(_is_empty_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(_is_empty_json_value(item) for item in value.values())
    return False


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(content)


def _active_tool_call_ids(raw_messages: list[Any]) -> set[str]:
    ids: set[str] = set()
    for message in raw_messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = str(tool_call.get("id") or "")
            if tool_call_id:
                ids.add(tool_call_id)
    return ids


def _active_tool_messages(raw_messages: list[Any]) -> list[Any]:
    active_ids = _active_tool_call_ids(raw_messages)
    if not active_ids:
        return []
    return [
        message
        for message in raw_messages
        if str(getattr(message, "type", "")) == "tool"
        and str(getattr(message, "tool_call_id", "") or "") in active_ids
    ]


def synthetic_ledger_messages(
    *,
    latest_human_text: str,
    ledger: CompactLayeredContextState,
    estimate_chars: int,
    threshold_chars: int,
    todo_snapshot: dict[str, Any] | None = None,
    local_semantic_summaries: list[dict[str, Any]] | None = None,
    global_fallback_summary: dict[str, Any] | None = None,
    include_deterministic_ledger: bool = True,
) -> list[Any]:
    """Represent compacted historical state as a protocol-valid tool observation."""
    tool_call_id = synthetic_ledger_tool_call_id(ledger)
    tool_args = build_context_ledger_tool_call_args(
        original_user_message=latest_human_text,
        estimate_chars=estimate_chars,
        threshold_chars=threshold_chars,
    )
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": tool_call_id,
                    "name": CONTEXT_LEDGER_TOOL_NAME,
                    "args": tool_args,
                }
            ],
        ),
        ToolMessage(
            content=build_context_ledger_tool_observation(
                original_user_message=latest_human_text,
                ledger=ledger,
                estimate_chars=estimate_chars,
                threshold_chars=threshold_chars,
                todo_snapshot=todo_snapshot,
                local_semantic_summaries=local_semantic_summaries or [],
                global_fallback_summary=global_fallback_summary,
                include_deterministic_ledger=include_deterministic_ledger,
            ),
            name=CONTEXT_LEDGER_TOOL_NAME,
            tool_call_id=tool_call_id,
        ),
    ]


def synthetic_ledger_tool_call_id(ledger: CompactLayeredContextState) -> str:
    """Return a stable synthetic call id for a compacted ledger."""
    digest = sha256(ledger.to_prompt_text().encode("utf-8")).hexdigest()[:16]
    return f"context_ledger_{digest}"


def _visible_ai_text_and_reasoning_count(content: Any) -> tuple[str, int]:
    if isinstance(content, str):
        return content.strip(), 0
    if not isinstance(content, list):
        return "", 0
    parts: list[str] = []
    reasoning_count = 0
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", ""))
        if block_type in {"reasoning", "reasoning_content"}:
            reasoning_count += 1
            continue
        if block_type in {"text", "output_text", "message"}:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            value = block.get("content")
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts).strip(), reasoning_count


def _content_block_tool_call_count(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    return sum(
        1
        for block in content
        if isinstance(block, dict)
        and str(block.get("type", "")) in {"function_call", "tool_call"}
    )
