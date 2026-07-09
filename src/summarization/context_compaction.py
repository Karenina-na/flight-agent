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
from src.prompt import (
    CONTEXT_LEDGER_TOOL_NAME,
    build_context_ledger_tool_call_args,
    build_context_ledger_tool_observation,
)


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
    if not compressed_messages:
        return None

    projection = project_layer_one_messages(compressed_messages)
    layered_state = build_layered_context_state(
        projection.messages,
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
    compact_request = request.override(
        messages=[
            *recent_messages,
            *synthetic_ledger_messages(
                latest_human_text=latest_human_text,
                ledger=layered_state,
                estimate_chars=estimate_chars,
                threshold_chars=threshold_chars,
            ),
        ],
    )
    return ContextCompactionResult(
        request=compact_request,
        ledger=layered_state,
        raw_message_count=len(recent_messages),
        layer_one_projection=projection,
    )


def recent_messages_with_current_goal(
    raw_messages: list[Any],
    *,
    current_user_goal: str,
) -> list[Any]:
    """Ensure the transient compacted view keeps the latest user goal visible."""
    if not current_user_goal:
        return raw_messages
    if any(
        str(getattr(message, "type", "")) == "human"
        and str(getattr(message, "content", "")) == current_user_goal
        for message in raw_messages
    ):
        return raw_messages
    if len(raw_messages) == 1 and str(getattr(raw_messages[0], "type", "")) == "human":
        return [HumanMessage(content=current_user_goal)]
    return [*raw_messages, HumanMessage(content=current_user_goal)]


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


def synthetic_ledger_messages(
    *,
    latest_human_text: str,
    ledger: CompactLayeredContextState,
    estimate_chars: int,
    threshold_chars: int,
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
