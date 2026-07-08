"""Trace serialization helpers for local web debugging."""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk

TRACE_DUMP_DIR = Path("logs/traces")


def is_assistant_message(message: object) -> bool:
    """Return True for assistant messages that are safe to render."""
    return isinstance(message, (AIMessage, AIMessageChunk))


def latest_assistant_message(invoke_output: object) -> object | None:
    """Return the last assistant message from an agent.invoke result."""
    if is_assistant_message(invoke_output):
        return invoke_output

    messages: list[object] = []
    if isinstance(invoke_output, dict):
        raw_messages = invoke_output.get("messages")
        if isinstance(raw_messages, list):
            messages = raw_messages
    else:
        raw_messages = getattr(invoke_output, "messages", None)
        if isinstance(raw_messages, list):
            messages = raw_messages

    for message in reversed(messages):
        if is_assistant_message(message):
            return message
    return None


def serialize_invoke_output(invoke_output: object) -> object:
    """Serialize an agent.invoke result for local JSON trace debugging."""
    if isinstance(invoke_output, dict):
        payload: dict[str, Any] = {"output_type": "dict"}
        for key, value in invoke_output.items():
            if key == "messages" and isinstance(value, list):
                payload[key] = [serialize_message(message) for message in value]
            else:
                payload[key] = json_safe(value)
        return payload

    if isinstance(invoke_output, list):
        return {
            "output_type": "list",
            "items": [json_safe(item) for item in invoke_output],
        }

    if is_assistant_message(invoke_output) or hasattr(invoke_output, "content"):
        return serialize_message(invoke_output)

    return json_safe(invoke_output)


def serialize_message(message: object) -> dict[str, Any]:
    """Serialize a LangChain message-like object for local trace dumps."""
    trace: dict[str, Any] = {
        "message_type": type(message).__name__,
        "role": str(getattr(message, "type", "unknown")),
        "content": getattr(message, "content", None),
    }

    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        trace["tool_calls"] = json_safe(tool_calls)

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        trace["tool_call_id"] = str(tool_call_id)

    name = getattr(message, "name", None)
    if name:
        trace["name"] = str(name)

    return trace


def message_text(message_chunk: object) -> str:
    """Extract visible assistant text from a LangChain message."""
    content = getattr(message_chunk, "content", message_chunk)
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content_blocks(message_chunk):
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))

    if parts:
        return "".join(parts)

    return chunk_text(content)


def chunk_text(content: object) -> str:
    """Extract text from a streamed or block-style content object."""
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def reasoning_text(message_chunk: object) -> str:
    """Extract reasoning text from LangChain standard content blocks."""
    block_parts: list[str] = []
    for block in content_blocks(message_chunk):
        if isinstance(block, dict):
            block_parts.extend(reasoning_from_mapping(block))

    return "".join(block_parts)


def content_blocks(message_chunk: object) -> list[object]:
    """Return content blocks from current LangChain message variants."""
    content_blocks_value = getattr(message_chunk, "content_blocks", None)
    if isinstance(content_blocks_value, list):
        return content_blocks_value

    content = getattr(message_chunk, "content", None)
    if isinstance(content, list):
        return content

    return []


def has_reasoning_block(message_chunk: object) -> bool:
    """Return True when a message includes a reasoning block."""
    return any(
        isinstance(block, dict) and block.get("type") == "reasoning"
        for block in content_blocks(message_chunk)
    )


def reasoning_from_mapping(mapping: dict) -> list[str]:
    """Extract reasoning strings from a content block mapping."""
    if mapping.get("type") != "reasoning":
        return []

    parts: list[str] = []
    for key in ("reasoning", "text", "content", "summary", "details"):
        parts.extend(reasoning_from_value(mapping.get(key)))

    return parts


def reasoning_from_value(value: object) -> list[str]:
    """Extract reasoning strings from nested content block values."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.extend(reasoning_from_mapping(item))
            else:
                parts.extend(reasoning_from_value(item))
        return parts
    if isinstance(value, dict):
        return reasoning_from_mapping(value)
    return []


def write_conversation_trace_dump(
    *,
    thread_id: str,
    turns: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    trace_dir: Path | None = None,
) -> Path:
    """Write the full multi-turn web trace to a local JSON file."""
    trace_dir = trace_dir or TRACE_DUMP_DIR
    trace_dir.mkdir(parents=True, exist_ok=True)
    output_path = trace_dump_path(thread_id, trace_dir=trace_dir)
    merged_turns = merge_trace_events_into_turns(turns, events or [])
    payload = {
        "thread_id": thread_id,
        "turn_count": len(merged_turns),
        "dumped_at": utc_timestamp(),
        "turns": merged_turns,
        "events": events or [],
    }
    output_path.write_text(
        json.dumps(
            payload,
            default=json_default,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_path


def merge_trace_events_into_turns(
    turns: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach ordered low-level call events to their owning conversation turns."""
    merged_turns = deepcopy(turns)
    turns_by_id = {
        str(turn.get("turn_id")): turn
        for turn in merged_turns
        if turn.get("turn_id") is not None
    }

    fallback_turn = merged_turns[-1] if len(merged_turns) == 1 else None
    call_indexes: dict[int, int] = {}
    for event in events:
        turn = turn_for_event(event, turns_by_id, fallback_turn)
        if turn is None:
            continue

        turn_id = id(turn)
        call_index = call_indexes.get(turn_id, 0)
        call_indexes[turn_id] = call_index + 1
        turn.setdefault("calls", []).append(call_trace(event, call_index))

    for turn in merged_turns:
        turn.setdefault("calls", [])
        turn.setdefault("stream_chunks", [])

    return merged_turns


def build_trace_tree(trace: dict[str, Any]) -> dict[str, Any]:
    """Build a parent-child trace tree for browser inspection."""
    turns = trace.get("turns", [])
    if not isinstance(turns, list):
        turns = []

    return {
        "id": str(trace.get("thread_id", "trace")),
        "type": "session",
        "label": f"Session {trace.get('thread_id', '-')}",
        "status": _trace_tree_status(turns),
        "meta": {
            "thread_id": trace.get("thread_id"),
            "turn_count": trace.get("turn_count", len(turns)),
            "event_count": trace.get("event_count", len(trace.get("events", []))),
        },
        "children": [_trace_turn_node(index, turn) for index, turn in enumerate(turns)],
    }


def _trace_tree_status(turns: list[object]) -> str:
    for turn in turns:
        if isinstance(turn, dict) and turn.get("status") == "error":
            return "error"
    return "success" if turns else "ready"


def _trace_turn_node(index: int, turn: object) -> dict[str, Any]:
    if not isinstance(turn, dict):
        return {
            "id": f"turn-{index}",
            "type": "turn",
            "label": f"Turn {index + 1}",
            "status": "unknown",
            "meta": {"raw": json_safe(turn)},
            "children": [],
        }

    user_input = str(turn.get("user_input", ""))
    preview = _preview(user_input, fallback="empty input")
    return {
        "id": str(turn.get("turn_id") or f"turn-{index}"),
        "type": "turn",
        "label": f"Turn {index + 1}: {preview}",
        "status": str(turn.get("status", "unknown")),
        "meta": {
            "turn_index": turn.get("turn_index", index),
            "turn_id": turn.get("turn_id"),
            "status": turn.get("status"),
            "started_at": turn.get("started_at"),
            "ended_at": turn.get("ended_at"),
            "user_input": turn.get("user_input"),
            "assistant_output_preview": _preview(
                str(turn.get("assistant_output") or ""),
                fallback="empty output",
            ),
            "assistant_output_chars": turn.get("assistant_output_chars"),
            "assistant_chunk_count": turn.get("assistant_chunk_count"),
            "reasoning_block_seen": turn.get("reasoning_block_seen"),
            "reasoning_text_seen": turn.get("reasoning_text_seen"),
            "empty_visible_output": turn.get("empty_visible_output"),
            "malformed_tool_call_text_seen": turn.get("malformed_tool_call_text_seen"),
            "error_type": turn.get("error_type"),
            "trace_path": turn.get("trace_path"),
        },
        "children": _trace_react_stage_nodes(turn),
    }


def _trace_react_stage_nodes(turn: dict[str, Any]) -> list[dict[str, Any]]:
    """Group one turn into ReAct-oriented stages for browser inspection."""
    turn_id = str(turn.get("turn_id") or f"turn-{turn.get('turn_index', 0)}")
    calls = turn.get("calls", [])
    if not isinstance(calls, list):
        calls = []

    children: list[dict[str, Any]] = [
        {
            "id": f"{turn_id}:user-input",
            "type": "react_input",
            "label": "User Input",
            "status": "completed",
            "meta": {
                "user_input": turn.get("user_input"),
                "calls": _json_safe_calls(
                    calls,
                    event_names={"conversation_turn_start"},
                ),
            },
            "children": [
                _trace_event_node(call)
                for call in calls
                if isinstance(call, dict)
                and call.get("event") == "conversation_turn_start"
            ],
        }
    ]

    agent_calls = [
        call
        for call in calls
        if isinstance(call, dict) and call.get("type") == "agent_run"
    ]
    if agent_calls:
        children.append(
            {
                "id": f"{turn_id}:agent-run",
                "type": "react_agent",
                "label": "Agent Run",
                "status": _stage_status(agent_calls),
                "meta": {
                    "event_count": len(agent_calls),
                    "calls": json_safe(agent_calls),
                },
                "children": [_trace_event_node(call) for call in agent_calls],
            }
        )

    children.extend(_trace_react_step_nodes(turn_id, calls))

    assistant_output = turn.get("assistant_output") or turn.get(
        "partial_assistant_output"
    )
    children.append(
        {
            "id": f"{turn_id}:final-response",
            "type": "react_final",
            "label": "Final Response",
            "status": str(turn.get("status", "unknown")),
            "meta": {
                "assistant_output": assistant_output,
                "reasoning_block_seen": turn.get("reasoning_block_seen"),
                "reasoning_text_seen": turn.get("reasoning_text_seen"),
                "error_type": turn.get("error_type"),
                "calls": _json_safe_calls(
                    calls,
                    event_names={"conversation_turn_end", "conversation_turn_error"},
                ),
            },
            "children": [
                _trace_event_node(call)
                for call in calls
                if isinstance(call, dict)
                and call.get("event")
                in {"conversation_turn_end", "conversation_turn_error"}
            ],
        }
    )
    children.append(
        {
            "id": f"{turn_id}:raw-call-timeline",
            "type": "raw_trace",
            "label": "Raw Debug",
            "status": "info",
            "meta": {
                "event_count": len(calls),
                "raw_turn": json_safe(turn),
                "calls": json_safe(calls),
            },
            "children": _trace_call_nodes(calls),
        }
    )
    return children


def _trace_react_step_nodes(
    turn_id: str,
    calls: list[object],
) -> list[dict[str, Any]]:
    """Group model and tool events into sequential ReAct steps."""
    steps: list[dict[str, Any]] = []
    current_step: dict[str, Any] | None = None
    current_model_calls: list[dict[str, Any]] = []
    current_tool_calls: list[dict[str, Any]] = []

    def new_step(step_number: int) -> dict[str, Any]:
        return {
            "id": f"{turn_id}:react-step-{step_number}",
            "type": "react_step",
            "label": f"ReAct Step {step_number}",
            "status": "started",
            "meta": {"step_index": step_number - 1},
            "_calls": [],
            "children": [],
        }

    def flush_step() -> None:
        nonlocal current_step, current_model_calls, current_tool_calls
        if current_step is None:
            return
        current_step["children"] = []
        if current_model_calls:
            current_step["children"].append(
                _react_stage_node(
                    current_step["id"],
                    "thought",
                    "Thought / Model Call",
                    current_model_calls,
                )
            )
        if current_tool_calls:
            current_step["children"].append(
                _react_stage_node(
                    current_step["id"],
                    "action",
                    "Action / Tool Call",
                    current_tool_calls,
                )
            )
        steps.append(current_step)
        current_step = None
        current_model_calls = []
        current_tool_calls = []

    for call in calls:
        if not isinstance(call, dict):
            continue
        call_type_value = call.get("type")
        if call.get("event") == "react_context_budget_compacted":
            flush_step()
            steps.append(_react_compaction_node(turn_id, len(steps) + 1, call))
            continue
        if call_type_value == "model" and call.get("event") == "model_call_start":
            flush_step()
            step_number = len(steps) + 1
            current_step = new_step(step_number)
            current_step["_calls"].append(call)
            current_model_calls.append(call)
            continue
        if call_type_value == "model":
            if current_step is None:
                step_number = len(steps) + 1
                current_step = new_step(step_number)
            current_step["_calls"].append(call)
            current_model_calls.append(call)
            continue
        if call_type_value == "tool":
            if current_step is None:
                step_number = len(steps) + 1
                current_step = new_step(step_number)
            current_step["_calls"].append(call)
            current_tool_calls.append(call)

    flush_step()
    for step in steps:
        child_statuses = [child.get("status") for child in step.get("children", [])]
        if "error" in child_statuses:
            step["status"] = "error"
        elif child_statuses and all(status == "completed" for status in child_statuses):
            step["status"] = "completed"
        else:
            step["status"] = "info"
        step["meta"]["stage_count"] = len(step.get("children", []))
        step["meta"]["event_count"] = len(step.pop("_calls", []))
    return steps


def _react_compaction_node(
    turn_id: str,
    step_number: int,
    call: dict[str, Any],
) -> dict[str, Any]:
    fields = call.get("fields") if isinstance(call.get("fields"), dict) else {}
    return {
        "id": f"{turn_id}:react-compaction-{step_number}",
        "type": "react_compaction",
        "label": "State-Preserving Context Compaction",
        "status": "completed",
        "meta": {
            "step_index": step_number - 1,
            "event_count": 1,
            "estimate_chars": fields.get("estimate_chars"),
            "threshold_chars": fields.get("threshold_chars"),
            "observation_count": fields.get("observation_count"),
            "preserved_observation_count": fields.get("preserved_observation_count"),
            "dropped_observation_count": fields.get("dropped_observation_count"),
            "preview_truncated_count": fields.get("preview_truncated_count"),
            "compacted_request_chars": fields.get("compacted_request_chars"),
            "compaction_mode": fields.get("compaction_mode"),
            "compacted_message_count": fields.get("compacted_message_count"),
            "compacted_tool_count": fields.get("compacted_tool_count"),
            "compacted_state_preview": fields.get("compacted_state_preview"),
            "compacted_state_preview_chars": fields.get("compacted_state_preview_chars"),
            "compacted_state_chars": fields.get("compacted_state_chars"),
            "compacted_state_sha256": fields.get("compacted_state_sha256"),
        },
        "children": [_trace_event_node(call)],
    }


def _react_stage_node(
    step_id: str,
    stage_name: str,
    label: str,
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": f"{step_id}:{stage_name}",
        "type": f"react_{stage_name}",
        "label": label,
        "status": _stage_status(calls),
        "meta": {
            "event_count": len(calls),
            "calls": json_safe(calls),
        },
        "children": _trace_call_nodes(calls),
    }


def _stage_status(calls: list[dict[str, Any]]) -> str:
    if any(str(call.get("event", "")).endswith("_error") for call in calls):
        return "error"
    if calls and str(calls[-1].get("event", "")).endswith("_end"):
        return "completed"
    if calls and str(calls[-1].get("event", "")).endswith("_start"):
        return "started"
    return "info" if calls else "empty"


def _trace_call_nodes(calls: object) -> list[dict[str, Any]]:
    if not isinstance(calls, list):
        return []

    nodes: list[dict[str, Any]] = []
    tool_nodes_by_id: dict[str, dict[str, Any]] = {}

    for call in calls:
        if not isinstance(call, dict):
            continue
        if call.get("type") == "tool" and call.get("tool_call_id"):
            node = _merge_tool_call_node(tool_nodes_by_id, call)
            if node not in nodes:
                nodes.append(node)
            continue
        nodes.append(_trace_event_node(call))

    return nodes


def _json_safe_calls(
    calls: list[object],
    *,
    event_names: set[str],
) -> list[object]:
    return json_safe(
        [
            call
            for call in calls
            if isinstance(call, dict) and str(call.get("event")) in event_names
        ]
    )


def _merge_tool_call_node(
    tool_nodes_by_id: dict[str, dict[str, Any]],
    call: dict[str, Any],
) -> dict[str, Any]:
    tool_call_id = str(call.get("tool_call_id"))
    node = tool_nodes_by_id.setdefault(
        tool_call_id,
        {
            "id": tool_call_id,
            "type": "tool",
            "label": f"Tool: {call.get('tool_name', '-')}",
            "status": "started",
            "meta": {
                "tool_call_id": call.get("tool_call_id"),
                "tool_name": call.get("tool_name"),
                "index": call.get("index"),
            },
            "children": [],
        },
    )

    node["meta"]["index"] = min(
        int(node["meta"].get("index") or call.get("index") or 0),
        int(call.get("index") or node["meta"].get("index") or 0),
    )
    node["meta"]["tool_name"] = node["meta"].get("tool_name") or call.get("tool_name")
    if "request" in call:
        node["meta"]["request"] = call["request"]
    if "response" in call:
        node["meta"]["response"] = call["response"]
    if call.get("event") == "tool_call_end":
        node["status"] = "completed"
    elif call.get("event") == "tool_call_error":
        node["status"] = "error"

    node["children"].append(_trace_event_node(call))
    return node


def _trace_event_node(call: dict[str, Any]) -> dict[str, Any]:
    event_name = str(call.get("event", "event"))
    node_type = str(call.get("type", "event"))
    return {
        "id": f"{event_name}-{call.get('index', len(str(call)))}",
        "type": node_type,
        "label": _trace_event_label(call),
        "status": _trace_event_status(call),
        "meta": _trace_event_meta(call),
        "children": [],
    }


def _trace_event_label(call: dict[str, Any]) -> str:
    event_name = str(call.get("event", "event"))
    if call.get("type") == "tool":
        return f"{event_name}: {call.get('tool_name', '-')}"
    if call.get("type") == "model":
        return event_name.replace("_", " ").title()
    if call.get("type") == "agent_run":
        return event_name.replace("_", " ").title()
    if call.get("type") == "conversation":
        return event_name.replace("_", " ").title()
    return event_name


def _trace_event_status(call: dict[str, Any]) -> str:
    event_name = str(call.get("event", ""))
    if event_name.endswith("_error"):
        return "error"
    if event_name.endswith("_end"):
        return "completed"
    if event_name.endswith("_start"):
        return "started"
    return "info"


def _trace_event_meta(call: dict[str, Any]) -> dict[str, Any]:
    meta = {
        "index": call.get("index"),
        "event": call.get("event"),
        "level": call.get("level"),
        "tool_call_id": call.get("tool_call_id"),
        "tool_name": call.get("tool_name"),
    }
    if "request" in call:
        meta["request"] = call["request"]
    if "response" in call:
        meta["response"] = call["response"]
    if "fields" in call:
        meta["fields"] = call["fields"]
    return meta


def _preview(value: str, *, fallback: str, limit: int = 42) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        return fallback
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def turn_for_event(
    event: dict[str, Any],
    turns_by_id: dict[str, dict[str, Any]],
    fallback_turn: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Find the conversation turn that owns an event."""
    fields = event.get("fields")
    if isinstance(fields, dict):
        turn_id = fields.get("turn_id")
        if turn_id is not None and str(turn_id) in turns_by_id:
            return turns_by_id[str(turn_id)]
    return fallback_turn


def call_trace(event: dict[str, Any], index: int) -> dict[str, Any]:
    """Render one observability event as a compact call trace item."""
    fields = event.get("fields") if isinstance(event.get("fields"), dict) else {}
    event_name = str(event.get("event", ""))
    call: dict[str, Any] = {
        "index": index,
        "type": call_type(event_name),
        "event": event_name,
        "level": str(event.get("level", "INFO")),
    }

    if event_name == "model_call_start" and "request_trace" in fields:
        call["request"] = fields["request_trace"]
    if event_name == "model_call_end" and "response_trace" in fields:
        call["response"] = fields["response_trace"]
    if event_name == "tool_call_start" and "tool_call" in fields:
        call["tool_call_id"] = fields.get("tool_call_id")
        call["tool_name"] = fields.get("tool_name")
        call["request"] = fields["tool_call"]
    if event_name in {"tool_call_end", "tool_call_error"}:
        call["tool_call_id"] = fields.get("tool_call_id")
        call["tool_name"] = fields.get("tool_name")
    if event_name == "tool_call_end" and "response_trace" in fields:
        call["response"] = fields["response_trace"]

    call["fields"] = fields
    return call


def call_type(event_name: str) -> str:
    """Classify an observability event for the debug UI."""
    if event_name.startswith("model_call"):
        return "model"
    if event_name.startswith("tool_call"):
        return "tool"
    if event_name.startswith("agent_run"):
        return "agent_run"
    if event_name.startswith("conversation_turn"):
        return "conversation"
    return "event"


def serialize_agent_input(agent_input: dict[str, Any]) -> dict[str, Any]:
    """Serialize agent input messages for JSON trace dumps."""
    return {
        "messages": [
            {
                "message_type": type(message).__name__,
                "role": str(getattr(message, "type", "unknown")),
                "content": getattr(message, "content", None),
            }
            for message in agent_input.get("messages", [])
        ]
    }


def json_safe(value: object) -> object:
    """Return a JSON-friendly representation without losing simple structures."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if hasattr(value, "content"):
        return serialize_message(value)
    return str(value)


def trace_dump_path(thread_id: str, *, trace_dir: Path | None = None) -> Path:
    """Return the JSON trace dump path for a web thread."""
    trace_dir = trace_dir or TRACE_DUMP_DIR
    return trace_dir / f"{thread_id}.json"


def utc_timestamp() -> str:
    """Return the current UTC timestamp as an ISO string."""
    return datetime.now(UTC).isoformat()


def json_default(value: object) -> str:
    """Fallback JSON serializer for trace dumps."""
    return str(value)


__all__ = [
    "TRACE_DUMP_DIR",
    "call_trace",
    "call_type",
    "build_trace_tree",
    "content_blocks",
    "has_reasoning_block",
    "is_assistant_message",
    "json_safe",
    "latest_assistant_message",
    "merge_trace_events_into_turns",
    "message_text",
    "reasoning_text",
    "serialize_agent_input",
    "serialize_invoke_output",
    "serialize_message",
    "trace_dump_path",
    "utc_timestamp",
    "write_conversation_trace_dump",
]
