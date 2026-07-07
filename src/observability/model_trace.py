"""Shared model request and message trace helpers."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from langchain.agents.middleware import ModelRequest


def model_request_trace(request: ModelRequest) -> dict[str, Any]:
    """Return the complete model request trace used by logs and guardrails."""
    return {
        "system_prompt": request.system_prompt,
        "messages": message_trace(request.messages),
        "tools": [tool_definition_trace(tool) for tool in request.tools],
    }


def model_request_trace_chars(request: ModelRequest) -> int:
    """Return the JSON character size of a model request trace."""
    return len(
        json.dumps(
            model_request_trace(request),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    )


def message_trace(messages: Any) -> list[dict[str, Any]]:
    """Return serializable traces for LangChain messages."""
    return [_single_message_trace(index, message) for index, message in enumerate(messages)]


def tool_call_trace(tool_call: Any) -> dict[str, Any]:
    """Return a serializable tool-call trace."""
    if not isinstance(tool_call, dict):
        return {"tool_call_type": type(tool_call).__name__}

    args = tool_call.get("args")
    return {
        "id": str(tool_call.get("id") or ""),
        "name": str(tool_call.get("name") or ""),
        "args": args,
        "argument_keys": sorted(str(key) for key in args) if isinstance(args, dict) else [],
    }


def tool_response_trace(response: Any) -> dict[str, Any]:
    """Return a serializable tool response trace."""
    trace = {
        "message_type": type(response).__name__,
        "status": str(getattr(response, "status", "success")),
        "content": getattr(response, "content", None),
    }
    trace.update(_content_trace(getattr(response, "content", None)))
    return trace


def tool_definition_trace(tool: Any) -> dict[str, Any]:
    """Return a complete tool definition trace for context-size estimation."""
    if isinstance(tool, dict):
        return tool

    trace = {
        "tool_type": type(tool).__name__,
        "name": str(getattr(tool, "name", "")),
        "description": str(getattr(tool, "description", "")),
    }
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        trace["args_schema"] = _schema_trace(args_schema)
    args = getattr(tool, "args", None)
    if args is not None:
        trace["args"] = args
    return trace


def _single_message_trace(index: int, message: Any) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "index": index,
        "message_type": type(message).__name__,
        "role": str(getattr(message, "type", "unknown")),
        "content": getattr(message, "content", None),
    }
    trace.update(_content_trace(getattr(message, "content", None)))

    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        trace["tool_calls"] = [tool_call_trace(tool_call) for tool_call in tool_calls]

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        trace["tool_call_id"] = str(tool_call_id)

    name = getattr(message, "name", None)
    if name:
        trace["name"] = str(name)

    return trace


def _schema_trace(schema: Any) -> Any:
    try:
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        if hasattr(schema, "schema"):
            return schema.schema()
        return str(schema)
    except Exception as exc:
        return {
            "schema_type": type(schema).__name__,
            "schema_repr": str(schema),
            "schema_error_type": type(exc).__name__,
            "schema_error": str(exc),
        }


def _content_trace(content: Any) -> dict[str, Any]:
    text = _content_text(content)
    encoded = text.encode("utf-8")
    trace: dict[str, Any] = {
        "content_type": type(content).__name__,
        "content_chars": len(text),
        "content_bytes": len(encoded),
        "content_sha256": sha256(encoded).hexdigest(),
    }
    if isinstance(content, list):
        trace["content_block_count"] = len(content)
        trace["content_block_types"] = [
            str(block.get("type", "unknown")) if isinstance(block, dict) else type(block).__name__
            for block in content
        ]
    return trace


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                for key in ("text", "content", "reasoning", "summary"):
                    parts.extend(_text_parts(block.get(key)))
            else:
                parts.extend(_text_parts(block))
        return "".join(parts)
    return "" if content is None else str(content)


def _text_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_text_parts(item))
        return parts
    if isinstance(value, dict):
        try:
            return [json.dumps(value, ensure_ascii=False, sort_keys=True)]
        except TypeError:
            return [str(value)]
    return [] if value is None else [str(value)]


__all__ = [
    "message_trace",
    "model_request_trace",
    "model_request_trace_chars",
    "tool_call_trace",
    "tool_definition_trace",
    "tool_response_trace",
]
