"""Interactive CLI entrypoint for the air ticket agent."""

import json
from copy import deepcopy
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain.messages import HumanMessage
from langchain_core.messages import AIMessage, AIMessageChunk

from src.agent import agent
from src.observability import (
    collect_trace_events,
    full_text_trace_fields,
    log_event,
    observe_agent_run,
)
from src.runtime import Context, build_default_context
from src.tools import get_tools

APP_USER_ID = "local-cli"
DEFAULT_WORKSPACE_ID = "local-cli"
TRACE_DUMP_DIR = Path("logs/traces")
HELP_TEXT = """可用命令：
/help   显示帮助
/new    开始一个新会话
/tools  查看当前已注册工具
/demo   运行一条机票报价示例
/exit   退出

直接输入问题即可调用机票事实查询 agent，例如：
北京到上海 2026-07-10 的机票大概多少钱？
CA981 这个航班有什么信息？"""
DEMO_MESSAGE = (
    "请查询北京到上海在 2026-07-10 的机票报价样本，"
    "并说明查到的事实、数据来源、查询时间和数据限制。"
)


@dataclass
class CliSession:
    """Current CLI conversation session."""

    thread_id: str
    turns: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def new(cls) -> "CliSession":
        return cls(thread_id=_new_thread_id())

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def context(self) -> Context:
        return build_default_context(
            user_id=APP_USER_ID,
            thread_id=self.thread_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            metadata={"entrypoint": "main.py"},
        )


def run_cli() -> None:
    """Run the interactive command-line agent shell."""
    session = CliSession.new()
    print("机票事实查询 Agent CLI。输入 /help 查看命令。")
    print(f"当前会话: {session.thread_id}")

    while True:
        try:
            raw_input = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        message = raw_input.strip()
        if not message:
            continue

        command_result = handle_command(message, session)
        if command_result == "exit":
            print("再见。")
            return
        if isinstance(command_result, CliSession):
            session = command_result
            print(f"已开始新会话: {session.thread_id}")
            continue
        if isinstance(command_result, str):
            print(command_result)
            continue

        stream_agent_response(message, session)


def handle_command(message: str, session: CliSession) -> str | CliSession | None:
    """Handle slash commands; return None when input should go to the agent."""
    command = message.lower()
    if command in {"/exit", "/quit", "exit", "quit"}:
        return "exit"
    if command == "/help":
        return HELP_TEXT
    if command == "/new":
        return CliSession.new()
    if command == "/tools":
        return format_tools()
    if command == "/demo":
        stream_agent_response(DEMO_MESSAGE, session)
        return ""
    if message.startswith("/"):
        return f"未知命令: {message}\n输入 /help 查看可用命令。"
    return None


def stream_agent_response(message: str, session: CliSession) -> None:
    """Run one user message through the agent and print a compact transcript."""
    with collect_trace_events(trace_id=session.thread_id) as trace_events:
        _stream_agent_response_with_trace(message, session, trace_events)


def _stream_agent_response_with_trace(
    message: str,
    session: CliSession,
    trace_events: list[dict[str, Any]],
) -> None:
    """Invoke one user message while collecting all structured trace events."""
    context = session.context()
    agent_input = {"messages": [HumanMessage(content=message)]}
    agent_config = session.config
    answer_started = False
    saw_reasoning = False
    saw_reasoning_block = False
    assistant_parts: list[str] = []
    assistant_chunk_count = 0
    turn_trace: dict[str, Any] = {
        "turn_index": len(session.turns),
        "status": "running",
        "started_at": _utc_timestamp(),
        "turn_id": context.request_id,
        "context": context.__dict__,
        "agent_input": _serialize_agent_input(agent_input),
        "agent_config": agent_config,
        "user_input": message,
        **full_text_trace_fields("user_input", message),
        "assistant_chunks": [],
        "reasoning_chunks": [],
        "stream_chunks": [],
        "invoke_output": None,
    }

    log_event(
        "conversation_turn_start",
        context=context,
        redact=False,
        entrypoint="main.stream_agent_response",
        **full_text_trace_fields("user_input", message),
    )

    try:
        with observe_agent_run(
            context,
            entrypoint="main.stream_agent_response",
            stream_mode="invoke",
        ):
            invoke_output = agent.invoke(
                agent_input,
                config=agent_config,
                context=context,
            )

        turn_trace["invoke_output"] = _serialize_invoke_output(invoke_output)
        assistant_message = _latest_assistant_message(invoke_output)
        if assistant_message is not None:
            saw_reasoning_block = _has_reasoning_block(assistant_message)
            reasoning = _reasoning_text(assistant_message)
            if reasoning:
                turn_trace["reasoning_chunks"].append(reasoning)
                print("思考流:")
                print(reasoning)
                saw_reasoning = True
            elif saw_reasoning_block:
                print("思考流: 检测到 reasoning block，但未暴露文本。\n")

            content = _message_text(assistant_message)
            if content:
                assistant_parts.append(content)
                turn_trace["assistant_chunks"].append(content)
                assistant_chunk_count = 1
                print("回复:")
                print(content)
                answer_started = True
        else:
            turn_trace["limitations"] = [
                "agent.invoke returned no assistant message to render."
            ]
            print("回复:")
            print("未获取到可展示的助手回复。")
            answer_started = True
    except Exception as exc:
        partial_output = "".join(assistant_parts)
        turn_trace.update(
            {
                "status": "error",
                "ended_at": _utc_timestamp(),
                "assistant_chunk_count": assistant_chunk_count,
                "reasoning_block_seen": saw_reasoning_block,
                "reasoning_text_seen": saw_reasoning,
                "error_type": type(exc).__name__,
                "partial_assistant_output": partial_output,
                **full_text_trace_fields(
                    "partial_assistant_output",
                    partial_output,
                ),
            }
        )
        session.turns.append(turn_trace)
        trace_path = trace_dump_path(session.thread_id)
        log_event(
            "conversation_turn_error",
            context=context,
            redact=False,
            entrypoint="main.stream_agent_response",
            assistant_chunk_count=assistant_chunk_count,
            reasoning_block_seen=saw_reasoning_block,
            reasoning_text_seen=saw_reasoning,
            error_type=type(exc).__name__,
            trace_dump_path=str(trace_path),
            **full_text_trace_fields("partial_assistant_output", partial_output),
        )
        session.events.extend(trace_events)
        write_conversation_trace_dump(
            thread_id=session.thread_id,
            turns=session.turns,
            events=session.events,
        )
        raise

    assistant_output = "".join(assistant_parts)
    turn_trace.update(
        {
            "status": "success",
            "ended_at": _utc_timestamp(),
            "assistant_chunk_count": assistant_chunk_count,
            "reasoning_block_seen": saw_reasoning_block,
            "reasoning_text_seen": saw_reasoning,
            "answer_started": answer_started,
            "assistant_output": assistant_output,
            **full_text_trace_fields("assistant_output", assistant_output),
        }
    )
    session.turns.append(turn_trace)
    trace_path = trace_dump_path(session.thread_id)
    log_event(
        "conversation_turn_end",
        context=context,
        redact=False,
        entrypoint="main.stream_agent_response",
        assistant_chunk_count=assistant_chunk_count,
        reasoning_block_seen=saw_reasoning_block,
        reasoning_text_seen=saw_reasoning,
        answer_started=answer_started,
        trace_dump_path=str(trace_path),
        **full_text_trace_fields("assistant_output", assistant_output),
    )
    session.events.extend(trace_events)
    write_conversation_trace_dump(
        thread_id=session.thread_id,
        turns=session.turns,
        events=session.events,
    )


def format_tools() -> str:
    """Render registered tools for CLI display."""
    lines = ["当前已注册工具："]
    for tool in get_tools():
        description = str(getattr(tool, "description", "")).strip()
        lines.append(f"- {tool.name}: {description}")
    return "\n".join(lines)


def _is_assistant_chunk(message_chunk: object) -> bool:
    """Return True for assistant message chunks that are safe to print."""
    return isinstance(message_chunk, (AIMessage, AIMessageChunk))


def _latest_assistant_message(invoke_output: object) -> object | None:
    """Return the last assistant message from an agent.invoke result."""
    if _is_assistant_chunk(invoke_output):
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
        if _is_assistant_chunk(message):
            return message
    return None


def _serialize_invoke_output(invoke_output: object) -> object:
    """Serialize an agent.invoke result for local JSON trace debugging."""
    if isinstance(invoke_output, dict):
        payload: dict[str, Any] = {"output_type": "dict"}
        for key, value in invoke_output.items():
            if key == "messages" and isinstance(value, list):
                payload[key] = [_serialize_message(message) for message in value]
            else:
                payload[key] = _json_safe(value)
        return payload

    if isinstance(invoke_output, list):
        return {
            "output_type": "list",
            "items": [_json_safe(item) for item in invoke_output],
        }

    if _is_assistant_chunk(invoke_output) or hasattr(invoke_output, "content"):
        return _serialize_message(invoke_output)

    return _json_safe(invoke_output)


def _serialize_message(message: object) -> dict[str, Any]:
    """Serialize a LangChain message-like object for local trace dumps."""
    trace: dict[str, Any] = {
        "message_type": type(message).__name__,
        "role": str(getattr(message, "type", "unknown")),
        "content": getattr(message, "content", None),
    }

    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        trace["tool_calls"] = _json_safe(tool_calls)

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        trace["tool_call_id"] = str(tool_call_id)

    name = getattr(message, "name", None)
    if name:
        trace["name"] = str(name)

    return trace


def _message_text(message_chunk: object) -> str:
    content = getattr(message_chunk, "content", message_chunk)
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in _content_blocks(message_chunk):
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))

    if parts:
        return "".join(parts)

    return _chunk_text(content)


def _chunk_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _reasoning_text(message_chunk: object) -> str:
    """Extract reasoning text from LangChain standard content blocks."""
    block_parts: list[str] = []
    for block in _content_blocks(message_chunk):
        if isinstance(block, dict):
            block_parts.extend(_reasoning_from_mapping(block))

    return "".join(block_parts)


def _content_blocks(message_chunk: object) -> list[object]:
    content_blocks = getattr(message_chunk, "content_blocks", None)
    if isinstance(content_blocks, list):
        return content_blocks

    content = getattr(message_chunk, "content", None)
    if isinstance(content, list):
        return content

    return []


def _has_reasoning_block(message_chunk: object) -> bool:
    return any(
        isinstance(block, dict) and block.get("type") == "reasoning"
        for block in _content_blocks(message_chunk)
    )


def _reasoning_from_mapping(mapping: dict) -> list[str]:
    if mapping.get("type") != "reasoning":
        return []

    parts: list[str] = []
    for key in ("reasoning", "text", "content", "summary", "details"):
        parts.extend(_reasoning_from_value(mapping.get(key)))

    return parts


def _reasoning_from_value(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.extend(_reasoning_from_mapping(item))
            else:
                parts.extend(_reasoning_from_value(item))
        return parts
    if isinstance(value, dict):
        return _reasoning_from_mapping(value)
    return []


def _new_thread_id() -> str:
    return f"cli-{uuid4().hex[:8]}"


def write_conversation_trace_dump(
    *,
    thread_id: str,
    turns: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    trace_dir: Path | None = None,
) -> Path:
    """Write the full multi-turn CLI trace to a local JSON file."""
    trace_dir = trace_dir or TRACE_DUMP_DIR
    trace_dir.mkdir(parents=True, exist_ok=True)
    output_path = trace_dump_path(thread_id, trace_dir=trace_dir)
    merged_turns = merge_trace_events_into_turns(turns, events or [])
    payload = {
        "thread_id": thread_id,
        "turn_count": len(merged_turns),
        "dumped_at": _utc_timestamp(),
        "turns": merged_turns,
        "events": events or [],
    }
    output_path.write_text(
        json.dumps(
            payload,
            default=_json_default,
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
        turn = _turn_for_event(event, turns_by_id, fallback_turn)
        if turn is None:
            continue

        turn_id = id(turn)
        call_index = call_indexes.get(turn_id, 0)
        call_indexes[turn_id] = call_index + 1
        turn.setdefault("calls", []).append(_call_trace(event, call_index))

    for turn in merged_turns:
        turn.setdefault("calls", [])
        turn.setdefault("stream_chunks", [])

    return merged_turns


def _turn_for_event(
    event: dict[str, Any],
    turns_by_id: dict[str, dict[str, Any]],
    fallback_turn: dict[str, Any] | None,
) -> dict[str, Any] | None:
    fields = event.get("fields")
    if isinstance(fields, dict):
        turn_id = fields.get("turn_id")
        if turn_id is not None and str(turn_id) in turns_by_id:
            return turns_by_id[str(turn_id)]
    return fallback_turn


def _call_trace(event: dict[str, Any], index: int) -> dict[str, Any]:
    fields = event.get("fields") if isinstance(event.get("fields"), dict) else {}
    event_name = str(event.get("event", ""))
    call: dict[str, Any] = {
        "index": index,
        "type": _call_type(event_name),
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


def _call_type(event_name: str) -> str:
    if event_name.startswith("model_call"):
        return "model"
    if event_name.startswith("tool_call"):
        return "tool"
    if event_name.startswith("agent_run"):
        return "agent_run"
    if event_name.startswith("conversation_turn"):
        return "conversation"
    return "event"


def _stream_chunk_trace(
    *,
    index: int,
    message_chunk: object,
    metadata: object,
) -> dict[str, Any]:
    return {
        "index": index,
        "message_type": type(message_chunk).__name__,
        "content": getattr(message_chunk, "content", None),
        "text": _message_text(message_chunk),
        "reasoning": _reasoning_text(message_chunk),
        "metadata": metadata,
    }


def _serialize_agent_input(agent_input: dict[str, Any]) -> dict[str, Any]:
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


def _json_safe(value: object) -> object:
    """Return a JSON-friendly representation without losing simple structures."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "content"):
        return _serialize_message(value)
    return str(value)


def trace_dump_path(thread_id: str, *, trace_dir: Path | None = None) -> Path:
    """Return the JSON trace dump path for a CLI thread."""
    trace_dir = trace_dir or TRACE_DUMP_DIR
    return trace_dir / f"{thread_id}.json"


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _json_default(value: object) -> str:
    return str(value)


if __name__ == "__main__":
    run_cli()
