import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from main import (
    DEMO_MESSAGE,
    HELP_TEXT,
    CliSession,
    format_tools,
    handle_command,
    merge_trace_events_into_turns,
    stream_agent_response,
    write_conversation_trace_dump,
    _has_reasoning_block,
    _is_assistant_chunk,
    _message_text,
    _reasoning_text,
)


def test_cli_help_describes_basic_commands():
    assert "/help" in HELP_TEXT
    assert "/new" in HELP_TEXT
    assert "/tools" in HELP_TEXT
    assert "/exit" in HELP_TEXT


def test_cli_new_command_starts_new_session():
    session = CliSession(thread_id="cli-old")

    new_session = handle_command("/new", session)

    assert isinstance(new_session, CliSession)
    assert new_session.thread_id.startswith("cli-")
    assert new_session.thread_id != session.thread_id


def test_cli_commands_handle_exit_help_tools_and_unknown():
    session = CliSession(thread_id="cli-test")

    assert handle_command("/exit", session) == "exit"
    assert handle_command("/help", session) == HELP_TEXT
    assert "resolve_flight_locations" in handle_command("/tools", session)
    assert "未知命令" in handle_command("/missing", session)
    assert handle_command("北京到上海多少钱", session) is None


def test_cli_demo_prompt_matches_air_ticket_mvp_tools():
    assert "北京到上海" in DEMO_MESSAGE
    assert "2026-07-10" in DEMO_MESSAGE
    assert "机票报价样本" in DEMO_MESSAGE
    assert "create_demo_task" not in DEMO_MESSAGE
    assert "inspect_runtime_context" not in DEMO_MESSAGE


def test_format_tools_lists_air_ticket_tools():
    rendered = format_tools()

    assert "resolve_flight_locations" in rendered
    assert "search_airfare_quotes" in rendered
    assert "query_flight_information" in rendered


def test_reasoning_text_reads_standard_content_blocks():
    chunk = AIMessageChunk(
        content=[
            {"type": "reasoning", "reasoning": "先检查工具。"},
            {"type": "text", "text": "demo 可以验证工具调用。"},
        ],
        response_metadata={"model_provider": "openai"},
    )

    assert _reasoning_text(chunk) == "先检查工具。"
    assert _has_reasoning_block(chunk)
    assert _message_text(chunk) == "demo 可以验证工具调用。"


def test_reasoning_text_reads_summary_blocks():
    chunk = AIMessageChunk(
        content=[
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "归纳步骤。"}],
            },
            {"type": "text", "text": "demo 可以验证流式输出。"},
        ],
        response_metadata={"model_provider": "openai"},
    )

    assert _reasoning_text(chunk) == "归纳步骤。"
    assert _has_reasoning_block(chunk)
    assert _message_text(chunk) == "demo 可以验证流式输出。"


def test_reasoning_block_can_exist_without_exposed_text():
    chunk = AIMessageChunk(
        content=[
            {
                "type": "reasoning",
                "extras": {"content": [], "status": "in_progress"},
            },
            {"type": "text", "text": "demo 可以验证流式输出。"},
        ],
        response_metadata={"model_provider": "openai"},
    )

    assert _has_reasoning_block(chunk)
    assert _reasoning_text(chunk) == ""
    assert _message_text(chunk) == "demo 可以验证流式输出。"


def test_cli_filters_tool_messages_from_rendered_answer():
    assert _is_assistant_chunk(AIMessageChunk(content="给用户看的回复"))
    assert not _is_assistant_chunk(
        ToolMessage(
            content='{"timezone":"Asia/Shanghai"}',
            name="query_current_date",
            tool_call_id="call-1",
        )
    )


def test_cli_logs_conversation_turn_trace(monkeypatch, capsys):
    class FakeAgent:
        def invoke(self, *args, **kwargs):
            return {"messages": [AIMessage(content="查到一些结果")]}

    events = []

    def fake_log_event(event, **kwargs):
        events.append((event, kwargs))

    monkeypatch.setattr("main.agent", FakeAgent())
    monkeypatch.setattr("main.log_event", fake_log_event)

    stream_agent_response("查询明天北京到上海", CliSession(thread_id="cli-test"))

    captured = capsys.readouterr()
    assert "查到一些结果" in captured.out
    assert [event for event, _ in events] == [
        "conversation_turn_start",
        "conversation_turn_end",
    ]
    assert events[0][1]["context"].thread_id == "cli-test"
    assert events[0][1]["user_input"] == "查询明天北京到上海"
    assert events[0][1]["user_input_chars"] == 9
    assert len(events[0][1]["user_input_sha256"]) == 64
    assert events[1][1]["assistant_chunk_count"] == 1
    assert events[1][1]["answer_started"] is True
    assert events[1][1]["assistant_output"] == "查到一些结果"
    assert len(events[1][1]["assistant_output_sha256"]) == 64


def test_cli_writes_multi_turn_json_trace_dump(monkeypatch, tmp_path, capsys):
    class FakeAgent:
        def invoke(self, *args, **kwargs):
            return {"messages": [AIMessage(content="第一轮回复")]}

    monkeypatch.setattr("main.agent", FakeAgent())
    monkeypatch.setattr("main.TRACE_DUMP_DIR", tmp_path / "traces")
    session = CliSession(thread_id="cli-test")

    stream_agent_response("第一轮问题", session)

    class SecondFakeAgent:
        def invoke(self, *args, **kwargs):
            return {"messages": [AIMessage(content="第二轮回复")]}

    monkeypatch.setattr("main.agent", SecondFakeAgent())

    stream_agent_response("第二轮问题", session)

    payload = json.loads((tmp_path / "traces" / "cli-test.json").read_text())

    assert payload["thread_id"] == "cli-test"
    assert payload["turn_count"] == 2
    assert payload["turns"][0]["user_input"] == "第一轮问题"
    assert payload["turns"][0]["assistant_output"] == "第一轮回复"
    assert payload["turns"][0]["status"] == "success"
    assert payload["turns"][0]["calls"][0]["type"] == "conversation"
    assert payload["turns"][0]["calls"][0]["event"] == "conversation_turn_start"
    assert payload["turns"][0]["calls"][1]["event"] == "agent_run_start"
    assert payload["turns"][0]["calls"][2]["type"] == "agent_run"
    assert payload["turns"][0]["calls"][2]["event"] == "agent_run_end"
    assert payload["turns"][0]["calls"][3]["type"] == "conversation"
    assert payload["turns"][0]["calls"][3]["event"] == "conversation_turn_end"
    assert payload["turns"][0]["stream_chunks"] == []
    assert payload["turns"][0]["invoke_output"]["messages"] == [
        {
            "content": "第一轮回复",
            "message_type": "AIMessage",
            "role": "ai",
        }
    ]
    assert payload["turns"][1]["user_input"] == "第二轮问题"
    assert payload["turns"][1]["assistant_output"] == "第二轮回复"
    assert payload["turns"][1]["status"] == "success"
    assert [event["event"] for event in payload["events"]] == [
        "conversation_turn_start",
        "agent_run_start",
        "agent_run_end",
        "conversation_turn_end",
        "conversation_turn_start",
        "agent_run_start",
        "agent_run_end",
        "conversation_turn_end",
    ]


def test_cli_logs_conversation_turn_error(monkeypatch):
    class BrokenAgent:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("boom")

    events = []

    def fake_log_event(event, **kwargs):
        events.append((event, kwargs))

    monkeypatch.setattr("main.agent", BrokenAgent())
    monkeypatch.setattr("main.log_event", fake_log_event)

    with pytest.raises(RuntimeError, match="boom"):
        stream_agent_response("查询明天北京到上海", CliSession(thread_id="cli-test"))

    assert [event for event, _ in events] == [
        "conversation_turn_start",
        "conversation_turn_error",
    ]
    assert events[1][1]["assistant_chunk_count"] == 0
    assert events[1][1]["error_type"] == "RuntimeError"
    assert events[1][1]["partial_assistant_output"] == ""
    assert len(events[1][1]["partial_assistant_output_sha256"]) == 64


def test_cli_writes_error_turn_to_json_trace_dump(monkeypatch, tmp_path):
    class BrokenAgent:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr("main.agent", BrokenAgent())
    monkeypatch.setattr("main.TRACE_DUMP_DIR", tmp_path / "traces")
    session = CliSession(thread_id="cli-error")

    with pytest.raises(RuntimeError, match="boom"):
        stream_agent_response("错误问题", session)

    payload = json.loads((tmp_path / "traces" / "cli-error.json").read_text())

    assert payload["thread_id"] == "cli-error"
    assert payload["turn_count"] == 1
    assert payload["turns"][0]["status"] == "error"
    assert payload["turns"][0]["user_input"] == "错误问题"
    assert payload["turns"][0]["partial_assistant_output"] == ""
    assert payload["turns"][0]["error_type"] == "RuntimeError"
    assert payload["turns"][0]["calls"][-1]["event"] == "conversation_turn_error"
    assert [event["event"] for event in payload["events"]] == [
        "conversation_turn_start",
        "agent_run_start",
        "agent_run_error",
        "conversation_turn_error",
    ]


def test_write_conversation_trace_dump_creates_json_file(tmp_path):
    output_path = write_conversation_trace_dump(
        thread_id="cli-direct",
        turns=[{"user_input": "你好", "assistant_output": "你好"}],
        trace_dir=tmp_path,
    )

    payload = json.loads(output_path.read_text())

    assert output_path == tmp_path / "cli-direct.json"
    assert payload["thread_id"] == "cli-direct"
    assert payload["turn_count"] == 1
    assert payload["turns"][0]["user_input"] == "你好"


def test_merge_trace_events_into_turns_builds_ordered_call_chain():
    turns = [
        {
            "turn_id": "request-1",
            "user_input": "北京到上海明天有航班吗",
            "stream_chunks": [],
        }
    ]
    events = [
        {
            "event": "conversation_turn_start",
            "level": "INFO",
            "fields": {"turn_id": "request-1", "user_input": "北京到上海明天有航班吗"},
        },
        {
            "event": "model_call_start",
            "level": "INFO",
            "fields": {
                "turn_id": "request-1",
                "request_trace": {
                    "system_prompt": "完整 prompt",
                    "messages": [{"role": "human", "content": "北京到上海明天有航班吗"}],
                },
            },
        },
        {
            "event": "model_call_end",
            "level": "INFO",
            "fields": {
                "turn_id": "request-1",
                "response_trace": [
                    {
                        "role": "ai",
                        "content": [
                            {
                                "type": "function_call",
                                "name": "resolve_flight_locations",
                                "arguments": '{"locations":["北京","上海"]}',
                            }
                        ],
                    }
                ],
            },
        },
        {
            "event": "tool_call_start",
            "level": "INFO",
            "fields": {
                "turn_id": "request-1",
                "tool_call_id": "call-1",
                "tool_name": "resolve_flight_locations",
                "tool_call": {
                    "name": "resolve_flight_locations",
                    "args": {"locations": ["北京", "上海"]},
                },
            },
        },
        {
            "event": "tool_call_end",
            "level": "INFO",
            "fields": {
                "turn_id": "request-1",
                "tool_call_id": "call-1",
                "tool_name": "resolve_flight_locations",
                "response_trace": {"content": '{"items":[{"input":"北京"}]}'},
            },
        },
    ]

    merged = merge_trace_events_into_turns(turns, events)

    assert merged[0]["calls"] == [
        {
            "index": 0,
            "type": "conversation",
            "event": "conversation_turn_start",
            "level": "INFO",
            "fields": {"turn_id": "request-1", "user_input": "北京到上海明天有航班吗"},
        },
        {
            "index": 1,
            "type": "model",
            "event": "model_call_start",
            "level": "INFO",
            "request": {
                "system_prompt": "完整 prompt",
                "messages": [{"role": "human", "content": "北京到上海明天有航班吗"}],
            },
            "fields": {
                "turn_id": "request-1",
                "request_trace": {
                    "system_prompt": "完整 prompt",
                    "messages": [{"role": "human", "content": "北京到上海明天有航班吗"}],
                },
            },
        },
        {
            "index": 2,
            "type": "model",
            "event": "model_call_end",
            "level": "INFO",
            "response": [
                {
                    "role": "ai",
                    "content": [
                        {
                            "type": "function_call",
                            "name": "resolve_flight_locations",
                            "arguments": '{"locations":["北京","上海"]}',
                        }
                    ],
                }
            ],
            "fields": {
                "turn_id": "request-1",
                "response_trace": [
                    {
                        "role": "ai",
                        "content": [
                            {
                                "type": "function_call",
                                "name": "resolve_flight_locations",
                                "arguments": '{"locations":["北京","上海"]}',
                            }
                        ],
                    }
                ],
            },
        },
        {
            "index": 3,
            "type": "tool",
            "event": "tool_call_start",
            "level": "INFO",
            "tool_call_id": "call-1",
            "tool_name": "resolve_flight_locations",
            "request": {
                "name": "resolve_flight_locations",
                "args": {"locations": ["北京", "上海"]},
            },
            "fields": {
                "turn_id": "request-1",
                "tool_call_id": "call-1",
                "tool_name": "resolve_flight_locations",
                "tool_call": {
                    "name": "resolve_flight_locations",
                    "args": {"locations": ["北京", "上海"]},
                },
            },
        },
        {
            "index": 4,
            "type": "tool",
            "event": "tool_call_end",
            "level": "INFO",
            "tool_call_id": "call-1",
            "tool_name": "resolve_flight_locations",
            "response": {"content": '{"items":[{"input":"北京"}]}'},
            "fields": {
                "turn_id": "request-1",
                "tool_call_id": "call-1",
                "tool_name": "resolve_flight_locations",
                "response_trace": {"content": '{"items":[{"input":"北京"}]}'},
            },
        },
    ]
