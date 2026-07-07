import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from main import (
    DEMO_MESSAGE,
    HELP_TEXT,
    CliSession,
    format_tools,
    handle_command,
    stream_agent_response,
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
        def stream(self, *args, **kwargs):
            yield AIMessageChunk(content="查到一些结果"), {}

    events = []

    def fake_log_event(event, **kwargs):
        events.append((event, kwargs))

    monkeypatch.setattr("main.agent", FakeAgent())
    monkeypatch.setattr("main.log_event", fake_log_event)
    monkeypatch.setattr("main.observe_agent_stream", lambda stream, *args, **kwargs: stream)

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


def test_cli_logs_conversation_turn_error(monkeypatch):
    class BrokenAgent:
        def stream(self, *args, **kwargs):
            yield AIMessageChunk(content="半截回复"), {}
            raise RuntimeError("boom")

    events = []

    def fake_log_event(event, **kwargs):
        events.append((event, kwargs))

    monkeypatch.setattr("main.agent", BrokenAgent())
    monkeypatch.setattr("main.log_event", fake_log_event)
    monkeypatch.setattr("main.observe_agent_stream", lambda stream, *args, **kwargs: stream)

    with pytest.raises(RuntimeError, match="boom"):
        stream_agent_response("查询明天北京到上海", CliSession(thread_id="cli-test"))

    assert [event for event, _ in events] == [
        "conversation_turn_start",
        "conversation_turn_error",
    ]
    assert events[1][1]["assistant_chunk_count"] == 1
    assert events[1][1]["error_type"] == "RuntimeError"
    assert events[1][1]["partial_assistant_output"] == "半截回复"
    assert len(events[1][1]["partial_assistant_output_sha256"]) == 64
