import pytest

from src.config import LoggingSettings
from src.observability import (
    full_text_trace_fields,
    observe_agent_run,
    observe_agent_stream,
    text_trace_fields,
)
from src.observability.logging import configure_logging
from src.runtime import Context


def test_observe_agent_run_logs_start_and_end(capsys):
    configure_logging(
        LoggingSettings(
            enabled=True,
            level="INFO",
            format="text",
            redact=True,
            console=True,
        )
    )
    context = Context(
        user_id="u1",
        thread_id="thread-1",
        request_id="request-1",
        run_id="run-1",
    )

    with observe_agent_run(
        context,
        entrypoint="test",
        stream_mode="updates",
    ):
        pass

    captured = capsys.readouterr()
    assert "event=agent_run_start" in captured.err
    assert "event=agent_run_end" in captured.err
    assert "entrypoint=test" in captured.err
    assert "stream_mode=updates" in captured.err
    assert "user_id=u1" in captured.err
    assert "request_id=request-1" in captured.err
    assert "run_id=run-1" in captured.err
    assert "trace_id=thread-1" in captured.err
    assert "turn_id=request-1" in captured.err
    assert "duration_ms=" in captured.err


def test_observe_agent_run_logs_error_and_reraises(capsys):
    configure_logging(
        LoggingSettings(
            enabled=True,
            level="INFO",
            format="text",
            redact=True,
            console=True,
        )
    )
    context = Context(user_id="u1", request_id="request-1", run_id="run-1")

    with pytest.raises(RuntimeError, match="request failed"):
        with observe_agent_run(
            context,
            entrypoint="test",
            stream_mode="updates",
        ):
            raise RuntimeError("request failed")

    captured = capsys.readouterr()
    assert "event=agent_run_start" in captured.err
    assert "ERROR event=agent_run_error" in captured.err
    assert "error_type=RuntimeError" in captured.err
    assert "request failed" not in captured.err


def test_observe_agent_stream_yields_items_and_logs_run(capsys):
    configure_logging(
        LoggingSettings(
            enabled=True,
            level="INFO",
            format="text",
            redact=True,
            console=True,
        )
    )
    context = Context(user_id="u1", request_id="request-1", run_id="run-1")

    items = list(
        observe_agent_stream(
            iter(["first", "second"]),
            context,
            entrypoint="test.stream",
            stream_mode="messages",
        )
    )

    captured = capsys.readouterr()
    assert items == ["first", "second"]
    assert "event=agent_run_start" in captured.err
    assert "event=agent_run_end" in captured.err
    assert "entrypoint=test.stream" in captured.err
    assert "stream_mode=messages" in captured.err


def test_text_trace_fields_records_safe_text_summary_only():
    fields = text_trace_fields("user_input", "查询明天北京到上海")

    assert fields["user_input_chars"] == 9
    assert fields["user_input_bytes"] == len("查询明天北京到上海".encode("utf-8"))
    assert len(fields["user_input_sha256"]) == 64
    assert "查询明天北京到上海" not in fields.values()


def test_full_text_trace_fields_records_raw_text_and_summary():
    fields = full_text_trace_fields("user_input", "查询明天北京到上海")

    assert fields["user_input"] == "查询明天北京到上海"
    assert fields["user_input_chars"] == 9
    assert fields["user_input_bytes"] == len("查询明天北京到上海".encode("utf-8"))
    assert len(fields["user_input_sha256"]) == 64
