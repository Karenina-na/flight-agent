import json
import logging

from src.config import LoggingSettings
from src.observability.events import log_event
from src.observability.logging import configure_logging, sanitize_fields
from src.runtime import Context


def test_text_logging_includes_event_and_context_ids(capsys):
    configure_logging(
        LoggingSettings(
            enabled=True,
            level="INFO",
            format="text",
            redact=True,
            output_path="",
            console=True,
        )
    )
    context = Context(
        user_id="u1",
        thread_id="thread-1",
        request_id="request-1",
        run_id="run-1",
        workspace_id="workspace-1",
    )

    log_event("model_call_start", context=context, message_count=2)

    captured = capsys.readouterr()
    assert "INFO event=model_call_start" in captured.err
    assert "user_id=u1" in captured.err
    assert "thread_id=thread-1" in captured.err
    assert "request_id=request-1" in captured.err
    assert "run_id=run-1" in captured.err
    assert "message_count=2" in captured.err


def test_json_logging_outputs_parseable_json(capsys):
    configure_logging(
        LoggingSettings(
            enabled=True,
            level="INFO",
            format="json",
            redact=True,
            output_path="",
            console=True,
        )
    )

    log_event("tool_call_end", context=Context(user_id="u1"), tool_name="demo")

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert payload["event"] == "tool_call_end"
    assert payload["level"] == "INFO"
    assert payload["user_id"] == "u1"
    assert payload["tool_name"] == "demo"


def test_sensitive_fields_are_redacted():
    fields = sanitize_fields(
        {
            "api_key": "secret-key",
            "nested": {"authorization": "Bearer secret"},
            "safe": "visible",
        },
        redact=True,
    )

    assert fields["api_key"] == "[REDACTED]"
    assert fields["nested"]["authorization"] == "[REDACTED]"
    assert fields["safe"] == "visible"


def test_disabled_logging_suppresses_events(capsys):
    configure_logging(
        LoggingSettings(
            enabled=False,
            level="INFO",
            format="text",
            redact=True,
            output_path="",
            console=True,
        )
    )

    log_event("model_call_start", level=logging.INFO)

    captured = capsys.readouterr()
    assert captured.err == ""


def test_logging_writes_to_configured_file(tmp_path):
    log_path = tmp_path / "logs" / "skypilot.log"
    configure_logging(
        LoggingSettings(
            enabled=True,
            level="INFO",
            format="text",
            redact=True,
            output_path=str(log_path),
            console=False,
        )
    )

    log_event("model_call_start", context=Context(user_id="u1"), message_count=2)

    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "INFO event=model_call_start" in log_text
    assert "user_id=u1" in log_text
    assert "message_count=2" in log_text


def test_logging_does_not_write_to_console_by_default(capsys, tmp_path):
    log_path = tmp_path / "logs" / "skypilot.log"
    configure_logging(
        LoggingSettings(
            enabled=True,
            level="INFO",
            format="text",
            redact=True,
            output_path=str(log_path),
        )
    )

    log_event("model_call_start", context=Context(user_id="u1"))

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "INFO event=model_call_start" in log_path.read_text(encoding="utf-8")
