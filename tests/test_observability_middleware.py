import logging
from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse, ToolCallRequest
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_openai import ChatOpenAI

from src.config import LoggingSettings
from src.observability import ObservabilityMiddleware
from src.observability.logging import configure_logging
from src.runtime import Context


def _model():
    return ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key="not-needed",
        model="google/gemma-4-e2b",
        profile={"max_input_tokens": 8192},
    )


def _request() -> ModelRequest:
    return ModelRequest(
        model=_model(),
        messages=[],
        system_prompt="Base prompt.",
        runtime=SimpleNamespace(
            context=Context(
                user_id="u1",
                thread_id="thread-1",
                request_id="request-1",
                run_id="run-1",
            )
        ),
    )


@tool
def demo_tool(title: str, api_key: str = "secret") -> str:
    """Demo tool used for observability middleware tests."""
    return f"created {title}"


def _tool_request() -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "id": "call-1",
            "name": "demo_tool",
            "args": {"title": "task", "api_key": "secret"},
        },
        tool=demo_tool,
        state={},
        runtime=ToolRuntime(
            state={},
            context=Context(
                user_id="u1",
                thread_id="thread-1",
                request_id="request-1",
                run_id="run-1",
            ),
            config={},
            stream_writer=lambda _: None,
            tool_call_id="call-1",
            store=None,
        ),
    )


def test_observability_middleware_does_not_expose_tools():
    middleware = ObservabilityMiddleware()

    assert middleware.tools == []


def test_observability_middleware_logs_successful_model_call(capsys):
    configure_logging(
        LoggingSettings(
            enabled=True,
            level="INFO",
            format="text",
            redact=True,
            console=True,
        )
    )
    middleware = ObservabilityMiddleware()

    def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="ok")])

    response = middleware.wrap_model_call(_request(), handler)

    captured = capsys.readouterr()
    assert response.result[0].content == "ok"
    assert "event=model_call_start" in captured.err
    assert "event=model_call_end" in captured.err
    assert "user_id=u1" in captured.err
    assert "duration_ms=" in captured.err


def test_observability_middleware_logs_error_and_reraises(capsys):
    configure_logging(
        LoggingSettings(
            enabled=True,
            level="INFO",
            format="text",
            redact=True,
            console=True,
        )
    )
    middleware = ObservabilityMiddleware()

    def handler(request: ModelRequest) -> ModelResponse:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        middleware.wrap_model_call(_request(), handler)

    captured = capsys.readouterr()
    assert "event=model_call_start" in captured.err
    assert "ERROR event=model_call_error" in captured.err
    assert "error_type=RuntimeError" in captured.err
    assert "boom" not in captured.err


def test_observability_middleware_logs_successful_tool_call(capsys):
    configure_logging(
        LoggingSettings(
            enabled=True,
            level="INFO",
            format="text",
            redact=True,
            console=True,
        )
    )
    middleware = ObservabilityMiddleware()

    def handler(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content="ok",
            name="demo_tool",
            tool_call_id=request.tool_call["id"],
        )

    response = middleware.wrap_tool_call(_tool_request(), handler)

    captured = capsys.readouterr()
    assert response.content == "ok"
    assert "event=tool_call_start" in captured.err
    assert "event=tool_call_end" in captured.err
    assert "tool_name=demo_tool" in captured.err
    assert "argument_keys=['api_key', 'title']" in captured.err
    assert "duration_ms=" in captured.err
    assert "secret" not in captured.err
    assert "task" not in captured.err


def test_observability_middleware_logs_tool_error_and_reraises(capsys):
    configure_logging(
        LoggingSettings(
            enabled=True,
            level="INFO",
            format="text",
            redact=True,
            console=True,
        )
    )
    middleware = ObservabilityMiddleware()

    def handler(request: ToolCallRequest) -> ToolMessage:
        raise RuntimeError("tool exploded")

    with pytest.raises(RuntimeError, match="tool exploded"):
        middleware.wrap_tool_call(_tool_request(), handler)

    captured = capsys.readouterr()
    assert "event=tool_call_start" in captured.err
    assert "ERROR event=tool_call_error" in captured.err
    assert "tool_name=demo_tool" in captured.err
    assert "error_type=RuntimeError" in captured.err
    assert "tool exploded" not in captured.err
