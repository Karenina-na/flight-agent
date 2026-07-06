import logging
from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.messages import AIMessage
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


def test_observability_middleware_does_not_expose_tools():
    middleware = ObservabilityMiddleware()

    assert middleware.tools == []


def test_observability_middleware_logs_successful_model_call(capsys):
    configure_logging(
        LoggingSettings(enabled=True, level="INFO", format="text", redact=True)
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
        LoggingSettings(enabled=True, level="INFO", format="text", redact=True)
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
