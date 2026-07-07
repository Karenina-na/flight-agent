from types import SimpleNamespace

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.guardrails import ContextBudgetGuard
from src.runtime import Context


def _model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key="not-needed",
        model="qwen3.5-4b-mlx",
        profile={"max_input_tokens": 8192},
    )


def _request(
    *,
    messages: list,
    system_prompt: str = "Base prompt.",
    tools: list | None = None,
) -> ModelRequest:
    return ModelRequest(
        model=_model(),
        messages=messages,
        system_prompt=system_prompt,
        tools=tools or [],
        runtime=SimpleNamespace(
            context=Context(
                user_id="u1",
                thread_id="thread-1",
                request_id="request-1",
                run_id="run-1",
            )
        ),
    )


def test_context_budget_guard_passes_through_small_requests():
    guard = ContextBudgetGuard(context_window_tokens=8192, max_fraction=0.85)
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    request = _request(messages=[HumanMessage(content="查询北京到上海")])
    response = guard.wrap_model_call(request, handler)

    assert response.result[0].content == "ok"
    assert seen_requests == [request]


def test_context_budget_guard_does_not_compact_without_tool_results():
    guard = ContextBudgetGuard(context_window_tokens=10, max_fraction=0.10)
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    request = _request(
        messages=[HumanMessage(content="很长" * 100)],
        system_prompt="system" * 100,
        tools=[{"name": "search_airfare_quotes", "description": "tool" * 50}],
    )
    response = guard.wrap_model_call(request, handler)

    assert response.result[0].content == "ok"
    assert seen_requests == [request]


def test_context_budget_guard_compacts_large_react_context_and_disables_tools():
    guard = ContextBudgetGuard(context_window_tokens=10, max_fraction=0.10)
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="summary")])

    request = _request(
        messages=[
            HumanMessage(content="查询未来10天的票价，然后给出一个建议"),
            AIMessage(content="", tool_calls=[]),
            ToolMessage(
                content=(
                    '{"query":{"origin":"北京","destination":"上海",'
                    '"departure_date":"2026-07-08"},'
                    '"captured_at":"2026-07-07T19:37:16+08:00",'
                    '"sources_used":["fliggy_mcp"],'
                    '"quotes":[{"price":400,"currency":"CNY"},'
                    '{"price":430,"currency":"CNY"}],'
                    '"limitations":["sample only"]}'
                ),
                name="search_airfare_quotes",
                tool_call_id="call-1",
            ),
        ],
        system_prompt="system" * 100,
        tools=[{"name": "search_airfare_quotes", "description": "tool" * 50}],
    )

    response = guard.wrap_model_call(request, handler)

    assert response.result[0].content == "summary"
    compact_request = seen_requests[0]
    assert compact_request is not request
    assert compact_request.tools == []
    assert compact_request.tool_choice == "none"
    assert len(compact_request.messages) == 1
    assert isinstance(compact_request.messages[0], HumanMessage)
    compact_prompt = compact_request.messages[0].content
    assert "不要再调用工具" in compact_prompt
    assert "查询未来10天的票价" in compact_prompt
    assert "2026-07-08" in compact_prompt
    assert "min_price=400" in compact_prompt
    assert "max_price=430" in compact_prompt
    assert isinstance(compact_request.system_message, SystemMessage)
    assert "必须生成面向用户的最终回答" in compact_request.system_prompt


def test_context_budget_guard_keeps_recent_tool_facts_bounded():
    guard = ContextBudgetGuard(
        context_window_tokens=10,
        max_fraction=0.10,
        max_tool_facts=2,
    )
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="summary")])

    request = _request(
        messages=[
            HumanMessage(content="查询未来10天的票价"),
            ToolMessage(
                content='{"query":{"departure_date":"2026-07-08"},"quotes":[]}',
                name="search_airfare_quotes",
                tool_call_id="call-1",
            ),
            ToolMessage(
                content='{"query":{"departure_date":"2026-07-09"},"quotes":[]}',
                name="search_airfare_quotes",
                tool_call_id="call-2",
            ),
            ToolMessage(
                content='{"query":{"departure_date":"2026-07-10"},"quotes":[]}',
                name="search_airfare_quotes",
                tool_call_id="call-3",
            ),
        ],
        system_prompt="system" * 100,
        tools=[{"name": "search_airfare_quotes", "description": "tool" * 50}],
    )

    guard.wrap_model_call(request, handler)

    compact_prompt = seen_requests[0].messages[0].content
    assert "2026-07-08" not in compact_prompt
    assert "2026-07-09" in compact_prompt
    assert "2026-07-10" in compact_prompt
