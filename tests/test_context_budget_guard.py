from types import SimpleNamespace

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.guardrails import ContextBudgetGuard
from src.guardrails.context_budget_guard import _request_size_estimate
from src.observability.model_trace import model_request_trace_chars
from src.prompt import build_system_prompt
from src.runtime import Context
from src.tools import get_tools


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


def _quote_payload(departure_date: str, low: int, high: int) -> str:
    return (
        '{"query":{"origin":"北京","destination":"上海",'
        f'"departure_date":"{departure_date}",'
        '"return_date":null,"cabin":"economy","adults":1,'
        '"children":0,"infants":0,"stops":0,"currency":"cny",'
        '"origin_airports":["PEK","PKX"],"destination_airports":["PVG","SHA"]},'
        '"captured_at":"2026-07-07T19:52:12+08:00",'
        '"sources_used":["fliggy_mcp"],'
        f'"quotes":[{{"price":{low},"currency":"CNY"}},'
        f'{{"price":{high},"currency":"CNY"}}],'
        '"limitations":["sample only"]}'
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


def test_context_budget_guard_uses_observability_request_trace_size():
    tools = get_tools()
    request = _request(
        messages=[
            HumanMessage(content="查询这个日期后十天的价格，并给出购买结论"),
            ToolMessage(
                content=_quote_payload("2026-07-10", 550, 700),
                name="search_airfare_quotes",
                tool_call_id="call-1",
            ),
        ],
        system_prompt=build_system_prompt(tools=tools),
        tools=tools,
    )

    assert _request_size_estimate(request) == model_request_trace_chars(request)


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
    assert '"quotes[].price"' in compact_prompt
    assert '"min": 400' in compact_prompt
    assert '"max": 430' in compact_prompt
    assert isinstance(compact_request.system_message, SystemMessage)
    assert "必须生成面向用户的最终回答" in compact_request.system_prompt


def test_context_budget_guard_compacts_web_41e6d813_like_request():
    guard = ContextBudgetGuard(context_window_tokens=8192, max_fraction=0.85)
    tools = get_tools()
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="summary")])

    request = _request(
        messages=[
            HumanMessage(content="请查询北京到上海在 2026-07-10 的机票报价样本"),
            ToolMessage(
                content=_quote_payload("2026-07-10", 550, 700),
                name="search_airfare_quotes",
                tool_call_id="call-1",
            ),
            AIMessage(content="2026-07-10 报价摘要。" + "历史回答" * 1500),
            HumanMessage(content="查询这个日期后十天的价格，并给出购买结论"),
            ToolMessage(
                content=_quote_payload("2026-07-20", 340, 460),
                name="search_airfare_quotes",
                tool_call_id="call-2",
            ),
            AIMessage(content="2026-07-20 报价摘要。" + "历史回答" * 1500),
            HumanMessage(content="请你查询后一个月每一天的机票，并做一个汇总表格"),
        ],
        system_prompt=build_system_prompt(tools=tools) + "上下文扩展。" * 500,
        tools=tools,
    )

    threshold = round(8192 * 4 * 0.85)
    assert _request_size_estimate(request) > threshold

    guard.wrap_model_call(request, handler)

    compact_request = seen_requests[0]
    assert compact_request.tools == []
    assert compact_request.tool_choice == "none"
    compact_prompt = compact_request.messages[0].content
    assert "后一个月" in compact_prompt
    assert "不要编造未查询结果" in compact_prompt
    assert "dropped_observation_count" in compact_prompt
    assert "2026-07-10" in compact_prompt
    assert "2026-07-20" in compact_prompt


def test_context_budget_guard_keeps_each_tool_observation_card_instead_of_recent_only():
    guard = ContextBudgetGuard(
        context_window_tokens=10,
        max_fraction=0.10,
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
    assert "工具观察账本" in compact_prompt
    assert "call-1" in compact_prompt
    assert "call-2" in compact_prompt
    assert "call-3" in compact_prompt
    assert "2026-07-08" in compact_prompt
    assert "2026-07-09" in compact_prompt
    assert "2026-07-10" in compact_prompt


def test_context_budget_guard_preserves_each_tool_observation_card():
    guard = ContextBudgetGuard(context_window_tokens=10, max_fraction=0.10)
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="summary")])

    request = _request(
        messages=[
            HumanMessage(content="批量查询 10 个通用对象并汇总"),
            *[
                ToolMessage(
                    content='{"query":{"slot":%d},"records":[{"amount":%d}]}' % (index, index),
                    name="generic_lookup",
                    tool_call_id=f"call-{index}",
                )
                for index in range(10)
            ],
        ],
        system_prompt="system" * 100,
        tools=[{"name": "generic_lookup", "description": "tool" * 50}],
    )

    guard.wrap_model_call(request, handler)

    compact_prompt = seen_requests[0].messages[0].content
    assert "工具观察账本" in compact_prompt
    assert "如果账本显示某些请求已调用成功，不要声称这些请求未完成" in compact_prompt
    for index in range(10):
        assert f"call-{index}" in compact_prompt
        assert f'"slot": {index}' in compact_prompt
