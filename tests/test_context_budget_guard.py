from types import SimpleNamespace

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.guardrails import ContextBudgetGuard
from src.guardrails.context_budget_guard import _request_size_estimate
from src.observability.model_trace import model_request_trace_chars
from src.prompt import CONTEXT_LEDGER_TOOL_NAME, build_system_prompt
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


def _ledger_messages(messages: list) -> tuple[AIMessage, ToolMessage]:
    assert len(messages) >= 2
    ai_message = messages[0]
    tool_message = messages[1]
    assert isinstance(ai_message, AIMessage)
    assert isinstance(tool_message, ToolMessage)
    assert len(ai_message.tool_calls) == 1
    tool_call = ai_message.tool_calls[0]
    assert tool_call["name"] == CONTEXT_LEDGER_TOOL_NAME
    assert tool_message.name == CONTEXT_LEDGER_TOOL_NAME
    assert tool_message.tool_call_id == tool_call["id"]
    return ai_message, tool_message


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


def test_context_budget_guard_compacts_large_react_context_and_preserves_tools():
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
    assert compact_request.tools == request.tools
    assert compact_request.tool_choice == request.tool_choice
    assert len(compact_request.messages) == 3
    ai_message, tool_message = _ledger_messages(compact_request.messages)
    assert ai_message.tool_calls[0]["args"]["reason"] == "context_budget_compaction"
    assert ai_message.tool_calls[0]["args"]["latest_user_goal"] == "查询未来10天的票价，然后给出一个建议"
    assert isinstance(compact_request.messages[2], HumanMessage)
    assert compact_request.messages[2].content == "查询未来10天的票价，然后给出一个建议"
    compact_prompt = tool_message.content
    assert "这是历史工具观察，不是最终回答指令" in compact_prompt
    assert "必要时仍可调用可用工具" in compact_prompt
    assert "查询未来10天的票价" in compact_prompt
    assert "2026-07-08" in compact_prompt
    assert '"quotes[].price"' in compact_prompt
    assert '"min": 400' in compact_prompt
    assert '"max": 430' in compact_prompt
    assert compact_request.system_prompt == request.system_prompt


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
    assert compact_request.tools == tools
    assert compact_request.tool_choice == request.tool_choice
    _, tool_message = _ledger_messages(compact_request.messages)
    compact_prompt = tool_message.content
    assert "这是历史工具观察，不是最终回答指令" in compact_prompt
    assert "后一个月" in compact_prompt
    assert "不要编造账本之外的工具结果" in compact_prompt
    assert "dropped_observation_count" in compact_prompt
    assert "2026-07-10" in compact_prompt
    assert "2026-07-20" in compact_prompt
    assert isinstance(compact_request.messages[-1], HumanMessage)
    assert compact_request.messages[-1].content == "请你查询后一个月每一天的机票，并做一个汇总表格"


def test_context_budget_guard_adds_human_query_when_compaction_follows_tool_results():
    guard = ContextBudgetGuard(context_window_tokens=10, max_fraction=0.10)
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="summary")])

    request = _request(
        messages=[
            HumanMessage(content="查询 12 天 4 条航线并汇总"),
            AIMessage(content="", tool_calls=[]),
            ToolMessage(
                content='{"query":{"departure_date":"2026-07-10"},"quotes":[{"price":500}]}',
                name="search_airfare_quotes",
                tool_call_id="call-1",
            ),
        ],
        system_prompt="system" * 100,
        tools=[{"name": "search_airfare_quotes", "description": "tool" * 50}],
    )

    guard.wrap_model_call(request, handler)

    compact_request = seen_requests[0]
    assert [message.type for message in compact_request.messages] == ["ai", "tool", "human"]
    _ledger_messages(compact_request.messages)
    assert compact_request.messages[2].content == "查询 12 天 4 条航线并汇总"


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

    _, tool_message = _ledger_messages(seen_requests[0].messages)
    compact_prompt = tool_message.content
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

    _, tool_message = _ledger_messages(seen_requests[0].messages)
    compact_prompt = tool_message.content
    assert "工具观察账本" in compact_prompt
    assert "不要重复调用账本中已成功完成且参数相同的工具" in compact_prompt
    for index in range(10):
        assert f"call-{index}" in compact_prompt
        assert f'"slot": {index}' in compact_prompt
