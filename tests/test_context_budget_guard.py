from types import SimpleNamespace

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.guardrails import ContextBudgetGuard
from src.guardrails.context_budget_guard import _request_size_estimate
from src.observability import collect_trace_events
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
    context: Context | None = None,
    state: dict | None = None,
) -> ModelRequest:
    return ModelRequest(
        model=_model(),
        messages=messages,
        system_prompt=system_prompt,
        tools=tools or [],
        state=state,
        runtime=SimpleNamespace(
            context=context
            or Context(
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
    ledger_index = next(
        index
        for index, message in enumerate(messages)
        if isinstance(message, AIMessage)
        and message.tool_calls
        and message.tool_calls[0]["name"] == CONTEXT_LEDGER_TOOL_NAME
    )
    assert len(messages) > ledger_index + 1
    ai_message = messages[ledger_index]
    tool_message = messages[ledger_index + 1]
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


def test_context_budget_guard_does_not_inject_todo_snapshot_under_budget():
    guard = ContextBudgetGuard(context_window_tokens=8192, max_fraction=0.85)
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    request = _request(
        messages=[HumanMessage(content="查询北京到上海")],
        state={
            "messages": [],
            "todos": [{"content": "查询报价", "status": "in_progress"}],
        },
    )
    response = guard.wrap_model_call(request, handler)

    assert response.result[0].content == "ok"
    assert seen_requests == [request]


def test_context_budget_guard_passes_through_old_react_history_under_budget():
    guard = ContextBudgetGuard(context_window_tokens=8192, max_fraction=0.85)
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    request = _request(
        messages=[
            HumanMessage(content="请查询北京到上海在 2026-07-10 的机票报价样本"),
            AIMessage(
                content=[
                    {
                        "type": "reasoning",
                        "content": [
                            {"type": "reasoning_text", "text": "旧思考不应回灌"}
                        ],
                    },
                    {"type": "text", "text": "我来查询北京到上海。"},
                    {
                        "type": "function_call",
                        "name": "search_airfare_quotes",
                        "arguments": '{"origin":"北京","destination":"上海"}',
                        "call_id": "call-old",
                    },
                ],
                tool_calls=[
                    {
                        "id": "call-old",
                        "name": "search_airfare_quotes",
                        "args": {"origin": "北京", "destination": "上海"},
                    }
                ],
            ),
            ToolMessage(
                content=_quote_payload("2026-07-10", 550, 700),
                name="search_airfare_quotes",
                tool_call_id="call-old",
            ),
            AIMessage(
                content=[
                    {
                        "type": "reasoning",
                        "content": [
                            {"type": "reasoning_text", "text": "旧总结思考不应回灌"}
                        ],
                    },
                    {"type": "text", "text": "北京到上海报价样本已查询完成。"},
                ]
            ),
            HumanMessage(content="查一下明天广州到香港的机票"),
        ],
        system_prompt="Base prompt.",
        tools=[{"name": "search_airfare_quotes", "description": "tool"}],
    )

    assert _request_size_estimate(request) < round(8192 * 4 * 0.85)

    guard.wrap_model_call(request, handler)

    assert seen_requests == [request]


def test_context_budget_guard_passes_through_active_turn_tool_protocol_under_budget():
    guard = ContextBudgetGuard(context_window_tokens=8192, max_fraction=0.85)
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    request = _request(
        messages=[
            HumanMessage(content="第一轮问题"),
            AIMessage(content="第一轮最终回答"),
            HumanMessage(content="当前问题"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-current",
                        "name": "query_current_date",
                        "args": {"days_offset": 1},
                    }
                ],
            ),
            ToolMessage(
                content='{"target_date":"2026-07-09"}',
                name="query_current_date",
                tool_call_id="call-current",
            ),
        ],
        system_prompt="Base prompt.",
        tools=[{"name": "query_current_date", "description": "tool"}],
    )

    guard.wrap_model_call(request, handler)

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
    assert isinstance(compact_request.messages[0], HumanMessage)
    assert compact_request.messages[0].content == "查询未来10天的票价，然后给出一个建议"
    ai_message, tool_message = _ledger_messages(compact_request.messages)
    assert ai_message.tool_calls[0]["args"]["reason"] == "context_budget_compaction"
    assert ai_message.tool_calls[0]["args"]["latest_user_goal"] == "查询未来10天的票价，然后给出一个建议"
    compact_prompt = tool_message.content
    assert "这是历史工具观察，不是最终回答指令" in compact_prompt
    assert "必要时仍可调用可用工具" in compact_prompt
    assert "查询未来10天的票价" in compact_prompt
    assert "2026-07-08" in compact_prompt
    assert '"quotes[].price"' in compact_prompt
    assert '"min": 400' in compact_prompt
    assert '"max": 430' in compact_prompt
    assert compact_request.system_prompt == request.system_prompt


def test_context_budget_guard_injects_todo_snapshot_only_after_compaction():
    guard = ContextBudgetGuard(context_window_tokens=10, max_fraction=0.10)
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="summary")])

    request = _request(
        messages=[
            HumanMessage(content="批量查询未来多天票价并汇总"),
            ToolMessage(
                content=_quote_payload("2026-07-08", 400, 430),
                name="search_airfare_quotes",
                tool_call_id="call-1",
            ),
        ],
        system_prompt="system" * 100,
        tools=[{"name": "search_airfare_quotes", "description": "tool" * 50}],
        state={
            "messages": [],
            "todos": [
                {
                    "content": "查询前五天报价",
                    "status": "completed",
                    "raw_tool_result": "do not expose",
                },
                {"content": "汇总价格区间", "status": "in_progress"},
            ],
        },
    )

    guard.wrap_model_call(request, handler)

    compact_request = seen_requests[0]
    _, tool_message = _ledger_messages(compact_request.messages)
    compact_prompt = tool_message.content
    assert "todo_snapshot" in compact_prompt
    assert "protected task state" in compact_prompt
    assert "查询前五天报价" in compact_prompt
    assert "completed" in compact_prompt
    assert "汇总价格区间" in compact_prompt
    assert "in_progress" in compact_prompt
    assert "raw_tool_result" not in compact_prompt


def test_context_budget_guard_compacts_without_todos_when_state_is_missing():
    guard = ContextBudgetGuard(context_window_tokens=10, max_fraction=0.10)
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="summary")])

    request = _request(
        messages=[
            HumanMessage(content="查询未来多天票价"),
            ToolMessage(
                content=_quote_payload("2026-07-08", 400, 430),
                name="search_airfare_quotes",
                tool_call_id="call-1",
            ),
        ],
        system_prompt="system" * 100,
        tools=[{"name": "search_airfare_quotes", "description": "tool" * 50}],
    )

    guard.wrap_model_call(request, handler)

    _, tool_message = _ledger_messages(seen_requests[0].messages)
    assert "todo_snapshot" not in tool_message.content


def test_context_budget_guard_logs_bounded_compacted_state_preview():
    guard = ContextBudgetGuard(context_window_tokens=10, max_fraction=0.10)

    def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="summary")])

    request = _request(
        messages=[
            HumanMessage(content="查询未来10天的票价，然后给出一个建议"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "search_airfare_quotes",
                        "args": {
                            "origin": "北京",
                            "destination": "上海",
                            "departure_date": "2026-07-08",
                        },
                    }
                ],
            ),
            ToolMessage(
                content=_quote_payload("2026-07-08", 400, 430),
                name="search_airfare_quotes",
                tool_call_id="call-1",
            ),
        ],
        system_prompt="system" * 100,
        tools=[{"name": "search_airfare_quotes", "description": "tool" * 50}],
    )

    with collect_trace_events(trace_id="thread-1") as events:
        guard.wrap_model_call(request, handler)

    compact_events = [
        event
        for event in events
        if event["event"] == "react_context_budget_compacted"
    ]
    assert len(compact_events) == 1
    fields = compact_events[0]["fields"]
    assert fields["compacted_state_preview"].startswith("## 压缩后的历史工作状态")
    assert "tool_observation_ledger" in fields["compacted_state_preview"]
    assert "2026-07-08" in fields["compacted_state_preview"]
    assert fields["compacted_state_preview_chars"] <= 4003
    assert fields["compacted_state_chars"] >= fields["compacted_state_preview_chars"]
    assert fields["compacted_state_sha256"]
    assert fields["todo_snapshot_item_count"] == 0


def test_context_budget_guard_logs_full_synthetic_observation_preview_with_todo():
    guard = ContextBudgetGuard(context_window_tokens=10, max_fraction=0.10)

    def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="summary")])

    request = _request(
        messages=[
            HumanMessage(content="批量查询未来多天票价并汇总"),
            ToolMessage(
                content=_quote_payload("2026-07-08", 400, 430),
                name="search_airfare_quotes",
                tool_call_id="call-1",
            ),
        ],
        system_prompt="system" * 100,
        tools=[{"name": "search_airfare_quotes", "description": "tool" * 50}],
        state={
            "messages": [],
            "todos": [{"content": "汇总价格区间", "status": "in_progress"}],
        },
    )

    with collect_trace_events(trace_id="thread-1") as events:
        guard.wrap_model_call(request, handler)

    compact_events = [
        event
        for event in events
        if event["event"] == "react_context_budget_compacted"
    ]
    assert len(compact_events) == 1
    fields = compact_events[0]["fields"]
    assert fields["compacted_state_preview"].startswith("## 压缩后的历史工作状态")
    assert "todo_snapshot" in fields["compacted_state_preview"]
    assert "汇总价格区间" in fields["compacted_state_preview"]
    assert fields["todo_snapshot_item_count"] == 1
    assert fields["todo_snapshot_dropped_count"] == 0
    assert fields["todo_snapshot_truncated_count"] == 0


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
    assert "2026-07-20" not in compact_prompt
    assert isinstance(compact_request.messages[0], HumanMessage)
    assert any(
        "2026-07-20" in str(getattr(message, "content", ""))
        for message in compact_request.messages
    )
    assert any(
        str(getattr(message, "content", "")) == "请你查询后一个月每一天的机票，并做一个汇总表格"
        for message in compact_request.messages
    )


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
    assert [message.type for message in compact_request.messages] == ["human", "ai", "tool"]
    _ledger_messages(compact_request.messages)
    assert compact_request.messages[0].content == "查询 12 天 4 条航线并汇总"


def test_context_budget_guard_keeps_human_query_before_synthetic_ledger():
    guard = ContextBudgetGuard(context_window_tokens=10, max_fraction=0.10)
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="summary")])

    request = _request(
        messages=[
            HumanMessage(content="执行复杂批量查询并汇总"),
            AIMessage(
                content="我先查第一批。",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "search_airfare_quotes",
                        "args": {
                            "origin": "北京",
                            "destination": "上海",
                            "departure_date": "2026-07-10",
                        },
                    }
                ],
            ),
            ToolMessage(
                content=_quote_payload("2026-07-10", 550, 700),
                name="search_airfare_quotes",
                tool_call_id="call-1",
            ),
        ],
        system_prompt="system" * 100,
        tools=[{"name": "search_airfare_quotes", "description": "tool" * 50}],
    )

    guard.wrap_model_call(request, handler)

    compact_request = seen_requests[0]
    assert [message.type for message in compact_request.messages] == ["human", "ai", "tool"]
    assert compact_request.messages[0].content == "执行复杂批量查询并汇总"
    _ledger_messages(compact_request.messages)


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


def test_context_budget_guard_preserves_tool_observations_after_layer1_trimming():
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
            HumanMessage(content="查询多个日期后汇总"),
            AIMessage(
                content=[
                    {"type": "reasoning", "content": "旧推理应在压缩视图中裁剪"},
                    {"type": "text", "text": "我先查第一天。"},
                    {
                        "type": "function_call",
                        "name": "search_airfare_quotes",
                        "arguments": '{"departure_date":"2026-07-08"}',
                        "call_id": "call-1",
                    },
                ],
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "search_airfare_quotes",
                        "args": {"departure_date": "2026-07-08"},
                    }
                ],
            ),
            ToolMessage(
                content='{"query":{"departure_date":"2026-07-08"},"quotes":[{"price":400}]}',
                name="search_airfare_quotes",
                tool_call_id="call-1",
            ),
            AIMessage(content="第一天查完。"),
            HumanMessage(content="继续查第二天"),
            AIMessage(
                content=[
                    {"type": "reasoning", "content": "第二段旧推理也应裁剪"},
                    {"type": "text", "text": "我继续查第二天。"},
                ],
                tool_calls=[
                    {
                        "id": "call-2",
                        "name": "search_airfare_quotes",
                        "args": {"departure_date": "2026-07-09"},
                    }
                ],
            ),
            ToolMessage(
                content='{"query":{"departure_date":"2026-07-09"},"quotes":[{"price":430}]}',
                name="search_airfare_quotes",
                tool_call_id="call-2",
            ),
            HumanMessage(content="现在汇总一下"),
        ],
        system_prompt="system" * 100,
        tools=[{"name": "search_airfare_quotes", "description": "tool" * 50}],
    )

    guard.wrap_model_call(request, handler)

    _, tool_message = _ledger_messages(seen_requests[0].messages)
    compact_prompt = tool_message.content
    compact_request = seen_requests[0]
    compact_request_text = "\n".join(
        str(getattr(message, "content", "")) for message in compact_request.messages
    )
    assert "call-1" in compact_prompt
    assert any(
        getattr(message, "tool_call_id", "") == "call-2"
        for message in compact_request.messages
    )
    assert "2026-07-08" in compact_prompt
    assert "2026-07-09" in compact_request_text
    assert "旧推理应在压缩视图中裁剪" not in compact_prompt
    assert "第二段旧推理也应裁剪" in compact_request_text
    assert "function_call" not in compact_prompt


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


def test_context_budget_guard_compacts_old_turns_without_duplicating_raw_suffix():
    guard = ContextBudgetGuard(context_window_tokens=10, max_fraction=0.10)
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="summary")])

    request = _request(
        messages=[
            HumanMessage(content="第一轮：查询北京到上海"),
            AIMessage(content="第一轮工具调用完成。"),
            ToolMessage(
                content='{"query":{"slot":"old"},"records":[{"amount":1}]}',
                name="generic_lookup",
                tool_call_id="old-call",
            ),
            AIMessage(content="第一轮最终摘要：old-tail-marker"),
            HumanMessage(content="第二轮：保留这个最近 turn 原文"),
            AIMessage(content="第二轮 assistant 原文：raw-suffix-marker"),
            HumanMessage(content="第三轮：最新问题"),
        ],
        system_prompt="system" * 100,
        tools=[{"name": "generic_lookup", "description": "tool" * 50}],
    )

    guard.wrap_model_call(request, handler)

    compact_request = seen_requests[0]
    assert [message.type for message in compact_request.messages] == ["human", "ai", "human", "ai", "tool"]
    _, tool_message = _ledger_messages(compact_request.messages)
    compact_prompt = tool_message.content
    assert "old-call" in compact_prompt
    assert "第一轮：查询北京到上海" in compact_prompt
    assert "raw-suffix-marker" not in compact_prompt
    assert compact_request.messages[0].content == "第二轮：保留这个最近 turn 原文"
    assert compact_request.messages[1].content == "第二轮 assistant 原文：raw-suffix-marker"
    assert compact_request.messages[2].content == "第三轮：最新问题"


def test_context_budget_guard_uses_runtime_user_goal_when_latest_human_is_summary():
    guard = ContextBudgetGuard(context_window_tokens=10, max_fraction=0.10)
    seen_requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="summary")])

    request = _request(
        messages=[
            HumanMessage(content="Here is a summary of the conversation to date:"),
            AIMessage(
                content="继续查询批量任务。",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "search_airfare_quotes",
                        "args": {
                            "origin": "北京",
                            "destination": "上海",
                            "departure_date": "2026-07-10",
                        },
                    }
                ],
            ),
            ToolMessage(
                content=_quote_payload("2026-07-10", 550, 700),
                name="search_airfare_quotes",
                tool_call_id="call-1",
            ),
        ],
        system_prompt="system" * 100,
        tools=[{"name": "search_airfare_quotes", "description": "tool" * 50}],
        context=Context(
            user_id="u1",
            thread_id="thread-1",
            request_id="request-1",
            run_id="run-1",
            current_user_input="请查询未来 10 天北京到上海机票并汇总报告",
            current_user_input_sha256="manual-test-hash",
        ),
    )

    guard.wrap_model_call(request, handler)

    compact_request = seen_requests[0]
    assert isinstance(compact_request.messages[0], HumanMessage)
    assert compact_request.messages[0].content == "请查询未来 10 天北京到上海机票并汇总报告"
    ai_message, tool_message = _ledger_messages(compact_request.messages)
    assert (
        ai_message.tool_calls[0]["args"]["latest_user_goal"]
        == "请查询未来 10 天北京到上海机票并汇总报告"
    )
    assert "最近用户目标：请查询未来 10 天北京到上海机票并汇总报告" in tool_message.content
    assert "最近用户目标：Here is a summary" not in tool_message.content
