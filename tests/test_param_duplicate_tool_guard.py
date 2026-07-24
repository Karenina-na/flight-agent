import json
from types import SimpleNamespace

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from src.guardrails import ParamAwareDuplicateToolCallGuard
from src.runtime import Context


@tool
def demo_lookup(origin: str, destination: str, limit: int = 20) -> str:
    """Demo lookup tool for duplicate-call guard tests."""
    return f"{origin}->{destination}:{limit}"


@tool
def other_lookup(origin: str, destination: str, limit: int = 20) -> str:
    """Second demo tool for duplicate-call guard tests."""
    return f"{origin}->{destination}:{limit}"


def _request(
    *,
    tool_name: str = "demo_lookup",
    args: dict | None = None,
    request_id: str = "request-1",
) -> ToolCallRequest:
    selected_tool = demo_lookup if tool_name == "demo_lookup" else other_lookup
    return ToolCallRequest(
        tool_call={
            "id": f"call-{tool_name}-{request_id}",
            "name": tool_name,
            "args": args or {"origin": "北京", "destination": "上海", "limit": 20},
        },
        tool=selected_tool,
        state={},
        runtime=ToolRuntime(
            state={},
            context=Context(
                user_id="u1",
                thread_id="thread-1",
                request_id=request_id,
                run_id="run-1",
            ),
            config={},
            stream_writer=lambda _: None,
            tool_call_id=f"call-{tool_name}-{request_id}",
            store=None,
        ),
    )


def test_duplicate_guard_allows_first_matching_tool_call():
    guard = ParamAwareDuplicateToolCallGuard()
    calls = 0

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal calls
        calls += 1
        return ToolMessage(
            content="real result",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    response = guard.wrap_tool_call(_request(), handler)

    assert calls == 1
    assert response.content == "real result"


def test_duplicate_guard_blocks_repeated_tool_and_same_arguments():
    guard = ParamAwareDuplicateToolCallGuard()
    calls = 0

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal calls
        calls += 1
        return ToolMessage(
            content="real result",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    first = guard.wrap_tool_call(_request(), handler)
    second = guard.wrap_tool_call(_request(), handler)
    payload = json.loads(second.content)

    assert calls == 1
    assert first.content == "real result"
    assert second.status == "error"
    assert second.name == "demo_lookup"
    assert second.tool_call_id == "call-demo_lookup-request-1"
    assert payload["status"] == "duplicate_blocked"
    assert payload["duplicate_count"] == 1
    assert payload["stop_requested"] is False
    assert "same tool with the same arguments" in payload["message"]
    assert "produce an answer" in payload["message"]


def test_duplicate_guard_allows_same_tool_with_different_arguments():
    guard = ParamAwareDuplicateToolCallGuard()
    calls = 0

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal calls
        calls += 1
        return ToolMessage(
            content=f"real result {calls}",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    first = guard.wrap_tool_call(
        _request(args={"origin": "北京", "destination": "上海", "limit": 20}),
        handler,
    )
    second = guard.wrap_tool_call(
        _request(args={"origin": "北京", "destination": "上海", "limit": 10}),
        handler,
    )

    assert calls == 2
    assert first.content == "real result 1"
    assert second.content == "real result 2"


def test_duplicate_guard_treats_argument_key_order_as_same_call():
    guard = ParamAwareDuplicateToolCallGuard()
    calls = 0

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal calls
        calls += 1
        return ToolMessage(
            content=f"real result {calls}",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    first = guard.wrap_tool_call(
        _request(args={"origin": "北京", "destination": "上海", "limit": 20}),
        handler,
    )
    second = guard.wrap_tool_call(
        _request(args={"limit": 20, "destination": "上海", "origin": "北京"}),
        handler,
    )

    assert calls == 1
    assert first.content == "real result 1"
    assert json.loads(second.content)["status"] == "duplicate_blocked"


def test_duplicate_guard_allows_different_tool_with_same_arguments():
    guard = ParamAwareDuplicateToolCallGuard()
    calls = 0

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal calls
        calls += 1
        return ToolMessage(
            content=f"real result {calls}",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    first = guard.wrap_tool_call(_request(tool_name="demo_lookup"), handler)
    second = guard.wrap_tool_call(_request(tool_name="other_lookup"), handler)

    assert calls == 2
    assert first.content == "real result 1"
    assert second.content == "real result 2"


def test_duplicate_guard_counts_repetitions_independently_per_tool_and_arguments():
    guard = ParamAwareDuplicateToolCallGuard(loop_stop_after=3)

    def handler(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content="real result",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    guard.wrap_tool_call(_request(tool_name="demo_lookup"), handler)
    guard.wrap_tool_call(_request(tool_name="demo_lookup"), handler)
    guard.wrap_tool_call(_request(tool_name="demo_lookup"), handler)
    guard.wrap_tool_call(_request(tool_name="other_lookup"), handler)
    other_duplicate = guard.wrap_tool_call(
        _request(tool_name="other_lookup"),
        handler,
    )

    payload = json.loads(other_duplicate.content)
    assert payload["duplicate_count"] == 1
    assert payload["status"] == "duplicate_blocked"
    assert payload["stop_requested"] is False


def test_duplicate_guard_scopes_seen_calls_to_request_id():
    guard = ParamAwareDuplicateToolCallGuard()
    calls = 0

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal calls
        calls += 1
        return ToolMessage(
            content=f"real result {calls}",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    first = guard.wrap_tool_call(_request(request_id="request-1"), handler)
    second = guard.wrap_tool_call(_request(request_id="request-2"), handler)

    assert calls == 2
    assert first.content == "real result 1"
    assert second.content == "real result 2"


def test_duplicate_guard_returns_stop_message_without_branch_jump_after_repeated_duplicates():
    guard = ParamAwareDuplicateToolCallGuard(loop_stop_after=3)
    calls = 0

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal calls
        calls += 1
        return ToolMessage(
            content="real result",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    guard.wrap_tool_call(_request(), handler)
    guard.wrap_tool_call(_request(), handler)
    guard.wrap_tool_call(_request(), handler)
    response = guard.wrap_tool_call(_request(), handler)

    assert calls == 1
    assert isinstance(response, ToolMessage)
    assert response.status == "error"
    payload = json.loads(response.content)
    assert payload["status"] == "react_loop_stop_requested"
    assert payload["duplicate_count"] == 3
    assert payload["loop_stop_after"] == 3
    assert payload["stop_requested"] is True
    assert "Stop calling tools" in payload["message"]


def test_duplicate_guard_ends_agent_run_after_repeated_identical_calls():
    class ToolCallingFakeModel(FakeMessagesListChatModel):
        invocation_count: int = 0

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

        def _generate(self, *args, **kwargs):
            self.invocation_count += 1
            return super()._generate(*args, **kwargs)

    repeated_args = {"origin": "北京", "destination": "上海", "limit": 20}
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call-{index}",
                        "name": "demo_lookup",
                        "args": repeated_args,
                    }
                ],
            )
            for index in range(1, 5)
        ]
    )
    guard = ParamAwareDuplicateToolCallGuard(loop_stop_after=3)
    test_agent = create_agent(
        model=model,
        tools=[demo_lookup],
        middleware=[guard],
    )

    result = test_agent.invoke(
        {"messages": [{"role": "user", "content": "查询北京到上海"}]},
        config={"recursion_limit": 20},
    )

    assert model.invocation_count == 4
    assert result["messages"][-1].content == ""
    assert (
        result["messages"][-1]
        .additional_kwargs["skypilot_react_loop_stop_requested"]
        is True
    )
    real_results = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.content == "北京->上海:20"
    ]
    assert len(real_results) == 1


def test_duplicate_guard_allows_parallel_tool_batch_to_merge_at_stop_threshold():
    class ToolCallingFakeModel(FakeMessagesListChatModel):
        invocation_count: int = 0

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

        def _generate(self, *args, **kwargs):
            self.invocation_count += 1
            return super()._generate(*args, **kwargs)

    repeated_args = {"origin": "北京", "destination": "上海", "limit": 20}
    new_args = {"origin": "广州", "destination": "上海", "limit": 20}
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call-seed", "name": "demo_lookup", "args": repeated_args}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call-duplicate-1", "name": "demo_lookup", "args": repeated_args}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call-new", "name": "demo_lookup", "args": new_args},
                    {"id": "call-duplicate-2", "name": "demo_lookup", "args": repeated_args},
                    {"id": "call-duplicate-3", "name": "demo_lookup", "args": repeated_args},
                ],
            ),
            AIMessage(content="已基于现有工具结果完成汇总。"),
        ]
    )
    test_agent = create_agent(
        model=model,
        tools=[demo_lookup],
        middleware=[ParamAwareDuplicateToolCallGuard(loop_stop_after=3)],
    )

    result = test_agent.invoke(
        {"messages": [{"role": "user", "content": "执行并行查询"}]},
        config={"recursion_limit": 20},
    )

    assert model.invocation_count == 4
    assert result["messages"][-1].content == "已基于现有工具结果完成汇总。"
    stop_messages = [
        json.loads(message.content)
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.content.startswith("{")
        and "react_loop_stop_requested" in message.content
    ]
    assert len(stop_messages) == 1
    assert stop_messages[0]["stop_requested"] is True


def test_duplicate_guard_hard_stop_closes_every_pending_tool_call_protocol():
    guard = ParamAwareDuplicateToolCallGuard(loop_stop_after=3)

    def handler(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content="real result",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    guard.wrap_tool_call(_request(), handler)
    guard.wrap_tool_call(_request(), handler)
    guard.wrap_tool_call(_request(), handler)
    guard.wrap_tool_call(_request(), handler)
    repeated_args = {"origin": "北京", "destination": "上海", "limit": 20}
    pending_message = AIMessage(
        content="",
        tool_calls=[
            {"id": "call-pending-1", "name": "demo_lookup", "args": repeated_args},
            {
                "id": "call-pending-2",
                "name": "other_lookup",
                "args": {"origin": "广州", "destination": "上海", "limit": 20},
            },
        ],
    )

    update = guard.after_model(
        {"messages": [pending_message]},
        SimpleNamespace(
            context=Context(
                user_id="u1",
                thread_id="thread-1",
                request_id="request-1",
                run_id="run-1",
            )
        ),
    )

    assert update is not None
    assert update["jump_to"] == "end"
    tool_messages = [
        message for message in update["messages"] if isinstance(message, ToolMessage)
    ]
    assert {message.tool_call_id for message in tool_messages} == {
        "call-pending-1",
        "call-pending-2",
    }
