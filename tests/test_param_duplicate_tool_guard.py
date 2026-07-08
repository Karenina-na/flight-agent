import json

from langchain.agents.middleware import ToolCallRequest
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool

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
    assert second.status == "success"
    assert second.name == "demo_lookup"
    assert second.tool_call_id == "call-demo_lookup-request-1"
    assert payload["status"] == "duplicate_blocked"
    assert payload["duplicate_count"] == 1
    assert "previous tool result" in payload["message"]


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


def test_duplicate_guard_requests_loop_stop_after_repeated_duplicates():
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
    payload = json.loads(response.content)

    assert calls == 1
    assert payload["status"] == "react_loop_stop_requested"
    assert payload["duplicate_count"] == 3
    assert payload["loop_stop_after"] == 3
    assert "Stop calling tools" in payload["message"]
