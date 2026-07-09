from types import SimpleNamespace

from langchain.messages import HumanMessage, ToolMessage

from src.summarization.context_compaction import (
    DEFAULT_TODO_SNAPSHOT_MAX_CONTENT_CHARS,
    DEFAULT_TODO_SNAPSHOT_MAX_ITEMS,
    build_todo_snapshot_from_request,
    project_layer_one_messages,
)


def test_layer_one_merges_adjacent_human_messages():
    projection = project_layer_one_messages(
        [
            HumanMessage(content="第一段用户补充"),
            HumanMessage(content="第二段用户补充"),
            ToolMessage(
                content='{"status":"ok"}',
                name="lookup",
                tool_call_id="call-1",
            ),
        ]
    )

    assert [message.type for message in projection.messages] == ["human", "tool"]
    assert projection.messages[0].content == "第一段用户补充\n\n第二段用户补充"
    assert projection.adjacent_human_merged_count == 1


def test_layer_one_replaces_duplicate_tool_outputs_with_short_markers():
    duplicate_content = '{"records":[{"id":1}]}'

    projection = project_layer_one_messages(
        [
            ToolMessage(
                content=duplicate_content,
                name="lookup",
                tool_call_id="call-1",
            ),
            ToolMessage(
                content=duplicate_content,
                name="lookup",
                tool_call_id="call-2",
            ),
        ]
    )

    assert len(projection.messages) == 2
    assert projection.messages[0].content == duplicate_content
    assert projection.messages[1].content.startswith(
        '{"status":"compacted_duplicate_tool_output"'
    )
    assert '"duplicate_of_tool_call_id":"call-1"' in projection.messages[1].content
    assert projection.duplicate_tool_output_count == 1


def test_layer_one_replaces_empty_tool_outputs_with_short_markers():
    projection = project_layer_one_messages(
        [
            ToolMessage(
                content='{"quotes":[],"limitations":[]}',
                name="search_airfare_quotes",
                tool_call_id="call-empty",
            ),
        ]
    )

    assert projection.messages[0].content.startswith(
        '{"status":"compacted_empty_tool_output"'
    )
    assert '"tool_call_id":"call-empty"' in projection.messages[0].content
    assert projection.empty_tool_output_count == 1


def test_build_todo_snapshot_keeps_only_order_content_and_status():
    request = SimpleNamespace(
        state={
            "todos": [
                {
                    "content": "查询第一批航线报价",
                    "status": "completed",
                    "extra": "do not expose",
                },
                {"content": "  ", "status": "pending"},
                {"content": "汇总结果", "status": "in_progress", "raw": {"x": 1}},
            ]
        }
    )

    snapshot = build_todo_snapshot_from_request(request)

    assert snapshot is not None
    assert snapshot["type"] == "todo_snapshot"
    assert snapshot["items"] == [
        {"index": 0, "content": "查询第一批航线报价", "status": "completed"},
        {"index": 2, "content": "汇总结果", "status": "in_progress"},
    ]
    assert "extra" not in str(snapshot)
    assert "raw" not in str(snapshot)


def test_build_todo_snapshot_returns_none_for_missing_or_invalid_state():
    assert build_todo_snapshot_from_request(SimpleNamespace()) is None
    assert build_todo_snapshot_from_request(SimpleNamespace(state={"todos": "bad"})) is None
    assert build_todo_snapshot_from_request(
        SimpleNamespace(state={"todos": [{"content": "", "status": "pending"}]})
    ) is None


def test_build_todo_snapshot_bounds_items_and_content_length():
    long_content = "任务内容" * 200
    request = SimpleNamespace(
        state={
            "todos": [
                {"content": f"{long_content}-{index}", "status": "pending"}
                for index in range(DEFAULT_TODO_SNAPSHOT_MAX_ITEMS + 3)
            ]
        }
    )

    snapshot = build_todo_snapshot_from_request(request)

    assert snapshot is not None
    assert len(snapshot["items"]) == DEFAULT_TODO_SNAPSHOT_MAX_ITEMS
    assert snapshot["total_count"] == DEFAULT_TODO_SNAPSHOT_MAX_ITEMS + 3
    assert snapshot["dropped_count"] == 3
    assert snapshot["truncated_count"] == DEFAULT_TODO_SNAPSHOT_MAX_ITEMS
    assert len(snapshot["items"][0]["content"]) <= DEFAULT_TODO_SNAPSHOT_MAX_CONTENT_CHARS + 3
    assert snapshot["items"][0]["content"].endswith("...")
