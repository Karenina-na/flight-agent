from langchain.messages import HumanMessage, ToolMessage

from src.summarization.context_compaction import project_layer_one_messages


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
