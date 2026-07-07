from langchain.messages import AIMessage, HumanMessage, ToolMessage

from src.guardrails.layered_context import (
    build_layered_context_state,
    has_compressible_history,
)


def test_layered_context_preserves_latest_user_outside_summary():
    messages = [
        HumanMessage(content="第一轮：查询北京到上海"),
        AIMessage(content="已经完成第一轮查询。"),
        HumanMessage(content="第二轮：继续查询上海到北京"),
    ]

    state = build_layered_context_state(messages, budget_chars=8000)
    data = state.to_dict()

    assert data["strategy"] == "layered_context_state"
    assert data["system_prompt_policy"] == "external_full_preserve"
    assert data["latest_user_message_policy"] == "external_full_preserve"
    assert state.old_user_message_count == 1
    assert state.old_user_messages[0]["content_summary"] == "第一轮：查询北京到上海"
    assert "第二轮：继续查询上海到北京" not in state.to_prompt_text()


def test_layered_context_summarizes_assistant_visible_state_and_tool_calls():
    messages = [
        HumanMessage(content="查询未来三天"),
        AIMessage(
            content=[
                {"type": "reasoning", "text": "内部推理不应进入压缩账本"},
                {"type": "text", "text": "我将调用报价工具并汇总。"},
            ],
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "search_airfare_quotes",
                    "args": {"origin": "北京", "destination": "上海"},
                }
            ],
        ),
        ToolMessage(
            content='{"query":{"origin":"北京","destination":"上海"},"quotes":[{"price":500}]}',
            name="search_airfare_quotes",
            tool_call_id="call-1",
        ),
        HumanMessage(content="请继续下一天"),
    ]

    state = build_layered_context_state(messages, budget_chars=8000)
    text = state.to_prompt_text()

    assert state.assistant_message_count == 1
    assert "我将调用报价工具并汇总" in text
    assert "内部推理不应进入压缩账本" not in text
    assert state.assistant_messages[0]["tool_calls"][0]["name"] == "search_airfare_quotes"
    assert state.observation_count == 1
    assert "call-1" in text
    assert '"origin": "北京"' in text


def test_layered_context_keeps_tool_observations_before_dropping_message_cards():
    messages = [
        HumanMessage(content="批量查询多个对象"),
        *[
            AIMessage(content=f"第 {index} 次查询完成。" + "长文本" * 80)
            for index in range(8)
        ],
        *[
            ToolMessage(
                content='{"query":{"slot":%d},"records":[{"amount":%d}]}' % (index, index),
                name="generic_lookup",
                tool_call_id=f"call-{index}",
            )
            for index in range(6)
        ],
        HumanMessage(content="请汇总已经查询的对象"),
    ]

    state = build_layered_context_state(messages, budget_chars=2600, message_preview_chars=500)
    text = state.to_prompt_text()

    assert state.assistant_message_count == 8
    assert state.dropped_assistant_message_count > 0
    assert state.observation_count == 6
    assert state.preserved_observation_count == 6
    for index in range(6):
        assert f"call-{index}" in text
        assert f'"slot": {index}' in text


def test_has_compressible_history_requires_more_than_latest_user_message():
    assert not has_compressible_history([HumanMessage(content="最新问题")])
    assert has_compressible_history(
        [
            HumanMessage(content="旧问题"),
            AIMessage(content="旧回答"),
            HumanMessage(content="最新问题"),
        ]
    )
