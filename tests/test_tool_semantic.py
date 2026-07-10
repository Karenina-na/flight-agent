import json

import pytest
from langchain.messages import AIMessage, ToolMessage

from src.summarization.tool_semantic import (
    ToolSummaryCandidate,
    build_tool_summary_candidates,
    chunk_tool_result,
    summarize_tool_candidates,
)


def test_chunk_tool_result_preserves_complete_json_records():
    payload = json.dumps(
        {
            "captured_at": "2026-07-10T13:04:18+08:00",
            "records": [
                {"id": f"record-{index}", "amount": index}
                for index in range(8)
            ],
        }
    )

    chunks = chunk_tool_result(payload, max_chars=180)

    parsed_chunks = [json.loads(chunk) for chunk in chunks]
    records = [
        record
        for chunk in parsed_chunks
        for record in chunk.get("records", [])
    ]
    assert [record["id"] for record in records] == [
        f"record-{index}" for index in range(8)
    ]
    assert all("captured_at" in chunk for chunk in parsed_chunks)


def test_build_tool_summary_candidates_keeps_full_tool_content():
    large_content = json.dumps(
        {"records": [{"amount": 1}], "padding": "x" * 1500}
    )
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "generic_lookup",
                    "args": {"scope": "all"},
                }
            ],
        ),
        ToolMessage(
            content=large_content,
            name="generic_lookup",
            tool_call_id="call-1",
        ),
    ]

    candidates = build_tool_summary_candidates(messages, min_chars=1200)

    assert len(candidates) == 1
    assert candidates[0].tool_name == "generic_lookup"
    assert candidates[0].args == {"scope": "all"}
    assert candidates[0].content == large_content


def test_summarize_tool_candidates_rejects_partial_number_matches():
    class FakeSummaryModel:
        def invoke(self, _messages):
            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        {
                            "facts": ["最低价格为 70 CNY"],
                            "omissions": [],
                        }
                    )
                },
            )()

    candidate = ToolSummaryCandidate(
        tool_call_id="call-1",
        tool_name="generic_lookup",
        args={"scope": "all"},
        content=json.dumps({"quotes": [{"price": 700}]}),
        result_stats={
            "arrays": {"quotes": {"length": 1}},
            "numbers": {"quotes[].price": {"min": 700, "max": 700}},
        },
    )

    with pytest.raises(
        ValueError,
        match="tool semantic summary contains no grounded facts",
    ):
        summarize_tool_candidates(
            FakeSummaryModel(),
            [candidate],
        )


def test_summarize_tool_candidates_ignores_reasoning_blocks_in_responses_content():
    class FakeSummaryModel:
        def invoke(self, _messages):
            return type(
                "Response",
                (),
                {
                    "content": [
                        {
                            "type": "reasoning",
                            "content": [
                                {
                                    "type": "reasoning_text",
                                    "text": "先分析工具结果。",
                                }
                            ],
                        },
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "facts": ["共 1 条报价，价格为 700 CNY"],
                                    "omissions": [],
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ]
                },
            )()

    candidate = ToolSummaryCandidate(
        tool_call_id="call-1",
        tool_name="generic_lookup",
        args={"scope": "all"},
        content=json.dumps({"quotes": [{"price": 700}]}),
        result_stats={
            "arrays": {"quotes": {"length": 1}},
            "numbers": {"quotes[].price": {"min": 700, "max": 700}},
        },
    )

    summaries = summarize_tool_candidates(
        FakeSummaryModel(),
        [candidate],
    )

    assert summaries["call-1"]["facts"] == ["共 1 条报价，价格为 700 CNY"]


def test_summarize_tool_candidates_does_not_include_global_user_task():
    class FakeSummaryModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        {
                            "facts": ["当前工具返回 1 条报价，价格为 700 CNY"],
                            "omissions": [],
                        },
                        ensure_ascii=False,
                    )
                },
            )()

    model = FakeSummaryModel()
    candidate = ToolSummaryCandidate(
        tool_call_id="call-1",
        tool_name="generic_lookup",
        args={"scope": "all"},
        content=json.dumps({"quotes": [{"price": 700}]}),
        result_stats={
            "arrays": {"quotes": {"length": 1}},
            "numbers": {"quotes[].price": {"min": 700, "max": 700}},
        },
    )

    summarize_tool_candidates(model, [candidate])

    prompt_text = "\n".join(message["content"] for message in model.calls[0])
    assert "连续查询12天" not in prompt_text
    assert "可直接支持当前用户目标" not in prompt_text
    assert "不判断整个用户任务是否完成" in prompt_text
    assert "未出现在当前工具结果中的任务" in prompt_text


def test_summarize_tool_candidates_emits_summary_lifecycle_events():
    class FakeSummaryModel:
        def invoke(self, _messages):
            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        {
                            "facts": ["当前工具返回 1 条报价，价格为 700 CNY"],
                            "omissions": [],
                        },
                        ensure_ascii=False,
                    )
                },
            )()

    candidate = ToolSummaryCandidate(
        tool_call_id="call-1",
        tool_name="generic_lookup",
        args={"scope": "all"},
        content=json.dumps({"quotes": [{"price": 700}]}),
        result_stats={
            "arrays": {"quotes": {"length": 1}},
            "numbers": {"quotes[].price": {"min": 700, "max": 700}},
        },
    )
    events = []

    summarize_tool_candidates(
        FakeSummaryModel(),
        [candidate],
        event_callback=lambda event, fields: events.append((event, fields)),
    )

    assert [event for event, _fields in events] == [
        "context_summary_start",
        "context_summary_end",
    ]
    assert events[0][1]["stage"] == "l3_tool_semantic"
    assert events[0][1]["tool_name"] == "generic_lookup"
    assert events[1][1]["status"] == "success"


def test_summarize_tool_candidates_emits_error_event_before_fallback():
    class FailingSummaryModel:
        def invoke(self, _messages):
            raise TimeoutError("summary timed out")

    candidate = ToolSummaryCandidate(
        tool_call_id="call-1",
        tool_name="generic_lookup",
        args={"scope": "all"},
        content=json.dumps({"quotes": [{"price": 700}]}),
        result_stats={
            "arrays": {"quotes": {"length": 1}},
            "numbers": {"quotes[].price": {"min": 700, "max": 700}},
        },
    )
    events = []

    with pytest.raises(TimeoutError, match="summary timed out"):
        summarize_tool_candidates(
            FailingSummaryModel(),
            [candidate],
            event_callback=lambda event, fields: events.append((event, fields)),
        )

    assert [event for event, _fields in events] == [
        "context_summary_start",
        "context_summary_error",
    ]
    assert events[1][1]["status"] == "error"
    assert events[1][1]["error_type"] == "TimeoutError"
