import json

import pytest
from langchain.messages import AIMessage, ToolMessage

from src.summarization.tool_semantic import (
    ToolSummaryCandidate,
    build_tool_summary_candidates,
    chunk_tool_result,
    summarize_tool_candidates,
)
from src.summarization.semantic_cache import SemanticSummaryCache
from src.summarization.structured_output import (
    SemanticSummaryCapability,
    SemanticSummaryUnavailableError,
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


def test_summarize_tool_candidates_keeps_plain_text_summary_content():
    class FakeSummaryModel:
        def invoke(self, _messages):
            return type(
                "Response",
                (),
                {"content": "当前工具结果显示 1 条报价，价格为 700 CNY。"},
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

    summaries = summarize_tool_candidates(FakeSummaryModel(), [candidate])

    assert summaries == {
        "call-1": {
            "content": "当前工具结果显示 1 条报价，价格为 700 CNY。",
        }
    }


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
                            "text": "共 1 条报价，价格为 700 CNY。",
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

    assert summaries["call-1"]["content"] == "共 1 条报价，价格为 700 CNY。"


def test_summarize_tool_candidates_invokes_model_directly_without_schema():
    class SummaryModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content="当前工具返回 1 条报价，价格为 700 CNY。")

    model = SummaryModel()
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

    summaries = summarize_tool_candidates(model, [candidate])

    assert summaries["call-1"]["content"] == "当前工具返回 1 条报价，价格为 700 CNY。"
    assert len(model.calls) == 1
    prompt_text = "\n".join(message["content"] for message in model.calls[0])
    assert "输出 schema" not in prompt_text
    assert "不要输出 JSON" in prompt_text


def test_summarize_tool_candidates_marks_reasoning_only_output_unavailable():
    class ReasoningOnlySummaryModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(
                content=[
                    {
                        "type": "reasoning",
                        "content": [
                            {
                                "type": "reasoning_text",
                                "text": "分析工具结果，但没有生成最终摘要。",
                            }
                        ],
                    }
                ]
            )

    model = ReasoningOnlySummaryModel()
    capability = SemanticSummaryCapability()
    events = []
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
        SemanticSummaryUnavailableError,
        match="reasoning_only_output",
    ):
        summarize_tool_candidates(
            model,
            [candidate],
            summary_capability=capability,
            event_callback=lambda event, fields: events.append((event, fields)),
        )

    assert capability.available is False
    assert capability.reason == "reasoning_only_output"
    assert len(model.calls) == 1
    assert [event for event, _fields in events] == [
        "context_summary_start",
        "context_summary_unavailable",
    ]
    assert events[-1][1]["fallback"] == "deterministic_compaction"


def test_summarize_tool_candidates_marks_empty_output_unavailable():
    class EmptySummaryModel:
        def invoke(self, _messages):
            return AIMessage(content="")

    capability = SemanticSummaryCapability()
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
        SemanticSummaryUnavailableError,
        match="empty_visible_output",
    ):
        summarize_tool_candidates(
            EmptySummaryModel(),
            [candidate],
            summary_capability=capability,
        )

    assert capability.available is False
    assert capability.reason == "empty_visible_output"


def test_summarize_tool_candidates_does_not_include_global_user_task():
    class FakeSummaryModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return type(
                "Response",
                (),
                {"content": "当前工具返回 1 条报价，价格为 700 CNY。"},
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
    assert "只总结当前一次工具调用" in prompt_text


def test_summarize_tool_candidates_emits_summary_lifecycle_events():
    class FakeSummaryModel:
        def invoke(self, _messages):
            return type(
                "Response",
                (),
                {"content": "当前工具返回 1 条报价，价格为 700 CNY。"},
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
    assert events[1][1]["summary_content"] == "当前工具返回 1 条报价，价格为 700 CNY。"


def test_summarize_tool_candidates_reuses_cached_summary_for_same_input():
    class FakeSummaryModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content=f"缓存摘要 {len(self.calls)}")

    model = FakeSummaryModel()
    cache = SemanticSummaryCache(max_items=16)
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

    first = summarize_tool_candidates(
        model,
        [candidate],
        summary_cache=cache,
        event_callback=lambda event, fields: events.append((event, fields)),
    )
    second = summarize_tool_candidates(
        model,
        [candidate],
        summary_cache=cache,
        event_callback=lambda event, fields: events.append((event, fields)),
    )

    assert first == second == {"call-1": {"content": "缓存摘要 1"}}
    assert len(model.calls) == 1
    end_events = [
        fields
        for event, fields in events
        if event == "context_summary_end"
    ]
    assert [fields["cached"] for fields in end_events] == [False, True]
    assert end_events[0]["cache_key"] == end_events[1]["cache_key"]


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
