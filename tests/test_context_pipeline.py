import json
from types import SimpleNamespace

from langchain.agents.middleware import ModelRequest
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.prompt import CONTEXT_LEDGER_TOOL_NAME
from src.summarization.semantic_cache import SemanticSummaryCache
from src.summarization.context_pipeline import build_context_pipeline_request
from src.summarization.structured_output import SemanticSummaryCapability


class FakeSummaryModel:
    def __init__(self, responses: list[str] | None = None, *, raises: bool = False):
        self.responses = responses or []
        self.raises = raises
        self.calls: list[object] = []

    def invoke(self, messages: object) -> AIMessage:
        self.calls.append(messages)
        if self.raises:
            raise RuntimeError("summary failed")
        content = self.responses.pop(0) if self.responses else "摘要事实：保留历史上下文。"
        return AIMessage(content=content)


def _model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key="not-needed",
        model="qwen3.5-4b-mlx",
        profile={"max_input_tokens": 8192},
    )


def _request(*, messages: list, state: dict | None = None) -> ModelRequest:
    return ModelRequest(
        model=_model(),
        messages=messages,
        system_prompt="system" * 100,
        tools=[{"name": "lookup", "description": "tool" * 50}],
        state=state,
        runtime=SimpleNamespace(context=None),
    )


def _tool_message(content: str, call_id: str = "call-1") -> ToolMessage:
    return ToolMessage(content=content, name="lookup", tool_call_id=call_id)


def _ledger_content(messages: list) -> str:
    ledger_index = next(
        index
        for index, message in enumerate(messages)
        if isinstance(message, AIMessage)
        and message.tool_calls
        and message.tool_calls[0]["name"] == CONTEXT_LEDGER_TOOL_NAME
    )
    return messages[ledger_index + 1].content


def test_pipeline_skips_semantic_model_when_l1_l3_fits_budget():
    summary_model = FakeSummaryModel()
    request = _request(
        messages=[
            HumanMessage(content="查询并汇总"),
            _tool_message('{"records":[{"amount":1}]}'),
        ]
    )

    result = build_context_pipeline_request(
        request,
        latest_human_text="查询并汇总",
        estimate_chars=5000,
        threshold_chars=2000,
        ledger_fraction=0.25,
        min_ledger_budget_chars=12000,
        raw_recent_turns=2,
        estimate_request_chars=lambda request: 1000,
        semantic_enabled=True,
        summary_model=summary_model,
    )

    assert result is not None
    assert result.compaction_level == "l1_l3"
    assert summary_model.calls == []
    assert result.semantic_summary_count == 0
    assert result.global_fallback_used is False
    assert result.semantic_skip_reason == "within_budget_after_l1_l3"


def test_pipeline_uses_l3_semantic_summary_for_large_tool_result():
    summary_model = FakeSummaryModel(
        [
            json.dumps(
                {
                    "facts": [
                        "共 3 条记录",
                        "amount 范围为 1 至 9",
                        "记录标识为 A、B、C",
                    ],
                    "omissions": ["省略了无关的 padding 文本"],
                },
                ensure_ascii=False,
            )
        ]
    )
    large_result = json.dumps(
        {
            "captured_at": "2026-07-10T13:04:18+08:00",
            "records": [
                {"id": "A", "amount": 1},
                {"id": "B", "amount": 5},
                {"id": "C", "amount": 9},
            ],
            "padding": "x" * 1800,
        },
        ensure_ascii=False,
    )
    request = _request(
        messages=[
            HumanMessage(content="汇总这些工具结果"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "lookup",
                        "args": {"scope": "all", "limit": 3},
                    }
                ],
            ),
            _tool_message(large_result),
        ]
    )
    estimates = iter([1500, 2500, 1200])

    result = build_context_pipeline_request(
        request,
        latest_human_text="汇总这些工具结果",
        estimate_chars=5000,
        threshold_chars=2000,
        ledger_fraction=0.25,
        min_ledger_budget_chars=12000,
        raw_recent_turns=2,
        estimate_request_chars=lambda request: next(estimates),
        semantic_enabled=True,
        summary_model=summary_model,
    )

    assert result is not None
    assert result.compaction_level == "l3_tool_semantic"
    assert result.tool_semantic_summary_count == 1
    assert result.tool_semantic_summary_failed is False
    assert len(summary_model.calls) == 1
    content = _ledger_content(result.request.messages)
    assert "共 3 条记录" in content
    assert "amount 范围为 1 至 9" in content
    assert "省略了无关的 padding 文本" in content
    assert '"scope":"all"' in content
    assert '"limit":3' in content
    assert "x" * 100 not in content
    assert "result_shape" not in content
    assert "result_stats" not in content


def test_pipeline_preserves_full_tool_result_before_l3_summary_when_budget_allows():
    summary_model = FakeSummaryModel(raises=True)
    large_result = json.dumps(
        {
            "records": [
                {"id": "A", "amount": 1},
                {"id": "B", "amount": 5},
                {"id": "C", "amount": 9},
            ],
            "padding": "x" * 1300,
            "tail_marker": "COMPLETE_TOOL_RESULT_RETAINED",
        },
        ensure_ascii=False,
    )
    request = _request(
        messages=[
            HumanMessage(content="汇总这些工具结果"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "lookup",
                        "args": {"scope": "all"},
                    }
                ],
            ),
            _tool_message(large_result),
        ]
    )
    estimates = iter([1500, 1900])

    result = build_context_pipeline_request(
        request,
        latest_human_text="汇总这些工具结果",
        estimate_chars=5000,
        threshold_chars=2000,
        ledger_fraction=0.25,
        min_ledger_budget_chars=12000,
        raw_recent_turns=2,
        estimate_request_chars=lambda request: next(estimates),
        semantic_enabled=True,
        summary_model=summary_model,
    )

    assert result is not None
    assert result.compaction_level == "l3_lossless_preserved"
    assert result.tool_semantic_summary_failed is False
    assert result.semantic_error_type is None
    assert summary_model.calls == []
    content = _ledger_content(result.request.messages)
    assert "COMPLETE_TOOL_RESULT_RETAINED" in content
    assert '"scope":"all"' in content


def test_pipeline_uses_l4_local_semantic_summary_when_l1_l3_still_over_budget():
    summary_model = FakeSummaryModel(
        ["保留局部事实；仍需继续汇总。"]
    )
    request = _request(
        messages=[
            HumanMessage(content="查询第一批"),
            _tool_message('{"records":[{"amount":1}]}', "call-1"),
            HumanMessage(content="现在汇总"),
        ]
    )
    estimates = iter([3000, 1000])

    result = build_context_pipeline_request(
        request,
        latest_human_text="现在汇总",
        estimate_chars=5000,
        threshold_chars=2000,
        ledger_fraction=0.25,
        min_ledger_budget_chars=12000,
        raw_recent_turns=1,
        estimate_request_chars=lambda request: next(estimates),
        semantic_enabled=True,
        summary_model=summary_model,
    )

    assert result is not None
    assert result.compaction_level == "l4_local_semantic"
    assert result.semantic_summary_count == 1
    assert result.semantic_summary_failed is False
    assert result.semantic_skip_reason is None
    assert len(summary_model.calls) == 1
    content = _ledger_content(result.request.messages)
    assert "### 历史摘要" in content
    assert "保留局部事实" in content
    assert "local_semantic_summary" not in content
    assert "evidence_refs" not in content


def test_pipeline_invokes_l4_summary_model_directly_without_schema():
    class SummaryModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content="保留局部事实；继续汇总。")

    summary_model = SummaryModel()
    request = _request(
        messages=[
            HumanMessage(content="查询第一批"),
            _tool_message('{"records":[{"amount":1}]}', "call-1"),
            HumanMessage(content="现在汇总"),
        ]
    )
    estimates = iter([3000, 1000])

    result = build_context_pipeline_request(
        request,
        latest_human_text="现在汇总",
        estimate_chars=5000,
        threshold_chars=2000,
        ledger_fraction=0.25,
        min_ledger_budget_chars=12000,
        raw_recent_turns=1,
        estimate_request_chars=lambda request: next(estimates),
        semantic_enabled=True,
        summary_model=summary_model,
    )

    assert result is not None
    assert result.compaction_level == "l4_local_semantic"
    assert len(summary_model.calls) == 1
    prompt_text = "\n".join(message["content"] for message in summary_model.calls[0])
    assert "输出 JSON schema" not in prompt_text
    assert "直接输出摘要正文" in prompt_text


def test_pipeline_disables_repeated_semantic_calls_after_reasoning_only_output():
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
                                "text": "只生成推理，没有最终摘要。",
                            }
                        ],
                    }
                ]
            )

    large_result = json.dumps(
        {
            "records": [{"id": "A", "amount": 700}],
            "padding": "x" * 1600,
        }
    )
    request = _request(
        messages=[
            HumanMessage(content="汇总工具结果"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "lookup",
                        "args": {"scope": "all"},
                    }
                ],
            ),
            _tool_message(large_result),
        ]
    )
    model = ReasoningOnlySummaryModel()
    capability = SemanticSummaryCapability()

    def compact_once():
        estimates = iter([1500, 2500])
        return build_context_pipeline_request(
            request,
            latest_human_text="汇总工具结果",
            estimate_chars=5000,
            threshold_chars=2000,
            ledger_fraction=0.25,
            min_ledger_budget_chars=12000,
            raw_recent_turns=1,
            estimate_request_chars=lambda _request: next(estimates),
            semantic_enabled=True,
            summary_model=model,
            summary_capability=capability,
        )

    first_result = compact_once()
    second_result = compact_once()

    assert first_result is not None
    assert second_result is not None
    assert len(model.calls) == 1
    assert first_result.semantic_summary_unavailable is True
    assert second_result.semantic_summary_unavailable is True
    assert first_result.semantic_unavailable_reason == "reasoning_only_output"
    assert second_result.semantic_unavailable_reason == "reasoning_only_output"
    assert first_result.semantic_skip_reason == "semantic_unavailable"
    assert second_result.semantic_skip_reason == "semantic_unavailable"


def test_pipeline_uses_l5_global_fallback_when_l4_still_over_budget():
    summary_model = FakeSummaryModel(
        [
            "局部事实。",
            "全局事实；待继续处理。",
        ]
    )
    request = _request(
        messages=[
            HumanMessage(content="查询第一批"),
            _tool_message('{"records":[{"amount":1}]}', "call-1"),
            HumanMessage(content="现在汇总"),
        ]
    )
    estimates = iter([3000, 2800, 1000])

    result = build_context_pipeline_request(
        request,
        latest_human_text="现在汇总",
        estimate_chars=5000,
        threshold_chars=2000,
        ledger_fraction=0.25,
        min_ledger_budget_chars=12000,
        raw_recent_turns=1,
        estimate_request_chars=lambda request: next(estimates),
        semantic_enabled=True,
        summary_model=summary_model,
    )

    assert result is not None
    assert result.compaction_level == "l5_global_fallback"
    assert result.semantic_summary_count == 2
    assert result.global_fallback_used is True
    assert len(summary_model.calls) == 2
    content = _ledger_content(result.request.messages)
    assert "全局事实" in content
    assert "较早的逐条工具结果已因上下文预算折叠" in content
    assert "global_fallback_summary" not in content
    assert "deterministic_history_omitted" not in content


def test_l5_global_fallback_omits_deterministic_ledger_details():
    summary_model = FakeSummaryModel(
        [
            "局部事实。",
            "只保留全局事实。",
        ]
    )
    request = _request(
        messages=[
            HumanMessage(content="查询第一批"),
            _tool_message('{"records":[{"secret_detail":"不能进入L5最终上下文"}]}', "call-1"),
            HumanMessage(content="现在汇总"),
        ]
    )
    estimates = iter([3000, 2800, 1000])

    result = build_context_pipeline_request(
        request,
        latest_human_text="现在汇总",
        estimate_chars=5000,
        threshold_chars=2000,
        ledger_fraction=0.25,
        min_ledger_budget_chars=12000,
        raw_recent_turns=1,
        estimate_request_chars=lambda request: next(estimates),
        semantic_enabled=True,
        summary_model=summary_model,
    )

    assert result is not None
    assert result.compaction_level == "l5_global_fallback"
    content = _ledger_content(result.request.messages)
    assert "只保留全局事实" in content
    assert "较早的逐条工具结果已因上下文预算折叠" in content
    assert "global_fallback_summary" not in content
    assert "deterministic_history_omitted" not in content
    assert "secret_detail" not in content
    assert "call-1" not in content


def test_pipeline_falls_back_to_l1_l3_when_summary_model_fails():
    summary_model = FakeSummaryModel(raises=True)
    request = _request(
        messages=[
            HumanMessage(content="查询第一批"),
            _tool_message('{"records":[{"amount":1}]}', "call-1"),
            HumanMessage(content="现在汇总"),
        ]
    )

    result = build_context_pipeline_request(
        request,
        latest_human_text="现在汇总",
        estimate_chars=5000,
        threshold_chars=2000,
        ledger_fraction=0.25,
        min_ledger_budget_chars=12000,
        raw_recent_turns=1,
        estimate_request_chars=lambda request: 3000,
        semantic_enabled=True,
        summary_model=summary_model,
    )

    assert result is not None
    assert result.compaction_level == "l1_l3"
    assert result.semantic_summary_failed is True
    assert "local_semantic_summary" not in _ledger_content(result.request.messages)


def test_pipeline_accepts_non_schema_summary_text_content():
    summary_model = FakeSummaryModel(["局部事实。"])
    events: list[tuple[str, dict]] = []
    request = _request(
        messages=[
            HumanMessage(content="查询第一批"),
            _tool_message('{"records":[{"amount":1}]}', "call-1"),
            HumanMessage(content="现在汇总"),
        ]
    )
    estimates = iter([3000, 1000])

    result = build_context_pipeline_request(
        request,
        latest_human_text="现在汇总",
        estimate_chars=5000,
        threshold_chars=2000,
        ledger_fraction=0.25,
        min_ledger_budget_chars=12000,
        raw_recent_turns=1,
        estimate_request_chars=lambda request: next(estimates),
        semantic_enabled=True,
        summary_model=summary_model,
        summary_event_callback=lambda event, fields: events.append((event, fields)),
    )

    assert result is not None
    assert result.semantic_summary_failed is False
    assert result.semantic_summary_unavailable is False
    assert result.compaction_level == "l4_local_semantic"
    assert [event for event, _ in events] == [
        "context_summary_start",
        "context_summary_end",
    ]
    assert events[-1][1]["stage"] == "local_semantic_summary"
    assert events[-1][1]["summary_content"] == "局部事实。"


def test_pipeline_reuses_cached_l4_summary_for_same_compacted_context():
    summary_model = FakeSummaryModel(["局部事实 1。", "局部事实 2。"])
    cache = SemanticSummaryCache(max_items=16)
    request = _request(
        messages=[
            HumanMessage(content="查询第一批"),
            _tool_message('{"records":[{"amount":1}]}', "call-1"),
            HumanMessage(content="现在汇总"),
        ]
    )
    events: list[tuple[str, dict]] = []

    def compact_once():
        estimates = iter([3000, 1000])
        return build_context_pipeline_request(
            request,
            latest_human_text="现在汇总",
            estimate_chars=5000,
            threshold_chars=2000,
            ledger_fraction=0.25,
            min_ledger_budget_chars=12000,
            raw_recent_turns=1,
            estimate_request_chars=lambda _request: next(estimates),
            semantic_enabled=True,
            summary_model=summary_model,
            summary_cache=cache,
            summary_event_callback=lambda event, fields: events.append((event, fields)),
        )

    first = compact_once()
    second = compact_once()

    assert first is not None
    assert second is not None
    assert len(summary_model.calls) == 1
    assert "局部事实 1。" in _ledger_content(first.request.messages)
    assert "局部事实 1。" in _ledger_content(second.request.messages)
    end_events = [
        fields
        for event, fields in events
        if event == "context_summary_end"
    ]
    assert [fields["cached"] for fields in end_events] == [False, True]
    assert end_events[0]["cache_key"] == end_events[1]["cache_key"]


def test_pipeline_reuses_cached_l5_summary_for_same_compacted_context():
    summary_model = FakeSummaryModel(
        [
            "局部事实。",
            "全局事实 1。",
            "不应生成的新局部事实。",
            "全局事实 2。",
        ]
    )
    cache = SemanticSummaryCache(max_items=16)
    request = _request(
        messages=[
            HumanMessage(content="查询第一批"),
            _tool_message('{"records":[{"amount":1}]}', "call-1"),
            HumanMessage(content="现在汇总"),
        ]
    )
    events: list[tuple[str, dict]] = []

    def compact_once():
        estimates = iter([3000, 2800, 1000])
        return build_context_pipeline_request(
            request,
            latest_human_text="现在汇总",
            estimate_chars=5000,
            threshold_chars=2000,
            ledger_fraction=0.25,
            min_ledger_budget_chars=12000,
            raw_recent_turns=1,
            estimate_request_chars=lambda _request: next(estimates),
            semantic_enabled=True,
            summary_model=summary_model,
            summary_cache=cache,
            summary_event_callback=lambda event, fields: events.append((event, fields)),
        )

    first = compact_once()
    second = compact_once()

    assert first is not None
    assert second is not None
    assert first.compaction_level == "l5_global_fallback"
    assert second.compaction_level == "l5_global_fallback"
    assert len(summary_model.calls) == 2
    assert "全局事实 1。" in _ledger_content(first.request.messages)
    assert "全局事实 1。" in _ledger_content(second.request.messages)
    assert "全局事实 2。" not in _ledger_content(second.request.messages)
    end_events = [
        fields
        for event, fields in events
        if event == "context_summary_end"
    ]
    assert [fields["cached"] for fields in end_events] == [
        False,
        False,
        True,
        True,
    ]


def test_pipeline_preserves_todo_snapshot_outside_semantic_summary_input():
    summary_model = FakeSummaryModel(
        ["局部事实。"]
    )
    request = _request(
        messages=[
            HumanMessage(content="查询第一批"),
            _tool_message('{"records":[{"amount":1}]}', "call-1"),
            HumanMessage(content="现在汇总"),
        ],
        state={"todos": [{"content": "汇总报价", "status": "in_progress"}]},
    )
    estimates = iter([3000, 1000])

    result = build_context_pipeline_request(
        request,
        latest_human_text="现在汇总",
        estimate_chars=5000,
        threshold_chars=2000,
        ledger_fraction=0.25,
        min_ledger_budget_chars=12000,
        raw_recent_turns=1,
        estimate_request_chars=lambda request: next(estimates),
        semantic_enabled=True,
        summary_model=summary_model,
    )

    assert result is not None
    content = _ledger_content(result.request.messages)
    assert "### 任务进度" in content
    assert "[进行中]" in content
    assert "汇总报价" in content
    assert "todo_snapshot" not in content
    assert "汇总报价" not in str(summary_model.calls[0])
