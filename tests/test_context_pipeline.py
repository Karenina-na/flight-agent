from types import SimpleNamespace

from langchain.agents.middleware import ModelRequest
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.prompt import CONTEXT_LEDGER_TOOL_NAME
from src.summarization.context_pipeline import build_context_pipeline_request


class FakeSummaryModel:
    def __init__(self, responses: list[str] | None = None, *, raises: bool = False):
        self.responses = responses or []
        self.raises = raises
        self.calls: list[object] = []

    def invoke(self, messages: object) -> AIMessage:
        self.calls.append(messages)
        if self.raises:
            raise RuntimeError("summary failed")
        content = self.responses.pop(0) if self.responses else '{"facts":["摘要事实"],"open_items":[],"evidence_refs":[]}'
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


def test_pipeline_uses_l4_local_semantic_summary_when_l1_l3_still_over_budget():
    summary_model = FakeSummaryModel(
        ['{"facts":["保留局部事实"],"open_items":["继续汇总"],"evidence_refs":["tool_call:call-1"]}']
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
    assert len(summary_model.calls) == 1
    assert "local_semantic_summary" in _ledger_content(result.request.messages)
    assert "保留局部事实" in _ledger_content(result.request.messages)


def test_pipeline_uses_l5_global_fallback_when_l4_still_over_budget():
    summary_model = FakeSummaryModel(
        [
            '{"facts":["局部事实"],"open_items":[],"evidence_refs":["message:0"]}',
            '{"facts":["全局事实"],"open_items":["待继续"],"evidence_refs":["summary:local"],"dropped_detail_notice":"细节已丢弃"}',
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
    assert "global_fallback_summary" in _ledger_content(result.request.messages)
    assert "全局事实" in _ledger_content(result.request.messages)
    assert "deterministic_history_omitted" in _ledger_content(result.request.messages)


def test_l5_global_fallback_omits_deterministic_ledger_details():
    summary_model = FakeSummaryModel(
        [
            '{"facts":["局部事实"],"open_items":[],"evidence_refs":["message:0"]}',
            '{"facts":["只保留全局事实"],"open_items":[],"evidence_refs":["summary:local"],"dropped_detail_notice":"明细已降级"}',
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
    assert "global_fallback_summary" in content
    assert "只保留全局事实" in content
    assert "deterministic_history_omitted" in content
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


def test_pipeline_preserves_todo_snapshot_outside_semantic_summary_input():
    summary_model = FakeSummaryModel(
        ['{"facts":["局部事实"],"open_items":[],"evidence_refs":["message:0"]}']
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
    assert "todo_snapshot" in content
    assert "汇总报价" in content
    assert "汇总报价" not in str(summary_model.calls[0])
