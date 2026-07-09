from langchain.messages import AIMessage, ToolMessage

from src.summarization.tool_observation import (
    build_tool_observations,
    compact_tool_observations,
    json_shape_summary,
    json_stats_summary,
)


def test_build_tool_observations_records_generic_tool_cards():
    messages = [
        ToolMessage(
            content='{"items":[{"name":"alpha","score":3},{"name":"beta","score":7}],"ok":true}',
            name="demo_tool",
            tool_call_id="call-1",
        )
    ]

    observations = build_tool_observations(messages)

    assert len(observations) == 1
    observation = observations[0]
    assert observation.tool_name == "demo_tool"
    assert observation.tool_call_id == "call-1"
    assert observation.status == "success"
    assert observation.result_shape["type"] == "object"
    assert observation.result_shape["keys"] == ["items", "ok"]
    assert observation.result_stats["arrays"]["items"]["length"] == 2
    assert observation.result_stats["numbers"]["items[].score"] == {"min": 3, "max": 7}
    assert observation.content_sha256


def test_build_tool_observations_prefers_original_tool_call_args():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "demo_tool",
                    "args": {"requested_slot": "from-tool-call"},
                }
            ],
        ),
        ToolMessage(
            content='{"query":{"requested_slot":"from-result"},"items":[]}',
            name="demo_tool",
            tool_call_id="call-1",
        ),
    ]

    observations = build_tool_observations(messages)

    assert len(observations) == 1
    assert observations[0].args == {"requested_slot": "from-tool-call"}


def test_json_summaries_do_not_depend_on_business_fields():
    payload = {
        "records": [
            {"category": "A", "amount": 10},
            {"category": "B", "amount": 20},
        ],
        "note": "sample",
    }

    shape = json_shape_summary(payload)
    stats = json_stats_summary(payload)

    assert shape["keys"] == ["note", "records"]
    assert shape["children"]["records"]["length"] == 2
    assert stats["arrays"]["records"]["length"] == 2
    assert stats["numbers"]["records[].amount"] == {"min": 10, "max": 20}
    assert stats["strings"]["records[].category"] == ["A", "B"]


def test_compact_tool_observations_preserves_all_cards_before_truncating_previews():
    observations = build_tool_observations(
        [
            ToolMessage(
                content=(
                    '{"query":{"slot":%d},"items":[{"value":%d}],'
                    '"large_text":"%s"}' % (index, index, "x" * 200)
                ),
                name="generic_lookup",
                tool_call_id=f"call-{index}",
            )
            for index in range(10)
        ]
    )

    ledger = compact_tool_observations(observations, budget_chars=4000, preview_chars=24)

    assert ledger.observation_count == 10
    assert ledger.preserved_observation_count == 10
    assert ledger.dropped_observation_count == 0
    assert ledger.preview_truncated_count == 10
    assert len(ledger.observations) == 10
    assert "call-0" in ledger.to_prompt_text()
    assert "call-9" in ledger.to_prompt_text()


def test_compact_tool_observations_drops_oldest_cards_only_when_budget_is_tiny():
    observations = build_tool_observations(
        [
            ToolMessage(
                content='{"value":%d}' % index,
                name="generic_lookup",
                tool_call_id=f"call-{index}",
            )
            for index in range(20)
        ]
    )

    ledger = compact_tool_observations(observations, budget_chars=1200, preview_chars=0)

    assert ledger.observation_count == 20
    assert ledger.preserved_observation_count < 20
    assert ledger.dropped_observation_count > 0
    assert ledger.observations[0]["tool_call_id"] != "call-0"
    assert "dropped_observation_count" in ledger.to_prompt_text()


def test_build_tool_observations_expands_batch_tool_results_into_task_cards():
    messages = [
        ToolMessage(
            content=(
                '{"batch_id":"batch_1",'
                '"summary":{"total_requested":2,"executed":2,"success":1,"empty":1,"failed":0,"skipped":0},'
                '"results":['
                '{"task_id":"d1","tool_name":"search_airfare_quotes","status":"success",'
                '"args":{"origin":"北京","destination":"上海","departure_date":"2026-07-10"},'
                '"result_shape":{"type":"object","keys":["query","quotes"]},'
                '"result_stats":{"arrays":{"quotes":{"length":2}},"numbers":{"quotes[].price":{"min":400,"max":700}}},'
                '"result_preview":{"quotes_count":2},"content_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},'
                '{"task_id":"d2","tool_name":"search_airfare_quotes","status":"empty",'
                '"args":{"origin":"北京","destination":"上海","departure_date":"2026-07-11"},'
                '"result_shape":{"type":"object","keys":["query","quotes"]},'
                '"result_stats":{"arrays":{"quotes":{"length":0}}},'
                '"result_preview":{"quotes_count":0},"content_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}'
                '],'
                '"limitations":["bounded summary"]}'
            ),
            name="run_tool_batch",
            tool_call_id="batch-call-1",
        )
    ]

    observations = build_tool_observations(messages)

    assert len(observations) == 1
    observation = observations[0]
    assert observation.tool_name == "run_tool_batch"
    assert observation.result_shape["batch_tool_result"] is True
    assert observation.result_shape["task_count"] == 2
    assert observation.result_stats["batch_summary"]["executed"] == 2
    assert observation.result_stats["batch_task_count"] == 2
    task_cards = observation.result_stats["batch_task_cards"]
    assert task_cards[0]["task_id"] == "d1"
    assert task_cards[0]["args"]["departure_date"] == "2026-07-10"
    assert task_cards[0]["result_stats"]["numbers"]["quotes[].price"] == {"min": 400, "max": 700}
    assert task_cards[1]["task_id"] == "d2"
    assert task_cards[1]["status"] == "empty"


def test_compact_tool_observations_keeps_batch_task_cards_in_essential_mode():
    observations = build_tool_observations(
        [
            ToolMessage(
                content=(
                    '{"batch_id":"batch_1",'
                    '"summary":{"total_requested":2,"executed":2,"success":2,"empty":0,"failed":0,"skipped":0},'
                    '"results":['
                    '{"task_id":"d1","tool_name":"generic_lookup","status":"success",'
                    '"args":{"slot":"a"},"result_shape":{"type":"object"},'
                    '"result_stats":{"numbers":{"value":{"min":1,"max":1}}},'
                    '"result_preview":{"value":1}},'
                    '{"task_id":"d2","tool_name":"generic_lookup","status":"success",'
                    '"args":{"slot":"b"},"result_shape":{"type":"object"},'
                    '"result_stats":{"numbers":{"value":{"min":2,"max":2}}},'
                    '"result_preview":{"value":2}}'
                    '],'
                    '"limitations":[]}'
                ),
                name="run_tool_batch",
                tool_call_id="batch-call-1",
            )
        ]
    )

    ledger = compact_tool_observations(observations, budget_chars=1200, preview_chars=0)

    assert ledger.preserved_observation_count == 1
    compact_text = ledger.to_prompt_text()
    assert '"batch_task_cards"' in compact_text
    assert '"task_id": "d1"' in compact_text
    assert '"task_id": "d2"' in compact_text
    assert '"slot": "a"' in compact_text
    assert '"slot": "b"' in compact_text
