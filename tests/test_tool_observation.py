from langchain.messages import ToolMessage

from src.guardrails.tool_observation import (
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

