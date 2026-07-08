import json

from src.tools import get_tools


def _tool_by_name(name: str):
    return next(tool for tool in get_tools() if tool.name == name)


def test_run_tool_batch_is_registered_and_guides_arguments():
    tool = _tool_by_name("run_tool_batch")
    schema = tool.args_schema.model_json_schema()

    assert schema["required"] == ["tasks"]
    assert "tool_name" in schema["properties"]["tasks"]["description"]
    assert "run_tool_batch 自身" in tool.description


def test_run_tool_batch_executes_registered_tools_serially():
    tool = _tool_by_name("run_tool_batch")

    payload = json.loads(
        tool.invoke(
            {
                "tasks": [
                    {
                        "task_id": "today",
                        "tool_name": "query_current_date",
                        "args": {
                            "days_offset": 0,
                            "timezone_name": "Asia/Shanghai",
                        },
                    },
                    {
                        "task_id": "tomorrow",
                        "tool_name": "query_current_date",
                        "args": {
                            "days_offset": 1,
                            "timezone_name": "Asia/Shanghai",
                        },
                    },
                ]
            }
        )
    )

    assert payload["batch_id"].startswith("batch_")
    assert payload["summary"] == {
        "total_requested": 2,
        "executed": 2,
        "success": 2,
        "empty": 0,
        "failed": 0,
        "skipped": 0,
    }
    assert [item["task_id"] for item in payload["results"]] == ["today", "tomorrow"]
    assert payload["results"][0]["tool_name"] == "query_current_date"
    assert payload["results"][0]["result_preview"]["timezone"] == "Asia/Shanghai"
    assert payload["results"][1]["result_preview"]["target_date"]


def test_run_tool_batch_rejects_recursive_and_unknown_tools():
    tool = _tool_by_name("run_tool_batch")

    payload = json.loads(
        tool.invoke(
            {
                "tasks": [
                    {
                        "task_id": "recursive",
                        "tool_name": "run_tool_batch",
                        "args": {"tasks": []},
                    },
                    {
                        "task_id": "unknown",
                        "tool_name": "missing_tool",
                        "args": {},
                    },
                ]
            }
        )
    )

    assert payload["summary"]["failed"] == 2
    assert payload["results"][0]["status"] == "failed"
    assert payload["results"][0]["error_type"] == "UnknownTool"
    assert payload["results"][1]["message"] == (
        "Tool 'missing_tool' is not available for batch execution."
    )


def test_run_tool_batch_limits_task_count():
    tool = _tool_by_name("run_tool_batch")

    payload = json.loads(
        tool.invoke(
            {
                "tasks": [
                    {
                        "task_id": "d0",
                        "tool_name": "query_current_date",
                        "args": {"days_offset": 0},
                    },
                    {
                        "task_id": "d1",
                        "tool_name": "query_current_date",
                        "args": {"days_offset": 1},
                    },
                ],
                "max_tasks": 1,
            }
        )
    )

    assert payload["summary"]["total_requested"] == 2
    assert payload["summary"]["executed"] == 1
    assert payload["summary"]["skipped"] == 1
    assert payload["results"][0]["task_id"] == "d0"
    assert "extra tasks were skipped" in payload["limitations"][1]
