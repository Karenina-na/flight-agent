"""Generic batch executor for registered tools."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Annotated, Any, Literal

from langchain.tools import tool
from pydantic import Field

from src.summarization.tool_observation import json_shape_summary, json_stats_summary
from src.tools.registry import get_tools, register_tool

BATCH_TOOL_NAME = "run_tool_batch"
DEFAULT_MAX_TASKS = 50
DEFAULT_PREVIEW_CHARS = 800

RUN_TOOL_BATCH_TASKS_DESCRIPTION = (
    "要批量执行的工具调用列表。每个任务必须包含 task_id、tool_name、args。"
    "tool_name 必须是已注册工具名，不能是 run_tool_batch 自身。"
)
RUN_TOOL_BATCH_MAX_TASKS_DESCRIPTION = (
    "本次最多执行多少个任务。默认 50；如果 tasks 更多，会跳过超出部分。"
)
RUN_TOOL_BATCH_STOP_ON_ERROR_DESCRIPTION = (
    "单个任务失败后是否停止整个批次。默认 false，表示继续执行后续任务。"
)
RUN_TOOL_BATCH_RETURN_MODE_DESCRIPTION = (
    "返回模式。summary 返回压缩摘要；preview 返回稍长预览。MVP 不返回完整原始结果。"
)

RUN_TOOL_BATCH_TOOL_DESCRIPTION = """批量执行多个已注册工具调用，并返回每个子任务的结构化摘要。

使用场景：
- 用户要求执行很多相似或可枚举的工具查询，例如多日期、多航线、多地点、多航班批量查询。
- 你已经能明确拆出一组 tool_name + args，不希望在 ReAct 循环里逐个调用导致上下文膨胀。

参数填写模板：
{
  "tasks": [
    {
      "task_id": "d1_route_1",
      "tool_name": "search_airfare_quotes",
      "args": {
        "origin": "北京",
        "destination": "上海",
        "departure_date": "2026-07-10",
        "cabin": "economy",
        "adults": 1,
        "currency": "cny",
        "limit": 20
      }
    }
  ],
  "max_tasks": 50,
  "stop_on_error": false,
  "return_mode": "summary"
}

使用规则：
- 本工具是通用批量执行器，不理解具体业务；业务参数必须填在每个 task.args 里。
- 不要把 write_todos、run_tool_batch 自身或记忆/技能类工具放进批量任务。
- 每个 task_id 应唯一，方便最终汇总和定位失败项。
- 如果任务数量很大，先提交一批可控任务；不要一次性提交无限任务。

返回边界：
- 返回每个任务的状态、参数、结果形状、统计和预览。
- 不返回完整原始大结果；需要完整事实时应单独调用对应原子工具。
"""


@tool(description=RUN_TOOL_BATCH_TOOL_DESCRIPTION)
def run_tool_batch(
    tasks: Annotated[list[dict[str, Any]], Field(description=RUN_TOOL_BATCH_TASKS_DESCRIPTION)],
    max_tasks: Annotated[int, Field(description=RUN_TOOL_BATCH_MAX_TASKS_DESCRIPTION)] = DEFAULT_MAX_TASKS,
    stop_on_error: Annotated[bool, Field(description=RUN_TOOL_BATCH_STOP_ON_ERROR_DESCRIPTION)] = False,
    return_mode: Annotated[
        Literal["summary", "preview"],
        Field(description=RUN_TOOL_BATCH_RETURN_MODE_DESCRIPTION),
    ] = "summary",
) -> str:
    """Execute registered tools serially and return bounded summaries."""
    tool_map = _batch_callable_tools()
    safe_max_tasks = _safe_max_tasks(max_tasks)
    selected_tasks = tasks[:safe_max_tasks] if isinstance(tasks, list) else []
    results: list[dict[str, Any]] = []
    stopped_after_error = False

    for index, task in enumerate(selected_tasks):
        result = _run_one_batch_task(
            task=task,
            index=index,
            tool_map=tool_map,
            return_mode=return_mode,
        )
        results.append(result)
        if stop_on_error and result["status"] == "failed":
            stopped_after_error = True
            break

    skipped_count = max(0, len(tasks or []) - len(selected_tasks))
    if stopped_after_error:
        skipped_count += max(0, len(selected_tasks) - len(results))

    payload = {
        "batch_id": _batch_id(tasks=tasks, results=results),
        "summary": _batch_summary(
            total_requested=len(tasks or []),
            executed=len(results),
            skipped=skipped_count,
            results=results,
        ),
        "results": results,
        "limitations": _batch_limitations(
            requested=len(tasks or []),
            executed=len(results),
            skipped=skipped_count,
            max_tasks=safe_max_tasks,
            stopped_after_error=stopped_after_error,
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def _batch_callable_tools() -> dict[str, Any]:
    """Return registered tools available for batch execution."""
    excluded = {
        BATCH_TOOL_NAME,
        "write_todos",
        "remember_user_fact",
        "recall_user_facts",
        "load_skill",
        "list_skill_files",
        "read_skill_file",
    }
    return {
        str(tool_obj.name): tool_obj
        for tool_obj in get_tools()
        if str(getattr(tool_obj, "name", "")) not in excluded
    }


def _run_one_batch_task(
    *,
    task: dict[str, Any],
    index: int,
    tool_map: dict[str, Any],
    return_mode: str,
) -> dict[str, Any]:
    task_id = str(task.get("task_id") or f"task_{index + 1}")
    tool_name = str(task.get("tool_name") or "")
    args = task.get("args")
    if not isinstance(args, dict):
        return _failed_task(
            task_id=task_id,
            tool_name=tool_name,
            args={},
            error_type="InvalidTaskArguments",
            message="task.args must be an object.",
        )
    tool_obj = tool_map.get(tool_name)
    if tool_obj is None:
        return _failed_task(
            task_id=task_id,
            tool_name=tool_name,
            args=args,
            error_type="UnknownTool",
            message=f"Tool '{tool_name}' is not available for batch execution.",
        )
    try:
        raw_result = tool_obj.invoke(args)
    except Exception as exc:  # pragma: no cover - exact third-party errors vary
        return _failed_task(
            task_id=task_id,
            tool_name=tool_name,
            args=args,
            error_type=type(exc).__name__,
            message=str(exc),
        )
    return _successful_task(
        task_id=task_id,
        tool_name=tool_name,
        args=args,
        raw_result=raw_result,
        return_mode=return_mode,
    )


def _successful_task(
    *,
    task_id: str,
    tool_name: str,
    args: dict[str, Any],
    raw_result: Any,
    return_mode: str,
) -> dict[str, Any]:
    raw_text = _result_text(raw_result)
    parsed = _parse_json(raw_text)
    result_value = parsed if parsed is not None else raw_text
    return {
        "task_id": task_id,
        "tool_name": tool_name,
        "status": _success_status(result_value),
        "args": args,
        "result_shape": json_shape_summary(result_value),
        "result_stats": json_stats_summary(result_value),
        "result_preview": _result_preview(result_value, return_mode=return_mode),
        "content_chars": len(raw_text),
        "content_sha256": sha256(raw_text.encode("utf-8")).hexdigest(),
    }


def _failed_task(
    *,
    task_id: str,
    tool_name: str,
    args: dict[str, Any],
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "tool_name": tool_name,
        "status": "failed",
        "args": args,
        "error_type": error_type,
        "message": message,
    }


def _success_status(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("quotes", "items", "flight_records", "relay_quotes"):
            if key in value and isinstance(value[key], list) and not value[key]:
                return "empty"
    return "success"


def _result_preview(value: Any, *, return_mode: str) -> Any:
    if isinstance(value, dict):
        preview: dict[str, Any] = {}
        for key in ("query", "captured_at", "sources_used", "limitations", "timezone", "target_date"):
            if key in value:
                preview[key] = value[key]
        for key in ("quotes", "items", "flight_records", "relay_quotes"):
            items = value.get(key)
            if isinstance(items, list):
                preview[f"{key}_count"] = len(items)
                if return_mode == "preview":
                    preview[f"{key}_sample"] = items[:2]
        return preview or _truncate(_json_text(value), DEFAULT_PREVIEW_CHARS)
    if isinstance(value, list):
        return {"items_count": len(value), "items_sample": value[:2] if return_mode == "preview" else []}
    return _truncate(str(value), DEFAULT_PREVIEW_CHARS)


def _batch_summary(
    *,
    total_requested: int,
    executed: int,
    skipped: int,
    results: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "total_requested": total_requested,
        "executed": executed,
        "success": sum(1 for result in results if result["status"] == "success"),
        "empty": sum(1 for result in results if result["status"] == "empty"),
        "failed": sum(1 for result in results if result["status"] == "failed"),
        "skipped": skipped,
    }


def _batch_limitations(
    *,
    requested: int,
    executed: int,
    skipped: int,
    max_tasks: int,
    stopped_after_error: bool,
) -> list[str]:
    limitations = [
        "Batch executor returns bounded summaries, not complete raw tool outputs."
    ]
    if requested > max_tasks:
        limitations.append(
            f"Requested {requested} tasks but max_tasks is {max_tasks}; extra tasks were skipped."
        )
    if stopped_after_error:
        limitations.append("Stopped early because stop_on_error=true and a task failed.")
    elif skipped and executed < requested:
        limitations.append(f"{skipped} tasks were not executed.")
    return limitations


def _batch_id(*, tasks: list[dict[str, Any]], results: list[dict[str, Any]]) -> str:
    seed = json.dumps(
        {"tasks": tasks, "result_hashes": [item.get("content_sha256") for item in results]},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return f"batch_{sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def _safe_max_tasks(max_tasks: int) -> int:
    if max_tasks <= 0:
        return 0
    return min(max_tasks, DEFAULT_MAX_TASKS)


def _parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


register_tool(run_tool_batch)
