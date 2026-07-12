"""Generic tool-observation summaries for context compaction."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any


DEFAULT_PREVIEW_CHARS = 500
MAX_SHAPE_DEPTH = 3
MAX_SAMPLE_ITEMS = 3
MAX_STRING_SAMPLES = 5
BATCH_TASK_PREVIEW_CHARS = 160


@dataclass(frozen=True)
class ToolObservation:
    """A compact, business-agnostic card for one completed tool message."""

    tool_name: str
    tool_call_id: str
    args: dict[str, Any]
    status: str
    result_shape: dict[str, Any]
    result_preview: str
    result_stats: dict[str, Any]
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "args": self.args,
            "status": self.status,
            "result_shape": self.result_shape,
            "result_preview": self.result_preview,
            "result_stats": self.result_stats,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class CompactObservationLedger:
    """A bounded set of tool-observation cards for compact state summaries."""

    observation_count: int
    preserved_observation_count: int
    dropped_observation_count: int
    preview_truncated_count: int
    observations: list[dict[str, Any]]
    budget_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "preserved_observation_count": self.preserved_observation_count,
            "dropped_observation_count": self.dropped_observation_count,
            "preview_truncated_count": self.preview_truncated_count,
            "budget_chars": self.budget_chars,
            "observations": self.observations,
        }

    def to_prompt_text(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)

    def to_model_text(self) -> str:
        """Render only the tool facts needed by the model."""
        if not self.observations:
            return "- 没有保留下来的历史工具结果。"

        lines = [
            _observation_model_line(index, observation)
            for index, observation in enumerate(self.observations, start=1)
        ]
        if self.dropped_observation_count:
            lines.append(
                f"- 另有 {self.dropped_observation_count} 条较早工具结果因上下文预算未保留。"
            )
        return "\n".join(lines)

    def with_semantic_summaries(
        self,
        summaries_by_tool_call_id: dict[str, dict[str, Any]],
        *,
        args_by_tool_call_id: dict[str, dict[str, Any]] | None = None,
    ) -> CompactObservationLedger:
        """Attach validated L3 summaries without changing diagnostic source data."""
        observations = []
        for observation in self.observations:
            tool_call_id = str(observation.get("tool_call_id") or "")
            summary = summaries_by_tool_call_id.get(tool_call_id)
            updated = dict(observation)
            if args_by_tool_call_id and tool_call_id in args_by_tool_call_id:
                updated["args"] = args_by_tool_call_id[tool_call_id]
            if summary:
                updated["semantic_summary"] = summary
            observations.append(updated)
        return replace(self, observations=observations)

    def with_lossless_results(
        self,
        results_by_tool_call_id: dict[str, str],
        *,
        args_by_tool_call_id: dict[str, dict[str, Any]] | None = None,
        names_by_tool_call_id: dict[str, str] | None = None,
    ) -> CompactObservationLedger:
        """Restore complete tool results when they fit the compacted budget."""
        observations: list[dict[str, Any]] = []
        restored_ids: set[str] = set()
        for observation in self.observations:
            updated = dict(observation)
            tool_call_id = str(updated.get("tool_call_id") or "")
            content = results_by_tool_call_id.get(tool_call_id)
            if content is not None:
                updated["result_preview"] = content
                updated["result_preview_truncated"] = False
                updated["lossless_result"] = True
                restored_ids.add(tool_call_id)
            if args_by_tool_call_id and tool_call_id in args_by_tool_call_id:
                updated["args"] = args_by_tool_call_id[tool_call_id]
            observations.append(updated)

        for tool_call_id, content in results_by_tool_call_id.items():
            if tool_call_id in restored_ids:
                continue
            observations.append(
                {
                    "tool_name": (
                        names_by_tool_call_id or {}
                    ).get(tool_call_id, "tool"),
                    "tool_call_id": tool_call_id,
                    "args": (args_by_tool_call_id or {}).get(tool_call_id, {}),
                    "status": "success",
                    "result_preview": content,
                    "result_preview_truncated": False,
                    "lossless_result": True,
                }
            )

        preserved_count = len(observations)
        observation_count = max(self.observation_count, preserved_count)
        return replace(
            self,
            observation_count=observation_count,
            preserved_observation_count=preserved_count,
            dropped_observation_count=max(observation_count - preserved_count, 0),
            preview_truncated_count=sum(
                1
                for observation in observations
                if observation.get("result_preview_truncated")
            ),
            observations=observations,
        )


def build_tool_observations(messages: list[Any]) -> list[ToolObservation]:
    """Build generic observation cards from completed tool messages."""
    observations: list[ToolObservation] = []
    tool_call_args_by_id: dict[str, dict[str, Any]] = {}
    tool_call_names_by_id: dict[str, str] = {}
    for message in messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = str(tool_call.get("id") or "")
            if not tool_call_id:
                continue
            args = tool_call.get("args")
            if isinstance(args, dict):
                tool_call_args_by_id[tool_call_id] = args
            name = str(tool_call.get("name") or "")
            if name:
                tool_call_names_by_id[tool_call_id] = name

        if str(getattr(message, "type", "")) != "tool":
            continue
        tool_call_id = str(getattr(message, "tool_call_id", "") or "")
        content = getattr(message, "content", "")
        content_text = _content_text(content)
        parsed = _parse_json(content)
        result_value: Any = parsed if parsed is not None else content_text
        result_shape = json_shape_summary(result_value)
        result_preview = _preview_text(result_value)
        result_stats = json_stats_summary(result_value)
        if _is_batch_tool_result(result_value):
            result_shape = _batch_result_shape(result_value)
            result_preview = _batch_result_preview(result_value)
            result_stats = _batch_result_stats(result_value)
        observations.append(
            ToolObservation(
                tool_name=str(getattr(message, "name", "") or tool_call_names_by_id.get(tool_call_id) or "tool"),
                tool_call_id=tool_call_id,
                args=tool_call_args_by_id.get(tool_call_id) or _infer_args(result_value),
                status=str(getattr(message, "status", "success") or "success"),
                result_shape=result_shape,
                result_preview=result_preview,
                result_stats=result_stats,
                content_sha256=sha256(content_text.encode("utf-8")).hexdigest(),
            )
        )
    return observations


def compact_tool_observations(
    observations: list[ToolObservation],
    *,
    budget_chars: int,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> CompactObservationLedger:
    """Compact observations while keeping one card per tool call when possible."""
    original_count = len(observations)
    cards = [_observation_card(observation, preview_chars=preview_chars) for observation in observations]
    preview_truncated_count = sum(
        1
        for observation, card in zip(observations, cards, strict=False)
        if len(observation.result_preview) > len(str(card.get("result_preview", "")))
    )
    if _ledger_chars(cards, original_count, preview_truncated_count, budget_chars) > budget_chars:
        cards = [_essential_observation_card(observation) for observation in observations]

    while cards and _ledger_chars(cards, original_count, preview_truncated_count, budget_chars) > budget_chars:
        cards = cards[1:]

    return CompactObservationLedger(
        observation_count=original_count,
        preserved_observation_count=len(cards),
        dropped_observation_count=original_count - len(cards),
        preview_truncated_count=preview_truncated_count,
        observations=cards,
        budget_chars=budget_chars,
    )


def json_shape_summary(value: Any, *, max_depth: int = MAX_SHAPE_DEPTH) -> dict[str, Any]:
    """Return a generic JSON shape summary."""
    return _shape(value, depth=0, max_depth=max_depth)


def json_stats_summary(value: Any) -> dict[str, Any]:
    """Return generic stats for JSON-like values."""
    stats: dict[str, Any] = {
        "arrays": {},
        "numbers": {},
        "strings": {},
        "booleans": {},
        "nulls": [],
    }
    number_values: dict[str, list[float]] = {}
    string_values: dict[str, list[str]] = {}
    boolean_counts: dict[str, dict[str, int]] = {}

    def visit(current: Any, path: str) -> None:
        if isinstance(current, dict):
            for key, item in current.items():
                child_path = f"{path}.{key}" if path else str(key)
                visit(item, child_path)
            return
        if isinstance(current, list):
            stats["arrays"][path or "$"] = {"length": len(current)}
            for item in current:
                visit(item, f"{path}[]" if path else "[]")
            return
        if isinstance(current, bool):
            bucket = boolean_counts.setdefault(path or "$", {"true": 0, "false": 0})
            bucket["true" if current else "false"] += 1
            return
        if isinstance(current, int | float):
            number_values.setdefault(path or "$", []).append(float(current))
            return
        if isinstance(current, str):
            if len(current) <= 80:
                values = string_values.setdefault(path or "$", [])
                if current not in values and len(values) < MAX_STRING_SAMPLES:
                    values.append(current)
            return
        if current is None:
            stats["nulls"].append(path or "$")

    visit(value, "")
    stats["numbers"] = {
        path: {"min": _clean_number(min(values)), "max": _clean_number(max(values))}
        for path, values in sorted(number_values.items())
    }
    stats["strings"] = {path: values for path, values in sorted(string_values.items())}
    stats["booleans"] = {path: counts for path, counts in sorted(boolean_counts.items())}
    stats["nulls"] = sorted(set(stats["nulls"]))
    return stats


def _shape(value: Any, *, depth: int, max_depth: int) -> dict[str, Any]:
    if isinstance(value, dict):
        summary: dict[str, Any] = {
            "type": "object",
            "keys": sorted(str(key) for key in value.keys()),
        }
        if depth < max_depth:
            summary["children"] = {
                str(key): _shape(item, depth=depth + 1, max_depth=max_depth)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))[:MAX_SAMPLE_ITEMS]
            }
        return summary
    if isinstance(value, list):
        item_shapes = [_shape(item, depth=depth + 1, max_depth=max_depth) for item in value[:MAX_SAMPLE_ITEMS]]
        return {
            "type": "array",
            "length": len(value),
            "sample_item_shapes": item_shapes,
        }
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int | float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string", "chars": len(value)}
    return {"type": type(value).__name__}


def _observation_card(observation: ToolObservation, *, preview_chars: int) -> dict[str, Any]:
    card = observation.to_dict()
    card["result_preview"] = _truncate(observation.result_preview, preview_chars)
    card["result_preview_truncated"] = len(observation.result_preview) > preview_chars
    return card


def _essential_observation_card(observation: ToolObservation) -> dict[str, Any]:
    return {
        "tool_name": observation.tool_name,
        "tool_call_id": observation.tool_call_id,
        "args": observation.args,
        "status": observation.status,
        "result_shape": _essential_shape(observation.result_shape),
        "result_stats": _essential_stats(observation.result_stats),
        "result_preview": "",
        "result_preview_truncated": bool(observation.result_preview),
        "content_sha256": observation.content_sha256[:16],
    }


def _essential_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in shape.items()
        if key in {"type", "keys", "length", "chars"}
    }


def _essential_stats(stats: dict[str, Any]) -> dict[str, Any]:
    if "batch_task_cards" in stats:
        compact_batch: dict[str, Any] = {
            "batch_task_count": stats.get("batch_task_count", 0),
            "batch_task_cards": stats.get("batch_task_cards", []),
        }
        if "batch_summary" in stats:
            compact_batch["batch_summary"] = stats["batch_summary"]
        return compact_batch
    arrays = {
        path: {"length": data.get("length")}
        for path, data in stats.get("arrays", {}).items()
        if isinstance(data, dict)
    }
    numbers = {
        path: data
        for path, data in stats.get("numbers", {}).items()
        if isinstance(data, dict)
    }
    compact: dict[str, Any] = {}
    if arrays:
        compact["arrays"] = arrays
    if numbers:
        compact["numbers"] = numbers
    return compact


def _observation_model_line(index: int, observation: dict[str, Any]) -> str:
    tool_name = str(observation.get("tool_name") or "未知工具")
    status = _model_status(str(observation.get("status") or ""))
    args = observation.get("args")
    args_text = (
        json.dumps(args, ensure_ascii=False, separators=(",", ":"), default=str)
        if isinstance(args, dict) and args
        else "无"
    )
    result_text = _model_result_text(observation)
    return (
        f"{index}. {tool_name}；参数：{args_text}；状态：{status}；"
        f"结果摘要：{result_text}"
    )


def _model_status(status: str) -> str:
    labels = {
        "success": "成功",
        "error": "失败",
        "failed": "失败",
        "duplicate_blocked": "重复调用已阻止",
    }
    return labels.get(status.strip().lower(), status or "未知")


def _model_result_text(observation: dict[str, Any]) -> str:
    preview = str(observation.get("result_preview") or "").strip()
    if observation.get("lossless_result") and preview:
        return f"完整工具结果：{preview}"

    semantic_summary = observation.get("semantic_summary")
    if isinstance(semantic_summary, dict):
        semantic_text = _semantic_summary_text(semantic_summary)
        if semantic_text:
            return semantic_text
    if isinstance(semantic_summary, str) and semantic_summary.strip():
        return semantic_summary.strip()

    stats_text = _model_stats_text(observation.get("result_stats"))
    preview_truncated = bool(observation.get("result_preview_truncated"))
    if stats_text:
        if preview and not preview_truncated:
            return f"{stats_text}；完整结果：{preview}"
        return stats_text
    if preview and not preview_truncated:
        return preview
    return "未保留结果明细。"


def _model_stats_text(stats: Any) -> str:
    if not isinstance(stats, dict):
        return ""
    parts: list[str] = []
    batch_summary = stats.get("batch_summary")
    if isinstance(batch_summary, dict) and batch_summary:
        parts.append(
            json.dumps(
                batch_summary,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        )
    for path, data in (stats.get("arrays") or {}).items():
        if isinstance(data, dict) and data.get("length") is not None:
            parts.append(f"{path} 共 {data['length']} 条")
    for path, data in (stats.get("numbers") or {}).items():
        if not isinstance(data, dict):
            continue
        minimum = data.get("min")
        maximum = data.get("max")
        if minimum is None and maximum is None:
            continue
        if minimum == maximum:
            parts.append(f"{path} 为 {minimum}")
        else:
            parts.append(f"{path} 范围 {minimum} 至 {maximum}")
    return "；".join(parts)


def _semantic_summary_text(summary: dict[str, Any]) -> str:
    content = summary.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    text = summary.get("summary")
    if isinstance(text, str) and text.strip():
        return text.strip()

    facts = [
        item.strip()
        for item in summary.get("facts", [])
        if isinstance(item, str) and item.strip()
    ]
    omissions = [
        item.strip()
        for item in summary.get("omissions", [])
        if isinstance(item, str) and item.strip()
    ]
    parts = list(facts)
    if omissions:
        parts.append("省略信息：" + "；".join(omissions))
    return "；".join(parts)


def _is_batch_tool_result(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("batch_id"), str)
        and isinstance(value.get("summary"), dict)
        and isinstance(value.get("results"), list)
    )


def _batch_result_shape(value: dict[str, Any]) -> dict[str, Any]:
    results = value.get("results") if isinstance(value.get("results"), list) else []
    task_fields: set[str] = set()
    task_tool_names: list[str] = []
    task_statuses: dict[str, int] = {}
    for task in results:
        if not isinstance(task, dict):
            continue
        task_fields.update(str(key) for key in task.keys())
        tool_name = str(task.get("tool_name") or "")
        if tool_name and tool_name not in task_tool_names and len(task_tool_names) < MAX_STRING_SAMPLES:
            task_tool_names.append(tool_name)
        status = str(task.get("status") or "")
        if status:
            task_statuses[status] = task_statuses.get(status, 0) + 1
    return {
        "type": "object",
        "keys": sorted(str(key) for key in value.keys()),
        "batch_tool_result": True,
        "task_count": len(results),
        "task_fields": sorted(task_fields),
        "task_tool_names_sample": task_tool_names,
        "task_status_counts": task_statuses,
    }


def _batch_result_preview(value: dict[str, Any]) -> str:
    preview = {
        "batch_id": value.get("batch_id"),
        "summary": value.get("summary"),
        "limitations": value.get("limitations", []),
    }
    return _preview_text(preview)


def _batch_result_stats(value: dict[str, Any]) -> dict[str, Any]:
    results = value.get("results") if isinstance(value.get("results"), list) else []
    task_cards = [
        _batch_task_card(task)
        for task in results
        if isinstance(task, dict)
    ]
    return {
        "batch_summary": value.get("summary") if isinstance(value.get("summary"), dict) else {},
        "batch_task_count": len(results),
        "batch_task_cards": task_cards,
    }


def _batch_task_card(task: dict[str, Any]) -> dict[str, Any]:
    card: dict[str, Any] = {
        "task_id": str(task.get("task_id") or ""),
        "tool_name": str(task.get("tool_name") or ""),
        "status": str(task.get("status") or ""),
        "args": task.get("args") if isinstance(task.get("args"), dict) else {},
    }
    if "error_type" in task or "message" in task:
        card["error_type"] = str(task.get("error_type") or "")
        card["message"] = _truncate(str(task.get("message") or ""), BATCH_TASK_PREVIEW_CHARS)
    result_shape = task.get("result_shape")
    if isinstance(result_shape, dict):
        card["result_shape"] = _essential_shape(result_shape)
    result_stats = task.get("result_stats")
    if isinstance(result_stats, dict):
        compact_stats = _essential_stats(result_stats)
        if compact_stats:
            card["result_stats"] = compact_stats
    result_preview = task.get("result_preview")
    if result_preview not in (None, "", {}, []):
        card["result_preview"] = _truncate(_preview_text(result_preview), BATCH_TASK_PREVIEW_CHARS)
    content_sha256 = str(task.get("content_sha256") or "")
    if content_sha256:
        card["content_sha256"] = content_sha256[:16]
    return card


def _ledger_chars(
    cards: list[dict[str, Any]],
    observation_count: int,
    preview_truncated_count: int,
    budget_chars: int,
) -> int:
    ledger = CompactObservationLedger(
        observation_count=observation_count,
        preserved_observation_count=len(cards),
        dropped_observation_count=observation_count - len(cards),
        preview_truncated_count=preview_truncated_count,
        observations=cards,
        budget_chars=budget_chars,
    )
    return len(json.dumps(ledger.to_dict(), ensure_ascii=False, separators=(",", ":"), default=str))


def _parse_json(content: Any) -> Any | None:
    if isinstance(content, dict | list):
        return content
    if not isinstance(content, str):
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _infer_args(result_value: Any) -> dict[str, Any]:
    if isinstance(result_value, dict):
        query = result_value.get("query")
        if isinstance(query, dict):
            return query
        args = result_value.get("args")
        if isinstance(args, dict):
            return args
    return {}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(content)


def _preview_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    return text if len(text) <= limit else f"{text[:limit]}..."


def _clean_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


__all__ = [
    "CompactObservationLedger",
    "ToolObservation",
    "build_tool_observations",
    "compact_tool_observations",
    "json_shape_summary",
    "json_stats_summary",
]
