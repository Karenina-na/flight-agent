"""Generic tool-observation summaries for context compaction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any


DEFAULT_PREVIEW_CHARS = 500
MAX_SHAPE_DEPTH = 3
MAX_SAMPLE_ITEMS = 3
MAX_STRING_SAMPLES = 5


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
        observations.append(
            ToolObservation(
                tool_name=str(getattr(message, "name", "") or tool_call_names_by_id.get(tool_call_id) or "tool"),
                tool_call_id=tool_call_id,
                args=tool_call_args_by_id.get(tool_call_id) or _infer_args(result_value),
                status=str(getattr(message, "status", "success") or "success"),
                result_shape=json_shape_summary(result_value),
                result_preview=_truncate(_preview_text(result_value), DEFAULT_PREVIEW_CHARS),
                result_stats=json_stats_summary(result_value),
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
        "content_sha256": observation.content_sha256[:16],
    }


def _essential_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in shape.items()
        if key in {"type", "keys", "length", "chars"}
    }


def _essential_stats(stats: dict[str, Any]) -> dict[str, Any]:
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
