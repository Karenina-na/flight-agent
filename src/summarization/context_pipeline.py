"""Dynamic L1-L5 context compaction pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal

from langchain.agents.middleware import ModelRequest

from src.prompt.context_summary import SummaryKind, build_semantic_summary_messages
from src.summarization.context_compaction import (
    ContextCompactionResult,
    build_context_compaction_request,
)


CompactionLevel = Literal["l1_l3", "l4_local_semantic", "l5_global_fallback"]


@dataclass(frozen=True)
class ContextPartition:
    """One named context partition used by the semantic compaction pipeline."""

    name: str
    payload: Any
    protected: bool = False


@dataclass(frozen=True)
class ContextAssembly:
    """Partitioned context view assembled across compaction stages."""

    partitions: list[ContextPartition]

    def to_dict(self) -> dict[str, Any]:
        return {
            partition.name: partition.payload
            for partition in self.partitions
        }


@dataclass(frozen=True)
class CompactionStageResult:
    """Metadata for one semantic compaction stage."""

    level: CompactionLevel
    request: ModelRequest
    chars: int
    summaries: list[dict[str, Any]]


def build_context_pipeline_request(
    request: ModelRequest,
    *,
    latest_human_text: str,
    estimate_chars: int,
    threshold_chars: int,
    ledger_fraction: float,
    min_ledger_budget_chars: int,
    raw_recent_turns: int,
    estimate_request_chars: Callable[[ModelRequest], int],
    semantic_enabled: bool,
    summary_model: Any | None,
) -> ContextCompactionResult | None:
    """Build a compacted request using deterministic L1-L3 and semantic L4-L5."""
    base_result = _deterministic_compaction_result(
        request,
        latest_human_text=latest_human_text,
        estimate_chars=estimate_chars,
        threshold_chars=threshold_chars,
        ledger_fraction=ledger_fraction,
        min_ledger_budget_chars=min_ledger_budget_chars,
        raw_recent_turns=raw_recent_turns,
    )
    if base_result is None and raw_recent_turns > 1:
        base_result = _deterministic_compaction_result(
            request,
            latest_human_text=latest_human_text,
            estimate_chars=estimate_chars,
            threshold_chars=threshold_chars,
            ledger_fraction=ledger_fraction,
            min_ledger_budget_chars=min_ledger_budget_chars,
            raw_recent_turns=1,
        )
    if base_result is None:
        return None

    base_chars = estimate_request_chars(base_result.request)
    base_result = replace(
        base_result,
        compaction_level="l1_l3",
        post_compaction_chars=base_chars,
        still_over_budget=base_chars > threshold_chars,
    )
    if base_chars <= threshold_chars or not semantic_enabled or summary_model is None:
        return base_result

    try:
        local_summary = _call_summary_model(
            summary_model,
            kind="local_semantic_summary",
            latest_user_goal=latest_human_text,
            bounded_context=_bounded_context(base_result),
        )
        l4_result = _with_semantic_summaries(
            base_result,
            local_semantic_summaries=[local_summary],
            global_fallback_summary=None,
            compaction_level="l4_local_semantic",
            semantic_summary_count=1,
        )
        l4_chars = estimate_request_chars(l4_result.request)
        l4_result = replace(
            l4_result,
            post_compaction_chars=l4_chars,
            still_over_budget=l4_chars > threshold_chars,
        )
        if l4_chars <= threshold_chars:
            return l4_result

        global_summary = _call_summary_model(
            summary_model,
            kind="global_fallback_summary",
            latest_user_goal=latest_human_text,
            bounded_context=_bounded_context(l4_result),
        )
        l5_result = _with_semantic_summaries(
            base_result,
            local_semantic_summaries=[],
            global_fallback_summary=global_summary,
            compaction_level="l5_global_fallback",
            semantic_summary_count=2,
        )
        l5_chars = estimate_request_chars(l5_result.request)
        return replace(
            l5_result,
            post_compaction_chars=l5_chars,
            still_over_budget=l5_chars > threshold_chars,
        )
    except Exception:
        return replace(base_result, semantic_summary_failed=True)


def _deterministic_compaction_result(
    request: ModelRequest,
    *,
    latest_human_text: str,
    estimate_chars: int,
    threshold_chars: int,
    ledger_fraction: float,
    min_ledger_budget_chars: int,
    raw_recent_turns: int,
) -> ContextCompactionResult | None:
    return build_context_compaction_request(
        request,
        latest_human_text=latest_human_text,
        estimate_chars=estimate_chars,
        threshold_chars=threshold_chars,
        ledger_fraction=ledger_fraction,
        min_ledger_budget_chars=min_ledger_budget_chars,
        raw_recent_turns=raw_recent_turns,
    )


def _with_semantic_summaries(
    result: ContextCompactionResult,
    *,
    local_semantic_summaries: list[dict[str, Any]],
    global_fallback_summary: dict[str, Any] | None,
    compaction_level: CompactionLevel,
    semantic_summary_count: int,
) -> ContextCompactionResult:
    compact_request = result.request.override(
        messages=[
            *result.raw_messages,
            *result.synthetic_message_builder(
                local_semantic_summaries=local_semantic_summaries,
                global_fallback_summary=global_fallback_summary,
            ),
        ]
    )
    return replace(
        result,
        request=compact_request,
        compaction_level=compaction_level,
        local_semantic_summaries=local_semantic_summaries,
        global_fallback_summary=global_fallback_summary,
        semantic_summary_count=semantic_summary_count,
        global_fallback_used=global_fallback_summary is not None,
    )


def _call_summary_model(
    summary_model: Any,
    *,
    kind: SummaryKind,
    latest_user_goal: str,
    bounded_context: dict[str, Any],
) -> dict[str, Any]:
    response = summary_model.invoke(
        build_semantic_summary_messages(
            kind=kind,
            latest_user_goal=latest_user_goal,
            bounded_context=bounded_context,
        )
    )
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            str(block.get("text") or block.get("content") or "")
            if isinstance(block, dict)
            else str(block)
            for block in content
        )
    parsed = json.loads(str(content))
    if not isinstance(parsed, dict):
        raise ValueError("semantic summary must be a JSON object")
    facts = parsed.get("facts")
    open_items = parsed.get("open_items")
    evidence_refs = parsed.get("evidence_refs")
    if not isinstance(facts, list) or not isinstance(open_items, list) or not isinstance(evidence_refs, list):
        raise ValueError("semantic summary missing required list fields")
    parsed["type"] = kind
    return parsed


def _bounded_context(result: ContextCompactionResult) -> dict[str, Any]:
    assembly = ContextAssembly(
        partitions=[
            ContextPartition(
                "protected_todo",
                _todo_snapshot_reference(result.todo_snapshot),
                protected=True,
            ),
            ContextPartition("deterministic_history_ledger", result.ledger.to_dict()),
            ContextPartition("local_semantic_summaries", result.local_semantic_summaries),
            ContextPartition("global_fallback_summary", result.global_fallback_summary),
        ]
    )
    return assembly.to_dict()


def _todo_snapshot_reference(todo_snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if not todo_snapshot:
        return None
    return {
        "type": "todo_snapshot_reference",
        "total_count": todo_snapshot.get("total_count", 0),
        "preserved_count": todo_snapshot.get("preserved_count", 0),
        "dropped_count": todo_snapshot.get("dropped_count", 0),
        "truncated_count": todo_snapshot.get("truncated_count", 0),
    }


__all__ = [
    "CompactionLevel",
    "CompactionStageResult",
    "ContextAssembly",
    "ContextPartition",
    "build_context_pipeline_request",
]
