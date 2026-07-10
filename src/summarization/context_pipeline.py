"""Dynamic L1-L5 context compaction pipeline."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Literal

from langchain.agents.middleware import ModelRequest

from src.prompt.context_summary import SummaryKind, build_semantic_summary_messages
from src.summarization.response_content import visible_response_text
from src.summarization.context_compaction import (
    ContextCompactionResult,
    build_context_compaction_request,
)
from src.summarization.tool_semantic import (
    SummaryEventCallback,
    summarize_tool_candidates,
)


CompactionLevel = Literal[
    "l1_l3",
    "l3_tool_semantic",
    "l3_lossless_preserved",
    "l4_local_semantic",
    "l5_global_fallback",
]


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
    summary_event_callback: SummaryEventCallback | None = None,
) -> ContextCompactionResult | None:
    """Build a compacted request using deterministic L1-L2 and semantic L3-L5."""
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
    working_result = base_result
    candidates = base_result.tool_semantic_candidates or []
    if candidates:
        lossless_result = _with_lossless_tool_results(
            base_result,
            candidates=candidates,
        )
        lossless_chars = estimate_request_chars(lossless_result.request)
        if lossless_chars <= threshold_chars:
            return replace(
                lossless_result,
                post_compaction_chars=lossless_chars,
                still_over_budget=False,
                semantic_skip_reason="within_budget_with_lossless_tool_results",
            )

    if candidates and semantic_enabled and summary_model is not None:
        try:
            tool_summaries = summarize_tool_candidates(
                summary_model,
                candidates,
                event_callback=summary_event_callback,
            )
            if tool_summaries:
                working_result = _with_tool_semantic_summaries(
                    base_result,
                    summaries_by_tool_call_id=tool_summaries,
                )
                l3_chars = estimate_request_chars(working_result.request)
                working_result = replace(
                    working_result,
                    post_compaction_chars=l3_chars,
                    still_over_budget=l3_chars > threshold_chars,
                )
        except Exception as exc:
            working_result = replace(
                base_result,
                semantic_summary_failed=True,
                tool_semantic_summary_failed=True,
                semantic_skip_reason="tool_semantic_summary_failed",
                semantic_error_stage="l3_tool_semantic",
                semantic_error_type=type(exc).__name__,
            )

    if working_result.post_compaction_chars <= threshold_chars:
        if working_result.compaction_level == "l3_tool_semantic":
            return replace(working_result, semantic_skip_reason="within_budget_after_l3")
        if candidates and not semantic_enabled:
            return replace(working_result, semantic_skip_reason="semantic_disabled")
        if candidates and summary_model is None:
            return replace(working_result, semantic_skip_reason="missing_summary_model")
        if working_result.tool_semantic_summary_failed:
            return working_result
        return replace(working_result, semantic_skip_reason="within_budget_after_l1_l3")
    if not semantic_enabled:
        return replace(working_result, semantic_skip_reason="semantic_disabled")
    if summary_model is None:
        return replace(working_result, semantic_skip_reason="missing_summary_model")

    try:
        local_summary = _call_summary_model(
            summary_model,
            kind="local_semantic_summary",
            latest_user_goal=latest_human_text,
            bounded_context=_bounded_context(working_result),
            event_callback=summary_event_callback,
        )
        l4_result = _with_semantic_summaries(
            working_result,
            local_semantic_summaries=[local_summary],
            global_fallback_summary=None,
            compaction_level="l4_local_semantic",
            semantic_summary_count=working_result.semantic_summary_count + 1,
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
            event_callback=summary_event_callback,
        )
        l5_result = _with_semantic_summaries(
            working_result,
            local_semantic_summaries=[],
            global_fallback_summary=global_summary,
            compaction_level="l5_global_fallback",
            semantic_summary_count=working_result.semantic_summary_count + 2,
        )
        l5_chars = estimate_request_chars(l5_result.request)
        return replace(
            l5_result,
            post_compaction_chars=l5_chars,
            still_over_budget=l5_chars > threshold_chars,
        )
    except Exception as exc:
        return replace(
            working_result,
            semantic_summary_failed=True,
            semantic_skip_reason="semantic_summary_failed",
            semantic_error_stage="l4_l5_semantic",
            semantic_error_type=type(exc).__name__,
        )


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
    include_deterministic_ledger = compaction_level != "l5_global_fallback"
    compact_request = result.request.override(
        messages=[
            *result.raw_messages,
            *result.synthetic_message_builder(
                ledger_override=result.ledger,
                local_semantic_summaries=local_semantic_summaries,
                global_fallback_summary=global_fallback_summary,
                include_deterministic_ledger=include_deterministic_ledger,
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
        deterministic_ledger_included=include_deterministic_ledger,
    )


def _with_tool_semantic_summaries(
    result: ContextCompactionResult,
    *,
    summaries_by_tool_call_id: dict[str, dict[str, Any]],
) -> ContextCompactionResult:
    candidate_args = {
        candidate.tool_call_id: candidate.args
        for candidate in result.tool_semantic_candidates or []
    }
    tool_ledger = result.ledger.tool_observation_ledger.with_semantic_summaries(
        summaries_by_tool_call_id,
        args_by_tool_call_id=candidate_args,
    )
    semantic_ledger = replace(result.ledger, tool_observation_ledger=tool_ledger)
    compact_request = result.request.override(
        messages=[
            *result.raw_messages,
            *result.synthetic_message_builder(
                ledger_override=semantic_ledger,
                local_semantic_summaries=[],
                global_fallback_summary=None,
                include_deterministic_ledger=True,
            ),
        ]
    )
    summary_count = len(summaries_by_tool_call_id)
    return replace(
        result,
        request=compact_request,
        ledger=semantic_ledger,
        compaction_level="l3_tool_semantic",
        semantic_summary_count=result.semantic_summary_count + summary_count,
        tool_semantic_summary_count=summary_count,
        tool_semantic_summary_failed=False,
        semantic_skip_reason=None,
    )


def _with_lossless_tool_results(
    result: ContextCompactionResult,
    *,
    candidates: list[Any],
) -> ContextCompactionResult:
    results_by_tool_call_id = {
        candidate.tool_call_id: candidate.content
        for candidate in candidates
        if candidate.tool_call_id
    }
    args_by_tool_call_id = {
        candidate.tool_call_id: candidate.args
        for candidate in candidates
        if candidate.tool_call_id
    }
    names_by_tool_call_id = {
        candidate.tool_call_id: candidate.tool_name
        for candidate in candidates
        if candidate.tool_call_id
    }
    tool_ledger = result.ledger.tool_observation_ledger.with_lossless_results(
        results_by_tool_call_id,
        args_by_tool_call_id=args_by_tool_call_id,
        names_by_tool_call_id=names_by_tool_call_id,
    )
    lossless_ledger = replace(result.ledger, tool_observation_ledger=tool_ledger)
    compact_request = result.request.override(
        messages=[
            *result.raw_messages,
            *result.synthetic_message_builder(
                ledger_override=lossless_ledger,
                local_semantic_summaries=[],
                global_fallback_summary=None,
                include_deterministic_ledger=True,
            ),
        ]
    )
    return replace(
        result,
        request=compact_request,
        ledger=lossless_ledger,
        compaction_level="l3_lossless_preserved",
        semantic_summary_failed=False,
        tool_semantic_summary_failed=False,
    )


def _call_summary_model(
    summary_model: Any,
    *,
    kind: SummaryKind,
    latest_user_goal: str,
    bounded_context: dict[str, Any],
    event_callback: SummaryEventCallback | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    input_chars = len(
        json.dumps(bounded_context, ensure_ascii=False, default=str)
    )
    _emit_summary_event(
        event_callback,
        "context_summary_start",
        stage=kind,
        input_chars=input_chars,
    )
    try:
        response = summary_model.invoke(
            build_semantic_summary_messages(
                kind=kind,
                latest_user_goal=latest_user_goal,
                bounded_context=bounded_context,
            )
        )
        response_text = visible_response_text(response)
        parsed = json.loads(response_text)
        if not isinstance(parsed, dict):
            raise ValueError("semantic summary must be a JSON object")
        facts = parsed.get("facts")
        open_items = parsed.get("open_items")
        evidence_refs = parsed.get("evidence_refs")
        if (
            not isinstance(facts, list)
            or not isinstance(open_items, list)
            or not isinstance(evidence_refs, list)
        ):
            raise ValueError("semantic summary missing required list fields")
        parsed["type"] = kind
    except Exception as exc:
        _emit_summary_event(
            event_callback,
            "context_summary_error",
            stage=kind,
            status="error",
            duration_ms=_duration_ms(started_at),
            error_type=type(exc).__name__,
        )
        raise
    _emit_summary_event(
        event_callback,
        "context_summary_end",
        stage=kind,
        status="success",
        duration_ms=_duration_ms(started_at),
        output_chars=len(response_text),
    )
    return parsed


def _emit_summary_event(
    callback: SummaryEventCallback | None,
    event: str,
    **fields: Any,
) -> None:
    if callback is not None:
        callback(event, fields)


def _duration_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _bounded_context(result: ContextCompactionResult) -> dict[str, Any]:
    assembly = ContextAssembly(
        partitions=[
            ContextPartition("history", result.ledger.to_model_text()),
            ContextPartition("previous_summaries", result.local_semantic_summaries),
        ]
    )
    return assembly.to_dict()


__all__ = [
    "CompactionLevel",
    "CompactionStageResult",
    "ContextAssembly",
    "ContextPartition",
    "build_context_pipeline_request",
]
