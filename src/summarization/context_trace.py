"""Trace helpers for staged context compaction."""

from __future__ import annotations

from typing import Any

from src.summarization.context_compaction import ContextCompactionResult
from src.summarization.tool_semantic import SummaryEventCallback


def emit_deterministic_layer_events(
    callback: SummaryEventCallback | None,
    *,
    result: ContextCompactionResult,
    estimate_chars: int,
    threshold_chars: int,
) -> None:
    """Emit L1/L2 events after deterministic compaction has run."""
    projection = result.layer_one_projection
    l1_change_count = (
        projection.adjacent_human_merged_count
        + projection.duplicate_tool_output_count
        + projection.empty_tool_output_count
    )
    l2_change_count = (
        projection.reasoning_block_removed_count
        + projection.tool_call_removed_count
        + projection.tool_message_removed_count
    )

    if l1_change_count:
        emit_context_compaction_layer(
            callback,
            layer="L1",
            layer_name="零成本修剪",
            status="completed",
            summary=(
                f"规则修剪完成，合并连续用户消息 {projection.adjacent_human_merged_count} 条，"
                f"重复工具输出 {projection.duplicate_tool_output_count} 条，"
                f"空工具输出 {projection.empty_tool_output_count} 条。"
            ),
            estimate_chars=estimate_chars,
            threshold_chars=threshold_chars,
            change_count=l1_change_count,
            adjacent_human_merged_count=projection.adjacent_human_merged_count,
            duplicate_tool_output_count=projection.duplicate_tool_output_count,
            empty_tool_output_count=projection.empty_tool_output_count,
        )
    if l2_change_count:
        emit_context_compaction_layer(
            callback,
            layer="L2",
            layer_name="ReAct 裁剪",
            status="completed",
            summary=(
                f"ReAct 历史裁剪完成，移除 reasoning block {projection.reasoning_block_removed_count} 个，"
                f"历史工具调用痕迹 {projection.tool_call_removed_count} 个。"
            ),
            estimate_chars=estimate_chars,
            threshold_chars=threshold_chars,
            change_count=l2_change_count,
            reasoning_block_removed_count=projection.reasoning_block_removed_count,
            tool_call_removed_count=projection.tool_call_removed_count,
            tool_message_removed_count=projection.tool_message_removed_count,
        )


def emit_l3_layer_event(
    callback: SummaryEventCallback | None,
    *,
    result: ContextCompactionResult,
    estimate_chars: int,
    threshold_chars: int,
) -> None:
    """Emit L3 after tool-result compression decisions are complete."""
    if not _should_emit_l3(result):
        return

    level = result.compaction_level
    if level == "l3_lossless_preserved":
        summary = "L3 检查完成，完整工具结果可以保留在压缩视图中。"
    elif level == "l3_tool_semantic" and result.still_over_budget:
        summary = (
            f"L3 工具结果语义压缩完成，候选 {len(result.tool_semantic_candidates or [])} 个，"
            f"成功摘要 {result.tool_semantic_summary_count} 个，但仍超过预算。"
        )
    elif level == "l3_tool_semantic":
        summary = (
            f"L3 工具结果语义压缩完成，候选 {len(result.tool_semantic_candidates or [])} 个，"
            f"成功摘要 {result.tool_semantic_summary_count} 个。"
        )
    elif result.still_over_budget:
        summary = "L3 已执行但仍超过预算，继续进入更高层压缩。"
    else:
        summary = "L3 结构化降维完成，未进入工具语义摘要。"

    emit_context_compaction_layer(
        callback,
        layer="L3",
        layer_name="工具结果压缩",
        status="completed",
        summary=summary,
        estimate_chars=estimate_chars,
        threshold_chars=threshold_chars,
        compaction_level=level,
        tool_semantic_candidate_count=len(result.tool_semantic_candidates or []),
        tool_semantic_summary_count=result.tool_semantic_summary_count,
        tool_semantic_summary_failed=result.tool_semantic_summary_failed,
        post_compaction_chars=result.post_compaction_chars,
        still_over_budget=result.still_over_budget,
    )


def emit_l4_layer_event(
    callback: SummaryEventCallback | None,
    *,
    result: ContextCompactionResult,
    estimate_chars: int,
    threshold_chars: int,
) -> None:
    """Emit L4 after local semantic summary has been applied."""
    emit_context_compaction_layer(
        callback,
        layer="L4",
        layer_name="局部语义摘要",
        status="completed",
        summary="L4 局部语义摘要已执行。",
        estimate_chars=estimate_chars,
        threshold_chars=threshold_chars,
        compaction_level=result.compaction_level,
        semantic_summary_count=result.semantic_summary_count,
        semantic_summary_failed=result.semantic_summary_failed,
        semantic_skip_reason=result.semantic_skip_reason,
        post_compaction_chars=result.post_compaction_chars,
        still_over_budget=result.still_over_budget,
    )


def emit_l5_layer_event(
    callback: SummaryEventCallback | None,
    *,
    result: ContextCompactionResult,
    estimate_chars: int,
    threshold_chars: int,
) -> None:
    """Emit L5 after global fallback summary has been applied."""
    emit_context_compaction_layer(
        callback,
        layer="L5",
        layer_name="全局兜底摘要",
        status="completed",
        summary="L5 全局兜底摘要已执行。",
        estimate_chars=estimate_chars,
        threshold_chars=threshold_chars,
        compaction_level=result.compaction_level,
        global_fallback_used=result.global_fallback_used,
        deterministic_ledger_included=result.deterministic_ledger_included,
        post_compaction_chars=result.post_compaction_chars,
        still_over_budget=result.still_over_budget,
    )


def emit_context_compaction_layer(
    callback: SummaryEventCallback | None,
    *,
    layer: str,
    layer_name: str,
    status: str,
    summary: str,
    estimate_chars: int,
    threshold_chars: int,
    **fields: Any,
) -> None:
    """Emit one context-compaction layer event through the pipeline callback."""
    if callback is None:
        return
    callback(
        "context_compaction_layer",
        {
            "layer": layer,
            "layer_name": layer_name,
            "status": status,
            "summary": summary,
            "estimate_chars": estimate_chars,
            "threshold_chars": threshold_chars,
            **fields,
        },
    )


def _should_emit_l3(result: ContextCompactionResult) -> bool:
    return bool(
        result.ledger.preview_truncated_count
        or result.tool_semantic_candidates
        or result.compaction_level
        in {
            "l3_lossless_preserved",
            "l3_tool_semantic",
            "l4_local_semantic",
            "l5_global_fallback",
        }
    )


__all__ = [
    "emit_context_compaction_layer",
    "emit_deterministic_layer_events",
    "emit_l3_layer_event",
    "emit_l4_layer_event",
    "emit_l5_layer_event",
]
