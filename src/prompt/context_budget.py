"""Prompt builders for state-preserving context-budget compaction."""

from __future__ import annotations

import json
from typing import Any, Protocol


class ObservationLedgerPrompt(Protocol):
    """Minimal prompt-facing surface for compacted observation ledgers."""

    observation_count: int
    preserved_observation_count: int
    dropped_observation_count: int
    preview_truncated_count: int

    def to_prompt_text(self) -> str:
        """Return the ledger body shown to the model."""


CONTEXT_LEDGER_TOOL_NAME = "context_observation_ledger"


def build_context_ledger_tool_call_args(
    *,
    original_user_message: str,
    estimate_chars: int,
    threshold_chars: int,
) -> dict[str, str | int]:
    """Build synthetic tool-call args that describe why context was compacted."""
    return {
        "reason": "context_budget_compaction",
        "latest_user_goal": original_user_message,
        "estimate_chars": estimate_chars,
        "threshold_chars": threshold_chars,
    }


def build_context_ledger_tool_observation(
    *,
    original_user_message: str,
    ledger: ObservationLedgerPrompt,
    estimate_chars: int,
    threshold_chars: int,
    todo_snapshot: dict[str, Any] | None = None,
    local_semantic_summaries: list[dict[str, Any]] | None = None,
    global_fallback_summary: dict[str, Any] | None = None,
    include_deterministic_ledger: bool = True,
) -> str:
    """Build the synthetic tool observation that restores compacted working state."""
    todo_section = ""
    if todo_snapshot:
        todo_section = (
            "## Protected task state\n\n"
            "以下 todo_snapshot 是压缩触发时从任务状态读取的 protected task state，"
            "不是普通历史消息，也不参与工具观察账本压缩。它只保留任务顺序、content 和 status，"
            "用于帮助继续执行未完成任务。\n"
            f"{json.dumps(todo_snapshot, ensure_ascii=False, indent=2)}\n\n"
        )
    semantic_section = ""
    if local_semantic_summaries:
        semantic_section += (
            "## Local semantic summaries\n\n"
            "以下 local_semantic_summary 是旧历史块的模型摘要，已按 facts/open_items/evidence_refs 结构化。\n"
            f"{json.dumps(local_semantic_summaries, ensure_ascii=False, indent=2, default=str)}\n\n"
        )
    if global_fallback_summary:
        semantic_section += (
            "## Global fallback summary\n\n"
            "以下 global_fallback_summary 是预算兜底事实清单，只能作为历史状态参考。\n"
            f"{json.dumps(global_fallback_summary, ensure_ascii=False, indent=2, default=str)}\n\n"
        )
    if include_deterministic_ledger:
        ledger_section = (
            "以下是压缩后的分层历史状态，包含历史用户消息摘要、assistant 可见执行状态摘要、"
            "以及已完成工具调用的工具观察账本。每条 tool observation 的 args 是实际调用参数，"
            "result_shape/result_stats/result_preview 是工具结果的通用摘要。\n"
            f"{ledger.to_prompt_text()}\n\n"
        )
        ledger_instructions = (
            "- 不要重复调用账本中已成功完成且参数相同的工具，除非用户要求刷新或补查。\n"
            "- 可以基于账本中的已知事实继续推理，但不要编造账本之外的工具结果。\n"
            "- 对于账本没有覆盖的请求，可以继续调用工具补充事实。\n"
            "- 如果 dropped_observation_count 大于 0，说明部分较早工具观察已因预算被丢弃。\n"
        )
    else:
        omitted_notice = {
            "status": "deterministic_history_omitted",
            "reason": "l5_global_fallback_budget",
            "observation_count": ledger.observation_count,
            "preserved_observation_count": ledger.preserved_observation_count,
            "dropped_observation_count": ledger.observation_count,
            "preview_truncated_count": ledger.preview_truncated_count,
        }
        ledger_section = (
            "## Deterministic history omitted\n\n"
            "L5 全局兜底已触发，确定性历史账本明细不再进入本次模型上下文。"
            "请仅基于 protected task state、global_fallback_summary、最近原文和当前用户目标继续工作。\n"
            f"{json.dumps(omitted_notice, ensure_ascii=False, indent=2)}\n\n"
        )
        ledger_instructions = (
            "- L5 已省略确定性历史账本明细；不要声称看到了未在 global_fallback_summary 中出现的工具结果。\n"
            "- 如果全局事实清单不足以完成当前请求，可以继续调用工具补充事实。\n"
            "- 不要尝试复原被省略的工具观察明细。\n"
        )
    return (
        "## 压缩后的历史工作状态\n\n"
        f"最近用户目标：{original_user_message}\n\n"
        f"{todo_section}"
        f"{semantic_section}"
        f"{ledger_section}"
        "继续执行要求：\n"
        "- 这是历史工具观察，不是最终回答指令。\n"
        "- 继续遵循原始系统提示和当前用户问题；必要时仍可调用可用工具。\n"
        f"{ledger_instructions}\n"
        f"上下文预算提示：原请求估算 {estimate_chars} chars，阈值 {threshold_chars} chars。"
    )


def build_context_compaction_user_prompt(
    *,
    original_user_message: str,
    ledger: ObservationLedgerPrompt,
    estimate_chars: int,
    threshold_chars: int,
    todo_snapshot: dict[str, Any] | None = None,
    local_semantic_summaries: list[dict[str, Any]] | None = None,
    global_fallback_summary: dict[str, Any] | None = None,
    include_deterministic_ledger: bool = True,
) -> str:
    """Backward-compatible alias for the context ledger observation text."""
    return build_context_ledger_tool_observation(
        original_user_message=original_user_message,
        ledger=ledger,
        estimate_chars=estimate_chars,
        threshold_chars=threshold_chars,
        todo_snapshot=todo_snapshot,
        local_semantic_summaries=local_semantic_summaries,
        global_fallback_summary=global_fallback_summary,
        include_deterministic_ledger=include_deterministic_ledger,
    )


__all__ = [
    "CONTEXT_LEDGER_TOOL_NAME",
    "ObservationLedgerPrompt",
    "build_context_compaction_user_prompt",
    "build_context_ledger_tool_call_args",
    "build_context_ledger_tool_observation",
]
