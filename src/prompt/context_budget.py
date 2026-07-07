"""Prompt builders for state-preserving context-budget compaction."""

from __future__ import annotations

from typing import Protocol


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
) -> str:
    """Build the synthetic tool observation that restores compacted working state."""
    return (
        "## 压缩后的历史工作状态\n\n"
        f"最近用户目标：{original_user_message}\n\n"
        "以下是压缩后的分层历史状态，包含历史用户消息摘要、assistant 可见执行状态摘要、"
        "以及已完成工具调用的工具观察账本。每条 tool observation 的 args 是实际调用参数，"
        "result_shape/result_stats/result_preview 是工具结果的通用摘要。\n"
        f"{ledger.to_prompt_text()}\n\n"
        "继续执行要求：\n"
        "- 这是历史工具观察，不是最终回答指令。\n"
        "- 继续遵循原始系统提示和当前用户问题；必要时仍可调用可用工具。\n"
        "- 不要重复调用账本中已成功完成且参数相同的工具，除非用户要求刷新或补查。\n"
        "- 可以基于账本中的已知事实继续推理，但不要编造账本之外的工具结果。\n"
        "- 对于账本没有覆盖的请求，可以继续调用工具补充事实。\n"
        "- 如果 dropped_observation_count 大于 0，说明部分较早工具观察已因预算被丢弃。\n\n"
        f"上下文预算提示：原请求估算 {estimate_chars} chars，阈值 {threshold_chars} chars。"
    )


def build_context_compaction_user_prompt(
    *,
    original_user_message: str,
    ledger: ObservationLedgerPrompt,
    estimate_chars: int,
    threshold_chars: int,
) -> str:
    """Backward-compatible alias for the context ledger observation text."""
    return build_context_ledger_tool_observation(
        original_user_message=original_user_message,
        ledger=ledger,
        estimate_chars=estimate_chars,
        threshold_chars=threshold_chars,
    )


__all__ = [
    "CONTEXT_LEDGER_TOOL_NAME",
    "ObservationLedgerPrompt",
    "build_context_compaction_user_prompt",
    "build_context_ledger_tool_call_args",
    "build_context_ledger_tool_observation",
]
