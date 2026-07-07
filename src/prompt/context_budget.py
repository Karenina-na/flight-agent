"""Prompt builders for context-budget compaction final answers."""

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


CONTEXT_COMPACTION_SYSTEM_PROMPT = (
    "你是机票价格与航班事实说明助手。当前上下文已经接近模型窗口上限，"
    "必须生成面向用户的最终回答。不要再调用工具，不要输出工具调用格式，"
    "只基于已提供的工具事实进行归纳。说明已查询到的事实、来源、查询时间、"
    "限制和缺失信息；不要做审计、违规或报销通过/驳回判断。"
)


def build_context_compaction_system_prompt() -> str:
    """Return the system prompt used after context compaction."""
    return CONTEXT_COMPACTION_SYSTEM_PROMPT


def build_context_compaction_user_prompt(
    *,
    original_user_message: str,
    ledger: ObservationLedgerPrompt,
    estimate_chars: int,
    threshold_chars: int,
) -> str:
    """Build the final-answer user prompt from a compact observation ledger."""
    return (
        "不要再调用工具。请基于以下已有工具结果，直接生成面向用户的最终回答。\n\n"
        f"用户最新问题：{original_user_message}\n\n"
        "以下是已完成工具调用的工具观察账本。每条 observation 的 args 是实际调用参数，"
        "result_shape/result_stats/result_preview 是工具结果的通用摘要。\n"
        f"{ledger.to_prompt_text()}\n\n"
        "回答要求：\n"
        "- 用中文回答。\n"
        "- 只能基于工具观察账本回答，不要编造账本之外的事实。\n"
        "- 每条记录的 args 是实际调用参数；如果账本显示某些请求已调用成功，不要声称这些请求未完成。\n"
        "- 对于账本中没有出现的请求，不要编造未查询结果。\n"
        "- 如果 dropped_observation_count 大于 0，说明部分较早工具观察已因预算被丢弃。\n"
        "- 可以给出基于样本的非强制性建议，但必须说明数据限制。\n"
        "- 输出普通 Markdown 正文，不要输出 function/tool/XML/JSON 调用格式。\n\n"
        f"上下文预算提示：原请求估算 {estimate_chars} chars，阈值 {threshold_chars} chars。"
    )


__all__ = [
    "CONTEXT_COMPACTION_SYSTEM_PROMPT",
    "ObservationLedgerPrompt",
    "build_context_compaction_system_prompt",
    "build_context_compaction_user_prompt",
]
