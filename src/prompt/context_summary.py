"""Prompt builders for semantic context compaction summaries."""

from __future__ import annotations

from typing import Any, Literal


SummaryKind = Literal["local_semantic_summary", "global_fallback_summary"]


def build_semantic_summary_messages(
    *,
    kind: SummaryKind,
    latest_user_goal: str,
    bounded_context: dict[str, Any],
) -> list[dict[str, str]]:
    """Build model messages for a bounded free-form context summary."""
    summary_scope = (
        "全局兜底摘要：只保留继续完成任务必须知道的事实、边界和待处理事项。"
        if kind == "global_fallback_summary"
        else "局部历史摘要：保留旧上下文里的关键事实、已完成工作和待处理事项。"
    )
    return [
        {
            "role": "system",
            "content": (
                "你是上下文压缩器。只基于输入的 bounded context 生成一段简洁中文摘要。"
                "不要调用工具，不要补充输入之外的事实，不要把摘要写成代码或 JSON schema。"
                "输出内容会被作为后续模型可读的历史上下文。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"最新用户目标：{latest_user_goal}\n"
                f"压缩范围：{summary_scope}\n"
                "输出要求：\n"
                "- 直接输出摘要正文，不要包裹 JSON、Markdown 代码块或额外说明。\n"
                "- 用短段落或短项目符号保留客观事实、已完成动作、信息边界和仍需继续的事项。\n"
                "- 如果事实不足，明确写出不足之处；不要编造工具结果。\n"
                "bounded context：\n"
                f"{bounded_context}"
            ),
        },
    ]


__all__ = ["SummaryKind", "build_semantic_summary_messages"]
