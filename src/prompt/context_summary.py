"""Prompt builders for semantic context compaction summaries."""

from __future__ import annotations

import json
from typing import Any, Literal


SummaryKind = Literal["local_semantic_summary", "global_fallback_summary"]


def build_semantic_summary_messages(
    *,
    kind: SummaryKind,
    latest_user_goal: str,
    bounded_context: dict[str, Any],
) -> list[dict[str, str]]:
    """Build model messages for a bounded semantic context summary."""
    schema = {
        "type": kind,
        "facts": ["客观事实，不能编造"],
        "open_items": ["仍需继续处理的事项"],
        "evidence_refs": ["message:<index> 或 tool_call:<id>"],
    }
    if kind == "global_fallback_summary":
        schema["dropped_detail_notice"] = "说明哪些细节已因预算被折叠"

    return [
        {
            "role": "system",
            "content": (
                "你是上下文压缩器。只基于输入的 bounded context 输出 JSON，"
                "不要输出 Markdown，不要调用工具，不要补充输入之外的事实。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"最新用户目标：{latest_user_goal}\n"
                f"输出 JSON schema：{json.dumps(schema, ensure_ascii=False)}\n"
                "bounded context：\n"
                f"{json.dumps(bounded_context, ensure_ascii=False, default=str)}"
            ),
        },
    ]


__all__ = ["SummaryKind", "build_semantic_summary_messages"]
