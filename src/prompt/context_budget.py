"""Prompt builders for state-preserving context-budget compaction."""

from __future__ import annotations

from typing import Any, Protocol


class ObservationLedgerPrompt(Protocol):
    """Minimal prompt-facing surface for compacted observation ledgers."""

    observation_count: int
    preserved_observation_count: int
    dropped_observation_count: int
    preview_truncated_count: int

    def to_prompt_text(self) -> str:
        """Return the ledger body shown to the model."""

    def to_model_text(self) -> str:
        """Return the simplified semantic history shown to the model."""


CONTEXT_LEDGER_TOOL_NAME = "context_observation_ledger"


def build_context_ledger_tool_call_args(
    *,
    original_user_message: str,
    estimate_chars: int,
    threshold_chars: int,
) -> dict[str, str | int]:
    """Build minimal synthetic args; diagnostics stay in trace metadata."""
    _ = original_user_message, estimate_chars, threshold_chars
    return {}


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
    _ = original_user_message, estimate_chars, threshold_chars
    todo_section = _render_todo_snapshot(todo_snapshot)
    semantic_section = _render_semantic_summaries(
        local_semantic_summaries=local_semantic_summaries or [],
        global_fallback_summary=global_fallback_summary,
    )
    if include_deterministic_ledger:
        ledger_section = (
            f"{_model_ledger_text(ledger)}\n\n"
        )
        ledger_instructions = (
            "- 不要重复调用上面已经成功完成且参数相同的工具，除非用户要求刷新或补查。\n"
            "- 可以基于已知事实继续推理，但不要编造未提供的工具结果。\n"
            "- 现有信息不足时，可以继续调用工具补充事实。\n"
        )
    else:
        ledger_section = (
            "### 信息边界\n\n"
            "较早的逐条工具结果已因上下文预算折叠，只保留上面的历史结论和待处理事项。\n\n"
        )
        ledger_instructions = (
            "- 不要声称看到了未在历史结论中出现的工具结果。\n"
            "- 现有事实不足时，可以继续调用工具补充。\n"
        )
    return (
        "## 压缩后的历史工作状态\n\n"
        f"{todo_section}"
        f"{semantic_section}"
        f"{ledger_section}"
        "继续执行要求：\n"
        "- 这是历史工具观察，不是最终回答指令。\n"
        "- 继续遵循原始系统提示和当前用户问题；必要时仍可调用可用工具。\n"
        f"{ledger_instructions}"
    )


def _model_ledger_text(ledger: ObservationLedgerPrompt) -> str:
    renderer = getattr(ledger, "to_model_text", None)
    if callable(renderer):
        return str(renderer())
    return ledger.to_prompt_text()


def _render_todo_snapshot(todo_snapshot: dict[str, Any] | None) -> str:
    if not todo_snapshot:
        return ""
    items = todo_snapshot.get("items")
    if not isinstance(items, list):
        return ""
    status_labels = {
        "pending": "待处理",
        "in_progress": "进行中",
        "completed": "已完成",
        "cancelled": "已取消",
    }
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        status = str(item.get("status") or "unknown").strip()
        lines.append(f"- [{status_labels.get(status, status)}] {content}")
    if not lines:
        return ""
    return "### 任务进度\n" + "\n".join(lines) + "\n\n"


def _render_semantic_summaries(
    *,
    local_semantic_summaries: list[dict[str, Any]],
    global_fallback_summary: dict[str, Any] | None,
) -> str:
    summaries = [
        summary
        for summary in [*local_semantic_summaries, global_fallback_summary]
        if isinstance(summary, dict)
    ]
    if not summaries:
        return ""

    facts: list[str] = []
    open_items: list[str] = []
    notices: list[str] = []
    for summary in summaries:
        facts.extend(_string_items(summary.get("facts")))
        open_items.extend(_string_items(summary.get("open_items")))
        notice = summary.get("dropped_detail_notice")
        if isinstance(notice, str) and notice.strip():
            notices.append(notice.strip())

    sections: list[str] = []
    if facts:
        sections.append("### 历史结论\n" + "\n".join(f"- {fact}" for fact in facts))
    if open_items:
        sections.append(
            "### 尚待处理\n" + "\n".join(f"- {item}" for item in open_items)
        )
    if notices:
        sections.append(
            "### 信息边界\n" + "\n".join(f"- {notice}" for notice in notices)
        )
    return "\n\n".join(sections) + ("\n\n" if sections else "")


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


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
        local_semantic_summaries=localff_semantic_summaries,
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
