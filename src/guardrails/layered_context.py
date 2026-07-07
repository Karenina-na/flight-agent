"""Layered context-state compaction for oversized ReAct histories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from src.guardrails.tool_observation import (
    CompactObservationLedger,
    build_tool_observations,
    compact_tool_observations,
)


DEFAULT_MESSAGE_PREVIEW_CHARS = 500
ESSENTIAL_MESSAGE_PREVIEW_CHARS = 180
MIN_TOOL_LEDGER_FRACTION = 0.65


@dataclass(frozen=True)
class MessageLayerCard:
    """A compact visible-message card for non-tool historical messages."""

    source_message_index: int
    role: str
    content_summary: str
    content_chars: int
    content_sha256: str
    content_truncated: bool
    tool_calls: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        card: dict[str, Any] = {
            "source_message_index": self.source_message_index,
            "role": self.role,
            "content_summary": self.content_summary,
            "content_chars": self.content_chars,
            "content_sha256": self.content_sha256,
            "content_truncated": self.content_truncated,
        }
        if self.tool_calls:
            card["tool_calls"] = self.tool_calls
        return card


@dataclass(frozen=True)
class CompactLayeredContextState:
    """A bounded, role-layered state summary for compacted model requests."""

    strategy: str
    budget_chars: int
    system_prompt_policy: str
    latest_user_message_policy: str
    old_user_message_count: int
    preserved_old_user_message_count: int
    dropped_old_user_message_count: int
    assistant_message_count: int
    preserved_assistant_message_count: int
    dropped_assistant_message_count: int
    old_user_preview_truncated_count: int
    assistant_preview_truncated_count: int
    tool_observation_ledger: CompactObservationLedger
    old_user_messages: list[dict[str, Any]]
    assistant_messages: list[dict[str, Any]]

    @property
    def observation_count(self) -> int:
        return self.tool_observation_ledger.observation_count

    @property
    def preserved_observation_count(self) -> int:
        return self.tool_observation_ledger.preserved_observation_count

    @property
    def dropped_observation_count(self) -> int:
        return self.tool_observation_ledger.dropped_observation_count

    @property
    def preview_truncated_count(self) -> int:
        return (
            self.tool_observation_ledger.preview_truncated_count
            + self.old_user_preview_truncated_count
            + self.assistant_preview_truncated_count
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "budget_chars": self.budget_chars,
            "system_prompt_policy": self.system_prompt_policy,
            "latest_user_message_policy": self.latest_user_message_policy,
            "counts": {
                "old_user_messages": {
                    "total": self.old_user_message_count,
                    "preserved": self.preserved_old_user_message_count,
                    "dropped": self.dropped_old_user_message_count,
                    "preview_truncated": self.old_user_preview_truncated_count,
                },
                "assistant_messages": {
                    "total": self.assistant_message_count,
                    "preserved": self.preserved_assistant_message_count,
                    "dropped": self.dropped_assistant_message_count,
                    "preview_truncated": self.assistant_preview_truncated_count,
                },
                "tool_observations": {
                    "total": self.observation_count,
                    "preserved": self.preserved_observation_count,
                    "dropped": self.dropped_observation_count,
                    "preview_truncated": self.tool_observation_ledger.preview_truncated_count,
                },
            },
            "layers": {
                "old_user_messages": self.old_user_messages,
                "assistant_messages": self.assistant_messages,
                "tool_observation_ledger": self.tool_observation_ledger.to_dict(),
            },
        }

    def to_prompt_text(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)


def build_layered_context_state(
    messages: list[Any],
    *,
    budget_chars: int,
    message_preview_chars: int = DEFAULT_MESSAGE_PREVIEW_CHARS,
) -> CompactLayeredContextState:
    """Build a generic layered summary from historical ReAct messages."""
    latest_human_index = _latest_human_index(messages)
    old_user_cards = _message_cards(
        messages,
        role="human",
        skip_index=latest_human_index,
        preview_chars=message_preview_chars,
    )
    assistant_cards = _message_cards(
        messages,
        role="ai",
        skip_index=None,
        preview_chars=message_preview_chars,
    )
    tool_ledger = compact_tool_observations(
        build_tool_observations(messages),
        budget_chars=_tool_budget_chars(budget_chars),
    )
    state = _state_from_cards(
        budget_chars=budget_chars,
        old_user_original_count=len(old_user_cards),
        assistant_original_count=len(assistant_cards),
        old_user_cards=old_user_cards,
        assistant_cards=assistant_cards,
        tool_ledger=tool_ledger,
    )
    if _state_chars(state) <= budget_chars:
        return state

    old_user_cards = _message_cards(
        messages,
        role="human",
        skip_index=latest_human_index,
        preview_chars=ESSENTIAL_MESSAGE_PREVIEW_CHARS,
    )
    assistant_cards = _message_cards(
        messages,
        role="ai",
        skip_index=None,
        preview_chars=ESSENTIAL_MESSAGE_PREVIEW_CHARS,
    )
    state = _state_from_cards(
        budget_chars=budget_chars,
        old_user_original_count=len(old_user_cards),
        assistant_original_count=len(assistant_cards),
        old_user_cards=old_user_cards,
        assistant_cards=assistant_cards,
        tool_ledger=tool_ledger,
    )

    while assistant_cards and _state_chars(state) > budget_chars:
        assistant_cards = assistant_cards[1:]
        state = _state_from_cards(
            budget_chars=budget_chars,
            old_user_original_count=state.old_user_message_count,
            assistant_original_count=state.assistant_message_count,
            old_user_cards=old_user_cards,
            assistant_cards=assistant_cards,
            tool_ledger=tool_ledger,
        )
    while old_user_cards and _state_chars(state) > budget_chars:
        old_user_cards = old_user_cards[1:]
        state = _state_from_cards(
            budget_chars=budget_chars,
            old_user_original_count=state.old_user_message_count,
            assistant_original_count=state.assistant_message_count,
            old_user_cards=old_user_cards,
            assistant_cards=assistant_cards,
            tool_ledger=tool_ledger,
        )
    return state


def has_compressible_history(messages: list[Any]) -> bool:
    """Return true when at least one message can be summarized or externalized."""
    latest_human_index = _latest_human_index(messages)
    return any(index != latest_human_index for index, _ in enumerate(messages))


def _state_from_cards(
    *,
    budget_chars: int,
    old_user_original_count: int,
    assistant_original_count: int,
    old_user_cards: list[MessageLayerCard],
    assistant_cards: list[MessageLayerCard],
    tool_ledger: CompactObservationLedger,
) -> CompactLayeredContextState:
    old_user_dicts = [card.to_dict() for card in old_user_cards]
    assistant_dicts = [card.to_dict() for card in assistant_cards]
    return CompactLayeredContextState(
        strategy="layered_context_state",
        budget_chars=budget_chars,
        system_prompt_policy="external_full_preserve",
        latest_user_message_policy="external_full_preserve",
        old_user_message_count=old_user_original_count,
        preserved_old_user_message_count=len(old_user_dicts),
        dropped_old_user_message_count=old_user_original_count - len(old_user_dicts),
        assistant_message_count=assistant_original_count,
        preserved_assistant_message_count=len(assistant_dicts),
        dropped_assistant_message_count=assistant_original_count - len(assistant_dicts),
        old_user_preview_truncated_count=sum(card.content_truncated for card in old_user_cards),
        assistant_preview_truncated_count=sum(card.content_truncated for card in assistant_cards),
        tool_observation_ledger=tool_ledger,
        old_user_messages=old_user_dicts,
        assistant_messages=assistant_dicts,
    )


def _message_cards(
    messages: list[Any],
    *,
    role: str,
    skip_index: int | None,
    preview_chars: int,
) -> list[MessageLayerCard]:
    cards: list[MessageLayerCard] = []
    for index, message in enumerate(messages):
        if index == skip_index:
            continue
        if str(getattr(message, "type", "")) != role:
            continue
        content = _visible_content_text(getattr(message, "content", ""))
        summary = _truncate(content, preview_chars)
        cards.append(
            MessageLayerCard(
                source_message_index=index,
                role=role,
                content_summary=summary,
                content_chars=len(content),
                content_sha256=sha256(content.encode("utf-8")).hexdigest(),
                content_truncated=len(summary) < len(content),
                tool_calls=_tool_call_cards(getattr(message, "tool_calls", None)),
            )
        )
    return cards


def _latest_human_index(messages: list[Any]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if str(getattr(messages[index], "type", "")) == "human":
            return index
    return None


def _visible_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = str(block.get("type", ""))
                if block_type in {"reasoning", "reasoning_content"}:
                    continue
                if block_type and block_type not in {"text", "output_text", "message"}:
                    continue
                for key in ("text", "content", "summary"):
                    value = block.get(key)
                    if isinstance(value, str):
                        parts.append(value)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return "" if content is None else str(content)


def _tool_call_cards(tool_calls: Any) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for tool_call in tool_calls or []:
        if not isinstance(tool_call, dict):
            continue
        args = tool_call.get("args")
        cards.append(
            {
                "id": str(tool_call.get("id") or ""),
                "name": str(tool_call.get("name") or ""),
                "args": args if isinstance(args, dict) else {},
            }
        )
    return cards


def _tool_budget_chars(budget_chars: int) -> int:
    if budget_chars <= 0:
        return 0
    return min(budget_chars, max(round(budget_chars * MIN_TOOL_LEDGER_FRACTION), 4000))


def _state_chars(state: CompactLayeredContextState) -> int:
    return len(json.dumps(state.to_dict(), ensure_ascii=False, separators=(",", ":"), default=str))


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    return text if len(text) <= limit else f"{text[:limit]}..."


__all__ = [
    "CompactLayeredContextState",
    "MessageLayerCard",
    "build_layered_context_state",
    "has_compressible_history",
]
