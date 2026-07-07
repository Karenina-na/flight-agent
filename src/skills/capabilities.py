"""Skill capability prompt layer generated from registered skill metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


SKILL_USE_PROMPT = """你可以参考以下已注册技能：

{skill_descriptions}"""


def _get_name(item: Any) -> str:
    return str(getattr(item, "name", None) or getattr(item, "__name__", "unnamed"))


def _get_description(item: Any) -> str:
    description = getattr(item, "description", None) or getattr(item, "__doc__", None)
    return str(description).strip() if description else "无描述"


def _build_capability_lines(capabilities: Iterable[Any]) -> str:
    return "\n".join(
        f"- {_get_name(capability)}：{_get_description(capability)}"
        for capability in capabilities
    )


def build_skill_prompt(skills: Iterable[Any] | None = None) -> str:
    """Build the skill prompt layer from optional skill descriptors."""
    skill_list = list(skills or [])
    if not skill_list:
        return "当前没有已注册技能。"
    return SKILL_USE_PROMPT.format(skill_descriptions=_build_capability_lines(skill_list))


__all__ = ["SKILL_USE_PROMPT", "build_skill_prompt"]
