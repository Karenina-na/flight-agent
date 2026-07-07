"""Tool capability prompt layer generated from registered tool metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


TOOL_USE_PROMPT = """你可以使用以下已注册工具：

{tool_descriptions}"""


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


def build_tool_prompt(tools: Iterable[Any] | None = None) -> str:
    """Build the tool prompt layer from registered tool objects."""
    tool_list = list(tools or [])
    if not tool_list:
        return "当前没有已注册工具。"
    return TOOL_USE_PROMPT.format(tool_descriptions=_build_capability_lines(tool_list))


__all__ = ["TOOL_USE_PROMPT", "build_tool_prompt"]
