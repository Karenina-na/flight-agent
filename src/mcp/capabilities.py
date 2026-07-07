"""MCP capability prompt layer generated from runtime metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


MCP_USE_PROMPT = """你可以连接以下 MCP 能力：

{mcp_descriptions}"""


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


def build_mcp_prompt(mcp_servers: Iterable[Any] | None = None) -> str:
    """Build the MCP prompt layer from optional MCP capability descriptors."""
    mcp_list = list(mcp_servers or [])
    if not mcp_list:
        return "当前没有已注册 MCP 能力。"
    return MCP_USE_PROMPT.format(mcp_descriptions=_build_capability_lines(mcp_list))


__all__ = ["MCP_USE_PROMPT", "build_mcp_prompt"]
