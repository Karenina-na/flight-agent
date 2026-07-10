"""Extract visible text from model responses used by summarization stages."""

from __future__ import annotations

from typing import Any


VISIBLE_TEXT_BLOCK_TYPES = {"text", "output_text", "message"}


def visible_response_text(response: Any) -> str:
    """Return user-visible text while excluding reasoning content blocks."""
    return visible_content_text(getattr(response, "content", response))


def visible_content_text(content: Any) -> str:
    """Extract visible text from string or Responses API content blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        if block_type and block_type not in VISIBLE_TEXT_BLOCK_TYPES:
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue
        nested_content = block.get("content")
        if isinstance(nested_content, str):
            parts.append(nested_content)
        elif isinstance(nested_content, list):
            parts.append(visible_content_text(nested_content))
    return "".join(parts)


__all__ = ["visible_content_text", "visible_response_text"]
