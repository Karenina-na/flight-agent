"""Checkpointer construction for LangGraph memory backends."""

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from src.config import MemorySettings


def build_checkpointer(settings: MemorySettings) -> Any:
    """Build the configured LangGraph checkpointer."""
    if settings.type == "in_memory":
        return InMemorySaver()

    raise ValueError(f"Unsupported memory type: {settings.type}")
