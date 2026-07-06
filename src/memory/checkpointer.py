"""Checkpointer construction for LangGraph graph-state memory."""

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from src.config import MemoryCheckpointerSettings


def build_checkpointer(settings: MemoryCheckpointerSettings) -> Any:
    """Build the configured LangGraph checkpointer."""
    if settings.type == "in_memory":
        return InMemorySaver()

    raise ValueError(f"Unsupported memory checkpointer type: {settings.type}")
