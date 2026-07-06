"""Store construction for LangGraph long-term memory."""

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from src.config import MemoryStoreSettings


def build_store(settings: MemoryStoreSettings) -> BaseStore | None:
    """Build the configured LangGraph store, or disable it."""
    if not settings.enabled:
        return None

    if settings.type == "in_memory":
        return InMemoryStore()

    raise ValueError(f"Unsupported memory store type: {settings.type}")
