from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from src.config import MemoryCheckpointerSettings, MemoryStoreSettings
from src.memory import build_checkpointer, build_store


def test_build_checkpointer_returns_in_memory_saver():
    checkpointer = build_checkpointer(MemoryCheckpointerSettings(type="in_memory"))

    assert isinstance(checkpointer, InMemorySaver)


def test_build_store_returns_in_memory_store_when_enabled():
    store = build_store(MemoryStoreSettings(enabled=True, type="in_memory"))

    assert isinstance(store, InMemoryStore)


def test_build_store_returns_none_when_disabled():
    store = build_store(MemoryStoreSettings(enabled=False, type="in_memory"))

    assert store is None
