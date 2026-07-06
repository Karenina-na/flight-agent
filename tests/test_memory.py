from langgraph.checkpoint.memory import InMemorySaver

from src.config import MemorySettings
from src.memory import build_checkpointer


def test_build_checkpointer_returns_in_memory_saver():
    checkpointer = build_checkpointer(MemorySettings(type="in_memory"))

    assert isinstance(checkpointer, InMemorySaver)
