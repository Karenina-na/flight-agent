"""Conversation session state for the web demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from src.runtime import Context, build_default_context

APP_USER_ID = "local-web"
DEFAULT_WORKSPACE_ID = "local-web"


@dataclass
class ChatSession:
    """Current browser conversation session."""

    thread_id: str
    turns: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def new(cls) -> "ChatSession":
        return cls(thread_id=new_thread_id())

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def context(self, *, current_user_input: str = "") -> Context:
        return build_default_context(
            user_id=APP_USER_ID,
            thread_id=self.thread_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            current_user_input=current_user_input,
            metadata={"entrypoint": "web-ui"},
        )


def new_thread_id() -> str:
    """Return a browser conversation thread id."""
    return f"web-{uuid4().hex[:8]}"


__all__ = ["APP_USER_ID", "ChatSession", "DEFAULT_WORKSPACE_ID", "new_thread_id"]
