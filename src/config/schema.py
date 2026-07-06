"""Typed application configuration."""

from dataclasses import dataclass
from typing import Literal

MemoryType = Literal["in_memory"]
WindowClauseType = Literal["fraction", "tokens", "messages"]


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float
    context_window_tokens: int


@dataclass(frozen=True)
class AgentSettings:
    default_thread_id: str


@dataclass(frozen=True)
class MemorySettings:
    type: MemoryType


@dataclass(frozen=True)
class WindowClauseSettings:
    type: WindowClauseType
    value: float | int


@dataclass(frozen=True)
class SummarizationSettings:
    enabled: bool
    model: str
    trigger: WindowClauseSettings
    keep: WindowClauseSettings
    trim_tokens_to_summarize: int | None


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    agent: AgentSettings
    memory: MemorySettings
    summarization: SummarizationSettings
