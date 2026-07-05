"""Typed application configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float


@dataclass(frozen=True)
class AgentSettings:
    default_thread_id: str


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    agent: AgentSettings
