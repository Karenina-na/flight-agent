"""Prompt package public interface."""

from src.prompt.base import CORE_PROMPT, DOMAIN_PROMPT
from src.prompt.build import SYSTEM_PROMPT, build_system_prompt
from src.prompt.context_budget import (
    CONTEXT_COMPACTION_SYSTEM_PROMPT,
    build_context_compaction_system_prompt,
    build_context_compaction_user_prompt,
)
from src.prompt.middleware import (
    build_memory_prompt_addendum,
    build_skill_prompt_addendum,
)

__all__ = [
    "CORE_PROMPT",
    "CONTEXT_COMPACTION_SYSTEM_PROMPT",
    "DOMAIN_PROMPT",
    "SYSTEM_PROMPT",
    "build_context_compaction_system_prompt",
    "build_context_compaction_user_prompt",
    "build_memory_prompt_addendum",
    "build_skill_prompt_addendum",
    "build_system_prompt",
]
