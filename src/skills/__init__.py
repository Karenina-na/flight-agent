"""Skill package public interface."""

from typing import Any

from src.skills.capabilities import SKILL_USE_PROMPT, build_skill_prompt

__all__ = [
    "SKILL_USE_PROMPT",
    "SkillCatalog",
    "SkillDescriptor",
    "SkillMiddleware",
    "build_skill_prompt",
    "build_skill_middleware",
    "load_skills_from_dir",
]


def __getattr__(name: str) -> Any:
    if name == "SkillCatalog":
        from src.skills.catalog import SkillCatalog

        return SkillCatalog
    if name == "SkillDescriptor":
        from src.skills.schema import SkillDescriptor

        return SkillDescriptor
    if name == "load_skills_from_dir":
        from src.skills.loader import load_skills_from_dir

        return load_skills_from_dir
    if name in {"SkillMiddleware", "build_skill_middleware"}:
        from src.skills.middleware import SkillMiddleware, build_skill_middleware

        return {
            "SkillMiddleware": SkillMiddleware,
            "build_skill_middleware": build_skill_middleware,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
