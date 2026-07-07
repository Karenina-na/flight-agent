"""Prompt addenda used by agent middleware."""

from __future__ import annotations


def build_memory_prompt_addendum() -> str:
    """Return long-term memory usage guidance."""
    return (
        "\n\n## Long-Term Memory\n\n"
        "可以按需使用 remember_user_fact(key, value) 保存稳定的用户偏好、"
        "项目背景或后续对话仍有价值的信息；需要回忆时调用 "
        "recall_user_facts()。只保存用户明确表达或对任务持续有用的信息。"
    )


def build_skill_prompt_addendum(skill_catalog: str) -> str:
    """Return skill catalog usage guidance."""
    return (
        "\n\n## Available Skills\n\n"
        f"{skill_catalog}\n\n"
        "Skill 使用规则：\n"
        "- 先阅读上面的名称和描述，判断是否需要技能。\n"
        "- 需要完整说明时，调用 load_skill(skill_name)。\n"
        "- 需要附属文件时，先调用 list_skill_files(skill_name)，"
        "再调用 read_skill_file(skill_name, relative_path)。"
    )


__all__ = [
    "build_memory_prompt_addendum",
    "build_skill_prompt_addendum",
]
