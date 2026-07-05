"""YAML-backed configuration loading."""

from pathlib import Path
from typing import Any

import yaml

from src.config.schema import AgentSettings, LLMSettings, Settings

DEFAULT_CONFIG_PATH = Path("config.yaml")
EXAMPLE_CONFIG_PATH = Path("config.example.yaml")


def load_settings(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load settings from YAML, falling back to the committed example file."""
    path = Path(config_path)
    if not path.exists() and path == DEFAULT_CONFIG_PATH:
        path = EXAMPLE_CONFIG_PATH

    raw_config = _load_yaml(path)
    llm_config = _require_mapping(raw_config, "llm")
    agent_config = _require_mapping(raw_config, "agent")

    return Settings(
        llm=LLMSettings(
            provider=_get_str(llm_config, "provider"),
            base_url=_get_str(llm_config, "base_url"),
            api_key=_get_str(llm_config, "api_key"),
            model=_get_str(llm_config, "model"),
            temperature=_get_float(llm_config, "temperature"),
        ),
        agent=AgentSettings(
            default_thread_id=_get_str(agent_config, "default_thread_id"),
        ),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return value


def _get_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if value is None:
        raise ValueError(f"Missing required config value: {key}")
    return str(value)


def _get_float(config: dict[str, Any], key: str) -> float:
    value = config.get(key)
    if value is None:
        raise ValueError(f"Missing required config value: {key}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config value '{key}' must be a float") from exc
