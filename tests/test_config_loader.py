from pathlib import Path

from src.config import load_settings


def test_load_settings_reads_yaml_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  provider: openai_compatible
  base_url: "http://localhost:9999/v1"
  api_key: "test-key"
  model: "test-model"
  temperature: 0.7

agent:
  default_thread_id: "test-thread"
""",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.llm.provider == "openai_compatible"
    assert settings.llm.base_url == "http://localhost:9999/v1"
    assert settings.llm.api_key == "test-key"
    assert settings.llm.model == "test-model"
    assert settings.llm.temperature == 0.7
    assert settings.agent.default_thread_id == "test-thread"


def test_load_settings_falls_back_to_example_config():
    settings = load_settings()

    assert settings.llm.base_url == "http://127.0.0.1:1234/v1"
    assert settings.llm.api_key == "not-needed"
    assert settings.llm.model == "google/gemma-4-e2b"
    assert settings.llm.temperature == 0.3
    assert settings.agent.default_thread_id == "1"


def test_load_settings_reports_missing_required_values(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  provider: openai_compatible
  base_url: "http://localhost:9999/v1"
  api_key: "test-key"
  temperature: 0.7

agent:
  default_thread_id: "test-thread"
""",
        encoding="utf-8",
    )

    try:
        load_settings(config_path)
    except ValueError as exc:
        assert "Missing required config value: model" in str(exc)
    else:
        raise AssertionError("Expected missing model config to raise ValueError")
