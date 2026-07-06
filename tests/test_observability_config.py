from pathlib import Path

from src.config import load_settings


def test_load_settings_reports_invalid_logging_level(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
observability:
  logging:
    enabled: true
    level: "VERBOSE"
    format: "text"
    redact: true
""",
        encoding="utf-8",
    )

    try:
        load_settings(config_path)
    except ValueError as exc:
        assert "observability.logging.level" in str(exc)
    else:
        raise AssertionError("Expected invalid logging level to raise ValueError")


def test_load_settings_reports_invalid_logging_format(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
observability:
  logging:
    enabled: true
    level: "INFO"
    format: "xml"
    redact: true
""",
        encoding="utf-8",
    )

    try:
        load_settings(config_path)
    except ValueError as exc:
        assert "observability.logging.format" in str(exc)
    else:
        raise AssertionError("Expected invalid logging format to raise ValueError")
