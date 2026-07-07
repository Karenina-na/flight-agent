import os
from types import SimpleNamespace

import pytest

from src.air_ticket import flyclaw_repo


def test_configure_repo_path_accepts_relative_and_absolute_paths(tmp_path):
    original = flyclaw_repo.FLYCLAW_REPO_PATH
    try:
        flyclaw_repo.configure_repo_path("external/FlyClaw")
        assert flyclaw_repo.FLYCLAW_REPO_PATH == flyclaw_repo.DEFAULT_FLYCLAW_REPO_PATH

        flyclaw_repo.configure_repo_path(tmp_path)
        assert flyclaw_repo.FLYCLAW_REPO_PATH == tmp_path
    finally:
        flyclaw_repo.configure_repo_path(original)


def test_run_json_command_parses_records_and_restores_proxy(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://old-proxy")
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)
    seen_proxy = {}

    def command(args):
        seen_proxy["http"] = os.environ.get("HTTP_PROXY")
        seen_proxy["https"] = os.environ.get("HTTPS_PROXY")
        seen_proxy["all"] = os.environ.get("ALL_PROXY")
        print('[{"source":"fake","flight_number":"CA981"}]')

    records = flyclaw_repo._run_json_command(
        command,
        SimpleNamespace(),
        proxy_url="socks5h://127.0.0.1:1082",
    )

    assert records == [{"source": "fake", "flight_number": "CA981"}]
    assert seen_proxy == {
        "http": "socks5h://127.0.0.1:1082",
        "https": "socks5h://127.0.0.1:1082",
        "all": "socks5h://127.0.0.1:1082",
    }
    assert os.environ["HTTP_PROXY"] == "http://old-proxy"
    assert "HTTPS_PROXY" not in os.environ
    assert "ALL_PROXY" not in os.environ


def test_run_json_command_raises_for_non_json_output():
    def command(args):
        print("not json")

    with pytest.raises(flyclaw_repo.FlyClawCommandError, match="non-JSON"):
        flyclaw_repo._run_json_command(command, SimpleNamespace(), proxy_url="")
