"""Load and run FlyClaw modules from the repository submodule."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
from importlib import import_module
import os
from pathlib import Path
import sys
import threading
from types import ModuleType
from types import SimpleNamespace
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLYCLAW_REPO_PATH = PROJECT_ROOT / "external" / "FlyClaw"
FLYCLAW_REPO_PATH = DEFAULT_FLYCLAW_REPO_PATH
_IMPORT_LOCK = threading.RLock()
_MODULE_CACHE: dict[str, ModuleType] = {}


class FlyClawCommandError(RuntimeError):
    """Raised when a FlyClaw command cannot return JSON records."""


def configure_repo_path(path: str | Path) -> None:
    """Configure the FlyClaw repository path used by provider calls."""
    global FLYCLAW_REPO_PATH
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    with _IMPORT_LOCK:
        if candidate != FLYCLAW_REPO_PATH:
            FLYCLAW_REPO_PATH = candidate
            _MODULE_CACHE.clear()


def get_airport_manager():
    """Return FlyClaw's airport manager singleton from the submodule."""
    module = import_flyclaw_module("airport_manager")
    return module.airport_manager


def resolve_airports(location: str, *, filter_inactive: bool = True) -> list[str]:
    """Resolve a city, airport, or IATA input using FlyClaw airport data."""
    manager = get_airport_manager()
    return list(manager.resolve_all(location, filter_inactive=filter_inactive))


def get_airport_info(code: str) -> dict | None:
    """Return FlyClaw airport metadata for a code when available."""
    return get_airport_manager().get_info(code)


def resolve_default_airport(location: str) -> str:
    """Resolve FlyClaw's default airport for a location."""
    return get_airport_manager().resolve(location) or ""


def run_search_command(
    *,
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None,
    cabin: str,
    adults: int,
    children: int,
    infants: int,
    stops: int | str,
    currency: str,
    limit: int | None,
    timeout_seconds: int,
    proxy_url: str,
) -> list[dict[str, Any]]:
    """Run FlyClaw's route search command and parse its JSON output."""
    flyclaw = import_flyclaw_module("flyclaw")
    args = SimpleNamespace(
        command="search",
        from_station=origin,
        to_station=destination,
        date=departure_date,
        return_date=return_date,
        cabin=cabin,
        adults=adults,
        children=children,
        infants=infants,
        stops=str(stops),
        currency=currency,
        limit=limit,
        timeout=timeout_seconds,
        return_time=None,
        sort=None,
        output="json",
        verbose=False,
        show_codeshare=False,
        layover_max_hours=None,
        compare=False,
        browser=False,
    )
    return _run_json_command(flyclaw.cmd_search, args, proxy_url=proxy_url)


def run_query_command(
    *,
    flight_number: str,
    date: str | None,
    include_price_relay: bool,
    currency: str,
    timeout_seconds: int,
    proxy_url: str,
) -> list[dict[str, Any]]:
    """Run FlyClaw's flight query command and parse its JSON output."""
    flyclaw = import_flyclaw_module("flyclaw")
    args = SimpleNamespace(
        command="query",
        flight=flight_number,
        date=date,
        currency=currency,
        timeout=timeout_seconds,
        return_time=None,
        output="json",
        verbose=False,
        show_codeshare=False,
        no_relay=not include_price_relay,
    )
    return _run_json_command(flyclaw.cmd_query, args, proxy_url=proxy_url)


def import_flyclaw_module(module_name: str) -> ModuleType:
    """Import a FlyClaw module with its repo root available on sys.path."""
    with _IMPORT_LOCK:
        cached = _MODULE_CACHE.get(module_name)
        if cached is not None:
            return cached
        with flyclaw_repo_imports():
            module = import_module(module_name)
        _MODULE_CACHE[module_name] = module
        return module


@contextmanager
def flyclaw_repo_imports() -> Iterator[None]:
    """Temporarily expose the FlyClaw submodule for its absolute imports."""
    if not FLYCLAW_REPO_PATH.exists():
        raise FileNotFoundError(
            "FlyClaw submodule is missing. Run: "
            "git submodule update --init --recursive"
        )

    repo_path = str(FLYCLAW_REPO_PATH)
    path_added = False
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        path_added = True
    try:
        yield
    finally:
        if path_added:
            try:
                sys.path.remove(repo_path)
            except ValueError:
                pass


def _run_json_command(command, args: SimpleNamespace, *, proxy_url: str) -> list[dict[str, Any]]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with _IMPORT_LOCK, _temporary_proxy(proxy_url), redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            command(args)
        except SystemExit as exc:
            raise FlyClawCommandError(_format_command_error(stderr.getvalue(), exc.code)) from exc
        except Exception as exc:
            raise FlyClawCommandError(_format_command_error(stderr.getvalue(), exc)) from exc

    output = stdout.getvalue().strip()
    if not output:
        return []
    try:
        records = json.loads(output)
    except json.JSONDecodeError as exc:
        raise FlyClawCommandError(f"FlyClaw returned non-JSON output: {output[:200]}") from exc
    if not isinstance(records, list):
        raise FlyClawCommandError("FlyClaw JSON output must be a list of records")
    return [record for record in records if isinstance(record, dict)]


def _format_command_error(stderr: str, error: object) -> str:
    details = stderr.strip()
    if details:
        return details
    return f"FlyClaw command failed: {error}"


@contextmanager
def _temporary_proxy(proxy_url: str) -> Iterator[None]:
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        if proxy_url:
            for key in keys:
                os.environ[key] = proxy_url
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
