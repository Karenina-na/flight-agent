"""Typed models for air ticket fact retrieval."""

from dataclasses import asdict, dataclass
import json
from typing import Any


@dataclass(frozen=True)
class LocationResolution:
    input: str
    airport_codes: tuple[str, ...]
    default_airport: str
    display_name: str
    source: str


@dataclass(frozen=True)
class AirfareQuote:
    source: str
    flight_number: str
    airline: str
    origin_iata: str
    destination_iata: str
    scheduled_departure: str
    scheduled_arrival: str
    cabin_class: str
    price: float | None
    currency: str
    stops: int | str | None
    duration_minutes: int | None


@dataclass(frozen=True)
class FlightRecord:
    source: str
    flight_number: str
    airline: str
    origin_iata: str
    destination_iata: str
    scheduled_departure: str
    scheduled_arrival: str
    status: str
    aircraft_type: str


@dataclass(frozen=True)
class AirfareQuoteResponse:
    query: dict[str, Any]
    captured_at: str
    sources_used: tuple[str, ...]
    quotes: tuple[AirfareQuote, ...]
    limitations: tuple[str, ...]

    def to_json(self) -> str:
        return _to_json(self)


@dataclass(frozen=True)
class FlightInformationResponse:
    flight_number: str
    date: str | None
    captured_at: str
    sources_used: tuple[str, ...]
    flight_records: tuple[FlightRecord, ...]
    relay_quotes: tuple[AirfareQuote, ...]
    limitations: tuple[str, ...]

    def to_json(self) -> str:
        return _to_json(self)


def to_json_payload(payload: Any) -> str:
    """Render a dataclass payload as stable JSON for tool responses."""
    return _to_json(payload)


def _to_json(payload: Any) -> str:
    return json.dumps(asdict(payload), ensure_ascii=False, separators=(",", ":"))
