"""Provider interface and providers for air ticket fact retrieval."""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Protocol

from src.air_ticket import flyclaw_repo
from src.air_ticket.models import (
    AirfareQuote,
    AirfareQuoteResponse,
    FlightInformationResponse,
    LocationResolution,
)
from src.air_ticket.normalization import (
    normalize_airfare_quote,
    normalize_flight_record,
)
from src.config.schema import AirTicketSettings
from src.config.schema import FlyClawSettings


DEFAULT_LIMITATIONS = (
    "Prices are point-in-time quotes.",
    "Current quotes may differ from historical ticketing prices.",
)


class AirTicketInfoProvider(Protocol):
    """Fact retrieval provider used by air ticket tools."""

    def resolve_locations(self, locations: Sequence[str]) -> tuple[LocationResolution, ...]:
        ...

    def search_airfare_quotes(
        self,
        *,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: str | None = None,
        cabin: str = "economy",
        adults: int = 1,
        children: int = 0,
        infants: int = 0,
        stops: int | str = 0,
        currency: str = "cny",
        limit: int = 20,
    ) -> AirfareQuoteResponse:
        ...

    def query_flight_information(
        self,
        *,
        flight_number: str,
        date: str | None = None,
        include_price_relay: bool = True,
        currency: str = "cny",
    ) -> FlightInformationResponse:
        ...


class MockAirTicketInfoProvider:
    """Deterministic provider for tests and local demos without network access."""

    _LOCATIONS = {
        "北京": ("PEK", "PKX", "PEK", "北京"),
        "beijing": ("PEK", "PKX", "PEK", "Beijing"),
        "上海": ("PVG", "SHA", "PVG", "上海"),
        "shanghai": ("PVG", "SHA", "PVG", "Shanghai"),
        "纽约": ("JFK", "EWR", "LGA", "JFK", "纽约"),
        "new york": ("JFK", "EWR", "LGA", "JFK", "New York"),
    }

    def resolve_locations(self, locations: Sequence[str]) -> tuple[LocationResolution, ...]:
        results = []
        for location in locations:
            key = location.strip().lower()
            match = self._LOCATIONS.get(key)
            if match:
                *airport_codes, default_airport, display_name = match
            else:
                code = location.strip().upper()
                airport_codes = [code] if len(code) == 3 else []
                default_airport = airport_codes[0] if airport_codes else ""
                display_name = location.strip()
            results.append(
                LocationResolution(
                    input=location,
                    airport_codes=tuple(airport_codes),
                    default_airport=default_airport,
                    display_name=display_name,
                    source="mock",
                )
            )
        return tuple(results)

    def search_airfare_quotes(
        self,
        *,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: str | None = None,
        cabin: str = "economy",
        adults: int = 1,
        children: int = 0,
        infants: int = 0,
        stops: int | str = 0,
        currency: str = "cny",
        limit: int = 20,
    ) -> AirfareQuoteResponse:
        records = [
            {
                "source": "mock_fliggy",
                "flight_number": "MU5101",
                "airline": "China Eastern",
                "origin_iata": "SHA",
                "destination_iata": "PEK",
                "scheduled_departure": f"{departure_date}T08:00:00+08:00",
                "scheduled_arrival": f"{departure_date}T10:15:00+08:00",
                "cabin_class": cabin,
                "price": 1120,
                "currency": "CNY",
                "stops": 0,
                "duration_minutes": 135,
            },
            {
                "source": "mock_google_flights",
                "flight_number": "CA1501",
                "airline": "Air China",
                "origin_iata": "PEK",
                "destination_iata": "SHA",
                "scheduled_departure": f"{departure_date}T12:30:00+08:00",
                "scheduled_arrival": f"{departure_date}T14:45:00+08:00",
                "cabin_class": cabin,
                "price": 1540,
                "currency": "CNY",
                "stops": 0,
                "duration_minutes": 135,
            },
            {
                "source": "mock_skiplagged",
                "flight_number": "HO1254",
                "airline": "Juneyao Air",
                "origin_iata": "PVG",
                "destination_iata": "PKX",
                "scheduled_departure": f"{departure_date}T17:20:00+08:00",
                "scheduled_arrival": f"{departure_date}T19:40:00+08:00",
                "cabin_class": cabin,
                "price": 1680,
                "currency": "CNY",
                "stops": 0,
                "duration_minutes": 140,
            },
        ][:limit]
        return AirfareQuoteResponse(
            query=_query_payload(
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                return_date=return_date,
                cabin=cabin,
                adults=adults,
                children=children,
                infants=infants,
                stops=stops,
                currency=currency,
            ),
            captured_at=_now_iso(),
            sources_used=tuple(record["source"] for record in records),
            quotes=tuple(normalize_airfare_quote(record) for record in records),
            limitations=DEFAULT_LIMITATIONS,
        )

    def query_flight_information(
        self,
        *,
        flight_number: str,
        date: str | None = None,
        include_price_relay: bool = True,
        currency: str = "cny",
    ) -> FlightInformationResponse:
        effective_date = date or "2026-07-10"
        flight_number = flight_number.strip().upper()
        flight_record = normalize_flight_record(
            {
                "source": "mock_fr24",
                "flight_number": flight_number,
                "airline": "Air China",
                "origin_iata": "PEK",
                "destination_iata": "JFK",
                "scheduled_departure": f"{effective_date}T13:00:00+08:00",
                "scheduled_arrival": f"{effective_date}T14:30:00-04:00",
                "status": "scheduled",
                "aircraft_type": "B77W",
            }
        )
        relay_quotes: tuple[AirfareQuote, ...] = ()
        sources = ["mock_fr24"]
        if include_price_relay:
            relay_quotes = (
                normalize_airfare_quote(
                    {
                        "source": "mock_google_flights",
                        "flight_number": flight_number,
                        "airline": "Air China",
                        "origin_iata": "PEK",
                        "destination_iata": "JFK",
                        "scheduled_departure": f"{effective_date}T13:00:00+08:00",
                        "scheduled_arrival": f"{effective_date}T14:30:00-04:00",
                        "cabin_class": "economy",
                        "price": 6200,
                        "currency": "CNY",
                        "stops": 0,
                        "duration_minutes": 840,
                    }
                ),
            )
            sources.append("mock_google_flights")
        return FlightInformationResponse(
            flight_number=flight_number,
            date=date,
            captured_at=_now_iso(),
            sources_used=tuple(sources),
            flight_records=(flight_record,),
            relay_quotes=relay_quotes,
            limitations=(
                "Flight status sources may not include ticket prices.",
                "Relay quotes are route-based references, not necessarily the exact ticket.",
            ),
        )

    @staticmethod
    def normalize_airfare_quote(record: dict) -> AirfareQuote:
        return normalize_airfare_quote(record)


class FlyClawProvider:
    """Provider that calls FlyClaw submodule command orchestration."""

    def __init__(self, settings: FlyClawSettings):
        self.settings = settings

    def resolve_locations(self, locations: Sequence[str]) -> tuple[LocationResolution, ...]:
        results = []
        for location in locations:
            codes = tuple(flyclaw_repo.resolve_airports(location, filter_inactive=True))
            default_airport = flyclaw_repo.resolve_default_airport(location) or (
                codes[0] if codes else ""
            )
            results.append(
                LocationResolution(
                    input=location,
                    airport_codes=codes,
                    default_airport=default_airport,
                    display_name=_display_name(
                        default_airport,
                        location,
                    ),
                    source="flyclaw",
                )
            )
        return tuple(results)

    def search_airfare_quotes(
        self,
        *,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: str | None = None,
        cabin: str = "economy",
        adults: int = 1,
        children: int = 0,
        infants: int = 0,
        stops: int | str = 0,
        currency: str = "cny",
        limit: int = 20,
    ) -> AirfareQuoteResponse:
        origin_airports = tuple(flyclaw_repo.resolve_airports(origin) or [origin])
        destination_airports = tuple(flyclaw_repo.resolve_airports(destination) or [destination])
        records = flyclaw_repo.run_search_command(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            return_date=return_date,
            cabin=cabin,
            adults=adults,
            children=children,
            infants=infants,
            stops=stops,
            currency=currency,
            limit=limit,
            timeout_seconds=self.settings.timeout_seconds,
            proxy_url=self.settings.proxy_url,
        )[:limit]
        return AirfareQuoteResponse(
            query={
                **_query_payload(
                    origin=origin,
                    destination=destination,
                    departure_date=departure_date,
                    return_date=return_date,
                    cabin=cabin,
                    adults=adults,
                    children=children,
                    infants=infants,
                    stops=stops,
                    currency=currency,
                ),
                "origin_airports": origin_airports,
                "destination_airports": destination_airports,
            },
            captured_at=_now_iso(),
            sources_used=_sources_from_records(records),
            quotes=tuple(normalize_airfare_quote(record) for record in records),
            limitations=DEFAULT_LIMITATIONS,
        )

    def query_flight_information(
        self,
        *,
        flight_number: str,
        date: str | None = None,
        include_price_relay: bool = True,
        currency: str = "cny",
    ) -> FlightInformationResponse:
        flight_number = flight_number.strip().upper()
        records = flyclaw_repo.run_query_command(
            flight_number=flight_number,
            date=date,
            include_price_relay=include_price_relay and self.settings.route_relay,
            currency=currency,
            timeout_seconds=self.settings.timeout_seconds,
            proxy_url=self.settings.proxy_url,
        )
        flight_records = [record for record in records if _is_flight_record(record)]
        relay_records = [record for record in records if record.get("price") is not None]
        return FlightInformationResponse(
            flight_number=flight_number,
            date=date,
            captured_at=_now_iso(),
            sources_used=_sources_from_records(records),
            flight_records=tuple(normalize_flight_record(record) for record in flight_records),
            relay_quotes=tuple(
                normalize_airfare_quote(record)
                for record in relay_records
                if record.get("price") is not None
            ),
            limitations=(
                "Flight status sources may not include ticket prices.",
                "Relay quotes are route-based references, not necessarily the exact ticket.",
            ),
        )

def build_air_ticket_provider(settings: AirTicketSettings) -> AirTicketInfoProvider:
    """Build the configured provider with a safe mock default."""
    if settings.provider == "flyclaw":
        return FlyClawProvider(settings.flyclaw)
    return MockAirTicketInfoProvider()


def _query_payload(
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
) -> dict:
    return {
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
        "return_date": return_date,
        "cabin": cabin,
        "adults": adults,
        "children": children,
        "infants": infants,
        "stops": stops,
        "currency": currency,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sources_from_records(records: Sequence[dict]) -> tuple[str, ...]:
    seen = []
    for record in records:
        source = str(record.get("source") or "")
        if source and source not in seen:
            seen.append(source)
    return tuple(seen)


def _is_flight_record(record: dict) -> bool:
    if record.get("status") or record.get("aircraft_type"):
        return True
    return record.get("price") is None


def _display_name(code: str, fallback: str) -> str:
    if not code:
        return fallback.strip()
    info = flyclaw_repo.get_airport_info(code)
    if not info:
        return code
    return info.get("city_cn") or info.get("city_en") or code
