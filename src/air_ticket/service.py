"""Service layer for air ticket fact retrieval."""

from collections.abc import Callable, Sequence
from functools import cached_property

from src.air_ticket.models import (
    AirfareQuoteResponse,
    FlightInformationResponse,
    LocationResolution,
)
from src.air_ticket.providers import (
    DEFAULT_LIMITATIONS,
    AirTicketInfoProvider,
    build_air_ticket_provider,
)
from src.air_ticket.providers import _now_iso, _query_payload
from src.config.schema import AirTicketSettings


class AirTicketService:
    """Business service used by tools to retrieve air ticket facts."""

    def __init__(
        self,
        settings: AirTicketSettings,
        provider_factory: Callable[[AirTicketSettings], AirTicketInfoProvider] | None = None,
    ):
        self.settings = settings
        self._provider_factory = provider_factory or build_air_ticket_provider

    @cached_property
    def provider(self) -> AirTicketInfoProvider:
        """Lazily build the configured provider."""
        return self._provider_factory(self.settings)

    def resolve_locations(self, locations: Sequence[str]) -> tuple[LocationResolution, ...]:
        try:
            return self.provider.resolve_locations(locations)
        except Exception as exc:
            return tuple(
                LocationResolution(
                    input=location,
                    airport_codes=(),
                    default_airport="",
                    display_name=location.strip(),
                    source="unavailable",
                )
                for location in locations
            )

    def search_quotes(
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
        try:
            return self.provider.search_airfare_quotes(
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
            )
        except Exception as exc:
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
                sources_used=(),
                quotes=(),
                limitations=DEFAULT_LIMITATIONS
                + (f"Air ticket facts are unavailable: {exc}",),
            )

    def query_flight(
        self,
        *,
        flight_number: str,
        date: str | None = None,
        include_price_relay: bool = True,
        currency: str = "cny",
    ) -> FlightInformationResponse:
        normalized_flight_number = flight_number.strip().upper()
        try:
            return self.provider.query_flight_information(
                flight_number=normalized_flight_number,
                date=date,
                include_price_relay=include_price_relay,
                currency=currency,
            )
        except Exception as exc:
            return FlightInformationResponse(
                flight_number=normalized_flight_number,
                date=date,
                captured_at=_now_iso(),
                sources_used=(),
                flight_records=(),
                relay_quotes=(),
                limitations=(
                    "Flight status sources may not include ticket prices.",
                    "Relay quotes are route-based references, not necessarily the exact ticket.",
                    f"Air ticket facts are unavailable: {exc}",
                ),
            )
