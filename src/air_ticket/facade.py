"""Public air ticket business interfaces used by tools."""

from collections.abc import Sequence

from src.air_ticket.models import (
    AirfareQuoteResponse,
    FlightInformationResponse,
    LocationResolution,
)
from src.air_ticket.service import AirTicketService
from src.config import load_settings
from src.config.schema import AirTicketSettings


def resolve_locations(
    locations: Sequence[str],
    *,
    settings: AirTicketSettings | None = None,
) -> tuple[LocationResolution, ...]:
    """Resolve city, airport, or IATA inputs into airport candidates."""
    return _service(settings).resolve_locations(locations)


def search_quotes(
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
    settings: AirTicketSettings | None = None,
) -> AirfareQuoteResponse:
    """Search route-level airfare quote facts."""
    return _service(settings).search_quotes(
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


def query_flight(
    *,
    flight_number: str,
    date: str | None = None,
    include_price_relay: bool = True,
    currency: str = "cny",
    settings: AirTicketSettings | None = None,
) -> FlightInformationResponse:
    """Query flight facts with optional route price references."""
    return _service(settings).query_flight(
        flight_number=flight_number,
        date=date,
        include_price_relay=include_price_relay,
        currency=currency,
    )


def _service(settings: AirTicketSettings | None = None) -> AirTicketService:
    air_ticket_settings = settings or load_settings().air_ticket
    return AirTicketService(air_ticket_settings)
