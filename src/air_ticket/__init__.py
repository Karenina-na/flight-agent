"""Air ticket fact retrieval domain package."""

from src.air_ticket.facade import query_flight, resolve_locations, search_quotes
from src.air_ticket.providers import (
    AirTicketInfoProvider,
    FlyClawProvider,
    MockAirTicketInfoProvider,
    build_air_ticket_provider,
)
from src.air_ticket.service import AirTicketService

__all__ = [
    "AirTicketInfoProvider",
    "AirTicketService",
    "FlyClawProvider",
    "MockAirTicketInfoProvider",
    "build_air_ticket_provider",
    "query_flight",
    "resolve_locations",
    "search_quotes",
]
