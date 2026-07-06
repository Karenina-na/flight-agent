"""Flight information lookup tool for air ticket fact retrieval."""

from langchain.tools import tool

from src.air_ticket import query_flight
from src.air_ticket.models import to_json_payload
from src.tools.registry import register_tool


@tool
def query_flight_information(
    flight_number: str,
    date: str | None = None,
    include_price_relay: bool = True,
    currency: str = "cny",
) -> str:
    """Query flight information facts for a flight number with optional route price references. Use when the user asks about a specific flight number."""
    response = query_flight(
        flight_number=flight_number,
        date=date,
        include_price_relay=include_price_relay,
        currency=currency,
    )
    return to_json_payload(response)


register_tool(query_flight_information)
