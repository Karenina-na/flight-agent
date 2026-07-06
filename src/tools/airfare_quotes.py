"""Airfare quote search tool for air ticket fact retrieval."""

from langchain.tools import tool

from src.air_ticket import search_quotes
from src.tools.registry import register_tool


@tool
def search_airfare_quotes(
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
) -> str:
    """Search airfare quote facts without judging price reasonableness."""
    response = search_quotes(
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
    return response.to_json()


register_tool(search_airfare_quotes)
