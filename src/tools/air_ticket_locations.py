"""Location resolution tool for air ticket fact retrieval."""

import json

from langchain.tools import tool

from src.air_ticket import resolve_locations
from src.tools.registry import register_tool


@tool
def resolve_flight_locations(locations: list[str]) -> str:
    """Resolve city, airport, or IATA inputs into flight location facts."""
    results = resolve_locations(locations)
    return json.dumps(
        {"items": [result.__dict__ for result in results]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


register_tool(resolve_flight_locations)
