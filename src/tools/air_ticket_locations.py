"""Location resolution tool for air ticket fact retrieval."""

import json
from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from src.air_ticket import resolve_locations
from src.tools.registry import register_tool


@tool
def resolve_flight_locations(
    locations: Annotated[
        list[str] | None,
        Field(
            description=(
                "List of city, airport, or IATA inputs to resolve, "
                "for example ['北京','上海']."
            )
        ),
    ] = None,
) -> str:
    """Resolve city, airport, or IATA inputs into flight location facts. Use when the user gives city or airport names instead of exact airport codes. Provide locations as a list, for example ["北京","上海"]."""
    if not locations:
        return json.dumps(
            {
                "items": [],
                "limitations": [
                    "locations is required; provide a list such as ['北京','上海']."
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    results = resolve_locations(locations)
    return json.dumps(
        {"items": [result.__dict__ for result in results]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


register_tool(resolve_flight_locations)
