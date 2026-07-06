import json

from src.tools import get_tools


def _tool_by_name(name: str):
    return next(tool for tool in get_tools() if tool.name == name)


def test_resolve_flight_locations_tool_returns_location_facts():
    tool = _tool_by_name("resolve_flight_locations")

    payload = json.loads(tool.invoke({"locations": ["北京", "上海"]}))

    assert payload["items"][0]["airport_codes"] == ["PEK", "PKX"]
    assert payload["items"][1]["airport_codes"] == ["PVG", "SHA"]
    assert payload["items"][0]["source"] == "mock"


def test_search_airfare_quotes_tool_returns_quote_facts_only():
    tool = _tool_by_name("search_airfare_quotes")

    payload = json.loads(
        tool.invoke(
            {
                "origin": "北京",
                "destination": "上海",
                "departure_date": "2026-07-10",
                "cabin": "economy",
                "limit": 2,
            }
        )
    )

    assert payload["query"]["origin"] == "北京"
    assert payload["sources_used"] == ["mock_fliggy", "mock_google_flights"]
    assert len(payload["quotes"]) == 2
    assert payload["quotes"][0]["price"] == 1120

    rendered = json.dumps(payload, ensure_ascii=False).lower()
    assert "reasonable" not in rendered
    assert "abnormal" not in rendered
    assert "audit" not in rendered


def test_query_flight_information_tool_returns_flight_and_relay_facts():
    tool = _tool_by_name("query_flight_information")

    payload = json.loads(
        tool.invoke(
            {
                "flight_number": "CA981",
                "date": "2026-07-10",
                "include_price_relay": True,
            }
        )
    )

    assert payload["flight_number"] == "CA981"
    assert payload["flight_records"][0]["origin_iata"] == "PEK"
    assert payload["relay_quotes"][0]["price"] == 6200
    assert payload["sources_used"] == ["mock_fr24", "mock_google_flights"]
