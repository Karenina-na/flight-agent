import json
from types import SimpleNamespace

import pytest

from src.tools import get_tools
from src.config.schema import AirTicketSettings, FlyClawSettings


@pytest.fixture(autouse=True)
def use_mock_air_ticket_provider(monkeypatch):
    settings = AirTicketSettings(
        provider="mock",
        flyclaw=FlyClawSettings(
            external_path="external/FlyClaw",
            timeout_seconds=20,
            proxy_url="",
            route_relay=True,
        ),
    )
    monkeypatch.setattr(
        "src.air_ticket.facade.load_settings",
        lambda: SimpleNamespace(air_ticket=settings),
    )


def _tool_by_name(name: str):
    return next(tool for tool in get_tools() if tool.name == name)


def test_resolve_flight_locations_tool_returns_location_facts():
    tool = _tool_by_name("resolve_flight_locations")

    payload = json.loads(tool.invoke({"locations": ["北京", "上海"]}))

    assert payload["items"][0]["airport_codes"] == ["PEK", "PKX"]
    assert payload["items"][1]["airport_codes"] == ["PVG", "SHA"]
    assert payload["items"][0]["source"] == "mock"


def test_resolve_flight_locations_tool_handles_missing_locations():
    tool = _tool_by_name("resolve_flight_locations")

    payload = json.loads(tool.invoke({"locations": []}))

    assert payload["items"] == []
    assert "locations is required" in payload["limitations"][0]


def test_location_and_date_tool_schemas_guide_model_arguments():
    location_schema = _tool_by_name(
        "resolve_flight_locations"
    ).args_schema.model_json_schema()
    date_schema = _tool_by_name("query_current_date").args_schema.model_json_schema()

    assert "['北京','上海']" in location_schema["properties"]["locations"]["description"]
    assert "明天/tomorrow=1" in date_schema["properties"]["days_offset"]["description"]
    assert "后天/day after tomorrow=2" in date_schema["properties"]["days_offset"]["description"]


def test_air_ticket_tool_schemas_match_required_argument_descriptions():
    location_schema = _tool_by_name(
        "resolve_flight_locations"
    ).args_schema.model_json_schema()
    quote_schema = _tool_by_name("search_airfare_quotes").args_schema.model_json_schema()
    flight_schema = _tool_by_name(
        "query_flight_information"
    ).args_schema.model_json_schema()

    assert location_schema["required"] == ["locations"]
    assert "default" not in location_schema["properties"]["locations"]
    assert location_schema["properties"]["locations"]["type"] == "array"
    assert "anyOf" not in location_schema["properties"]["locations"]

    assert quote_schema["required"] == ["origin", "destination", "departure_date"]
    assert flight_schema["required"] == ["flight_number"]


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


def test_query_current_date_tool_returns_date_facts():
    tool = _tool_by_name("query_current_date")

    payload = json.loads(
        tool.invoke({"days_offset": 1, "timezone_name": "Asia/Shanghai"})
    )

    assert payload["timezone"] == "Asia/Shanghai"
    assert payload["days_offset"] == 1
    assert payload["current_date"]
    assert payload["current_datetime"]
    assert payload["target_date"] >= payload["current_date"]
    assert payload["limitations"] == []


def test_query_current_date_tool_falls_back_for_unknown_timezone():
    tool = _tool_by_name("query_current_date")

    payload = json.loads(tool.invoke({"timezone_name": "Missing/Timezone"}))

    assert payload["timezone"] == "Asia/Shanghai"
    assert payload["limitations"]
