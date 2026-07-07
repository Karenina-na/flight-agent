from src.air_ticket.providers import (
    FlyClawProvider,
    MockAirTicketInfoProvider,
    build_air_ticket_provider,
)
from src.config.schema import AirTicketSettings, FlyClawSettings


def test_mock_provider_resolves_multi_airport_city_candidates():
    provider = MockAirTicketInfoProvider()

    results = provider.resolve_locations(["北京", "上海"])

    assert [result.input for result in results] == ["北京", "上海"]
    assert results[0].airport_codes == ("PEK", "PKX")
    assert results[0].default_airport == "PEK"
    assert results[0].source == "mock"
    assert results[1].airport_codes == ("PVG", "SHA")
    assert results[1].default_airport == "PVG"


def test_mock_provider_returns_airfare_quote_facts_without_judgment():
    provider = MockAirTicketInfoProvider()

    response = provider.search_airfare_quotes(
        origin="北京",
        destination="上海",
        departure_date="2026-07-10",
        cabin="economy",
        limit=2,
    )

    assert response.query["origin"] == "北京"
    assert response.sources_used == ("mock_fliggy", "mock_google_flights")
    assert len(response.quotes) == 2
    assert response.quotes[0].flight_number == "MU5101"
    assert response.quotes[0].price == 1120
    assert response.quotes[0].currency == "CNY"
    assert "point-in-time quotes" in response.limitations[0]

    rendered = response.to_json()
    assert "reasonable" not in rendered.lower()
    assert "abnormal" not in rendered.lower()
    assert "audit" not in rendered.lower()


def test_mock_provider_returns_flight_information_with_relay_quotes():
    provider = MockAirTicketInfoProvider()

    response = provider.query_flight_information(
        flight_number="CA981",
        date="2026-07-10",
        include_price_relay=True,
    )

    assert response.flight_number == "CA981"
    assert response.sources_used == ("mock_fr24", "mock_google_flights")
    assert response.flight_records[0].origin_iata == "PEK"
    assert response.flight_records[0].destination_iata == "JFK"
    assert response.relay_quotes[0].flight_number == "CA981"
    assert response.relay_quotes[0].price == 6200


def test_flyclaw_style_records_are_normalized_to_project_models():
    provider = MockAirTicketInfoProvider()
    raw_record = {
        "source": "google_flights",
        "flight_number": "CA0981",
        "airline": "Air China",
        "origin_iata": "PEK",
        "destination_iata": "JFK",
        "scheduled_departure": "2026-07-10T13:00:00+08:00",
        "scheduled_arrival": "2026-07-10T14:30:00-04:00",
        "price": 6200,
        "currency": "CNY",
        "stops": 0,
        "duration_minutes": 840,
    }

    quote = provider.normalize_airfare_quote(raw_record)

    assert quote.source == "google_flights"
    assert quote.flight_number == "CA0981"
    assert quote.origin_iata == "PEK"
    assert quote.destination_iata == "JFK"
    assert quote.price == 6200
    assert quote.currency == "CNY"


def test_flyclaw_provider_resolves_locations_from_embedded_airport_data():
    provider = FlyClawProvider(
        FlyClawSettings(
            external_path="external/FlyClaw",
            timeout_seconds=20,
            proxy_url="",
            route_relay=True,
        )
    )

    results = provider.resolve_locations(["北京", "上海"])

    assert results[0].airport_codes == ("PEK", "PKX")
    assert results[0].default_airport == "PEK"
    assert results[0].display_name == "北京"
    assert results[0].source == "flyclaw"
    assert results[1].airport_codes == ("PVG", "SHA")


def test_flyclaw_provider_searches_quotes_with_command_runner(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "src.air_ticket.flyclaw_repo.run_search_command",
        lambda **kwargs: _fake_search_records(captured, **kwargs),
    )
    provider = FlyClawProvider(
        FlyClawSettings(
            external_path="external/FlyClaw",
            timeout_seconds=20,
            proxy_url="socks5h://127.0.0.1:1082",
            route_relay=True,
        )
    )

    response = provider.search_airfare_quotes(
        origin="北京",
        destination="上海",
        departure_date="2026-07-10",
        cabin="economy",
        limit=1,
    )

    assert response.query["origin_airports"] == ("PEK", "PKX")
    assert response.query["destination_airports"] == ("PVG", "SHA")
    assert captured["origin"] == "北京"
    assert captured["destination"] == "上海"
    assert captured["timeout_seconds"] == 20
    assert captured["proxy_url"] == "socks5h://127.0.0.1:1082"
    assert response.sources_used == ("fake_fliggy",)
    assert len(response.quotes) == 1
    assert response.quotes[0].flight_number == "MU5101"
    assert response.quotes[0].price == 1110
    assert response.limitations == (
        "Prices are point-in-time quotes.",
        "Current quotes may differ from historical ticketing prices.",
    )


def test_flyclaw_provider_queries_flight_and_relay_quotes(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "src.air_ticket.flyclaw_repo.run_query_command",
        lambda **kwargs: _fake_query_records(captured, **kwargs),
    )
    provider = FlyClawProvider(
        FlyClawSettings(
            external_path="external/FlyClaw",
            timeout_seconds=20,
            proxy_url="",
            route_relay=True,
        )
    )

    response = provider.query_flight_information(
        flight_number="CA981",
        date="2026-07-10",
        include_price_relay=True,
    )

    assert response.flight_records[0].source == "fake_fr24"
    assert response.flight_records[0].origin_iata == "PEK"
    assert response.flight_records[0].destination_iata == "JFK"
    assert len(response.flight_records) == 1
    assert response.relay_quotes[0].source == "fake_google_flights"
    assert response.relay_quotes[0].price == 6200
    assert response.sources_used == ("fake_fr24", "fake_google_flights")
    assert captured["include_price_relay"] is True


def test_flyclaw_provider_honors_route_relay_config(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "src.air_ticket.flyclaw_repo.run_query_command",
        lambda **kwargs: _fake_query_records(captured, **kwargs),
    )
    provider = FlyClawProvider(
        FlyClawSettings(
            external_path="external/FlyClaw",
            timeout_seconds=20,
            proxy_url="",
            route_relay=False,
        )
    )

    provider.query_flight_information(
        flight_number="CA981",
        date="2026-07-10",
        include_price_relay=True,
    )

    assert captured["include_price_relay"] is False


def test_build_air_ticket_provider_supports_embedded_flyclaw():
    settings = AirTicketSettings(
        provider="flyclaw",
        flyclaw=FlyClawSettings(
            external_path="external/FlyClaw",
            timeout_seconds=20,
            proxy_url="",
            route_relay=True,
        ),
    )

    provider = build_air_ticket_provider(settings)

    assert isinstance(provider, FlyClawProvider)


def _fake_search_records(captured: dict, **kwargs):
    captured.update(kwargs)
    return [
        {
            "source": "fake_fliggy",
            "flight_number": "MU5101",
            "airline": "China Eastern",
            "origin_iata": "SHA",
            "destination_iata": "PEK",
            "scheduled_departure": f"{kwargs['departure_date']}T08:00:00+08:00",
            "scheduled_arrival": f"{kwargs['departure_date']}T10:15:00+08:00",
            "cabin_class": kwargs.get("cabin", "economy"),
            "price": 1110,
            "currency": "CNY",
            "stops": 0,
            "duration_minutes": 135,
        }
    ]


def _fake_query_records(captured: dict, **kwargs):
    captured.update(kwargs)
    flight_number = kwargs["flight_number"]
    date = kwargs["date"] or "2026-07-10"
    records = [
        {
            "source": "fake_fr24",
            "flight_number": flight_number,
            "airline": "Air China",
            "origin_iata": "PEK",
            "destination_iata": "JFK",
            "scheduled_departure": f"{date}T13:00:00+08:00",
            "scheduled_arrival": f"{date}T14:30:00-04:00",
            "status": "scheduled",
            "aircraft_type": "B77W",
        }
    ]
    if kwargs["include_price_relay"]:
        records.append(
            {
                "source": "fake_google_flights",
                "flight_number": flight_number,
                "airline": "Air China",
                "origin_iata": "PEK",
                "destination_iata": "JFK",
                "scheduled_departure": f"{date}T13:00:00+08:00",
                "scheduled_arrival": f"{date}T14:30:00-04:00",
                "cabin_class": "economy",
                "price": 6200,
                "currency": "CNY",
                "stops": 0,
                "duration_minutes": 840,
            }
        )
    return records
