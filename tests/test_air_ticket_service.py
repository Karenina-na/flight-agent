from src.air_ticket.models import (
    AirfareQuote,
    AirfareQuoteResponse,
    FlightInformationResponse,
    FlightRecord,
    LocationResolution,
)
from src.air_ticket.service import AirTicketService
from src.config.schema import AirTicketSettings, FlyClawSettings


def _settings() -> AirTicketSettings:
    return AirTicketSettings(
        provider="mock",
        flyclaw=FlyClawSettings(
            external_path="external/FlyClaw",
            timeout_seconds=20,
            proxy_url="",
            route_relay=True,
        ),
    )


def test_service_builds_provider_lazily():
    created = []

    def provider_factory(settings):
        created.append(settings)
        return _FakeProvider()

    service = AirTicketService(_settings(), provider_factory=provider_factory)

    assert created == []

    results = service.resolve_locations(["北京"])

    assert created == [service.settings]
    assert results[0].airport_codes == ("PEK",)


def test_service_returns_empty_quote_response_when_provider_fails():
    service = AirTicketService(_settings(), provider_factory=lambda settings: _FailingProvider())

    response = service.search_quotes(
        origin="北京",
        destination="上海",
        departure_date="2026-07-10",
    )

    assert response.query["origin"] == "北京"
    assert response.sources_used == ()
    assert response.quotes == ()
    assert "Air ticket facts are unavailable" in response.limitations[-1]
    assert "audit" not in response.to_json().lower()


def test_service_normalizes_empty_optional_quote_arguments():
    service = AirTicketService(_settings(), provider_factory=lambda settings: _FakeProvider())

    response = service.search_quotes(
        origin="北京",
        destination="上海",
        departure_date="2026-07-10",
        return_date="",
        stops="",
    )

    assert response.query["return_date"] is None
    assert response.query["stops"] == 0


def test_service_returns_empty_flight_response_when_provider_fails():
    service = AirTicketService(_settings(), provider_factory=lambda settings: _FailingProvider())

    response = service.query_flight(flight_number=" ca981 ", date="2026-07-10")

    assert response.flight_number == "CA981"
    assert response.flight_records == ()
    assert response.relay_quotes == ()
    assert response.sources_used == ()
    assert "Air ticket facts are unavailable" in response.limitations[-1]


class _FakeProvider:
    def resolve_locations(self, locations):
        return (
            LocationResolution(
                input=locations[0],
                airport_codes=("PEK",),
                default_airport="PEK",
                display_name="北京",
                source="fake",
            ),
        )

    def search_airfare_quotes(self, **kwargs):
        quote = AirfareQuote(
            source="fake",
            flight_number="CA1501",
            airline="Air China",
            origin_iata="PEK",
            destination_iata="SHA",
            scheduled_departure="2026-07-10T12:30:00+08:00",
            scheduled_arrival="2026-07-10T14:45:00+08:00",
            cabin_class="economy",
            price=1200,
            currency="CNY",
            stops=0,
            duration_minutes=135,
        )
        return AirfareQuoteResponse(
            query=kwargs,
            captured_at="2026-07-06T00:00:00+08:00",
            sources_used=("fake",),
            quotes=(quote,),
            limitations=(),
        )

    def query_flight_information(self, **kwargs):
        record = FlightRecord(
            source="fake",
            flight_number=kwargs["flight_number"],
            airline="Air China",
            origin_iata="PEK",
            destination_iata="JFK",
            scheduled_departure="2026-07-10T13:00:00+08:00",
            scheduled_arrival="2026-07-10T14:30:00-04:00",
            status="scheduled",
            aircraft_type="B77W",
        )
        return FlightInformationResponse(
            flight_number=kwargs["flight_number"],
            date=kwargs.get("date"),
            captured_at="2026-07-06T00:00:00+08:00",
            sources_used=("fake",),
            flight_records=(record,),
            relay_quotes=(),
            limitations=(),
        )


class _FailingProvider:
    def resolve_locations(self, locations):
        raise RuntimeError("source down")

    def search_airfare_quotes(self, **kwargs):
        raise RuntimeError("source down")

    def query_flight_information(self, **kwargs):
        raise RuntimeError("source down")
