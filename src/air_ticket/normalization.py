"""Normalize FlyClaw-style records into project-owned air ticket models."""

from src.air_ticket.models import AirfareQuote, FlightRecord


def normalize_airfare_quote(record: dict) -> AirfareQuote:
    """Normalize a quote-like record from FlyClaw or a compatible provider."""
    return AirfareQuote(
        source=str(record.get("source") or ""),
        flight_number=str(record.get("flight_number") or ""),
        airline=str(record.get("airline") or ""),
        origin_iata=str(record.get("origin_iata") or ""),
        destination_iata=str(record.get("destination_iata") or ""),
        scheduled_departure=str(record.get("scheduled_departure") or ""),
        scheduled_arrival=str(record.get("scheduled_arrival") or ""),
        cabin_class=str(record.get("cabin_class") or record.get("cabin") or ""),
        price=record.get("price"),
        currency=str(record.get("currency") or ""),
        stops=record.get("stops"),
        duration_minutes=record.get("duration_minutes"),
    )


def normalize_flight_record(record: dict) -> FlightRecord:
    """Normalize a flight-status-like record from FlyClaw or compatible data."""
    return FlightRecord(
        source=str(record.get("source") or ""),
        flight_number=str(record.get("flight_number") or ""),
        airline=str(record.get("airline") or ""),
        origin_iata=str(record.get("origin_iata") or ""),
        destination_iata=str(record.get("destination_iata") or ""),
        scheduled_departure=str(record.get("scheduled_departure") or ""),
        scheduled_arrival=str(record.get("scheduled_arrival") or ""),
        status=str(record.get("status") or ""),
        aircraft_type=str(record.get("aircraft_type") or ""),
    )
