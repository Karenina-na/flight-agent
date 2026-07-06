from src.prompt import (
    CORE_PROMPT,
    DOMAIN_PROMPT,
    build_system_prompt,
    build_tool_prompt,
)
from src.prompt.build import build_system_prompt as build_system_prompt_from_module
from src.prompt.capabilities import build_tool_prompt as build_tool_prompt_from_module
from src.tools import get_tools


def test_base_prompt_layers_do_not_name_concrete_tools():
    assert "resolve_flight_locations" not in DOMAIN_PROMPT
    assert "search_airfare_quotes" not in DOMAIN_PROMPT
    assert "query_flight_information" not in DOMAIN_PROMPT


def test_domain_prompt_positions_agent_for_air_ticket_fact_explanations():
    assert "机票价格与航班事实说明助手" in DOMAIN_PROMPT
    assert "不要直接做审计" in DOMAIN_PROMPT
    assert "实时报价" in DOMAIN_PROMPT
    assert "历史出票价格" in DOMAIN_PROMPT


def test_tool_layer_is_generated_from_registered_tools():
    tool_prompt = build_tool_prompt(get_tools())

    assert "resolve_flight_locations" in tool_prompt
    assert "search_airfare_quotes" in tool_prompt
    assert "query_flight_information" in tool_prompt
    assert "Resolve city, airport, or IATA inputs" in tool_prompt
    assert "Search airfare quote facts" in tool_prompt
    assert "Query flight information facts" in tool_prompt


def test_prompt_package_exposes_build_modules():
    assert build_system_prompt_from_module is build_system_prompt
    assert build_tool_prompt_from_module is build_tool_prompt


def test_system_prompt_combines_layers():
    system_prompt = build_system_prompt(tools=get_tools())

    assert CORE_PROMPT in system_prompt
    assert DOMAIN_PROMPT in system_prompt
    assert "resolve_flight_locations" in system_prompt
    assert "search_airfare_quotes" in system_prompt
    assert "query_flight_information" in system_prompt
