from src.tools import get_tools


def test_get_tools_discovers_registered_tools():
    tool_names = {tool.name for tool in get_tools()}

    assert tool_names == {
        "resolve_flight_locations",
        "search_airfare_quotes",
        "query_flight_information",
    }


def test_get_tools_returns_copy():
    tools = get_tools()
    tools.clear()

    assert {tool.name for tool in get_tools()} == {
        "resolve_flight_locations",
        "search_airfare_quotes",
        "query_flight_information",
    }
