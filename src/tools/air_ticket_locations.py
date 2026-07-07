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
                "必填。城市名、机场名或 IATA 代码列表。"
                "如果用户说“北京到上海”，填写 ['北京','上海']；"
                "如果用户说“PEK 到 SHA”，填写 ['PEK','SHA']。"
                "不要留空，不要传 null。"
            )
        ),
    ] = None,
) -> str:
    """解析城市/机场/IATA 为机场候选事实。

    使用场景：
    - 用户给出城市名、机场名或 IATA 代码，需要转成标准机场候选。
    - 用户说“北京到上海”“广州飞东京”“PEK 到 SHA”这类航线表达时，先调用本工具解析地点。

    参数填写模板：
    - 北京到上海：{"locations":["北京","上海"]}
    - 北京首都到上海虹桥：{"locations":["北京首都","上海虹桥"]}
    - PEK 到 SHA：{"locations":["PEK","SHA"]}

    参数规则：
    - locations 必须是字符串数组，至少包含 1 个地点，通常航线查询包含 2 个地点。
    - 不要传空对象 {}，不要传 {"locations":null}，不要传 {"locations":[]}。
    - 不要要求用户自己提供 IATA；先用本工具解析。
    """
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
