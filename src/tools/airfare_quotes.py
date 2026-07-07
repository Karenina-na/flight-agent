"""Airfare quote search tool for air ticket fact retrieval."""

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from src.air_ticket import search_quotes
from src.tools.registry import register_tool


@tool
def search_airfare_quotes(
    origin: Annotated[
        str,
        Field(
            description=(
                "必填。出发地城市、机场名或 IATA 代码。"
                "例如北京、北京首都、PEK、PKX。不要留空。"
            )
        ),
    ],
    destination: Annotated[
        str,
        Field(
            description=(
                "必填。目的地城市、机场名或 IATA 代码。"
                "例如上海、上海虹桥、SHA、PVG。不要留空。"
            )
        ),
    ],
    departure_date: Annotated[
        str,
        Field(
            description=(
                "必填。出发日期，格式 YYYY-MM-DD。"
                "如果用户说“明天”，必须先用日期工具得到 target_date，"
                "再把 target_date 填到这里。"
            )
        ),
    ],
    return_date: Annotated[
        str | None,
        Field(description="可选。返程日期，格式 YYYY-MM-DD；单程查询填写 null 或省略。"),
    ] = None,
    cabin: Annotated[
        str,
        Field(
            description=(
                "舱位。默认 economy。可用值通常为 economy、premium_economy、business、first。"
            )
        ),
    ] = "economy",
    adults: Annotated[int, Field(description="成人乘客数量，默认 1。")] = 1,
    children: Annotated[int, Field(description="儿童乘客数量，默认 0。")] = 0,
    infants: Annotated[int, Field(description="婴儿乘客数量，默认 0。")] = 0,
    stops: Annotated[
        int | str,
        Field(description="经停偏好。默认 0 表示优先直飞；也可传 any 表示不限。"),
    ] = 0,
    currency: Annotated[
        str,
        Field(description="报价币种，默认 cny。中国用户通常填写 cny。"),
    ] = "cny",
    limit: Annotated[int, Field(description="最多返回报价条数，默认 20。")] = 20,
) -> str:
    """查询某航线某日期的公开机票报价事实，不判断价格是否合理。

    使用场景：
    - 用户问“北京到上海明天机票多少钱”“广州到东京 2026-07-10 价格范围”。
    - 已经能确定 origin、destination、departure_date 时调用本工具。

    参数填写模板：
    - 北京到上海，2026-07-08，单人经济舱：
      {"origin":"北京","destination":"上海","departure_date":"2026-07-08","cabin":"economy","adults":1,"currency":"cny","limit":20}
    - PEK 到 SHA，2026-07-08：
      {"origin":"PEK","destination":"SHA","departure_date":"2026-07-08","cabin":"economy","adults":1,"currency":"cny"}

    调用前规则：
    - 如果用户使用“明天/后天”等相对日期，先调用日期工具得到 target_date。
    - 如果用户地点是城市名，可先调用地点解析工具获得候选机场；也可以用城市名直接查询报价样本。
    - origin、destination、departure_date 三个参数必须填写，不要传空对象 {}。

    回答边界：
    - 本工具只返回报价样本事实，不代表历史出票价，不输出“合理/异常/违规/审计通过”等判断。
    """
    response = search_quotes(
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        cabin=cabin,
        adults=adults,
        children=children,
        infants=infants,
        stops=stops,
        currency=currency,
        limit=limit,
    )
    return response.to_json()


register_tool(search_airfare_quotes)
