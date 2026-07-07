"""Flight information lookup tool for air ticket fact retrieval."""

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from src.air_ticket import query_flight
from src.air_ticket.models import to_json_payload
from src.tools.registry import register_tool


@tool
def query_flight_information(
    flight_number: Annotated[
        str,
        Field(
            description=(
                "必填。航班号，例如 CA981、MU5105、CZ3102。"
                "必须从用户问题中提取，不要留空。"
            )
        ),
    ],
    date: Annotated[
        str | None,
        Field(
            description=(
                "可选。查询日期，格式 YYYY-MM-DD。"
                "如果用户说“明天/后天”，先用日期工具得到 target_date 再填写。"
            )
        ),
    ] = None,
    include_price_relay: Annotated[
        bool,
        Field(description="是否附带航线报价参考，默认 true。用户只问执飞动态时也可以保持 true。"),
    ] = True,
    currency: Annotated[
        str,
        Field(description="报价参考币种，默认 cny。"),
    ] = "cny",
) -> str:
    """按航班号查询航班事实，并可附带航线报价参考。

    使用场景：
    - 用户给出具体航班号，如“CA981 这个航班怎么样”“MU5105 明天多少钱”。
    - 问题核心是某个航班号的执飞事实、航线信息或价格参考。

    参数填写模板：
    - 查询 CA981，不指定日期：
      {"flight_number":"CA981","include_price_relay":true,"currency":"cny"}
    - 查询 MU5105 在 2026-07-08：
      {"flight_number":"MU5105","date":"2026-07-08","include_price_relay":true,"currency":"cny"}

    参数规则：
    - flight_number 必须填写，不要传空对象 {}。
    - 如果用户给的是航线而不是航班号，例如“北京到上海”，不要用本工具；应使用地点解析和报价查询工具。
    - 如果用户说“明天”，先调用日期工具得到 target_date，再填入 date。

    回答边界：
    - 本工具返回航班事实和可选报价参考，不输出审计结论或价格合理性裁定。
    """
    response = query_flight(
        flight_number=flight_number,
        date=date,
        include_price_relay=include_price_relay,
        currency=currency,
    )
    return to_json_payload(response)


register_tool(query_flight_information)
