"""Date query tool for grounding relative travel dates."""

import json
from datetime import datetime, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.tools import tool
from pydantic import Field

from src.tools.registry import register_tool

DEFAULT_TIMEZONE = "Asia/Shanghai"


@tool
def query_current_date(
    days_offset: Annotated[
        int,
        Field(
            description=(
                "Relative day offset from current date: 今天/today=0, "
                "明天/tomorrow=1, 后天/day after tomorrow=2, N天后=N."
            )
        ),
    ] = 0,
    timezone_name: Annotated[
        str,
        Field(description="IANA timezone name, default Asia/Shanghai."),
    ] = DEFAULT_TIMEZONE,
) -> str:
    """查询当前日期/时间，并把“今天、明天、后天、N天后”等相对日期落到明确日期。用户查询航班或机票但只给出相对日期时，先用本工具获得 target_date，再继续查询事实。"""
    try:
        timezone = ZoneInfo(timezone_name)
        limitations: list[str] = []
    except ZoneInfoNotFoundError:
        timezone = ZoneInfo(DEFAULT_TIMEZONE)
        limitations = [
            f"Unknown timezone '{timezone_name}', fell back to {DEFAULT_TIMEZONE}."
        ]

    now = datetime.now(timezone)
    target = now.date() + timedelta(days=days_offset)
    payload = {
        "timezone": timezone.key,
        "current_date": now.date().isoformat(),
        "current_datetime": now.isoformat(timespec="seconds"),
        "days_offset": days_offset,
        "target_date": target.isoformat(),
        "limitations": limitations,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


register_tool(query_current_date)
