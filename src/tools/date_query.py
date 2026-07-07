"""Date query tool for grounding relative travel dates."""

import json
from datetime import datetime, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.tools import tool
from pydantic import Field

from src.tools.registry import register_tool

DEFAULT_TIMEZONE = "Asia/Shanghai"
QUERY_CURRENT_DATE_DAYS_OFFSET_DESCRIPTION = (
    "相对今天的天数偏移。今天/today=0；明天/tomorrow=1；"
    "后天/day after tomorrow=2；3天后=3；N天后=N。"
    "当用户说“明天”时必须填写 1，不要省略。"
)
QUERY_CURRENT_DATE_TIMEZONE_DESCRIPTION = (
    "IANA 时区名称。中国语境默认 Asia/Shanghai；"
    "除非用户明确指定其他时区，否则填写 Asia/Shanghai。"
)
QUERY_CURRENT_DATE_TOOL_DESCRIPTION = """查询当前日期/时间，并把相对日期落到明确日期。

使用场景：
- 用户说“今天、明天、后天、N天后、下周”等相对日期。
- 用户查询航班/机票但没有给 YYYY-MM-DD，而是说“明天北京到上海”。

参数填写模板：
- 今天：{"days_offset":0,"timezone_name":"Asia/Shanghai"}
- 明天：{"days_offset":1,"timezone_name":"Asia/Shanghai"}
- 后天：{"days_offset":2,"timezone_name":"Asia/Shanghai"}
- 3天后：{"days_offset":3,"timezone_name":"Asia/Shanghai"}

参数规则：
- 用户说“明天”时，days_offset 必须是 1。
- 用户说“后天”时，days_offset 必须是 2。
- 不要传空对象 {} 来表示明天；空对象只表示今天。
- 拿到返回的 target_date 后，再用于机票报价或航班事实查询。
"""


@tool(description=QUERY_CURRENT_DATE_TOOL_DESCRIPTION)
def query_current_date(
    days_offset: Annotated[
        int,
        Field(description=QUERY_CURRENT_DATE_DAYS_OFFSET_DESCRIPTION),
    ] = 0,
    timezone_name: Annotated[
        str,
        Field(description=QUERY_CURRENT_DATE_TIMEZONE_DESCRIPTION),
    ] = DEFAULT_TIMEZONE,
) -> str:
    """Return current or offset date facts."""
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
