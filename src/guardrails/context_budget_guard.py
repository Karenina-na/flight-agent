"""Guard model calls from continuing ReAct loops near the context limit."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from hashlib import sha256
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import HumanMessage, SystemMessage

from src.observability import log_event
from src.observability.model_trace import model_request_trace_chars
from src.runtime import Context


DEFAULT_MAX_FRACTION = 0.85
DEFAULT_CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOOL_FACTS = 6


class ContextBudgetGuard(AgentMiddleware):
    """Compact large ReAct contexts into a final-answer request."""

    tools: list[Any] = []

    def __init__(
        self,
        *,
        context_window_tokens: int,
        max_fraction: float = DEFAULT_MAX_FRACTION,
        chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
        max_tool_facts: int = DEFAULT_MAX_TOOL_FACTS,
    ) -> None:
        self.context_window_tokens = context_window_tokens
        self.max_fraction = max_fraction
        self.chars_per_token = chars_per_token
        self.max_tool_facts = max_tool_facts

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Compact oversized ReAct requests before synchronous model calls."""
        return handler(self._guarded_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Compact oversized ReAct requests before asynchronous model calls."""
        return await handler(self._guarded_request(request))

    def _guarded_request(self, request: ModelRequest) -> ModelRequest:
        estimate = _request_size_estimate(request)
        threshold = self.context_window_tokens * self.chars_per_token * self.max_fraction
        tool_facts = _tool_fact_summaries(request.messages, max_items=self.max_tool_facts)
        if estimate <= threshold or not tool_facts:
            return request

        compact_request = request.override(
            system_message=SystemMessage(content=_final_answer_system_prompt()),
            messages=[
                HumanMessage(
                    content=_final_answer_prompt(
                        original_user_message=_latest_human_text(request.messages),
                        tool_facts=tool_facts,
                        estimate_chars=estimate,
                        threshold_chars=round(threshold),
                    )
                )
            ],
            tools=[],
            tool_choice="none",
        )
        _log_context_budget_compacted(
            request,
            estimate_chars=estimate,
            threshold_chars=round(threshold),
            tool_fact_count=len(tool_facts),
        )
        return compact_request


def build_context_budget_guard(
    *,
    context_window_tokens: int,
    max_fraction: float = DEFAULT_MAX_FRACTION,
) -> ContextBudgetGuard:
    """Build context budget guard middleware."""
    return ContextBudgetGuard(
        context_window_tokens=context_window_tokens,
        max_fraction=max_fraction,
    )


def _request_size_estimate(request: ModelRequest) -> int:
    return model_request_trace_chars(request)


def _tool_fact_summaries(messages: list[Any], *, max_items: int) -> list[str]:
    facts = [_tool_fact_summary(message) for message in messages]
    facts = [fact for fact in facts if fact]
    if max_items <= 0:
        return facts
    return facts[-max_items:]


def _tool_fact_summary(message: Any) -> str:
    if str(getattr(message, "type", "")) != "tool":
        return ""

    tool_name = str(getattr(message, "name", "") or "tool")
    content = getattr(message, "content", "")
    data = _parse_json_object(content)
    if not isinstance(data, dict):
        return f"- {tool_name}: {_truncate(str(content), 500)}"

    query = data.get("query") if isinstance(data.get("query"), dict) else {}
    quotes = data.get("quotes") if isinstance(data.get("quotes"), list) else []
    prices = [
        quote.get("price")
        for quote in quotes
        if isinstance(quote, dict) and isinstance(quote.get("price"), int | float)
    ]
    parts = [f"- {tool_name}"]
    for key in ("origin", "destination", "departure_date", "return_date", "cabin", "currency"):
        value = query.get(key)
        if value:
            parts.append(f"{key}={value}")
    if prices:
        parts.append(f"quote_count={len(quotes)}")
        parts.append(f"min_price={min(prices):g}")
        parts.append(f"max_price={max(prices):g}")
        first_currency = next(
            (
                quote.get("currency")
                for quote in quotes
                if isinstance(quote, dict) and quote.get("currency")
            ),
            None,
        )
        if first_currency:
            parts.append(f"currency={first_currency}")
    elif "target_date" in data:
        parts.append(f"target_date={data.get('target_date')}")
    captured_at = data.get("captured_at") or data.get("current_datetime")
    if captured_at:
        parts.append(f"captured_at={captured_at}")
    sources = data.get("sources_used")
    if isinstance(sources, list) and sources:
        parts.append(f"sources={','.join(str(source) for source in sources)}")
    limitations = data.get("limitations")
    if isinstance(limitations, list) and limitations:
        parts.append("limitations=" + "; ".join(str(item) for item in limitations[:3]))
    return " | ".join(parts)


def _parse_json_object(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _latest_human_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if str(getattr(message, "type", "")) == "human":
            content = getattr(message, "content", "")
            return str(content)
    return ""


def _final_answer_system_prompt() -> str:
    return (
        "你是机票价格与航班事实说明助手。当前上下文已经接近模型窗口上限，"
        "必须生成面向用户的最终回答。不要再调用工具，不要输出工具调用格式，"
        "只基于已提供的工具事实进行归纳。说明已查询到的事实、来源、查询时间、"
        "限制和缺失信息；不要做审计、违规或报销通过/驳回判断。"
    )


def _final_answer_prompt(
    *,
    original_user_message: str,
    tool_facts: list[str],
    estimate_chars: int,
    threshold_chars: int,
) -> str:
    facts = "\n".join(tool_facts)
    return (
        "不要再调用工具。请基于以下已有工具结果，直接生成面向用户的最终回答。\n\n"
        f"用户最新问题：{original_user_message}\n\n"
        "已获取的工具事实摘要：\n"
        f"{facts}\n\n"
        "回答要求：\n"
        "- 用中文回答。\n"
        "- 汇总已查询日期的票价样本，能给价格区间就给区间。\n"
        "- 如果用户要求更长日期范围（例如未来 10 天、后一个月或每天汇总）但没有全部查完，明确说明只覆盖已查询日期，不要编造未查询日期。\n"
        "- 建议用户新开会话、缩小日期范围，或分批查询后再生成完整表格。\n"
        "- 可以给出基于样本的非强制性出行建议，但必须说明数据限制。\n"
        "- 输出普通 Markdown 正文，不要输出 function/tool/XML/JSON 调用格式。\n\n"
        f"上下文预算提示：原请求估算 {estimate_chars} chars，阈值 {threshold_chars} chars。"
    )


def _log_context_budget_compacted(
    request: ModelRequest,
    *,
    estimate_chars: int,
    threshold_chars: int,
    tool_fact_count: int,
) -> None:
    log_event(
        "react_context_budget_compacted",
        context=_request_context(request),
        redact=False,
        estimate_chars=estimate_chars,
        threshold_chars=threshold_chars,
        tool_fact_count=tool_fact_count,
        original_message_count=len(request.messages),
        original_tool_count=len(request.tools),
        compacted_prompt_sha256=sha256(
            _latest_human_text(request.messages).encode("utf-8")
        ).hexdigest(),
    )


def _request_context(request: ModelRequest) -> Context | None:
    runtime = getattr(request, "runtime", None)
    context = getattr(runtime, "context", None)
    return context if isinstance(context, Context) else None


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."
