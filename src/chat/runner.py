"""Agent invocation runner shared by the web UI."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from langchain.messages import HumanMessage

from src.agent import agent as default_agent
from src.chat.session import ChatSession
from src.chat.trace import (
    build_trace_tree,
    has_reasoning_block,
    json_safe,
    latest_assistant_message,
    merge_trace_events_into_turns,
    message_text,
    reasoning_text,
    serialize_agent_input,
    serialize_invoke_output,
    trace_dump_path,
    utc_timestamp,
    write_conversation_trace_dump,
)
from src.observability import (
    collect_trace_events,
    full_text_trace_fields,
    log_event,
    observe_agent_run,
)


EMPTY_VISIBLE_OUTPUT_FALLBACK = (
    "模型返回了工具调用格式文本，但未生成可展示回答。"
    "本轮工具调用和原始模型响应已经写入 Trace；请查看调试页中的工具结果，"
    "或重新提问让模型生成面向用户的文字说明。"
)


@dataclass(frozen=True)
class ChatTurnResult:
    """Rendered result for one browser chat turn."""

    thread_id: str
    answer: str
    status: str
    trace_path: Path
    reasoning: str = ""
    reasoning_block_seen: bool = False
    reasoning_text_seen: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    execution_steps: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly response payload for the web API."""
        payload = {
            "thread_id": self.thread_id,
            "answer": self.answer,
            "status": self.status,
            "trace_path": str(self.trace_path),
            "reasoning": self.reasoning,
            "reasoning_block_seen": self.reasoning_block_seen,
            "reasoning_text_seen": self.reasoning_text_seen,
            "tool_calls": self.tool_calls,
            "execution_steps": self.execution_steps,
            "trace": self.trace,
        }
        if self.error_type:
            payload["error_type"] = self.error_type
        return payload


def run_agent_turn(
    message: str,
    session: ChatSession,
    *,
    agent_instance: Any = default_agent,
    trace_dir: Path | None = None,
    entrypoint: str = "web-ui.chat",
) -> ChatTurnResult:
    """Run one user message through the agent and persist a full trace dump."""
    with collect_trace_events(trace_id=session.thread_id) as trace_events:
        return _run_agent_turn_with_trace(
            message,
            session,
            trace_events,
            agent_instance=agent_instance,
            trace_dir=trace_dir,
            entrypoint=entrypoint,
        )


def _run_agent_turn_with_trace(
    message: str,
    session: ChatSession,
    trace_events: list[dict[str, Any]],
    *,
    agent_instance: Any,
    trace_dir: Path | None,
    entrypoint: str,
) -> ChatTurnResult:
    context = session.context(current_user_input=message)
    agent_input = {"messages": [HumanMessage(content=message)]}
    agent_config = session.config
    assistant_parts: list[str] = []
    assistant_chunk_count = 0
    saw_reasoning = False
    saw_reasoning_block = False
    answer_started = False
    turn_trace: dict[str, Any] = {
        "turn_index": len(session.turns),
        "status": "running",
        "started_at": utc_timestamp(),
        "turn_id": context.request_id,
        "context": context.__dict__,
        "agent_input": serialize_agent_input(agent_input),
        "agent_config": agent_config,
        "user_input": message,
        **full_text_trace_fields("user_input", message),
        "assistant_chunks": [],
        "reasoning_chunks": [],
        "stream_chunks": [],
        "invoke_output": None,
    }
    session.live_turn = turn_trace
    session.live_events = trace_events

    log_event(
        "conversation_turn_start",
        context=context,
        redact=False,
        entrypoint=entrypoint,
        **full_text_trace_fields("user_input", message),
    )

    try:
        with observe_agent_run(
            context,
            entrypoint=entrypoint,
            stream_mode="invoke",
        ):
            invoke_output = agent_instance.invoke(
                agent_input,
                config=agent_config,
                context=context,
            )

        turn_trace["invoke_output"] = serialize_invoke_output(invoke_output)
        assistant_message = latest_assistant_message(invoke_output)
        if assistant_message is not None:
            saw_reasoning_block = has_reasoning_block(assistant_message)
            reasoning = reasoning_text(assistant_message)
            if reasoning:
                turn_trace["reasoning_chunks"].append(reasoning)
                saw_reasoning = True

            content = message_text(assistant_message)
            if content:
                assistant_parts.append(content)
                turn_trace["assistant_chunks"].append(content)
                assistant_chunk_count = 1
                answer_started = True
            else:
                malformed_tool_call_text_seen = _has_malformed_tool_call_text(
                    assistant_message
                )
                current_calls = merge_trace_events_into_turns(
                    [turn_trace],
                    trace_events,
                )[0].get("calls", [])
                tool_fallback = fallback_answer_from_tool_results(current_calls)
                tool_loop_stop_requested = _has_tool_loop_stop_requested(current_calls)
                turn_trace["empty_visible_output"] = True
                turn_trace["malformed_tool_call_text_seen"] = (
                    malformed_tool_call_text_seen
                )
                turn_trace["limitations"] = [
                    (
                        "The final assistant message did not include visible "
                        "text content to render."
                    )
                ]
                if malformed_tool_call_text_seen:
                    turn_trace["limitations"].append(
                        "The final assistant message contained tool-call-like "
                        "text outside the structured tool_calls protocol."
                    )
                visible_fallback = _visible_fallback_for_empty_output(
                    tool_fallback=tool_fallback,
                    tool_loop_stop_requested=tool_loop_stop_requested,
                )
                if tool_fallback:
                    turn_trace["limitations"].append(
                        "A deterministic fallback answer was generated from "
                        "successful current-turn tool results."
                    )
                    turn_trace["tool_result_fallback_used"] = True
                if tool_loop_stop_requested:
                    turn_trace["limitations"].append(
                        "A repeated tool-call stop signal was observed; the runner "
                        "generated a visible fallback instead of leaving the turn empty."
                    )
                    turn_trace["tool_loop_stop_fallback_used"] = True
                assistant_parts.append(visible_fallback)
                turn_trace["assistant_chunks"].append(visible_fallback)
                assistant_chunk_count = 1
                answer_started = True
        else:
            turn_trace["limitations"] = [
                "agent.invoke returned no assistant message to render."
            ]
            assistant_parts.append("未获取到可展示的助手回复。")
            answer_started = True
    except Exception as exc:
        return _finish_error_turn(
            exc=exc,
            session=session,
            turn_trace=turn_trace,
            trace_events=trace_events,
            context=context,
            assistant_parts=assistant_parts,
            assistant_chunk_count=assistant_chunk_count,
            saw_reasoning=saw_reasoning,
            saw_reasoning_block=saw_reasoning_block,
            trace_dir=trace_dir,
            entrypoint=entrypoint,
        )

    assistant_output = "".join(assistant_parts)
    turn_trace.update(
        {
            "status": "success",
            "ended_at": utc_timestamp(),
            "assistant_chunk_count": assistant_chunk_count,
            "reasoning_block_seen": saw_reasoning_block,
            "reasoning_text_seen": saw_reasoning,
            "answer_started": answer_started,
            "assistant_output": assistant_output,
            **full_text_trace_fields("assistant_output", assistant_output),
        }
    )
    session.turns.append(turn_trace)
    trace_path = trace_dump_path(session.thread_id, trace_dir=trace_dir)
    log_event(
        "conversation_turn_end",
        context=context,
        redact=False,
        entrypoint=entrypoint,
        assistant_chunk_count=assistant_chunk_count,
        reasoning_block_seen=saw_reasoning_block,
        reasoning_text_seen=saw_reasoning,
        answer_started=answer_started,
        trace_dump_path=str(trace_path),
        **full_text_trace_fields("assistant_output", assistant_output),
    )
    session.events.extend(trace_events)
    session.live_turn = None
    session.live_events = []
    trace_path = write_conversation_trace_dump(
        thread_id=session.thread_id,
        turns=session.turns,
        events=session.events,
        trace_dir=trace_dir,
    )
    merged_turn = merge_trace_events_into_turns([turn_trace], trace_events)[0]
    return ChatTurnResult(
        thread_id=session.thread_id,
        answer=assistant_output,
        status="success",
        trace_path=trace_path,
        reasoning="".join(turn_trace["reasoning_chunks"]),
        reasoning_block_seen=saw_reasoning_block,
        reasoning_text_seen=saw_reasoning,
        tool_calls=tool_call_summaries(merged_turn.get("calls", [])),
        execution_steps=execution_step_summaries(merged_turn.get("calls", [])),
        trace=conversation_trace_payload(session),
    )


def _has_malformed_tool_call_text(message: object) -> bool:
    """Return True when raw model content looks like an unparsed tool call."""
    raw_text = str(json_safe(getattr(message, "content", "")))
    markers = ("<function=", "</function>", "<tool_call", "</tool_call>")
    return any(marker in raw_text for marker in markers)


def fallback_answer_from_tool_results(calls: list[dict[str, Any]]) -> str | None:
    """Build a deterministic answer from successful current-turn tool results."""
    for call in reversed(calls):
        if call.get("type") != "tool" or call.get("event") != "tool_call_end":
            continue
        if call.get("tool_name") == "search_airfare_quotes":
            answer = _airfare_quotes_fallback(call)
            if answer:
                return answer
    return None


def _visible_fallback_for_empty_output(
    *,
    tool_fallback: str | None,
    tool_loop_stop_requested: bool,
) -> str:
    if tool_loop_stop_requested and tool_fallback:
        return (
            "模型检测到重复工具调用已被系统拦截，我先基于已经成功返回的工具结果给你摘要：\n\n"
            f"{tool_fallback}"
        )
    if tool_loop_stop_requested:
        return (
            "模型检测到重复工具调用已被系统拦截，但本轮没有可用于生成摘要的成功工具结果。"
            "请查看 Trace 中最近的工具结果，或缩小问题后重新发起查询。"
        )
    return tool_fallback or EMPTY_VISIBLE_OUTPUT_FALLBACK


def _has_tool_loop_stop_requested(calls: list[dict[str, Any]]) -> bool:
    for call in calls:
        event = call.get("event")
        if event == "react_duplicate_tool_call_blocked":
            fields = call.get("fields")
            if isinstance(fields, dict) and fields.get("stop_requested") is True:
                return True
        if call.get("type") != "tool" or event != "tool_call_end":
            continue
        response = call.get("response")
        if not isinstance(response, dict):
            continue
        raw_content = response.get("content")
        if not isinstance(raw_content, str):
            continue
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("stop_requested") is True:
            return True
        if payload.get("status") == "react_loop_stop_requested":
            return True
    return False


def _airfare_quotes_fallback(call: dict[str, Any]) -> str | None:
    raw_content = _tool_response_content(call)
    if not isinstance(raw_content, str):
        return None
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        return None

    if _is_tool_guard_response(payload):
        return None

    query = payload.get("query") or {}
    quotes = payload.get("quotes") or []
    sources = payload.get("sources_used") or []
    limitations = payload.get("limitations") or []
    origin = _string_or_joined(query.get("origin")) or "出发地"
    destination = _string_or_joined(query.get("destination")) or "目的地"
    departure_date = str(query.get("departure_date") or "未知日期")
    captured_at = str(payload.get("captured_at") or "未知")

    lines = [
        "模型最终回答生成失败，我先根据已经成功返回的工具结果给你一个兜底摘要：",
        "",
        f"**查询条件：** {origin} → {destination}，出发日期 {departure_date}",
        f"**查询时间：** {captured_at}",
        f"**数据来源：** {', '.join(str(source) for source in sources) if sources else '未返回来源'}",
        "",
    ]
    if quotes:
        lines.append("**报价样本：**")
        for quote in quotes[:10]:
            if not isinstance(quote, dict):
                continue
            flight = str(quote.get("flight_number") or "未知航班")
            airline = str(quote.get("airline") or "未知航司")
            price = quote.get("price")
            currency = str(quote.get("currency") or "")
            origin_iata = str(quote.get("origin_iata") or "")
            destination_iata = str(quote.get("destination_iata") or "")
            route = (
                f"{origin_iata}->{destination_iata}"
                if origin_iata or destination_iata
                else "未知航线"
            )
            departure = str(quote.get("scheduled_departure") or "未知起飞时间")
            lines.append(
                f"- {flight}（{airline}）：{price} {currency}，{route}，起飞 {departure}"
            )
    else:
        lines.append("**报价样本：** 当前工具结果未返回可见报价。")

    if limitations:
        lines.extend(["", "**限制：**"])
        lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines)


def _tool_response_content(call: dict[str, Any]) -> str | None:
    response = call.get("response")
    if isinstance(response, dict) and isinstance(response.get("content"), str):
        return response["content"]

    fields = call.get("fields")
    if not isinstance(fields, dict):
        return None
    response_trace = fields.get("response_trace")
    if isinstance(response_trace, dict) and isinstance(response_trace.get("content"), str):
        return response_trace["content"]
    return None


def _is_tool_guard_response(payload: dict[str, Any]) -> bool:
    status = payload.get("status")
    if status in {"duplicate_blocked", "react_loop_stop_requested"}:
        return True
    return payload.get("stop_requested") is True


def _string_or_joined(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "/".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _finish_error_turn(
    *,
    exc: Exception,
    session: ChatSession,
    turn_trace: dict[str, Any],
    trace_events: list[dict[str, Any]],
    context: Any,
    assistant_parts: list[str],
    assistant_chunk_count: int,
    saw_reasoning: bool,
    saw_reasoning_block: bool,
    trace_dir: Path | None,
    entrypoint: str,
) -> ChatTurnResult:
    partial_output = "".join(assistant_parts)
    user_answer = f"调用失败：{type(exc).__name__}: {exc}"
    turn_trace.update(
        {
            "status": "error",
            "ended_at": utc_timestamp(),
            "assistant_chunk_count": assistant_chunk_count,
            "reasoning_block_seen": saw_reasoning_block,
            "reasoning_text_seen": saw_reasoning,
            "error_type": type(exc).__name__,
            "partial_assistant_output": partial_output,
            "assistant_output": user_answer,
            **full_text_trace_fields(
                "partial_assistant_output",
                partial_output,
            ),
        }
    )
    session.turns.append(turn_trace)
    trace_path = trace_dump_path(session.thread_id, trace_dir=trace_dir)
    log_event(
        "conversation_turn_error",
        context=context,
        redact=False,
        entrypoint=entrypoint,
        assistant_chunk_count=assistant_chunk_count,
        reasoning_block_seen=saw_reasoning_block,
        reasoning_text_seen=saw_reasoning,
        error_type=type(exc).__name__,
        trace_dump_path=str(trace_path),
        **full_text_trace_fields("partial_assistant_output", partial_output),
    )
    session.events.extend(trace_events)
    session.live_turn = None
    session.live_events = []
    trace_path = write_conversation_trace_dump(
        thread_id=session.thread_id,
        turns=session.turns,
        events=session.events,
        trace_dir=trace_dir,
    )
    merged_turn = merge_trace_events_into_turns([turn_trace], trace_events)[0]
    return ChatTurnResult(
        thread_id=session.thread_id,
        answer=user_answer,
        status="error",
        trace_path=trace_path,
        reasoning="".join(turn_trace["reasoning_chunks"]),
        reasoning_block_seen=saw_reasoning_block,
        reasoning_text_seen=saw_reasoning,
        tool_calls=tool_call_summaries(merged_turn.get("calls", [])),
        execution_steps=execution_step_summaries(merged_turn.get("calls", [])),
        trace=conversation_trace_payload(session),
        error_type=type(exc).__name__,
    )


def conversation_trace_payload(session: ChatSession) -> dict[str, Any]:
    """Return the full in-memory multi-turn trace for the Web UI."""
    turns = list(session.turns)
    events = list(session.events)
    if session.live_turn is not None:
        turns.append(session.live_turn)
        events.extend(session.live_events)
    merged_turns = merge_trace_events_into_turns(turns, events)
    payload = {
        "thread_id": session.thread_id,
        "turn_count": len(merged_turns),
        "event_count": len(events),
        "turns": json_safe(merged_turns),
        "events": json_safe(events),
    }
    payload["tree"] = build_trace_tree(payload)
    return payload


def debug_summary_payload(
    trace: dict[str, Any],
    *,
    model_name: str,
    context_window_tokens: int,
) -> dict[str, Any]:
    """Return compact debug metrics for the browser side panel."""
    turns = trace.get("turns") if isinstance(trace.get("turns"), list) else []
    last_turn = turns[-1] if turns and isinstance(turns[-1], dict) else {}
    calls = last_turn.get("calls") if isinstance(last_turn.get("calls"), list) else []

    model_starts = [
        call
        for call in calls
        if call.get("type") == "model" and call.get("event") == "model_call_start"
    ]
    model_ends = [
        call
        for call in calls
        if call.get("type") == "model" and call.get("event") == "model_call_end"
    ]
    tool_starts = [
        call
        for call in calls
        if call.get("type") == "tool" and call.get("event") == "tool_call_start"
    ]
    tool_ends = [
        call
        for call in calls
        if call.get("type") == "tool" and call.get("event") == "tool_call_end"
    ]
    tool_errors = [
        call
        for call in calls
        if call.get("type") == "tool" and call.get("event") == "tool_call_error"
    ]

    last_model_start = model_starts[-1] if model_starts else {}
    request = last_model_start.get("request")
    request_messages = []
    if isinstance(request, dict) and isinstance(request.get("messages"), list):
        request_messages = request["messages"]

    prompt_char_counts = [
        len(_compact_json(call.get("request"))) for call in model_starts
    ]
    estimated_prompt_chars = (
        len(_compact_json(last_model_start.get("request")))
        if last_model_start
        else 0
    )
    max_prompt_chars = max(prompt_char_counts, default=0)
    total_react_prompt_chars = sum(prompt_char_counts)
    estimated_response_chars = sum(
        len(_compact_json(call.get("response"))) for call in model_ends
    )
    context_usage_estimate = (
        round(estimated_prompt_chars / max(context_window_tokens * 4, 1), 4)
        if context_window_tokens
        else None
    )

    sources = _source_summary(tool_ends)
    warnings = _debug_warnings(last_turn, tool_errors)

    return {
        "session": {
            "thread_id": trace.get("thread_id"),
            "turn_count": trace.get("turn_count", len(turns)),
            "event_count": trace.get("event_count", len(trace.get("events") or [])),
            "last_status": last_turn.get("status", "ready"),
        },
        "model": {
            "model_name": model_name,
            "context_window_tokens": context_window_tokens,
            "last_message_count": len(request_messages),
            "estimated_prompt_chars": estimated_prompt_chars,
            "max_prompt_chars": max_prompt_chars,
            "total_react_prompt_chars": total_react_prompt_chars,
            "estimated_response_chars": estimated_response_chars,
            "context_usage_estimate": context_usage_estimate,
        },
        "execution": {
            "model_call_count": len(model_starts),
            "tool_call_count": len(tool_starts),
            "tool_success_count": len(tool_ends),
            "tool_error_count": len(tool_errors),
            "recent_tools": _unique_recent_tools(tool_starts),
        },
        "sources": sources,
        "warnings": warnings,
    }


def _compact_json(value: object) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"))


def _unique_recent_tools(tool_starts: list[dict[str, Any]]) -> list[str]:
    tools: list[str] = []
    seen: set[str] = set()
    for call in tool_starts:
        tool_name = call.get("tool_name")
        if not tool_name:
            request = call.get("request")
            if isinstance(request, dict):
                tool_name = request.get("name")
        if not tool_name:
            continue
        name = str(tool_name)
        if name in seen:
            continue
        seen.add(name)
        tools.append(name)
    return tools[-6:]


def _source_summary(tool_ends: list[dict[str, Any]]) -> dict[str, Any]:
    sources_used: list[str] = []
    limitations: list[str] = []
    fact_counts: dict[str, int] = {}
    captured_at: str | None = None

    for call in tool_ends:
        payload = _tool_response_payload(call)
        if not isinstance(payload, dict):
            continue
        captured_at = captured_at or _optional_str(payload.get("captured_at"))
        _extend_unique(sources_used, payload.get("sources_used"))
        _extend_unique(limitations, payload.get("limitations"))
        for key in ("items", "quotes", "flight_records", "relay_quotes"):
            value = payload.get(key)
            if isinstance(value, list):
                fact_counts[key] = fact_counts.get(key, 0) + len(value)

    return {
        "sources_used": sources_used,
        "captured_at": captured_at,
        "fact_counts": fact_counts,
        "limitations": limitations,
    }


def _tool_response_payload(call: dict[str, Any]) -> object | None:
    response = call.get("response")
    if not isinstance(response, dict):
        return None
    content = response.get("content")
    if isinstance(content, (dict, list)):
        return content
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _extend_unique(target: list[str], value: object) -> None:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = [item for item in value if isinstance(item, str)]
    else:
        return
    for item in values:
        if item not in target:
            target.append(item)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _debug_warnings(
    last_turn: dict[str, Any],
    tool_errors: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if last_turn.get("empty_visible_output"):
        warnings.append("模型未生成可展示文本。")
    if last_turn.get("malformed_tool_call_text_seen"):
        warnings.append("模型返回了伪工具调用文本。")
    if tool_errors:
        warnings.append(f"{len(tool_errors)} 次工具调用失败。")
    return warnings


def tool_call_summaries(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact tool-call rows for the debug panel."""
    summaries_by_id: dict[str, dict[str, Any]] = {}
    anonymous_summaries: list[dict[str, Any]] = []
    for call in calls:
        if call.get("type") != "tool":
            continue

        tool_call_id = call.get("tool_call_id")
        if not tool_call_id:
            anonymous_summaries.append(_single_tool_call_summary(call))
            continue

        key = str(tool_call_id)
        summary = summaries_by_id.setdefault(
            key,
            {
                "index": call.get("index"),
                "tool_name": call.get("tool_name"),
                "tool_call_id": tool_call_id,
                "status": "started",
            },
        )
        summary["index"] = min(
            int(summary["index"] or call.get("index") or 0),
            int(call.get("index") or summary["index"] or 0),
        )
        summary["tool_name"] = summary.get("tool_name") or call.get("tool_name")

        if "request" in call:
            summary["request"] = call["request"]
        if "response" in call:
            summary["response"] = call["response"]
        if call.get("event") == "tool_call_end":
            summary["status"] = "completed"
        elif call.get("event") == "tool_call_error":
            summary["status"] = "error"

    summaries = list(summaries_by_id.values()) + anonymous_summaries
    return sorted(summaries, key=lambda item: int(item.get("index") or 0))


def execution_step_summaries(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact ReAct steps for the main chat UI."""
    steps: list[dict[str, Any]] = []
    index = 0
    while index < len(calls):
        call = calls[index]
        call_type = call.get("type")
        event = call.get("event")
        if event == "react_context_budget_compacted":
            steps.append(_context_compaction_step(call))
            index += 1
            continue
        if event == "context_summary_start":
            summary_calls = [call]
            if index + 1 < len(calls):
                next_call = calls[index + 1]
                if next_call.get("event") in {
                    "context_summary_end",
                    "context_summary_error",
                }:
                    summary_calls.append(next_call)
                    index += 1
            steps.append(_context_summary_step(summary_calls))
            index += 1
            continue
        if event in {"context_summary_end", "context_summary_error"}:
            steps.append(_context_summary_step([call]))
            index += 1
            continue
        if call_type == "model" and event == "model_call_start":
            model_calls = [call]
            next_index = index + 1
            if next_index < len(calls):
                next_call = calls[next_index]
                if (
                    next_call.get("type") == "model"
                    and next_call.get("event") == "model_call_end"
                ):
                    model_calls.append(next_call)
                    index = next_index
            tool_groups: list[list[dict[str, Any]]] = []
            while index + 1 < len(calls):
                next_call = calls[index + 1]
                if (
                    next_call.get("type") != "tool"
                    or next_call.get("event") != "tool_call_start"
                ):
                    break
                tool_batch: list[dict[str, Any]] = []
                index += 1
                while index < len(calls):
                    batch_call = calls[index]
                    if batch_call.get("type") != "tool":
                        break
                    tool_batch.append(batch_call)
                    if index + 1 >= len(calls):
                        break
                    following_call = calls[index + 1]
                    if following_call.get("type") != "tool":
                        break
                    index += 1
                tool_groups.extend(_group_tool_call_events(tool_batch))
            steps.append(_react_execution_step(len(steps) + 1, model_calls, tool_groups))
        elif call_type == "tool" and event == "tool_call_start":
            tool_batch = [call]
            while index + 1 < len(calls) and calls[index + 1].get("type") == "tool":
                index += 1
                tool_batch.append(calls[index])
            steps.append(
                _react_execution_step(
                    len(steps) + 1,
                    [],
                    _group_tool_call_events(tool_batch),
                )
            )
        index += 1
    return steps


def _group_tool_call_events(calls: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped_by_id: dict[str, list[dict[str, Any]]] = {}
    anonymous_groups: list[list[dict[str, Any]]] = []
    order: list[str] = []

    for call in calls:
        tool_call_id = call.get("tool_call_id")
        if not tool_call_id:
            anonymous_groups.append([call])
            continue
        key = str(tool_call_id)
        if key not in grouped_by_id:
            grouped_by_id[key] = []
            order.append(key)
        grouped_by_id[key].append(call)

    groups = [grouped_by_id[key] for key in order]
    groups.extend(anonymous_groups)
    return groups


def _react_execution_step(
    step_number: int,
    model_calls: list[dict[str, Any]],
    tool_groups: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    stages = []
    if model_calls:
        stages.append(_thought_execution_stage(model_calls))
    if len(tool_groups) > 1:
        stages.append(_action_batch_execution_stage(tool_groups))
    else:
        stages.extend(_action_execution_stage(group) for group in tool_groups)

    status = _combined_status(stage["status"] for stage in stages)
    event_count = len(model_calls) + sum(len(group) for group in tool_groups)
    first_call = (model_calls or (tool_groups[0] if tool_groups else [{}]))[0]
    tool_count = len(tool_groups)

    return {
        "index": first_call.get("index"),
        "kind": "react_step",
        "title": f"ReAct Step {step_number}",
        "status": status,
        "event_count": event_count,
        "summary": _react_step_summary(model_calls, tool_count),
        "stages": stages,
    }


def _context_compaction_step(call: dict[str, Any]) -> dict[str, Any]:
    fields = call.get("fields") if isinstance(call.get("fields"), dict) else {}
    observation_count = int(fields.get("observation_count") or 0)
    preserved_count = int(fields.get("preserved_observation_count") or 0)
    dropped_count = int(fields.get("dropped_observation_count") or 0)
    compacted_state_details = _context_compaction_state_details(fields)
    return {
        "index": call.get("index"),
        "kind": "context_compaction",
        "title": "上下文状态压缩",
        "status": "completed",
        "event_count": 1,
        "summary": (
            f"上下文超过预算，压缩历史状态并保留 {preserved_count}/{observation_count} "
            f"条工具观察，丢弃 {dropped_count} 条；Agent 可继续调用工具。"
        ),
        "stages": [
            {
                "kind": "context_compaction",
                "title": "状态压缩",
                "status": "completed",
                "summary": (
                    f"原请求估算 {fields.get('estimate_chars')} chars，"
                    f"阈值 {fields.get('threshold_chars')} chars。"
                ),
                "details": {
                    "estimate_chars": fields.get("estimate_chars"),
                    "threshold_chars": fields.get("threshold_chars"),
                    "observation_count": observation_count,
                    "preserved_observation_count": preserved_count,
                    "dropped_observation_count": dropped_count,
                    "preview_truncated_count": fields.get("preview_truncated_count"),
                    "compacted_request_chars": fields.get("compacted_request_chars"),
                    "compacted_state_text_chars": fields.get("compacted_state_text_chars"),
                    "final_model_request_chars": fields.get("final_model_request_chars"),
                    "post_compaction_chars": fields.get("post_compaction_chars"),
                    "still_over_budget": fields.get("still_over_budget"),
                    "compaction_level": fields.get("compaction_level"),
                    "semantic_skip_reason": fields.get("semantic_skip_reason"),
                    "compaction_mode": fields.get("compaction_mode"),
                    "compacted_message_count": fields.get("compacted_message_count"),
                    "compacted_tool_count": fields.get("compacted_tool_count"),
                },
            }
        ]
        + (
            [
                {
                    "kind": "context_compaction_state",
                    "title": "压缩后信息",
                    "status": "completed",
                    "summary": (
                        f"压缩后状态 {compacted_state_details.get('compacted_state_chars')} chars，"
                        f"当前展示前 {compacted_state_details.get('compacted_state_preview_chars')} chars。"
                    ),
                    "details": compacted_state_details,
                }
            ]
            if compacted_state_details
            else []
        ),
        "details": {
            "estimate_chars": fields.get("estimate_chars"),
            "threshold_chars": fields.get("threshold_chars"),
            "observation_count": observation_count,
            "preserved_observation_count": preserved_count,
            "dropped_observation_count": dropped_count,
            "preview_truncated_count": fields.get("preview_truncated_count"),
            "compacted_request_chars": fields.get("compacted_request_chars"),
            "compacted_state_text_chars": fields.get("compacted_state_text_chars"),
            "final_model_request_chars": fields.get("final_model_request_chars"),
            "post_compaction_chars": fields.get("post_compaction_chars"),
            "still_over_budget": fields.get("still_over_budget"),
            "compaction_level": fields.get("compaction_level"),
            "semantic_skip_reason": fields.get("semantic_skip_reason"),
            "compaction_mode": fields.get("compaction_mode"),
            "compacted_message_count": fields.get("compacted_message_count"),
            "compacted_tool_count": fields.get("compacted_tool_count"),
            **compacted_state_details,
        },
    }


def _context_summary_step(calls: list[dict[str, Any]]) -> dict[str, Any]:
    first_call = calls[0] if calls else {}
    last_call = calls[-1] if calls else {}
    start_fields = (
        first_call.get("fields")
        if isinstance(first_call.get("fields"), dict)
        else {}
    )
    end_fields = (
        last_call.get("fields")
        if isinstance(last_call.get("fields"), dict)
        else {}
    )
    details = {**start_fields, **end_fields}
    last_event = str(last_call.get("event") or "")
    if last_event == "context_summary_error":
        status = "error"
    elif last_event == "context_summary_end":
        status = "completed"
    else:
        status = "running"
    stage = str(details.get("stage") or "context_summary")
    title = {
        "l3_tool_semantic": "工具结果语义压缩",
        "local_semantic_summary": "历史上下文摘要",
        "global_fallback_summary": "全局兜底摘要",
    }.get(stage, "上下文摘要")
    if status == "completed":
        summary = f"摘要完成，用时 {details.get('duration_ms', 0)} ms。"
    elif status == "error":
        summary = (
            f"摘要失败：{details.get('error_type') or '未知错误'}；"
            "系统将使用压缩回退结果继续执行。"
        )
    else:
        tool_name = details.get("tool_name")
        summary = (
            f"正在压缩工具 {tool_name} 的返回结果。"
            if tool_name
            else "正在生成上下文摘要。"
        )
    return {
        "index": first_call.get("index"),
        "kind": "context_summary",
        "title": title,
        "status": status,
        "event_count": len(calls),
        "summary": summary,
        "stages": [
            {
                "kind": "context_summary",
                "title": title,
                "status": status,
                "summary": summary,
                "details": details,
            }
        ],
    }


def _context_compaction_state_details(fields: dict[str, Any]) -> dict[str, Any]:
    preview = fields.get("compacted_state_preview")
    if not isinstance(preview, str) or not preview:
        return {}
    return {
        "compacted_state_preview": preview,
        "compacted_state_preview_chars": fields.get("compacted_state_preview_chars"),
        "compacted_state_chars": fields.get("compacted_state_chars"),
        "compacted_state_sha256": fields.get("compacted_state_sha256"),
    }


def _thought_execution_stage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    start = calls[0]
    end = calls[-1] if len(calls) > 1 else {}
    status = "completed" if end.get("event") == "model_call_end" else "started"
    if end.get("event") == "model_call_error":
        status = "error"

    message_count = _request_message_count(start.get("request"))
    tool_count = _request_tool_count(start.get("request"))
    response_block_types = _response_block_types(end.get("response"))
    response_preview = _model_response_preview(end.get("response"))
    requested_tools = _model_requested_tools(end.get("response"))
    response_text = (
        "、".join(_readable_response_block_type(value) for value in response_block_types)
        if response_block_types
        else "未知响应"
    )

    return {
        "kind": "thought",
        "title": "模型响应",
        "status": status,
        "summary": f"模型读取 {message_count} 条上下文消息，响应包含 {response_text}。",
        "details": {
            "message_count": message_count,
            "tool_count": tool_count,
            "response_block_types": response_block_types,
            "response_preview": response_preview,
            "requested_tools": requested_tools,
        },
    }


def _action_execution_stage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    start = calls[0]
    end = calls[-1] if len(calls) > 1 else {}
    tool_name = str(start.get("tool_name") or "tool")
    status = "started"
    if end.get("event") == "tool_call_end":
        status = "completed"
    elif end.get("event") == "tool_call_error":
        status = "error"
    argument_keys = _tool_argument_keys(start.get("request"))
    argument_text = ", ".join(argument_keys) if argument_keys else "无"
    return {
        "kind": "action",
        "title": "工具调用",
        "status": status,
        "summary": f"调用 {tool_name}，参数：{argument_text}。",
        "details": {
            "tool_name": tool_name,
            "tool_call_id": start.get("tool_call_id"),
            "argument_keys": argument_keys,
            "response_preview": _response_preview(end.get("response")),
        },
    }


def _action_batch_execution_stage(tool_groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    tool_items = [_action_execution_stage(group) for group in tool_groups]
    tool_names = [
        str(item["details"].get("tool_name") or "tool")
        for item in tool_items
    ]
    unique_tool_names = sorted(set(tool_names))
    title_tool = unique_tool_names[0] if len(unique_tool_names) == 1 else "多个工具"
    status = _combined_status(item["status"] for item in tool_items)
    return {
        "kind": "action_batch",
        "title": "工具批次",
        "status": status,
        "summary": f"批量调用 {title_tool} × {len(tool_items)}。",
        "details": {
            "tool_count": len(tool_items),
            "tool_names": unique_tool_names,
            "tools": [
                {
                    "tool_name": item["details"].get("tool_name"),
                    "tool_call_id": item["details"].get("tool_call_id"),
                    "argument_keys": item["details"].get("argument_keys"),
                    "response_preview": item["details"].get("response_preview"),
                    "status": item["status"],
                }
                for item in tool_items
            ],
        },
    }


def _combined_status(statuses: object) -> str:
    status_list = list(statuses)
    if "error" in status_list:
        return "error"
    if status_list and all(status == "completed" for status in status_list):
        return "completed"
    if "started" in status_list:
        return "started"
    return "info"


def _react_step_summary(model_calls: list[dict[str, Any]], tool_count: int) -> str:
    if tool_count:
        return f"模型响应中请求调用 {tool_count} 个工具。"
    if model_calls:
        return "模型生成最终回复。"
    return "执行工具调用。"


def _request_message_count(request: object) -> int:
    if not isinstance(request, dict):
        return 0
    messages = request.get("messages")
    return len(messages) if isinstance(messages, list) else 0


def _request_tool_count(request: object) -> int:
    if not isinstance(request, dict):
        return 0
    tools = request.get("tools")
    return len(tools) if isinstance(tools, list) else 0


def _response_block_types(response: object) -> list[str]:
    if not isinstance(response, list):
        return []
    block_types: list[str] = []
    for item in response:
        if not isinstance(item, dict):
            continue
        values = item.get("content_block_types")
        if isinstance(values, list):
            block_types.extend(str(value) for value in values)
            continue
        value = item.get("type") or item.get("role")
        if value:
            block_types.append(str(value))
    return block_types


def _model_response_preview(response: object, limit: int = 180) -> str:
    if not isinstance(response, list):
        return ""

    parts: list[str] = []
    for item in response:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(_text_from_content_blocks(content))

    preview = " ".join(part.strip() for part in parts if part and part.strip())
    return preview if len(preview) <= limit else f"{preview[:limit]}..."


def _text_from_content_blocks(blocks: list[object]) -> list[str]:
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text") or block.get("content")
            if text:
                parts.append(str(text))
    return parts


def _model_requested_tools(response: object) -> list[dict[str, Any]]:
    if not isinstance(response, list):
        return []

    tools: list[dict[str, Any]] = []
    for item in response:
        if not isinstance(item, dict):
            continue
        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list):
            tools.extend(_tool_request_summary(tool_call) for tool_call in tool_calls)
        content = item.get("content")
        if isinstance(content, list):
            tools.extend(
                _content_block_tool_request_summary(block)
                for block in content
                if isinstance(block, dict) and block.get("type") in {"function_call", "tool_call"}
            )

    return _dedupe_tool_requests(
        [tool for tool in tools if tool.get("name") or tool.get("argument_keys")]
    )


def _dedupe_tool_requests(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for tool in tools:
        key = (
            str(tool.get("id") or ""),
            str(tool.get("name") or ""),
            tuple(str(value) for value in tool.get("argument_keys") or []),
        )
        fallback_key = ("", key[1], key[2])
        if key in seen or (key[0] and fallback_key in seen):
            continue
        if not key[0]:
            matching_with_id = any(
                existing_key[1] == key[1]
                and existing_key[2] == key[2]
                and bool(existing_key[0])
                for existing_key in seen
            )
            if matching_with_id:
                continue
        seen.add(key)
        deduped.append(tool)
    return deduped


def _tool_request_summary(tool_call: object) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        return {"name": "", "argument_keys": []}
    args = tool_call.get("args")
    return {
        "name": str(tool_call.get("name") or ""),
        "id": str(tool_call.get("id") or ""),
        "argument_keys": sorted(str(key) for key in args) if isinstance(args, dict) else [],
    }


def _content_block_tool_request_summary(block: dict[str, Any]) -> dict[str, Any]:
    args = block.get("args") or block.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return {
        "name": str(block.get("name") or block.get("function_name") or ""),
        "id": str(block.get("id") or block.get("call_id") or ""),
        "argument_keys": sorted(str(key) for key in args) if isinstance(args, dict) else [],
    }


def _readable_response_block_type(value: object) -> str:
    mapping = {
        "reasoning": "内部推理标记",
        "text": "文本回复",
        "function_call": "工具调用请求",
        "tool_call": "工具调用请求",
        "message": "消息",
        "ai": "模型消息",
        "assistant": "模型消息",
    }
    key = str(value)
    return mapping.get(key, key)


def _tool_argument_keys(request: object) -> list[str]:
    if not isinstance(request, dict):
        return []
    args = request.get("args")
    if not isinstance(args, dict):
        return []
    return sorted(str(key) for key in args)


def _response_preview(response: object, limit: int = 300) -> str:
    if not isinstance(response, dict):
        return ""
    content = response.get("content")
    preview = str(content or "")
    return preview if len(preview) <= limit else f"{preview[:limit]}..."


def _status_text(status: str) -> str:
    if status == "error":
        return "失败"
    if status == "started":
        return "开始"
    return status


def _single_tool_call_summary(call: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "index": call.get("index"),
        "tool_name": call.get("tool_name"),
        "tool_call_id": call.get("tool_call_id"),
        "status": _tool_status(call.get("event")),
    }
    if "request" in call:
        summary["request"] = call["request"]
    if "response" in call:
        summary["response"] = call["response"]
    return summary


def _tool_status(event: object) -> str:
    if event == "tool_call_end":
        return "completed"
    if event == "tool_call_error":
        return "error"
    return "started"


__all__ = [
    "ChatTurnResult",
    "conversation_trace_payload",
    "debug_summary_payload",
    "execution_step_summaries",
    "fallback_answer_from_tool_results",
    "run_agent_turn",
    "tool_call_summaries",
]
