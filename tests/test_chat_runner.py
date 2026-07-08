import json

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from src.chat.runner import (
    debug_summary_payload,
    execution_step_summaries,
    run_agent_turn,
    tool_call_summaries,
)
from src.chat.session import ChatSession
from src.chat.trace import (
    build_trace_tree,
    has_reasoning_block,
    is_assistant_message,
    merge_trace_events_into_turns,
    message_text,
    reasoning_text,
    write_conversation_trace_dump,
)


def test_chat_session_uses_web_thread_ids():
    session = ChatSession.new()

    assert session.thread_id.startswith("web-")
    assert session.config == {"configurable": {"thread_id": session.thread_id}}
    assert session.context().metadata == {"entrypoint": "web-ui"}


def test_reasoning_text_reads_standard_content_blocks():
    chunk = AIMessageChunk(
        content=[
            {"type": "reasoning", "reasoning": "先检查工具。"},
            {"type": "text", "text": "demo 可以验证工具调用。"},
        ],
        response_metadata={"model_provider": "openai"},
    )

    assert reasoning_text(chunk) == "先检查工具。"
    assert has_reasoning_block(chunk)
    assert message_text(chunk) == "demo 可以验证工具调用。"


def test_reasoning_text_reads_summary_blocks():
    chunk = AIMessageChunk(
        content=[
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "归纳步骤。"}],
            },
            {"type": "text", "text": "demo 可以验证非流式输出。"},
        ],
        response_metadata={"model_provider": "openai"},
    )

    assert reasoning_text(chunk) == "归纳步骤。"
    assert has_reasoning_block(chunk)
    assert message_text(chunk) == "demo 可以验证非流式输出。"


def test_reasoning_block_can_exist_without_exposed_text():
    chunk = AIMessageChunk(
        content=[
            {
                "type": "reasoning",
                "extras": {"content": [], "status": "in_progress"},
            },
            {"type": "text", "text": "demo 可以验证工具调用。"},
        ],
        response_metadata={"model_provider": "openai"},
    )

    assert has_reasoning_block(chunk)
    assert reasoning_text(chunk) == ""
    assert message_text(chunk) == "demo 可以验证工具调用。"


def test_trace_filters_tool_messages_from_rendered_answer():
    assert is_assistant_message(AIMessageChunk(content="给用户看的回复"))
    assert not is_assistant_message(
        ToolMessage(
            content='{"timezone":"Asia/Shanghai"}',
            name="query_current_date",
            tool_call_id="call-1",
        )
    )


def test_tool_call_summaries_collapse_start_and_end_events():
    calls = [
        {
            "index": 3,
            "type": "tool",
            "event": "tool_call_start",
            "tool_name": "query_current_date",
            "tool_call_id": "call-1",
            "request": {
                "name": "query_current_date",
                "args": {"days_offset": 1},
            },
        },
        {
            "index": 4,
            "type": "tool",
            "event": "tool_call_end",
            "tool_name": "query_current_date",
            "tool_call_id": "call-1",
            "response": {"content": '{"target_date":"2026-07-08"}'},
        },
        {
            "index": 7,
            "type": "tool",
            "event": "tool_call_start",
            "tool_name": "search_airfare_quotes",
            "tool_call_id": "call-2",
            "request": {
                "name": "search_airfare_quotes",
                "args": {"origin": "北京", "destination": "上海"},
            },
        },
        {
            "index": 8,
            "type": "tool",
            "event": "tool_call_end",
            "tool_name": "search_airfare_quotes",
            "tool_call_id": "call-2",
            "response": {"content": '{"quotes":[]}'},
        },
    ]

    summaries = tool_call_summaries(calls)

    assert summaries == [
        {
            "index": 3,
            "tool_name": "query_current_date",
            "tool_call_id": "call-1",
            "status": "completed",
            "request": {
                "name": "query_current_date",
                "args": {"days_offset": 1},
            },
            "response": {"content": '{"target_date":"2026-07-08"}'},
        },
        {
            "index": 7,
            "tool_name": "search_airfare_quotes",
            "tool_call_id": "call-2",
            "status": "completed",
            "request": {
                "name": "search_airfare_quotes",
                "args": {"origin": "北京", "destination": "上海"},
            },
            "response": {"content": '{"quotes":[]}'},
        },
    ]


def test_execution_step_summaries_group_react_steps_without_full_session_trace():
    calls = [
        {
            "index": 1,
            "type": "model",
            "event": "model_call_start",
            "request": {"messages": [{"role": "human", "content": "明天北京到上海"}]},
        },
        {
            "index": 2,
            "type": "model",
            "event": "model_call_end",
            "response": [{"content_block_types": ["reasoning", "function_call"]}],
        },
        {
            "index": 3,
            "type": "tool",
            "event": "tool_call_start",
            "tool_name": "query_current_date",
            "tool_call_id": "call-1",
            "request": {
                "name": "query_current_date",
                "args": {"days_offset": 1, "timezone_name": "Asia/Shanghai"},
            },
        },
        {
            "index": 4,
            "type": "tool",
            "event": "tool_call_end",
            "tool_name": "query_current_date",
            "tool_call_id": "call-1",
            "response": {"content": '{"target_date":"2026-07-08"}'},
        },
        {
            "index": 5,
            "type": "model",
            "event": "model_call_start",
            "request": {"messages": [{"role": "tool"}]},
        },
        {
            "index": 6,
            "type": "model",
            "event": "model_call_end",
            "response": [{"content_block_types": ["text"]}],
        },
    ]

    steps = execution_step_summaries(calls)

    assert steps == [
        {
            "index": 1,
            "kind": "react_step",
            "title": "ReAct Step 1",
            "status": "completed",
            "event_count": 4,
            "summary": "模型响应中请求调用 1 个工具。",
            "stages": [
                {
                    "kind": "thought",
                    "title": "模型响应",
                    "status": "completed",
                    "summary": "模型读取 1 条上下文消息，响应包含 内部推理标记、工具调用请求。",
                    "details": {
                        "message_count": 1,
                        "tool_count": 0,
                        "response_block_types": ["reasoning", "function_call"],
                        "response_preview": "",
                        "requested_tools": [],
                    },
                },
                {
                    "kind": "action",
                    "title": "工具调用",
                    "status": "completed",
                    "summary": "调用 query_current_date，参数：days_offset, timezone_name。",
                    "details": {
                        "tool_name": "query_current_date",
                        "tool_call_id": "call-1",
                        "argument_keys": ["days_offset", "timezone_name"],
                        "response_preview": '{"target_date":"2026-07-08"}',
                    },
                },
            ],
        },
        {
            "index": 5,
            "kind": "react_step",
            "title": "ReAct Step 2",
            "status": "completed",
            "event_count": 2,
            "summary": "模型生成最终回复。",
            "stages": [
                {
                    "kind": "thought",
                    "title": "模型响应",
                    "status": "completed",
                    "summary": "模型读取 1 条上下文消息，响应包含 文本回复。",
                    "details": {
                        "message_count": 1,
                        "tool_count": 0,
                        "response_block_types": ["text"],
                        "response_preview": "",
                        "requested_tools": [],
                    },
                },
            ],
        },
    ]
    assert "request" not in steps[0]["stages"][0]["details"]
    assert "messages" not in steps[0]["stages"][0]["details"]


def test_execution_step_summaries_pairs_batched_tool_start_and_end_events():
    calls = [
        {
            "index": 1,
            "type": "model",
            "event": "model_call_start",
            "request": {"messages": [{"role": "human", "content": "查未来两天票价"}]},
        },
        {
            "index": 2,
            "type": "model",
            "event": "model_call_end",
            "response": [{"content_block_types": ["reasoning", "function_call", "function_call"]}],
        },
        {
            "index": 3,
            "type": "tool",
            "event": "tool_call_start",
            "tool_name": "search_airfare_quotes",
            "tool_call_id": "call-1",
            "request": {"args": {"departure_date": "2026-07-08", "origin": "北京"}},
        },
        {
            "index": 4,
            "type": "tool",
            "event": "tool_call_start",
            "tool_name": "search_airfare_quotes",
            "tool_call_id": "call-2",
            "request": {"args": {"departure_date": "2026-07-09", "origin": "北京"}},
        },
        {
            "index": 5,
            "type": "tool",
            "event": "tool_call_end",
            "tool_name": "search_airfare_quotes",
            "tool_call_id": "call-1",
            "response": {"content": '{"quotes":[{"price":500}]}'},
        },
        {
            "index": 6,
            "type": "tool",
            "event": "tool_call_end",
            "tool_name": "search_airfare_quotes",
            "tool_call_id": "call-2",
            "response": {"content": '{"quotes":[{"price":420}]}'},
        },
        {
            "index": 7,
            "type": "model",
            "event": "model_call_start",
            "request": {"messages": [{"role": "tool"}]},
        },
        {
            "index": 8,
            "type": "model",
            "event": "model_call_end",
            "response": [{"content_block_types": ["text"]}],
        },
    ]

    steps = execution_step_summaries(calls)

    assert steps[0]["status"] == "completed"
    assert steps[0]["summary"] == "模型响应中请求调用 2 个工具。"
    assert [stage["status"] for stage in steps[0]["stages"]] == [
        "completed",
        "completed",
    ]
    batch = steps[0]["stages"][1]
    assert batch["kind"] == "action_batch"
    assert batch["summary"] == "批量调用 search_airfare_quotes × 2。"
    assert batch["details"]["tool_count"] == 2
    assert [item["tool_call_id"] for item in batch["details"]["tools"]] == ["call-1", "call-2"]
    assert batch["details"]["tools"][0]["response_preview"] == '{"quotes":[{"price":500}]}'
    assert batch["details"]["tools"][1]["response_preview"] == '{"quotes":[{"price":420}]}'
    assert steps[1]["summary"] == "模型生成最终回复。"


def test_execution_step_summaries_exposes_context_compaction_stage():
    calls = [
        {
            "index": 1,
            "type": "tool",
            "event": "tool_call_start",
            "tool_name": "generic_lookup",
            "tool_call_id": "call-1",
            "request": {"args": {"slot": 1}},
        },
        {
            "index": 2,
            "type": "tool",
            "event": "tool_call_end",
            "tool_name": "generic_lookup",
            "tool_call_id": "call-1",
            "response": {"content": '{"value":1}'},
        },
        {
            "index": 3,
            "type": "event",
            "event": "react_context_budget_compacted",
            "fields": {
                "estimate_chars": 56000,
                "threshold_chars": 27853,
                "observation_count": 10,
                "preserved_observation_count": 10,
                "dropped_observation_count": 0,
                "preview_truncated_count": 4,
                "compacted_state_preview": '{"layers":{"tool_observation_ledger":{"observations":[{"tool_name":"generic_lookup"}]}}}',
                "compacted_state_preview_chars": 91,
                "compacted_state_chars": 91,
                "compacted_state_sha256": "abc123",
            },
        },
        {
            "index": 4,
            "type": "model",
            "event": "model_call_start",
            "request": {"messages": [{"role": "human", "content": "final"}], "tools": []},
        },
        {
            "index": 5,
            "type": "model",
            "event": "model_call_end",
            "response": [{"content_block_types": ["text"]}],
        },
    ]

    steps = execution_step_summaries(calls)

    assert steps[0]["stages"][0]["kind"] == "action"
    assert steps[1]["kind"] == "context_compaction"
    assert steps[1]["title"] == "上下文状态压缩"
    assert steps[1]["summary"] == (
        "上下文超过预算，压缩历史状态并保留 10/10 条工具观察，"
        "丢弃 0 条；Agent 可继续调用工具。"
    )
    assert steps[1]["details"]["estimate_chars"] == 56000
    assert steps[1]["stages"][1]["title"] == "压缩后信息"
    assert steps[1]["stages"][1]["details"]["compacted_state_preview"].startswith(
        '{"layers"'
    )
    assert steps[1]["details"]["compacted_state_sha256"] == "abc123"
    assert steps[2]["summary"] == "模型生成最终回复。"


def test_execution_step_summaries_exposes_model_text_and_requested_tools():
    calls = [
        {
            "index": 1,
            "type": "model",
            "event": "model_call_start",
            "request": {"messages": [{"role": "human", "content": "查明天机票"}]},
        },
        {
            "index": 2,
            "type": "model",
            "event": "model_call_end",
            "response": [
                {
                    "content_block_types": ["reasoning", "text", "function_call"],
                    "content": [
                        {"type": "reasoning", "text": "hidden"},
                        {"type": "text", "text": "我先获取日期，再查票价。"},
                        {
                            "type": "function_call",
                            "name": "query_current_date",
                            "arguments": '{"days_offset":1,"timezone_name":"Asia/Shanghai"}',
                        },
                    ],
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "query_current_date",
                            "args": {
                                "days_offset": 1,
                                "timezone_name": "Asia/Shanghai",
                            },
                        }
                    ],
                }
            ],
        },
    ]

    steps = execution_step_summaries(calls)

    details = steps[0]["stages"][0]["details"]
    assert details["response_preview"] == "我先获取日期，再查票价。"
    assert details["requested_tools"] == [
        {
            "name": "query_current_date",
            "id": "call-1",
            "argument_keys": ["days_offset", "timezone_name"],
        },
    ]


def test_debug_summary_payload_extracts_recent_turn_metrics():
    trace = {
        "thread_id": "web-debug",
        "turn_count": 1,
        "event_count": 8,
        "turns": [
            {
                "status": "success",
                "user_input": "查询明天北京到上海",
                "assistant_output": "查到 7 条报价。",
                "empty_visible_output": True,
                "malformed_tool_call_text_seen": False,
                "calls": [
                    {
                        "index": 1,
                        "type": "model",
                        "event": "model_call_start",
                        "request": {
                            "messages": [{"role": "human", "content": "查询明天北京到上海"}],
                        },
                    },
                    {
                        "index": 2,
                        "type": "model",
                        "event": "model_call_end",
                        "response": [{"content": "调用日期工具"}],
                    },
                    {
                        "index": 3,
                        "type": "tool",
                        "event": "tool_call_start",
                        "tool_name": "search_airfare_quotes",
                        "tool_call_id": "call-1",
                        "request": {
                            "name": "search_airfare_quotes",
                            "args": {"origin": "北京", "destination": "上海"},
                        },
                    },
                    {
                        "index": 4,
                        "type": "tool",
                        "event": "tool_call_end",
                        "tool_name": "search_airfare_quotes",
                        "tool_call_id": "call-1",
                        "response": {
                            "content": (
                                '{"captured_at":"2026-07-07T16:34:27+08:00",'
                                '"sources_used":["fliggy_mcp"],'
                                '"quotes":[{"price":400},{"price":700}],'
                                '"limitations":["sample only"]}'
                            )
                        },
                    },
                ],
            }
        ],
        "events": [],
    }

    summary = debug_summary_payload(
        trace,
        model_name="google/gemma-4-e2b",
        context_window_tokens=8192,
    )

    assert summary["session"] == {
        "thread_id": "web-debug",
        "turn_count": 1,
        "event_count": 8,
        "last_status": "success",
    }
    assert summary["model"]["model_name"] == "google/gemma-4-e2b"
    assert summary["model"]["context_window_tokens"] == 8192
    assert summary["model"]["last_message_count"] == 1
    assert summary["model"]["estimated_prompt_chars"] > 0
    assert summary["model"]["estimated_response_chars"] > 0
    assert summary["execution"]["model_call_count"] == 1
    assert summary["execution"]["tool_call_count"] == 1
    assert summary["execution"]["tool_success_count"] == 1
    assert summary["execution"]["tool_error_count"] == 0
    assert summary["execution"]["recent_tools"] == ["search_airfare_quotes"]
    assert summary["sources"]["sources_used"] == ["fliggy_mcp"]
    assert summary["sources"]["captured_at"] == "2026-07-07T16:34:27+08:00"
    assert summary["sources"]["fact_counts"] == {"quotes": 2}
    assert summary["sources"]["limitations"] == ["sample only"]
    assert summary["warnings"] == ["模型未生成可展示文本。"]


def test_debug_summary_context_estimate_uses_single_model_request_not_react_sum():
    first_request = {
        "messages": [{"role": "human", "content": "查询明天北京到上海"}],
        "tools": [{"name": "query_current_date", "description": "date tool"}],
    }
    second_request = {
        "messages": [
            {"role": "human", "content": "查询明天北京到上海"},
            {"role": "ai", "content": "调用日期工具"},
            {"role": "tool", "content": '{"target_date":"2026-07-08"}'},
        ],
        "tools": [{"name": "query_current_date", "description": "date tool"}],
    }
    trace = {
        "thread_id": "web-debug",
        "turn_count": 1,
        "event_count": 4,
        "turns": [
            {
                "status": "success",
                "calls": [
                    {
                        "index": 1,
                        "type": "model",
                        "event": "model_call_start",
                        "request": first_request,
                    },
                    {
                        "index": 2,
                        "type": "model",
                        "event": "model_call_start",
                        "request": second_request,
                    },
                ],
            }
        ],
        "events": [],
    }

    summary = debug_summary_payload(
        trace,
        model_name="qwen3.5-4b-mlx",
        context_window_tokens=128,
    )

    assert summary["model"]["last_message_count"] == 3
    assert summary["model"]["estimated_prompt_chars"] == len(
        json.dumps(second_request, ensure_ascii=False, separators=(",", ":"))
    )
    assert summary["model"]["max_prompt_chars"] == max(
        len(json.dumps(first_request, ensure_ascii=False, separators=(",", ":"))),
        len(json.dumps(second_request, ensure_ascii=False, separators=(",", ":"))),
    )
    assert summary["model"]["total_react_prompt_chars"] == (
        len(json.dumps(first_request, ensure_ascii=False, separators=(",", ":")))
        + len(json.dumps(second_request, ensure_ascii=False, separators=(",", ":")))
    )
    assert summary["model"]["context_usage_estimate"] == round(
        summary["model"]["estimated_prompt_chars"] / (128 * 4),
        4,
    )


def test_run_agent_turn_logs_and_writes_multi_turn_trace(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.responses = ["第一轮回复", "第二轮回复"]

        def invoke(self, *args, **kwargs):
            return {"messages": [AIMessage(content=self.responses.pop(0))]}

    session = ChatSession(thread_id="web-test")
    fake_agent = FakeAgent()

    first = run_agent_turn(
        "第一轮问题",
        session,
        agent_instance=fake_agent,
        trace_dir=tmp_path / "traces",
    )
    second = run_agent_turn(
        "第二轮问题",
        session,
        agent_instance=fake_agent,
        trace_dir=tmp_path / "traces",
    )

    payload = json.loads((tmp_path / "traces" / "web-test.json").read_text())

    assert first.answer == "第一轮回复"
    assert second.answer == "第二轮回复"
    assert payload["thread_id"] == "web-test"
    assert payload["turn_count"] == 2
    assert payload["turns"][0]["user_input"] == "第一轮问题"
    assert payload["turns"][0]["assistant_output"] == "第一轮回复"
    assert payload["turns"][0]["status"] == "success"
    assert payload["turns"][0]["stream_chunks"] == []
    assert payload["turns"][0]["invoke_output"]["messages"] == [
        {
            "content": "第一轮回复",
            "message_type": "AIMessage",
            "role": "ai",
        }
    ]
    assert payload["turns"][1]["user_input"] == "第二轮问题"
    assert payload["turns"][1]["assistant_output"] == "第二轮回复"
    assert second.trace["thread_id"] == "web-test"
    assert second.trace["turn_count"] == 2
    assert second.trace["turns"][0]["user_input"] == "第一轮问题"
    assert second.trace["turns"][1]["assistant_output"] == "第二轮回复"
    assert second.trace["event_count"] == 8
    assert second.trace["tree"]["type"] == "session"
    assert second.trace["tree"]["children"][0]["type"] == "turn"
    assert second.trace["tree"]["children"][0]["label"] == "Turn 1: 第一轮问题"
    assert second.execution_steps == []
    assert [event["event"] for event in payload["events"]] == [
        "conversation_turn_start",
        "agent_run_start",
        "agent_run_end",
        "conversation_turn_end",
        "conversation_turn_start",
        "agent_run_start",
        "agent_run_end",
        "conversation_turn_end",
    ]


def test_run_agent_turn_falls_back_when_final_message_has_no_visible_text(tmp_path):
    class ReasoningOnlyAgent:
        def invoke(self, *args, **kwargs):
            return {
                "messages": [
                    ToolMessage(
                        content='{"flight_number":"KN5977","flight_records":[]}',
                        name="query_flight_information",
                        tool_call_id="call-1",
                    ),
                    AIMessage(
                        content=[
                            {
                                "type": "reasoning",
                                "content": [
                                    {
                                        "type": "reasoning_text",
                                        "text": (
                                            "<function=query_current_date>\n"
                                            "<parameter=days_offset>\n"
                                            "1\n"
                                            "</parameter>\n"
                                            "</function>"
                                        ),
                                    }
                                ],
                            }
                        ],
                    ),
                ]
            }

    session = ChatSession(thread_id="web-empty-answer")

    result = run_agent_turn(
        "查一下KN5977的具体信息",
        session,
        agent_instance=ReasoningOnlyAgent(),
        trace_dir=tmp_path / "traces",
    )
    payload = json.loads((tmp_path / "traces" / "web-empty-answer.json").read_text())
    turn = payload["turns"][0]

    assert result.status == "success"
    assert result.answer
    assert "模型返回了工具调用格式文本" in result.answer
    assert turn["empty_visible_output"] is True
    assert turn["malformed_tool_call_text_seen"] is True
    assert turn["answer_started"] is True
    assert turn["assistant_output"] == result.answer


def test_run_agent_turn_returns_error_payload_and_trace(tmp_path):
    class BrokenAgent:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("boom")

    session = ChatSession(thread_id="web-error")

    result = run_agent_turn(
        "错误问题",
        session,
        agent_instance=BrokenAgent(),
        trace_dir=tmp_path / "traces",
    )
    payload = json.loads((tmp_path / "traces" / "web-error.json").read_text())

    assert result.status == "error"
    assert result.error_type == "RuntimeError"
    assert "RuntimeError: boom" in result.answer
    assert result.trace["thread_id"] == "web-error"
    assert result.trace["turn_count"] == 1
    assert result.trace["turns"][0]["status"] == "error"
    assert payload["thread_id"] == "web-error"
    assert payload["turn_count"] == 1
    assert payload["turns"][0]["status"] == "error"
    assert payload["turns"][0]["user_input"] == "错误问题"
    assert payload["turns"][0]["partial_assistant_output"] == ""
    assert payload["turns"][0]["error_type"] == "RuntimeError"
    assert payload["turns"][0]["calls"][-1]["event"] == "conversation_turn_error"
    assert [event["event"] for event in payload["events"]] == [
        "conversation_turn_start",
        "agent_run_start",
        "agent_run_error",
        "conversation_turn_error",
    ]


def test_write_conversation_trace_dump_creates_json_file(tmp_path):
    output_path = write_conversation_trace_dump(
        thread_id="web-direct",
        turns=[{"user_input": "你好", "assistant_output": "你好"}],
        trace_dir=tmp_path,
    )

    payload = json.loads(output_path.read_text())

    assert output_path == tmp_path / "web-direct.json"
    assert payload["thread_id"] == "web-direct"
    assert payload["turn_count"] == 1
    assert payload["turns"][0]["user_input"] == "你好"


def test_merge_trace_events_into_turns_builds_ordered_call_chain():
    turns = [
        {
            "turn_id": "request-1",
            "user_input": "北京到上海明天有航班吗",
            "stream_chunks": [],
        }
    ]
    events = [
        {
            "event": "conversation_turn_start",
            "level": "INFO",
            "fields": {"turn_id": "request-1", "user_input": "北京到上海明天有航班吗"},
        },
        {
            "event": "model_call_start",
            "level": "INFO",
            "fields": {
                "turn_id": "request-1",
                "request_trace": {
                    "system_prompt": "完整 prompt",
                    "messages": [{"role": "human", "content": "北京到上海明天有航班吗"}],
                },
            },
        },
        {
            "event": "model_call_end",
            "level": "INFO",
            "fields": {
                "turn_id": "request-1",
                "response_trace": [
                    {
                        "role": "ai",
                        "content": [
                            {
                                "type": "function_call",
                                "name": "resolve_flight_locations",
                                "arguments": '{"locations":["北京","上海"]}',
                            }
                        ],
                    }
                ],
            },
        },
        {
            "event": "tool_call_start",
            "level": "INFO",
            "fields": {
                "turn_id": "request-1",
                "tool_call_id": "call-1",
                "tool_name": "resolve_flight_locations",
                "tool_call": {
                    "name": "resolve_flight_locations",
                    "args": {"locations": ["北京", "上海"]},
                },
            },
        },
        {
            "event": "tool_call_end",
            "level": "INFO",
            "fields": {
                "turn_id": "request-1",
                "tool_call_id": "call-1",
                "tool_name": "resolve_flight_locations",
                "response_trace": {"content": '{"items":[{"input":"北京"}]}'},
            },
        },
    ]

    merged = merge_trace_events_into_turns(turns, events)

    assert [call["event"] for call in merged[0]["calls"]] == [
        "conversation_turn_start",
        "model_call_start",
        "model_call_end",
        "tool_call_start",
        "tool_call_end",
    ]
    assert merged[0]["calls"][1]["request"]["system_prompt"] == "完整 prompt"
    assert merged[0]["calls"][3]["tool_name"] == "resolve_flight_locations"
    assert merged[0]["calls"][3]["request"]["args"] == {
        "locations": ["北京", "上海"]
    }
    assert merged[0]["calls"][4]["response"] == {
        "content": '{"items":[{"input":"北京"}]}'
    }


def test_build_trace_tree_groups_turns_into_react_stages():
    trace = {
        "thread_id": "web-tree",
        "turn_count": 1,
        "event_count": 5,
        "events": [],
        "turns": [
            {
                "turn_id": "request-1",
                "turn_index": 0,
                "status": "success",
                "user_input": "北京到上海明天有航班吗",
                "assistant_output": "查到一些结果",
                "calls": [
                    {
                        "index": 0,
                        "type": "conversation",
                        "event": "conversation_turn_start",
                        "level": "INFO",
                        "fields": {"turn_id": "request-1"},
                    },
                    {
                        "index": 1,
                        "type": "model",
                        "event": "model_call_start",
                        "level": "INFO",
                        "request": {"messages": [{"role": "human"}]},
                        "fields": {"turn_id": "request-1"},
                    },
                    {
                        "index": 2,
                        "type": "model",
                        "event": "model_call_end",
                        "level": "INFO",
                        "response": [{"role": "ai", "content": "准备调用工具"}],
                        "fields": {"turn_id": "request-1"},
                    },
                    {
                        "index": 3,
                        "type": "tool",
                        "event": "tool_call_start",
                        "tool_name": "resolve_flight_locations",
                        "tool_call_id": "call-1",
                        "request": {
                            "name": "resolve_flight_locations",
                            "args": {"locations": ["北京", "上海"]},
                        },
                    },
                    {
                        "index": 4,
                        "type": "tool",
                        "event": "tool_call_end",
                        "tool_name": "resolve_flight_locations",
                        "tool_call_id": "call-1",
                        "response": {"content": '{"items":[]}'},
                    },
                    {
                        "index": 5,
                        "type": "conversation",
                        "event": "conversation_turn_end",
                        "level": "INFO",
                        "fields": {"turn_id": "request-1"},
                    },
                ],
            }
        ],
    }

    tree = build_trace_tree(trace)

    assert tree["type"] == "session"
    assert tree["label"] == "Session web-tree"
    assert tree["status"] == "success"
    assert tree["meta"] == {
        "thread_id": "web-tree",
        "turn_count": 1,
        "event_count": 5,
    }
    turn = tree["children"][0]
    assert turn["type"] == "turn"
    assert turn["label"] == "Turn 1: 北京到上海明天有航班吗"
    assert turn["meta"]["user_input"] == "北京到上海明天有航班吗"
    assert "raw_turn" not in turn["meta"]
    assert turn["meta"]["assistant_output_preview"] == "查到一些结果"
    assert [child["type"] for child in turn["children"]] == [
        "react_input",
        "react_step",
        "react_final",
        "raw_trace",
    ]
    assert turn["children"][0]["label"] == "User Input"
    assert turn["children"][0]["meta"]["calls"][0]["event"] == "conversation_turn_start"
    step = turn["children"][1]
    assert step["label"] == "ReAct Step 1"
    assert step["meta"]["event_count"] == 4
    assert "calls" not in step["meta"]
    assert [child["type"] for child in step["children"]] == [
        "react_thought",
        "react_action",
    ]
    thought = step["children"][0]
    assert thought["label"] == "Thought / Model Call"
    assert [call["event"] for call in thought["meta"]["calls"]] == [
        "model_call_start",
        "model_call_end",
    ]
    assert thought["children"][0]["type"] == "model"
    action = step["children"][1]
    assert action["label"] == "Action / Tool Call"
    assert [call["event"] for call in action["meta"]["calls"]] == [
        "tool_call_start",
        "tool_call_end",
    ]
    tool = action["children"][0]
    assert tool["label"] == "Tool: resolve_flight_locations"
    assert tool["status"] == "completed"
    assert tool["meta"]["request"]["args"] == {"locations": ["北京", "上海"]}
    assert tool["meta"]["response"] == {"content": '{"items":[]}'}
    assert [child["label"] for child in tool["children"]] == [
        "tool_call_start: resolve_flight_locations",
        "tool_call_end: resolve_flight_locations",
    ]
    final = turn["children"][2]
    assert final["label"] == "Final Response"
    assert final["meta"]["assistant_output"] == "查到一些结果"
    assert final["meta"]["calls"][0]["event"] == "conversation_turn_end"
    raw = turn["children"][3]
    assert raw["label"] == "Raw Debug"
    assert raw["meta"]["raw_turn"]["calls"][0]["event"] == "conversation_turn_start"
    assert [call["event"] for call in raw["meta"]["calls"]] == [
        "conversation_turn_start",
        "model_call_start",
        "model_call_end",
        "tool_call_start",
        "tool_call_end",
        "conversation_turn_end",
    ]
