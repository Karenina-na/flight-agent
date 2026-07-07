import json
import threading

from src.chat.runner import ChatTurnResult
from src.web_ui.server import DEMO_MESSAGE, INDEX_HTML, TRACE_HTML, WebApp


def test_web_app_starts_with_web_session():
    app = WebApp()

    assert app.session.thread_id.startswith("web-")
    assert app.messages == []


def test_web_app_new_session_resets_messages():
    app = WebApp()
    app.messages = [{"role": "user", "content": "旧问题"}]
    old_thread_id = app.session.thread_id

    payload = app.new_session()

    assert payload["thread_id"].startswith("web-")
    assert payload["thread_id"] != old_thread_id
    assert payload["messages"] == []
    assert payload["trace"]["turn_count"] == 0
    assert payload["trace"]["events"] == []
    assert payload["debug_summary"]["session"]["turn_count"] == 0
    assert app.messages == []


def test_web_app_lists_registered_air_ticket_tools():
    app = WebApp()

    tools = app.list_tools()
    tool_names = {tool["name"] for tool in tools}

    assert "resolve_flight_locations" in tool_names
    assert "search_airfare_quotes" in tool_names
    assert "query_flight_information" in tool_names
    assert "query_current_date" in tool_names


def test_web_app_empty_chat_does_not_call_runner():
    app = WebApp()

    payload = app.chat("   ")

    assert payload["status"] == "empty"
    assert payload["answer"] == "请输入问题后再发送。"
    assert payload["messages"] == []
    assert payload["trace"]["turn_count"] == 0
    assert payload["debug_summary"]["session"]["last_status"] == "ready"


def test_web_app_trace_state_uses_revision_to_skip_full_payload():
    app = WebApp()

    first_payload = app.trace_state()
    cached_payload = app.trace_state(first_payload["trace_revision"])

    assert first_payload["status"] == "ready"
    assert first_payload["trace"]["turn_count"] == 0
    assert first_payload["debug_summary"]["session"]["turn_count"] == 0
    assert cached_payload == {
        "thread_id": app.session.thread_id,
        "status": "not_modified",
        "trace_revision": first_payload["trace_revision"],
    }


def test_web_app_chat_appends_user_and_assistant_messages(monkeypatch, tmp_path):
    def fake_run_agent_turn(message, session, entrypoint):
        assert message == "查一下北京机场"
        assert entrypoint == "web-ui.chat"
        return ChatTurnResult(
            thread_id=session.thread_id,
            answer="北京有 PEK 和 PKX。",
            status="success",
            trace_path=tmp_path / "trace.json",
            tool_calls=[{"event": "tool_call_start", "tool_name": "resolve_flight_locations"}],
            execution_steps=[
                {
                    "kind": "tool",
                    "title": "调用工具 resolve_flight_locations",
                    "status": "completed",
                    "summary": "工具 resolve_flight_locations 调用完成。",
                    "details": {"tool_name": "resolve_flight_locations"},
                }
            ],
            trace={
                "thread_id": session.thread_id,
                "turn_count": 1,
                "event_count": 4,
                "turns": [{"user_input": message, "assistant_output": "北京有 PEK 和 PKX。"}],
                "events": [{"event": "conversation_turn_start"}],
            },
        )

    monkeypatch.setattr("src.web_ui.server.run_agent_turn", fake_run_agent_turn)
    app = WebApp()

    payload = app.chat(" 查一下北京机场 ")

    assert payload["status"] == "success"
    assert payload["answer"] == "北京有 PEK 和 PKX。"
    assert payload["messages"] == [
        {"role": "user", "content": "查一下北京机场"},
        {
            "role": "assistant",
            "content": "北京有 PEK 和 PKX。",
            "execution_steps": [
                {
                    "kind": "tool",
                    "title": "调用工具 resolve_flight_locations",
                    "status": "completed",
                    "summary": "工具 resolve_flight_locations 调用完成。",
                    "details": {"tool_name": "resolve_flight_locations"},
                }
            ],
        },
    ]
    assert payload["tool_calls"] == [
        {"event": "tool_call_start", "tool_name": "resolve_flight_locations"}
    ]
    assert payload["execution_steps"][0]["title"] == "调用工具 resolve_flight_locations"
    assert payload["trace_path"] == str(tmp_path / "trace.json")
    assert payload["trace"]["turn_count"] == 1
    assert payload["trace"]["turns"][0]["user_input"] == "查一下北京机场"
    assert payload["trace"]["events"][0]["event"] == "conversation_turn_start"
    assert payload["debug_summary"]["session"]["turn_count"] == 1
    assert payload["debug_summary"]["model"]["model_name"]
    assert payload["debug_summary"]["execution"]["tool_call_count"] == 0


def test_web_app_rejects_duplicate_chat_while_request_is_running(monkeypatch, tmp_path):
    started = threading.Event()
    finish = threading.Event()
    calls = []

    def slow_run_agent_turn(message, session, entrypoint):
        calls.append(message)
        started.set()
        finish.wait(timeout=2)
        return ChatTurnResult(
            thread_id=session.thread_id,
            answer="第一条回复",
            status="success",
            trace_path=tmp_path / "trace.json",
        )

    monkeypatch.setattr("src.web_ui.server.run_agent_turn", slow_run_agent_turn)
    app = WebApp()
    results = []

    first_thread = threading.Thread(
        target=lambda: results.append(app.chat("查一下北京到上海")),
    )
    first_thread.start()
    assert started.wait(timeout=1)

    duplicate = app.chat("查一下北京到上海")

    finish.set()
    first_thread.join(timeout=2)

    assert len(calls) == 1
    assert duplicate["status"] == "busy"
    assert duplicate["answer"] == "上一条消息仍在处理中，请稍后再试。"
    assert duplicate["trace"]["turn_count"] == 0
    assert duplicate["debug_summary"]["session"]["last_status"] == "ready"
    assert results[0]["status"] == "success"
    assert app.messages == [
        {"role": "user", "content": "查一下北京到上海"},
        {"role": "assistant", "content": "第一条回复", "execution_steps": []},
    ]


def test_demo_message_matches_air_ticket_mvp():
    assert "北京到上海" in DEMO_MESSAGE
    assert "2026-07-10" in DEMO_MESSAGE
    assert "机票报价样本" in DEMO_MESSAGE
    assert "create_demo_task" not in DEMO_MESSAGE


def test_main_page_left_sidebar_is_minimal_and_demo_fills_input_only():
    assert "快捷操作" in INDEX_HTML
    assert "运行示例" in INDEX_HTML
    assert "新建会话" in INDEX_HTML
    assert "能力范围" not in INDEX_HTML
    assert "示例问题" not in INDEX_HTML
    assert "工具目录" not in INDEX_HTML
    assert "查看工具" not in INDEX_HTML
    assert 'id="toolsBtn"' not in INDEX_HTML
    assert 'id="toolCatalog"' not in INDEX_HTML
    assert 'document.querySelector("#toolsBtn")' not in INDEX_HTML
    assert 'fetch("/api/tools")' not in INDEX_HTML
    assert 'postJson("/api/demo")' not in INDEX_HTML
    assert "inputEl.value = DEMO_PROMPT" in INDEX_HTML
    assert "const DEMO_PROMPT" in INDEX_HTML


def test_main_page_keeps_debug_sidebar_lightweight():
    assert "Debug Summary" in INDEX_HTML
    assert "打开完整 Trace" in INDEX_HTML
    assert 'href="/trace"' in INDEX_HTML
    assert 'id="traceTree"' not in INDEX_HTML
    assert 'id="rawTrace"' not in INDEX_HTML
    assert "renderAssistantContent(message.content)" in INDEX_HTML
    assert "messagesFromPayload(payload, pendingMessages)" in INDEX_HTML
    assert "payload.answer" in INDEX_HTML
    assert "未获取到可展示的助手回复。" in INDEX_HTML
    assert 'message.role === "assistant"' in INDEX_HTML
    assert "let currentMessages = []" in INDEX_HTML
    assert "currentMessages.concat([" in INDEX_HTML
    assert 'document.querySelectorAll(".msg")' not in INDEX_HTML
    assert "normalizeMarkdownTables" in INDEX_HTML
    assert "renderTable(tableLines)" in INDEX_HTML
    assert "for (let index = 0; index < lines.length; index += 1)" in INDEX_HTML
    assert ".msg.assistant table" in INDEX_HTML
    assert "renderExecutionSteps(message.execution_steps)" in INDEX_HTML
    assert INDEX_HTML.index("renderExecutionSteps(message.execution_steps)") < INDEX_HTML.index("renderAssistantContent(message.content)")
    assert "execution-steps" in INDEX_HTML
    assert "assistant-output" in INDEX_HTML
    assert "assistant-answer" in INDEX_HTML
    assert "执行过程" in INDEX_HTML
    assert "最终输出" in INDEX_HTML
    assert "execution_steps" in INDEX_HTML
    assert "模型上下文" in INDEX_HTML
    assert "执行概览" in INDEX_HTML
    assert "数据来源" not in INDEX_HTML
    assert "最近工具" not in INDEX_HTML
    assert "提醒" in INDEX_HTML
    assert 'id="modelName"' in INDEX_HTML
    assert 'id="contextWindow"' in INDEX_HTML
    assert 'id="lastMessageCount"' in INDEX_HTML
    assert 'id="estimatedPromptChars"' in INDEX_HTML
    assert 'id="estimatedResponseChars"' in INDEX_HTML
    assert 'id="contextUsageEstimate"' in INDEX_HTML
    assert 'id="modelCallCount"' in INDEX_HTML
    assert 'id="toolCallCount"' in INDEX_HTML
    assert 'id="toolSuccessCount"' in INDEX_HTML
    assert 'id="toolErrorCount"' in INDEX_HTML
    assert 'id="sourcesUsed"' not in INDEX_HTML
    assert 'id="capturedAt"' not in INDEX_HTML
    assert 'id="factCounts"' not in INDEX_HTML
    assert 'id="limitations"' not in INDEX_HTML
    assert 'id="toolSummary"' not in INDEX_HTML
    assert 'id="debugWarnings"' in INDEX_HTML
    assert "payload.debug_summary" in INDEX_HTML
    assert "renderReactStage" in INDEX_HTML
    assert "step.stages" in INDEX_HTML
    assert "renderToolSummary" not in INDEX_HTML


def test_trace_page_renders_full_synced_trace_viewer():
    assert 'id="traceTree"' in TRACE_HTML
    assert 'id="rawTrace"' in TRACE_HTML
    assert 'id="expandTraceBtn"' in TRACE_HTML
    assert 'id="collapseTraceBtn"' in TRACE_HTML
    assert 'href="/"' in TRACE_HTML
    assert 'fetch(`/api/trace-state?revision=${encodeURIComponent(currentTraceRevision)}`)' in TRACE_HTML
    assert "window.setInterval(() => loadTrace(), 3000)" in TRACE_HTML
    assert "traceOpenState" in TRACE_HTML
    assert "captureTraceOpenState()" in TRACE_HTML
    assert "renderRawTrace(trace, payload.trace_revision" in TRACE_HTML
    assert "json-key" in TRACE_HTML
    assert "captureScrollState()" in TRACE_HTML
    assert "restoreScrollState(scrollState)" in TRACE_HTML
    assert "window.requestAnimationFrame(() => restoreScrollState(scrollState))" in TRACE_HTML
    assert "metaScrollState" in TRACE_HTML
    assert 'document.querySelectorAll(".node-meta")' in TRACE_HTML
    assert "meta.dataset.metaKey = nodeKey" in TRACE_HTML
    assert "currentTraceTreeSignature" in TRACE_HTML
    assert "nextTreeSignature !== currentTraceTreeSignature" in TRACE_HTML
    assert "loadTrace({forceTree: true})" in TRACE_HTML
    assert "payload.status === \"not_modified\"" in TRACE_HTML
    assert "currentRawTraceSignature" in TRACE_HTML
    assert "rawTraceEl.innerHTML = highlightedJson(trace)" in TRACE_HTML
    assert "react_step" in TRACE_HTML
    assert "react_thought" in TRACE_HTML
    assert "react_action" in TRACE_HTML
    assert "react_final" in TRACE_HTML


def test_chat_turn_result_serializes_to_json_payload(tmp_path):
    result = ChatTurnResult(
        thread_id="web-test",
        answer="回复",
        status="success",
        trace_path=tmp_path / "trace.json",
        reasoning_block_seen=True,
        tool_calls=[{"tool_name": "query_current_date"}],
        execution_steps=[{"title": "调用工具 query_current_date"}],
        trace={"thread_id": "web-test", "turn_count": 2, "turns": [], "events": []},
    )

    payload = result.as_dict()

    assert payload["thread_id"] == "web-test"
    assert payload["trace_path"] == str(tmp_path / "trace.json")
    assert payload["reasoning_block_seen"] is True
    assert payload["execution_steps"] == [{"title": "调用工具 query_current_date"}]
    assert payload["trace"]["turn_count"] == 2
    assert json.loads(json.dumps(payload, ensure_ascii=False))["answer"] == "回复"
