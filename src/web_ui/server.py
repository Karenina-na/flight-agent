"""Small stdlib web UI for the air ticket agent demo."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.chat import (
    ChatSession,
    conversation_trace_payload,
    debug_summary_payload,
    run_agent_turn,
)
from src.config import load_settings
from src.tools import get_tools

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
DEMO_MESSAGE = (
    "请查询北京到上海在 2026-07-10 的机票报价样本，"
    "并说明查到的事实、信息出处、查询时间和数据限制。"
)


class WebApp:
    """In-memory application state for the local browser demo."""

    def __init__(self) -> None:
        self.session = ChatSession.new()
        self.messages: list[dict[str, Any]] = []
        self.settings = load_settings()
        self._chat_lock = Lock()

    def new_session(self) -> dict[str, Any]:
        """Start a fresh browser conversation."""
        self.session = ChatSession.new()
        self.messages = []
        return {
            "thread_id": self.session.thread_id,
            "messages": self.messages,
            "trace": self.trace_payload(),
            "debug_summary": self.debug_summary(),
        }

    def trace_payload(self) -> dict[str, Any]:
        """Return the current full multi-turn trace for the debug panel."""
        return conversation_trace_payload(self.session)

    def trace_revision(self) -> str:
        """Return a cheap revision marker for trace polling."""
        last_turn = self.session.turns[-1] if self.session.turns else {}
        last_marker = ""
        if isinstance(last_turn, dict):
            last_marker = str(
                last_turn.get("ended_at")
                or last_turn.get("started_at")
                or last_turn.get("status")
                or ""
            )
        return (
            f"{self.session.thread_id}:"
            f"{len(self.session.turns)}:"
            f"{len(self.session.events)}:"
            f"{last_marker}"
        )

    def trace_state(self, known_revision: str | None = None) -> dict[str, Any]:
        """Return full trace only when it changed since the client revision."""
        revision = self.trace_revision()
        if known_revision and known_revision == revision:
            return {
                "thread_id": self.session.thread_id,
                "status": "not_modified",
                "trace_revision": revision,
            }

        trace = self.trace_payload()
        return {
            "thread_id": self.session.thread_id,
            "status": "ready",
            "trace_revision": revision,
            "trace": trace,
            "debug_summary": self.debug_summary(trace),
        }

    def debug_summary(self, trace: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return compact debug metrics for the right side panel."""
        return debug_summary_payload(
            trace or self.trace_payload(),
            model_name=self.settings.llm.model,
            context_window_tokens=self.settings.llm.context_window_tokens,
        )

    def list_tools(self) -> list[dict[str, str]]:
        """Return registered tool metadata for the UI."""
        tools = []
        for tool in get_tools():
            tools.append(
                {
                    "name": str(tool.name),
                    "description": str(getattr(tool, "description", "")).strip(),
                }
            )
        return tools

    def chat(self, message: str) -> dict[str, Any]:
        """Run one chat turn and return UI-ready JSON."""
        cleaned = message.strip()
        if not cleaned:
            return {
                "thread_id": self.session.thread_id,
                "answer": "请输入问题后再发送。",
                "status": "empty",
                "messages": self.messages,
                "trace_path": "",
                "tool_calls": [],
                "trace": self.trace_payload(),
                "debug_summary": self.debug_summary(),
            }

        if not self._chat_lock.acquire(blocking=False):
            return {
                "thread_id": self.session.thread_id,
                "answer": "上一条消息仍在处理中，请稍后再试。",
                "status": "busy",
                "messages": self.messages,
                "trace_path": "",
                "tool_calls": [],
                "trace": self.trace_payload(),
                "debug_summary": self.debug_summary(),
            }

        try:
            self.messages.append({"role": "user", "content": cleaned})
            result = run_agent_turn(cleaned, self.session, entrypoint="web-ui.chat")
            self.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "execution_steps": result.execution_steps,
                }
            )
            payload = result.as_dict()
            payload["messages"] = self.messages
            payload["debug_summary"] = self.debug_summary(payload.get("trace"))
            return payload
        finally:
            self._chat_lock.release()


def run_web_ui(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Run the local browser UI."""
    app = WebApp()
    handler_class = build_handler(app)
    server = ThreadingHTTPServer((host, port), handler_class)
    print(f"机票事实查询 Web UI: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n再见。")
    finally:
        server.server_close()


def build_handler(app: WebApp) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to one WebApp instance."""

    class WebUIRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            if path == "/":
                self._send_html(INDEX_HTML)
                return
            if path == "/trace":
                self._send_html(TRACE_HTML)
                return
            if path == "/api/trace-state":
                query = parse_qs(parsed_url.query)
                revision = (query.get("revision") or [""])[0]
                self._send_json(app.trace_state(revision))
                return
            if path == "/api/state":
                self._send_json(
                    {
                        "thread_id": app.session.thread_id,
                        "messages": app.messages,
                        "trace": app.trace_payload(),
                        "debug_summary": app.debug_summary(),
                    }
                )
                return
            if path == "/api/tools":
                self._send_json({"tools": app.list_tools()})
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/chat":
                payload = self._read_json()
                self._send_json(app.chat(str(payload.get("message", ""))))
                return
            if path == "/api/new":
                self._send_json(app.new_session())
                return
            if path == "/api/demo":
                self._send_json(app.chat(DEMO_MESSAGE))
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw_body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(raw_body)
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}

        def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = html.encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(
            self,
            payload: dict[str, Any],
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return WebUIRequestHandler


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SkyPilot 机票事实查询</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7fb;
      --panel: #ffffff;
      --ink: #18202f;
      --muted: #697386;
      --line: #dfe3eb;
      --accent: #2563eb;
      --accent-dark: #1e40af;
      --soft: #eef4ff;
      --ok: #0f766e;
      --warn: #b45309;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .shell {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr) 280px;
      gap: 16px;
      height: 100vh;
      padding: 16px;
    }
    aside, main, section {
      min-height: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    aside, section {
      padding: 16px;
      overflow: auto;
    }
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .brand {
      display: grid;
      grid-template-columns: 40px minmax(0, 1fr);
      gap: 10px;
      align-items: center;
    }
    .brand-mark {
      width: 40px;
      height: 40px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      letter-spacing: 0;
    }
    .brand h1 {
      font-size: 19px;
      line-height: 1.1;
    }
    .brand-subtitle {
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
    }
    .sidebar-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcff;
      padding: 12px;
    }
    .sidebar-heading {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }
    .sidebar-heading h2 {
      margin-bottom: 0;
    }
    .sidebar-tag {
      border-radius: 999px;
      padding: 2px 7px;
      background: #e8f5f2;
      color: var(--ok);
      font-size: 11px;
      white-space: nowrap;
    }
    main {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      overflow: hidden;
    }
    header {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
    }
    h1, h2 {
      margin: 0;
      letter-spacing: 0;
    }
    h1 { font-size: 20px; }
    h2 { font-size: 14px; margin-bottom: 10px; }
    .muted { color: var(--muted); }
    .thread {
      padding: 10px;
      background: var(--soft);
      border-radius: 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      word-break: break-all;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 9px 12px;
      cursor: pointer;
      font: inherit;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button:hover { border-color: var(--accent); }
    button.primary:hover { background: var(--accent-dark); }
    .actions {
      display: grid;
      gap: 8px;
    }
    #messages {
      overflow: auto;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .msg {
      max-width: min(760px, 88%);
      padding: 12px 14px;
      border-radius: 8px;
      overflow-wrap: anywhere;
    }
    .msg.user {
      align-self: flex-end;
      background: var(--accent);
      color: #fff;
    }
    .msg.assistant {
      align-self: flex-start;
      background: transparent;
      color: var(--ink);
      padding: 0;
      display: grid;
      gap: 8px;
    }
    .msg.user {
      white-space: pre-wrap;
    }
    .msg.assistant p {
      margin: 0 0 8px;
    }
    .msg.assistant p:last-child {
      margin-bottom: 0;
    }
    .msg.assistant ul, .msg.assistant ol {
      margin: 6px 0 8px 20px;
      padding: 0;
    }
    .msg.assistant li {
      margin: 3px 0;
    }
    .msg.assistant code {
      padding: 1px 4px;
      border-radius: 4px;
      background: #e5e7eb;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.92em;
    }
    .msg.assistant pre {
      margin: 8px 0;
      padding: 10px;
      background: #f8fafc;
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: auto;
      font-size: 12px;
      white-space: pre;
    }
    .msg.assistant blockquote {
      margin: 8px 0;
      padding-left: 10px;
      border-left: 3px solid #cbd5e1;
      color: var(--muted);
    }
    .msg.assistant .table-wrap {
      max-width: 100%;
      overflow: auto;
      margin: 8px 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }
    .msg.assistant table {
      width: 100%;
      min-width: 680px;
      border-collapse: collapse;
      font-size: 13px;
      white-space: nowrap;
    }
    .msg.assistant th,
    .msg.assistant td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    .msg.assistant th {
      background: #f8fafc;
      font-weight: 600;
      color: var(--ink);
    }
    .msg.assistant tr:last-child td {
      border-bottom: 0;
    }
    .assistant-output {
      background: #f2f5f9;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .assistant-section-title {
      margin-bottom: 8px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
    }
    .execution-steps {
      background: #fbfcff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      display: grid;
      gap: 10px;
    }
    .execution-title {
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
    }
    .execution-step {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      overflow: hidden;
    }
    .execution-step summary {
      cursor: pointer;
      padding: 8px 10px;
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      list-style: none;
    }
    .execution-step summary::-webkit-details-marker { display: none; }
    .execution-step summary::before {
      content: "›";
      color: var(--muted);
      transition: transform 0.15s ease;
    }
    .execution-step[open] summary::before {
      transform: rotate(90deg);
    }
    .execution-step-title {
      min-width: 0;
      font-size: 12px;
      font-weight: 600;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .execution-step-status {
      font-size: 11px;
      color: var(--muted);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 1px 7px;
      background: #f8fafc;
    }
    .execution-step[data-kind="tool"] {
      border-left: 3px solid #d97706;
    }
    .execution-step[data-kind="model"] {
      border-left: 3px solid var(--accent);
    }
    .execution-step[data-kind="react_step"] {
      border-left: 3px solid var(--accent);
    }
    .execution-step-body {
      padding: 0 10px 10px 28px;
      display: grid;
      gap: 8px;
    }
    .execution-step-summary {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .execution-step pre {
      max-height: 220px;
      margin: 0;
      overflow-x: auto;
      white-space: pre;
      overflow-wrap: normal;
    }
    .execution-stage {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
      padding: 8px;
      position: relative;
    }
    .execution-stage[data-kind="thought"] {
      border-left: 3px solid var(--accent);
    }
    .execution-stage[data-kind="action"] {
      border-left: 3px solid #d97706;
    }
    .execution-stage-flow {
      display: grid;
      gap: 6px;
      margin-top: 6px;
    }
    .execution-stage-row {
      display: grid;
      grid-template-columns: 86px minmax(0, 1fr);
      gap: 8px;
      font-size: 12px;
      align-items: start;
    }
    .execution-stage-key {
      color: var(--muted);
      white-space: nowrap;
    }
    .execution-stage-value {
      min-width: 0;
      color: var(--ink);
      overflow-wrap: anywhere;
    }
    .execution-stage-value code {
      background: #eef2f7;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 1px 4px;
      font-size: 11px;
    }
    .execution-status-completed {
      color: var(--ok);
    }
    .execution-status-error {
      color: var(--warn);
    }
    .execution-status-started {
      color: #475569;
    }
    .execution-stage-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      margin-bottom: 4px;
    }
    .execution-stage-title {
      min-width: 0;
      font-size: 12px;
      font-weight: 600;
    }
    .execution-stage-status {
      font-size: 11px;
      color: var(--muted);
    }
    .execution-stage-summary {
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .execution-stage details {
      margin-top: 4px;
    }
    .execution-stage summary {
      cursor: pointer;
      color: var(--accent);
      font-size: 12px;
      list-style: none;
      padding: 0;
      display: inline;
    }
    .execution-stage summary::-webkit-details-marker { display: none; }
    .execution-stage pre {
      max-height: 180px;
      margin-top: 6px;
      background: #fff;
      overflow-x: auto;
      white-space: pre;
      overflow-wrap: normal;
    }
    form {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      padding: 14px;
      border-top: 1px solid var(--line);
    }
    textarea {
      resize: none;
      min-height: 48px;
      max-height: 160px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 11px 12px;
      font: inherit;
    }
    textarea:focus {
      outline: 2px solid #bfdbfe;
      border-color: var(--accent);
    }
    pre {
      margin: 0;
      padding: 12px;
      background: #f8fafc;
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: auto;
      font-size: 12px;
    }
    .debug-card {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      margin-top: 12px;
      background: #fbfcff;
    }
    .debug-grid {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      font-size: 12px;
    }
    .debug-value {
      color: var(--ink);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      word-break: break-all;
    }
    .debug-list {
      display: grid;
      gap: 6px;
      margin-top: 8px;
      font-size: 12px;
    }
    .debug-note {
      border-left: 3px solid #94a3b8;
      border-radius: 5px;
      background: #f8fafc;
      padding: 7px 8px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .debug-note.warn {
      border-left-color: var(--warn);
      background: #fff7ed;
      color: #92400e;
    }
    .tool-list {
      display: grid;
      gap: 6px;
      margin-top: 8px;
    }
    .tool-pill {
      border-left: 3px solid #d97706;
      border-radius: 5px;
      background: #fff7ed;
      padding: 7px 8px;
      font-size: 12px;
      color: #92400e;
      overflow-wrap: anywhere;
    }
    .trace-link {
      display: block;
      text-align: center;
      margin-top: 12px;
      text-decoration: none;
      border: 1px solid var(--accent);
      border-radius: 6px;
      padding: 9px 12px;
      color: #fff;
      background: var(--accent);
    }
    .trace-link:hover { background: var(--accent-dark); }
    .empty-note {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    .status {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      background: #e8f5f2;
      color: var(--ok);
      font-size: 12px;
      margin-top: 8px;
    }
    .status.error {
      background: #fff7ed;
      color: var(--warn);
    }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; height: auto; min-height: 100vh; }
      main { min-height: 70vh; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">SP</div>
        <div>
          <h1>SkyPilot</h1>
          <div class="brand-subtitle">机票事实查询 Agent</div>
        </div>
      </div>

      <div class="sidebar-card">
        <div class="sidebar-heading">
          <h2>当前会话</h2>
          <span class="sidebar-tag">local</span>
        </div>
        <div class="thread" id="threadId">loading...</div>
      </div>

      <div class="sidebar-card">
        <div class="sidebar-heading">
          <h2>快捷操作</h2>
        </div>
        <div class="actions">
          <button class="primary" id="newBtn" type="button">新建会话</button>
          <button id="demoBtn" type="button">运行示例</button>
        </div>
      </div>
    </aside>
    <main>
      <header>
        <h1>航班与票价事实问答</h1>
        <div class="muted">输入自然语言问题，系统会按需调用日期、地点、票价和航班事实工具。</div>
      </header>
      <div id="messages"></div>
      <form id="chatForm">
        <textarea id="input" placeholder="例如：查询明天北京到上海的机票有哪些"></textarea>
        <button class="primary" id="sendBtn" type="submit">发送</button>
      </form>
    </main>
    <section>
      <h2>Debug Summary</h2>
      <div id="lastStatus" class="status">ready</div>
      <div class="debug-card">
        <h2>会话</h2>
        <div class="debug-grid"><span class="muted">Thread</span><span class="debug-value" id="debugThread">-</span></div>
        <div class="debug-grid"><span class="muted">Turns</span><span class="debug-value" id="debugTurns">0</span></div>
        <div class="debug-grid"><span class="muted">Events</span><span class="debug-value" id="debugEvents">0</span></div>
      </div>
      <div class="debug-card">
        <h2>模型上下文</h2>
        <div class="debug-grid"><span class="muted">Model</span><span class="debug-value" id="modelName">-</span></div>
        <div class="debug-grid"><span class="muted">Context</span><span class="debug-value" id="contextWindow">0</span></div>
        <div class="debug-grid"><span class="muted">Messages</span><span class="debug-value" id="lastMessageCount">0</span></div>
        <div class="debug-grid"><span class="muted">Prompt chars</span><span class="debug-value" id="estimatedPromptChars">0</span></div>
        <div class="debug-grid"><span class="muted">Response chars</span><span class="debug-value" id="estimatedResponseChars">0</span></div>
        <div class="debug-grid"><span class="muted">Context est.</span><span class="debug-value" id="contextUsageEstimate">-</span></div>
      </div>
      <div class="debug-card">
        <h2>执行概览</h2>
        <div class="debug-grid"><span class="muted">Model calls</span><span class="debug-value" id="modelCallCount">0</span></div>
        <div class="debug-grid"><span class="muted">Tool calls</span><span class="debug-value" id="toolCallCount">0</span></div>
        <div class="debug-grid"><span class="muted">Tool success</span><span class="debug-value" id="toolSuccessCount">0</span></div>
        <div class="debug-grid"><span class="muted">Tool errors</span><span class="debug-value" id="toolErrorCount">0</span></div>
      </div>
      <div class="debug-card">
        <h2>提醒</h2>
        <div class="debug-list" id="debugWarnings"></div>
      </div>
      <a class="trace-link" href="/trace" target="_blank" rel="noreferrer">打开完整 Trace</a>
    </section>
  </div>
  <script>
    const messagesEl = document.querySelector("#messages");
    const inputEl = document.querySelector("#input");
    const formEl = document.querySelector("#chatForm");
    const sendBtn = document.querySelector("#sendBtn");
    const demoBtn = document.querySelector("#demoBtn");
    const newBtn = document.querySelector("#newBtn");
    const threadEl = document.querySelector("#threadId");
    const statusEl = document.querySelector("#lastStatus");
    const debugThreadEl = document.querySelector("#debugThread");
    const debugTurnsEl = document.querySelector("#debugTurns");
    const debugEventsEl = document.querySelector("#debugEvents");
    const modelNameEl = document.querySelector("#modelName");
    const contextWindowEl = document.querySelector("#contextWindow");
    const lastMessageCountEl = document.querySelector("#lastMessageCount");
    const estimatedPromptCharsEl = document.querySelector("#estimatedPromptChars");
    const estimatedResponseCharsEl = document.querySelector("#estimatedResponseChars");
    const contextUsageEstimateEl = document.querySelector("#contextUsageEstimate");
    const modelCallCountEl = document.querySelector("#modelCallCount");
    const toolCallCountEl = document.querySelector("#toolCallCount");
    const toolSuccessCountEl = document.querySelector("#toolSuccessCount");
    const toolErrorCountEl = document.querySelector("#toolErrorCount");
    const debugWarningsEl = document.querySelector("#debugWarnings");
    const DEMO_PROMPT = "请查询北京到上海在 2026-07-10 的机票报价样本，并说明查到的事实、信息出处、查询时间和数据限制。";
    let isSending = false;
    let currentMessages = [];

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function renderInlineMarkdown(text) {
      let html = escapeHtml(text);
      html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
      html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
      html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
      return html;
    }

    function normalizeMarkdownTables(text) {
      return String(text || "")
        .replace(/\|\s+\|/g, "|\n|")
        .replace(/\|\s*(?=\|)/g, "|\n");
    }

    function isTableRow(line) {
      return /^\s*\|.*\|\s*$/.test(line);
    }

    function isTableDivider(line) {
      return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
    }

    function splitTableRow(line) {
      return line
        .trim()
        .replace(/^\|/, "")
        .replace(/\|$/, "")
        .split("|")
        .map((cell) => cell.trim());
    }

    function renderTable(lines) {
      const rows = lines.map(splitTableRow);
      const headers = rows[0] || [];
      const bodyRows = rows.slice(2);
      const headerHtml = headers
        .map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`)
        .join("");
      const bodyHtml = bodyRows
        .map((row) => `<tr>${row.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`)
        .join("");
      return `<div class="table-wrap"><table><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table></div>`;
    }

    function renderMarkdown(text) {
      const lines = normalizeMarkdownTables(text).split(/\r?\n/);
      const blocks = [];
      let paragraph = [];
      let listItems = [];
      let orderedList = false;
      let inCode = false;
      let codeLines = [];
      let codeLang = "";

      function flushParagraph() {
        if (!paragraph.length) return;
        blocks.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
        paragraph = [];
      }

      function flushList() {
        if (!listItems.length) return;
        const tag = orderedList ? "ol" : "ul";
        blocks.push(`<${tag}>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</${tag}>`);
        listItems = [];
      }

      for (let index = 0; index < lines.length; index += 1) {
        const line = lines[index];
        const codeFence = line.match(/^```(\w*)\s*$/);
        if (codeFence) {
          if (inCode) {
            blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
            inCode = false;
            codeLines = [];
            codeLang = "";
          } else {
            flushParagraph();
            flushList();
            inCode = true;
            codeLang = codeFence[1] || "";
          }
          continue;
        }
        if (inCode) {
          codeLines.push(line);
          continue;
        }
        if (!line.trim()) {
          flushParagraph();
          flushList();
          continue;
        }
        const heading = line.match(/^(#{1,3})\s+(.+)$/);
        if (heading) {
          flushParagraph();
          flushList();
          const level = heading[1].length + 2;
          blocks.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
          continue;
        }
        if (
          isTableRow(line)
          && index + 1 < lines.length
          && isTableDivider(lines[index + 1])
        ) {
          flushParagraph();
          flushList();
          const tableLines = [line, lines[index + 1]];
          index += 2;
          while (index < lines.length && isTableRow(lines[index])) {
            tableLines.push(lines[index]);
            index += 1;
          }
          index -= 1;
          blocks.push(renderTable(tableLines));
          continue;
        }
        const bullet = line.match(/^\s*[-*]\s+(.+)$/);
        const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
        if (bullet || ordered) {
          flushParagraph();
          const nextOrdered = Boolean(ordered);
          if (listItems.length && orderedList !== nextOrdered) flushList();
          orderedList = nextOrdered;
          listItems.push((bullet || ordered)[1]);
          continue;
        }
        const quote = line.match(/^>\s?(.+)$/);
        if (quote) {
          flushParagraph();
          flushList();
          blocks.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
          continue;
        }
        paragraph.push(line.trim());
      }

      if (inCode) {
        blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      }
      flushParagraph();
      flushList();
      return blocks.join("") || "";
    }

    function renderAssistantContent(content) {
      const raw = String(content || "");
      if (!raw.trim()) {
        return '<p class="empty-note">（空回复）</p>';
      }
      try {
        const rendered = renderMarkdown(raw);
        return rendered || `<p>${renderInlineMarkdown(raw)}</p>`;
      } catch (error) {
        console.error("Markdown render failed", error);
        return `<p>${renderInlineMarkdown(raw)}</p>`;
      }
    }

    function renderExecutionSteps(steps) {
      const items = Array.isArray(steps) ? steps : [];
      if (!items.length) return "";
      const renderedSteps = items.map((step, index) => {
        const title = step.title || `执行步骤 ${index + 1}`;
        const status = step.status || "info";
        const kind = step.kind || "step";
        const summary = step.summary || "";
        const stages = Array.isArray(step.stages) ? step.stages : [];
        const body = stages.length
          ? stages.map((stage) => renderReactStage(stage)).join("")
          : renderLegacyExecutionDetails(step);
        return `
          <details class="execution-step" data-kind="${escapeHtml(kind)}">
            <summary>
              <span class="execution-step-title">${escapeHtml(index + 1)}. ${escapeHtml(title)}</span>
              <span class="execution-step-status">${escapeHtml(status)}</span>
            </summary>
            <div class="execution-step-body">
              ${summary ? `<p class="execution-step-summary">${escapeHtml(summary)}</p>` : ""}
              ${body}
            </div>
          </details>
        `;
      }).join("");
      return `
        <div class="execution-steps">
          <div class="execution-title">执行过程</div>
          ${renderedSteps}
        </div>
      `;
    }

    function renderReactStage(stage) {
      const kind = stage.kind || "stage";
      const title = stage.title || "阶段";
      const status = stage.status || "info";
      const summary = stage.summary || "";
      const details = JSON.stringify(stage.details || {}, null, 2);
      return `
        <div class="execution-stage" data-kind="${escapeHtml(kind)}">
          <div class="execution-stage-head">
            <span class="execution-stage-title">${escapeHtml(title)}</span>
            <span class="execution-stage-status ${executionStatusClass(status)}">${escapeHtml(statusText(status))}</span>
          </div>
          ${summary ? `<p class="execution-stage-summary">${escapeHtml(summary)}</p>` : ""}
          ${renderStageFlow(stage)}
          <details>
            <summary>查看结构化详情</summary>
            <pre><code>${escapeHtml(details)}</code></pre>
          </details>
        </div>
      `;
    }

    function renderStageFlow(stage) {
      const details = stage && typeof stage.details === "object" && !Array.isArray(stage.details)
        ? stage.details
        : {};
      if (stage.kind === "thought") {
        return `
          <div class="execution-stage-flow">
            ${renderStageRow("上下文", `${formatNumber(details.message_count)} 条消息`)}
            ${renderStageRow("可用工具", `${formatNumber(details.tool_count)} 个`)}
            ${renderStageRow("响应类型", formatBlockTypes(details.response_block_types))}
            ${renderStageRow("文本摘要", formatModelResponsePreview(details.response_preview))}
            ${renderStageRow("工具请求", formatRequestedTools(details.requested_tools))}
          </div>
        `;
      }
      if (stage.kind === "action") {
        return `
          <div class="execution-stage-flow">
            ${renderStageRow("工具", `<code>${escapeHtml(details.tool_name || "tool")}</code>`)}
            ${renderStageRow("参数", formatArgumentKeys(details.argument_keys))}
            ${renderStageRow("结果", formatResponsePreview(details.response_preview))}
          </div>
        `;
      }
      return "";
    }

    function renderStageRow(label, valueHtml) {
      return `
        <div class="execution-stage-row">
          <span class="execution-stage-key">${escapeHtml(label)}</span>
          <span class="execution-stage-value">${valueHtml || "-"}</span>
        </div>
      `;
    }

    function formatArgumentKeys(keys) {
      const items = Array.isArray(keys) ? keys : [];
      if (!items.length) return "-";
      return items.map((key) => `<code>${escapeHtml(key)}</code>`).join(" ");
    }

    function formatBlockTypes(values) {
      const items = Array.isArray(values) ? values : [];
      if (!items.length) return "未知响应";
      return items.map((value) => `<code>${escapeHtml(readableBlockType(value))}</code>`).join(" ");
    }

    function formatModelResponsePreview(value) {
      const raw = String(value || "").trim();
      if (!raw) return "无可展示文本";
      return escapeHtml(raw.length > 180 ? `${raw.slice(0, 180)}...` : raw);
    }

    function formatRequestedTools(tools) {
      const items = Array.isArray(tools) ? tools : [];
      if (!items.length) return "未请求工具";
      return items.map((tool) => {
        const name = tool && tool.name ? String(tool.name) : "tool";
        const keys = tool && Array.isArray(tool.argument_keys) ? tool.argument_keys : [];
        const suffix = keys.length ? ` · ${keys.join(", ")}` : "";
        return `<code>${escapeHtml(name + suffix)}</code>`;
      }).join(" ");
    }

    function readableBlockType(value) {
      const mapping = {
        reasoning: "内部推理标记",
        text: "文本回复",
        function_call: "工具调用请求",
        tool_call: "工具调用请求",
        message: "消息",
        ai: "模型消息",
        assistant: "模型消息",
      };
      const key = String(value || "");
      return mapping[key] || key || "未知响应";
    }

    function formatResponsePreview(value) {
      const raw = String(value || "").trim();
      if (!raw) return "暂无返回内容";
      try {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          const facts = [];
          if (Array.isArray(parsed.quotes)) facts.push(`${parsed.quotes.length} 条报价`);
          if (Array.isArray(parsed.items)) facts.push(`${parsed.items.length} 个地点`);
          if (Array.isArray(parsed.flight_records)) facts.push(`${parsed.flight_records.length} 条航班记录`);
          if (Array.isArray(parsed.relay_quotes)) facts.push(`${parsed.relay_quotes.length} 条参考报价`);
          if (parsed.target_date) facts.push(`目标日期 ${parsed.target_date}`);
          if (facts.length) return escapeHtml(facts.join("，"));
        }
      } catch (error) {
        // Keep the raw preview below when tool output is not JSON.
      }
      return escapeHtml(raw.length > 160 ? `${raw.slice(0, 160)}...` : raw);
    }

    function statusText(status) {
      if (status === "completed") return "完成";
      if (status === "error") return "失败";
      if (status === "started") return "执行中";
      return String(status || "info");
    }

    function executionStatusClass(status) {
      if (status === "completed") return "execution-status-completed";
      if (status === "error") return "execution-status-error";
      if (status === "started") return "execution-status-started";
      return "";
    }

    function renderLegacyExecutionDetails(step) {
      const details = JSON.stringify(step.details || {}, null, 2);
      return `<pre><code>${escapeHtml(details)}</code></pre>`;
    }

    function renderMessages(messages) {
      currentMessages = (messages || []).map((message) => ({
        role: String(message.role || ""),
        content: String(message.content || ""),
        execution_steps: Array.isArray(message.execution_steps) ? message.execution_steps : [],
      }));
      messagesEl.innerHTML = "";
      for (const message of currentMessages) {
        const el = document.createElement("div");
        el.className = `msg ${message.role}`;
        if (message.role === "assistant") {
          el.innerHTML = renderExecutionSteps(message.execution_steps)
            + `
              <div class="assistant-output">
                <div class="assistant-section-title">最终输出</div>
                <div class="assistant-answer">${renderAssistantContent(message.content)}</div>
              </div>
            `;
        } else {
          el.textContent = message.content;
        }
        messagesEl.appendChild(el);
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function renderDebug(payload) {
      threadEl.textContent = payload.thread_id || threadEl.textContent;
      statusEl.textContent = payload.status || "ready";
      statusEl.className = payload.status === "error" ? "status error" : "status";
      const trace = payload.trace || {};
      const summary = payload.debug_summary || {};
      const session = summary.session || {};
      const model = summary.model || {};
      const execution = summary.execution || {};
      debugThreadEl.textContent = payload.thread_id || session.thread_id || trace.thread_id || "-";
      debugTurnsEl.textContent = String(session.turn_count ?? trace.turn_count ?? 0);
      debugEventsEl.textContent = String(session.event_count ?? trace.event_count ?? 0);
      modelNameEl.textContent = model.model_name || "-";
      contextWindowEl.textContent = formatNumber(model.context_window_tokens);
      lastMessageCountEl.textContent = formatNumber(model.last_message_count);
      estimatedPromptCharsEl.textContent = formatNumber(model.estimated_prompt_chars);
      estimatedResponseCharsEl.textContent = formatNumber(model.estimated_response_chars);
      contextUsageEstimateEl.textContent = formatPercent(model.context_usage_estimate);
      modelCallCountEl.textContent = formatNumber(execution.model_call_count);
      toolCallCountEl.textContent = formatNumber(execution.tool_call_count);
      toolSuccessCountEl.textContent = formatNumber(execution.tool_success_count);
      toolErrorCountEl.textContent = formatNumber(execution.tool_error_count);
      renderDebugNotes(debugWarningsEl, summary.warnings || [], "暂无提醒", "warn");
    }

    function formatNumber(value) {
      const number = Number(value || 0);
      return Number.isFinite(number) ? number.toLocaleString("zh-CN") : "0";
    }

    function formatPercent(value) {
      if (value === null || value === undefined || value === "") return "-";
      const number = Number(value);
      if (!Number.isFinite(number)) return "-";
      return `${(number * 100).toFixed(1)}%`;
    }

    function renderDebugNotes(container, values, emptyText, className = "") {
      container.innerHTML = "";
      const items = Array.isArray(values) ? values : [];
      if (!items.length) {
        const empty = document.createElement("div");
        empty.className = `debug-note ${className}`.trim();
        empty.textContent = emptyText;
        container.appendChild(empty);
        return;
      }
      for (const value of items.slice(-4)) {
        const item = document.createElement("div");
        item.className = `debug-note ${className}`.trim();
        item.textContent = String(value);
        container.appendChild(item);
      }
    }

    async function postJson(path, payload = {}) {
      const res = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      return await res.json();
    }

    function messagesFromPayload(payload, pendingMessages) {
      const answer = String(payload.answer || "");
      if (Array.isArray(payload.messages) && payload.messages.length) {
        const messages = payload.messages.map((message) => ({
          role: String(message.role || ""),
          content: String(message.content || ""),
          execution_steps: Array.isArray(message.execution_steps) ? message.execution_steps : [],
        }));
        const last = messages[messages.length - 1];
        if (
          last
          && last.role === "assistant"
          && !last.content.trim()
          && answer.trim()
        ) {
          return messages.slice(0, -1).concat([
            {
              role: "assistant",
              content: answer,
              execution_steps: Array.isArray(payload.execution_steps)
                ? payload.execution_steps
                : [],
            },
          ]);
        }
        return messages;
      }
      if (answer.trim()) {
        return pendingMessages.slice(0, -1).concat([
          {
            role: "assistant",
            content: answer,
            execution_steps: Array.isArray(payload.execution_steps)
              ? payload.execution_steps
              : [],
          },
        ]);
      }
      return pendingMessages.slice(0, -1).concat([
        {role: "assistant", content: "未获取到可展示的助手回复。"},
      ]);
    }

    async function loadState() {
      const res = await fetch("/api/state");
      const payload = await res.json();
      threadEl.textContent = payload.thread_id;
      renderMessages(payload.messages || []);
      renderDebug({
        thread_id: payload.thread_id,
        status: "ready",
        trace: payload.trace,
        debug_summary: payload.debug_summary,
      });
    }

    async function sendMessage(message) {
      if (!message.trim()) return;
      if (isSending) return;
      setSending(true);
      inputEl.value = "";
      let pendingMessages = currentMessages;
      try {
        pendingMessages = currentMessages.concat([
          {role: "user", content: message},
          {role: "assistant", content: "查询中..."},
        ]);
        renderMessages(pendingMessages);
        const payload = await postJson("/api/chat", {message});
        renderMessages(messagesFromPayload(payload, pendingMessages));
        renderDebug(payload);
      } catch (error) {
        renderMessages(pendingMessages.slice(0, -1).concat([
          {role: "assistant", content: `请求失败：${String(error)}`},
        ]));
      } finally {
        setSending(false);
      }
    }

    function setSending(value) {
      isSending = value;
      sendBtn.disabled = value;
      demoBtn.disabled = value;
      newBtn.disabled = value;
      sendBtn.textContent = value ? "查询中..." : "发送";
    }

    formEl.addEventListener("submit", async (event) => {
      event.preventDefault();
      await sendMessage(inputEl.value);
    });

    newBtn.addEventListener("click", async () => {
      if (isSending) return;
      const payload = await postJson("/api/new");
      threadEl.textContent = payload.thread_id;
      renderMessages([]);
      renderDebug({
        thread_id: payload.thread_id,
        status: "ready",
        trace: payload.trace,
        debug_summary: payload.debug_summary,
      });
    });

    demoBtn.addEventListener("click", async () => {
      if (isSending) return;
      inputEl.value = DEMO_PROMPT;
      inputEl.focus();
    });

    loadState();
  </script>
</body>
</html>
"""


TRACE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SkyPilot Debug Trace</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7fb;
      --panel: #ffffff;
      --ink: #18202f;
      --muted: #697386;
      --line: #dfe3eb;
      --accent: #2563eb;
      --accent-dark: #1e40af;
      --soft: #eef4ff;
      --session: #1d4ed8;
      --turn: #0f766e;
      --conversation: #7c3aed;
      --agent: #b45309;
      --model: #be123c;
      --tool: #0369a1;
      --event: #4b5563;
      --react-input: #0f766e;
      --react-step: #7c3aed;
      --react-thought: #be123c;
      --react-action: #0369a1;
      --react-final: #1d4ed8;
      --json-key: #1d4ed8;
      --json-string: #047857;
      --json-number: #b45309;
      --json-boolean: #7c3aed;
      --json-null: #64748b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .page {
      width: min(1280px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 18px 0 28px;
    }
    header, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: start;
      padding: 18px;
      margin-bottom: 16px;
    }
    h1, h2 {
      margin: 0;
      letter-spacing: 0;
    }
    h1 { font-size: 21px; }
    h2 { font-size: 15px; margin-bottom: 10px; }
    .muted { color: var(--muted); }
    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    button, a.button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 9px 12px;
      cursor: pointer;
      font: inherit;
      text-decoration: none;
      line-height: 1.2;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button:hover, a.button:hover { border-color: var(--accent); }
    button.primary:hover { background: var(--accent-dark); }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .metric-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .metric-value {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      overflow-wrap: anywhere;
    }
    .content {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(340px, 0.8fr);
      gap: 16px;
    }
    .panel {
      min-height: 0;
      padding: 16px;
    }
    .tree-panel {
      max-height: calc(100vh - 196px);
      overflow: auto;
    }
    .raw-panel {
      max-height: calc(100vh - 196px);
      overflow: auto;
    }
    .trace-tree {
      display: grid;
      gap: 8px;
    }
    .empty {
      color: var(--muted);
      padding: 14px;
      border: 1px dashed var(--line);
      border-radius: 6px;
      background: #fbfcff;
    }
    details.trace-node {
      border: 1px solid var(--line);
      border-left: 5px solid var(--event);
      border-radius: 7px;
      background: #fff;
      overflow: hidden;
    }
    details.trace-node[data-type="session"] { border-left-color: var(--session); }
    details.trace-node[data-type="turn"] { border-left-color: var(--turn); }
    details.trace-node[data-type="conversation"] { border-left-color: var(--conversation); }
    details.trace-node[data-type="agent_run"] { border-left-color: var(--agent); }
    details.trace-node[data-type="model"] { border-left-color: var(--model); }
    details.trace-node[data-type="tool"] { border-left-color: var(--tool); }
    details.trace-node[data-type="react_input"] { border-left-color: var(--react-input); }
    details.trace-node[data-type="react_agent"] { border-left-color: var(--agent); }
    details.trace-node[data-type="react_step"] { border-left-color: var(--react-step); }
    details.trace-node[data-type="react_thought"] { border-left-color: var(--react-thought); }
    details.trace-node[data-type="react_action"] { border-left-color: var(--react-action); }
    details.trace-node[data-type="react_final"] { border-left-color: var(--react-final); }
    details.trace-node summary {
      display: grid;
      grid-template-columns: auto auto minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      list-style: none;
      padding: 9px 10px;
      cursor: pointer;
    }
    details.trace-node summary::-webkit-details-marker { display: none; }
    .node-caret {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      width: 10px;
    }
    details[open] > summary .node-caret { transform: rotate(90deg); }
    .node-type {
      border-radius: 999px;
      padding: 2px 7px;
      color: #fff;
      background: var(--event);
      font-size: 11px;
      text-transform: uppercase;
    }
    details[data-type="session"] > summary .node-type { background: var(--session); }
    details[data-type="turn"] > summary .node-type { background: var(--turn); }
    details[data-type="conversation"] > summary .node-type { background: var(--conversation); }
    details[data-type="agent_run"] > summary .node-type { background: var(--agent); }
    details[data-type="model"] > summary .node-type { background: var(--model); }
    details[data-type="tool"] > summary .node-type { background: var(--tool); }
    details[data-type="react_input"] > summary .node-type { background: var(--react-input); }
    details[data-type="react_agent"] > summary .node-type { background: var(--agent); }
    details[data-type="react_step"] > summary .node-type { background: var(--react-step); }
    details[data-type="react_thought"] > summary .node-type { background: var(--react-thought); }
    details[data-type="react_action"] > summary .node-type { background: var(--react-action); }
    details[data-type="react_final"] > summary .node-type { background: var(--react-final); }
    .node-label {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      min-width: 0;
    }
    .node-status {
      color: var(--muted);
      font-size: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .node-body {
      border-top: 1px solid var(--line);
      padding: 10px;
      background: #fbfcff;
    }
    .node-meta {
      margin: 0;
      padding: 10px;
      background: #f8fafc;
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: auto;
      font-size: 12px;
    }
    .node-children {
      display: grid;
      gap: 8px;
      margin-top: 8px;
    }
    pre {
      margin: 0;
      padding: 12px;
      background: #f8fafc;
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: auto;
      font-size: 12px;
      max-height: calc(100vh - 260px);
    }
    .json-key { color: var(--json-key); }
    .json-string { color: var(--json-string); }
    .json-number { color: var(--json-number); }
    .json-boolean { color: var(--json-boolean); }
    .json-null { color: var(--json-null); font-style: italic; }
    @media (max-width: 980px) {
      header { grid-template-columns: 1fr; }
      .toolbar { justify-content: flex-start; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .content { grid-template-columns: 1fr; }
      .tree-panel, .raw-panel { max-height: none; }
    }
  </style>
</head>
<body>
  <div class="page">
    <header>
      <div>
        <h1>Debug Trace</h1>
        <div class="muted">独立展示完整多轮对话链路，与主聊天页面通过同一个会话状态同步。</div>
      </div>
      <div class="toolbar">
        <a class="button" href="/">返回聊天</a>
        <button id="refreshTraceBtn" type="button">刷新</button>
        <button id="expandTraceBtn" type="button">全部展开</button>
        <button id="collapseTraceBtn" type="button">全部收起</button>
      </div>
    </header>

    <div class="summary">
      <div class="metric">
        <div class="metric-label">Thread</div>
        <div class="metric-value" id="threadId">-</div>
      </div>
      <div class="metric">
        <div class="metric-label">Turns</div>
        <div class="metric-value" id="turnCount">0</div>
      </div>
      <div class="metric">
        <div class="metric-label">Events</div>
        <div class="metric-value" id="eventCount">0</div>
      </div>
      <div class="metric">
        <div class="metric-label">Status</div>
        <div class="metric-value" id="traceStatus">ready</div>
      </div>
    </div>

    <div class="content">
      <section class="panel tree-panel">
        <h2>Trace Tree</h2>
        <div id="traceTree" class="trace-tree"></div>
      </section>
      <section class="panel raw-panel">
        <h2>Raw JSON</h2>
        <pre id="rawTrace">{}</pre>
      </section>
    </div>
  </div>

  <script>
    const traceTreeEl = document.querySelector("#traceTree");
    const rawTraceEl = document.querySelector("#rawTrace");
    const treePanelEl = document.querySelector(".tree-panel");
    const rawPanelEl = document.querySelector(".raw-panel");
    const threadIdEl = document.querySelector("#threadId");
    const turnCountEl = document.querySelector("#turnCount");
    const eventCountEl = document.querySelector("#eventCount");
    const traceStatusEl = document.querySelector("#traceStatus");
    const refreshTraceBtn = document.querySelector("#refreshTraceBtn");
    const expandTraceBtn = document.querySelector("#expandTraceBtn");
    const collapseTraceBtn = document.querySelector("#collapseTraceBtn");
    const traceOpenState = new Map();
    const metaScrollState = new Map();
    let traceOpenStateInitialized = false;
    let currentTraceTreeSignature = "";
    let currentRawTraceSignature = "";
    let currentTraceRevision = "";

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function safeJson(value) {
      return JSON.stringify(value || {}, null, 2);
    }

    function highlightedJson(value) {
      return escapeHtml(safeJson(value)).replace(
        /("(?:\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\btrue\b|\bfalse\b|\bnull\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
        (match) => {
          if (match.endsWith(":")) return `<span class="json-key">${match}</span>`;
          if (match.startsWith('"')) return `<span class="json-string">${match}</span>`;
          if (match === "true" || match === "false") return `<span class="json-boolean">${match}</span>`;
          if (match === "null") return `<span class="json-null">${match}</span>`;
          return `<span class="json-number">${match}</span>`;
        }
      );
    }

    function captureTraceOpenState() {
      for (const item of document.querySelectorAll("details.trace-node")) {
        if (item.dataset.nodeKey) {
          traceOpenState.set(item.dataset.nodeKey, item.open);
        }
      }
    }

    function captureScrollState() {
      for (const item of document.querySelectorAll(".node-meta")) {
        if (item.dataset.metaKey) {
          metaScrollState.set(item.dataset.metaKey, {
            top: item.scrollTop,
            left: item.scrollLeft,
          });
        }
      }
      return {
        pageX: window.scrollX,
        pageY: window.scrollY,
        treeTop: treePanelEl.scrollTop,
        treeLeft: treePanelEl.scrollLeft,
        rawPanelTop: rawPanelEl.scrollTop,
        rawPanelLeft: rawPanelEl.scrollLeft,
        rawTop: rawTraceEl.scrollTop,
        rawLeft: rawTraceEl.scrollLeft,
      };
    }

    function restoreScrollState(state) {
      treePanelEl.scrollTop = state.treeTop;
      treePanelEl.scrollLeft = state.treeLeft;
      rawPanelEl.scrollTop = state.rawPanelTop;
      rawPanelEl.scrollLeft = state.rawPanelLeft;
      rawTraceEl.scrollTop = state.rawTop;
      rawTraceEl.scrollLeft = state.rawLeft;
      for (const item of document.querySelectorAll(".node-meta")) {
        const metaState = metaScrollState.get(item.dataset.metaKey);
        if (metaState) {
          item.scrollTop = metaState.top;
          item.scrollLeft = metaState.left;
        }
      }
      window.scrollTo(state.pageX, state.pageY);
    }

    function traceTreeSignature(tree) {
      if (!tree) return "";
      const children = Array.isArray(tree.children) ? tree.children : [];
      return [
        tree.id || "",
        tree.type || "",
        tree.label || "",
        tree.status || "",
        children.length,
        children.map((child) => traceTreeSignature(child)).join("|"),
      ].join(":");
    }

    function renderRawTrace(trace, revision, options = {}) {
      const rawSignature = revision || `${trace.thread_id || ""}:${trace.turn_count || 0}:${trace.event_count || 0}`;
      if (!options.forceRaw && rawSignature === currentRawTraceSignature) return;
      rawTraceEl.innerHTML = highlightedJson(trace);
      currentRawTraceSignature = rawSignature;
    }

    function renderTrace(payload, options = {}) {
      if (payload.status === "not_modified") {
        traceStatusEl.textContent = "ready";
        currentTraceRevision = payload.trace_revision || currentTraceRevision;
        return;
      }
      const scrollState = captureScrollState();
      const trace = payload.trace || {};
      const nextTreeSignature = traceTreeSignature(trace.tree || null);
      currentTraceRevision = payload.trace_revision || currentTraceRevision;
      threadIdEl.textContent = trace.thread_id || payload.thread_id || "-";
      turnCountEl.textContent = String(trace.turn_count || 0);
      eventCountEl.textContent = String(trace.event_count || 0);
      traceStatusEl.textContent = "ready";
      renderRawTrace(trace, payload.trace_revision, options);
      if (options.forceTree || nextTreeSignature !== currentTraceTreeSignature) {
        renderTraceTree(trace.tree);
        currentTraceTreeSignature = nextTreeSignature;
      }
      restoreScrollState(scrollState);
      window.requestAnimationFrame(() => restoreScrollState(scrollState));
    }

    function renderTraceTree(tree) {
      captureTraceOpenState();
      traceTreeEl.innerHTML = "";
      if (!tree) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "暂无 trace。发起一次对话后这里会展示完整链路。";
        traceTreeEl.appendChild(empty);
        return;
      }
      traceTreeEl.appendChild(traceNodeElement(tree, 0, "0"));
      traceOpenStateInitialized = true;
    }

    function traceNodeKey(node, path) {
      return node.id || `${path}:${node.type || "event"}:${node.label || ""}`;
    }

    function traceNodeElement(node, depth, path) {
      const nodeKey = traceNodeKey(node, path);
      const details = document.createElement("details");
      details.className = "trace-node";
      details.dataset.type = node.type || "event";
      details.dataset.nodeKey = nodeKey;
      details.open = traceOpenStateInitialized && traceOpenState.has(nodeKey)
        ? traceOpenState.get(nodeKey)
        : depth < 2;
      details.addEventListener("toggle", () => {
        traceOpenState.set(nodeKey, details.open);
      });

      const summary = document.createElement("summary");
      summary.style.paddingLeft = `${10 + depth * 12}px`;

      const caret = document.createElement("span");
      caret.className = "node-caret";
      caret.textContent = "›";

      const type = document.createElement("span");
      type.className = "node-type";
      type.textContent = node.type || "event";

      const label = document.createElement("span");
      label.className = "node-label";
      label.textContent = node.label || node.id || "(unnamed)";

      const status = document.createElement("span");
      status.className = "node-status";
      status.textContent = node.status || "";

      summary.append(caret, type, label, status);
      details.appendChild(summary);

      const body = document.createElement("div");
      body.className = "node-body";

      if (node.meta && Object.keys(node.meta).length) {
        const meta = document.createElement("pre");
        meta.className = "node-meta";
        meta.dataset.metaKey = nodeKey;
        meta.innerHTML = highlightedJson(node.meta);
        body.appendChild(meta);
      }

      const children = node.children || [];
      if (children.length) {
        const childWrap = document.createElement("div");
        childWrap.className = "node-children";
        children.forEach((child, index) => {
          childWrap.appendChild(traceNodeElement(child, depth + 1, `${path}.${index}`));
        });
        body.appendChild(childWrap);
      }

      if (!body.childNodes.length) {
        const empty = document.createElement("div");
        empty.className = "muted";
        empty.textContent = "无子节点";
        body.appendChild(empty);
      }

      details.appendChild(body);
      return details;
    }

    function setTraceOpen(open) {
      for (const item of document.querySelectorAll("details.trace-node")) {
        item.open = open;
        if (item.dataset.nodeKey) {
          traceOpenState.set(item.dataset.nodeKey, open);
        }
      }
      traceOpenStateInitialized = true;
    }

    async function loadTrace(options = {}) {
      traceStatusEl.textContent = "loading";
      try {
        const res = await fetch(`/api/trace-state?revision=${encodeURIComponent(currentTraceRevision)}`);
        const payload = await res.json();
        renderTrace(payload, options);
      } catch (error) {
        traceStatusEl.textContent = "error";
        rawTraceEl.textContent = String(error);
      }
    }

    refreshTraceBtn.addEventListener("click", () => loadTrace({forceTree: true}));
    expandTraceBtn.addEventListener("click", () => setTraceOpen(true));
    collapseTraceBtn.addEventListener("click", () => setTraceOpen(false));

    loadTrace();
    window.setInterval(() => loadTrace(), 3000);
  </script>
</body>
</html>
"""


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEMO_MESSAGE",
    "TRACE_HTML",
    "WebApp",
    "build_handler",
    "run_web_ui",
]
