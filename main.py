"""Interactive CLI entrypoint for the air ticket agent."""

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import uuid4

from langchain.messages import HumanMessage
from langchain_core.messages import AIMessageChunk

from src.agent import agent
from src.observability import full_text_trace_fields, log_event, observe_agent_stream
from src.runtime import Context, build_default_context
from src.tools import get_tools

APP_USER_ID = "local-cli"
DEFAULT_WORKSPACE_ID = "local-cli"
HELP_TEXT = """可用命令：
/help   显示帮助
/new    开始一个新会话
/tools  查看当前已注册工具
/demo   运行一条机票报价示例
/exit   退出

直接输入问题即可调用机票事实查询 agent，例如：
北京到上海 2026-07-10 的机票大概多少钱？
CA981 这个航班有什么信息？"""
DEMO_MESSAGE = (
    "请查询北京到上海在 2026-07-10 的机票报价样本，"
    "并说明查到的事实、数据来源、查询时间和数据限制。"
)


@dataclass
class CliSession:
    """Current CLI conversation session."""

    thread_id: str

    @classmethod
    def new(cls) -> "CliSession":
        return cls(thread_id=_new_thread_id())

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def context(self) -> Context:
        return build_default_context(
            user_id=APP_USER_ID,
            thread_id=self.thread_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            metadata={"entrypoint": "main.py"},
        )


def run_cli() -> None:
    """Run the interactive command-line agent shell."""
    session = CliSession.new()
    print("机票事实查询 Agent CLI。输入 /help 查看命令。")
    print(f"当前会话: {session.thread_id}")

    while True:
        try:
            raw_input = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        message = raw_input.strip()
        if not message:
            continue

        command_result = handle_command(message, session)
        if command_result == "exit":
            print("再见。")
            return
        if isinstance(command_result, CliSession):
            session = command_result
            print(f"已开始新会话: {session.thread_id}")
            continue
        if isinstance(command_result, str):
            print(command_result)
            continue

        stream_agent_response(message, session)


def handle_command(message: str, session: CliSession) -> str | CliSession | None:
    """Handle slash commands; return None when input should go to the agent."""
    command = message.lower()
    if command in {"/exit", "/quit", "exit", "quit"}:
        return "exit"
    if command == "/help":
        return HELP_TEXT
    if command == "/new":
        return CliSession.new()
    if command == "/tools":
        return format_tools()
    if command == "/demo":
        stream_agent_response(DEMO_MESSAGE, session)
        return ""
    if message.startswith("/"):
        return f"未知命令: {message}\n输入 /help 查看可用命令。"
    return None


def stream_agent_response(message: str, session: CliSession) -> None:
    """Stream one user message through the agent and print a compact transcript."""
    context = session.context()
    answer_started = False
    reasoning_started = False
    saw_reasoning = False
    saw_reasoning_block = False
    assistant_parts: list[str] = []
    assistant_chunk_count = 0

    log_event(
        "conversation_turn_start",
        context=context,
        redact=False,
        entrypoint="main.stream_agent_response",
        **full_text_trace_fields("user_input", message),
    )

    try:
        for message_chunk, metadata in observe_agent_stream(
            agent.stream(
                {"messages": [HumanMessage(content=message)]},
                config=session.config,
                context=context,
                stream_mode="messages",
            ),
            context,
            entrypoint="main.stream_agent_response",
            stream_mode="messages",
        ):
            if not _is_assistant_chunk(message_chunk):
                continue

            saw_reasoning_block = saw_reasoning_block or _has_reasoning_block(
                message_chunk
            )
            reasoning = _reasoning_text(message_chunk)
            if reasoning:
                if not reasoning_started:
                    print("思考流:")
                    reasoning_started = True
                print(reasoning, end="", flush=True)
                saw_reasoning = True

            content = _message_text(message_chunk)
            if not content:
                continue

            assistant_parts.append(content)
            assistant_chunk_count += 1

            if not answer_started:
                if saw_reasoning:
                    print("\n")
                elif saw_reasoning_block:
                    print("思考流: 检测到 reasoning block，但未暴露文本。\n")
                print("回复:")
                answer_started = True

            print(content, end="", flush=True)
    except Exception as exc:
        partial_output = "".join(assistant_parts)
        log_event(
            "conversation_turn_error",
            context=context,
            redact=False,
            entrypoint="main.stream_agent_response",
            assistant_chunk_count=assistant_chunk_count,
            reasoning_block_seen=saw_reasoning_block,
            reasoning_text_seen=saw_reasoning,
            error_type=type(exc).__name__,
            **full_text_trace_fields("partial_assistant_output", partial_output),
        )
        raise

    if answer_started:
        print()

    assistant_output = "".join(assistant_parts)
    log_event(
        "conversation_turn_end",
        context=context,
        redact=False,
        entrypoint="main.stream_agent_response",
        assistant_chunk_count=assistant_chunk_count,
        reasoning_block_seen=saw_reasoning_block,
        reasoning_text_seen=saw_reasoning,
        answer_started=answer_started,
        **full_text_trace_fields("assistant_output", assistant_output),
    )


def format_tools() -> str:
    """Render registered tools for CLI display."""
    lines = ["当前已注册工具："]
    for tool in get_tools():
        description = str(getattr(tool, "description", "")).strip()
        lines.append(f"- {tool.name}: {description}")
    return "\n".join(lines)


def _is_assistant_chunk(message_chunk: object) -> bool:
    """Return True for assistant message chunks that are safe to print."""
    return isinstance(message_chunk, AIMessageChunk)


def _message_text(message_chunk: object) -> str:
    content = getattr(message_chunk, "content", message_chunk)
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in _content_blocks(message_chunk):
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))

    if parts:
        return "".join(parts)

    return _chunk_text(content)


def _chunk_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _reasoning_text(message_chunk: object) -> str:
    """Extract reasoning text from LangChain standard content blocks."""
    block_parts: list[str] = []
    for block in _content_blocks(message_chunk):
        if isinstance(block, dict):
            block_parts.extend(_reasoning_from_mapping(block))

    return "".join(block_parts)


def _content_blocks(message_chunk: object) -> list[object]:
    content_blocks = getattr(message_chunk, "content_blocks", None)
    if isinstance(content_blocks, list):
        return content_blocks

    content = getattr(message_chunk, "content", None)
    if isinstance(content, list):
        return content

    return []


def _has_reasoning_block(message_chunk: object) -> bool:
    return any(
        isinstance(block, dict) and block.get("type") == "reasoning"
        for block in _content_blocks(message_chunk)
    )


def _reasoning_from_mapping(mapping: dict) -> list[str]:
    if mapping.get("type") != "reasoning":
        return []

    parts: list[str] = []
    for key in ("reasoning", "text", "content", "summary", "details"):
        parts.extend(_reasoning_from_value(mapping.get(key)))

    return parts


def _reasoning_from_value(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.extend(_reasoning_from_mapping(item))
            else:
                parts.extend(_reasoning_from_value(item))
        return parts
    if isinstance(value, dict):
        return _reasoning_from_mapping(value)
    return []


def _new_thread_id() -> str:
    return f"cli-{uuid4().hex[:8]}"


if __name__ == "__main__":
    run_cli()
