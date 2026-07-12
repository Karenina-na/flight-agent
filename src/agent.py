from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware, ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI

from src.config import load_settings
from src.guardrails import (
    build_agent_state_compaction_middleware,
    build_param_aware_duplicate_tool_call_guard,
)
from src.memory import build_checkpointer, build_memory_middleware, build_store
from src.observability import configure_logging, build_observability_middleware
from src.prompt import build_system_prompt
from src.runtime import Context
from src.skills import build_skill_middleware
from src.summarization import build_summary_model
from src.tools import get_tools

settings = load_settings()
configure_logging(settings.observability.logging)

model = ChatOpenAI(
    base_url=settings.llm.base_url,
    api_key=settings.llm.api_key,
    model=settings.llm.model,
    temperature=settings.llm.temperature,
    output_version="responses/v1",
    use_responses_api=True,
    profile={"max_input_tokens": settings.llm.context_window_tokens},
    timeout=settings.llm.timeout_seconds,
    max_retries=settings.llm.max_retries,
)
summary_model = build_summary_model(settings.llm, settings.summarization)
summary_cache_max_items = (
    settings.summarization.cache_max_items
    if settings.summarization.cache_enabled
    else 0
)

tools = get_tools()
checkpointer = build_checkpointer(settings.memory.checkpointer)
store = build_store(settings.memory.store)
middleware = [
    build_skill_middleware(skills_root=Path("skills")),
    build_memory_middleware(),
    TodoListMiddleware(),
    build_agent_state_compaction_middleware(
        context_window_tokens=settings.llm.context_window_tokens,
        summary_model=summary_model,
        semantic_enabled=settings.summarization.enabled,
        summary_cache_max_items=summary_cache_max_items,
    ),
    build_observability_middleware(redact=settings.observability.logging.redact),
    build_param_aware_duplicate_tool_call_guard(loop_stop_after=3),
    ToolCallLimitMiddleware(run_limit=64, exit_behavior="end"),
]

agent = create_agent(
    model=model,
    system_prompt=build_system_prompt(tools=tools),
    tools=tools,
    context_schema=Context,
    middleware=middleware,
    checkpointer=checkpointer,
    store=store,
)
