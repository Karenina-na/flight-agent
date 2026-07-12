from langchain.agents.middleware import TodoListMiddleware, ToolCallLimitMiddleware

from src.agent import middleware, model, settings, store
from src.guardrails import (
    AgentStateCompactionMiddleware,
    ParamAwareDuplicateToolCallGuard,
)
from src.memory import MemoryMiddleware
from src.observability import ObservabilityMiddleware
from src.skills import SkillMiddleware


def test_agent_middleware_is_flat_and_includes_observability_skills_and_memory():
    assert len(middleware) == 7
    assert isinstance(middleware[0], SkillMiddleware)
    assert isinstance(middleware[1], MemoryMiddleware)
    assert isinstance(middleware[2], TodoListMiddleware)
    assert isinstance(middleware[3], AgentStateCompactionMiddleware)
    assert middleware[3].summary_model is not None
    assert middleware[3].semantic_enabled is True
    assert middleware[3].summary_model.request_timeout == settings.summarization.timeout_seconds
    assert middleware[3].summary_model.max_tokens is None
    assert middleware[3].summary_model.max_retries == 0
    assert middleware[3].summary_model.temperature == 0
    assert middleware[3].summary_model.extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    expected_cache_items = (
        settings.summarization.cache_max_items
        if settings.summarization.cache_enabled
        else 0
    )
    assert middleware[3].summary_cache.max_items == expected_cache_items
    assert isinstance(middleware[4], ObservabilityMiddleware)
    assert isinstance(middleware[5], ParamAwareDuplicateToolCallGuard)
    assert middleware[5].loop_stop_after == 3
    assert isinstance(middleware[6], ToolCallLimitMiddleware)
    assert middleware[6].run_limit == 64
    assert middleware[6].exit_behavior == "end"
    assert all(not isinstance(item, list) for item in middleware)


def test_agent_builds_memory_store():
    assert store is not None


def test_agent_main_model_has_bounded_request_behavior():
    assert model.request_timeout == settings.llm.timeout_seconds
    assert model.max_retries == settings.llm.max_retries
