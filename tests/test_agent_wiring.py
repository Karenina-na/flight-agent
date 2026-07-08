from langchain.agents.middleware import (
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
)

from src.agent import middleware, store
from src.guardrails import ContextBudgetGuard, ParamAwareDuplicateToolCallGuard
from src.memory import MemoryMiddleware
from src.observability import ObservabilityMiddleware
from src.skills import SkillMiddleware


def test_agent_middleware_is_flat_and_includes_observability_skills_and_memory():
    assert len(middleware) == 8
    assert isinstance(middleware[0], SummarizationMiddleware)
    assert isinstance(middleware[1], SkillMiddleware)
    assert isinstance(middleware[2], MemoryMiddleware)
    assert isinstance(middleware[3], TodoListMiddleware)
    assert isinstance(middleware[4], ContextBudgetGuard)
    assert isinstance(middleware[5], ObservabilityMiddleware)
    assert isinstance(middleware[6], ParamAwareDuplicateToolCallGuard)
    assert middleware[6].loop_stop_after == 3
    assert isinstance(middleware[7], ToolCallLimitMiddleware)
    assert middleware[7].run_limit == 64
    assert middleware[7].exit_behavior == "end"
    assert all(not isinstance(item, list) for item in middleware)


def test_agent_builds_memory_store():
    assert store is not None
