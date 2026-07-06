from langchain.agents.middleware import SummarizationMiddleware

from src.agent import middleware, store
from src.memory import MemoryMiddleware
from src.observability import ObservabilityMiddleware
from src.skills import SkillMiddleware


def test_agent_middleware_is_flat_and_includes_observability_skills_and_memory():
    assert len(middleware) == 4
    assert isinstance(middleware[0], ObservabilityMiddleware)
    assert isinstance(middleware[1], SummarizationMiddleware)
    assert isinstance(middleware[2], SkillMiddleware)
    assert isinstance(middleware[3], MemoryMiddleware)
    assert all(not isinstance(item, list) for item in middleware)


def test_agent_builds_memory_store():
    assert store is not None
