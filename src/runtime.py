"""Runtime context shared across one agent invocation."""

from dataclasses import dataclass, field
from hashlib import sha256
from uuid import uuid4


@dataclass(frozen=True)
class Context:
    """Request-scoped context available to tools and middleware."""

    user_id: str
    thread_id: str | None = None
    request_id: str | None = None
    run_id: str | None = None
    tenant_id: str | None = None
    workspace_id: str | None = None
    locale: str = "zh-CN"
    timezone: str = "Asia/Shanghai"
    environment: str = "local"
    permissions: tuple[str, ...] = ()
    current_user_input: str = ""
    current_user_input_sha256: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def build_default_context(
    *,
    user_id: str,
    thread_id: str | None = None,
    request_id: str | None = None,
    run_id: str | None = None,
    tenant_id: str | None = None,
    workspace_id: str | None = None,
    locale: str = "zh-CN",
    timezone: str = "Asia/Shanghai",
    environment: str = "local",
    permissions: tuple[str, ...] = (),
    current_user_input: str = "",
    metadata: dict[str, str] | None = None,
) -> Context:
    """Build the default runtime context used by local demos and tests."""
    current_user_input_sha256 = (
        sha256(current_user_input.encode("utf-8")).hexdigest()
        if current_user_input
        else ""
    )
    return Context(
        user_id=user_id,
        thread_id=thread_id,
        request_id=request_id or _new_id("req"),
        run_id=run_id or _new_id("run"),
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        locale=locale,
        timezone=timezone,
        environment=environment,
        permissions=permissions,
        current_user_input=current_user_input,
        current_user_input_sha256=current_user_input_sha256,
        metadata=metadata or {},
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


__all__ = ["Context", "build_default_context"]
