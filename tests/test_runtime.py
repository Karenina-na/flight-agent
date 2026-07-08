from dataclasses import FrozenInstanceError

import pytest

from src.runtime import Context, build_default_context


def test_context_keeps_user_id_as_required_minimum():
    context = Context(user_id="u1")

    assert context.user_id == "u1"
    assert context.thread_id is None
    assert context.request_id is None
    assert context.run_id is None
    assert context.tenant_id is None
    assert context.workspace_id is None
    assert context.locale == "zh-CN"
    assert context.timezone == "Asia/Shanghai"
    assert context.environment == "local"
    assert context.permissions == ()
    assert context.metadata == {}


def test_build_default_context_generates_correlation_ids():
    context = build_default_context(user_id="u1")

    assert context.request_id is not None
    assert context.request_id.startswith("req_")
    assert context.run_id is not None
    assert context.run_id.startswith("run_")
    assert context.current_user_input == ""
    assert context.current_user_input_sha256 == ""


def test_build_default_context_records_current_user_input_hash():
    context = build_default_context(
        user_id="u1",
        current_user_input="查询未来 10 天北京到上海机票并汇总",
    )

    assert context.current_user_input == "查询未来 10 天北京到上海机票并汇总"
    assert len(context.current_user_input_sha256) == 64


def test_context_is_request_scoped_and_immutable():
    context = Context(user_id="u1")

    with pytest.raises(FrozenInstanceError):
        context.user_id = "u2"


def test_build_default_context_populates_generic_fields():
    context = build_default_context(
        user_id="u1",
        thread_id="thread-1",
        request_id="request-1",
        run_id="run-1",
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        locale="en-US",
        timezone="UTC",
        environment="test",
        permissions=("memory:write",),
        current_user_input="原始用户问题",
        metadata={"entrypoint": "pytest"},
    )

    assert context == Context(
        user_id="u1",
        thread_id="thread-1",
        request_id="request-1",
        run_id="run-1",
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        locale="en-US",
        timezone="UTC",
        environment="test",
        permissions=("memory:write",),
        current_user_input="原始用户问题",
        current_user_input_sha256=context.current_user_input_sha256,
        metadata={"entrypoint": "pytest"},
    )
