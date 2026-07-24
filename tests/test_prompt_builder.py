from src.prompt import (
    CORE_PROMPT,
    CONTEXT_LEDGER_TOOL_NAME,
    DOMAIN_PROMPT,
    TODO_GUIDANCE_PROMPT,
    build_context_ledger_tool_call_args,
    build_context_ledger_tool_observation,
    build_memory_prompt_addendum,
    build_skill_prompt_addendum,
    build_system_prompt,
)
from src.prompt.build import build_system_prompt as build_system_prompt_from_module
from src.tools import build_tool_prompt, get_tools
from src.tools.capabilities import build_tool_prompt as build_tool_prompt_from_module


def test_base_prompt_layers_do_not_name_concrete_tools():
    assert "resolve_flight_locations" not in DOMAIN_PROMPT
    assert "search_airfare_quotes" not in DOMAIN_PROMPT
    assert "query_flight_information" not in DOMAIN_PROMPT
    assert "query_current_date" not in DOMAIN_PROMPT


def test_domain_prompt_positions_agent_for_air_ticket_fact_explanations():
    assert "机票事实查询助手" in DOMAIN_PROMPT
    assert "相对日期" in DOMAIN_PROMPT
    assert "先用已注册能力获得当前日期和目标日期" in DOMAIN_PROMPT
    assert "不要直接要求用户改写成 YYYY-MM-DD" in DOMAIN_PROMPT
    assert "先用已注册能力解析候选机场" in DOMAIN_PROMPT
    assert "报销审批" in DOMAIN_PROMPT
    assert "不做结论" in DOMAIN_PROMPT
    assert "当前报价不等于历史出票价" in DOMAIN_PROMPT
    assert "历史出票价格" in DOMAIN_PROMPT
    assert "行李额" not in DOMAIN_PROMPT
    assert "准点率" not in DOMAIN_PROMPT
    assert "历史平均票价区间" not in DOMAIN_PROMPT
    assert "能力调用 ID" not in DOMAIN_PROMPT


def test_core_prompt_defines_react_guard_and_ledger_semantics():
    assert "context_observation_ledger" in CORE_PROMPT
    assert "duplicate_blocked" in CORE_PROMPT
    assert "react_loop_stop_requested" in CORE_PROMPT
    assert "stop_requested=true" in CORE_PROMPT
    assert "立即停止工具调用" in CORE_PROMPT
    assert "基于已有事实回答" in CORE_PROMPT


def test_tool_layer_is_generated_from_registered_tools():
    tool_prompt = build_tool_prompt(get_tools())

    assert "query_current_date" in tool_prompt
    assert "resolve_flight_locations" in tool_prompt
    assert "search_airfare_quotes" in tool_prompt
    assert "query_flight_information" in tool_prompt
    assert "相对日期" in tool_prompt
    assert "解析城市/机场/IATA 为机场候选事实" in tool_prompt
    assert '{"locations":["北京","上海"]}' in tool_prompt
    assert "查询某航线某日期的公开机票报价事实" in tool_prompt
    assert '"origin":"北京"' in tool_prompt
    assert "按航班号查询航班事实" in tool_prompt
    assert '"flight_number":"CA981"' in tool_prompt


def test_prompt_package_exposes_large_prompt_build_modules():
    assert build_system_prompt_from_module is build_system_prompt
    assert build_tool_prompt_from_module is build_tool_prompt


def test_system_prompt_combines_layers():
    system_prompt = build_system_prompt(tools=get_tools())

    assert CORE_PROMPT in system_prompt
    assert DOMAIN_PROMPT in system_prompt
    assert "resolve_flight_locations" in system_prompt
    assert "search_airfare_quotes" in system_prompt
    assert "query_flight_information" in system_prompt
    assert "query_current_date" in system_prompt


def test_todo_guidance_prompt_defines_complex_task_lifecycle_without_domain_tools():
    assert "调用第一个业务工具前" in TODO_GUIDANCE_PROMPT
    assert "三个或更多" in TODO_GUIDANCE_PROMPT
    assert "in_progress" in TODO_GUIDANCE_PROMPT
    assert "pending" in TODO_GUIDANCE_PROMPT
    assert "completed" in TODO_GUIDANCE_PROMPT
    assert "每完成一个子任务" in TODO_GUIDANCE_PROMPT
    assert "可验证的执行子任务" in TODO_GUIDANCE_PROMPT
    assert "匹配的成功工具结果" in TODO_GUIDANCE_PROMPT
    assert "最后一次 write_todos 调用之后" in TODO_GUIDANCE_PROMPT
    assert "search_airfare_quotes" not in TODO_GUIDANCE_PROMPT
    assert "query_current_date" not in TODO_GUIDANCE_PROMPT


def test_context_budget_prompts_live_in_prompt_package():
    class FakeLedger:
        def to_model_text(self) -> str:
            return "- 已完成 generic_lookup，参数：{\"slot\":1}；结果：共 2 条记录。"

        def to_prompt_text(self) -> str:
            return '{"observation_count":1,"result_shape":{"type":"object"}}'

    assert CONTEXT_LEDGER_TOOL_NAME == "context_observation_ledger"
    tool_args = build_context_ledger_tool_call_args(
        original_user_message="请汇总",
        estimate_chars=100,
        threshold_chars=80,
    )
    prompt = build_context_ledger_tool_observation(
        original_user_message="请汇总",
        ledger=FakeLedger(),
        estimate_chars=100,
        threshold_chars=80,
        todo_snapshot={
            "type": "todo_snapshot",
            "items": [{"index": 0, "content": "汇总报价", "status": "in_progress"}],
        },
    )

    assert tool_args == {}
    assert "这是历史工具观察，不是最终回答指令" in prompt
    assert "必要时仍可调用可用工具" in prompt
    assert "已完成 generic_lookup" in prompt
    assert "任务进度" in prompt
    assert "受保护状态" in prompt
    assert "权威任务状态" in prompt
    assert "汇总报价" in prompt
    assert "进行中" in prompt
    assert "observation_count" not in prompt
    assert "result_shape" not in prompt
    assert "todo_snapshot" not in prompt
    assert "estimate_chars" not in prompt
    assert "原请求估算" not in prompt


def test_middleware_prompt_addenda_live_in_prompt_package():
    assert "## Long-Term Memory" in build_memory_prompt_addendum()
    assert "remember_user_fact(key, value)" in build_memory_prompt_addendum()
    skill_addendum = build_skill_prompt_addendum("- concise-writer: Writes concise answers")
    assert "## Available Skills" in skill_addendum
    assert "- concise-writer: Writes concise answers" in skill_addendum
    assert "load_skill(skill_name)" in skill_addendum
