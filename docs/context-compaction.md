# 上下文压缩实现规范

本文档是 SkyPilot 当前上下文压缩实现的统一说明和事实来源。它描述
主 Agent 实际运行的代码，不是未来计划。设计取舍见
[`ADR-002`](decisions/ADR-002-state-preserving-context-compaction.md)，历史实施步骤见
[`2026-07-08 context compaction pipeline plan`](superpowers/plans/2026-07-08-context-compaction-pipeline.md)。

## 1. 目标与边界

上下文压缩用于在长对话和长 ReAct 任务接近模型上下文上限时，构造一个更小的
**临时模型请求视图**。它要同时满足：

- 预算内请求完全不改写。
- 原始 system prompt、最新用户目标、可用工具和 `tool_choice` 保持不变。
- 尽量保留已经完成的工作、工具事实和 Todo 进度。
- 压缩后 Agent 仍可继续调用工具，不强制提前生成最终答案。
- 不修改 LangGraph checkpoint、会话原始消息、完整 trace 或 debug 数据。
- 压缩逻辑保持领域无关，不理解机票、价格、日期等业务字段。

它不承诺“无限上下文”。L5 后仍可能超预算；语义摘要也可能遗漏细节或不可用。

## 2. 代码职责

| 模块 | 当前职责 |
| --- | --- |
| `src/guardrails/context_budget_guard.py` | 估算完整请求、判断是否触发、调用 pipeline、记录最终压缩事件 |
| `src/guardrails/agent_state_compaction.py` | 对外提供压缩 middleware builder |
| `src/summarization/context_pipeline.py` | 编排 L1-L5、逐层重新估算、选择最终临时视图 |
| `src/summarization/context_compaction.py` | 分区、L1/L2 投影、Todo snapshot、synthetic tool 协议组装 |
| `src/summarization/layered_context.py` | 构造历史 user/assistant/tool 的分层状态并按预算裁剪 |
| `src/summarization/tool_observation.py` | 通用工具观察、shape/stats/preview 和 observation ledger |
| `src/summarization/tool_semantic.py` | L3 单次工具结果的分块与语义摘要 |
| `src/summarization/semantic_cache.py` | 进程内 LRU 语义摘要缓存 |
| `src/summarization/structured_output.py` | 提取摘要模型可见文本并维护摘要能力状态 |
| `src/summarization/context_trace.py` | 按实际执行顺序发出 L1-L5 trace 事件 |
| `src/prompt/context_budget.py` | 渲染压缩后的 synthetic context ledger |
| `src/prompt/tool_summary.py` | L3 工具结果摘要 prompt |
| `src/prompt/context_summary.py` | L4/L5 上下文摘要 prompt |

`guardrails` 只决定“是否需要压缩”；分区、改写和组装属于 `summarization`。

## 3. 主链路与触发条件

主 Agent 在 `src/agent.py` 中按以下顺序接入相关 middleware：

```text
Skill
-> Memory
-> TodoListMiddleware
-> AgentStateCompactionMiddleware
-> Observability
-> ParamAwareDuplicateToolCallGuard
-> ToolCallLimitMiddleware
```

每次 model call 前，`ContextBudgetGuard` 使用 observability 相同的完整请求序列化口径
估算字符数，范围包括：

- system prompt；
- 全部 messages；
- 工具名称、描述和参数 schema。

当前计算方式：

```text
threshold_chars = context_window_tokens * 4 * 0.85
```

默认值：

- `chars_per_token = 4`
- `max_fraction = 0.85`
- observation ledger 目标预算为阈值的 `25%`，且最少 `12000 chars`
- tool/message 普通 preview 默认 `500 chars`
- Todo 最多保留 20 项，每项最多 300 字符
- L3 候选工具结果最小长度为 1200 字符
- L3 单个摘要 chunk 最大约 12000 字符

只有同时满足以下条件才触发：

1. 完整请求估算值大于阈值；
2. 除最新用户消息外存在可压缩历史。

否则 middleware 将原始 `ModelRequest` 原样交给下游 handler，不执行 L1-L5、不注入
Todo snapshot，也不生成 synthetic ledger。

> 当前主链路的阈值是上述固定 `85%`。`config.yaml` 中
> `summarization.trigger`、`keep` 和 `trim_tokens_to_summarize` 是通用
> `SummarizationMiddleware` 的配置；该 middleware 当前未接入主 Agent，因此这些字段
> 不控制本文描述的 L1-L5 pipeline。

## 4. 临时视图和受保护内容

压缩只调用：

```python
request.override(messages=compacted_messages)
```

原始 checkpoint 和 trace 不被替换。下一次 model call 仍从 LangGraph 的原始状态开始，
重新估算；如果再次超预算，再根据当前完整状态构造临时视图。相同摘要输入通常会命中
语义缓存，因此不会要求摘要模型重新生成不同文本。

以下内容受保护：

- **System prompt**：始终由 `ModelRequest.system_prompt` 原样保留，不进入摘要。
- **最新用户目标**：优先从 request runtime 的 `current_user_input` 读取；没有时才回退到
  最新 `HumanMessage`。
- **活跃工具协议尾部**：最新用户目标之后尚未消费完的
  `AIMessage.tool_calls -> ToolMessage` 链保持协议合法。
- **工具面**：原 `tools` 和 `tool_choice` 不变。
- **Todo state**：不参与普通历史摘要，以受保护 snapshot 单独注入。

完整的旧 turn 不会作为 raw suffix 保留。raw suffix 实际只包含最新用户目标和可选的
活跃工具协议尾部，避免旧目标、旧 tool call 和大结果重新膨胀请求。

## 5. L1-L5 Pipeline

各层是**逐层输入**关系，不是五个彼此独立的候选视图。只有前一层仍不能满足预算时，
才继续进入下一层；已经执行的确定性处理会成为后续层的输入。

### L1：零成本修剪

L1 只处理可压缩历史，完全确定性，不调用模型：

- 连续历史 `HumanMessage` 在协议安全时合并；
- Tool 输出按完整内容 SHA-256 去重；
- 空字符串、`null`、空数组、空对象等历史 Tool 结果替换为短标记；
- 保留唯一且非空的工具结果供后续 ledger 处理。

重复和空结果使用短 `ToolMessage` 标记，不把旧原文继续带入临时视图。L1 不修改
原始消息。

### L2：ReAct 轨迹修剪

L2 同样是确定性处理：

- 删除旧 AI reasoning block；
- 删除旧 function call/tool call 模板；
- 对含 tool call 的旧 AI step，不保留其可见计划文本，避免“接下来调用某工具”在压缩后
  被模型误当作当前指令；
- 只保留没有 tool call 的历史 AI 可见回答。

L2 不生成“推理梗概”。需要语义归纳的旧历史统一交给 L4/L5。

### L3：工具结果保真与语义压缩

L3 面向大 ToolMessage，但先尝试无损保留：

1. 找出长度至少 1200 字符的候选工具结果。
2. 将候选原文恢复到临时 ledger/活跃协议尾部并重新估算。
3. 如果完整工具结果可以落入阈值，返回 `l3_lossless_preserved`，不调用摘要模型。
4. 只有完整结果放不下，才逐个调用摘要模型，返回 `l3_tool_semantic`。

工具摘要输入包含：

- `tool_name`
- 实际 `args`
- 当前完整 chunk
- 确定性的通用统计
- chunk 序号和总数

JSON 只在完整记录边界分块，文本只在完整行边界分块。摘要模型直接返回自由文本
`content`，不要求 JSON schema。prompt 要求保留实体、时间、来源、单位、限制和关键数值，
并在记录数量可控时逐条保留标识及关键值。

多 chunk 摘要会按顺序合并为该工具调用的一段语义结果。L3 只总结单个工具结果，
不判断整个用户任务是否完成，也不接收 Todo 作为摘要对象。

如果 L3 摘要失败，pipeline 使用确定性 ledger 继续；如果摘要能力被判定不可用，后续
请求直接跳过语义摘要，避免重复失败。

### L4：局部历史语义摘要

仅当 L3 后仍超过阈值、语义摘要已启用且摘要模型可用时执行。

L4 的输入不是原始完整 transcript，而是当前 bounded view：

```json
{
  "history": "当前分层 ledger 的模型可读文本",
  "previous_summaries": []
}
```

摘要模型输出一段自由文本，保留旧上下文中的客观事实、已完成工作、信息边界和待处理
事项。结果作为 `local_semantic_summary` 注入 synthetic ledger，同时仍保留确定性 ledger。

L4 完成后重新估算；满足预算就停止，否则进入 L5。

### L5：全局兜底摘要

L5 是最后的语义降级层。它只读取 L4 后的 bounded assembly，不回读原始完整历史。

最终临时视图保留：

- 原始 system prompt；
- 最新用户目标；
- 活跃工具协议尾部；
- Todo protected snapshot；
- 全局自由文本事实摘要；
- “较早逐条工具结果已折叠”的信息边界。

L5 不再渲染确定性 observation ledger 的逐条内容，
`deterministic_ledger_included=false`。工具仍可用，Agent 可以补查缺失事实。

L5 完成后即返回当前视图，即使 `still_over_budget=true`。当前没有再关闭工具或强制
最终回答的第六层。

## 6. 分层历史状态和工具观察

L1/L2 后的历史被组织为三个领域无关分区：

1. 旧用户消息；
2. AI 可见执行状态；
3. 工具观察。

工具观察卡内部可包含：

```json
{
  "tool_name": "some_tool",
  "tool_call_id": "call_123",
  "args": {"key": "value"},
  "status": "success",
  "result_shape": {},
  "result_preview": "...",
  "result_stats": {},
  "content_sha256": "..."
}
```

这些字段主要用于内部压缩和 trace。实际给业务模型的 ledger 使用简化的模型可读文本，
避免要求模型解释大量诊断字段。

当确定性 ledger 超出自己的预算时，降级顺序为：

1. 尽量保留全部 observation 及 preview；
2. 裁剪 observation preview，保留调用身份、参数、shape 和统计；
3. 裁剪旧 user/assistant 文本 preview；
4. 丢弃更老的 assistant card；
5. 丢弃更老的 user card；
6. 最后才丢弃最老的 observation card，并记录 dropped count。

因此它优先保留“哪些调用实际发生过”，但不保证始终保留完整结果。

## 7. Todo Protected Snapshot

Todo 来自 `request.state["todos"]`，通过 duck typing 读取。只有触发压缩后才提取：

```json
{
  "type": "todo_snapshot",
  "total_count": 3,
  "preserved_count": 3,
  "dropped_count": 0,
  "truncated_count": 0,
  "items": [
    {"index": 0, "content": "查询第一天", "status": "completed"},
    {"index": 1, "content": "查询第二天", "status": "in_progress"},
    {"index": 2, "content": "汇总结果", "status": "pending"}
  ]
}
```

规则：

- 不存在、格式错误或没有有效 item 时不注入；
- 每项只保留顺序、`content` 和 `status`；
- 最多 20 项，每项最多 300 字符；
- 不进入 L3-L5 的待摘要历史；
- 在最终 ledger 中标记为“受保护状态”和当前任务进度的权威来源；
- 摘要模型不会调用 `write_todos`，业务 Agent 仍按 system prompt 管理 Todo 生命周期。

Todo 使用仍由模型决定，prompt 不能保证小模型一定创建 Todo。缺失 Todo 不会阻断业务
工具调用或压缩。

## 8. Synthetic Tool 注入协议

压缩历史通过一对协议合法的消息注入，而不是孤立插入 ToolMessage：

```json
[
  {
    "role": "user",
    "content": "最新用户目标"
  },
  {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "id": "context_ledger_<stable-hash>",
        "name": "context_observation_ledger",
        "args": {}
      }
    ]
  },
  {
    "role": "tool",
    "name": "context_observation_ledger",
    "tool_call_id": "context_ledger_<stable-hash>",
    "content": "压缩后的历史工作状态、Todo、摘要和信息边界"
  }
]
```

synthetic call id 由确定性 ledger 文本 SHA-256 的前 16 位生成。相同 ledger 得到稳定 ID。
synthetic args 当前为空，预算诊断只进入 trace，不增加模型上下文负担。

消息顺序为 raw live suffix 在前、synthetic ledger pair 在后，兼容要求先出现 user query 的
本地模型模板。该工具不是注册给模型主动调用的业务工具，只是 Harness 内部上下文载体。

## 9. 语义摘要模型与缓存

摘要模型使用独立 `ChatOpenAI` client：

- `temperature=0`
- 可使用主模型或 `summarization.model` 指定的模型
- 默认关闭 thinking：`enable_thinking=false`
- 不设置固定 `max_output_tokens`
- timeout、retry 来自 summarization 配置
- 不绑定业务工具

L3-L5 都接受任意非空的可见文本作为摘要正文，不解析 JSON metadata。只返回 reasoning、
空文本或不可提取内容时，`SemanticSummaryCapability` 会在当前进程标记为 unavailable。

进程内 LRU 缓存 key 由以下内容构成：

- 摘要模型身份；
- 规范化后的摘要 messages。

相同输入返回完全相同的缓存文本。默认最多 256 项；设置
`summarization.cache_enabled=false` 时 middleware 使用容量 0。缓存不是持久化缓存，服务
重启后清空；历史新增消息或 bounded view 变化会产生新 key。

摘要 capability 在进程内记录“摘要能力不可用”状态，避免反复调用一个只输出 reasoning
的模型；该状态不存入 LRU 文本缓存。普通超时或瞬态异常不等同于永久不可用，当前请求
回退到最近成功的确定性视图。

## 10. 每次 ReAct 中的重复压缩

临时视图不会写回 checkpoint，因此同一 turn 内每个后续 model call 都会从原始完整状态
重新进入 guard：

```text
原始 checkpoint + 新 ToolMessage
-> 完整估算
-> 再次触发 L1-L5
-> 构造新的临时视图
-> 调用业务模型
```

这能保证原始证据不被不可逆摘要，但意味着长 ReAct turn 可能多次执行压缩。语义缓存只
复用输入完全相同的摘要；新增工具结果会使相关摘要产生新 key。

上一轮 synthetic ledger 本身不会进入 checkpoint，因为它只存在于
`request.override(messages=...)` 的临时请求中，所以下一轮不会递归压缩“压缩结果”。

## 11. 失败与回退矩阵

| 情况 | 行为 |
| --- | --- |
| 请求未超预算 | 原样通过 |
| 超预算但无可压缩历史 | 原样通过，由模型服务决定是否接受 |
| L3 完整工具结果可放下 | 无损保留，不调用摘要模型 |
| L3 摘要瞬态失败 | 使用确定性 L1/L2/observation ledger 结果 |
| 摘要模型不可用 | 标记 capability unavailable，后续跳过 L3-L5 语义调用 |
| L4 失败 | 回退到 L3 后的工作视图 |
| L5 失败 | 当前实现回退到进入 L4 前的 L3 工作视图 |
| L5 后仍超预算 | 返回 L5 视图并记录 `still_over_budget=true` |
| 压缩后事实不足 | 工具保持可用，模型应补查或说明信息不足 |

重复工具调用拦截、总工具调用次数限制和最终回答 fallback 是独立 guardrail，不属于
L1-L5。压缩事件中出现它们的结果，只代表这些结果成为待压缩历史。

## 12. Trace 与 UI

只展示实际执行的层级，不把未进入的 L4/L5 标记为完成。

逐层事件：

```text
context_compaction_layer
```

语义摘要调用事件：

```text
context_summary_start
context_summary_end
context_summary_error
context_summary_unavailable
```

每次完整 pipeline 的结果事件：

```text
react_context_budget_compacted
```

关键字段包括：

- 原始估算与阈值：`estimate_chars`、`threshold_chars`
- 最终大小：`post_compaction_chars`、`still_over_budget`
- 最终层级：`compaction_level`
- observation 保留/丢弃/preview 裁剪数量
- old user 和 assistant card 保留/丢弃数量
- 原始/压缩后 message 和 tool 数量
- Todo 总数、保留数、丢弃数、状态统计、是否全部完成
- L3 候选数、摘要成功数、失败或跳过原因
- L4/L5 摘要数量、是否使用 global fallback
- 是否仍包含确定性 ledger
- compacted state preview、字符数和 SHA-256

完整 trace 保存原始请求和模型响应；UI 的“上下文压缩”面板按一次 pipeline 分组，再按
真实时间顺序展示实际进入的 L 层和摘要调用。

## 13. 压缩前后示例

压缩前 checkpoint 可能是：

```json
[
  {"role": "user", "content": "查询三天票价"},
  {"role": "assistant", "reasoning": "先查第一天", "tool_calls": [{"id": "c1", "name": "search", "args": {"day": 1}}]},
  {"role": "tool", "tool_call_id": "c1", "content": "很大的第一天结果"},
  {"role": "assistant", "reasoning": "继续第二天", "tool_calls": [{"id": "c2", "name": "search", "args": {"day": 2}}]},
  {"role": "tool", "tool_call_id": "c2", "content": "很大的第二天结果"},
  {"role": "user", "content": "继续完成并汇总"},
  {"role": "assistant", "tool_calls": [{"id": "c3", "name": "search", "args": {"day": 3}}]},
  {"role": "tool", "tool_call_id": "c3", "content": "当前活跃结果"}
]
```

达到 L3 后，本次业务模型看到的临时 messages 形态是：

```json
[
  {"role": "user", "content": "继续完成并汇总"},
  {"role": "assistant", "tool_calls": [{"id": "c3", "name": "search", "args": {"day": 3}}]},
  {"role": "tool", "tool_call_id": "c3", "content": "当前活跃结果或其L3摘要"},
  {"role": "assistant", "content": "", "tool_calls": [{"id": "context_ledger_hash", "name": "context_observation_ledger", "args": {}}]},
  {"role": "tool", "tool_call_id": "context_ledger_hash", "content": "旧用户目标、已完成工具事实、Todo和继续执行边界"}
]
```

达到 L4 时，最后一条 ledger 同时包含局部历史摘要和确定性工具观察。达到 L5 时，最后
一条 ledger 只保留 Todo、全局事实摘要和“逐条历史已折叠”的边界，不再包含逐条
observation 详情。

## 14. 当前限制与维护规则

- 字符估算不等于模型 tokenizer，尤其对中文、JSON 和长 schema 可能存在偏差。
- 85% 阈值和 `4 chars/token` 当前是代码默认值，不是 YAML 配置。
- 自由文本摘要兼容小模型，但无法像严格 JSON 一样做字段级验证。
- L3 摘要必须在压缩率和事实保真之间取舍，精确回答可能需要重新调用工具。
- Todo snapshot 无损保护状态结构，但模型是否主动创建、更新 Todo 仍受模型能力影响。
- 进程缓存不会跨服务重启共享，也不会消除新增历史导致的新摘要调用。
- L5 是压缩兜底，不是请求成功兜底；极端情况下仍可能超预算。
- 修改 pipeline 行为时，应同步更新本文档、ADR（如决策变化）和对应测试。

建议重点回归：

```bash
.venv/bin/python -m pytest tests/test_context_compaction.py tests/test_context_budget_guard.py tests/test_tool_observation.py -q
.venv/bin/python -m pytest tests/test_agent_wiring.py tests/test_chat_runner.py tests/test_web_ui.py -q
.venv/bin/python -m pytest
```
