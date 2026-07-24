# Context Compaction Pipeline Plan

> 本文是历史实施计划。当前运行行为以
> [`docs/context-compaction.md`](../../context-compaction.md) 为准。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace always-on ReAct hygiene cleanup with an on-demand, layered context compaction pipeline that activates only when the model request approaches or exceeds the context budget.

**Architecture:** Normal agent runs keep original LangGraph/checkpointer/session messages untouched and send them to the model as-is. When a model request crosses the configured compaction threshold, `ContextBudgetGuard` delegates to the `summarization` context pipeline, which builds a transient compacted request view through five deterministic-to-expensive layers, then calls the model with `request.override(messages=...)`. Raw checkpoint, trace, and debug data remain complete.

**Tech Stack:** Python 3.12, LangChain/LangGraph middleware, pytest, existing `src/guardrails/context_budget_guard.py`, `src/summarization/context_compaction.py`, `src/summarization/layered_context.py`, `src/summarization/tool_observation.py`, `src/chat/trace.py`, and `src/chat/runner.py`.

---

## Decision

Do **not** compact or sanitize ordinary under-budget conversations.

The previous plan proposed ReAct hygiene compaction whenever old completed turns contained reasoning, tool calls, or tool results. That is now superseded. The new policy is:

```text
If request estimate is below threshold:
  pass original messages through unchanged.

If request estimate crosses threshold:
  enter the five-layer compaction pipeline.
```

This keeps normal LangChain/LangGraph interaction semantics intact and avoids unnecessary prompt/cache churn.

---

## Protocol Model

Raw checkpoint/session/trace messages stay complete:

```json
[
  {"role": "user", "content": "第一轮问题"},
  {"role": "assistant", "content": [{"type": "reasoning"}, {"type": "function_call"}], "tool_calls": [{"id": "call-1"}]},
  {"role": "tool", "tool_call_id": "call-1", "content": "完整工具结果"},
  {"role": "assistant", "content": [{"type": "reasoning"}, {"type": "text", "text": "最终回答"}]},
  {"role": "user", "content": "第二轮问题"}
]
```

Only after compaction triggers, the transient model request may become:

```json
[
  {"role": "user", "content": "第一轮问题"},
  {"role": "assistant", "content": "第一轮事实摘要"},
  {"role": "user", "content": "第二轮问题"}
]
```

If the active turn contains valid tool protocol messages, preserve those pairings unless the pipeline reaches a hard fallback layer:

```json
[
  {"role": "user", "content": "当前问题"},
  {"role": "assistant", "tool_calls": [{"id": "call-2", "name": "query_current_date"}]},
  {"role": "tool", "tool_call_id": "call-2", "content": "工具结果或压缩后的工具结果"}
]
```

---

## Trigger Policy

Each model call starts with a size estimate using the same serialized request view as observability.

Recommended thresholds:

```text
soft_threshold = context_window * chars_per_token * 0.75
hard_threshold = context_window * chars_per_token * 0.85
```

- Below `soft_threshold`: send original request unchanged.
- Between `soft_threshold` and `hard_threshold`: run deterministic layers first.
- Above `hard_threshold`: run deterministic layers, then allow LLM-backed folding layers if still oversized.

The exact threshold can stay at the existing `max_fraction` initially; the important change is that compaction is threshold-triggered, not always-on.

---

## Five-Layer Pipeline

| Layer | Trigger Timing | Compression Target | Strategy | Cost |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1: Zero-Cost Trimming** | After compaction mode starts | Historical messages | Rule filtering only: remove duplicate Tool outputs, drop low-value empty Tool-result rounds, merge adjacent User messages where protocol-safe. No semantic summary. | Very low |
| **Layer 2: ReAct Trimming** | After Layer 1 if still large | Historical AI `reasoning` and tool-call traces | Deterministically remove old reasoning blocks and old function/tool-call templates from the transient compacted prefix. Do not generate reasoning summaries here. | Low |
| **Layer 3: Tool Result Semantic Compression** | After Layer 2 when large Tool results are present | Individual large `ToolMessage` content | Preserve the raw ToolMessage outside the transient request, split JSON/text only on complete record or line boundaries, then use the summary model to retain goal-relevant facts, values, sources, times, units, and limitations. Invalid summaries fall back to deterministic counts and numeric ranges. | Medium |
| **Layer 4: Historical Turn Folding** | If still near/above budget | Older complete `User-AI-Tool` turns | LLM or deterministic summarizer creates an objective fact summary under 200 Chinese characters per folded block. | Medium |
| **Layer 5: Global Fallback Summary** | If still above hard budget | Everything except protected anchors | Preserve system prompt, latest user goal, and active protocol anchors; compress the rest into a fact list under 400 Chinese characters. | High |

Protected anchors:

- System prompt.
- Latest user message / current user goal.
- Required active tool protocol pairings.
- Tool call IDs needed to keep current protocol valid.
- Todo snapshot, if available, as compact state rather than raw conversation.

---

## Todo State Policy

Todo state should **participate in the compacted context**, but should **not be compressed like ordinary chat/tool messages**.

Reason:

- Todo is working state, not conversational noise.
- Compressing todo items through a generic text summarizer may lose status, ordering, or pending work.
- The model needs todo progress after compaction to continue the task coherently.

Policy:

```text
Normal under-budget request:
  Do not inject extra todo state solely for compaction.

Compaction-triggered request:
  Include a compact todo snapshot as protected state.
```

Suggested compact todo shape:

```json
{
  "type": "todo_snapshot",
  "items": [
    {"content": "查询广州到香港报价", "status": "completed"},
    {"content": "汇总报价结果", "status": "in_progress"}
  ],
  "instruction": "Continue from pending/in_progress items. If task state changes, update todos with the todo tool."
}
```

Todo should therefore be **outside the five generic compression targets**. It is injected or preserved as a compact structured snapshot when compaction is active.

Current implementation:

- `ContextBudgetGuard` only reads todo state after context-budget compaction is triggered.
- Todo state is read by duck-typing `ModelRequest.state["todos"]`; if the state is missing or malformed, compaction continues without todo injection.
- The compact snapshot keeps only `index`, `content`, and `status`; raw todo tool calls, reasoning, extra metadata, and historical messages are not copied into the compacted context.
- The compact snapshot is bounded: it preserves at most 20 todo items, truncates each content field to 300 chars, and records `total_count`, `preserved_count`, `dropped_count`, and `truncated_count`.
- The snapshot is rendered inside the synthetic context ledger tool observation before the compressed historical state, so it acts as protected task state rather than ordinary chat/tool history.
- Under-budget requests remain unchanged and do not receive any todo snapshot.

---

## Tool Result Policy

Tool results are not compressed immediately after every tool call under normal operation.

Flow:

```text
Tool returns raw result
  -> raw result remains in checkpoint/trace/debug
  -> next model call estimates context size
  -> if compaction triggers, Layer 3 compresses ToolMessage content in the transient request view
```

This avoids changing normal tool semantics and keeps trace/debug fidelity.

Layer 3 model-facing output is rendered as concise facts rather than truncated
JSON. For example:

```text
search_airfare_quotes 查询成功。
共获得 7 条报价，价格范围为 540–910 CNY。
最低报价为 KN5977，540 CNY；最高报价为 CA1883，910 CNY。
数据来源为 fliggy_mcp，报价为查询时点样本。
```

---

## Implementation Tasks

### Task 1: Revert Always-On Hygiene Semantics

**Files:**
- Modify: `src/guardrails/context_budget_guard.py`
- Modify: `tests/test_context_budget_guard.py`

- [x] Update tests so under-budget requests pass through unchanged even when history contains old reasoning/tool messages.
- [x] Keep existing oversized request tests green.
- [x] Remove or rename tests that expect under-budget hygiene compaction.

Expected behavior:

```python
request = _request(messages=[old_react_history, HumanMessage(content="new task")])
guard = ContextBudgetGuard(context_window_tokens=8192, max_fraction=0.85)

guard.wrap_model_call(request, handler)

assert seen_requests[0] is request
```

### Task 2: Add Threshold-Triggered Pipeline Skeleton

**Files:**
- Modify: `src/guardrails/context_budget_guard.py`
- Test: `tests/test_context_budget_guard.py`

- [x] Extract current compaction request construction into `src/summarization/context_compaction.py`.
- [x] Add a pipeline method `_compaction_pipeline_request`.
- [x] Preserve current `request.override(messages=...)` behavior only when the estimate crosses threshold.

### Task 3: Implement Deterministic Layers 1-2 and Tool Observation Fallback

**Files:**
- Modify: `src/guardrails/context_budget_guard.py`
- Modify or reuse: `src/summarization/context_compaction.py`, `src/summarization/layered_context.py`, `src/summarization/tool_observation.py`
- Test: `tests/test_context_compaction.py`, `tests/test_context_budget_guard.py`, `tests/test_layered_context.py`, `tests/test_tool_observation.py`

- [x] Layer 1: replace duplicate historical Tool outputs by content hash in the transient request view.
- [x] Layer 1: replace low-value empty Tool outputs with short markers.
- [x] Layer 1: merge adjacent User messages only when no tool protocol edge is between them.
- [x] Layer 2: remove old reasoning blocks only inside compaction mode; any semantic reasoning summary belongs to Layer 4.
- [x] Layer 3 fallback: preserve deterministic array counts and numeric ranges when semantic tool-result compression is unavailable.

### Task 4: Add Todo Snapshot Preservation

**Files:**
- Modify: `src/guardrails/context_budget_guard.py`
- Modify: `src/summarization/context_compaction.py`
- Modify: `src/prompt/context_budget.py`
- Test: `tests/test_context_compaction.py`, `tests/test_context_budget_guard.py`, `tests/test_prompt_builder.py`

- [x] Read todo state from `ModelRequest.state["todos"]` with duck typing, without importing middleware-private state types.
- [x] Include compact todo snapshot only after context-budget compaction triggers.
- [x] Keep under-budget requests unchanged, even if todo state exists.
- [x] Skip malformed todo state and empty todo items without raising.
- [x] Render the todo snapshot as protected task state inside the synthetic context ledger observation.
- [x] Add trace metadata for `todo_snapshot_item_count`.
- [x] Add trace metadata for `todo_snapshot_total_count`, `todo_snapshot_dropped_count`, and `todo_snapshot_truncated_count`.
- [x] Record `compacted_state_preview` from the full synthetic context ledger observation, including protected todo state when present.
- [x] Bound todo snapshot size to avoid re-inflating compacted context.
- [x] Do not make the model call todo tools during compression.

### Task 5: Add LLM-Backed Layers 3-5

**Files:**
- Add: `src/summarization/context_pipeline.py`
- Add: `src/prompt/context_summary.py`
- Test: `tests/test_context_pipeline.py`
- Modify: `src/guardrails/context_budget_guard.py`, `src/guardrails/agent_state_compaction.py`, `src/agent.py`

- [x] Move L1-L5 orchestration into `src/summarization/context_pipeline.py`.
- [x] Keep `ContextBudgetGuard` focused on threshold detection, pipeline invocation, and trace metadata.
- [x] Layer 3: split oversized ToolMessage JSON/text on complete boundaries and summarize each result independently.
- [x] Layer 4: when L1-L3 remains over budget, call the configured summary model with a bounded context view and inject a `local_semantic_summary`.
- [x] Layer 5: when L4 remains over budget, call the summary model again using the current bounded assembly and inject a `global_fallback_summary`.
- [x] Summary prompts live in `src/prompt/context_summary.py` and require structured JSON with `facts`, `open_items`, and `evidence_refs`.
- [x] L3 summary failure falls back to deterministic counts/ranges; L4/L5 failure falls back to the latest successful compacted view and records semantic failure metadata.
- [x] Todo remains protected state: it is excluded from L3-L5 summary inputs, while the final synthetic ledger keeps the compact todo snapshot.

Current implementation:

- `build_context_pipeline_request()` first applies deterministic L1-L2 trimming and constructs the observation ledger through `build_context_compaction_request()`.
- If oversized ToolMessages are present, the pipeline first restores their complete results into the transient ledger and re-estimates the request. When that lossless view fits, it returns `compaction_level="l3_lossless_preserved"` without invoking the summary model.
- If complete ToolMessages do not fit and semantic compression is available, L3 summarizes each result independently and sets `compaction_level="l3_tool_semantic"` when the resulting request fits.
- L3 never cuts JSON in the middle of a record. Large JSON arrays are divided into complete-record chunks; text is divided on complete line boundaries.
- The L3 prompt receives only the current tool name, actual call arguments, the current complete-boundary chunk, and deterministic statistics. It does not receive the global user goal and does not decide whether the overall task is complete.
- If no L3 candidate exists and the deterministic compacted request is already within budget, no summary model is called and `compaction_level="l1_l3"`.
- If L3 remains oversized, L4 generates a local semantic summary from the compacted history ledger and existing semantic summaries.
- If L4 is still oversized, L5 switches to strict global fallback mode: raw recent messages, the latest user goal, protected todo snapshot, and the protocol-valid synthetic context ledger pair are preserved, but deterministic history ledger details are omitted from the ledger body and replaced by a `deterministic_history_omitted` notice plus the global fallback summary.
- `settings.summarization.enabled=false` disables L3-L5 model summaries and keeps the pipeline deterministic.
- The summary client is isolated from the main agent client and uses `temperature=0`, configurable timeout/output limits, no retries by default, and disabled thinking by default.
- L3/L4/L5 summary calls emit `context_summary_start`, `context_summary_end`, and `context_summary_error` lifecycle events. Invalid JSON and invalid output schemas both terminate with an error event and fall back to the latest successful compacted view.
- Trace metadata includes `compaction_level`, `tool_semantic_candidate_count`, `tool_semantic_summary_count`, `tool_semantic_summary_failed`, `semantic_summary_count`, `semantic_summary_failed`, `global_fallback_used`, `deterministic_ledger_included`, `post_compaction_chars`, and `still_over_budget`.

---

## Non-Goals

- Do not rewrite checkpoint/session history.
- Do not compress under-budget requests.
- Do not let L3-L5 summary-model calls invoke todo tools. The business Agent
  still follows the system-level Todo lifecycle after receiving a protected
  snapshot in its compacted request.
- Do not hide raw trace/debug data.
- Do not make reasoning summaries a normal always-on behavior.
