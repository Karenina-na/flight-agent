# Context Compaction Pipeline Plan

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
| **Layer 3: Tool Result Reduction** | After Layer 2 if still large | Large `ToolMessage` content | Deterministically reduce large outputs. JSON: keys, counts, numeric stats, samples. Text: head/tail/hash/omitted length. Logs/code: error stack and key lines only. | Low |
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

Layer 3 output examples:

Large JSON:

```json
{
  "status": "compacted_tool_result",
  "raw_sha256": "...",
  "shape": {"top_level_keys": ["query", "quotes", "limitations"], "quotes.length": 120},
  "stats": {"quotes.price.min": 400, "quotes.price.max": 1880},
  "samples": {"quotes.first": ["..."], "quotes.last": ["..."]}
}
```

Large text:

```json
{
  "status": "compacted_tool_result",
  "raw_sha256": "...",
  "text_head": "...",
  "text_tail": "...",
  "omitted_chars": 32000
}
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

### Task 3: Implement Deterministic Layers 1-3

**Files:**
- Modify: `src/guardrails/context_budget_guard.py`
- Modify or reuse: `src/summarization/context_compaction.py`, `src/summarization/layered_context.py`, `src/summarization/tool_observation.py`
- Test: `tests/test_context_compaction.py`, `tests/test_context_budget_guard.py`, `tests/test_layered_context.py`, `tests/test_tool_observation.py`

- [x] Layer 1: replace duplicate historical Tool outputs by content hash in the transient request view.
- [x] Layer 1: replace low-value empty Tool outputs with short markers.
- [x] Layer 1: merge adjacent User messages only when no tool protocol edge is between them.
- [x] Layer 2: remove old reasoning blocks only inside compaction mode; any semantic reasoning summary belongs to Layer 4.
- [x] Layer 3: compact large ToolMessage JSON/text into bounded structured summaries.

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

### Task 5: Add LLM-Backed Layers 4-5

**Files:**
- Add: `src/summarization/context_pipeline.py`
- Add: `src/prompt/context_summary.py`
- Test: `tests/test_context_pipeline.py`
- Modify: `src/guardrails/context_budget_guard.py`, `src/guardrails/agent_state_compaction.py`, `src/agent.py`

- [x] Move L1-L5 orchestration into `src/summarization/context_pipeline.py`.
- [x] Keep `ContextBudgetGuard` focused on threshold detection, pipeline invocation, and trace metadata.
- [x] Layer 4: when L1-L3 remains over budget, call the configured summary model with a bounded context view and inject a `local_semantic_summary`.
- [x] Layer 5: when L4 remains over budget, call the summary model again using the current bounded assembly and inject a `global_fallback_summary`.
- [x] Summary prompts live in `src/prompt/context_summary.py` and require structured JSON with `facts`, `open_items`, and `evidence_refs`.
- [x] LLM summary failure falls back to the deterministic L1-L3 compacted view and records `semantic_summary_failed=true`.
- [x] Todo remains protected state: the semantic summary model receives only a count-level todo reference, while the final synthetic ledger keeps the compact todo snapshot.

Current implementation:

- `build_context_pipeline_request()` first builds the deterministic L1-L3 compacted request through `build_context_compaction_request()`.
- If the compacted request is already within budget, no summary model is called and `compaction_level="l1_l3"`.
- If semantic compression is enabled and a summary model is available, L4 generates a local semantic summary from a bounded assembly containing protected todo metadata, deterministic history ledger, and existing semantic summaries.
- If L4 is still oversized, L5 replaces local semantic detail with a global fallback summary while preserving raw recent messages, the latest user goal, protected todo snapshot, and the protocol-valid synthetic context ledger pair.
- `settings.summarization.enabled=false` disables L4/L5 and keeps the pipeline deterministic.
- Trace metadata includes `compaction_level`, `semantic_summary_count`, `semantic_summary_failed`, `global_fallback_used`, `post_compaction_chars`, and `still_over_budget`.

---

## Non-Goals

- Do not rewrite checkpoint/session history.
- Do not compress under-budget requests.
- Do not ask the model to call todo tools during compaction.
- Do not hide raw trace/debug data.
- Do not make reasoning summaries a normal always-on behavior.
