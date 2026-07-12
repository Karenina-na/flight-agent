# ADR-002: Use State-Preserving Context Compaction for ReAct Turns

## Status
Accepted

## Date
2026-07-07

## Context
The agent can enter long multi-step ReAct workflows, especially when it needs to
query many dates, routes, or tool-backed facts before producing a useful answer.
Earlier context-budget protection compressed oversized requests into a forced
final-answer request:

- Replace the original message history with one final-answer prompt.
- Disable tools with `tools=[]`.
- Set `tool_choice="none"`.
- Tell the model not to call tools again and to answer only from existing facts.

This protected the model from exceeding the context window, but it also changed
the task semantics. A long-running workflow could be cut short just because the
context was large, even when the agent still needed to call more tools to finish
the user's request.

For example, if the user asked to query a range of dates and the agent had only
queried part of the range, forced final-answer compaction could correctly avoid
fabricating missing results, but it could not continue the remaining work.

## Decision
Use `AgentStateCompactionMiddleware` as the default state-preserving
context-budget behavior for the active agent. The middleware keeps guardrail
responsibilities thin: it estimates request size, decides whether compaction is
needed, calls the `summarization` context pipeline, and records trace metadata.
The L1-L5 partitioning, rewriting, semantic summary, and final transient message
assembly live in the summarization layer.

When the request approaches the configured context budget, the agent now:

1. Splits the message history into a compressible prefix and a protected live
   suffix.
2. Compresses older turns through the `summarization` context pipeline.
3. Injects that state as a protocol-valid synthetic tool observation.
4. Preserves the original system prompt outside the compacted state.
5. Preserves the current user request from runtime context as a non-compressible
   goal anchor.
6. Preserves the live protocol tail only when there is an unconsumed
   `AIMessage.tool_calls -> ToolMessage` sequence after the latest user goal.
7. Preserves available tools and the original tool choice.
8. Lets the ReAct workflow continue from the compressed state.

The compacted observation explicitly says it is historical working state, not a
final-answer instruction. The model may continue calling tools when the compact
state does not cover the user's current request.

The active agent no longer wires the generic `SummarizationMiddleware` before
state compaction. That middleware is still available as a module, but the main
agent path avoids chat-summary messages as an intermediate representation for
tool-heavy ReAct work.

## Design

### Trigger
`AgentStateCompactionMiddleware` estimates the full model request size using the same
request trace representation used by observability:

- system prompt
- message history
- tool definitions and schemas

Compaction triggers when:

- estimated request size exceeds the configured context threshold, currently
  `85%` of the model context window, and
- the history contains compressible messages beyond the latest user message.

If the only message is the latest user request, the guard does not compact
because there is no historical state to preserve.

### Current User Goal Anchor
The current user request is stored in request-scoped runtime context as
`current_user_input`, with `current_user_input_sha256` available for trace
correlation. During compaction this value is treated as the active user goal and
is preferred over the latest `HumanMessage` found in the message list.

This distinction matters because upstream framework features or model-specific
prompt templates can introduce summary-shaped human messages such as:

```text
Here is a summary of the conversation to date:
```

Those messages are useful as historical summaries, but they are not the user's
current task. If the raw suffix does not already contain the runtime user goal,
the compactor inserts the goal as a raw `HumanMessage`. If the suffix contains
only a single summary-like human message, it is replaced by the runtime goal.

This keeps the compacted request renderable by prompt templates that require a
user message, while preventing the agent from losing the original task after
multiple tool calls and compaction passes.

### Turn-Aware Partitioning
Before building the compact state, the guard partitions messages into:

- a compressed prefix, containing older turns; and
- a raw suffix, containing the latest user goal and, when present, the active
  tool-call protocol tail.

The implementation deliberately avoids keeping previous complete turns raw. A
complete old turn can carry stale tool calls, old user goals, and large tool
results back into the next model request. Instead, the raw suffix is limited to:

- the latest `HumanMessage`, replaced with the runtime `current_user_input`
  when available; and
- the latest unconsumed assistant tool-call sequence after that user message.

If a later assistant message has already consumed the tool result, that
assistant/tool sequence is not kept raw. It is folded into the compressed
prefix. This handles long one-shot ReAct workflows such as "query many dates and
then summarize" without letting every intermediate ReAct step accumulate in the
live prompt.

### Layered Context Pipeline

After budget pressure triggers, `src/summarization/context_pipeline.py`
constructs the transient compacted request in stages:

1. L1 zero-cost trimming and L2 ReAct trace trimming build the deterministic
   base request via `build_context_compaction_request()`.
2. For each oversized ToolMessage, first restore the complete tool result into
   the transient ledger and re-estimate the request. If that lossless view fits,
   the pipeline returns `l3_lossless_preserved` without calling a summary model.
3. Only when the complete tool results do not fit, run L3 per-tool semantic
   compression. JSON is split only between complete records and text only
   between complete lines.
4. Run L4 local semantic summary if the L3 request is still over budget.
5. Run L5 global fallback summary if L4 is still over budget.

The pipeline uses partitioned context assembly rather than appending summary
text directly to ordinary history. Protected todo state, recent raw messages,
the active user goal, deterministic history ledger, local semantic summaries,
and global fallback summary are treated as separate partitions and rendered into
one protocol-valid synthetic context observation.

L3 summary prompts live in `src/prompt/tool_summary.py`. The prompt is scoped to
one tool result and does not receive the global user goal, so it cannot
reinterpret one partial result as the completion state of the whole task. L4/L5
prompts live in `src/prompt/context_summary.py` and receive only a bounded
history view, not the original full transcript.

Semantic summaries are intentionally free-form visible text, not strict parsed
JSON. This makes the compression path compatible with more local models: the
system treats any non-empty visible model content as the summary body. If the
model returns only reasoning or an empty visible result, the shared semantic
summary capability is marked unavailable for the current process. The current
request falls back to the latest deterministic compacted view, and later
requests skip L3-L5 semantic model calls instead of repeating the same failure.
Trace events distinguish this `context_summary_unavailable` fallback from a
transient `context_summary_error`.

Semantic summary calls are cached with a process-local LRU cache keyed by the
summary model identity and normalized summary messages. This keeps repeated
compaction of the same historical view stable across model calls and avoids
paying the summary cost again when the input has not changed.

Todo is protected state. It is not included in L3-L5 summary inputs. The final
context ledger still receives the bounded todo snapshot so status and ordering
survive.

L5 is the strict fallback. When it activates, the synthetic context ledger pair
is still protocol-valid, but the deterministic history ledger body is no longer
rendered into the model request. Instead, the observation contains a
`deterministic_history_omitted` notice with aggregate counts and the
`global_fallback_summary`. The model may continue using recent raw messages,
the latest user goal, protected todo state, and the global fact list, but it
must not claim access to omitted tool-observation details.

### Layered Context State
The compressed prefix is represented as a generic layered context state with
three layers:

1. historical user messages;
2. assistant visible execution state; and
3. tool observations.

The original system prompt is never moved into the compact state. The active
runtime user goal is also kept outside the compact state.

Historical user messages are stored as compact cards containing:

- source message index
- role
- visible text summary
- original character count
- content hash
- truncation flag

Assistant messages are stored similarly, but only visible assistant content is
included. Reasoning blocks are deliberately excluded from the compact state.
Assistant tool-call requests are kept as structured metadata so the model can
see which actions were attempted without preserving long natural-language
assistant text.

### Observation Ledger
Historical `ToolMessage` objects are converted into generic observation cards.
Each card contains:

- `tool_name`
- `tool_call_id`
- `args`
- `status`
- `result_shape`
- `result_preview`
- `result_stats`
- `content_sha256`

The ledger is business-agnostic. It does not understand airfare, dates, prices,
routes, or any domain-specific field names. It summarizes JSON-like results by
structure and generic statistics such as array lengths, numeric min/max values,
short string samples, null paths, and boolean counts.

### Budgeting Strategy
The layered compactor first budgets the tool-observation ledger, then keeps
historical user and assistant cards when space allows.

The fallback order is:

1. Preserve all tool observation cards with previews.
2. Trim tool previews and keep shape/stat summaries.
3. Trim historical user and assistant visible-text previews.
4. Drop older assistant cards if the layered state is still too large.
5. Drop older historical user cards if needed.
6. Drop the oldest observation cards only when the compact tool ledger itself is
   still too large.

This avoids the earlier "keep only the latest N facts" behavior. A 10-day or
20-day batch query should still show that each successful tool call happened,
even if older raw results are summarized aggressively.

### Compacted Request Shape
After compaction, the request keeps the original execution surface:

```text
system_prompt: original system prompt

messages:
  1. raw live suffix, including the runtime current user goal and optional active protocol tail
  2. AIMessage: synthetic tool call to context_observation_ledger
  3. ToolMessage: layered compact state for compressed older turns

tools: original tools
tool_choice: original tool_choice
```

The synthetic tool observation is protocol-valid: the injected `ToolMessage`
always has a matching preceding `AIMessage.tool_calls[].id`. The guard does not
insert an orphan `ToolMessage` directly into the message list.

The raw live suffix is kept before the synthetic ledger pair so model prompt
templates that require an early user query can still render the compacted
request. The synthetic ledger remains part of the same request and supplies the
compressed historical working state after the live user context is visible.

Example shape:

```text
HumanMessage(runtime current user goal)
optional AIMessage(tool_calls=[active real tool call])
optional ToolMessage(tool_call_id=active real tool call id)
AIMessage(tool_calls=[context_observation_ledger])
ToolMessage(tool_call_id=same id, content=layered compact state)

tools: original tools
tool_choice: original tool_choice
```

The compact state message tells the model:

- this is historical working state, not a final-answer instruction;
- continue following the original system prompt and current user request;
- available tools may still be called when needed;
- do not repeat tool calls with the same successful arguments unless the user
  asks to refresh or fill a gap;
- do not fabricate tool results that are not represented in the ledger;
- if observations were dropped, older tool state was removed due to budget.

## Trace and UI Behavior
Each executed compression stage is emitted only when that stage actually runs.
Skipped L4/L5 stages are not shown as completed work.

Layer events use:

```text
context_compaction_layer
```

Semantic model calls use:

```text
context_summary_start
context_summary_end
context_summary_error
context_summary_unavailable
```

The final compaction event remains:

```text
react_context_budget_compacted
```

The event now represents state-preserving compaction rather than forced
final-answer fallback. It records:

- `compaction_mode="layered_context_state"` for the current implementation
- `estimate_chars`
- `threshold_chars`
- `observation_count`
- `preserved_observation_count`
- `dropped_observation_count`
- `preview_truncated_count`
- `old_user_message_count`
- `preserved_old_user_message_count`
- `dropped_old_user_message_count`
- `assistant_message_count`
- `preserved_assistant_message_count`
- `dropped_assistant_message_count`
- `raw_message_count`
- `compacted_request_chars`
- `original_message_count`
- `compacted_message_count`
- `original_tool_count`
- `compacted_tool_count`
- `todo_snapshot_item_count`
- `todo_snapshot_total_count`
- `todo_snapshot_dropped_count`
- `todo_snapshot_truncated_count`
- `compaction_level`
- `semantic_summary_count`
- `semantic_summary_failed`
- `global_fallback_used`
- `deterministic_ledger_included`
- `post_compaction_chars`
- `still_over_budget`
- `compacted_state_preview`
- `compacted_state_sha256`
- `compacted_prompt_sha256`

`compacted_state_preview` is generated from the full synthetic context ledger
observation shown to the model, not only from the raw observation ledger JSON.
When todo protected state is present, the preview can therefore show the
`todo_snapshot` section as well as the layered historical state.

The UI execution summary displays this as one "上下文压缩" panel and explains
that the agent can continue calling tools after compression. The chat execution
panel groups one compaction trigger as a single expandable item, for example:

```text
上下文压缩：L3 工具结果语义压缩
  1. L3 摘要调用：工具结果
  2. L3 层完成
  3. 压缩结果：L3 工具结果语义压缩
```

If the pipeline reaches L4 or L5, the same grouping keeps the chronological
order and labels the internal steps with their L-level.

## Current Assessment

The current strategy meets the main design goal: context pressure no longer
forces the agent into a final-answer-only mode. The model receives a compacted
working-state view and can continue tool use when more facts are required.

The strongest parts of the design are:

- compaction is transient and does not mutate checkpoint history;
- the original system prompt, latest user goal, tools, and tool choice remain
  outside the compressed state;
- the synthetic context ledger is protocol-valid because every synthetic
  `ToolMessage` has a matching synthetic `AIMessage.tool_calls` entry;
- L3 first tries lossless preservation before spending a semantic summary call;
- semantic summaries are cached and free-form, reducing both repeated cost and
  model-compatibility failures;
- Todo state is protected and injected as a compact snapshot only when
  compaction triggers.

The remaining risks are:

- budgeting is still character-based, so it can differ from the model's true
  tokenizer;
- a reasoning-only local summary model can disable semantic summaries for the
  process, causing fallback to deterministic compression;
- free-form summaries are easier for local models but less machine-checkable
  than strict JSON;
- L5 can still report `still_over_budget=true` in very large sessions because
  even the anchor-only prompt may be too large;
- if omitted details are needed for a precise final answer, the agent must call
  a tool again or clearly state the missing detail.

## Alternatives Considered

### Forced Final-Answer Compaction
Pros:
- Simple and safe when the model is close to the context limit.
- Prevents further tool-call loops.
- Produces a user-visible answer instead of failing from context overflow.

Cons:
- Interrupts unfinished work.
- Disables tools even when more facts are required.
- Changes the agent from an executor into a summarizer at the exact point where
  it may need to continue acting.

Rejected as the default behavior. It may still be useful later as a hard
fallback when state-preserving compaction cannot fit within budget.

### Keep Only Recent Messages
Pros:
- Easy to implement.
- Preserves the latest conversational surface.

Cons:
- Drops successful older tool calls.
- Can make the model believe earlier requested work was never completed.
- Performs poorly for batch workflows where many tool calls are equally
  important.

Rejected because batch factual retrieval needs broad coverage, not only recency.

### Compress Everything and Re-Append the Tail After the Last Tool
Pros:
- Easy to implement.
- Preserves the latest visible working edge after a tool result.

Cons:
- Can duplicate recent assistant/tool messages: once inside the compact ledger
  and once as raw messages.
- Does not align with user-turn boundaries.
- Can still exceed budget if the post-tool tail is large.

Rejected after implementation review. The current approach partitions by human
turns before compaction, then compresses only the older prefix.

### Domain-Specific Summary
Pros:
- Could create more compact summaries for airfare-specific data such as date,
  route, quote count, and price range.

Cons:
- Couples context-budget middleware to air-ticket business logic.
- Does not generalize to other tools, skills, or MCP-backed capabilities.
- Risks losing unknown but important fields.

Rejected for the generic guardrail layer. Domain-specific summarization can be
added later inside specific tools or providers if needed.

## Consequences

- Long ReAct turns can continue after compaction instead of being forced to
  answer prematurely.
- Tools remain available after compaction.
- The current user request is anchored outside message-history summaries.
- The compressed prompt is less likely to lose successful historical tool calls.
- The latest user goal and active tool protocol tail can remain raw while older
  context is compressed.
- Complete previous turns are no longer duplicated inside the compact ledger and
  raw suffix at the same time.
- The model gets explicit guidance not to repeat identical successful calls.
- The guardrail remains generic and does not depend on air-ticket fields.
- State compaction is now a memory-management behavior, not a final-answer
  behavior.

## Known Limitations

- L5 semantic compaction now omits deterministic ledger details as a strict
  fallback, but it still uses character-based estimates and can report
  `still_over_budget=true` if even the anchor-only global fallback view remains
  too large.
- The ledger uses character-based budgeting, not tokenizer-level accounting.
- The raw suffix policy is intentionally strict. It keeps the latest user goal
  and active protocol tail, not arbitrary recent turns. This avoids stale
  context duplication but can make the compacted view rely heavily on the
  synthetic ledger for older conversational nuance.
- Free-form semantic summaries are easier for local models but cannot be
  validated as strongly as structured JSON summaries.
- Observation previews are lossy by design. If a final answer needs full raw tool
  output that was trimmed, the agent may need to call a tool again or explain the
  missing detail.
- Duplicate tool-call prevention remains a separate guardrail. Its fallback
  language may still ask the model to use previous results and answer, but that
  path is independent from context compaction.

## Follow-Up Work

- Consider whether `still_over_budget=true` should trigger a stricter
  final-resort policy for extremely large sessions.
- Improve retention of recent valid `AIMessage(tool_calls) -> ToolMessage`
  sequences if future workflows need to preserve active tool-call structure.
- Consider an optional tokenizer-backed estimator for models where character
  estimates are too loose.
- Expose compacted ledger details more clearly in the debug trace page.
- Consider removing the old `ContextBudgetGuard` compatibility name after
  downstream imports have moved to `AgentStateCompactionMiddleware`.
