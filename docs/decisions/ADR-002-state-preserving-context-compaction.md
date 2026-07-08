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
context-budget behavior for the active agent.

When the request approaches the configured context budget, the agent now:

1. Splits the message history on user-turn boundaries.
2. Compresses older turns into a layered context state.
3. Injects that state as a protocol-valid synthetic tool observation.
4. Preserves the original system prompt outside the compacted state.
5. Preserves the current user request from runtime context as a non-compressible
   goal anchor.
6. Preserves a small number of recent turns as raw messages when they are still
   useful working context.
7. Expands generic batch-tool results into per-task observation cards.
8. Preserves available tools and the original tool choice.
9. Lets the ReAct workflow continue from the compressed state.

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
- a raw suffix, containing the most recent user turns.

A turn starts at a `HumanMessage` and includes following assistant/tool messages
until the next `HumanMessage`. By default, the guard keeps the latest two user
turns raw and compresses older turns. This avoids the earlier bug where the same
recent assistant/tool messages could appear both in the compacted ledger and as
raw messages.

For single-user-turn histories, the latest `HumanMessage` is kept raw, while the
execution messages after that user request can still be compressed. This handles
long one-shot ReAct workflows such as "query many dates and then summarize".

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

### Batch Tool Results
Generic batch execution returns one outer `ToolMessage`, but that outer message
can represent many completed sub-tasks. Treating it as one opaque JSON blob made
the compacted state hard for the model and debug UI to inspect.

When a tool result has the generic batch shape:

```text
batch_id
summary
results[]
limitations
```

the observation builder creates a batch-aware card:

- outer shape records that this is a batch result;
- summary counts are preserved;
- each sub-task keeps `task_id`, `tool_name`, `status`, `args`, compact
  `result_shape`, compact `result_stats`, bounded preview, and result hash;
- essential-mode compaction still keeps those sub-task cards before dropping the
  whole outer observation.

The compactor still does not interpret business fields inside task args or
results. It only recognizes the generic batch envelope so one batch call can
remain traceable as many completed tool actions.

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
  1. raw recent turn messages, including the runtime current user goal
  2. AIMessage: synthetic tool call to context_observation_ledger
  3. ToolMessage: layered compact state for compressed older turns

tools: original tools
tool_choice: original tool_choice
```

The synthetic tool observation is protocol-valid: the injected `ToolMessage`
always has a matching preceding `AIMessage.tool_calls[].id`. The guard does not
insert an orphan `ToolMessage` directly into the message list.

Raw recent turns are kept before the synthetic ledger pair so model prompt
templates that require an early user query can still render the compacted
request. The synthetic ledger remains part of the same request and supplies the
compressed historical working state after the recent user context is visible.

Example shape:

```text
HumanMessage(...)
AIMessage(...)
ToolMessage(...)
HumanMessage(runtime current user goal)
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
The trace event remains:

```text
react_context_budget_compacted
```

The event now represents state-preserving compaction rather than forced
final-answer fallback. It records:

- `compaction_mode="state_preserving"`
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
- `compacted_state_preview`
- `compacted_state_sha256`
- `compacted_prompt_sha256`

The UI execution summary displays this as "上下文状态压缩" and explains that the
agent can continue calling tools after compression.

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
- Batch tool calls remain inspectable at the sub-task level after compaction.
- Recent user turns can remain raw while older turns are compressed.
- The same recent raw messages are no longer duplicated inside the compact
  ledger.
- The model gets explicit guidance not to repeat identical successful calls.
- The guardrail remains generic and does not depend on air-ticket fields.
- State compaction is now a memory-management behavior, not a final-answer
  behavior.

## Known Limitations

- There is not yet a second-level hard fallback if the state-preserving compacted
  request is still too large.
- The ledger uses character-based budgeting, not tokenizer-level accounting.
- The recent-turn retention rule currently keeps a fixed number of recent user
  turns raw. A future version may adapt this count based on remaining budget.
- The batch-aware compactor recognizes a generic `batch_id` / `summary` /
  `results` envelope, so other batch tools should reuse that shape for best
  compaction behavior.
- Observation previews are lossy by design. If a final answer needs full raw tool
  output that was trimmed, the agent may need to call a tool again or explain the
  missing detail.
- Duplicate tool-call prevention remains a separate guardrail. Its fallback
  language may still ask the model to use previous results and answer, but that
  path is independent from context compaction.

## Follow-Up Work

- Add a hard fallback mode only when state-preserving compaction still exceeds
  the request budget.
- Add tests for compacted requests that remain over budget.
- Improve retention of recent valid `AIMessage(tool_calls) -> ToolMessage`
  sequences if future workflows need to preserve active tool-call structure.
- Consider an optional tokenizer-backed estimator for models where character
  estimates are too loose.
- Expose compacted ledger details more clearly in the debug trace page.
- Consider removing the old `ContextBudgetGuard` compatibility name after
  downstream imports have moved to `AgentStateCompactionMiddleware`.
