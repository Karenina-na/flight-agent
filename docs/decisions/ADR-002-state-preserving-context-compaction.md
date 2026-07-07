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
Use state-preserving context compaction as the default context-budget behavior.

When the request approaches the configured context budget, the agent now:

1. Builds a compact observation ledger from historical tool results.
2. Replaces old verbose history with a compact state summary.
3. Preserves the original system prompt.
4. Preserves available tools and the original tool choice.
5. Preserves the current working edge after the most recent tool result.
6. Lets the ReAct workflow continue from the compressed state.

The compaction prompt explicitly says it is a historical state summary, not a
final-answer instruction. The model may continue calling tools when the ledger
does not cover the user's current request.

## Design

### Trigger
`ContextBudgetGuard` estimates the full model request size using the same
request trace representation used by observability:

- system prompt
- message history
- tool definitions and schemas

Compaction triggers when:

- estimated request size exceeds the configured context threshold, currently
  `85%` of the model context window, and
- historical tool results exist.

If there are no tool results, the guard does not compact because there is no
tool-backed state to preserve in an observation ledger.

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
The compactor tries to preserve one observation card per completed tool call.

The fallback order is:

1. Preserve all observation cards with previews.
2. Trim previews and keep shape/stat summaries.
3. Drop the oldest observation cards only if the compact ledger is still too
   large.

This avoids the earlier "keep only the latest N facts" behavior. A 10-day or
20-day batch query should still show that each successful tool call happened,
even if older raw results are summarized aggressively.

### Compacted Request Shape
After compaction, the request keeps the original execution surface:

```text
system_prompt: original system prompt

messages:
  1. SystemMessage: compacted historical working state + observation ledger
  2. recent messages after the latest ToolMessage

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
- `estimate_chars`
- `threshold_chars`
- `observation_count`
- `preserved_observation_count`
- `dropped_observation_count`
- `preview_truncated_count`
- `compacted_request_chars`
- `original_message_count`
- `compacted_message_count`
- `original_tool_count`
- `compacted_tool_count`

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
- The compressed prompt is less likely to lose successful historical tool calls.
- The model gets explicit guidance not to repeat identical successful calls.
- The guardrail remains generic and does not depend on air-ticket fields.
- State compaction is now a memory-management behavior, not a final-answer
  behavior.

## Known Limitations

- There is not yet a second-level hard fallback if the state-preserving compacted
  request is still too large.
- The ledger uses character-based budgeting, not tokenizer-level accounting.
- The recent-message retention rule keeps messages after the latest tool result;
  more nuanced retention may be needed for complex multi-branch workflows.
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
