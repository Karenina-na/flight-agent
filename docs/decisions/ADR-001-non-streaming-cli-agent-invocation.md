# ADR-001: Use Non-Streaming Invocation for the CLI Agent

## Status
Accepted

## Date
2026-07-07

## Context
The local CLI agent originally used `agent.stream(..., stream_mode="messages")`
to render assistant responses incrementally. During local LM Studio testing, the
air-ticket tool path repeatedly failed when the user asked:

```text
查一下北京的机场叫什么
```

The failing trace showed that the model selected the correct tool
`resolve_flight_locations`, but emitted function-call arguments as an empty
string. LangChain then parsed the call as empty tool arguments:

```json
{"name":"resolve_flight_locations","arguments":""}
```

which became:

```json
{"args":{}}
```

Because `locations` is a required argument, the tool failed with:

```text
locations: Field required
```

The stream-based run then repeated the same invalid tool call many times in the
same turn. A representative trace showed:

- 59 `resolve_flight_locations` function calls.
- 59 empty `arguments` values.
- 59 tool validation errors.
- No successful location resolution.

After switching the CLI path to non-streaming `agent.invoke(...)`, the same user
request produced a single valid tool call:

```json
{"locations":["北京"]}
```

The tool completed successfully and returned the FlyClaw-backed airport facts:

```json
{
  "items": [
    {
      "input": "北京",
      "airport_codes": ["PEK", "PKX"],
      "default_airport": "PEK",
      "display_name": "北京",
      "source": "flyclaw"
    }
  ]
}
```

This indicates the earlier issue was most likely caused by the interaction
between streaming function calling and the local OpenAI-compatible model/runtime,
rather than by the air-ticket service layer, FlyClaw integration, or tool
registration.

## Decision
Use non-streaming `agent.invoke(...)` as the default CLI execution path for the
MVP air-ticket agent.

The public CLI helper name remains `stream_agent_response()` for compatibility
with existing tests and commands, but internally it now performs one invoke call
and prints the final assistant message.

The JSON trace dump now records:

- `invoke_output` for the full final agent result.
- `stream_chunks` as an empty list for non-streaming CLI runs.
- model/tool lifecycle events collected through observability middleware.

## Alternatives Considered

### Keep Streaming Invocation

Pros:
- Incremental user-visible output.
- Useful for long-running answers once the model/tool stack is stable.

Cons:
- Local testing produced repeated empty function-call arguments.
- The CLI surfaced noisy partial content and tool errors.
- Debug traces became dominated by repeated invalid calls.

Rejected for the MVP CLI because tool-call correctness is more important than
incremental rendering.

### Add a Tool-Argument Repair Layer First

Pros:
- Could recover from empty tool arguments by inferring values from the user
  message, for example mapping `查一下北京的机场叫什么` to
  `{"locations":["北京"]}`.
- Useful as a future robustness layer.

Cons:
- It would mask the root issue in the model/runtime interaction.
- It adds another behavior layer before the basic invocation path is stable.

Deferred until after the non-streaming path is validated across more scenarios.

### Change the Local Model

Pros:
- A stronger tool-use model may handle streaming function calls correctly.

Cons:
- It depends on local LM Studio configuration and model availability.
- The current codebase still needs a stable default behavior.

Deferred. The CLI should work with the currently configured local endpoint first.

## Consequences

- CLI responses are no longer streamed token-by-token.
- Tool calls are more stable in the tested LM Studio setup.
- Trace files are cleaner and easier to inspect because one turn records the
  final invoke result plus ordered model/tool events.
- If streaming is reintroduced later, it should be guarded by tests or a config
  flag and validated against real tool-calling traces.

## Follow-Up Work

- Restrict the MVP CLI tool surface to air-ticket tools only, instead of also
  exposing skill and memory tools.
- Add repeated empty-argument protection in middleware to prevent tool-call
  loops from flooding logs.
- Keep full debug trace JSON local and ignored by Git; keep runtime logs concise.
- Re-evaluate streaming after testing a model/runtime combination with reliable
  streamed function-call arguments.
