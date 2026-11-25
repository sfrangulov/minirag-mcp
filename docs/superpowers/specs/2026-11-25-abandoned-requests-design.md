# Abandoned requests stop costing the server — design

2026-11-25. A small change; approach chosen by the owner the same day.

## Why

A client that gives up on a call should stop costing the server. Skach, the owner's live
interview assistant, searches the `aida` index every few seconds and abandons a search
after 5 s. Each abandoned `find_mentions` still ran to the end: 3–6 s of CPU on a
1068-document index. The next search arrived on top of it, the process sat at 98 % CPU,
and every abandoned call logged an `AssertionError: Request already responded to`
traceback. Skach is fixing its own side (shorter and fewer literal lookups). This change
is the server's side, and it serves every client.

## How it works today

Every tool is a synchronous function behind `guard`. FastMCP runs it with
`anyio.to_thread.run_sync`, whose cancellation waits for the thread to finish. When the
client sends `notifications/cancelled`, the SDK answers "Request cancelled" at once and
cancels the handler's scope. But the handler only notices when the thread returns, and
then it answers a second time, which trips the assert in `mcp/shared/session.py`. If the
handler raises the cancellation instead, the SDK suppresses the duplicate on its own
(`mcp/server/lowlevel/server.py`, the `get_cancelled_exc_class()` branch).

The signal is `notifications/cancelled`. mcp ≥ 2 clients, Skach's 2.3 among them, send it
when they abandon a call (`cancel_on_abandon`). A client that only closes its session sends
nothing the server acts on: the handler is not cancelled, and the work runs to the end.
That case is follow-up minirag-mcp-m32, not this change.

## Goals

1. A cancelled call ends at once: no second response, no traceback.
2. A cancelled read stops its work at its next checkpoint.
3. A call cancelled while it waits for a worker thread never starts.

## Non-goals

- Interrupting one LanceDB query, or one HTTP call to an endpoint, while it runs.
- A cap on concurrent searches per index. Once abandoned work stops, the pile-up has no
  source. Revisit only if a measurement says otherwise.
- Making `find_mentions` itself faster.

## Anti-goal

A write that stops halfway. Ingest, delete, sync and label writes get no checkpoints.
Abandoned, they run to the end in their thread, as they do now; only the handler stops
waiting.

## Approaches

- **A (chosen).** `guard` becomes async. It runs the synchronous body with
  `run_sync(..., abandon_on_cancel=True)`. On cancellation it sets a `threading.Event`
  and re-raises. The event reaches the thread through a `ContextVar`, which `run_sync`
  copies into the worker. The core's read loops call `checkpoint()`, which raises when the
  event is set. Outside a server call the variable is unset and `checkpoint()` does
  nothing, so the CLI is unchanged.
- **B.** An async core end to end. That means rewriting the store and the pipeline, out
  of proportion to the problem.
- **C.** A per-index semaphore alone. It bounds the pile, but abandoned work still burns
  the CPU that live calls need.

## Design

- A new module holds the event variable, `checkpoint()`, and the exception it raises.
  The exception must not be turned into a `ToolError` by `guard`.
- Checkpoints go between units of read work, never inside a write:
  - per Arrow batch in `Store.literal_candidates`, which is the `find_mentions` scan;
  - between the stages of `run_query`: embedding, first-stage search, rerank,
    look-alikes and parents.

  The exact spots are the builder's call.
- `guard` keeps its signature passthrough, because FastMCP builds the tool schema from
  it. It also keeps `_ANSWERING` and the failure mapping. The HTTP request context still
  reaches the thread through the same copied context.
- Behaviour is the same over HTTP and stdio, since both go through `guard`.

## Tests

Each test must fail on the current code before the change makes it pass.

- A read is cancelled through a real MCP client while its scan is in progress. The scan
  stops within about a second. Nothing logs "Request already responded to".
- `checkpoint()` does nothing outside a call, and raises inside a cancelled one.
- A write that is cancelled still completes.

## Left to the builder

- How the test cancels a call: the in-memory client, or a raw `ClientSession` over the
  ASGI app.
- Where exactly the checkpoints go.

## Delivery

The owner's `serve` runs the installed uv tool, so the fix reaches it with the next
release (2.0.0a3). Publishing is the owner's call. Until then, Skach's side alone removes
the live timeouts.
