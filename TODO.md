# TODO

## Runtime / Agent

### `execute_python`: model-side Codex alignment
Status: deferred

Current state:
- UI/event layer is now Codex-like: `execute_python` emits a dedicated `pythonExecution` item lifecycle and streams stdout/stderr to the frontend.
- Runtime semantics remain synchronous tool-in-turn.
- The model still only receives the final tool result after execution completes.

Deferred follow-up:
- Evaluate whether `execute_python` should evolve beyond a normal `StructuredTool` return and become a more runtime-managed execution primitive.
- If pursued, prefer a staged approach:
  1. Return a more structured execution summary to the model instead of only a prompt-friendly free-text blob.
  2. Only after that, evaluate whether execution lifecycle should become a first-class runtime item on the model side as well.

Explicit non-goal for now:
- Do not convert default `execute_python` into a detached async job/polling model.
- Keep shared `adata` and shared Python variable scope as the default behavior.
