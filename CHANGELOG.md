# Changelog

## Unreleased

- Reworked custom algorithm development around an authoritative active algorithm context, proposal registry, dirty workspace tracking, and disk-first proposal/campaign state refresh after resume.
- Added algorithm campaign lifecycle tooling: staged panels, trial budgets, archived trials, automatic promote/reject, baseline refresh, claim-metric adapters, Stage 2 simulation registration, Stage 3 guardrails, and final locking.
- Added benchmark dataset management, builtin baseline recording, saved trajectory reuse, and listable baseline artifacts for debugging.
- Added a benchmark artifact manifest plus `cytobridge-agent benchmarks` install/build commands so large datasets, baseline metrics, trajectories, and checkpoints can be distributed outside git history.
- Expanded builtin flow-matching algorithms and docs, including balanced OT-CFM, SF2M, VGFM naming, trajectory-based evaluation, deterministic sigma-zero support, and memory/scalability preflight checks.
- Added custom training isolation in a subprocess with timeout handling so memory or CUDA failures in custom algorithms do not kill the main agent loop.
- Rebuilt compaction around boundary projection, microcompact, LLM summary prompts, and post-compact state rehydration.
- Added guarded terminal command, Web search/fetch, theory-book RAG search, provider-aware LLM configuration, slash commands, stop hooks, and durable checkpointer support.
- Refactored algorithm skills and package docs into clearer algorithm, runtime, theory, and benchmark workflow entrypoints.
- Improved Web UI setup, provider controls, timeline rendering, markdown/LaTeX rendering, tool-call commentary, progress grouping, and event cards.
