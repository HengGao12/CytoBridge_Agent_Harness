# CytoBridge Agent

CytoBridge Agent is a single-agent scientific workflow runtime for time-resolved single-cell data analysis on top of `CytoBridge-main`.

It is designed for datasets with biologically meaningful time points and supports:

- preprocessing and AnnData sanity checks
- theory/model selection for dynamical OT / flow-matching style modeling
- model training and evaluation
- downstream analysis and figure generation
- report writing
- custom training algorithm authoring inside `~/.cellcompass/training_algorithms`
- registry-backed experiment management for custom training algorithms

The current runtime is `runtime_v2`: one agent loop, one shared state tree, one conversation history, one Web timeline.

## Repository layout

- `CytoBridge-main/`
  - core package: modeling, training, downstream analysis, docs
- `cytobridge_agent/`
  - agent runtime, CLI, Web server, prompts, skills, tools
- `web/`
  - frontend

## Installation

Install the agent and the core package in editable mode:

```bash
cd /lustre/home/2501111653/CytoBridge-agent
pip install -e .
pip install -r requirements-agent.txt

cd CytoBridge-main
pip install -e .
```

If you use the Web UI, build the frontend once:

```bash
cd /lustre/home/2501111653/CytoBridge-agent/web
npm install
npm run build
```

## Benchmark Evaluation

```bash
# without harness
bash baseline.sh

# with harness
bash harness.sh
```


## Runtime model

This repository no longer uses the old multi-agent runtime as the main path.

The main runtime is:

- single LLM-driven agent
- LangGraph-backed checkpointing
- event-based Web timeline restore
- local skills as docs/catalog rather than prompt-injected subagents

Recent runtime additions:

- active algorithm context with registry-backed proposal, snapshot, dirty-workspace, review, and campaign state
- autoresearch-style algorithm campaigns with staged trial budgets, archived trials, automatic promote/reject decisions, baseline gates, and final locking
- benchmark dataset registry plus builtin baseline records and reusable saved trajectories
- isolated custom training subprocesses with training/inference timeout and memory-preflight protection
- boundary-based context compaction with microcompact and post-compact rehydrate
- guarded terminal, Web search/fetch, slash commands, stop hooks, and provider-aware LLM configuration
- theory/literature RAG, including separate theory-book search support
- Web UI improvements for timeline events, setup/provider configuration, markdown/LaTeX rendering, and tool-call commentary display

The main code path starts here:

- `cytobridge_agent/session_controller.py`
- `cytobridge_agent/interactive.py`
- `cytobridge_agent/runtime_v2/agent.py`
- `cytobridge_agent/runtime_v2/tool_registry.py`

## CLI usage

The preferred command is `cellcompass`; `cytobridge-agent` is the same entry
point name for environments that already install it.

Common entrypoints:

```bash
# Start an interactive terminal session.
cellcompass

# Start the Rich terminal UI (falls back to plain output if Rich is unavailable).
cellcompass tui

# Run one non-interactive turn and exit.
cellcompass exec "Analyze this dataset"

# Shorthand for one non-interactive turn.
cellcompass "Analyze this dataset"

# Resume from an interactive history picker.
cellcompass resume
cellcompass resume --grep packer

# Resume the latest saved session directly.
cellcompass continue
cellcompass c
cellcompass continue --grep packer

# Inspect saved sessions.
cellcompass session list --limit 20
cellcompass session list --grep packer --json
cellcompass session show <session_id>
cellcompass session tail <session_id> --last 50

# Check runtime health without opening the Web UI.
cellcompass doctor
cellcompass doctor --json

# Summarize runtime event counts and recorded durations.
cellcompass perf report --session <session_id>

# Inspect active/stale local runtime jobs.
cellcompass jobs --active
cellcompass jobs --stale
```

Inside interactive mode, use `/help` for slash commands. `/resume` with no
arguments opens the same numbered history picker, and `/resume --last` resumes
the latest saved session.

The terminal commands share one session-opening path: interactive sessions,
one-shot `exec` turns, `/resume`, `continue`, and slash-command resume all use
the same saved-session selection and resume logic. This keeps CLI behavior
consistent as the terminal UI evolves separately from the Web UI.

Runtime health and profiling are also exposed through slash commands:
`/doctor` checks local conversation storage, job registry, and UI dependencies;
`/perf [session_id]` summarizes saved timeline events for bottleneck triage;
`/jobs` lists active runtime jobs.
The Web API exposes the same runtime state at `/api/runtime/status`,
`/api/runtime/doctor`, and `/api/runtime/perf/{session_id}`.
`/status`, `cellcompass session show`, and `/api/runtime/status` include a
canonical `snapshot` object with separate `runtime`, `session`, `model`,
`workflow`, `data`, `algorithm`, `artifacts`, and `jobs` sections. Saved
conversation snapshots are marked as `runtime.status=saved`, so supervisors do
not confuse checkpoint state with a live agent process.
For custom algorithm work, `snapshot.algorithm.lifecycle` is the canonical
lifecycle state machine view: it treats Stage 1/2/3 promotion as progress, but
only `algorithm_lifecycle_status=complete` plus a final-regression
`locked_release` counts as complete.
Campaign state is resolved registry-first: the campaign JSON registry is the
source of truth for stages, gates, stage panels, locked releases, and lifecycle
status; compact `active_algorithm_campaign` values in session state are only a
fallback and their source is reported in `snapshot.algorithm.campaign_resolution`.
Active sessions also carry an ownership record in
`~/.cellcompass/runtime_sessions.json`; a live session is not silently taken over
by another process unless the previous owner is stale or in the same process.
The Web server uses FastAPI lifespan startup/shutdown hooks to install runtime
callbacks, restore matplotlib hooks, release session ownership, and clean up
managed training subprocesses.

Tool execution is a runtime safety boundary. Python/script tools and training
already run in managed subprocesses, and the runtime wraps high-risk
read/search/web/terminal tools in a child-process guard. If a guarded tool
raises `MemoryError`, times out, segfaults, or is killed by the kernel/OOM
killer, the parent agent process receives a normal tool-result error with the
return code/signal instead of crashing. State-mutating workflow tools still run
in-process so session and AnnData bindings are preserved, but they catch
`BaseException` and return recoverable tool errors.

AnnData handling follows the same rule: the main agent process binds active
`.h5ad` paths but does not eagerly load full AnnData objects during session
resume or `load_or_switch_adata`. Python analysis, scripts, training, and
downstream workers read the bound path inside their isolated subprocesses. If a
dataset is too large or corrupt, the worker fails and returns diagnostics while
the agent runtime stays alive.

Every runtime tool declares a machine-checkable contract in its metadata:
`cytobridge_isolation`, `cytobridge_mutates_state`,
`cytobridge_reads_adata`, `cytobridge_writes_artifacts`, and
`cytobridge_may_use_gpu`. Only `process` and `managed_subprocess` tools declare
`cytobridge_timeout_sec` and `cytobridge_timeout_enforced=true`; in-process
state tools deliberately leave `cytobridge_timeout_sec=null` because interrupting
them mid-state-update would be unsafe.

## Algorithm benchmark artifact bundle

The repository intentionally does not commit benchmark `.h5ad` files, saved
trajectories, or baseline checkpoints. They are distributed as a release asset
described by the bundled manifest.

Install the published benchmark bundle into the runtime default location:

```bash
cellcompass benchmarks install
```

If the GitHub repository or release is private, provide access through
`GITHUB_TOKEN`, `GH_TOKEN`, or the normal git credential store for `github.com`;
the installer uses those credentials for release downloads.

Install from a local bundle, which is useful on clusters without GitHub release
access:

```bash
cellcompass benchmarks install \
  --asset /path/to/cellcompass-algorithm-benchmarks-2026-05-03.tar.zst
```

Inspect the manifest or build a maintainer bundle from a prepared local
`~/.cellcompass/algorithm_benchmarks` tree:

```bash
cellcompass benchmarks manifest
cellcompass benchmarks build-bundle \
  --output /path/to/cellcompass-algorithm-benchmarks-2026-05-03.tar.zst
```

The bundle is installed under `~/.cellcompass/algorithm_benchmarks`, so existing
agent tools can find registered datasets, leaderboards, saved trajectories, and
baseline checkpoints without retraining.
New benchmark datasets should register the preprocessing routine with
`preprocessing_script_path`; the registration tool copies that script under the
dataset directory and records its SHA256 in `dataset.json` for later audit and
gene-level downstream analysis.

## Data contract

The standard workflow assumes:

- `adata.obs["time_point_processed"]`
  - canonical processed time field
- `adata.obsm["X_latent"]`
  - canonical transcriptomic backbone used by training

Custom algorithms may read extra `obs`, `obsm`, or `layers` fields, but should not replace those two runtime contracts.

## Global configuration

Persistent local config is stored in:

- `~/.cellcompass/config.json`

Typical example:

```json
{
  "openai_api_key": "your-key",
  "llm_base_url": "https://openrouter.ai/api/v1",
  "llm_model": "gpt-5.4",
  "llm_auth_mode": "codex_oauth",
  "report_format": "html",
  "algorithm_proposal_review_mode": "agent_decide",
  "device": "cuda"
}
```

The agent also stores local runtime state under:

- `~/.cellcompass/conversations/`
- `~/.cellcompass/training_algorithms/`
- `~/.cellcompass/auth_profiles.json`
- `~/.cellcompass/gemini_credentials.json`

## Authentication modes

Supported LLM auth modes:

- `api_key`
  - OpenAI-compatible API via `openai_api_key` and optional `llm_base_url`
- `gemini_oauth`
  - Gemini OAuth flow
- `codex_oauth`
  - OpenAI/Codex OAuth via local sidecar
- `auto`
  - runtime chooses based on config/model

### Codex OAuth

Login:

```bash
cellcompass auth codex-login
```

Inspect stored profiles:

```bash
cellcompass auth profiles list
cellcompass auth profiles status
cellcompass auth profiles usage --all-profiles
```

Enable/disable/reorder profiles:

```bash
cellcompass auth profiles disable <profile_id>
cellcompass auth profiles enable <profile_id>
cellcompass auth profiles order --provider openai-codex --ids id1,id2,id3
```

## CLI

CytoBridge exposes three primary run modes. `cellcompass` is the preferred entry
point; `cytobridge-agent` is only a command alias to the same runtime.

### Inspect a dataset

```bash
cellcompass inspect /path/to/data.h5ad
```

### Direct one-shot mode

Run one agent turn and exit:

```bash
cellcompass "Analyze the available dataset and propose the next workflow"
cellcompass exec "Continue the report" --resume <SESSION_ID>
cellcompass exec "Summarize current status" --resume <SESSION_ID> --json
```

Dataset workflow mode is a structured one-shot prompt into the same unified
runtime used by `exec`, interactive, TUI, and Web:

```bash
cellcompass run /path/to/data.h5ad \
  --question "Analyze the main fate branches and driver genes" \
  --output ./output_run \
  --device cuda
```

`input` is optional. You can also start with no dataset and provide a path later in the conversation.

### Interactive mode

```bash
cellcompass
cellcompass interactive /path/to/data.h5ad \
  --question "Check the dataset first, then propose a workflow" \
  --output ./output_interactive
```

Resume or inspect existing chat sessions:

```bash
cellcompass resume <SESSION_ID>
cellcompass resume --last
cellcompass session list
cellcompass session status <SESSION_ID>
cellcompass session tail <SESSION_ID> --last 50
```

Interactive sessions support slash commands including `/help`, `/new`,
`/resume`, `/session`, `/status`, `/stop`, `/model`, `/provider`,
`/reasoning`, `/compact`, `/hooks`, and `/quit`.

### Web UI

```bash
cellcompass web --host 0.0.0.0 --port 8000
```

Then open:

- `http://localhost:8000`

Useful startup overrides:

```bash
cellcompass web \
  --input /path/to/data.h5ad \
  --output /path/to/output \
  --device cuda \
  --llm-model gpt-4o
```

## Common runtime options

Some frequently used flags:

- `--llm-model`
- `--llm-base-url`
- `--llm-api-key`
- `--llm-auth-mode {auto,api_key,gemini_oauth,codex_oauth}`
- `--llm-profile-id`
- `--report-format {html,md}`
- `--disable-multimodal`

Notes:

- HTML is the default report format
- multimodal mode allows pasted/uploaded images into the chat context

## Custom algorithm experiment registry

Custom algorithms under:

- `~/.cellcompass/training_algorithms/<algorithm_id>/`

now maintain a local experiment registry:

- `registry/algorithm_registry.json`
- `registry/proposals/`
- `registry/decisions.jsonl`
- `registry/runs/`
- `registry/obsolete.jsonl`
- `registry/workspace_snapshots/`

This registry tracks:

- active proposal / snapshot / baseline run
- proposal revisions and supersession
- training run verdicts and promotion/rejection
- obsolete proposals or runs that should no longer be used as evidence
- latest persisted training preview artifacts

The registry is the source of truth for custom algorithm experiment history. Conversation state only caches a summary.

## Training preview and inspection

`preview_training_run` is the main dry-run tool for training inspection.

It now does three things at once:

- resolves the effective training target and merged config
- performs training-data and backend preflight checks
- records the effective runtime call chain, including the concrete objects and source file/line locations actually used by the run

Each preview is persisted as:

- `<output_dir>/.runtime/previews/*.preview.md`
- `<output_dir>/.runtime/previews/*.preview.json`

For custom algorithms, the preview paths are also written into the experiment registry.

## Web UI features

The Web UI supports:

- session initialization and restore
- switching LLM configuration without discarding the chat
- stop current run
- timeline replay from saved events
- proposal review for custom algorithms
- skill visibility toggles
- training algorithm visibility toggles
- manual context compaction

## Literature RAG

The runtime now exposes:

- `search_literature`
  - retrieval over the local literature knowledge base with citation-backed formatting
- `read_file`
  - unified text / image / PDF reader for direct PDF inspection after retrieval

The RAG stack uses:

- local structured knowledge and precomputed paragraph embeddings from `cytobridge_agent/rag/`
- the main agent LLM client for HyDE expansion and reranking
- isolated RAG prompts, not the main conversation history

Important boundary:

- the repository includes the RAG code, APA citation table, structured knowledge base, and precomputed paragraph embeddings
- the repository does **not** include the large local model download or the raw PDF library in `main`
- `search_literature` is intended to work from the shipped structured data and paragraph indexes
- direct PDF reading with `read_file` still requires the local PDF library to be present under:
  - `cytobridge_agent/rag/literature_db/literature/`

If you want full local-detail reading on a fresh machine, you need three things locally:

- `sentence-transformers`
- `spacy`
- `PyMuPDF`

and, for best performance:

- a local embedding model cache under `cytobridge_agent/rag/all-mpnet-base-v2/`
- a local PDF library under `cytobridge_agent/rag/literature_db/literature/`

## PDF workflow support

The workflow skill catalog now includes:

- `workflow/pdf`

Current PDF support is intentionally pragmatic:

- the unified `read_file(...)` tool can now inspect:
  - text files with `offset` / `limit`
  - image files directly
  - PDFs via rendered page images, with `pages="..."` required for large PDFs
- PDF text extraction is still available through `execute_python` using `pdfplumber` or `pypdf` when structured extraction is needed
- PDF page rendering is still available through `execute_python` plus `pdftoppm` when persistent custom render artifacts are needed
- the environment is currently suitable for PDF reading/review, not guaranteed PDF generation

Current dependency status expected by the skill:

- `pdfplumber`
- `pypdf`
- `Pillow`
- `cairosvg`
- system `pdftoppm` from Poppler

`reportlab` is not treated as a required dependency at the moment because PDF generation is not a primary CytoBridge workflow.

## Context compaction

The runtime compacts context before LLM invocation when prompt tokens exceed a threshold.

Compaction behavior:

- keep the stored transcript append-only
- create a compact boundary and project model-facing history from that boundary forward
- run a cheap microcompact pass before full LLM compaction
- use an isolated LLM compaction prompt with a structured summary contract
- rehydrate active workflow, algorithm, proposal, campaign, plan, and artifact state after full compaction
- keep legacy compacted sessions readable through compatibility projection

Implementation:

- `cytobridge_agent/tools/context_compaction.py`
- `cytobridge_agent/runtime_v2/middleware.py`

## Custom training algorithms

The agent can create custom algorithms under:

- `~/.cellcompass/training_algorithms/<algorithm_id>/`

Each algorithm workspace typically contains:

- `manifest.yaml`
- `config.yaml`
- `algorithm.py`
- `PROPOSAL.md`
- `PROPOSAL.json`
- `risk.md`
- `IMPLEMENTATION_MAP.md`
- `README.md`

Current authoring rules:

1. create and approve a proposal before editing implementation files
2. edit only the active algorithm workspace unless explicitly activating another workspace
3. implement the approved proposal directly; if the implementation needs semantic simplification, revise the proposal first
4. keep inference and metrics based on model-generated trajectories from `t0`, not future observed targets or post-hoc metric hacks
5. use `run_campaign_trial` for algorithm tuning so trials are archived and automatically promoted or rejected
6. use direct `run_training` only for manual diagnostics that should not update campaign active-best state

Campaign trials archive workspace/config state and automatically manage snapshots; agents do not need to manually snapshot before `run_campaign_trial`.

Primary references:

- `CytoBridge-main/docs/INDEX.md`
- `CytoBridge-main/docs/runtime/custom-algorithms/README.md`
- `CytoBridge-main/docs/runtime/flow-matching/README.md`
- `CytoBridge-main/docs/theory/README.md`
- `cytobridge_agent/skills/algorithm/algorithm-orchestrator/SKILL.md`
- `cytobridge_agent/skills/algorithm/proposal-theory/SKILL.md`
- `cytobridge_agent/skills/algorithm/authoring/SKILL.md`
- `cytobridge_agent/skills/algorithm/campaign-tuning/SKILL.md`
- `cytobridge_agent/skills/algorithm/review-and-training/SKILL.md`

## Skills

Skills are local markdown instructions used as structured guidance for the agent.

Main workflow-facing skills live under:

- `cytobridge_agent/skills/workflow/`

Algorithm-design skills live under:

- `cytobridge_agent/skills/algorithm/`

Planner and downstream skills live under:

- `cytobridge_agent/skills/planner/`
- `cytobridge_agent/skills/downstream/`

The important point is:

- skills are local docs/catalog
- they are not the old runtime’s persistent subagent system

## Current architecture summary

The main runtime path is:

1. build prompt from current state
2. register unified tool surface
3. run single-agent turn in LangGraph
4. stream tool / status / artifact events to Web UI
5. checkpoint runtime state and persist conversation timeline

Key files:

- `cytobridge_agent/runtime_v2/agent.py`
- `cytobridge_agent/runtime_v2/graph.py`
- `cytobridge_agent/runtime_v2/prompt_builder.py`
- `cytobridge_agent/runtime_v2/tool_registry.py`
- `cytobridge_agent/interactive.py`
- `cytobridge_agent/web_server.py`

## Testing

Common verification steps for this repo are:

```bash
conda run -n agent python -m unittest cytobridge_agent.runtime_v2.proposal_versioning_test -v
conda run -n agent python -m unittest cytobridge_agent.runtime_v2.algorithm_campaign_test -v
conda run -n agent python -m unittest cytobridge_agent.runtime_v2.training_isolation_test -v
```

Frontend build check:

```bash
cd web
npm run build
```

## Current limitations

- The workflow is intended for multi-time-point snapshot data with meaningful temporal structure.
- `obs["time_point_processed"]` and `obsm["X_latent"]` remain fixed runtime contracts.
- direct PDF reading via `read_file` depends on a local raw PDF library and is not guaranteed to work on a clean clone unless that library is provisioned separately.
- The main branch carries structured literature data and paragraph indexes, not the full raw document archive or local embedding model weights.
- Custom algorithms are powerful, but they are still expected to reuse the package training flow rather than replace it wholesale.

## Minimal example

```bash
cellcompass run \
  --question "Run preprocessing, choose a model, train it, perform downstream analysis, and write a report." \
  --output ./output_demo \
  --llm-model gpt-4o
```
