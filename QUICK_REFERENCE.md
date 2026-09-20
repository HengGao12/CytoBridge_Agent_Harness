# CytoBridge Agent Quick Reference

This is the short operational reference for the current runtime.

Current main path:
- single-agent `runtime_v2`
- LangGraph checkpointing
- Web timeline restore from saved events
- local skills as docs/catalog

## Key entry points

- runtime entry:
  - `cytobridge_agent/graph.py`
- runtime implementation:
  - `cytobridge_agent/runtime_v2/agent.py`
  - `cytobridge_agent/runtime_v2/graph.py`
  - `cytobridge_agent/runtime_v2/tool_registry.py`
- Web server:
  - `cytobridge_agent/web_server.py`
- interactive session / persistence:
  - `cytobridge_agent/interactive.py`
  - `cytobridge_agent/conversation_store.py`

## Core data contract

The standard workflow assumes:
- `adata.obs["time_point_processed"]`
- `adata.obsm["X_latent"]`

Custom algorithms may augment these with extra modalities, but should not replace them as the primary runtime contract.

## Main commands

### Run one-shot workflow

```bash
cytobridge-agent run /path/to/data.h5ad \
  --question "Analyze the major lineage branches and driver genes" \
  --output ./output_run
```

### Start interactive mode

```bash
cytobridge-agent interactive /path/to/data.h5ad \
  --question "Inspect the dataset, then propose the correct workflow" \
  --output ./output_interactive
```

### Resume interactive session

```bash
cytobridge-agent interactive --resume <SESSION_ID>
```

### Start Web UI

```bash
cytobridge-agent web --host 0.0.0.0 --port 8000
```

## Auth quick reference

Global config:
- `~/.cellcompass/config.json`

Common auth modes:
- `api_key`
- `gemini_oauth`
- `codex_oauth`
- `auto`

### Codex OAuth login

```bash
cytobridge-agent auth codex-login
```

### Inspect stored OAuth profiles

```bash
cytobridge-agent auth profiles list
cytobridge-agent auth profiles status
cytobridge-agent auth profiles usage --all-profiles
```

## Current tool surface

The runtime agent mainly uses:
- `execute_python`
- `inspect_adata_state`
- `load_or_switch_adata`
- `persist_runtime_adata`
- file read tools:
  - `list_path`
  - `read_file`
  - `find_files`
  - `grep_files`
- planning tools:
  - `set_plan_from_text`
  - `update_plan`
  - `get_plan_status`
- theory:
  - `search_theory`
- training:
  - `run_training`
- workflow commit:
  - `commit_workflow_state`
- custom algorithm workspace tools:
  - `init_training_algorithm_workspace`
  - `list_workspace_tree`
  - `read_workspace_file`
  - `create_workspace_file`
  - `replace_workspace_file`
  - `apply_workspace_patch`
  - `preview_workspace_diff`

## Custom algorithm locations

Agent-created algorithm workspaces live under:
- `~/.cellcompass/training_algorithms/<algorithm_id>/`

Typical files:
- `manifest.yaml`
- `config.yaml`
- `algorithm.py`
- `PROPOSAL.md`
- `README.md`

Primary docs:
- `CytoBridge-main/docs/custom_training_workflow.md`
- `CytoBridge-main/docs/flow_matching_developer_api.md`

Primary skills:
- `~/.cellcompass/skills/planner/algorithm-development/SKILL.md`
- `~/.cellcompass/skills/planner/algorithm-authoring/SKILL.md`
- `~/.cellcompass/skills/planner/algorithm-review-and-training/SKILL.md`

## Context compaction

Current default policy:
- `context_window = 128000`
- `trigger_ratio = 0.82`
- `keep_last_turns = 6`

Behavior:
- keep the first system message
- summarize older history
- keep recent turns
- persist compacted history back into runtime state

Code:
- `cytobridge_agent/tools/context_compaction.py`
- `cytobridge_agent/runtime_v2/middleware.py`

## Where to look when debugging

### Runtime orchestration
- `cytobridge_agent/runtime_v2/agent.py`
- `cytobridge_agent/runtime_v2/graph.py`
- `cytobridge_agent/runtime_v2/prompt_builder.py`

### Tool exposure / workflow tools
- `cytobridge_agent/runtime_v2/tool_registry.py`
- `cytobridge_agent/tools/planner_tools.py`

### Training
- `cytobridge_agent/tools/training_tools.py`
- `CytoBridge-main/CytoBridge/tl/fit.py`
- `CytoBridge-main/CytoBridge/tl/trainer.py`
- `CytoBridge-main/CytoBridge/tl/flow_matching_backends.py`

### Session restore
- `cytobridge_agent/interactive.py`
- `cytobridge_agent/conversation_store.py`
- `cytobridge_agent/runtime_v2/resume.py`

### Web UI
- `cytobridge_agent/web_server.py`
- `web/src/App.jsx`
- `web/src/hooks/useAgentSocket.js`

## Practical warnings

- This workflow is for biologically meaningful multi-time-point snapshot data.
- `pilot` / `final` are mainly run labels and context tags, not separate training pipelines.
- If you want a smaller exploratory run, prepare a smaller dataset explicitly and pass it to `run_training`.
- Custom algorithms should reuse the package training flow rather than forking the trainer.
