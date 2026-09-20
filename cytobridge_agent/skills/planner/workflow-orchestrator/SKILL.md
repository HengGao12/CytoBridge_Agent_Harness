---
name: workflow-orchestrator
description: Read this first for the standard CytoBridge workflow. It is the high-level routing guide that tells you which stage-specific skill to read next.
---

# Workflow Orchestrator

Use this skill as the top-level routing guide for the normal CytoBridge workflow:

1. preprocessing
2. theory selection
3. training
4. downstream analysis
5. report authoring

This skill is intentionally not the detailed execution manual for every stage. Its job is to:

- identify the current workflow phase
- decide the next required artifact
- tell you which stage-specific workflow skill to read next
- prevent phase skipping, redundant reruns, or stale-artifact reasoning

## Skill Loading Discipline

For the normal CytoBridge workflow, this should be the first workflow skill you read.

After reading this skill:

- identify the current stage first
- read exactly the stage-specific workflow skill needed for that stage
- do not preload many stage skills at once just because they are available
- load auxiliary skills only when the current stage genuinely needs them

Examples:

- if you are still deciding whether preprocessing is complete, read only `preprocessing-execution`
- if the workflow has already reached training, read `training-orchestration` at that point
- if you later move from training to downstream analysis, then read `downstream-analysis`

The goal is to keep context narrow and stage-correct. Do not read five workflow skills up front and then mix their rules together.

## Task Profiling

Before substantial work, set a multi-axis task profile instead of inventing a
single task label in free text.

Use:

- `set_task_profile(...)`
- `check_workflow_gate(...)`

Why:

- many real tasks mix reproduction, tuning, integration, and downstream work
- a single `task_mode` is too lossy
- the gate tool should see which layers are actually being changed

At minimum, declare:

- `current_stage`
- `primary_goal`
- `change_axes`

Typical `change_axes` examples:

- config-only tuning:
  - `{"data_contract": false, "solver": false, "coupling": false, "path": false, "mass": false, "loss": false, "evaluation": false, "downstream": false}`
- paper reproduction with solver/path semantics:
  - `{"solver": true, "path": true, "mass": true}`
- downstream-only biological interpretation:
  - `{"downstream": true}`

## Custom Algorithm Experiment Registry

For custom algorithm work under:

- `~/.cellcompass/training_algorithms/<algorithm_id>/`

do not treat the workspace as an untracked scratch folder anymore.

There is now a file-backed experiment registry under:

- `~/.cellcompass/training_algorithms/<algorithm_id>/registry/`

Key files:

- `algorithm_registry.json`
- `proposals/<proposal_id>.json`
- `decisions.jsonl`
- `runs/<run_id>.json`
- `obsolete.jsonl`
- `workspace_snapshots/<snapshot_id>/`

Use the registry as the source of truth for:

- active proposal
- active workspace snapshot
- active baseline run
- latest accepted run
- recent decisions
- obsolete experimental evidence

Do not reason only from current file contents when the task is a custom
algorithm lifecycle question.

## Active Algorithm Context

Keep the agent-side model simple:

- `active_algorithm_context` is the default authoring/editing target
- `activate_algorithm_workspace(...)` is the way to change that target
- read-only inspection of another algorithm does not change the active target
- workspace writes must stay under the active algorithm workspace
- workspace edits make that algorithm dirty until a clean snapshot is created
- campaign tuning creates snapshots automatically in `run_campaign_trial(...)`
- manual review or `run_training(...)` requires a clean active snapshot; if `dirty_since_snapshot=true`, call `snapshot_active_algorithm_workspace(...)` first
- approving a proposal automatically activates that algorithm as the authoring target
- `run_training(training_algorithm_id="...")` uses the explicit training target and does not switch the active authoring target
- proposal review is version-bound by `proposal_id`, not by whatever file happens to be latest on disk

So if you need to edit B while A is active, first call:

- `activate_algorithm_workspace(algorithm_id="B")`

If you only need to inspect B while continuing to edit A, just use:

- `list_experiment_history(algorithm_id="B")`
- file read tools

Before major custom-algorithm actions, inspect registry state first with:

- `list_experiment_history(...)`
- `check_workflow_gate(...)`
- `preview_training_run(...)`

When needed, mutate registry state with:

- patch the algorithm `PROPOSAL.md` with `apply_workspace_patch(...)` to create a reviewed proposal revision
- `record_decision(...)`
- `mark_result_obsolete(...)`
- `rollback_algorithm_workspace(...)`
- `set_active_baseline_run(...)`
- `compare_algorithm_runs(...)`

## Critical Rule

When you are about to execute a workflow stage, **first read the corresponding stage skill** with normal file tools. Do not rely on memory or this overview alone.

Read only the skill for the current stage unless the task genuinely requires a second stage or a specialized auxiliary skill.

Required stage-skill mapping:

- paper or article data intake / candidate dataset discovery
  - `~/.cellcompass/skills/workflow/paper-to-adata-intake/SKILL.md`
  - Use this when the user asks which datasets from a paper are analyzable, where repository accessions can be downloaded, or whether paper resources can be materialized.
- preprocessing
  - `~/.cellcompass/skills/workflow/preprocessing-execution/SKILL.md`
- theory selection
  - `~/.cellcompass/skills/workflow/theory-selection/SKILL.md`
- training
  - `~/.cellcompass/skills/workflow/training-orchestration/SKILL.md`
- downstream analysis
  - `~/.cellcompass/skills/workflow/downstream-analysis/SKILL.md`
- downstream figure/report-facing visualization
  - `~/.cellcompass/skills/workflow/downstream-analysis/SKILL.md`
- report writing or report revision
  - `~/.cellcompass/skills/workflow/report-authoring/SKILL.md`

If you have not read the stage skill in the current run, do that first.

Do not jump ahead to later-stage skills until the current stage decision is settled or the workflow has actually advanced.

## Primary Objective

Move from raw or partially prepared temporal snapshot data to validated biological conclusions with the fewest correct steps, while preserving artifact lineage and avoiding unnecessary recomputation.

## Research Question Framing Integration

Not every theory-selection task starts with "choose a model family from known options."

Sometimes the real bottleneck is earlier:

- the scientific question is still vague
- the motivation is generic
- the gap statement is not venue-specific
- the user wants a biology-journal problem framing versus an ML-topconf framing
- the user wants a few plausible algorithm direction families without committing to a concrete method
- the user still has not clarified whether the question really belongs inside the CytoBridge package frame at all

In those cases, load:

- `~/.cellcompass/skills/planner/research-question-framing/SKILL.md`
- `~/.cellcompass/skills/planner/idea-lifecycle/SKILL.md`
- `~/.cellcompass/skills/workflow/theory-selection/references/package-capability-map.md`

## Idea vs Algorithm Boundary

Do not route every research-planning task into `idea` work.

Use the `idea` path only when the **problem itself** is still under-specified.

That means at least one of these is still unclear:

- what exact scientific question is being asked
- what exact ambiguity or failure mode is worth solving
- what venue-style framing is intended
- what counts as success
- whether the task is even worth turning into a persistent research asset

Typical signals that the task should go through `research-question-framing` and possibly `idea-lifecycle`:

- the user asks for a good idea, motivation, or gap
- the user asks what problem is worth solving
- the user wants biology-journal vs ML-topconf framing
- the user only has a broad direction, not a concrete question
- the user wants to explore a few feasible directions before committing
- the user has not yet shown that the target problem belongs to deep-learning continuous generative dynamics on multi-timepoint single-cell snapshots

Do **not** use the `idea` path when the problem is already clear enough and the real task is to design a solution.

Skip straight to algorithm work when the user already provides a concrete target such as:

- a specific scientific question
- a specific ambiguity to resolve
- a concrete dataset/task setting
- a clear success criterion
- a request to improve, extend, or replace an existing method for that problem

In those cases:

- do not spend turns inventing a new `idea`
- do not force creation of a persistent research idea asset
- move directly to method-level reasoning

Typical direct-to-algorithm cases:

- "for this clearly defined snapshot dynamics problem, design a better method"
- "improve RUOT / FM / SB for this setting"
- "propose an algorithm for separating transport from growth under these assumptions"
- "the question is fixed; now choose the theory and design the method"

Preferred routing rule:

- if the question is vague, unstable, or venue-framing-dependent:
  - `research-question-framing`
  - calibrate against `workflow/theory-selection/references/package-capability-map.md`
  - then `idea-lifecycle` if it is worth persisting
- if the question is already sharp and the task is solution design:
  - skip `idea-lifecycle`
  - go directly to:
    - `~/.cellcompass/skills/workflow/theory-selection/SKILL.md` when choosing among existing CytoBridge families
    - `~/.cellcompass/skills/algorithm/proposal-theory/SKILL.md` when proposing or justifying a new algorithm
    - `~/.cellcompass/skills/algorithm/algorithm-orchestrator/SKILL.md` when the proposal is stable enough to implement

Short test:

- if the next useful artifact is a **problem statement**, use the `idea` path
- if the next useful artifact is a **method proposal**, skip the `idea` path

Use `research-question-framing` before normal theory selection when the task is to:

- formulate a paper-worthy scientific question
- separate biology-journal framing from ML-topconf framing
- sharpen a motivation or gap statement
- propose bounded, feasible idea directions rather than a full algorithm

Routing rule:

- if the question itself is unstable, read `research-question-framing` first
- before creating a persistent idea, check the current package boundary with `workflow/theory-selection/references/package-capability-map.md`
- before creating a persistent idea, check that the core object belongs to the CytoBridge frame: deep-learning continuous generative dynamics rather than generic traditional bioinformatics
- if framing is good enough to become a persistent problem asset, then read `idea-lifecycle` and create/review the idea before method design
- if the question is already fixed and the task is choosing or justifying a method family, continue with normal theory selection and do not detour through `idea-lifecycle`
- if the question is already fixed and the task is to propose a concrete new algorithm, go directly to `proposal-theory`
- if framing stabilizes and the next issue is recoverability or theoretical soundness, then read `~/.cellcompass/skills/algorithm/proposal-theory/SKILL.md`

## Literature Retrieval Integration

### Determining Whether Literature Retrieval is Needed

Before executing any stage that may require literature support, check:

1. **Stage type**: Theory selection and downstream analysis ALWAYS require literature retrieval.
2. **Workflow state**: Check if literature has already been retrieved for the current context.
3. **User request**: If user explicitly requests a new analysis or comparison, retrieval may be needed even if previously done.

### Literature Retrieval Decision Logic

```python
def needs_literature_retrieval(stage: str, state: dict) -> bool:
    """Determine if literature retrieval is needed for the current stage."""
    
    # Stages that always require literature
    lit_required_stages = ["theory_selection", "downstream_analysis"]
    
    if stage not in lit_required_stages:
        return False
    
    # Check if we already have literature results in state
    if "literature_results" in state.get("artifact_index", {}):
        # Check if retrieval is still valid (e.g., dataset hasn't changed)
        last_retrieval = state["artifact_index"]["literature_results"]
        if last_retrieval.get("dataset_id") == state.get("dataset_id"):
            return False  # Reuse existing results
    
    return True
```

### Literature Retrieval Workflow

When literature retrieval is needed:

**Step 1: Load literature citation skill**
- read_file(file_path='~/.cellcompass/skills/planner/literature-citation/SKILL.md')

**Step 2: Execute literature retrieval**
- Construct search queries based on dataset characteristics and stage requirements
- Call `search_literature(query=constructed_query)`
- Store results with citation guidelines in context

**Step 3: Record retrieval in workflow state**
- commit_workflow_state(
    phase="literature_retrieval",
    literature_results={
      "queries": ["query1", "query2"],
      "citations": ["citation1", "citation2"],
      "dataset_id": current_dataset_id,
      "timestamp": "current_time"
    }
  )

**Step 4: Proceed to target stage** with literature context loaded

## Core Workflow Logic

### 1. Determine current workflow state
Inspect the following state variables:

- `workflow_phase` - Current phase name
- `phase_status` - Status of current phase (not_started, in_progress, completed)
- `artifact_index` - Dictionary of all generated artifacts
- `preprocessed_path` - Path to preprocessed data
- `preprocessing_script_path` - Saved script that reproduces the final preprocessing/materialization path
- `preprocessing_script_sha256` - Hash of the saved preprocessing script when available
- `preprocessing_input_paths` - Raw/materialized inputs consumed by the preprocessing script
- `preprocessing_output_paths` - Stable `.h5ad` and sidecar outputs produced by preprocessing
- `candidates` - Selected model candidates from theory selection
- `training_runs` - Record of training runs
- `final_config` - Final model configuration used
- `final_metrics` - Final evaluation metrics
- `downstream_results` - Results from downstream analysis
- `downstream_figures` - Generated figures
- `report_path` - Path to generated report
- `literature_results` - Literature retrieval results

Always reason from current state and actual artifacts, not from assumptions about what "should already exist".

### 2. Determine the next required artifact

For the user's request, ask:

1. What is the next missing artifact?
2. Which stage produces it?
3. Has that stage already been completed and validated?
4. Does this stage require literature retrieval, and has it been done?

**Artifact-to-Stage Mapping:**

| Missing Artifact | Required Stage | Literature Needed? |
|------------------|----------------|-------------------|
| No valid `preprocessed_path` | preprocessing | No |
| No defensible model family/candidate | theory selection | Yes |
| No trained model / valid final run | training | No |
| No evidence-backed downstream outputs | downstream analysis | Yes |
| No figures for report | downstream visualization | No |
| No report or outdated report | report authoring | No |

### 3. Handle literature retrieval if needed

If the identified stage requires literature and retrieval hasn't been done:

1. Load literature citation skill
2. Execute `search_literature` with appropriate queries
3. Commit results to state using `commit_workflow_state`
4. Then proceed to the target stage

### 4. Read the stage skill before acting

Once the next stage is identified and literature needs are satisfied, read that stage's `SKILL.md` and follow it.

This is mandatory because the detailed contracts, preferred execution style, common failure modes, and required commit fields now live in the stage skills, not in this orchestrator.

## Reuse vs Rerun Policy

Reuse a prior stage result only if all of the following are true:

- the relevant artifact exists
- it matches the current dataset/model context
- validation passes
- the user did not explicitly request a rerun, comparison, or change in method
- for literature results: dataset hasn't changed and queries are still relevant

Rerun when:

- the dataset changed
- the artifact is missing or stale
- validation fails
- the user explicitly requests a rerun/comparison
- an upstream change invalidates downstream artifacts
- literature needs updating (new queries, different focus)

Do not rerun just to feel safe. Validate first.

## Phase Transition Policy

Workflow phase should be advanced only when a concrete stage result has been produced and committed.

Do not treat "I have a plan" or "I think it should be done" as completion.

When a stage is actually complete, call:

- `commit_workflow_state(...)`

The exact fields to commit are defined in the corresponding stage skill.

For custom algorithm work, also update the experiment registry, not just workflow
state.

Minimum expectations by phase:

- proposal:
  - proposal record exists in `registry/proposals/`
- authoring:
  - active workspace snapshot exists
- review:
  - decision log reflects approval / rejection / revision
- training:
  - run is registered in `registry/runs/`
  - campaign tuning trials have an automatic promote / reject decision
  - direct manual runs have an explicit legacy decision if they are used as evidence
- rollback / evidence invalidation:
  - obsolete records and decision log are updated

## What This Skill Should Not Do

This skill should not be treated as:

- the preprocessing manual
- the theory-selection manual
- the training manual
- the downstream manual
- the report-writing manual

Those are separate workflow skills. Read them first.

## Read-First References

Use these when the interface or artifact contract is unclear:

- package API map:
  - `CytoBridge-main/docs/INDEX.md`
- custom training workflow:
  - `CytoBridge-main/docs/runtime/custom-algorithms/README.md`
- flow-matching extension API:
  - `CytoBridge-main/docs/runtime/flow-matching/README.md`

If the package interface is still unclear after reading the relevant stage skill, inspect the docs/source directly instead of guessing.

## Common Failure Modes

- jumping into a stage without first reading that stage skill
- skipping theory selection and going straight to training without a justified model choice
- skipping literature retrieval when it's required
- generating downstream narrative before generating downstream artifacts
- writing or revising the report before artifact paths and evidence are stable
- rerunning whole stages even though validation would have shown reuse was possible
- committing partial or vague state updates that do not correspond to real artifacts

## Minimal Working Discipline

For every workflow turn:

1. inspect current state
2. set/update `task_profile`
3. determine the next required artifact
4. run `check_workflow_gate(...)` for the intended stage
5. check if the target stage requires literature retrieval
6. if needed, execute literature retrieval and commit results
7. identify the stage that produces the artifact
8. read that stage skill
9. execute the minimal correct work
10. commit the real result

That is the entire purpose of this skill.
