# Literature Reading Order For Idea Work

This folder is a packaged copy of the curated CytoBridge literature survey.

Use it when the agent needs field familiarity for:

- framing a research idea
- reviewing whether an idea is sharp enough
- selecting a theoretical direction with literature awareness

## Default Order

1. Read [synthesis.md](synthesis.md).
   - Use this first for the field map, major trends, and unresolved gaps.
2. Read [timeline.md](timeline.md) if chronology matters.
   - Use this when you need to know what changed across years or why a method mattered at its time.
3. Read a **small** number of specific notes under [notes/](notes/).
   - Prefer 2-6 notes that are directly tied to the current question.
4. Read the package capability map:
   - `~/.cellcompass/skills/workflow/theory-selection/references/package-capability-map.md`
   - Use this to understand what CytoBridge already implements before you claim a new gap.
5. Check existing persistent research assets before proposing something new.
   - Use `list_research_ideas(...)`
   - Use `get_research_idea_status(...)`
   - Use `get_algorithm_proposal_status()`
   - Use `list_experiment_history(...)` when a related algorithm already exists
6. Use `search_literature` / RAG only after the note-level summaries, package-boundary check, and existing-asset check are no longer enough.
   - Use RAG for targeted citation recovery, metadata checks, or broader retrieval.
7. Read the original PDFs only for the few papers whose exact method, theorem, or limitation now matters.

Do not start by reading all copied notes.  
Do not start by reading concrete source code either.  
At idea stage, the intended order is:

`overview -> selected notes -> package boundary -> existing ideas/proposals -> RAG -> PDF`.

## Suggested Starting Sets

### If the task is `biology-journal` idea framing

Start with:

- [synthesis.md](synthesis.md)
- [2018 RNA velocity](notes/2018__journal__nature__rna_velocity_of_single_cells.md)
- [2019 WOT](notes/2019__journal__cell__optimal_transport_analysis_of_single_cell_gene_expression_identifies_developmental_trajectories_in_reprogramming.md)
- [2020 scVelo](notes/2020__journal__nature_biotechnology__generalizing_rna_velocity_to_transient_cell_states_through_dynamical_modeling.md)
- [2021 PRESCIENT](notes/2021__journal__nature_communications__prescient.md)
- [2021 LineageOT](notes/2021__journal__nature_communications__lineageot.md)

Then branch out to:

- spatial / interaction: `SpaceFlow`, `stlearn_psts`, `stt`, `FlowSig`
- multiview / fate aggregation: `CellRank 2`, `moslin`
- recent systems: `moscot`, `genetrajectory`, `scdiffeq`, `MultistageOT`

### If the task is `ml-topconf` idea framing

Start with:

- [synthesis.md](synthesis.md)
- [2018 dynamic UOT](notes/2018__math__journal_of_functional_analysis__unbalanced_optimal_transport_dynamic_and_kantorovich_formulations.md)
- [2020 TrajectoryNet](notes/2020__math__icml__trajectorynet.md)
- [2021 Deep Generative Learning via Schrödinger Bridge](notes/2021__math__icml__deep_generative_learning_via_schrodinger_bridge.md)
- [2023 Conditional Flow Matching](notes/2023__math__arxiv__conditional_flow_matching.md)
- [2023 Generalized Schrödinger Bridge Matching](notes/2023__math__iclr__generalized_schrodinger_bridge_matching.md)
- [2024 Flow Matching on General Geometries](notes/2024__mlconf__iclr__flow_matching_on_general_geometries.md)
- [2024 Metric Flow Matching](notes/2024__mlconf__neurips__metric_flow_matching_for_smooth_interpolations_on_the_data_manifold.md)

Then branch out to:

- partial observability
- mean-field / interaction-aware dynamics
- unbalanced stochastic OT / RUOT
- geometry-aware or manifold-aware transport / flow

## What The Copied Notes Are Good For

Use the copied notes for:

- problem formulation
- identifying real failure modes of prior work
- seeing what previous papers actually contributed
- quickly checking whether a direction is already crowded or still open

Do **not** treat the copied notes as a substitute for full-paper verification when:

- a theorem statement matters
- an engineering detail determines feasibility
- a venue / publication-status claim must be exact
- you need to quote or attribute a narrow limitation precisely

## File Map

- [synthesis.md](synthesis.md): high-level field synthesis
- [timeline.md](timeline.md): chronological literature timeline
- [candidate_registry.csv](candidate_registry.csv): structured survey registry
- [inventory_summary.md](inventory_summary.md): survey scope summary
- [web_verification/publication_status_20260417.md](web_verification/publication_status_20260417.md): publication-status checks
- [notes/](notes/): per-paper reading notes
