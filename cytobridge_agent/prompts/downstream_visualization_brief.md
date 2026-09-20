[PLANNER_VIZ_BRIEF_AUTO]
Visualization intent is auto-inferred for this downstream turn. Treat this block as a rendering contract.

User goal:
{user_goal}

User instruction:
{user_instruction}

Structured Viz Brief (JSON):
```json
{viz_brief_json}
```

Execution contract:
1. Rerun the relevant analysis chain before plotting; do not only reuse old figures.
2. Visualization priority:
   - P1: rerun relevant CytoBridge analysis chain to refresh evidence
   - P2: custom redraw via `execute_python` + `save_pubfig`
   - P3: add extra CytoBridge APIs in `execute_python` when key evidence is still missing
3. Save figures with descriptive names and publication-quality exports.
4. Report policy default: `main_only` (core figure coverage, no appendix auto-generation).
5. If `basis_preference` is present (and especially when `strict_basis=true`), do not switch to another embedding basis silently.
6. Empirical result panels must be generated from locked evidence artifacts:
   final-regression metrics, downstream tables, saved model outputs, or the
   saved `evaluation_trajectory.npz`. Do not use random/synthetic placeholder
   trajectories or invented coordinates in a panel that is captioned or
   discussed as a model result. If a conceptual schematic is useful, label it
   explicitly as a schematic and keep it separate from evidence panels.
7. Trajectory visualizations should preserve artifact provenance in the script
   or caption. Prefer model-native trajectory arrays and package trajectory
   plotting helpers when compatible. If the locked trajectory has outliers,
   show a robust view plus an outlier/support diagnostic instead of silently
   replacing the trajectory with a cleaner synthetic drawing.
[/PLANNER_VIZ_BRIEF_AUTO]
