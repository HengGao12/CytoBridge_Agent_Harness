# Scalable Tuning And Failure Handling

Use this reference when a trial is slow, memory-heavy, blocked by review, or no
longer improving.

## Scalability Discipline

Treat scalability as a first-class biological requirement.

- Inspect cells per time point before choosing a panel.
- Implement for real biological scale even when debugging on tiny simulations.
- Avoid full `n_t x n_{t+1}` dense cost, affinity, bias, clone-mask, kernel,
  logits, or transport-plan state on medium/large real datasets.
- Use true mini-batch, chunking, streaming, GPU-vectorized, sparse, landmark, or
  coreset approximations when the proposal permits them.
- A `use_mini_batch` flag or `chunk_size` setting does not prove scalability if
  the implementation still builds full adjacent-pair state before chunking.
- Preview/training OOM or timeout is evidence of an implementation scalability
  bug, not an infrastructure issue to bypass.

The accepted approximation changes the estimator/state representation, not the
scientific evidence boundary.

## OT/UOT Pattern

For large adjacent time gaps:

- split `(t_k, t_{k+1})` into bounded source/target chunks, sparse candidates,
  landmarks, coresets, or another bounded-memory representation
- solve balanced OT or UOT only inside bounded solver-scale blocks
- sample endpoint pairs from resulting transport mass
- sample source rows proportional to row mass, then targets conditionally within
  the selected row
- do not downsample by default; observed cell-count changes across time may be
  real proliferation/death or total-mass evidence, and more valid cells usually
  improve coupling and training signal
- if debugging forces a smaller dataset, downsample with the same ratio per time
  point, not a fixed number per time point
- treat smaller panels as smoke/debug evidence unless a larger or full prepared
  real-data run confirms the same claim; formal biological Stage 1/2/3/final
  evidence should use the full prepared real dataset whenever computationally
  feasible

For balanced chunks, normalize local source/target masses. For UOT chunks, keep
the sliced source/target mass vectors and let the solver handle local mass
relaxation; sampled row mass should still define the terminal mass-ratio target
used by the growth head.

References:

- `CytoBridge-main/CytoBridge/tl/flow_matching_backends.py`
- `CytoBridge-main/docs/runtime/flow-matching/README.md`
- `CytoBridge-main/docs/runtime/flow-matching/extension-points.md`

When implementing or debugging chunked OT/UOT/WFR, inspect the package source
before writing new coupling code:

- `BalancedOTCouplingStrategy` for balanced OT-CFM/SF2M-style endpoint pairing
- `UnbalancedOTCouplingStrategy` for VGFM/CRUFM-style UOT endpoint pairing
- `WFROETCouplingStrategy` for WFR-FM semi-coupling and mass targets
- `_split_transport_chunks(...)` for adjacent-time chunk partitioning
- `_solve_uot_from_pairwise_cost(...)` and
  `_solve_balanced_ot_from_pairwise_cost(...)` for sampler-compatible chunked
  solver metadata

Use the builtin configs under `CytoBridge-main/CytoBridge/configs/` to confirm
which `flow_matching.backend`, `coupling.use_mini_batch`, and `chunk_size`
fields activate those paths.

## Coupling Quality Before Blind Tuning

Builtin baselines are fixed comparators during a custom algorithm lifecycle.
New campaigns default to `strict_all_builtin`, where gates compare against the
full default builtin comparator set. Explicit shorter builtin baseline lists on
refresh are repair scopes for named missing/stale baselines; they do not weaken
the gate, and missing required comparators remain blockers. Use `permissive`
only when the user explicitly asks for the older flexible baseline mode. Do not tune builtin baselines by default; tune the
candidate algorithm. Refresh a builtin only when baseline evidence is
missing/stale, a config path is wrong, or concrete abnormal evidence exists. If
an agent-proposed OT/UOT/WFR method has abnormal W1/TMV, mass collapse, NaNs, or
an unexplained gap to nearby baselines, inspect the candidate coupling quality
before more optimizer sweeps:

- resolved config and dataset-scoped builtin overrides;
- feature/cost scale and finite cost entries;
- solver convergence, total plan mass, row/column marginal coverage, and plan
  sparsity or collapse;
- chunk or mini-batch equivalence to the intended adjacent-time coupling;
- terminal mass targets implied by the coupling;
- whether coupling construction runs once in setup or repeatedly in the epoch
  hot path.

Data-sensitive builtin knobs to check, when builtin tuning is justified, include
VGFM/CRUFM UOT `reg`, `reg_m`, and regularization strategy; WFR-FM `delta` or
auto-delta settings; chunk size; batch size; and schedule length. Use manual
`run_training(...)` or preview for diagnostic degenerations and ablations; do
not promote those diagnostics as campaign candidates unless the proposal is
honestly revised.

## Epoch And Initialization Fairness

- Builtin flow-matching methods use 3000 epochs by default.
- Custom flow-matching configs should start from the seeded 3000-epoch default.
- Do not reduce epochs merely to satisfy the wall-clock budget.
- A shorter schedule is acceptable only after convergence evidence shows it is
  enough from scratch.
- Do not fine-tune from an already trained run, cached fitted state, or
  target-specific precomputed predictions to make a trial look fast.
- Warm starts are allowed only when the approved proposal defines them as part
  of the algorithm and they would be available on every new dataset under the
  same data contract.

If runtime is too high, first improve semantics-preserving efficiency:
chunking, streaming, GPU/vectorization, caching invariant costs, or parallelizing
independent per-gap work.

## GPU OT/UOT/WFR Coupling Trick

When the bottleneck is POT coupling construction, check whether the solver is
accidentally running on CPU:

- `ot.unbalanced.sinkhorn_unbalanced(...)` supports torch tensors on CUDA and is
  the package pattern used by UOT/VGFM-style coupling.
- `ot.unbalanced.mm_unbalanced(...)` also supports torch tensors on CUDA and can
  accelerate strict WFR-OET/WFR-FM coupling without changing the WFR objective.
- Keep source masses, target masses, and cost blocks as CUDA tensors for the
  solver call. Convert only the returned bounded plan/subplan back to NumPy if
  the package sampler needs CPU storage.
- A CPU-bound process with GPU utilization near 0% during coupling precompute is
  usually evidence that the implementation converted the solver inputs to NumPy
  too early.

Do not conclude that a large `chunk_size` is impossible only because the current
CPU implementation times out. First compare against the nearest builtin
implementation lifecycle:

- cost blocks that can be expressed as torch tensor operations should be built
  on the target device in bounded row/chunk blocks;
- the expensive OT/UOT/WFR solve and invariant coupling metadata should happen
  in `build_state` or equivalent setup whenever the approved method permits it;
- the epoch hot path should usually only sample from cached bounded plans or
  lightweight metadata, not rebuild pairwise costs or solve OT/UOT/WFR again;
- if the custom method only changes pairwise geometry, keep the builtin
  chunk-splitting and solver/sampler lifecycle and replace only the cost module.

Shrinking `chunk_size` is a last-resort diagnostic, not the first explanation.
If builtin methods can use a much larger chunk on the same dataset, investigate
GPU/vectorized cost construction, precomputed coupling state, and sampler
metadata before declaring a proposal-level scalability failure.

This is a safe acceleration only when the solver and objective are unchanged.
Changing from exact/mm WFR-OET to entropic Sinkhorn, changing KL placement, or
changing marginal constraints is an algorithmic approximation and must be
justified against the approved proposal.

Setup-time timeout is also a scalability failure. A custom method can avoid the
epoch hot path and still be unusable if `build_state` performs too many nested
Python loops, repeated solver calls, or oversized block-pair constructions. When
preview chain inspection times out before a smoke epoch:

- verify the nearest builtin-degenerate path first, with the new component
  neutralized where the proposal permits it;
- profile setup by adjacent time gap and by component: cost, solver, sampler
  metadata, mass target construction, path pretraining;
- reduce algorithmic setup complexity with vectorized batched blocks, sparse or
  sampled candidate pairs, cached per-gap tensors, and fewer Python-level loops;
- only after setup completes should you tune training hyperparameters.

## Stage 2 Simulations

Prefer existing benchmark datasets. Create a new Stage 2 simulation only when:

- the proposal claim cannot be fairly tested by existing benchmarks
- the simulation isolates a specific proposal mechanism
- the expected result and claim metric are observable from rollout artifacts
- builtin/reference baselines can run on the same frozen `simulation_version`
- the reason for not using existing datasets is recorded
- the claim metric is meaningful for the proposed mechanism and includes a
  claim-absent, shuffled, collapsed, or other control where the metric should
  fail

Do not create a simulation because W1/TMV is bad, the gate is hard, or a weaker
synthetic case would be easier to pass.

## Failure Handling

- If `implementation_map_required`, fill and self-check `IMPLEMENTATION_MAP.md`.
- If `implementation_review_required`, wait for the read-only evaluator and fix
  code or patch/review the proposal before retrying.
- If `inference_review_required`, wait for the read-only inference evaluator.
  This should be triggered by inference-surface changes such as
  `simulation_hook(...)` or `inference_context_builder(...)`, not by
  claim-metric-only edits.
- If a rejected branch still looks promising, use `resume_rejected_trial(...)`;
  it does not change active best unless the new trial promotes.
- If several trials fail to promote, stop blind search and read
  `../tuning-playbook/SKILL.md`.

Use `risk.md` as a severity-ordered diagnosis queue. Inspect active-best
model/run artifacts, compare against relevant builtin/reference baselines, and
make visual diagnostics such as observed-vs-predicted time slices, rollout
trajectories, mass/weight curves, and claim-metric residual plots.
