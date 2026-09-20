# A Public-Data Scientific Harness for CytoBridge Agent

## Abstract

CytoBridge Agent is an agentic workflow for analyzing single-cell temporal dynamics and producing the six artifacts evaluated by DynBench: velocity fields, growth-rate estimates and growth-driver rankings, held-out cell-state distributions, per-cell fate predictions, perturbation responses, and gene regulatory networks (GRNs). Although the original agent can complete this workflow end to end, a single-pass execution may produce outputs that are structurally valid but scientifically unstable. In particular, global signed-gradient aggregation can obscure state-dependent growth drivers, strong perturbations can cause fate classifiers to respond to out-of-distribution states, and independently generated artifacts may lack cross-task consistency.

This repository adds a public-data-only scientific harness around the CytoBridge execution loop. Before inference, the harness summarizes the public task package and supplies metric-aware analysis constraints. After the first pass, it audits the six outputs for schema validity, numerical consistency, distributional collapse, unstable growth-driver evidence, and unsupported perturbation effects. Detected issues are returned to the same agent session for a configurable revision round. The final outputs then undergo deterministic calibration. Growth drivers are re-ranked using stratified bootstrap ExtraTrees surrogates of the agent's own per-cell growth predictions, with time and fate included as nuisance variables. Perturbation effects are filtered through a robust consensus between normalized fate-change magnitude and outgoing GRN strength, using a median-plus-MAD threshold. Raw artifacts are preserved to maintain provenance and auditability.

The harness does not read DynBench ground truth, historical reference outputs, or evaluator scores. In a replay of one existing DeepSeek run on `S_balanced_easy_01_seed42`, fixing a separate NumPy classifier-loading incompatibility established an effective baseline of `0.800`. The public-data calibration increased M2 growth-driver recovery from `0.333` to `1.000` and M5 perturbation accuracy from `0.786` to `1.000`, yielding a total score of `0.947`. M1, M3, M4, and M6 were unchanged by the scientific calibration. This `+0.147` result is evidence from a single-scenario artifact replay rather than a general benchmark claim; multi-scenario paired ablations remain necessary to estimate the average treatment effect of the complete harness.

**Keywords:** agentic scientific analysis, single-cell dynamics, harness engineering, bootstrap stability, perturbation analysis, gene regulatory networks, robust statistics, DynBench

## Motivation

Scientific agents may produce plausible individual analyses without delivering a coherent and reproducible scientific result. In DynBench, CytoBridge must complete several coupled tasks in one run, including dynamics modeling, growth-driver ranking, fate prediction, perturbation analysis, and GRN inference. A single-pass workflow can therefore preserve unstable rankings, spurious perturbation effects, inconsistent artifacts, or output-contract errors until final evaluation.

The harness was introduced as a public-data-only control layer around this workflow. It makes task requirements explicit before execution, audits the first-pass outputs, returns targeted revision feedback, and applies reproducible calibration to results with identifiable statistical weaknesses. This approach improves reliability without reading hidden ground truth, changing the evaluator, or retraining the underlying CytoBridge model.

The central question is whether public-data diagnostics, stability estimation, and cross-artifact consistency checks can reduce avoidable scientific errors in autonomous analysis. The current implementation provides a testable mechanism for doing so, while its general effectiveness must be established through paired, multi-scenario ablations.

## Research Objective

The harness addresses three failure classes in scientific agent execution:

1. **Context and contract failures.** The agent may overlook time keys, label structure, held-out targets, output schemas, or probability constraints.
2. **Scientific interpretation failures.** Global gradient summaries may cancel state-dependent effects, while strong interventions may turn classifier sensitivity into false causal claims.
3. **Single-pass execution failures.** A plausible first answer may be submitted without checking whether the six outputs are mutually consistent and supported by the fitted model.

The design objective is to reduce these failures using only information available to a legitimate benchmark participant.

## Method

The implemented workflow is:

```text
Public task package
        |
        v
Public-data profile and metric-aware guidance
        |
        v
Initial CytoBridge Agent execution
        |
        v
Scientific and contract audit
        |
        +---- issues detected ----> in-session targeted revision
        |                                  |
        +----------------------------------+
        |
        v
Deterministic M2/M5 calibration
        |
        v
Final audit, schema validation, and DynBench evaluation
```

### End-to-End Execution Protocol

The diagram above summarizes an eight-stage execution protocol.

1. **Run isolation and task staging.** `run_dynbench.py` resolves the requested scenario and creates a new output directory. It copies only public task inputs into `_workspace/`: `train.h5ad`, `TASK.md`, `prediction_targets.json`, the public fate classifier, and the public output verifier. The output directory is reset at the start of the run so stale artifacts cannot be mistaken for newly generated results.

2. **Public profile construction.** If `harness_revisions > 0`, the harness reads `train.h5ad` and writes `_workspace/scientific_harness_context.json`. This operation occurs before the LLM session starts. It summarizes observable data structure but does not infer or import benchmark answers.

3. **Prompt augmentation.** The original task prompt is augmented with the profile path, the six-file delivery contract, workspace restrictions, the public verifier command, and five scientific gates: contract identification, model selection, shared-model consistency, scientific calibration, and artifact validation. The guidance asks the agent to validate interpolation on public pseudo-holdouts, retain distributional diversity, quantify driver stability, use matched perturbation controls, and distinguish upstream regulators from downstream reporters.

4. **Initial agent execution.** `cytobridge_runner.py` creates one `SessionController`, opens the task session, and runs the initial turn. The agent is responsible for fitting its model and materializing all six required files directly in the output directory:

   - `velocity_field.csv`;
   - `growth_rates.csv`;
   - `holdout_prediction.csv`;
   - `per_cell_fate.json`;
   - `perturbation_results.json`;
   - `driver_genes.json`.

   The harness does not synthesize missing primary outputs as a fallback. If the agent cannot train or export its model, the run is reported as failed.

5. **Public scientific audit.** The harness reads the public training data and the six generated artifacts. It computes diagnostics and emits a structured JSON record containing an issue code, severity, evidence, and recommended action for each detected problem. The audit sets `requires_revision=true` whenever at least one issue is present.

6. **Targeted in-session revision.** When revision is required, the issue records are rendered into a new prompt and sent to the same `SessionController`. The agent is instructed to preserve strong artifacts, reuse its fitted model, and regenerate only weak or inconsistent outputs. The audit/revision cycle stops when no issue remains or when the configured revision limit is reached. The default limit is one round.

7. **Deterministic calibration and final audit.** After the revision loop, the harness backs up the raw driver and perturbation JSON files. It then applies the M2 growth-driver calibration and M5 perturbation-GRN consensus rule described below. A final public audit is run on the calibrated outputs, and all actions are written to `scientific_harness_manifest.json`.

8. **Contract validation and scoring.** The benchmark validates that all six deliverables exist, are non-empty, are newly generated, and can be parsed. Only after the harness has finished does the official DynBench evaluator access its evaluation assets and calculate M1-M6 and TOTAL. Harness diagnostics never receive the resulting scores.

With `--harness-revisions 0`, stages 2, 3's harness-specific guidance, 5, 6, and 7 are disabled. This provides the no-harness condition required for a paired ablation.

### Public-Data Profiling

Before the agent runs, `build_public_data_profile()` summarizes `train.h5ad` and the public prediction targets. It records:

- the number of cells and measured genes;
- gene names and the detected time and label columns;
- observed time values and cell counts at each time;
- cell counts for every observed `time x label` stratum;
- global and per-gene expression ranges, means, and standard deviations;
- mean gene expression at each observed time;
- finite-value coverage and the public holdout target specification.

The profile is a deterministic compression of public inputs. It reduces repeated exploratory tool calls and makes critical task variables salient to the agent. It does not add biological knowledge beyond the task package and does not guarantee that the LLM will use the information correctly.

The appended guidance organizes analysis into five gates:

1. **Contract gate:** identify measured and latent spaces, observed times, holdout targets, source cells, classifier feature order, and output schemas.
2. **Model-selection gate:** compare candidate dynamics and growth choices using public evidence, including leave-one-observed-time-out checks when feasible.
3. **Shared-model gate:** derive velocity, rollout, fate, perturbation, GRN, and driver outputs from one coherent fitted model or documented projections.
4. **Scientific-calibration gate:** inspect distributional coverage, driver stability, matched-control perturbations, and edge-sign stability.
5. **Artifact gate:** run the public verifier and inspect numerical and cross-artifact consistency before submission.

### Audit and Revision Loop

Quality control is divided among the public schema verifier, `audit_public_outputs()`, and the final benchmark validator. The verifier checks the public file contract, the scientific audit compares current outputs with public data and cross-artifact evidence, and the final validator rejects missing, empty, stale, or unparseable deliverables.

The scientific audit itself checks for:

- missing required outputs and unreadable driver or perturbation JSON;
- growth-rate alignment with the public training cells;
- non-finite or distributionally collapsed holdout predictions;
- fate predictions collapsed to a single terminal state;
- violations of `delta = perturbed - control`;
- nearly universal perturbation activity;
- disagreement between reported growth drivers and a stable nonlinear surrogate;
- perturbation effects lacking upstream GRN support.

Growth-driver diagnostics combine raw expression-growth Spearman association, partial association after removing time and label effects, the agent-reported ranking, and a stratified nonlinear surrogate. A disagreement issue is raised only when the reported leader differs from a surrogate leader that ranks first in at least 75% of the audit resamples. This avoids triggering revision for every small ranking fluctuation.

Perturbation diagnostics first verify the identity

$$
\Delta_{gk}=p_{gk}^{\mathrm{perturbed}}-p_{gk}^{\mathrm{control}}.
$$

They then calculate an adaptive activity floor and the fraction of candidate genes called active. If at least 80% are active, the output is flagged as potentially unselective. The audit also calculates joint perturbation-GRN support, but it does not consult hidden labels to decide which genes are correct.

For holdout predictions, the median per-gene spread is compared with the training distribution. A spread ratio below `0.1` indicates that the exported particles may have collapsed toward a mean trajectory. For fate predictions, a warning is raised when more than 98% of cells share the same top fate in a multi-fate task.

Each issue is serialized as:

```json
{
  "code": "growth_driver_evidence_disagreement",
  "severity": "warning",
  "summary": "...",
  "evidence": {"reported_top": "...", "stable_surrogate_top": "..."},
  "recommended_action": "..."
}
```

The revision prompt contains these issue records and explicitly asks for model-native recalculation, public pseudo-holdout validation, and preservation of already strong artifacts. Because the same controller and session are reused, the revision can access the model and scripts created during the first pass instead of restarting the analysis. This mechanism provides an opportunity for correction, but an LLM revision is stochastic and is not guaranteed to improve an output.

### Stable Growth-Driver Calibration

Let `X` be the measured gene-expression matrix and `y` the growth rate predicted by the agent for each cell. For bootstrap replicate `b`, the harness samples cells within `time x fate` strata and fits an ExtraTrees surrogate:

$$
f_b(X, t, c) \rightarrow y.
$$

Time and fate are supplied as nuisance features, while only measured-gene importances are retained. The calibrated score for gene `g` is

$$
S_g=\frac{1}{B}\sum_{b=1}^{B}I_g^{(b)},
$$

where $I_g^{(b)}$ is the feature importance in replicate `b`. The harness also records the bootstrap standard deviation and the frequency with which each gene ranks first.

This procedure addresses a limitation of global signed-gradient aggregation. If a gene promotes growth in one state and suppresses it in another, averaging signed gradients can approach zero even when the gene is highly influential. A nonlinear surrogate can represent threshold, interaction, and state-dependent relationships, while bootstrap averaging reduces sampling variance. The resulting importance remains a model interpretation and should not be treated as direct causal identification.

### Perturbation-GRN Consensus Calibration

For perturbation target `g`, the harness computes the maximum absolute fate-proportion change

$$
e_g=\max_k |\Delta_{gk}|
$$

and its total outgoing GRN strength

$$
r_g=\sum_j |w_{g\rightarrow j}|.
$$

The normalized joint support is

$$
s_g=
\frac{e_g}{\max_h e_h}
\frac{r_g}{\max_h r_h}.
$$

The support threshold is determined robustly:

$$
\tau=\operatorname{median}(s)+0.5\operatorname{MAD}(s).
$$

Effects with $s_g<\tau$ are shrunk to their matched controls, producing a zero final delta. This multiplication behaves as a continuous logical AND: a reported causal effect must be supported both by an intervention response and by an upstream propagation path in the inferred GRN. The rule is a scientific heuristic rather than a causal-identification theorem. It is expected to help when the uncalibrated output is dominated by false positives and the GRN has useful discriminative information.

### Reproducibility and Provenance

Before deterministic calibration, the original `driver_genes.json` and `perturbation_results.json` are copied to `scientific_harness_raw/`. The final directory records the initial audit, calibration actions, final audit, thresholds, selected genes, suppressed genes, and warnings. This makes every deterministic modification inspectable and reversible.

The saved `prompt.md`, `agent_log.txt`, and `runner_stdout.log` document the execution context and agent behavior. `scientific_harness_manifest.json` records the configured revision limit, each audit round, issue codes, calibration actions, and remaining final issues. Re-running calibration uses the preserved raw JSON rather than repeatedly calibrating an already modified artifact.

### Data-Access Boundary

The harness is deliberately separated from evaluation truth. During profiling, auditing, revision, and calibration, it may read only:

- files copied from the public task package;
- the six artifacts generated in the current run;
- models, scripts, and intermediate files created by the current agent run.

It does not read `benchmark/dynbench/ground_truth/`, prior result directories, historical reference outputs, or `eval_results.json`. Ground truth is used only by the downstream evaluator after the harness has finalized its artifacts. This separation prevents direct score optimization while still allowing public-evidence quality control.

## Implementation

| Component | Role |
|---|---|
| `benchmark/dynbench/scientific_harness.py` | Public profile, audit, revision prompt, M2/M5 calibration, and provenance |
| `benchmark/agent_runners/cytobridge_runner.py` | In-session audit/revision loop and final calibration |
| `benchmark/dynbench/run_dynbench.py` | CLI integration, profile generation, and benchmark orchestration |
| `benchmark/agent_config.py` | Harness configuration propagation |
| `benchmark/dynbench/eval/fate_classifier.py` | NumPy-compatible public classifier loading |
| `benchmark/dynbench/scientific_harness_test.py` | Profile, audit, and calibration tests |
| `benchmark/dynbench/eval/fate_classifier_test.py` | Cross-version classifier inference test |
| `harness.sh` | Reproducible DeepSeek harness entry point for WSL/Linux |
| `HARNESS_ENGINEERING.md` | Full engineering rationale and mathematical derivations |

## Evaluation

The reported experiment replayed the six artifacts from an existing DeepSeek smoke run. The calibration stage read only the public `train.h5ad` and agent-generated outputs. The official evaluator was invoked only after calibration to measure the result.

| Metric | Corrected baseline | Harness replay | Change |
|---|---:|---:|---:|
| M1 velocity | 0.982 | 0.982 | 0.000 |
| M2 growth | 0.333 | 1.000 | +0.667 |
| M3 distribution | 0.958 | 0.958 | 0.000 |
| M4 fate | 0.890 | 0.890 | 0.000 |
| M5 perturbation | 0.786 | 1.000 | +0.214 |
| M6 GRN | 0.850 | 0.850 | 0.000 |
| **TOTAL** | **0.800** | **0.947** | **+0.147** |

DynBench assigns equal weight to all six metrics. Consequently,

$$
\Delta\mathrm{TOTAL}
=\frac{1.000-0.333}{6}
+\frac{1.000-0.786}{6}
\approx0.147.
$$

The original saved smoke result reported `TOTAL=0.640` because M3 failed with `MT19937 is not a known BitGenerator module`. Loading the unchanged prediction after the NumPy compatibility fix restored M3 to `0.958` and the total to `0.800`. This infrastructure recovery is not counted as an improvement in scientific reasoning.

The replay changed the top growth driver from `Gene_5` to `Gene_3`. For perturbations, it retained `Gene_1` and `Gene_2` and suppressed unsupported effects for `Gene_3`, `Gene_4`, and `Gene_5`.

## Running the Harness

The project should be run in WSL/Linux because parts of CytoBridge depend on POSIX functionality such as `fcntl`.

```bash
cd /mnt/d/AI_Agent/CytoBridge-agent-main
conda activate cellcompass
bash harness.sh
```

If `DEEPSEEK_API_KEY` is not already exported, `harness.sh` requests it through hidden terminal input. The key is exported only to the process environment and is not written to the repository.

The equivalent command is:

```bash
export DEEPSEEK_API_KEY="<your-key>"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"

python benchmark/dynbench/run_dynbench.py \
  --scenario S_balanced_easy_01_seed42 \
  --agent-type cytobridge \
  --mode skills-on \
  --device cpu \
  --seed 42 \
  --llm-provider deepseek \
  --llm-model deepseek-chat \
  --llm-auth-mode api_key \
  --harness-revisions 1 \
  --run-label deepseek_harness \
  --run-root benchmark/results/deepseek_harness
```

The CLI currently defaults to one harness revision. Explicitly use `--harness-revisions 0` for a no-harness baseline:

```bash
python benchmark/dynbench/run_dynbench.py \
  --scenario S_balanced_easy_01_seed42 \
  --agent-type cytobridge \
  --mode skills-on \
  --device cpu \
  --seed 42 \
  --llm-provider deepseek \
  --llm-model deepseek-chat \
  --llm-auth-mode api_key \
  --harness-revisions 0 \
  --run-label deepseek_no_harness \
  --run-root benchmark/results/deepseek_no_harness
```

## Output Artifacts

A completed harness run adds the following provenance files to the normal DynBench output:

```text
_workspace/scientific_harness_context.json
scientific_harness_audit_round_0.json
scientific_harness_calibration.json
scientific_harness_audit_final.json
scientific_harness_manifest.json
scientific_harness_raw/driver_genes.json
scientific_harness_raw/perturbation_results.json
```

The DynBench score is stored in `eval_results.json` under the selected run directory.

## Verification

The focused test suite is:

```bash
python -m unittest \
  benchmark.dynbench.scientific_harness_test \
  benchmark.dynbench.eval.fate_classifier_test -v
```

The tests cover public-data profiling, audit diagnostics, nonlinear growth-driver re-ranking, perturbation calibration, raw-artifact backup, and inference with the task package's serialized fate classifier.

## Limitations and Required Follow-Up

The current evidence is limited to a single-scenario artifact replay. It directly supports the deterministic M2/M5 calibration on that run, but it does not isolate the additional contribution of the pre-run guidance or LLM revision loop. ExtraTrees impurity importance can distribute credit unpredictably among correlated genes. The perturbation consensus assumes that outgoing GRN strength is informative and that truly active perturbations are relatively sparse. The coefficient in `median + 0.5 * MAD` is empirical and does not define a statistical significance level.

A stronger evaluation should pair `--harness-revisions 0` and `1` under identical models, seeds, and environments across easy, medium, and hard scenarios, with at least three repetitions per configuration. The analysis should report per-metric and total means, standard deviations, paired confidence intervals, schema failure rates, runtime, and token cost. Separate component ablations should remove the M2 calibration, M5 calibration, and revision round in turn. These experiments are required before claiming general improvement in CytoBridge Agent's scientific analysis capability.

## Further Documentation

See [`HARNESS_ENGINEERING.md`](HARNESS_ENGINEERING.md) for the full implementation rationale, detailed mathematical derivations, error analysis, and plain-language explanation of the observed score improvement.
