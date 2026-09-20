#!/usr/bin/env python
"""
DynBench v2 Runner — Run a selected agent on a DynBench task package.

Usage:
    conda activate CytoCompass
    python benchmark/dynbench/run_dynbench.py \
        --scenario S1_v2 \
        --mode skills-on

This will:
1. Load the task package (train.h5ad + TASK.md)
2. Send the same task prompt and public inputs to the selected agent runner
3. Collect the 6 output files
4. Run dynbench_eval_v2 and save results
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from benchmark.result_paths import resolve_dynbench_paths
from benchmark.cost_tracker import CostTracker



EXPECTED_DYNBENCH_OUTPUT_FILES = [
    "velocity_field.csv",
    "growth_rates.csv",
    "holdout_prediction.csv",
    "per_cell_fate.json",
    "perturbation_results.json",
    "driver_genes.json",
]

MODEL_NATIVE_ARTIFACT_NAMES = {
    "model_artifact.json",
    "model_state.pt",
    "resolved_config.yaml",
    "config.yaml",
}

MODEL_NATIVE_LOG_TERMS = (
    "preview_training_run",
    "run_training",
    "candidate_name': 'vgfm",
    'candidate_name": "vgfm',
    "candidate_name': 'crufm",
    'candidate_name": "crufm',
    "candidate_name': 'wfrfm",
    'candidate_name": "wfrfm',
    "DynamicalModel",
    "simulate_trajectory",
)

CUSTOM_PIPELINE_TERMS = (
    "dynbench_export",
    "dynbench_pipeline",
    "sklearn",
    "Ridge",
    "RandomForest",
    "torch.autograd",
    "Jacobian",
    "jacobian",
)

STALE_RESULT_FILES = EXPECTED_DYNBENCH_OUTPUT_FILES + [
    "error.json",
    "eval_results.json",
    "agent_trace.json",
    "recovery_manifest.json",
]

RESETTABLE_OUTPUT_DIRS = [
    "training_runs",
    "figures",
    "_biomni_runtime",
    "_workspace",
    ".runtime",
    "__pycache__",
]


def _missing_expected_outputs(output_dir: Path, *, started_at: float | None = None) -> list[str]:
    missing: list[str] = []
    for name in EXPECTED_DYNBENCH_OUTPUT_FILES:
        path = output_dir / name
        if not path.exists():
            missing.append(name)
            continue
        if path.stat().st_size <= 0:
            missing.append(name)
            continue
        if started_at is not None and path.stat().st_mtime < started_at:
            missing.append(name)
    return missing

def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_agent_outputs(output_dir: Path, *, started_at: float | None = None) -> None:
    """Validate that all six benchmark outputs exist and are parseable."""
    import json as _json
    for name in EXPECTED_DYNBENCH_OUTPUT_FILES:
        path = output_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Output validation failed: {name} is missing")
        if path.stat().st_size <= 0:
            raise ValueError(f"Output validation failed: {name} is empty")
        if started_at is not None and path.stat().st_mtime < started_at:
            raise ValueError(f"Output validation failed: {name} is stale from before this run")
        if name.endswith(".json"):
            try:
                with open(path) as f:
                    _json.load(f)
            except Exception as exc:
                raise ValueError(f"Output validation failed: {name} is not valid JSON: {exc}") from exc
        if name.endswith(".csv"):
            try:
                with open(path) as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                if len(lines) < 2:
                    raise ValueError(f"Output validation failed: {name} has fewer than 2 rows")
            except Exception as exc:
                raise ValueError(f"Output validation failed: {name} read error: {exc}") from exc


def _audit_cytobridge_model_native(output_dir: Path) -> dict:
    """Classify whether a DynBench CytoBridge run used a package-native model.

    This is intentionally an evidence audit, not a hidden-metric computation.
    It checks only public run artifacts/logs and records whether the delivered
    files were backed by a CytoBridge model artifact/training path.
    """
    artifact_paths = [
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name in MODEL_NATIVE_ARTIFACT_NAMES
    ]

    readable_paths: list[Path] = []
    for name in ("runner_stdout.log", "agent_log.txt", "prompt.md"):
        path = output_dir / name
        if path.exists():
            readable_paths.append(path)
    for subdir in ("scripts", "training_runs", ".runtime"):
        root = output_dir / subdir
        if root.exists():
            readable_paths.extend(path for path in root.rglob("*.py") if path.is_file())
            readable_paths.extend(path for path in root.rglob("*.yaml") if path.is_file())
            readable_paths.extend(path for path in root.rglob("*.json") if path.is_file())

    text_parts: list[str] = []
    inspected_files: list[str] = []
    for path in readable_paths:
        try:
            text_parts.append(path.read_text(encoding="utf-8", errors="replace")[:500_000])
            inspected_files.append(str(path.relative_to(output_dir)))
        except Exception:
            continue
    text = "\n".join(text_parts)
    text_lower = text.lower()

    model_hits = [term for term in MODEL_NATIVE_LOG_TERMS if term.lower() in text_lower]
    custom_hits = [term for term in CUSTOM_PIPELINE_TERMS if term.lower() in text_lower]
    has_model_artifact = any(path.endswith("model_artifact.json") for path in artifact_paths)
    has_model_state = any(path.endswith("model_state.pt") for path in artifact_paths)
    has_resolved_config = any(path.endswith("resolved_config.yaml") or path.endswith("config.yaml") for path in artifact_paths)
    has_training_run = any(path.startswith("training_runs/") for path in artifact_paths)
    model_native = bool(has_model_artifact and has_model_state and has_resolved_config and (has_training_run or model_hits))

    if model_native:
        classification = "model_native"
    elif custom_hits:
        classification = "custom_pipeline"
    else:
        classification = "unclear"

    return {
        "classification": classification,
        "model_native": model_native,
        "has_model_artifact": has_model_artifact,
        "has_model_state": has_model_state,
        "has_resolved_config": has_resolved_config,
        "has_training_run": has_training_run,
        "model_hits": model_hits,
        "custom_hits": custom_hits,
        "artifact_paths": artifact_paths[:50],
        "inspected_files": inspected_files[:50],
    }


def _build_output_format_examples() -> str:
    """6.7 Build explicit output format specifications for the TASK.md template."""
    return (
        "\n\n## Required Output File Formats\n"
        "Each file MUST follow these exact formats:\n\n"
        "### 1. holdout_prediction.csv\n"
        "CSV with gene columns + 'time' column. Each row = one predicted cell. Write with index=False.\n"
        "```csv\n"
        "gene_A,gene_B,gene_C,...,time\n"
        "0.123,0.456,0.789,...,5.0\n"
        "```\n\n"
        "### 2. velocity_field.csv\n"
        "CSV with columns named 'velocity_{gene}' for every gene in the training data.\n"
        "One row per cell in training data, same order as train.h5ad. Write with index=False.\n"
        "```csv\n"
        "velocity_gene_A,velocity_gene_B,velocity_gene_C,...\n"
        "0.01,-0.02,0.03,...\n"
        "```\n\n"
        "### 3. growth_rates.csv\n"
        "Single column 'growth_rate'. One row per cell in training data, same order as train.h5ad. Write with index=False.\n"
        "```csv\n"
        "growth_rate\n"
        "0.05\n"
        "```\n\n"
        "### 4. per_cell_fate.json\n"
        "Dict mapping cell_id -> {fate_label: probability}. For root/time-0 cells only.\n"
        "```json\n"
        "{\"cell_001\": {\"Fate_A\": 0.7, \"Fate_B\": 0.3}}\n"
        "```\n\n"
        "### 5. perturbation_results.json\n"
        "List of dicts, each with 'gene_name', 'control', 'perturbed', 'delta' keys.\n"
        "```json\n"
        "[{\"gene_name\": \"G1\", \"control\": {\"Fate_A\": 0.5}, \"perturbed\": {\"Fate_A\": 0.3}, \"delta\": {\"Fate_A\": -0.2}}]\n"
        "```\n\n"
        "### 6. driver_genes.json\n"
        "Dict with 'grn_edges' (list of {source, target, score}) and 'growth_drivers' (dict gene->float).\n"
        "```json\n"
        "{\"grn_edges\": [{\"source\": \"G1\", \"target\": \"G2\", \"score\": 0.8}], \"growth_drivers\": {\"G1\": 0.5}}\n"
        "```\n"
    )



def _prepare_agent_workspace(task_dir: Path, output_dir: Path) -> dict[str, Path]:
    workspace_dir = output_dir / "_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for name in ["train.h5ad", "TASK.md", "fate_classifier.pkl", "prediction_targets.json"]:
        source = task_dir / name
        if not source.exists():
            continue
        target = workspace_dir / name
        shutil.copy2(source, target)
        copied[name] = target
    verifier_source = Path(__file__).resolve().parent / "verify_dynbench_outputs.py"
    if verifier_source.exists():
        verifier_target = workspace_dir / "verify_dynbench_outputs.py"
        shutil.copy2(verifier_source, verifier_target)
        copied["verify_dynbench_outputs.py"] = verifier_target
    return copied


class _TeeStream:
    """Mirror stdout/stderr into a log file without hiding terminal output."""

    def __init__(self, *streams):
        self._streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


def run_dynbench(
    scenario: str = "S1_v2",
    mode: str = "skills-on",
    device: str = "cpu",
    seed: int = None,
    run_label: str = None,
    agent_type: str = None,
    llm_model: str = None,
    llm_provider: str = None,
    llm_base_url: str = None,
    llm_api_key: str = None,
    llm_auth_mode: str = None,
    llm_profile_id: str = None,
    llm_thinking_level: str = None,
    harness_revisions: int = 1,
    biomni_path: str = None,
    biomni_source: str = None,
    timeout_sec: float = None,
    run_root: str = None,
):
    """
    Run a selected agent on a DynBench task.
    
    Args:
        scenario: scenario name (e.g. S1_v2)
        mode: "skills-on" (normal) or "skills-off" (mask downstream skills)
        device: compute device
        seed: random seed
        llm_model: override LLM model name
    """
    base_dir = Path(__file__).resolve().parent
    task_dir = base_dir / "task_packages" / scenario
    gt_dir = base_dir / "ground_truth" / scenario

    # v2 scenario search: also look under run_root/dynbench_assets/ (batch runner puts data there)
    if not task_dir.exists() and run_root:
        run_root_resolved = Path(run_root).resolve()
        # native_run_root is usually <batch_dir>/native_runs; dynbench_assets is at <batch_dir>/dynbench_assets
        for candidate_root in [run_root_resolved, run_root_resolved.parent]:
            v2_task = candidate_root / "dynbench_assets" / "task_packages" / scenario
            if v2_task.exists():
                task_dir = v2_task
                gt_v2 = candidate_root / "dynbench_assets" / "ground_truth" / scenario
                if gt_v2.exists():
                    gt_dir = gt_v2
                break
    from benchmark.agent_config import load_saved_config, resolve_agent_selection

    selection = resolve_agent_selection(
        agent_type=agent_type,
        saved_config=load_saved_config(),
        llm_model=llm_model,
        llm_provider=llm_provider,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_auth_mode=llm_auth_mode,
        llm_profile_id=llm_profile_id,
        llm_thinking_level=llm_thinking_level,
        harness_revisions=harness_revisions,
        biomni_path=biomni_path,
        biomni_source=biomni_source,
    )

    # Set up output directory
    if run_label:
        run_name = f"agent_{selection.run_label}_{mode}_{run_label}"
    elif seed is not None:
        run_name = f"agent_{selection.run_label}_{mode}_seed{seed}"
    else:
        run_name = f"agent_{selection.run_label}_{mode}"
    _, results_dir, output_dir = resolve_dynbench_paths(
        scenario=scenario,
        run_name=run_name,
        run_root=run_root,
    )
    _reset_output_dir(output_dir)
    summary_file = results_dir / f"{run_name}.json"
    summary_file.unlink(missing_ok=True)

    # ── 6.8 Create CostTracker for this run ──
    cost_tracker = CostTracker()

    # Validate paths
    workspace_files = _prepare_agent_workspace(task_dir, output_dir)
    workspace_dir = output_dir / "_workspace"
    train_path = workspace_files.get("train.h5ad", task_dir / "train.h5ad")
    task_md = workspace_files.get("TASK.md", task_dir / "TASK.md")
    fate_classifier_path = workspace_files.get("fate_classifier.pkl", task_dir / "fate_classifier.pkl")
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}\nRun snapshot_builder_v2.py first.")
    if not task_md.exists():
        raise FileNotFoundError(f"Task description not found: {task_md}")
    
    # Load TASK.md as the agent prompt
    with open(task_md) as f:
        task_prompt = f.read()

    harness_profile_path = None
    if selection.agent_type == "cytobridge" and int(harness_revisions or 0) > 0:
        from benchmark.dynbench.scientific_harness import (
            build_initial_harness_guidance,
            write_public_data_profile,
        )

        harness_profile_path = workspace_dir / "scientific_harness_context.json"
        write_public_data_profile(
            train_h5ad=train_path,
            output_path=harness_profile_path,
            prediction_targets_path=workspace_files.get("prediction_targets.json"),
        )
        task_prompt += build_initial_harness_guidance(harness_profile_path)

    # Default to the neutral/relaxed wording used by the public task contract.
    # The old "model-native" wording can be re-enabled only for controlled
    # compatibility checks.
    relax_model_native_prompt = os.environ.get("DYNBENCH_STRICT_MODEL_NATIVE_PROMPT") != "1"
    if relax_model_native_prompt:
        execution_clause = (
            "If you can preprocess and compute the requested outputs, then immediately run the task export contract.\n"
        )
        rerun_clause = (
            "If any artifact is missing or invalid, rerun the export path for that artifact rather than filling placeholders or statistical approximations.\n"
        )
        pipeline_clause = "You MUST produce all 6 required output files using your OWN analysis pipeline.\n"
    else:
        execution_clause = (
            "If you can preprocess and train, then immediately run model-native downstream computation and the task export contract.\n"
        )
        rerun_clause = (
            "If any artifact is missing or invalid, rerun the model-native export path rather than filling placeholders or statistical approximations.\n"
        )
        pipeline_clause = "You MUST produce all 6 required output files using your OWN trained model pipeline.\n"

    benchmark_contract = (
        "\n\n## Delivery Discipline\n"
        "This is a strict file-delivery task, not an open-ended research task.\n"
        "Highest priority is to materialize the six required output files for this run and verify their public schema.\n"
        "Do not spend runtime budget on literature review, theory brainstorming, or generic methodology comparison unless a hard blocker prevents execution.\n"
        f"{execution_clause}"
        "Do not treat plan updates, narrative summaries, or claimed completion as progress unless files now exist in the output directory.\n"
        "After each materialization step, immediately verify file existence, non-zero size, and basic schema before doing more reasoning.\n"
        "Avoid partial completion: keep working until all required files exist or a concrete unsupported blocker is recorded.\n"
        "Treat missing any required file as a failure until you materialize it in the output directory.\n"
    )
    task_prompt += benchmark_contract
    task_prompt += _build_output_format_examples()
    verifier_path = workspace_files.get("verify_dynbench_outputs.py", workspace_dir / "verify_dynbench_outputs.py")
    task_prompt += (
        "\n\n## Public Output Format Verifier\n"
        "Before final delivery, run the public verifier script from the task workspace:\n"
        f"`python {verifier_path} --output-dir {output_dir} --train-h5ad {train_path}`\n"
        "This verifier checks only public file format/schema constraints; it does not read hidden answers, judge scientific correctness, or repair outputs.\n"
        "If it reports a failed output file, fix the corresponding artifact and rerun the verifier before final response.\n"
    )
    task_prompt += (
        "\n\n## CRITICAL: NO FALLBACK ALLOWED\n"
        f"{pipeline_clause}"
        "DO NOT use any deterministic/statistical shortcut, interpolation, or simple difference method.\n"
        f"{rerun_clause}"
        "If your model training fails, report the error explicitly and stop; do not produce fake outputs.\n"
        "No downstream checker will repair invalid, missing, stale, or schema-mismatched output files.\n"
    )
    task_prompt += (
        "\n\n## Restricted Workspace\n"
        f"Read task inputs from this task workspace: {workspace_dir}\n"
        f"Write final deliverables directly under the designated output directory: {output_dir}\n"
        "- Treat the task workspace as read-only input/provenance. Do not modify or overwrite files there.\n"
        "- Only rely on files present in the task workspace plus files you create inside the designated output directory.\n"
        "- Do not read hidden answer/reference directories, prior run outputs, or other result folders.\n"
        "- Do not inspect non-public implementation files outside the task workspace.\n"
    )

    task_prompt += (
        "\n## Execution Discipline\n"
        "Do not claim a step is complete until the corresponding files exist directly under the requested output directory.\n"
        "If you write files into a workspace or scratch path first, copy or move them into the requested output directory before ending.\n"
        "Before final response, re-check all six required files with existence and basic shape/schema validation.\n"
    )

    # Add output directory instruction
    task_prompt += f"\n\n## Output Directory\nSave all outputs to: {output_dir}\n"
    task_prompt += f"\n## Additional Files\n- Fate classifier: {fate_classifier_path}\n"
    
    logger.info("=" * 60)
    logger.info(f"DynBench v2 Runner")
    logger.info(f"  Scenario: {scenario}")
    logger.info(f"  Agent type: {selection.agent_type}")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Scientific harness revisions: {harness_revisions if selection.agent_type == 'cytobridge' else 0}")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Workspace: {workspace_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)
    
    # ── Set up file logging for traceability ──
    log_file = output_dir / "agent_log.txt"
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    # Save the prompt for traceability
    with open(output_dir / "prompt.md", "w") as f:
        f.write(task_prompt)

    transcript_path = output_dir / "runner_stdout.log"
    transcript_fp = open(transcript_path, "w", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(orig_stdout, transcript_fp)
    sys.stderr = _TeeStream(orig_stderr, transcript_fp)
    
    try:
        # ── Run the agent ──
        t0 = time.time()
        
        # If skills-off mode, set environment variable to mask downstream skills
        if mode == "skills-off":
            os.environ["CYTOBRIDGE_DISABLE_DOWNSTREAM_SKILLS"] = "1"
            logger.info("Skills-off mode: downstream analysis skills disabled")
        else:
            os.environ.pop("CYTOBRIDGE_DISABLE_DOWNSTREAM_SKILLS", None)

        try:
            from benchmark.benchmark_run import get_runner
            from benchmark.agent_runners.auth_utils import clear_benchmark_auth_lock

            clear_benchmark_auth_lock()
            runner = get_runner(selection.runner_id, **selection.runner_kwargs)
            result = runner.run(
                train_h5ad=train_path,
                task_prompt=task_prompt,
                output_dir=output_dir,
                scenario=scenario,
                seed=seed,
                device=device,
                time_key="time",
                timeout_sec=timeout_sec,
                expected_output_files=EXPECTED_DYNBENCH_OUTPUT_FILES,
                workspace_dir=workspace_dir,
                blocked_read_roots=[gt_dir, base_dir / "eval", base_dir / "ground_truth"],
                cost_tracker=cost_tracker,
            )
            clear_benchmark_auth_lock()
            runtime = time.time() - t0
            # ── 6.8 Stop cost tracker ──
            cost_tracker.stop()

            # ── 6.7 Validate outputs ──
            if not result.success:
                logger.error(f"Agent failed after {runtime:.1f}s: {result.error_message}")
                with open(output_dir / "error.json", "w") as f:
                    json.dump({
                        "agent_type": selection.agent_type,
                        "error": result.error_message,
                        "runtime_sec": runtime,
                        "logs": result.logs,
                        "run_result": result.to_dict(),
                    }, f, indent=2)
                return

            # Keep the native runner error when execution failed. Validating
            # files first would replace it with a secondary missing-file error.
            _validate_agent_outputs(output_dir, started_at=t0)
            model_native_audit = _audit_cytobridge_model_native(output_dir)
            with open(output_dir / "model_native_audit.json", "w") as f:
                json.dump(model_native_audit, f, indent=2, default=str)

            logger.info(f"Agent completed in {runtime:.1f}s")
        except Exception as e:
            runtime = time.time() - t0
            cost_tracker.stop()
            try:
                from benchmark.agent_runners.auth_utils import clear_benchmark_auth_lock

                clear_benchmark_auth_lock()
            except Exception:
                pass
            logger.error(f"Agent failed after {runtime:.1f}s: {e}", exc_info=True)
            with open(output_dir / "error.json", "w") as f:
                json.dump(
                    {
                        "agent_type": selection.agent_type,
                        "error": str(e),
                        "runtime_sec": runtime,
                        "run_result": {
                            "success": False,
                            "runtime_sec": runtime,
                            "error_message": str(e),
                            "native_status": "native_error",
                            "format_status": "error",
                            "artifact_origin": "none",
                            "native_error_message": str(e),
                        },
                    },
                    f,
                    indent=2,
                )
            return
        
        # ── Save agent trace for debugging ──
        trace = {
            "scenario": scenario,
            "mode": mode,
            "seed": seed,
            "agent_type": selection.agent_type,
            "runner_id": selection.runner_id,
            "runner_kwargs": {k: v for k, v in selection.runner_kwargs.items() if "key" not in k},
            "runtime_sec": runtime,
            "workspace_dir": str(workspace_dir),
            "output_files": [f.name for f in output_dir.iterdir() if f.is_file()],
            "model_native_audit": model_native_audit,
        }
        with open(output_dir / "agent_trace.json", "w") as f:
            json.dump(trace, f, indent=2, default=str)
        
        # ── Run evaluation ──
        logger.info("\n" + "=" * 60)
        logger.info("Running DynBench v2 Evaluation...")
        logger.info("=" * 60)
        
        from benchmark.dynbench.eval.dynbench_eval_v2 import evaluate_all
        
        results = evaluate_all(str(output_dir), str(gt_dir))
        
        # Save results
        result_file = output_dir / "eval_results.json"
        with open(result_file, "w") as f:
            json.dump({
                "scenario": scenario,
                "mode": mode,
                "seed": seed,
                "device": device,
                "agent_type": selection.agent_type,
                "runtime_sec": runtime,
                "run_result": result.to_dict(),
                "results": results,
                "auxiliary": {
                    "cost_summary": cost_tracker.to_dict(),
                    "model_native_audit": model_native_audit,
                },
            }, f, indent=2, default=str)

        # ── 6.8 Save standalone cost summary ──
        cost_summary_file = output_dir / "cost_summary.json"
        with open(cost_summary_file, "w") as f:
            json.dump(cost_tracker.to_dict(), f, indent=2, default=str)

        
        logger.info(f"\nResults saved to: {result_file}")
        
        # Also copy to a summary file
        with open(summary_file, "w") as f:
            json.dump({
                "scenario": scenario,
                "mode": mode,
                "seed": seed,
                "agent_type": selection.agent_type,
                "runtime_sec": runtime,
                "run_result": result.to_dict(),
                "results": results,
                "model_native_audit": model_native_audit,
            }, f, indent=2, default=str)
        try:
            (output_dir / "error.json").unlink()
        except FileNotFoundError:
            pass
        logger.info(f"Summary saved to: {summary_file}")
    finally:
        logging.getLogger().removeHandler(file_handler)
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        transcript_fp.close()


def main():
    parser = argparse.ArgumentParser(description="Run DynBench v2 benchmark")
    parser.add_argument("--scenario", default="S1_v2", help="Scenario name")
    parser.add_argument("--mode", default="skills-on", choices=["skills-on", "skills-off"],
                       help="Agent mode")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional). When not set, directory name omits seed.")
    parser.add_argument("--run-label", default=None, help="Custom run label (e.g. run1). Overrides seed in directory name.")
    parser.add_argument("--agent-type", default=None, choices=["codex", "biomini", "biomni", "cytobridge"])
    parser.add_argument("--llm-model", default=None, help="Override LLM model")
    parser.add_argument("--llm-provider", default=None, help="Override LLM provider, e.g. xiaomi, deepseek, openai-compatible")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-auth-mode", default=None, choices=["auto", "api_key", "gemini_oauth", "codex_oauth"])
    parser.add_argument("--llm-profile-id", default=None)
    parser.add_argument("--llm-thinking-level", default=None, choices=["off", "minimal", "none", "low", "medium", "high"])
    parser.add_argument(
        "--harness-revisions",
        type=int,
        default=1,
        help="Public-data scientific audit/revision rounds for the CytoBridge runner (0 disables)",
    )
    parser.add_argument("--biomni-path", default=None)
    parser.add_argument("--biomni-source", default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--run-root", default=None, help="Benchmark result root under benchmark/results")
    
    args = parser.parse_args()
    run_dynbench(
        scenario=args.scenario,
        mode=args.mode,
        device=args.device,
        seed=args.seed,
        run_label=args.run_label,
        agent_type=args.agent_type,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_auth_mode=args.llm_auth_mode,
        llm_profile_id=args.llm_profile_id,
        llm_thinking_level=args.llm_thinking_level,
        harness_revisions=args.harness_revisions,
        biomni_path=args.biomni_path,
        biomni_source=args.biomni_source,
        timeout_sec=args.timeout,
        run_root=args.run_root,
    )


if __name__ == "__main__":
    main()
