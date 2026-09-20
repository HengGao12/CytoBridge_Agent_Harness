#!/usr/bin/env python
"""
benchmark_run.py — Orchestrate benchmark agent runs.

For each (fold, seed), calls the selected AgentRunner with a standardized prompt,
captures results, and saves a run manifest.

Usage:
    python -m benchmark.benchmark_run \
        --config benchmark/configs/weinreb_2020.yaml \
        --seeds 42 \
        --agent cytobridge \
        --device cpu

    # Single fold only
    python -m benchmark.benchmark_run \\
        --config benchmark/configs/weinreb_2020.yaml \\
        --fold middle_holdout \\
        --seeds 42 \\
        --agent cytobridge

    # Dry run (print what would be executed)
    python -m benchmark.benchmark_run \\
        --config benchmark/configs/weinreb_2020.yaml \\
        --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from .benchmark_utils import (
    load_task_card,
    get_fold_data_dir,
    get_run_dir,
    save_run_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_prompt(card: dict, fold: dict) -> str:
    """Fill the prompt template from the task card with fold-specific values."""
    template = card["prompt_template"]
    return template.format(
        training_timepoints=", ".join(str(t) for t in fold["training_timepoints"]),
        held_out_timepoint=str(fold["held_out_timepoint"]),
        time_key=card["time_key"],
    )


def get_runner(agent_id: str, **kwargs):
    """Instantiate the appropriate AgentRunner."""
    if agent_id == "codex":
        from .agent_runners.codex_runner import CodexRunner
        return CodexRunner(**kwargs)
    if agent_id == "cytobridge":
        from .agent_runners.cytobridge_runner import CytoBridgeRunner
        return CytoBridgeRunner(**kwargs)
    if agent_id in {"biomini", "biomni"}:
        from .agent_runners.biomni_runner import BiomniRunner
        return BiomniRunner(**kwargs)
    else:
        raise ValueError(
            f"Unknown agent '{agent_id}'. Supported automated runners: codex, cytobridge, biomini.\n"
            f"  1. python -m benchmark.make_task_package ...\n"
            f"  2. Give task package to agent\n"
            f"  3. python -m benchmark.eval_external --prediction <path> ..."
        )


def _build_runner_kwargs(agent_id: str, **cli_kwargs) -> dict:
    """Extract the runner-specific kwargs from CLI args."""
    if agent_id == "codex":
        keys = ["llm_model", "llm_profile_id"]
    elif agent_id == "cytobridge":
        keys = ["llm_model", "llm_provider", "llm_base_url", "llm_api_key", "llm_auth_mode",
                "llm_profile_id", "llm_thinking_level"]
    elif agent_id in {"biomini", "biomni"}:
        keys = [
            "biomni_repo", "llm_model", "llm_source", "llm_base_url", "llm_api_key",
            "llm_auth_mode", "llm_profile_id", "llm_thinking_level",
        ]
    else:
        keys = []
    return {k: v for k, v in cli_kwargs.items() if k in keys and v is not None}


def run_benchmark(
    config_path: str,
    agent_id: str = "cytobridge",
    agent_type: Optional[str] = None,
    fold_filter: Optional[str] = None,
    seeds: List[int] = None,
    device: str = "cpu",
    dry_run: bool = False,
    timeout_sec: Optional[float] = None,
    # LLM config passthrough
    llm_model: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_auth_mode: str = "auto",
    llm_profile_id: Optional[str] = None,
    llm_thinking_level: Optional[str] = "low",
    biomni_path: Optional[str] = None,
    biomni_source: Optional[str] = None,
) -> None:
    """Run the benchmark for a dataset config."""
    if seeds is None:
        seeds = [42]

    card = load_task_card(config_path)
    dataset_id = card["dataset_id"]
    time_key = card["time_key"]

    folds = card["folds"]
    if fold_filter:
        folds = [f for f in folds if f["fold_id"] == fold_filter]
        if not folds:
            raise ValueError(f"Fold '{fold_filter}' not found in config")

    total_runs = len(folds) * len(seeds)
    from cytobridge_agent.utils.config_manager import get_saved_config
    from .agent_config import resolve_agent_selection

    selection = resolve_agent_selection(
        agent_type=agent_type or agent_id,
        saved_config=get_saved_config(),
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_auth_mode=llm_auth_mode,
        llm_profile_id=llm_profile_id,
        llm_thinking_level=llm_thinking_level,
        biomni_path=biomni_path,
        biomni_source=biomni_source,
    )
    agent_id = selection.run_label
    runner_id = selection.runner_id

    logger.info(
        f"Benchmark: dataset={dataset_id}, agent={agent_id}, "
        f"folds={len(folds)}, seeds={len(seeds)}, total_runs={total_runs}"
    )

    if dry_run:
        for fold in folds:
            prompt = build_prompt(card, fold)
            for seed in seeds:
                run_dir = get_run_dir(agent_id, dataset_id, fold["fold_id"], seed)
                fold_dir = get_fold_data_dir(dataset_id, fold["fold_id"])
                print(f"\n{'='*60}")
                print(f"[DRY RUN] Agent={agent_id} | Fold={fold['fold_id']} | Seed={seed}")
                print(f"  Train: {fold_dir / 'train.h5ad'}")
                print(f"  Output: {run_dir}")
                print(f"  Device: {device}")
                print(f"  Prompt: {prompt[:200]}...")
        print(f"\n{'='*60}")
        print(f"Total: {total_runs} runs (dry run, nothing executed)")
        return

    # Instantiate runner with agent-specific kwargs
    runner_kwargs = _build_runner_kwargs(runner_id, **selection.runner_kwargs)
    runner = get_runner(runner_id, **runner_kwargs)

    run_idx = 0
    for fold in folds:
        prompt = build_prompt(card, fold)
        fold_dir = get_fold_data_dir(dataset_id, fold["fold_id"])
        train_path = fold_dir / "train.h5ad"

        if not train_path.exists():
            logger.error(
                f"Train data not found: {train_path}. "
                f"Run `python -m benchmark.benchmark_prepare --config {config_path}` first."
            )
            continue

        for seed in seeds:
            run_idx += 1
            run_dir = get_run_dir(agent_id, dataset_id, fold["fold_id"], seed)
            run_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"\n[{run_idx}/{total_runs}] "
                f"Fold={fold['fold_id']} Seed={seed} → {run_dir}"
            )

            result = runner.run(
                train_h5ad=train_path,
                task_prompt=prompt,
                output_dir=run_dir,
                seed=seed,
                device=device,
                time_key=time_key,
                timeout_sec=timeout_sec,
            )

            # Save manifest
            manifest = {
                "dataset_id": dataset_id,
                "fold_id": fold["fold_id"],
                "held_out_timepoint": fold["held_out_timepoint"],
                "training_timepoints": fold["training_timepoints"],
                "agent_id": agent_id,
                "seed": seed,
                "device": device,
                "prompt": prompt,
                **result.to_dict(),
            }
            save_run_manifest(run_dir, manifest)

            status = "✓" if result.success else "✗"
            logger.info(
                f"  {status} success={result.success} "
                f"runtime={result.runtime_sec:.1f}s "
                f"prediction={'found' if result.predicted_h5ad else 'MISSING'}"
            )

    logger.info(f"\nBenchmark complete. Results in: benchmark/results/{agent_id}/{dataset_id}/")


def main():
    from .agent_config import load_saved_config

    saved = load_saved_config()

    parser = argparse.ArgumentParser(description="Run benchmark agent tasks")
    parser.add_argument("--config", required=True, help="Task-card YAML config")
    parser.add_argument("--agent", default=None, help="Legacy runner id (cytobridge/biomini)")
    parser.add_argument("--agent-type", default=None, choices=["codex", "biomini", "biomni", "cytobridge"],
                        help="Agent backend to use. Overrides ~/.cellcompass Codex defaults.")
    parser.add_argument("--fold", default=None, help="Run only this fold")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds (default: 42)")
    parser.add_argument("--device", default=saved.get("device", "cpu"), choices=["cpu", "cuda", "mps"],
                        help="Compute device")
    parser.add_argument("--timeout", type=float, default=None, help="Per-run timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without executing")

    # LLM config (falls back to ~/.cellcompass/config.json)
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument("--llm-base-url", default=None, help="LLM API base URL")
    parser.add_argument("--llm-api-key", default=None, help="LLM API key")
    parser.add_argument("--llm-auth-mode", default="auto",
                        choices=["auto", "api_key", "gemini_oauth", "codex_oauth"])
    parser.add_argument("--llm-profile-id", default=None)
    parser.add_argument("--llm-thinking-level", default=None,
                        choices=["off", "minimal", "low", "medium", "high"])
    parser.add_argument("--biomni-path", default=None, help="Path to Biomni repo (default: <project>/Biomni)")
    parser.add_argument("--biomni-source", default=None, help="Biomni LLM source, e.g. Anthropic/OpenAI/Gemini/Custom")

    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    try:
        run_benchmark(
            config_path=args.config,
            agent_id=args.agent or "codex",
            agent_type=args.agent_type,
            fold_filter=args.fold,
            seeds=seeds,
            device=args.device,
            dry_run=args.dry_run,
            timeout_sec=args.timeout,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            llm_auth_mode=args.llm_auth_mode,
            llm_profile_id=args.llm_profile_id,
            llm_thinking_level=args.llm_thinking_level,
            biomni_path=args.biomni_path,
            biomni_source=args.biomni_source,
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
