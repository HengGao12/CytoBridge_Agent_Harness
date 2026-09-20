#!/usr/bin/env python
"""Run the full benchmark pipeline.

Usage:
    conda activate CytoCompass
    python benchmark/run_all.py
    python benchmark/run_all.py --execute --device mps
    python benchmark/run_all.py --execute --fold middle_holdout --seeds 42
    python benchmark/run_all.py --prepare-packages
"""
from __future__ import annotations

import argparse
import sys
import os

# Ensure the project root is importable.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmark.benchmark_prepare import prepare_dataset
from benchmark.benchmark_run import run_benchmark
from benchmark.benchmark_eval import evaluate_dataset
from benchmark.benchmark_report import generate_report
from benchmark.benchmark_utils import load_task_card
from benchmark.agent_config import normalize_agent_type


def _prepare_packages(config_path: str, fold_filter: str | None, seeds: list[int]):
    """Generate task packages for external agents (Codex, Gemini, etc.)."""
    from benchmark.make_task_package import main as make_pkg_main
    card = load_task_card(config_path)

    folds = card["folds"]
    if fold_filter:
        folds = [f for f in folds if f["fold_id"] == fold_filter]

    out_base = os.path.join(ROOT, "benchmark", "task_packages")
    for fold in folds:
        fold_id = fold["fold_id"]
        for seed in seeds:
            out_dir = os.path.join(out_base, f"{card['dataset_id']}_{fold_id}_seed{seed}")
            sys.argv = [
                "make_task_package",
                "--config", config_path,
                "--fold", fold_id,
                "--output-dir", out_dir,
                "--seed", str(seed),
            ]
            make_pkg_main()

    print(f"\n✓ Task packages ready in: {out_base}/")
    print(f"  Give each folder to an external agent (Codex App, Antigravity new session, etc.)")
    print(f"  After agent produces predicted_heldout.h5ad, evaluate with:")
    print(f"    python -m benchmark.eval_external --prediction <path> --config {config_path} --fold <fold> --agent <name> --seed <seed>")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full benchmark pipeline: prepare -> run -> eval -> report"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "benchmark", "configs", "weinreb_2020.yaml"),
        help="Task-card YAML (default: weinreb_2020)",
    )
    parser.add_argument("--agent", default=None, help="Legacy Agent ID")
    parser.add_argument("--agent-type", default=None, choices=["codex", "biomini", "biomni", "cytobridge"],
                        help="Agent backend to use")
    parser.add_argument("--fold", default=None, help="Run only this fold (default: all)")
    parser.add_argument("--seeds", default="42", help="Comma-separated random seeds (default: 42)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Compute device")
    parser.add_argument("--execute", action="store_true", help="Actually execute runs (default: dry-run)")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip fold preparation when data already exists")
    parser.add_argument("--skip-run", action="store_true", help="Skip agent runs and only evaluate existing outputs")
    parser.add_argument("--prepare-packages", action="store_true",
                        help="Generate task packages for external agents (Codex, Gemini, etc.)")

    # LLM
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-auth-mode", default="auto")
    parser.add_argument("--llm-profile-id", default=None)
    parser.add_argument("--llm-thinking-level", default="low")
    parser.add_argument("--biomni-path", default=None)
    parser.add_argument("--biomni-source", default=None)

    parser.add_argument("--latex", action="store_true", help="Print a LaTeX table")
    parser.add_argument("--output", default=None, help="Save report to a file (.csv / .tex)")

    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    report_agent_id = normalize_agent_type(args.agent_type or args.agent or "codex")

    # ── Special mode: prepare packages for external agents ──
    if args.prepare_packages:
        print("=" * 60)
        print("Generating Task Packages for External Agents")
        print("=" * 60)
        prepare_dataset(args.config, args.fold)
        _prepare_packages(args.config, args.fold, seeds)
        return

    print("=" * 60)
    print("CytoBridge Benchmark — Full Pipeline")
    print("=" * 60)
    print(f"  Config:  {args.config}")
    print(f"  Agent:   {report_agent_id}")
    print(f"  Fold:    {args.fold or 'all'}")
    print(f"  Seeds:   {seeds}")
    print(f"  Device:  {args.device}")
    print(f"  Mode:    {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print("=" * 60)

    # ── Step 1: Prepare folds ──
    if not args.skip_prepare:
        print("\n▶ Step 1/4: Preparing folds...")
        prepare_dataset(args.config, args.fold)
        print("  ✓ Folds prepared")
    else:
        print("\n▶ Step 1/4: Skipped (--skip-prepare)")

    # ── Step 2: Run agent ──
    if not args.skip_run:
        print(f"\n▶ Step 2/4: Running agent ({report_agent_id})...")
        run_benchmark(
            config_path=args.config,
            agent_id=args.agent or "codex",
            agent_type=args.agent_type,
            fold_filter=args.fold,
            seeds=seeds,
            device=args.device,
            dry_run=not args.execute,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            llm_auth_mode=args.llm_auth_mode,
            llm_profile_id=args.llm_profile_id,
            llm_thinking_level=args.llm_thinking_level,
            biomni_path=args.biomni_path,
            biomni_source=args.biomni_source,
        )
        if args.execute:
            print("  ✓ Agent runs complete")
        else:
            print("  ⓘ Dry-run only. Add --execute to actually run.")
            return
    else:
        print("\n▶ Step 2/4: Skipped (--skip-run)")

    # ── Step 3: Evaluate ──
    print(f"\n▶ Step 3/4: Evaluating W1/W2...")
    evaluate_dataset(
        config_path=args.config,
        agent_id=report_agent_id,
        fold_filter=args.fold,
    )
    print("  ✓ Evaluation complete")

    # ── Step 4: Report ──
    print(f"\n▶ Step 4/4: Generating report...")
    generate_report(
        agent_ids=[report_agent_id],
        output_path=args.output,
        latex=args.latex,
    )
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
