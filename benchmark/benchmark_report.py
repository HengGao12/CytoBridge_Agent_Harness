#!/usr/bin/env python
"""
benchmark_report.py — Aggregate and visualize benchmark results.

Collects all metrics.json files from benchmark/results/ and produces:
- A summary table (printed to console)
- A CSV/LaTeX-ready table
- Optionally bar charts comparing agents

Usage:
    python -m benchmark.benchmark_report
    python -m benchmark.benchmark_report --agent cytobridge --output benchmark_summary.csv
    python -m benchmark.benchmark_report --latex
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .benchmark_utils import collect_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def aggregate_metrics(
    all_metrics: List[Dict],
) -> List[Dict]:
    """
    Group metrics by (agent, dataset, fold) and compute stats across seeds.

    Returns a list of summary dicts, each with mean/std for W1/W2 and
    success count.
    """
    key_fn = lambda m: (m["agent_id"], m["dataset_id"], m["fold_id"])

    groups = defaultdict(list)
    for m in all_metrics:
        groups[key_fn(m)].append(m)

    summaries = []
    for (agent, dataset, fold), runs in sorted(groups.items()):
        n_total = len(runs)
        n_success = sum(1 for r in runs if r.get("success"))

        w1_vals = [r["W1"] for r in runs if r.get("W1") is not None]
        w2_vals = [r["W2"] for r in runs if r.get("W2") is not None]

        import numpy as np

        summary = {
            "agent_id": agent,
            "dataset_id": dataset,
            "fold_id": fold,
            "held_out": runs[0].get("held_out_timepoint", "?"),
            "n_runs": n_total,
            "n_success": n_success,
            "success_rate": f"{n_success}/{n_total}",
            "W1_mean": float(np.mean(w1_vals)) if w1_vals else None,
            "W1_std": float(np.std(w1_vals)) if len(w1_vals) > 1 else 0.0,
            "W2_mean": float(np.mean(w2_vals)) if w2_vals else None,
            "W2_std": float(np.std(w2_vals)) if len(w2_vals) > 1 else 0.0,
            "weighted": any(r.get("weighted") for r in runs),
        }
        summaries.append(summary)

    return summaries


def format_table(summaries: List[Dict]) -> str:
    """Format summaries as a readable ASCII table."""
    if not summaries:
        return "No results found."

    headers = [
        "Agent", "Dataset", "Fold", "Held-out",
        "Success", "W1 (mean±std)", "W2 (mean±std)", "Weighted",
    ]
    rows = []
    for s in summaries:
        w1_str = f"{s['W1_mean']:.4f}±{s['W1_std']:.4f}" if s["W1_mean"] is not None else "N/A"
        w2_str = f"{s['W2_mean']:.4f}±{s['W2_std']:.4f}" if s["W2_mean"] is not None else "N/A"
        rows.append([
            s["agent_id"], s["dataset_id"], s["fold_id"], s["held_out"],
            s["success_rate"], w1_str, w2_str, "Y" if s["weighted"] else "N",
        ])

    # Compute column widths
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_line = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, widths)) + "|"

    lines = [sep, header_line, sep]
    for row in rows:
        line = "|" + "|".join(f" {str(v):<{w}} " for v, w in zip(row, widths)) + "|"
        lines.append(line)
    lines.append(sep)

    return "\n".join(lines)


def format_latex(summaries: List[Dict]) -> str:
    """Format summaries as a LaTeX table."""
    if not summaries:
        return "% No results"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Benchmark results: held-out time-point prediction}",
        r"\label{tab:benchmark}",
        r"\begin{tabular}{llllccc}",
        r"\toprule",
        r"Agent & Dataset & Fold & Held-out & Success & W1 & W2 \\",
        r"\midrule",
    ]

    for s in summaries:
        w1 = f"${s['W1_mean']:.3f} \\pm {s['W1_std']:.3f}$" if s["W1_mean"] is not None else "---"
        w2 = f"${s['W2_mean']:.3f} \\pm {s['W2_std']:.3f}$" if s["W2_mean"] is not None else "---"
        lines.append(
            f"{s['agent_id']} & {s['dataset_id']} & {s['fold_id']} & "
            f"{s['held_out']} & {s['success_rate']} & {w1} & {w2} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def format_csv(summaries: List[Dict]) -> str:
    """Format summaries as CSV."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "agent_id", "dataset_id", "fold_id", "held_out",
        "n_runs", "n_success", "success_rate",
        "W1_mean", "W1_std", "W2_mean", "W2_std", "weighted",
    ])
    writer.writeheader()
    writer.writerows(summaries)
    return output.getvalue()


def generate_report(
    agent_ids: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    latex: bool = False,
) -> None:
    """Generate and print/save the benchmark report."""
    all_metrics = collect_all_metrics(agent_ids)
    if not all_metrics:
        logger.warning("No metrics found in benchmark/results/")
        print("No benchmark results found. Run benchmark_run.py and benchmark_eval.py first.")
        return

    summaries = aggregate_metrics(all_metrics)

    # Print to console
    print("\n" + "=" * 80)
    print("BENCHMARK REPORT")
    print("=" * 80 + "\n")
    print(format_table(summaries))

    if latex:
        print("\n" + "=" * 80)
        print("LaTeX Table")
        print("=" * 80 + "\n")
        print(format_latex(summaries))

    # Save if output path specified
    if output_path:
        output_path = Path(output_path)
        suffix = output_path.suffix.lower()
        if suffix == ".csv":
            content = format_csv(summaries)
        elif suffix == ".tex":
            content = format_latex(summaries)
        else:
            content = format_table(summaries)
        output_path.write_text(content)
        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--agent", nargs="*", default=None, help="Filter by agent(s)")
    parser.add_argument("--output", default=None, help="Save report to file (.csv, .tex, or .txt)")
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX table")

    args = parser.parse_args()

    try:
        generate_report(
            agent_ids=args.agent,
            output_path=args.output,
            latex=args.latex,
        )
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
