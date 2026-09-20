#!/usr/bin/env python
"""Generate a Markdown summary from run_all_benchmarks.py output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_records(input_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(input_dir.glob("*/eval_results.json")):
        try:
            records.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            records.append(
                {
                    "status": "failed",
                    "benchmark": path.parent.name,
                    "agent_type": "",
                    "seed": "",
                    "runtime_sec": 0.0,
                    "scores": {},
                    "error": f"Invalid JSON: {path}",
                }
            )
    return records


def score_text(scores: dict[str, Any]) -> str:
    total = scores.get("total_score")
    if isinstance(total, (int, float)):
        return f"{total:.3f}"
    return ""


def render_markdown(records: list[dict[str, Any]], input_dir: Path) -> str:
    succeeded = sum(1 for record in records if record.get("status") == "succeeded")
    failed = len(records) - succeeded
    lines = [
        "# CytoBridge Benchmark Report",
        "",
        f"Input directory: `{input_dir}`",
        f"Runs: {len(records)}; succeeded: {succeeded}; failed: {failed}",
        "",
        "| Benchmark | Agent | Seed | Status | Runtime sec | Total score | Error |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for record in records:
        err = (record.get("error") or "").replace("\n", " ")[:220]
        lines.append(
            f"| {record.get('benchmark', '')} | {record.get('agent_type', '')} | "
            f"{record.get('seed', '')} | {record.get('status', '')} | "
            f"{float(record.get('runtime_sec') or 0.0):.1f} | "
            f"{score_text(record.get('scores') or {})} | {err} |"
        )

    lines.extend(["", "## Metric Details", ""])
    for record in records:
        scores = record.get("scores") or {}
        if not scores:
            continue
        lines.append(f"### {record.get('benchmark')} seed={record.get('seed')}")
        for key, value in scores.items():
            if isinstance(value, (int, float)):
                lines.append(f"- `{key}`: {value:.4f}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown summary for benchmark batch results.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    records = load_records(input_dir)
    Path(args.output).write_text(render_markdown(records, input_dir))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
