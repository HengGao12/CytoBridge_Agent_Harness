#!/usr/bin/env python
"""Summarize CytoBridge DynBench LLM usage and prompt-cache hit rates."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


USAGE_RE = re.compile(
    r"LLM usage prompt=(?P<prompt>\d+) completion=(?P<completion>\d+) "
    r"total=(?P<total>\d+) cached_prompt=(?P<cached>\d+) cache_hit=(?P<cache_hit>[0-9.]+)%"
)


def parse_usage(run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for log_path in sorted((run_root / "batch_runner_logs").glob("*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for idx, match in enumerate(USAGE_RE.finditer(text), 1):
            row = {
                "log": log_path.name,
                "call_index_in_log": idx,
                "prompt_tokens": int(match.group("prompt")),
                "completion_tokens": int(match.group("completion")),
                "total_tokens": int(match.group("total")),
                "cached_prompt_tokens": int(match.group("cached")),
                "cache_hit_pct": float(match.group("cache_hit")),
            }
            row["uncached_prompt_tokens"] = row["prompt_tokens"] - row["cached_prompt_tokens"]
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]], run_root: Path) -> dict[str, object]:
    prompt = sum(int(row["prompt_tokens"]) for row in rows)
    completion = sum(int(row["completion_tokens"]) for row in rows)
    total = sum(int(row["total_tokens"]) for row in rows)
    cached = sum(int(row["cached_prompt_tokens"]) for row in rows)
    uncached = sum(int(row["uncached_prompt_tokens"]) for row in rows)
    logs = {str(row["log"]) for row in rows}
    return {
        "run_root": str(run_root),
        "llm_call_count": len(rows),
        "logs_with_usage": len(logs),
        "prompt_tokens": prompt,
        "uncached_prompt_tokens": uncached,
        "cached_prompt_tokens": cached,
        "completion_tokens": completion,
        "total_tokens": total,
        "aggregate_prompt_cache_hit_pct": round(100.0 * cached / prompt, 4) if prompt else None,
        "mean_call_cache_hit_pct": round(sum(float(row["cache_hit_pct"]) for row in rows) / len(rows), 4)
        if rows
        else None,
        "latest_calls": rows[-10:],
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    rows = parse_usage(run_root)
    summary = summarize(rows, run_root)
    (run_root / "llm_usage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(rows, run_root / "llm_usage_calls.csv")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
