#!/usr/bin/env python
"""
generate_phase2_scenarios.py — Generate all 12 Phase 2 controlled-variable scenarios.

Usage:
    conda activate CytoCompass
    python benchmark/dynbench/generate_phase2_scenarios.py              # dry-run
    python benchmark/dynbench/generate_phase2_scenarios.py --execute    # actually generate

This reads phase2_scenarios.json and calls scenario_factory.py for each.
All scenarios use scenario_seed=42 (Phase 2 design).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DYNBENCH_DIR = Path(__file__).resolve().parent
SCENARIOS_JSON = DYNBENCH_DIR / "phase2_scenarios.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DYNBENCH_DIR / "simulators"))


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 2 scenarios")
    parser.add_argument("--execute", action="store_true",
                        help="Actually generate (default: dry-run)")
    parser.add_argument("--group", default=None,
                        choices=["A", "B", "C"],
                        help="Only generate one group (A=gene, B=fate, C=topo)")
    parser.add_argument("--scenario", default=None,
                        help="Only generate one scenario name (e.g. A1_gene_low)")
    args = parser.parse_args()

    if not SCENARIOS_JSON.exists():
        print(f"ERROR: {SCENARIOS_JSON} not found")
        sys.exit(1)

    with open(SCENARIOS_JSON) as f:
        all_configs = json.load(f)

    # Filter by group or scenario
    configs = all_configs
    if args.group:
        configs = [c for c in configs if c["name"].startswith(args.group.lower())]
    if args.scenario:
        configs = [c for c in configs if c["name"] == args.scenario]

    if not configs:
        print("No matching scenarios found.")
        sys.exit(1)

    print("=" * 70)
    print("  Phase 2 Scenario Generation")
    print(f"  Total: {len(configs)} scenarios")
    print(f"  Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print("=" * 70)

    # Group summary
    groups = {}
    for c in configs:
        g = c["name"][0]
        groups.setdefault(g, []).append(c)
    for g, items in sorted(groups.items()):
        label = {"A": "Gene gradient", "B": "Fate gradient", "C": "Topology contrast"}
        names = [c["name"] for c in items]
        print(f"\n  Group {g} ({label.get(g, '?')}):")
        for n in names:
            c = next(c for c in items if c["name"] == n)
            print(f"    {n}: {c['n_fates']} fates, "
                  f"{c.get('n_confounders', 0)}+{c.get('n_noise_genes', 0)} conf+noise, "
                  f"seed={c['seed']}")

    if not args.execute:
        print("\n  ⓘ Dry-run. Add --execute to generate.")
        return

    # Write a temporary combined config for scenario_factory
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(configs, f, indent=2)
        tmp_config = f.name

    try:
        from scenario_factory import batch_from_configs

        output_base = DYNBENCH_DIR
        results = batch_from_configs(tmp_config, output_base, verbose=True)

        # Update batch_summary.json — append Phase 2 results to existing
        summary_path = DYNBENCH_DIR / "batch_summary.json"
        existing = []
        if summary_path.exists():
            with open(summary_path) as f:
                existing = json.load(f)

        # Add phase2 tag to new results
        for r in results:
            r["phase"] = "phase2"

        combined = existing + results
        with open(summary_path, 'w') as f:
            json.dump(combined, f, indent=2, default=str)

        print(f"\n✓ Updated {summary_path} ({len(combined)} total scenarios)")

    finally:
        os.unlink(tmp_config)


if __name__ == "__main__":
    main()
