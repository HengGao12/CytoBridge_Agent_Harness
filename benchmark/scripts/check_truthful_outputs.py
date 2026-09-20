#!/usr/bin/env python3
"""Check that agent outputs are genuinely produced by the agent, not by fallback/recovery mechanisms.

Usage:
    python scripts/check_truthful_outputs.py <intermediates_dir> [--agent-log <path>]

Checks performed:
    A. No recovery/fallback markers in logs
    B. velocity_field.json: per-cell variation (not constant)
    C. growth_rates.json: per-cell variation (not constant)
    D. Performance sanity: scores should differ from known fallback values
    E. Constant column detection (BioMini-style {data:[], columns:[]} format)
    F. No recovery_manifest.json present
    G. per_cell_fate.json: not all identical
    H. holdout_prediction.json: not all identical predictions
    I. artifact_origin must be "native"

Exit code 0 = all checks passed, 1 = at least one check failed.
"""

import json
import sys
from pathlib import Path


def check_no_fallback_in_logs(agent_log: Path) -> list[str]:
    """Check agent_log.txt and error.json for fallback/recovery markers."""
    issues = []
    markers = [
        "deterministic fallback",
        "deterministic_export",
        "recovery_mode",
        "intermediate_contract_conversion",
        "intermediate_conversion",
        "recovered_complete",
        "artifact_origin=recovered",
    ]
    
    for fname in ["agent_log.txt", "error.json", "runner_stdout.log", "agent_log.json"]:
        fpath = agent_log.parent / fname
        if not fpath.exists():
            continue
        content = fpath.read_text(encoding="utf-8", errors="ignore")
        for marker in markers:
            if marker.lower() in content.lower():
                issues.append(f"FOUND '{marker}' in {fname}")
    
    return issues


def check_velocity_per_cell(intermediates_dir: Path) -> list[str]:
    """Check velocity_field.json has per-cell variation."""
    issues = []
    vf = intermediates_dir / "velocity_field.json"
    if not vf.exists():
        issues.append("velocity_field.json does not exist")
        return issues
    
    with open(vf) as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        # Could be {gene_name: [values...]} or {metadata..., data: [...]}
        # Check for common keys
        genes = {k: v for k, v in data.items() 
                 if isinstance(v, list) and len(v) > 10}
        if not genes:
            # Check nested structure
            for k, v in data.items():
                if isinstance(v, dict):
                    sub_genes = {sk: sv for sk, sv in v.items() 
                                if isinstance(sv, list) and len(sv) > 10}
                    if sub_genes:
                        genes = sub_genes
                        break
        
        if not genes:
            issues.append("velocity_field.json: could not find gene velocity arrays")
            return issues
        
        for gene, values in genes.items():
            unique_vals = set()
            for v in values[:100]:  # sample first 100
                if isinstance(v, (int, float)):
                    unique_vals.add(round(v, 8))
                elif isinstance(v, str):
                    unique_vals.add(v)
            if len(unique_vals) == 1:
                issues.append(f"velocity_field.json: gene '{gene}' has CONSTANT velocity {values[0]} across all cells")
    
    elif isinstance(data, list):
        # List of dicts - check if all rows are identical
        if len(data) > 1:
            try:
                keys = [k for k in data[0].keys() if k.startswith("velocity_") or k.startswith("vel_")]
                for k in keys[:5]:  # check first 5 velocity columns
                    vals = set()
                    for row in data[:100]:
                        v = row.get(k)
                        if isinstance(v, (int, float)):
                            vals.add(round(v, 8))
                    if len(vals) == 1:
                        issues.append(f"velocity_field.json: column '{k}' has CONSTANT value across all cells")
            except (KeyError, TypeError) as e:
                issues.append(f"velocity_field.json: error reading list format: {e}")
    
    return issues


def check_growth_per_cell(intermediates_dir: Path) -> list[str]:
    """Check growth_rates.json has per-cell variation."""
    issues = []
    gr = intermediates_dir / "growth_rates.json"
    if not gr.exists():
        issues.append("growth_rates.json does not exist")
        return issues
    
    with open(gr) as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        rates = data.get("growth_rate") or data.get("rates")
        if isinstance(rates, list) and len(rates) > 10:
            unique_vals = set(round(v, 8) for v in rates[:100] if isinstance(v, (int, float)))
            if len(unique_vals) == 1:
                issues.append(f"growth_rates.json: ALL cells have CONSTANT growth_rate {rates[0]}")
    elif isinstance(data, list) and len(data) > 10:
        # Check first column or 'growth_rate' key
        try:
            if "growth_rate" in data[0]:
                vals = set(round(row["growth_rate"], 8) for row in data[:100] if isinstance(row.get("growth_rate"), (int, float)))
                if len(vals) == 1:
                    issues.append(f"growth_rates.json: ALL cells have CONSTANT growth_rate {data[0]['growth_rate']}")
        except (KeyError, TypeError) as e:
            issues.append(f"growth_rates.json: error reading list format: {e}")
    
    return issues


def check_performance_sanity(eval_results: Path) -> list[str]:
    """Check that known fallback values are not present."""
    issues = []
    if not eval_results.exists():
        return issues  # no eval results yet, skip
    
    with open(eval_results) as f:
        data = json.load(f)
    
    results = data.get("results", {})
    pm = results.get("per_metric", {})
    
    # Known fallback values from previous analysis
    fallback_signatures = {
        ("branching_16fates", "M1_velocity"): 0.2787,
        ("branching_16fates", "M2_growth"): 0.5282,
        ("branching_16fates", "M6_grn"): 0.0271,
        ("asymmetric_tree", "M1_velocity"): 0.3170,
        ("asymmetric_tree", "M2_growth"): 0.5755,
        ("asymmetric_tree", "M4_fate"): 0.0843,
        ("asymmetric_tree", "M6_grn"): 0.0371,
    }
    
    for (scenario, metric), fallback_val in fallback_signatures.items():
        score = pm.get(metric, {}).get("score")
        if score is not None and abs(score - fallback_val) < 0.001:
            issues.append(f"PERFORMANCE: {metric}={score:.4f} matches known fallback value {fallback_val} (scenario: {scenario})")
    
    # Check artifact_origin
    run_result = data.get("run_result", {})
    origin = run_result.get("artifact_origin", "")
    recovery = run_result.get("recovery_method", "none")
    if origin == "recovered":
        issues.append(f"PERFORMANCE: artifact_origin='recovered' — outputs are from recovery, not native agent")
    if recovery not in ("none",):
        issues.append(f"PERFORMANCE: recovery_method='{recovery}' — outputs may not be from agent's own pipeline")
    
    return issues


def check_no_constant_columns(intermediates_dir: Path) -> list[str]:
    """Extra check: detect BioMini-style constant outputs at column level."""
    issues = []
    vf = intermediates_dir / "velocity_field.json"
    if not vf.exists():
        return issues
    
    with open(vf) as f:
        data = json.load(f)
    
    # Handle dict format {data: [...], columns: [...]}
    if isinstance(data, dict):
        rows = data.get("data", [])
        cols = data.get("columns", [])
    # Handle list-of-dicts format
    elif isinstance(data, list) and len(data) > 0:
        rows = data
        cols = list(data[0].keys())
    else:
        return issues
    
    if not rows or not cols:
        return issues
    
    # Check ALL columns, not just first 5
    constant_count = 0
    for col in cols:
        try:
            vals = [row[col] for row in rows[:200] if col in row]
            if vals and all(abs(v - vals[0]) < 1e-10 for v in vals):
                constant_count += 1
                if constant_count <= 3:
                    issues.append(f"CONSTANT COLUMN: velocity_field '{col}' has identical values across all sampled cells: {vals[0]}")
        except (KeyError, TypeError):
            continue
    if constant_count > 3:
        issues.append(f"CONSTANT COLUMN: ... and {constant_count - 3} more constant columns detected")
    if constant_count > 0:
        issues.append(f"CONSTANT COLUMN: {constant_count}/{len(cols)} velocity columns are constant — strong fallback indicator")
    
    return issues


def check_no_recovery_manifest(intermediates_dir: Path) -> list[str]:
    """Check that no old-style recovery_manifest.json exists.
    
    Note: biomni_replay_manifest.json is expected — it's the primary execution
    mechanism for BioMini, not a recovery/fallback artifact.
    """
    issues = []
    for name in ["recovery_manifest.json", "format_recovery_manifest.json"]:
        p = intermediates_dir.parent / name
        if p.exists():
            issues.append(f"RECOVERY ARTIFACT: {name} exists — recovery was applied")
    # Also check in intermediates dir itself
    for name in ["recovery_manifest.json"]:
        p = intermediates_dir / name
        if p.exists():
            issues.append(f"RECOVERY ARTIFACT: {name} exists in intermediates/ — recovery was applied")
    return issues


def check_per_cell_fate(intermediates_dir: Path) -> list[str]:
    """Check per_cell_fate.json is not all identical."""
    issues = []
    fpath = intermediates_dir / "per_cell_fate.json"
    if not fpath.exists():
        return issues  # missing is OK (evaluator gives 0)
    
    with open(fpath) as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        # Check if all cells have the same fate
        fates = data.get("fate") or data.get("fates") or data.get("predicted_fate")
        if isinstance(fates, list) and len(fates) > 10:
            unique = set(str(v) for v in fates[:200])
            if len(unique) == 1:
                issues.append(f"per_cell_fate.json: ALL cells have identical fate '{fates[0]}'")
    elif isinstance(data, list) and len(data) > 10:
        try:
            fate_key = next((k for k in data[0] if "fate" in k.lower()), None)
            if fate_key:
                unique = set(str(row.get(fate_key)) for row in data[:200])
                if len(unique) == 1:
                    issues.append(f"per_cell_fate.json: ALL cells have identical {fate_key} '{data[0][fate_key]}'")
        except (StopIteration, KeyError, TypeError):
            pass
    
    return issues


def check_holdout_prediction(intermediates_dir: Path) -> list[str]:
    """Check holdout_prediction.json is not all identical."""
    issues = []
    fpath = intermediates_dir / "holdout_prediction.json"
    if not fpath.exists():
        return issues  # missing is OK (evaluator gives 0)
    
    try:
        with open(fpath) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        issues.append("holdout_prediction.json: invalid JSON")
        return issues
    
    if isinstance(data, dict):
        # Check expression values
        expr = data.get("expression") or data.get("predicted_expression")
        if isinstance(expr, dict):
            for gene, vals in list(expr.items())[:5]:
                if isinstance(vals, list) and len(vals) > 10:
                    unique = set(round(v, 8) for v in vals[:200] if isinstance(v, (int, float)))
                    if len(unique) == 1:
                        issues.append(f"holdout_prediction.json: gene '{gene}' has CONSTANT predicted expression {vals[0]}")
    
    return issues


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <intermediates_dir> [--agent-log <path>]")
        sys.exit(2)
    
    intermediates_dir = Path(sys.argv[1])
    agent_log_dir = intermediates_dir.parent  # default: same dir as intermediates
    
    # Parse --agent-log
    if "--agent-log" in sys.argv:
        idx = sys.argv.index("--agent-log")
        if idx + 1 < len(sys.argv):
            agent_log_dir = Path(sys.argv[idx + 1])
    
    all_issues = []
    
    # A. No fallback markers
    print("=== Check A: No fallback/recovery markers ===")
    issues = check_no_fallback_in_logs(agent_log_dir)
    if issues:
        all_issues.extend(issues)
        for i in issues:
            print(f"  FAIL: {i}")
    else:
        print("  PASS: No fallback markers found")
    
    # B. Velocity per-cell variation
    print("\n=== Check B: velocity_field.json per-cell variation ===")
    issues = check_velocity_per_cell(intermediates_dir)
    if issues:
        all_issues.extend(issues)
        for i in issues:
            print(f"  FAIL: {i}")
    else:
        print("  PASS: velocity has per-cell variation")
    
    # C. Growth rate per-cell variation
    print("\n=== Check C: growth_rates.json per-cell variation ===")
    issues = check_growth_per_cell(intermediates_dir)
    if issues:
        all_issues.extend(issues)
        for i in issues:
            print(f"  FAIL: {i}")
    else:
        print("  PASS: growth rates have per-cell variation")
    
    # D. Performance sanity
    print("\n=== Check D: Performance sanity ===")
    eval_results = intermediates_dir.parent / "eval_results.json"
    issues = check_performance_sanity(eval_results)
    if issues:
        all_issues.extend(issues)
        for i in issues:
            print(f"  FAIL: {i}")
    else:
        print("  PASS: No fallback performance signatures detected")
    
    # E. Constant column detection (BioMini-style)
    print("\n=== Check E: Constant column detection ===")
    issues = check_no_constant_columns(intermediates_dir)
    if issues:
        all_issues.extend(issues)
        for i in issues:
            print(f"  FAIL: {i}")
    else:
        print("  PASS: No constant columns detected in velocity field")
    
    # F. No recovery manifest
    print("\n=== Check F: No recovery artifacts ===")
    issues = check_no_recovery_manifest(intermediates_dir)
    if issues:
        all_issues.extend(issues)
        for i in issues:
            print(f"  FAIL: {i}")
    else:
        print("  PASS: No recovery manifests found")
    
    # G. per_cell_fate check
    print("\n=== Check G: per_cell_fate.json variation ===")
    issues = check_per_cell_fate(intermediates_dir)
    if issues:
        all_issues.extend(issues)
        for i in issues:
            print(f"  FAIL: {i}")
    else:
        print("  PASS: per_cell_fate has variation or is absent")
    
    # H. holdout_prediction check
    print("\n=== Check H: holdout_prediction.json variation ===")
    issues = check_holdout_prediction(intermediates_dir)
    if issues:
        all_issues.extend(issues)
        for i in issues:
            print(f"  FAIL: {i}")
    else:
        print("  PASS: holdout_prediction has variation or is absent")
    
    # Summary
    print(f"\n{'='*60}")
    if all_issues:
        print(f"RESULT: FAILED ({len(all_issues)} issues)")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
