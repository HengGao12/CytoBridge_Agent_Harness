#!/usr/bin/env python3
"""Full benchmark iteration loop: 3 agents × 3 seeds on sparse_4fates.

Runs each agent/seed combination, validates outputs, and reports results.
Stops after max_rounds attempts per combination (retry on transient failures).
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark"
TOPOLOGY_SPEC = BENCHMARK_ROOT / "dynbench/simulators/test_configs/branching_tree_4fates_sparse.json"
CHECK_SCRIPT = BENCHMARK_ROOT / "scripts/check_truthful_outputs.py"
CONDA_ENV = "cytobridge"

AGENTS = ["biomini", "codex", "cytobridge"]
SEEDS = [42, 137, 256]
MAX_RETRIES = 2  # per combination

def run_benchmark(agent: str, seed: int, output_dir: Path) -> dict:
    """Run a single agent/seed combination."""
    cmd = [
        "conda", "run", "-n", CONDA_ENV, "--no-capture-output",
        "python", str(BENCHMARK_ROOT / "batch_benchmark_runner.py"),
        "--topology-spec-file", str(TOPOLOGY_SPEC),
        "--difficulties", "medium",
        "--seeds", str(seed),
        "--agents", agent,
        "--output-dir", str(output_dir),
        "--timeout-sec", "900" if agent == "cytobridge" else "600",
    ]
    
    print(f"\n{'='*60}")
    print(f"RUNNING: {agent} / seed {seed}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1200,
        cwd=str(PROJECT_ROOT)
    )
    
    # Parse output for pass/fail
    output = result.stdout + result.stderr
    passed = "passed:" in output.lower()
    total_match = None
    for line in output.split("\n"):
        if "total=" in line.lower():
            try:
                total_str = line.split("total=")[-1].strip()
                if total_str:
                    total_match = float(total_str)
            except (ValueError, IndexError):
                pass
    
    return {
        "agent": agent,
        "seed": seed,
        "passed": passed,
        "total_score": total_match,
        "output": output[-500:] if len(output) > 500 else output,
    }


def find_agent_output(base_dir: Path, agent: str, seed: int) -> Path | None:
    """Find the agent output directory from batch results."""
    # Search for the agent output directory
    for pattern in [
        f"**/agent_{agent}_skills-on_seed{seed}",
        f"**/agent_{agent}_seed{seed}",
    ]:
        matches = list(base_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def check_outputs(agent_dir: Path) -> dict:
    """Run the check script on agent outputs."""
    intermediates_dir = agent_dir / "intermediates"
    if not intermediates_dir.exists():
        return {"passed": False, "error": "intermediates/ directory not found"}
    
    cmd = [
        "conda", "run", "-n", CONDA_ENV, "--no-capture-output",
        "python", str(CHECK_SCRIPT),
        str(intermediates_dir),
        "--agent-log", str(agent_dir),
    ]
    
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60,
        cwd=str(PROJECT_ROOT)
    )
    
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    
    # Parse scores from eval_results.json
    eval_results = agent_dir / "eval_results.json"
    scores = {}
    artifact_origin = "unknown"
    if eval_results.exists():
        try:
            with open(eval_results) as f:
                data = json.load(f)
            pm = data.get("results", {}).get("per_metric", {})
            for k, v in pm.items():
                scores[k] = v.get("score", 0)
            scores["TOTAL"] = data.get("results", {}).get("total_score", 0)
            rr = data.get("run_result", {})
            artifact_origin = rr.get("artifact_origin", "unknown")
        except Exception:
            pass
    
    return {
        "passed": passed,
        "scores": scores,
        "artifact_origin": artifact_origin,
        "check_output": output,
    }


def main():
    results_dir = PROJECT_ROOT / "benchmark/results/truthful_final"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for agent in AGENTS:
        for seed in SEEDS:
            batch_dir = results_dir / f"batch_{agent}_seed{seed}"
            
            for attempt in range(1, MAX_RETRIES + 1):
                print(f"\n>>> Attempt {attempt}/{MAX_RETRIES} for {agent} / seed {seed}")
                
                run_result = run_benchmark(agent, seed, batch_dir)
                
                if run_result["passed"]:
                    # Find and validate outputs
                    agent_dir = find_agent_output(batch_dir, agent, seed)
                    if agent_dir:
                        check = check_outputs(agent_dir)
                        result_entry = {
                            "agent": agent,
                            "seed": seed,
                            "attempt": attempt,
                            "benchmark_passed": True,
                            "check_passed": check["passed"],
                            "scores": check.get("scores", {}),
                            "artifact_origin": check.get("artifact_origin", "unknown"),
                            "check_output": check.get("check_output", ""),
                        }
                        all_results.append(result_entry)
                        
                        if check["passed"]:
                            print(f"  ✓ {agent} seed {seed}: PASS (total={check['scores'].get('TOTAL', 'N/A'):.4f})")
                            break
                        else:
                            print(f"  ✗ {agent} seed {seed}: Check FAILED")
                            if attempt < MAX_RETRIES:
                                print(f"    Retrying...")
                    else:
                        print(f"  ✗ {agent} seed {seed}: Could not find agent output dir")
                        all_results.append({
                            "agent": agent, "seed": seed, "attempt": attempt,
                            "benchmark_passed": True, "check_passed": False,
                            "error": "output dir not found"
                        })
                        break
                else:
                    print(f"  ✗ {agent} seed {seed}: Benchmark FAILED")
                    all_results.append({
                        "agent": agent, "seed": seed, "attempt": attempt,
                        "benchmark_passed": False, "check_passed": False,
                        "total_score": run_result.get("total_score"),
                    })
                    if attempt < MAX_RETRIES:
                        print(f"    Retrying...")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Agent':<12} {'Seed':<8} {'Status':<10} {'M1_vel':<10} {'M2_grw':<10} {'M3_dist':<10} {'M4_fate':<10} {'M5_pert':<10} {'M6_grn':<10} {'TOTAL':<10} {'Origin':<10}")
    print("-"*120)
    
    for r in all_results:
        scores = r.get("scores", {})
        status = "PASS" if r.get("check_passed") else "FAIL"
        print(f"{r['agent']:<12} {r['seed']:<8} {status:<10} "
              f"{scores.get('M1_velocity', 'N/A'):<10} "
              f"{scores.get('M2_growth', 'N/A'):<10} "
              f"{scores.get('M3_distribution', 'N/A'):<10} "
              f"{scores.get('M4_fate', 'N/A'):<10} "
              f"{scores.get('M5_perturbation', 'N/A'):<10} "
              f"{scores.get('M6_grn', 'N/A'):<10} "
              f"{scores.get('TOTAL', 'N/A'):<10} "
              f"{r.get('artifact_origin', 'N/A'):<10}")
    
    # Save results
    results_file = results_dir / "final_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    # Count passes
    passes = sum(1 for r in all_results if r.get("check_passed"))
    total = len(all_results)
    print(f"\nOverall: {passes}/{total} combinations passed all checks")
    
    return 0 if passes == len(AGENTS) * len(SEEDS) else 1


if __name__ == "__main__":
    sys.exit(main())
