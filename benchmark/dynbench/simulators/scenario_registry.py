"""
DynBench v2 Scenario Registry — deterministic scenario definitions.

All 12 v2 scenarios are defined by config files in test_configs/v2/.
Scenario names are derived purely from config parameters (no seed suffix).
Data files are MD5-frozen for reproducibility.

Usage:
    from dynbench.simulators.scenario_registry import V2_REGISTRY, get_baseline, get_gene_gradient_group
    scenarios = get_gene_gradient_group()  # returns list of config dicts
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_DIR = Path(__file__).resolve().parent / "test_configs" / "v2"

# ══════════════════════════════════════════════════════════════════════════════
#  V2 Scenario Registry
# ══════════════════════════════════════════════════════════════════════════════

# Canonical ordering and metadata for all 12 v2 scenarios
V2_SCENARIO_META: List[Dict[str, Any]] = [
    # ── Baseline ──
    {
        "name": "baseline_25g_8f",
        "config_file": "baseline_25g_8f.json",
        "group": "baseline",
        "difficulty": "medium",
        "description": "Baseline: 25 genes, 8 fates, σ=0.12, 6 timepoints, symmetric binary tree GRN",
    },
    # ── Group A: Gene Gradient (fixed 8 fates, varying gene count) ──
    {
        "name": "gene_gradient_12g_8f",
        "config_file": "gene_gradient_12g_8f.json",
        "group": "gene_gradient",
        "difficulty": "easy",
        "description": "Group A low: 12 genes, 8 fates, minimal reporters",
    },
    {
        "name": "gene_gradient_18g_8f",
        "config_file": "gene_gradient_18g_8f.json",
        "group": "gene_gradient",
        "difficulty": "medium",
        "description": "Group A mid: 18 genes, 8 fates, moderate complexity",
    },
    {
        "name": "gene_gradient_35g_8f",
        "config_file": "gene_gradient_35g_8f.json",
        "group": "gene_gradient",
        "difficulty": "hard",
        "description": "Group A high: 35 genes, 8 fates, many reporters/confounders/noise",
    },
    # ── Group B: Fate Gradient (fixed 25 genes, varying fate count) ──
    {
        "name": "fate_gradient_25g_2f",
        "config_file": "fate_gradient_25g_2f.json",
        "group": "fate_gradient",
        "difficulty": "easy",
        "description": "Group B low: 25 genes, 2 fates, 1 toggle pair, many confounders",
    },
    {
        "name": "fate_gradient_25g_4f",
        "config_file": "fate_gradient_25g_4f.json",
        "group": "fate_gradient",
        "difficulty": "medium",
        "description": "Group B mid: 25 genes, 4 fates, 2 toggle pairs",
    },
    {
        "name": "fate_gradient_25g_16f",
        "config_file": "fate_gradient_25g_16f.json",
        "group": "fate_gradient",
        "difficulty": "hard",
        "description": "Group B high: 25 genes, 16 fates, 4 toggle pairs, dense TF network",
    },
    # ── Group C: Topology (fixed 25 genes, 4 fates, varying topology) ──
    {
        "name": "topology_dag_25g_4f",
        "config_file": "topology_dag_25g_4f.json",
        "group": "topology",
        "difficulty": "medium",
        "description": "Group C DAG: 25 genes, 4 fates, standard layered branching tree",
    },
    {
        "name": "topology_feedback_25g_4f",
        "config_file": "topology_feedback_25g_4f.json",
        "group": "topology",
        "difficulty": "hard",
        "description": "Group C feedback: 25 genes, 4 fates, reporter-to-TF feedback edges",
    },
    {
        "name": "topology_redundant_25g_4f",
        "config_file": "topology_redundant_25g_4f.json",
        "group": "topology",
        "difficulty": "hard",
        "description": "Group C redundant: 25 genes, 4 fates, parallel regulation paths",
    },
    # ── Difficulty Variants ──
    {
        "name": "baseline_25g_8f_medium",
        "config_file": "baseline_25g_8f_medium.json",
        "group": "difficulty_variant",
        "difficulty": "hard",
        "description": "Baseline variant: 25 genes, 8 fates, higher noise σ=0.15",
    },
    {
        "name": "baseline_25g_8f_extreme",
        "config_file": "baseline_25g_8f_extreme.json",
        "group": "difficulty_variant",
        "difficulty": "extreme",
        "description": "Baseline variant: 25 genes, 8 fates, 3 holdouts, σ=0.20",
    },
]

# Lookup by name
V2_REGISTRY: Dict[str, Dict[str, Any]] = {
    meta["name"]: meta for meta in V2_SCENARIO_META
}

V2_ALL_NAMES: List[str] = [meta["name"] for meta in V2_SCENARIO_META]


# ══════════════════════════════════════════════════════════════════════════════
#  Config Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_v2_config(name: str) -> Dict[str, Any]:
    """Load a v2 scenario config by name."""
    meta = V2_REGISTRY.get(name)
    if meta is None:
        raise KeyError(f"Unknown v2 scenario: {name}. Available: {V2_ALL_NAMES}")
    config_path = CONFIG_DIR / meta["config_file"]
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def load_all_v2_configs() -> Dict[str, Dict[str, Any]]:
    """Load all 12 v2 scenario configs."""
    configs = {}
    for meta in V2_SCENARIO_META:
        configs[meta["name"]] = load_v2_config(meta["name"])
    return configs


# ══════════════════════════════════════════════════════════════════════════════
#  Group Accessors
# ══════════════════════════════════════════════════════════════════════════════

def get_baseline() -> Dict[str, Any]:
    """Return the baseline config."""
    return load_v2_config("baseline_25g_8f")


def get_gene_gradient_group() -> List[Dict[str, Any]]:
    """Return all gene gradient configs (Group A), ordered by gene count."""
    names = ["gene_gradient_12g_8f", "gene_gradient_18g_8f",
             "baseline_25g_8f", "gene_gradient_35g_8f"]
    return [load_v2_config(n) for n in names]


def get_fate_gradient_group() -> List[Dict[str, Any]]:
    """Return all fate gradient configs (Group B), ordered by fate count."""
    names = ["fate_gradient_25g_2f", "fate_gradient_25g_4f",
             "fate_gradient_25g_16f"]
    return [load_v2_config(n) for n in names]


def get_topology_group() -> List[Dict[str, Any]]:
    """Return all topology variant configs (Group C)."""
    names = ["topology_dag_25g_4f", "topology_feedback_25g_4f",
             "topology_redundant_25g_4f"]
    return [load_v2_config(n) for n in names]


def get_difficulty_variants() -> List[Dict[str, Any]]:
    """Return difficulty variant configs (medium and extreme noise)."""
    names = ["baseline_25g_8f_medium", "baseline_25g_8f_extreme"]
    return [load_v2_config(n) for n in names]


def get_all_v2_names() -> List[str]:
    """Return all v2 scenario names."""
    return list(V2_ALL_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
#  MD5 Frozen Data Verification
# ══════════════════════════════════════════════════════════════════════════════

def compute_file_md5(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_scenario_md5(gt_dir: Path) -> Dict[str, str]:
    """Compute MD5 hashes for all ground truth files in a scenario directory."""
    hashes = {}
    if not gt_dir.exists():
        return hashes
    for f in sorted(gt_dir.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            hashes[f.name] = compute_file_md5(f)
    return hashes


def store_md5_manifest(gt_dir: Path) -> Path:
    """Compute and store MD5 manifest for a scenario's ground truth files."""
    hashes = compute_scenario_md5(gt_dir)
    manifest_path = gt_dir / "md5_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(hashes, f, indent=2)
    return manifest_path


def verify_md5_manifest(gt_dir: Path) -> Dict[str, Any]:
    """
    Verify ground truth files against stored MD5 manifest.
    Returns dict with 'passed' bool, 'mismatches' list, and 'missing' list.
    """
    manifest_path = gt_dir / "md5_manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "error": "No md5_manifest.json found",
                "mismatches": [], "missing": []}

    with open(manifest_path) as f:
        expected = json.load(f)

    mismatches = []
    missing = []
    for fname, expected_md5 in expected.items():
        fpath = gt_dir / fname
        if not fpath.exists():
            missing.append(fname)
        else:
            actual_md5 = compute_file_md5(fpath)
            if actual_md5 != expected_md5:
                mismatches.append({"file": fname, "expected": expected_md5,
                                   "actual": actual_md5})

    return {
        "passed": len(mismatches) == 0 and len(missing) == 0,
        "mismatches": mismatches,
        "missing": missing,
    }
