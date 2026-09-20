from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from benchmark.agent_config import load_saved_config
from cytobridge_agent.utils.llm_factory import instantiate_llm


ALLOWED_TOPOLOGY_FAMILIES = {
    "branching_tree",
    "asymmetric_tree",
    "branching_with_feedback",
    "competitive_multistable",
    "compact_toggle",
}


def _strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return value


def _response_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def normalize_topology_spec(spec: dict[str, Any], *, request_text: str, difficulty: str, seed: int) -> dict[str, Any]:
    family_raw = str(spec.get("topology_family") or "branching_tree").strip().lower().replace("-", "_")
    alias_map = {
        "tree": "branching_tree",
        "branching": "branching_tree",
        "hierarchical_tree": "branching_tree",
        "feedback_tree": "branching_with_feedback",
        "feedback_branching": "branching_with_feedback",
        "competitive": "competitive_multistable",
        "multistable": "competitive_multistable",
        "toggle": "compact_toggle",
        "toggle_switch": "compact_toggle",
    }
    family = alias_map.get(family_raw, family_raw)
    if family not in ALLOWED_TOPOLOGY_FAMILIES:
        raise ValueError(
            f"Unsupported topology_family '{family}'. Expected one of: {', '.join(sorted(ALLOWED_TOPOLOGY_FAMILIES))}."
        )

    normalized = {
        "schema_version": "dynbench_topology_spec_v1",
        "request_text": request_text,
        "difficulty": difficulty,
        "seed": seed,
        "topology_family": family,
        "n_fates": int(spec.get("n_fates", 4)),
        "n_time_bins": int(spec.get("n_time_bins", 8)),
        "n_init_cells": int(spec.get("n_init_cells", 2000)),
        "n_reporters_per_tf": int(spec.get("n_reporters_per_tf", 1)),
        "n_confounders": int(spec.get("n_confounders", 2)),
        "n_noise_genes": int(spec.get("n_noise_genes", 0)),
        "growth_type": str(spec.get("growth_type", "multi")).strip().lower(),
        "noise_sigma": float(spec.get("noise_sigma", 0.12)),
        "t_end": float(spec.get("t_end", 12.0)),
        "holdout_bins": [int(x) for x in spec.get("holdout_bins", [2, 3])],
        "feedback_strength": float(spec.get("feedback_strength", 0.0)),
        "cross_branch_coupling": float(spec.get("cross_branch_coupling", 0.0)),
        "asymmetry_strength": float(spec.get("asymmetry_strength", 0.0)),
        "density_bias": str(spec.get("density_bias", "medium")).strip().lower(),
        "transient_state_depth": int(spec.get("transient_state_depth", 1)),
        "notes": str(spec.get("notes", "")).strip(),
    }

    normalized["n_fates"] = max(2, min(16, normalized["n_fates"]))
    normalized["n_reporters_per_tf"] = max(1, min(2, normalized["n_reporters_per_tf"]))
    normalized["n_confounders"] = max(0, normalized["n_confounders"])
    normalized["n_noise_genes"] = max(0, normalized["n_noise_genes"])
    normalized["n_time_bins"] = max(4, normalized["n_time_bins"])
    normalized["n_init_cells"] = max(200, normalized["n_init_cells"])
    normalized["feedback_strength"] = max(0.0, min(1.0, normalized["feedback_strength"]))
    normalized["cross_branch_coupling"] = max(0.0, min(1.0, normalized["cross_branch_coupling"]))
    normalized["asymmetry_strength"] = max(0.0, min(1.0, normalized["asymmetry_strength"]))
    normalized["transient_state_depth"] = max(0, min(3, normalized["transient_state_depth"]))
    if normalized["growth_type"] not in {"single", "multi", "interaction"}:
        normalized["growth_type"] = "multi"
    if normalized["density_bias"] not in {"low", "medium", "high"}:
        normalized["density_bias"] = "medium"
    if family == "compact_toggle":
        normalized.update(
            {
                "n_fates": 2,
                "n_reporters_per_tf": 1,
                "n_confounders": 0,
                "n_noise_genes": 0,
                "growth_type": "single",
            }
        )
    return normalized


def design_topology_spec(
    *,
    request_text: str,
    difficulty: str,
    seed: int,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_auth_mode: Optional[str] = None,
    llm_profile_id: Optional[str] = None,
    llm_thinking_level: Optional[str] = None,
) -> dict[str, Any]:
    saved = load_saved_config()
    llm, _ = instantiate_llm(
        model=llm_model or saved.get("llm_model", "gpt-5.4"),
        base_url=llm_base_url or saved.get("llm_base_url"),
        api_key=llm_api_key or saved.get("openai_api_key"),
        auth_mode=llm_auth_mode or saved.get("llm_auth_mode", "auto"),
        preferred_profile_id=llm_profile_id or saved.get("llm_profile_id"),
        thinking_level=llm_thinking_level or saved.get("llm_thinking_level", "low"),
    )

    prompt = f"""
You are designing a DynBench synthetic topology specification.
Return JSON only. No markdown. No prose.

User request:
{request_text}

Difficulty: {difficulty}
Seed: {seed}

Allowed topology_family values:
- branching_tree
- asymmetric_tree
- branching_with_feedback
- competitive_multistable
- compact_toggle

Return exactly one JSON object with these fields:
- topology_family
- n_fates
- n_time_bins
- n_init_cells
- n_reporters_per_tf
- n_confounders
- n_noise_genes
- growth_type
- noise_sigma
- t_end
- holdout_bins
- feedback_strength
- cross_branch_coupling
- asymmetry_strength
- density_bias
- transient_state_depth
- notes

Constraints:
- keep n_fates between 2 and 16
- holdout_bins must be a list of integer time bins
- growth_type must be one of single, multi, interaction
- density_bias must be one of low, medium, high
- if the request is compact/simple, prefer compact_toggle
- if the request mentions tree/hierarchy, prefer branching_tree or asymmetric_tree
- if the request mentions feedback/competition, prefer branching_with_feedback or competitive_multistable
""".strip()
    response = llm.invoke(prompt)
    text = _strip_code_fence(_response_to_text(response))
    parsed = json.loads(text)
    return normalize_topology_spec(parsed, request_text=request_text, difficulty=difficulty, seed=seed)


def save_topology_spec(spec: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return target
