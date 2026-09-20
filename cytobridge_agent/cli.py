#!/usr/bin/env python
"""
CellCompass CLI - unified runtime for single-cell dynamics workflows.

Usage:
    # Inspect data
    cellcompass inspect path/to/data.h5ad
    
    # Run a dataset workflow through the unified runtime
    cellcompass run path/to/data.h5ad \\
        --question "What are the terminal cell fates in this developmental trajectory?" \\
        --time-key "day" \\
        --label-key "celltype" \\
        --device cuda
    
    # With a configured LLM endpoint
    cellcompass run path/to/data.h5ad \\
        --llm-base-url http://localhost:8000/v1 \\
        --llm-model qwen2-72b
    # Run an interactive session
    cellcompass interactive \\
        --llm-base-url http://localhost:8000/v1 \\
        --llm-model qwen2-72b
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from cytobridge_agent.utils.config_manager import get_saved_config
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cytobridge_agent")


def _normalize_llm_provider(value: Any) -> str:
    from cytobridge_agent.utils.llm_providers import normalize_llm_provider

    return normalize_llm_provider(value)


def _normalize_auth_mode(value: Any) -> str:
    from cytobridge_agent.utils.llm_factory import normalize_auth_mode

    return normalize_auth_mode(value)


def _normalize_llm_reasoning_effort(value: Any, *, default: str = "low") -> str:
    from cytobridge_agent.utils.llm_factory import normalize_llm_reasoning_effort

    return normalize_llm_reasoning_effort(value, default=default)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    saved_config = get_saved_config()

    parser = argparse.ArgumentParser(
        prog="cellcompass",
        description="CytoBridge / CellCompass agent. Starts an interactive session by default.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="cytobridge-agent 1.0.0",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # =========================================================================
    # auth command
    # =========================================================================
    auth_parser = subparsers.add_parser(
        "auth",
        help="Manage OAuth-backed LLM profiles",
    )
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command", help="Auth commands")

    auth_codex_parser = auth_subparsers.add_parser(
        "codex-login",
        help="Login with Codex OAuth and store the profile locally",
    )
    auth_codex_parser.add_argument(
        "--profile-id",
        type=str,
        default=None,
        help="Optional explicit profile id. Defaults to openai-codex:<email>.",
    )

    auth_profiles_parser = auth_subparsers.add_parser(
        "profiles",
        help="Inspect and manage stored OAuth profiles",
    )
    profiles_subparsers = auth_profiles_parser.add_subparsers(
        dest="auth_profiles_command",
        help="Profile operations",
    )

    profiles_list_parser = profiles_subparsers.add_parser("list", help="List OAuth profiles")
    profiles_list_parser.add_argument(
        "--provider",
        type=str,
        default="openai-codex",
        help="Provider id to list (default: openai-codex)",
    )

    profiles_status_parser = profiles_subparsers.add_parser("status", help="Show provider status")
    profiles_status_parser.add_argument(
        "--provider",
        type=str,
        default="openai-codex",
        help="Provider id to inspect (default: openai-codex)",
    )

    profiles_usage_parser = profiles_subparsers.add_parser(
        "usage",
        help="Query Codex account usage for one or more profiles",
    )
    profiles_usage_parser.add_argument(
        "--provider",
        type=str,
        default="openai-codex",
        help="Provider id to inspect (default: openai-codex)",
    )
    profiles_usage_parser.add_argument(
        "--profile-id",
        type=str,
        default=None,
        help="Specific profile id to query",
    )
    profiles_usage_parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Query usage for all stored profiles instead of only the selected/first profile",
    )

    profiles_enable_parser = profiles_subparsers.add_parser("enable", help="Enable a profile")
    profiles_enable_parser.add_argument("profile_id", type=str, help="Profile id to enable")

    profiles_disable_parser = profiles_subparsers.add_parser("disable", help="Disable a profile")
    profiles_disable_parser.add_argument("profile_id", type=str, help="Profile id to disable")

    profiles_order_parser = profiles_subparsers.add_parser(
        "order",
        help="Set provider profile preference order",
    )
    profiles_order_parser.add_argument(
        "--provider",
        type=str,
        default="openai-codex",
        help="Provider id to reorder (default: openai-codex)",
    )
    profiles_order_parser.add_argument(
        "--ids",
        type=str,
        required=True,
        help="Comma-separated profile ids in desired order",
    )

    # =========================================================================
    # benchmarks command
    # =========================================================================
    benchmarks_parser = subparsers.add_parser(
        "benchmarks",
        help="Install or build reusable algorithm benchmark artifacts",
    )
    benchmarks_subparsers = benchmarks_parser.add_subparsers(
        dest="benchmarks_command",
        help="Benchmark artifact operations",
    )
    benchmarks_subparsers.add_parser(
        "manifest",
        help="Print the bundled benchmark artifact manifest",
    )

    benchmarks_install_parser = benchmarks_subparsers.add_parser(
        "install",
        help="Install benchmark datasets, baseline metrics, and saved artifacts",
    )
    benchmarks_install_parser.add_argument("--manifest", type=str, default=None)
    benchmarks_install_parser.add_argument("--asset", type=str, default=None, help="Local archive path")
    benchmarks_install_parser.add_argument("--url", type=str, default=None, help="Release asset URL")
    benchmarks_install_parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Install root. Defaults to ~/.cellcompass",
    )
    benchmarks_install_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that already exist locally",
    )
    benchmarks_install_parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Install only one dataset id. Repeat for multiple datasets.",
    )
    benchmarks_install_parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip top-level builtin_runs entries if present in the archive.",
    )
    benchmarks_install_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip sha256 verification.",
    )

    benchmarks_build_parser = benchmarks_subparsers.add_parser(
        "build-bundle",
        help="Build a local benchmark bundle from ~/.cellcompass/algorithm_benchmarks",
    )
    benchmarks_build_parser.add_argument(
        "--source-root",
        type=str,
        default=None,
        help="Benchmark root to bundle. Defaults to ~/.cellcompass/algorithm_benchmarks",
    )
    benchmarks_build_parser.add_argument("--output", type=str, required=True)
    benchmarks_build_parser.add_argument(
        "--include-builtin-runs",
        action="store_true",
        help="Include top-level builtin_runs history. Defaults to false.",
    )

    # =========================================================================
    # inspect command
    # =========================================================================
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect an h5ad file and show data summary",
    )
    inspect_parser.add_argument(
        "input",
        type=str,
        help="Path to h5ad file",
    )
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    
    # =========================================================================
    # run command
    # =========================================================================
    run_parser = subparsers.add_parser(
        "run",
        help="Run a dataset workflow through the unified Runtime Core",
    )
    
    # Input/output
    run_parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Optional path to input data (h5ad or raw). Can also be provided later in chat.",
    )
    run_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: <input_dir>/cytobridge_output)",
    )
    
    # Scientific question
    run_parser.add_argument(
        "-q", "--question",
        type=str,
        default="",
        help="Your scientific question in natural language",
    )
    
    # Data columns
    run_parser.add_argument(
        "-t", "--time-key",
        type=str,
        default=None,
        help="Column name for time points (auto-detected if not specified)",
    )
    run_parser.add_argument(
        "-l", "--label-key",
        type=str,
        default=None,
        help="Column name for cell type labels (auto-detected if not specified)",
    )
    
    # Analyses
    run_parser.add_argument(
        "-a", "--analyses",
        type=str,
        nargs="+",
        choices=[
            "trajectory_fate",
            "drivers_genes",
            "growth_mass",
            "stochasticity_score",
            "interaction",
            "counterfactual",
            "limitations",
        ],
        default=["trajectory_fate", "drivers_genes", "growth_mass"],
        help="Downstream analyses to run (default: trajectory_fate drivers_genes growth_mass)",
    )
    
    # Gene sets for enrichment
    run_parser.add_argument(
        "--gene-sets-gmt",
        type=str,
        default=None,
        help="Path to GMT file for gene set enrichment analysis",
    )
    
    # Training parameters
    run_parser.add_argument(
        "-d", "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for training (default: cuda)",
    )
    run_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max pilot training retries on failure (default: 3)",
    )
    run_parser.add_argument(
        "--pilot-epochs",
        type=int,
        default=100,
        help="Max epochs per stage in pilot training (default: 100)",
    )
    run_parser.add_argument(
        "--pilot-max-cells",
        type=int,
        default=500,
        help="Max cells per time point in pilot (default: 500)",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    run_parser.add_argument(
        "--disable-llm-overrides",
        action="store_true",
        help="禁用 LLM 配置覆写，直接使用 YAML 文件中的默认配置（便于调试）",
    )
    
    # Multimodal toggle
    run_parser.add_argument(
        "--disable-multimodal",
        action="store_true",
        help="Disable all visual model inputs, including uploaded images and read_file image/PDF rendering.",
    )
    # Report format
    run_parser.add_argument(
        "--report-format",
        type=str,
        default="html",
        choices=["md", "html"],
        help="Report output format (default: html)",
    )
    
    # =========================================================================
    # interactive command
    # =========================================================================
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Run the CytoBridge agent in interactive mode"
    )
    
    # Input argument
    interactive_parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Optional path to input data. Can also be provided later in chat.",
    )
    
    # Optional initial question
    interactive_parser.add_argument(
        "-q", "--question",
        type=str,
        default="",
        help="Initial scientific question",
    )
    
    # Output directory
    interactive_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory",
    )

    # Device configuration
    interactive_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training (default: cuda)",
    )
    interactive_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="SESSION_ID",
        help="Resume a saved interactive session from ~/.cellcompass/conversations/<SESSION_ID>.json",
    )
    interactive_parser.add_argument(
        "--last",
        action="store_true",
        help="Resume the most recent saved session instead of starting a new one.",
    )

    # =========================================================================
    # tui command
    # =========================================================================
    tui_parser = subparsers.add_parser(
        "tui",
        help="Run the Rich terminal UI. Falls back to plain terminal output when Rich is unavailable.",
    )
    tui_parser.add_argument("input", type=str, nargs="?", default=None, help="Optional path to input data")
    tui_parser.add_argument("-q", "--question", type=str, default="", help="Initial scientific question")
    tui_parser.add_argument("-o", "--output", type=str, default=None, help="Output directory")
    tui_parser.add_argument("--device", type=str, default="cuda", help="Device for training (default: cuda)")
    tui_parser.add_argument("--resume", type=str, default=None, metavar="SESSION_ID", help="Resume a saved session")
    tui_parser.add_argument("--last", action="store_true", help="Resume the most recent saved session")
    tui_parser.add_argument("--grep", type=str, default=None, help="Resume the latest session matching text with --last")
    tui_parser.add_argument("--cwd", type=str, default=None, help="Resume the latest session matching cwd/output path with --last")
    tui_parser.add_argument("--limit", type=int, default=20, help="History search limit for --grep/--cwd")

    # =========================================================================
    # exec command
    # =========================================================================
    exec_parser = subparsers.add_parser(
        "exec",
        help="Run one non-interactive agent turn and exit",
    )
    exec_parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt to run. Quote multi-word prompts or pass words normally.",
    )
    exec_parser.add_argument("--input", type=str, default=None, help="Optional input data path")
    exec_parser.add_argument("-o", "--output", type=str, default=None, help="Output directory")
    exec_parser.add_argument("--resume", type=str, default=None, metavar="SESSION_ID", help="Resume a saved session")
    exec_parser.add_argument("--last", action="store_true", help="Resume the most recent saved session")
    exec_parser.add_argument("--json", action="store_true", help="Emit JSONL events")
    exec_parser.add_argument("--device", type=str, default="cuda", help="Device for training (default: cuda)")
    exec_parser.add_argument(
        "--disable-multimodal",
        action="store_true",
        help="Disable multimodal input support for the new session.",
    )

    # =========================================================================
    # resume command
    # =========================================================================
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume a saved session in interactive mode. Opens a picker when SESSION_ID is omitted.",
    )
    resume_parser.add_argument("session_id", nargs="?", default=None, help="Session id to resume")
    resume_parser.add_argument("--last", action="store_true", help="Resume the most recent saved session")
    resume_parser.add_argument("--limit", type=int, default=20, help="Number of sessions to show in the picker")
    resume_parser.add_argument("--grep", type=str, default=None, help="Filter picker/latest selection by text")
    resume_parser.add_argument("--cwd", type=str, default=None, help="Filter picker/latest selection by cwd/output path")

    # =========================================================================
    # continue command
    # =========================================================================
    continue_parser = subparsers.add_parser(
        "continue",
        aliases=["c"],
        help="Resume the most recent saved session in interactive mode",
    )
    continue_parser.set_defaults(command="continue")
    continue_parser.add_argument("--limit", type=int, default=20, help="Fallback picker size if no latest session exists")
    continue_parser.add_argument("--grep", type=str, default=None, help="Resume the latest session matching text")
    continue_parser.add_argument("--cwd", type=str, default=None, help="Resume the latest session matching cwd/output path")

    # =========================================================================
    # session command
    # =========================================================================
    session_parser = subparsers.add_parser(
        "session",
        help="Inspect or manage saved sessions",
    )
    session_subparsers = session_parser.add_subparsers(dest="session_command", help="Session commands")
    session_list = session_subparsers.add_parser("list", help="List saved sessions")
    session_list.add_argument("--limit", type=int, default=20)
    session_list.add_argument("--plain", action="store_true", help="Print a compact non-numbered list")
    session_list.add_argument("--json", action="store_true", help="Print raw session summaries as JSON")
    session_list.add_argument("--grep", type=str, default=None, help="Filter sessions by text")
    session_list.add_argument("--cwd", type=str, default=None, help="Filter sessions by cwd/output path")
    session_status = session_subparsers.add_parser("status", help="Show session status")
    session_status.add_argument("session_id", type=str)
    session_show = session_subparsers.add_parser("show", help="Show a readable session summary and recent events")
    session_show.add_argument("session_id", type=str)
    session_show.add_argument("--last", type=int, default=8, help="Number of recent events to include")
    session_tail = session_subparsers.add_parser("tail", help="Show recent timeline events")
    session_tail.add_argument("session_id", type=str)
    session_tail.add_argument("--last", type=int, default=50)
    session_compact = session_subparsers.add_parser("compact", help="Compact a saved session")
    session_compact.add_argument("session_id", type=str)
    session_compact.add_argument("--keep-last-turns", type=int, default=6)
    session_delete = session_subparsers.add_parser("delete", help="Delete a saved session")
    session_delete.add_argument("session_id", type=str)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check local runtime health for CLI/TUI/Web operation",
    )
    doctor_parser.add_argument("--json", action="store_true", help="Emit machine-readable diagnostics")

    perf_parser = subparsers.add_parser(
        "perf",
        help="Inspect runtime performance from saved session events",
    )
    perf_subparsers = perf_parser.add_subparsers(dest="perf_command", help="Performance commands")
    perf_report = perf_subparsers.add_parser("report", help="Summarize event counts and recorded durations")
    perf_report.add_argument("--session", required=True, help="Session id to inspect")
    perf_report.add_argument("--json", action="store_true", help="Emit machine-readable report")

    jobs_parser = subparsers.add_parser(
        "jobs",
        help="Inspect runtime job registry",
    )
    jobs_parser.add_argument("--session", type=str, default=None, help="Filter jobs by session id")
    jobs_parser.add_argument("--active", action="store_true", help="Only show active/running jobs")
    jobs_parser.add_argument("--stale", action="store_true", help="Only show stale jobs with dead PIDs")
    jobs_parser.add_argument("--json", action="store_true", help="Emit machine-readable job records")

    # =========================================================================
    # web command
    # =========================================================================
    web_parser = subparsers.add_parser(
        "web",
        help="Start the CytoBridge agent Web UI"
    )
    web_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    # Pre-configuration args
    web_parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to h5ad file (pre-fill)"
    )
    web_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output directory (pre-fill)"
    )
    web_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (pre-fill)"
    )
    # --- Common LLM arguments ---
    llm_group = parser.add_argument_group("LLM Configuration")
    llm_group.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="Base URL for LLM API (default: provider-specific config, then ~/.cellcompass/config.json when applicable)",
    )
    llm_group.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="API key for LLM (default: provider-specific config, then ~/.cellcompass/config.json or OPENAI_API_KEY when applicable)",
    )
    llm_group.add_argument(
        "--llm-model",
        type=str,
        default=saved_config.get("llm_model", "gpt-4o"),
        help="LLM model name (default: from ~/.cellcompass/config.json or gpt-4o)",
    )
    llm_group.add_argument(
        "--llm-provider",
        type=str,
        default=saved_config.get("llm_provider", "auto"),
        help="LLM provider id, e.g. auto, openai, openrouter, xiaomi, deepseek, zai, kimi, codex_oauth",
    )
    llm_group.add_argument(
        "--llm-auth-mode",
        type=str,
        default=saved_config.get("llm_auth_mode", "auto"),
        choices=["auto", "api_key", "gemini_oauth", "codex_oauth"],
        help="LLM auth mode selection (default: from config or auto)",
    )
    llm_group.add_argument(
        "--llm-profile-id",
        type=str,
        default=saved_config.get("llm_profile_id"),
        help="Preferred OAuth profile id for supported providers",
    )
    llm_group.add_argument(
        "--llm-thinking-level",
        type=str,
        default=saved_config.get("llm_thinking_level", "low"),
        choices=["off", "minimal", "low", "medium", "high", "xhigh"],
        help="Model reasoning/thinking effort when supported (default: from config or low)",
    )
    
    # Checkpoint/Resume mode (for HPC environments)
    checkpoint_group = run_parser.add_argument_group("Checkpoint Mode (for HPC)")
    checkpoint_group.add_argument(
        "--checkpoint",
        action="store_true",
        help="Enable checkpoint mode: pause before training and save state for manual execution on compute node",
    )
    checkpoint_group.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="LEGACY_CHECKPOINT_DIR",
        help="Legacy training-checkpoint resume path. Interactive session resume now uses: interactive --resume <SESSION_ID>.",
    )
    
    # Verbosity
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    run_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    
    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    """Execute the inspect command."""
    import scanpy as sc
    from .tools.data_tools import inspect_adata
    
    input_path = None
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            return 1
    
    logger.info(f"Loading {input_path}...")
    adata = sc.read_h5ad(input_path)
    
    summary = inspect_adata(adata)
    
    if args.json:
        print(json.dumps(summary.model_dump(), indent=2))
    else:
        print("\n" + "=" * 60)
        print("CytoBridge Agent - Data Inspection")
        print("=" * 60)
        print(f"\nFile: {input_path}")
        print(f"Cells: {summary.n_obs:,}")
        print(f"Genes: {summary.n_vars:,}")
        
        if summary.sparsity is not None:
            print(f"Sparsity: {summary.sparsity:.1%}")
        if summary.var_is_log_norm is not None:
            print(f"Log-normalized: {'Yes' if summary.var_is_log_norm else 'No'}")
        
        print(f"\nLatent space: {'X_latent' if summary.has_latent else 'Not computed'}")
        print(f"PCA: {'Available' if summary.has_pca else 'Not computed'}")
        print(f"UMAP: {'Available' if summary.has_umap else 'Not computed'}")
        
        if summary.time_candidates:
            print("\nTime Point Candidates:")
            for tc in summary.time_candidates[:5]:
                print(f"  - {tc.key}: {len(tc.levels)} levels, mass CV={tc.mass_variance:.2f}")
        
        if summary.label_candidates:
            print(f"\nLabel Candidates: {', '.join(summary.label_candidates)}")
        
        print("\nObservation columns:")
        for col in summary.obs_keys[:10]:
            print(f"  - {col}")
        if len(summary.obs_keys) > 10:
            print(f"  ... and {len(summary.obs_keys) - 10} more")
        
        print()
    
    return 0


def _normalize_provider_secret_map(raw: Any) -> Dict[str, str]:
    data = raw if isinstance(raw, dict) else {}
    out: Dict[str, str] = {}
    for key, value in data.items():
        provider = _normalize_llm_provider(key)
        text = str(value or "").strip()
        if provider and text:
            out[provider] = text
    return out


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _env_text_or_file(name: str, default: str) -> str:
    raw_file = os.environ.get(f"{name}_FILE")
    if raw_file and raw_file.strip():
        try:
            return Path(raw_file.strip()).read_text(encoding="utf-8").strip()
        except OSError:
            logger.warning("Could not read %s_FILE=%s; falling back to %s/default", name, raw_file, name)
    raw = os.environ.get(name)
    if raw is not None and raw.strip():
        return raw.strip()
    return str(default)


def _resolve_cli_llm_runtime_args(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    """Resolve CLI LLM args without leaking legacy OpenAI-compatible config across providers."""
    saved_config = get_saved_config()
    from cytobridge_agent.utils.llm_providers import get_llm_provider, resolve_provider_api_key, resolve_provider_base_url

    provider = _normalize_llm_provider(getattr(args, "llm_provider", "auto"))
    saved_provider = _normalize_llm_provider(saved_config.get("llm_provider") or "auto")

    explicit_api_key = str(getattr(args, "llm_api_key", "") or "").strip() or None
    provider_api_keys = _normalize_provider_secret_map(saved_config.get("provider_api_keys"))
    provider_api_key = provider_api_keys.get(provider)
    provider_env_api_key = (
        resolve_provider_api_key(get_llm_provider(provider), None)
        if provider != "auto"
        else None
    )
    saved_api_key = saved_config.get("openai_api_key") if provider in {"auto", saved_provider} else None
    api_key = (
        explicit_api_key
        or provider_env_api_key
        or provider_api_key
        or (str(saved_api_key or "").strip() or None)
        or (os.environ.get("OPENAI_API_KEY") if provider == "auto" else None)
    )

    explicit_base_url = str(getattr(args, "llm_base_url", "") or "").strip() or None
    provider_base_urls = _normalize_provider_secret_map(saved_config.get("provider_base_urls"))
    provider_base_url = provider_base_urls.get(provider)
    saved_base_url = saved_config.get("llm_base_url") if provider in {"auto", saved_provider} else None
    base_url = explicit_base_url or provider_base_url or (str(saved_base_url or "").strip() or None)
    if provider != "auto":
        base_url = resolve_provider_base_url(get_llm_provider(provider), base_url)

    return {
        "model": getattr(args, "llm_model", None),
        "base_url": base_url,
        "api_key": api_key,
        "auth_mode": getattr(args, "llm_auth_mode", "auto"),
        "provider": provider,
        "profile_id": getattr(args, "llm_profile_id", None),
        "thinking_level": getattr(args, "llm_thinking_level", None),
    }


def _controller_from_args(args: argparse.Namespace):
    """Build a process-local session controller from CLI LLM arguments."""
    from .session_controller import SessionController
    from cytobridge_agent.runtime_v2.state import (
        DEFAULT_STOP_HOOK_ENABLED,
        DEFAULT_STOP_HOOK_MAX_TRIGGERS,
        DEFAULT_STOP_HOOK_MODE,
        DEFAULT_STOP_HOOK_PROMPT,
    )

    runtime = _resolve_cli_llm_runtime_args(args)
    saved_config = get_saved_config()
    initial_config = {
        "llm_model": runtime["model"],
        "llm_base_url": runtime["base_url"],
        "llm_api_key": runtime["api_key"],
        "llm_auth_mode": _normalize_auth_mode(runtime["auth_mode"]),
        "llm_provider": runtime["provider"] or "auto",
        "llm_profile_id": runtime["profile_id"],
        "llm_thinking_level": _normalize_llm_reasoning_effort(runtime["thinking_level"], default="low"),
        "stop_hook_enabled": _env_bool(
            "CYTOBRIDGE_STOP_HOOK_ENABLED",
            bool(saved_config.get("stop_hook_enabled", DEFAULT_STOP_HOOK_ENABLED)),
        ),
        "stop_hook_mode": _env_text_or_file(
            "CYTOBRIDGE_STOP_HOOK_MODE",
            str(saved_config.get("stop_hook_mode") or DEFAULT_STOP_HOOK_MODE),
        ),
        "stop_hook_prompt": _env_text_or_file(
            "CYTOBRIDGE_STOP_HOOK_PROMPT",
            str(saved_config.get("stop_hook_prompt") or DEFAULT_STOP_HOOK_PROMPT),
        ),
        "stop_hook_max_triggers": _env_int(
            "CYTOBRIDGE_STOP_HOOK_MAX_TRIGGERS",
            int(saved_config.get("stop_hook_max_triggers", DEFAULT_STOP_HOOK_MAX_TRIGGERS)),
        ),
    }

    def llm_factory(config: Optional[Dict[str, Any]] = None):
        from cytobridge_agent.utils.llm_factory import instantiate_llm

        cfg = dict(initial_config)
        cfg.update(dict(config or {}))
        return instantiate_llm(
            model=cfg.get("llm_model") or cfg.get("model") or "gpt-4o",
            base_url=cfg.get("llm_base_url") or cfg.get("base_url"),
            api_key=cfg.get("llm_api_key") or cfg.get("api_key"),
            auth_mode=_normalize_auth_mode(cfg.get("llm_auth_mode") or cfg.get("auth_mode") or "auto"),
            provider=_normalize_llm_provider(cfg.get("llm_provider") or cfg.get("provider") or "auto"),
            preferred_profile_id=cfg.get("llm_profile_id") or cfg.get("profile_id"),
            thinking_level=_normalize_llm_reasoning_effort(
                cfg.get("llm_thinking_level") or cfg.get("thinking_level"),
                default="low",
            ),
        )[0]

    return SessionController(llm_factory, initial_config=initial_config)


def _print_repl_banner(controller, *, resumed: bool = False) -> None:
    session = controller.active_session
    print("=" * 60)
    print("CellCompass / CytoBridge Agent")
    print("=" * 60)
    if session is not None:
        print(f"Session: {session.session_id}")
        print(f"Data: {session.input_path or '(not set)'}")
        print(f"Output: {session.output_dir or '(not set)'}")
        if resumed:
            print("Resumed existing session.")
    print("Type /help for commands, /quit to exit.")


def _run_repl(controller) -> None:
    from .cli_slash import QUIT_COMMANDS, handle_cli_slash_command, is_slash_command
    from .cli_runtime import SessionOpenSpec, open_or_create_session
    from .cli_terminal import enable_readline_history

    enable_readline_history()
    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
            if user_input.lower() in QUIT_COMMANDS:
                print("Session ended.")
                return
            if is_slash_command(user_input):
                try:
                    output = handle_cli_slash_command(controller, user_input)
                except EOFError:
                    print("Session ended.")
                    return
                if output:
                    print(output)
                continue
            if controller.active_session is None:
                open_or_create_session(
                    controller,
                    SessionOpenSpec(question=user_input),
                )
            result = controller.run_turn(user_input)
            print(f"\nAgent > {result.response}")
        except KeyboardInterrupt:
            print("\nInterrupted. Session checkpoint is preserved if a turn completed or failed cleanly.")
            return
        except EOFError:
            print("\nSession ended.")
            return


def cmd_interactive(args: argparse.Namespace) -> int:
    from .cli_runtime import SessionOpenSpec, open_or_create_session

    controller = _controller_from_args(args)
    try:
        if getattr(args, "last", False) and not controller.latest_session_id():
            logger.error("No saved session found for --last.")
            return 1
        _, resumed = open_or_create_session(
            controller,
            SessionOpenSpec(
                input_path=getattr(args, "input", None),
                question=getattr(args, "question", "") or "",
                output_dir=getattr(args, "output", None),
                device=getattr(args, "device", "cuda"),
                session_id=getattr(args, "resume", None),
                last=bool(getattr(args, "last", False)),
            ),
        )
        _print_repl_banner(controller, resumed=resumed)
        initial_question = str(getattr(args, "question", "") or "").strip()
        if initial_question and not resumed:
            result = controller.run_turn(initial_question)
            print(f"\nAgent > {result.response}")
        _run_repl(controller)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.exception("Interactive session failed: %s", exc)
        return 1


def cmd_tui(args: argparse.Namespace) -> int:
    from .cli_runtime import SessionOpenSpec
    from .cli_tui import run_tui

    if getattr(args, "resume", None) and getattr(args, "input", None):
        logger.error("--resume cannot be combined with an input path in tui mode.")
        return 1
    if getattr(args, "last", False) and getattr(args, "input", None):
        logger.error("--last cannot be combined with an input path in tui mode.")
        return 1
    controller = _controller_from_args(args)
    return run_tui(
        controller,
        SessionOpenSpec(
            input_path=getattr(args, "input", None),
            question=getattr(args, "question", "") or "",
            output_dir=getattr(args, "output", None),
            device=getattr(args, "device", "cuda"),
            session_id=getattr(args, "resume", None),
            last=bool(getattr(args, "last", False)),
            require_resume=bool(getattr(args, "last", False)),
            picker=False,
            picker_limit=max(1, int(getattr(args, "limit", 20) or 20)),
            query=getattr(args, "grep", None),
            cwd=getattr(args, "cwd", None),
        ),
    )


def cmd_exec(args: argparse.Namespace) -> int:
    from .cli_runtime import SessionOpenSpec, emit_session_open_event, open_or_create_session
    from .session_controller import print_jsonl_event

    prompt = " ".join(getattr(args, "prompt", []) or []).strip()
    if not prompt:
        logger.error("exec requires a prompt.")
        return 1
    controller = _controller_from_args(args)
    try:
        if getattr(args, "last", False) and not controller.latest_session_id():
            logger.error("No saved session found for --last.")
            return 1
        session, resumed = open_or_create_session(
            controller,
            SessionOpenSpec(
                input_path=getattr(args, "input", None),
                question=prompt,
                output_dir=getattr(args, "output", None),
                device=getattr(args, "device", "cuda"),
                session_id=getattr(args, "resume", None),
                last=bool(getattr(args, "last", False)),
                enable_multimodal=not getattr(args, "disable_multimodal", False),
            ),
        )
        emit_session_open_event(json_enabled=bool(args.json), session=session, resumed=resumed)
        result = controller.run_turn(prompt)
        if args.json:
            print_jsonl_event(
                "turn_complete",
                {"session_id": result.session_id, "response": result.response},
            )
        else:
            print(result.response)
            print(f"\nSession: {result.session_id}")
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if args.json:
            print_jsonl_event("error", {"message": str(exc)})
        logger.exception("exec failed: %s", exc)
        return 1


def cmd_resume(args: argparse.Namespace) -> int:
    from .cli_runtime import SessionOpenSpec, resolve_resume_session_id

    controller = _controller_from_args(args)
    try:
        session_id = resolve_resume_session_id(
            controller,
            SessionOpenSpec(
                session_id=getattr(args, "session_id", None),
                last=bool(getattr(args, "last", False)),
                picker=True,
                picker_limit=max(1, int(getattr(args, "limit", 20) or 20)),
                query=getattr(args, "grep", None),
                cwd=getattr(args, "cwd", None),
            ),
        )
        if not session_id:
            print("Resume cancelled.")
            return 0
        controller.resume_session(session_id)
        _print_repl_banner(controller, resumed=True)
        _run_repl(controller)
        return 0
    except RuntimeError as exc:
        logger.error("%s Use: cellcompass resume SESSION_ID or cellcompass resume --last", exc)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.exception("Failed to resume session: %s", exc)
        return 1


def cmd_session(args: argparse.Namespace) -> int:
    from .cli_terminal import filter_sessions, format_session_detail, format_session_listing

    controller = _controller_from_args(args)
    command = getattr(args, "session_command", None) or "list"
    if command == "list":
        list_limit = max(1, int(args.limit or 20))
        search_limit = max(list_limit, 200) if getattr(args, "grep", None) or getattr(args, "cwd", None) else list_limit
        sessions = filter_sessions(
            controller.list_sessions(limit=search_limit),
            query=getattr(args, "grep", None),
            cwd=getattr(args, "cwd", None),
        )[:list_limit]
        if getattr(args, "json", False):
            print(json.dumps(sessions, indent=2, ensure_ascii=False))
            return 0
        print(
            format_session_listing(
                sessions,
                numbered=not getattr(args, "plain", False),
            )
        )
        return 0
    if command == "status":
        print(json.dumps(controller.session_status(args.session_id), indent=2, ensure_ascii=False))
        return 0
    if command == "show":
        print(
            format_session_detail(
                controller.session_status(args.session_id),
                controller.tail_events(args.session_id, limit=max(0, int(args.last or 0))),
            )
        )
        return 0
    if command == "tail":
        events = controller.tail_events(args.session_id, limit=args.last)
        for event in events:
            print(json.dumps(event, ensure_ascii=False))
        return 0
    if command == "compact":
        print(json.dumps(controller.compact_session(args.session_id, keep_last_turns=args.keep_last_turns), indent=2, ensure_ascii=False))
        return 0
    if command == "delete":
        print(json.dumps(controller.delete_session(args.session_id), indent=2, ensure_ascii=False))
        return 0
    logger.error("Unknown session command.")
    return 1


def cmd_doctor(args: argparse.Namespace) -> int:
    from .runtime_diagnostics import format_doctor_report, run_doctor

    report = run_doctor()
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_doctor_report(report))
    return 0 if report.get("status") == "ok" else 2


def cmd_perf(args: argparse.Namespace) -> int:
    from .runtime_diagnostics import format_perf_report, session_perf_report

    command = getattr(args, "perf_command", None) or "report"
    if command != "report":
        logger.error("Unknown perf command.")
        return 1
    report = session_perf_report(args.session)
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_perf_report(report))
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    from .job_registry import JobRegistry

    registry = JobRegistry()
    jobs = registry.stale_jobs() if getattr(args, "stale", False) else registry.list(
        session_id=getattr(args, "session", None),
        active_only=bool(getattr(args, "active", False)),
    )
    if getattr(args, "json", False):
        print(json.dumps(jobs, indent=2, ensure_ascii=False))
        return 0
    if not jobs:
        print("No runtime jobs found.")
        return 0
    for job in jobs:
        print(
            f"{job.get('job_id')}  {job.get('kind')}  {job.get('status')}  "
            f"session={job.get('session_id') or '-'} pid={job.get('pid') or '-'} alive={job.get('pid_alive')}"
        )
    return 0


def cmd_auth(args: argparse.Namespace) -> int:
    from cytobridge_agent.utils.codex_sidecar_client import codex_sidecar_call

    if args.auth_command == "codex-login":
        result = codex_sidecar_call("auth.codex.login", {"profile_id": args.profile_id})
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.auth_command == "profiles":
        if args.auth_profiles_command == "list":
            result = codex_sidecar_call("auth.profiles.list", {"provider": args.provider})
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.auth_profiles_command == "status":
            result = codex_sidecar_call("auth.runtime.status", {"provider": args.provider})
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.auth_profiles_command == "usage":
            result = codex_sidecar_call(
                "auth.runtime.usage",
                {
                    "provider": args.provider,
                    "profile_id": args.profile_id,
                    "all_profiles": bool(args.all_profiles),
                    "timeout_ms": 5000,
                },
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.auth_profiles_command == "enable":
            result = codex_sidecar_call("auth.profiles.enable", {"profile_id": args.profile_id})
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.auth_profiles_command == "disable":
            result = codex_sidecar_call("auth.profiles.disable", {"profile_id": args.profile_id})
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.auth_profiles_command == "order":
            ids = [item.strip() for item in args.ids.split(",") if item.strip()]
            result = codex_sidecar_call(
                "auth.profiles.order.set",
                {"provider": args.provider, "ids": ids},
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0

    logger.error("Unknown auth command.")
    return 1


def cmd_benchmarks(args: argparse.Namespace) -> int:
    from cytobridge_agent.benchmark_artifacts import (
        DEFAULT_BENCHMARK_ROOT,
        build_algorithm_benchmark_bundle,
        install_algorithm_benchmarks,
        load_manifest,
    )

    if args.benchmarks_command == "manifest":
        print(json.dumps(load_manifest(), indent=2, ensure_ascii=False))
        return 0
    if args.benchmarks_command == "install":
        result = install_algorithm_benchmarks(
            manifest_path=args.manifest,
            output_root=args.output_root,
            asset_path=args.asset,
            url=args.url,
            force=args.force,
            datasets=args.dataset,
            include_baselines=not args.no_baselines,
            verify_sha256=not args.no_verify,
        )
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
        return 0
    if args.benchmarks_command == "build-bundle":
        result = build_algorithm_benchmark_bundle(
            source_root=args.source_root or DEFAULT_BENCHMARK_ROOT,
            output_path=args.output,
            include_builtin_runs=args.include_builtin_runs,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    logger.error("Unknown benchmarks command.")
    return 1


def _runtime_run_prompt(args: argparse.Namespace) -> str:
    parts = [
        "Run a complete CytoBridge/CellCompass scientific workflow using the unified Runtime Core.",
        "Inspect the input data, choose appropriate preprocessing and dynamics modeling steps, train/evaluate models, run downstream biological analysis, and write the requested report.",
    ]
    if getattr(args, "question", None):
        parts.append(f"Scientific question: {args.question}")
    if getattr(args, "time_key", None):
        parts.append(f"Use this time column when appropriate: {args.time_key}")
    if getattr(args, "label_key", None):
        parts.append(f"Use this label column when appropriate: {args.label_key}")
    analyses = getattr(args, "analyses", None) or []
    if analyses:
        parts.append(f"Requested downstream analyses: {', '.join(analyses)}")
    if getattr(args, "gene_sets_gmt", None):
        parts.append(f"Gene-set GMT resource: {args.gene_sets_gmt}")
    parts.append(f"Preferred device: {getattr(args, 'device', 'cuda')}")
    parts.append(f"Report format: {getattr(args, 'report_format', 'html')}")
    if getattr(args, "disable_llm_overrides", False):
        parts.append("Do not override model/training configs unless needed for correctness.")
    return "\n".join(parts)


def cmd_run(args: argparse.Namespace) -> int:
    """Run a dataset workflow through the unified Runtime Core.

    ``run`` is no longer a separate graph pipeline. It is a structured one-shot
    prompt into the same session runtime used by exec/interactive/TUI/Web.
    """
    from .cli_runtime import SessionOpenSpec, emit_session_open_event, open_or_create_session

    input_path = None
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            return 1
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    if getattr(args, "checkpoint", False) or getattr(args, "resume", None):
        logger.error(
            "Legacy pipeline checkpoint/resume has been removed. Use session-level runtime resume: "
            "cellcompass resume <SESSION_ID> or cellcompass exec --resume <SESSION_ID> ..."
        )
        return 1

    logger.info("Starting CytoBridge Agent through unified Runtime Core...")
    logger.info(f"Input: {input_path or '(not provided)'}")
    logger.info(f"Question: {args.question or '(none)'}")
    logger.info(f"Analyses: {args.analyses}")
    logger.info(f"Device: {args.device}")

    try:
        controller = _controller_from_args(args)
        prompt = _runtime_run_prompt(args)
        session, resumed = open_or_create_session(
            controller,
            SessionOpenSpec(
                input_path=str(input_path) if input_path else None,
                question=prompt,
                output_dir=args.output,
                device=args.device,
                report_format=args.report_format,
                enable_multimodal=not args.disable_multimodal,
            ),
        )
        emit_session_open_event(json_enabled=False, session=session, resumed=resumed)
        result = controller.run_turn(prompt)
        print(result.response)
        print(f"\nSession: {result.session_id}")
        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as e:
        logger.exception(f"Runtime run failed: {e}")
        return 1


def cmd_web(args: argparse.Namespace) -> int:
    """Execute the web command."""
    from .web_server import start_server
    
    print(f"Starting Web UI at http://{args.host}:{args.port}...")
    # Add open browser command
    print(f"Please open http://localhost:{args.port} in your browser")
    
    try:
        start_server(
            host=args.host, 
            port=args.port,
            initial_config={
                "input_path": args.input,
                "output_path": args.output,
                "device": args.device,
                "llm_base_url": args.llm_base_url,
                "openai_api_key": args.llm_api_key,
                "llm_model": args.llm_model,
                "llm_provider": _normalize_llm_provider(getattr(args, "llm_provider", "auto")),
                "llm_auth_mode": _normalize_auth_mode(args.llm_auth_mode),
                "llm_profile_id": args.llm_profile_id,
                "llm_thinking_level": _normalize_llm_reasoning_effort(
                    getattr(args, "llm_thinking_level", None),
                    default="low",
                ),
            }
        )
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        logger.error(f"Web server failed: {e}")
        return 1


def _preprocess_argv(argv: Optional[List[str]] = None) -> List[str]:
    """Map Claude/Codex-style defaults onto argparse subcommands."""
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        return ["interactive"]
    commands = {
        "auth",
        "benchmarks",
        "inspect",
        "run",
        "interactive",
        "tui",
        "exec",
        "resume",
        "continue",
        "c",
        "session",
        "doctor",
        "perf",
        "jobs",
        "web",
        "-h",
        "--help",
        "--version",
    }
    if raw[0] in commands:
        return raw
    global_value_options = {
        "--llm-base-url",
        "--llm-api-key",
        "--llm-model",
        "--llm-provider",
        "--llm-auth-mode",
        "--llm-profile-id",
        "--llm-thinking-level",
    }
    idx = 0
    while idx < len(raw):
        token = raw[idx]
        if token in commands:
            return raw
        if token.startswith("--"):
            if "=" not in token and token in global_value_options:
                idx += 2
            else:
                idx += 1
            continue
        if token.startswith("-"):
            idx += 1
            continue
        return [*raw[:idx], "exec", *raw[idx:]]
    return raw


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(_preprocess_argv(argv))
    
    if args.command is None:
        args.command = "interactive"
    
    if args.command == "inspect":
        return cmd_inspect(args)
    elif args.command == "auth":
        return cmd_auth(args)
    elif args.command == "benchmarks":
        return cmd_benchmarks(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "interactive":
        if args.resume and args.input:
            logger.error("--resume cannot be combined with an input path in interactive mode.")
            return 1
        if args.last and args.input:
            logger.error("--last cannot be combined with an input path in interactive mode.")
            return 1
        return cmd_interactive(args)
    elif args.command == "tui":
        return cmd_tui(args)
    elif args.command == "exec":
        return cmd_exec(args)
    elif args.command == "resume":
        return cmd_resume(args)
    elif args.command in {"continue", "c"}:
        args.last = True
        args.session_id = None
        return cmd_resume(args)
    elif args.command == "session":
        return cmd_session(args)
    elif args.command == "doctor":
        return cmd_doctor(args)
    elif args.command == "perf":
        return cmd_perf(args)
    elif args.command == "jobs":
        return cmd_jobs(args)
    elif args.command == "web":
        return cmd_web(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
