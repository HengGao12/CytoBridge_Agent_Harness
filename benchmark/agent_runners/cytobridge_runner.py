"""CytoBridge/CellCompass runner for the benchmark harness.

Uses the same session-open path as the modern ``cellcompass exec`` CLI while
keeping benchmark-only environment isolation and result collection in-process.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from benchmark.dynbench.format_recovery import recover_outputs_format_only
from benchmark.dynbench.intermediate_contract import materialize_intermediates_from_files
from benchmark.dynbench.scientific_harness import (
    apply_public_scientific_calibration,
    build_revision_prompt,
    write_public_audit,
)
from benchmark.cost_tracker import CostTracker

from .auth_utils import clear_benchmark_auth_lock
from .base import AgentRunner, RunResult

logger = logging.getLogger(__name__)

class CytoBridgeRunner(AgentRunner):
    """
    Benchmark runner that drives the CytoBridge agent.

    LLM configuration can be passed explicitly or falls back to
    ~/.cellcompass/config.json (same as `cytobridge-agent run`).
    """

    agent_id = "cytobridge"

    def __init__(
        self,
        llm_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_auth_mode: str = "auto",
        llm_profile_id: Optional[str] = None,
        llm_thinking_level: Optional[str] = None,
        recursion_limit: Optional[int] = None,
        harness_revisions: int = 1,
    ):
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_auth_mode = llm_auth_mode
        self.llm_profile_id = llm_profile_id
        self.llm_thinking_level = llm_thinking_level
        self.recursion_limit = recursion_limit
        self.harness_revisions = max(0, int(harness_revisions))

    def _resolve_llm_config(self) -> Dict[str, Any]:
        """
        Merge explicit args with ~/.cellcompass/config.json fallback.
        Mirrors the pattern used in `cytobridge_agent.cli`.
        """
        from benchmark.agent_config import load_saved_config

        saved = load_saved_config()
        provider_keys = saved.get("provider_api_keys")
        saved_api_key = saved.get("openai_api_key")
        saved_provider = str(self.llm_provider or saved.get("llm_provider") or "auto").strip()
        if not saved_api_key and isinstance(provider_keys, dict) and provider_keys:
            if saved_provider and provider_keys.get(saved_provider):
                saved_api_key = provider_keys[saved_provider]
            elif len(provider_keys) == 1:
                saved_api_key = next(iter(provider_keys.values()))
            elif "xiaomimimo" in str(saved.get("llm_base_url") or "").lower():
                saved_api_key = provider_keys.get("xiaomi")
        provider_base_urls = saved.get("provider_base_urls")
        saved_base_url = saved.get("llm_base_url")
        if not self.llm_base_url and isinstance(provider_base_urls, dict) and provider_base_urls.get(saved_provider):
            saved_base_url = provider_base_urls[saved_provider]
        return {
            "model": self.llm_model or saved.get("llm_model", "gpt-5.4"),
            "provider": saved_provider,
            "base_url": self.llm_base_url or saved_base_url,
            "api_key": self.llm_api_key or saved_api_key,
            "auth_mode": self.llm_auth_mode or saved.get("llm_auth_mode", "auto"),
            "preferred_profile_id": self.llm_profile_id or saved.get("llm_profile_id"),
            "thinking_level": self.llm_thinking_level or saved.get("llm_thinking_level", "low"),
        }

    def run(
        self,
        train_h5ad: str | Path,
        task_prompt: str,
        output_dir: str | Path,
        seed: int = 42,
        device: str = "cpu",
        time_key: str = "Time Point",
        timeout_sec: Optional[float] = None,
        cost_tracker: Optional[CostTracker] = None,
        **kwargs,
    ) -> RunResult:
        """
        Run the CytoBridge agent on a benchmark task.
        """
        train_h5ad = str(Path(train_h5ad).resolve())
        output_dir = str(Path(output_dir).resolve())
        workspace_dir = Path(kwargs.get("workspace_dir") or Path(output_dir)).resolve()
        blocked_read_roots = [str(Path(p).resolve()) for p in kwargs.get("blocked_read_roots", [])]

        # Do not silently cap benchmark agents below the runtime default.
        # A small graph recursion limit can terminate valid long-running file
        # delivery runs before the public verifier/fix loop completes.
        if self.recursion_limit is not None:
            os.environ["CYTOBRIDGE_RUNTIME_RECURSION_LIMIT"] = str(self.recursion_limit)
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cytobridge_numba_cache")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/cytobridge_mplconfig")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        # DynBench computes its own M1-M6 after the agent writes outputs. The
        # package-level trainer.evaluate() can be memory-heavy on S3 and is not
        # needed for benchmark scoring.
        os.environ.setdefault("CYTOBRIDGE_SKIP_TRAINER_EVALUATE", "1")
        os.environ.setdefault("CYTOBRIDGE_DISABLE_RAG_FOR_BENCHMARK", "1")
        os.environ["CYTOBRIDGE_BENCHMARK_RESTRICT_READS"] = "1"
        os.environ["CYTOBRIDGE_BENCHMARK_READ_ROOTS"] = os.pathsep.join([str(workspace_dir), output_dir])
        os.environ["CYTOBRIDGE_BENCHMARK_BLOCKED_ROOTS"] = os.pathsep.join(blocked_read_roots)
        expected_output_files = [str(x) for x in kwargs.get("expected_output_files", [])]

        # Seed injection removed — LLM natural variation is the source of variation

        # Build user goal dict
        user_goal = {
            "raw_question": task_prompt,
            "requested_analyses": [],  # Benchmark: agent decides
            "time_key": time_key,
            "device": device,
            "max_retries": 3,
            "report_format": "md",
            "benchmark_strict": True,
            "benchmark_no_literature": True,
            "benchmark_required_outputs": list(expected_output_files),
            "benchmark_export_strategy": "runner_intermediate_completion",
        }

        # Resolve LLM config
        llm_config = self._resolve_llm_config()

        logs = []
        t0 = time.time()
        clear_benchmark_auth_lock()
        final_state = None

        # ── 6.8 Start cost tracking ──
        if cost_tracker is not None:
            cost_tracker.start()

        try:
            from cytobridge_agent.cli_runtime import SessionOpenSpec, open_or_create_session
            from cytobridge_agent.session_controller import SessionController
            from cytobridge_agent.utils.llm_factory import instantiate_llm

            def llm_factory(config=None):
                cfg = dict(llm_config)
                cfg.update(dict(config or {}))
                llm, _ = instantiate_llm(
                    model=cfg.get("llm_model") or cfg.get("model"),
                    base_url=cfg.get("llm_base_url") or cfg.get("base_url"),
                    api_key=cfg.get("llm_api_key") or cfg.get("api_key"),
                    auth_mode=cfg.get("llm_auth_mode") or cfg.get("auth_mode") or "auto",
                    provider=cfg.get("llm_provider") or cfg.get("provider") or "auto",
                    preferred_profile_id=cfg.get("llm_profile_id") or cfg.get("preferred_profile_id"),
                    thinking_level=cfg.get("llm_thinking_level") or cfg.get("thinking_level"),
                )
                if cost_tracker is not None:
                    return self._wrap_llm_for_cost_tracking(llm, cost_tracker) or llm
                return llm

            logs.append(f"LLM: {llm_config['model']}")
            logs.append(f"Device: {device}")
            logs.append(f"Seed: {seed}")
            logs.append(f"Train data: {train_h5ad}")
            logs.append(f"Workspace: {workspace_dir}")
            logs.append("Runner attempt: 1/1")
            logs.append(f"Intermediate contract dir: {Path(output_dir) / 'intermediates'}")

            previous_cwd = Path.cwd()
            os.chdir(workspace_dir)
            try:
                controller = SessionController(
                    llm_factory,
                    initial_config={
                        "llm_provider": llm_config.get("provider") or "auto",
                        "llm_base_url": llm_config.get("base_url"),
                        "llm_api_key": llm_config.get("api_key"),
                        "llm_model": llm_config.get("model") or "gpt-4o",
                        "llm_auth_mode": llm_config.get("auth_mode") or "auto",
                        "llm_profile_id": llm_config.get("preferred_profile_id"),
                        "llm_thinking_level": llm_config.get("thinking_level"),
                        # DynBench is a strict file-delivery benchmark. Stop hooks
                        # can keep a completed run in an open-ended continuation
                        # loop after deliverables already pass the public verifier.
                        "stop_hook_enabled": False,
                    },
                )
                session, _ = open_or_create_session(
                    controller,
                    SessionOpenSpec(
                        input_path=train_h5ad,
                        question=task_prompt,
                        output_dir=output_dir,
                        device=device,
                        report_format="md",
                        enable_multimodal=False,
                        user_goal_overrides=user_goal,
                    ),
                )
                controller.run_turn(task_prompt)

                harness_rounds = []
                if self.harness_revisions > 0:
                    for revision_index in range(self.harness_revisions):
                        audit_path = Path(output_dir) / f"scientific_harness_audit_round_{revision_index}.json"
                        audit = write_public_audit(
                            train_h5ad=train_h5ad,
                            output_dir=output_dir,
                            audit_path=audit_path,
                        )
                        harness_rounds.append(
                            {
                                "round": revision_index,
                                "audit_path": str(audit_path),
                                "requires_revision": bool(audit.get("requires_revision")),
                                "issue_codes": [
                                    str(item.get("code")) for item in audit.get("issues", [])
                                ],
                            }
                        )
                        if not audit.get("requires_revision"):
                            break
                        controller.run_turn(build_revision_prompt(audit, audit_path))

                    calibration = apply_public_scientific_calibration(
                        train_h5ad=train_h5ad,
                        output_dir=output_dir,
                    )
                    final_audit_path = Path(output_dir) / "scientific_harness_audit_final.json"
                    final_audit = write_public_audit(
                        train_h5ad=train_h5ad,
                        output_dir=output_dir,
                        audit_path=final_audit_path,
                    )
                    (Path(output_dir) / "scientific_harness_manifest.json").write_text(
                        json.dumps(
                            {
                                "version": 1,
                                "public_inputs_only": True,
                                "max_revisions": self.harness_revisions,
                                "rounds": harness_rounds,
                                "calibration": calibration,
                                "final_audit_path": str(final_audit_path),
                                "final_issue_codes": [
                                    str(item.get("code")) for item in final_audit.get("issues", [])
                                ],
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                final_state = dict(session.state or {})
            finally:
                os.chdir(previous_cwd)

            runtime = time.time() - t0
            logs.append(f"Runtime: {runtime:.1f}s")

            if expected_output_files:
                intermediates_manifest = materialize_intermediates_from_files(
                    output_dir=output_dir,
                    search_roots=[output_dir, workspace_dir],
                    started_at=t0,
                )
                manifest = recover_outputs_format_only(
                    output_dir=output_dir,
                    expected_output_files=expected_output_files,
                    search_roots=[workspace_dir, Path(output_dir) / "training_runs"],
                    started_at=t0,
                )
                missing = list(manifest.get("missing_files") or [])
                if manifest.get("recovered_files"):
                    logs.append(
                        "Format-only recovery copied outputs: "
                        + ", ".join(item["name"] for item in manifest["recovered_files"])
                    )
                if not missing:
                    recovery_method = str(manifest.get("recovery_method") or "none")
                    if recovery_method == "format_copy":
                        recovery_method = "workspace_copy"
                    return RunResult(
                        success=True,
                        runtime_sec=runtime,
                        agent_state=final_state,
                        logs=logs,
                        native_status="native_complete" if manifest.get("format_status") == "native_complete" else "native_incomplete",
                        format_status=str(manifest.get("format_status") or "unknown"),
                        recovery_method=recovery_method,
                        recovery_status=str(manifest.get("recovery_status") or "not_applied"),
                        artifact_origin=str(manifest.get("artifact_origin") or "native"),
                        recovery_manifest=manifest.get("recovery_manifest"),
                        cost_summary=cost_tracker.to_dict() if cost_tracker else None,
                    )
                # DISABLED: No intermediate conversion recovery.
                # Agent must produce outputs natively or fail.
                final_error = f"Missing expected output files: {', '.join(missing)}"
                logs.append(f"DISABLED: intermediate conversion recovery removed. {final_error}")
                logs.append(f"WARNING: {final_error}")
                recovery_method = str(manifest.get("recovery_method") or "none")
                if recovery_method == "format_copy":
                    recovery_method = "workspace_copy"
                return RunResult(
                    success=False,
                    runtime_sec=runtime,
                    error_message=final_error,
                    agent_state=final_state,
                    logs=logs,
                    native_status="native_incomplete",
                    format_status=str(manifest.get("format_status") or "incomplete"),
                    recovery_method=recovery_method,
                    recovery_status=str(manifest.get("recovery_status") or "not_applied"),
                    artifact_origin=str(manifest.get("artifact_origin") or "none"),
                    recovery_manifest=manifest.get("recovery_manifest"),
                    native_error_message=final_error,
                    cost_summary=cost_tracker.to_dict() if cost_tracker else None,
                )

            predicted = self._find_predicted_h5ad(output_dir)
            if predicted:
                logs.append(f"Found prediction: {predicted}")
                return RunResult(
                    success=True,
                    predicted_h5ad=predicted,
                    runtime_sec=runtime,
                    agent_state=final_state,
                    logs=logs,
                    native_status="native_complete",
                    format_status="native_complete",
                    recovery_method="none",
                    recovery_status="not_applied",
                    artifact_origin="native",
                    cost_summary=cost_tracker.to_dict() if cost_tracker else None,
                )
            final_error = "Agent completed but no predicted_heldout.h5ad found"
            return RunResult(
                success=False,
                runtime_sec=runtime,
                error_message=final_error,
                agent_state=final_state,
                logs=logs,
                native_status="native_incomplete",
                format_status="incomplete",
                recovery_method="none",
                recovery_status="not_applied",
                artifact_origin="none",
                native_error_message=final_error,
                cost_summary=cost_tracker.to_dict() if cost_tracker else None,
            )
        except Exception as e:
            runtime = time.time() - t0
            final_error = f"{type(e).__name__}: {e}"
            logs.append(f"FAILED: {final_error}")
            logger.error(f"CytoBridge run failed: {final_error}", exc_info=True)
            return RunResult(
                success=False,
                runtime_sec=runtime,
                error_message=final_error,
                logs=logs,
                native_status="native_error",
                format_status="error",
                recovery_method="none",
                recovery_status="not_applied",
                artifact_origin="none",
                native_error_message=final_error,
                cost_summary=cost_tracker.to_dict() if cost_tracker else None,
            )

    def _wrap_llm_for_cost_tracking(self, llm, cost_tracker: CostTracker) -> None:
        """Wrap the LLM object to intercept invoke() calls for cost tracking.

        Uses a proxy class instead of monkey-patching because Pydantic v2
        models do not allow setting non-field attributes on instances.
        """
        if not hasattr(llm, "invoke"):
            return
        original_invoke = llm.invoke

        def _tracked_invoke(*args, **kwargs):
            response = original_invoke(*args, **kwargs)
            # Extract token usage from LangChain-style response
            usage_metadata = getattr(response, "usage_metadata", None) or {}
            if not usage_metadata and hasattr(response, "response_metadata"):
                meta = response.response_metadata or {}
                usage_metadata = meta.get("token_usage", {}) or meta.get("usage", {})
            tokens_in = int(usage_metadata.get("input_tokens", 0) or usage_metadata.get("prompt_tokens", 0) or 0)
            tokens_out = int(usage_metadata.get("output_tokens", 0) or usage_metadata.get("completion_tokens", 0) or 0)
            if tokens_in or tokens_out:
                cost_tracker.record_call(
                    provider="openai",
                    model=str(getattr(llm, "model_name", "") or getattr(llm, "model", "unknown")),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )
            return response

        class _LLMProxy:
            """Transparent proxy that forwards attribute access to the real LLM,
            with invoke() intercepted for cost tracking."""
            def __init__(self, real_llm):
                object.__setattr__(self, "_real", real_llm)
            def __getattr__(self, name):
                return getattr(object.__getattribute__(self, "_real"), name)
            def invoke(self, *args, **kwargs):
                return _tracked_invoke(*args, **kwargs)
            def __repr__(self):
                return f"<LLMProxy of {type(object.__getattribute__(self, '_real')).__name__}>"

        proxy = _LLMProxy(llm)
        # Replace the caller's reference to llm via the run() local variable
        # We return the proxy so the caller can rebind it
        return proxy
