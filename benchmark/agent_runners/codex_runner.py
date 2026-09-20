from __future__ import annotations

import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from benchmark.dynbench.format_recovery import recover_outputs_format_only
from benchmark.dynbench.intermediate_contract import materialize_intermediates_from_files
from benchmark.cost_tracker import CostTracker

from .auth_utils import clear_benchmark_auth_lock
from .base import AgentRunner, RunResult


CODEX_TRANSPORT_FAILURE_PATTERNS = (
    "failed to connect to websocket",
    "tls handshake eof",
    "stream disconnected before completion",
    "error sending request",
    "transport channel closed",
    "responses_websocket",
)


class CodexRunner(AgentRunner):
    """Run benchmark tasks through the external Codex CLI."""

    agent_id = "codex"

    def __init__(
        self,
        llm_model: Optional[str] = None,
        llm_profile_id: Optional[str] = None,
        timeout_seconds: int = 1800,
    ) -> None:
        self.llm_model = llm_model or "gpt-5.5"
        self.llm_profile_id = llm_profile_id
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        train_h5ad: str | Path,
        task_prompt: str,
        output_dir: str | Path,
        seed: int = 42,
        device: str = "cpu",
        time_key: str = "time",
        timeout_sec: Optional[float] = None,
        cost_tracker: Optional[CostTracker] = None,
        **kwargs,
    ) -> RunResult:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        train_h5ad = Path(train_h5ad).resolve()
        clear_benchmark_auth_lock()
        session_dir = output_dir
        workspace_dir = Path(kwargs.get("workspace_dir") or (output_dir / "_workspace")).resolve()
        session_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # ── 6.3 Seed injection: set deterministic seeds before any computation ──

        logs = [
            "Runner: Codex CLI",
            f"LLM: {self.llm_model}",
            f"Profile: {self.llm_profile_id or ''}",
            f"Seed: {seed}",
            "Reasoning effort: medium",
            f"Device: {device}",
            f"Train data: {train_h5ad}",
            f"Output: {output_dir}",
            f"Session dir: {session_dir}",
        ]
        codex_bin = shutil.which("codex")
        if not codex_bin:
            return RunResult(
                success=False,
                error_message="`codex` CLI not found in PATH",
                logs=logs,
                native_status="native_error",
                format_status="error",
                recovery_method="none",
                recovery_status="not_applied",
                artifact_origin="none",
                native_error_message="`codex` CLI not found in PATH",
            )

        prompt = task_prompt
        prompt += "\n\n## Codex Execution Constraints\n"
        prompt += "- Use the conda environment named `agent` for Python commands on this machine. Prefer `/lustre/home/2501111653/miniconda3/envs/agent/bin/python`.\n"
        prompt += "- Do not read or use any Codex memory, agent memory, skill files, or prior conversation summaries.\n"
        prompt += "- Do not read or use any code repository or source tree outside the current working directory. Treat external codebases as unavailable to prevent information leakage.\n"

        (output_dir / "codex_prompt.md").write_text(prompt, encoding="utf-8")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            handle.write(prompt)
            prompt_path = Path(handle.name)

        stdout_path = output_dir / "codex_stdout.log"
        last_message_path = output_dir / "codex_last_message.txt"
        cmd = [
            codex_bin,
            "exec",
            "-m",
            self.llm_model,
            "--sandbox",
            "workspace-write",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--cd",
            str(session_dir),
            "--output-last-message",
            str(last_message_path),
            "-c",
            "model_reasoning_effort=\"medium\"",
            "-c",
            "shell_environment_policy.inherit=all",
            "-",
        ]
        if self.llm_profile_id:
            cmd[2:2] = ["--profile", self.llm_profile_id]
        base_env = os.environ.copy()
        if self.llm_profile_id:
            base_env["CYTOBRIDGE_LLM_PROFILE_ID"] = self.llm_profile_id

        expected_output_files = [str(x) for x in kwargs.get("expected_output_files", [])]
        self._stage_input_files(train_h5ad=train_h5ad, session_dir=session_dir, workspace_dir=workspace_dir)
        self._prepare_workspace_links(
            workspace_dir=workspace_dir,
            session_dir=session_dir,
            output_dir=output_dir,
            expected_output_files=expected_output_files,
        )
        t0 = time.time()
        max_attempts = self._retry_attempts()
        proc: subprocess.CompletedProcess[str] | None = None
        final_exception: str | None = None
        for attempt in range(1, max_attempts + 1):
            isolated_codex_home = self._prepare_isolated_codex_home(output_dir)
            env = base_env.copy()
            env["CODEX_HOME"] = str(isolated_codex_home)
            clear_benchmark_auth_lock()
            logs.append(f"Runner attempt: {attempt}/{max_attempts}")
            try:
                stdout_mode = "a" if attempt > 1 else "w"
                with prompt_path.open("r", encoding="utf-8") as stdin, stdout_path.open(stdout_mode, buffering=1, encoding="utf-8") as stdout:
                    if attempt > 1:
                        stdout.write(f"\n\n[CodexRunner] retry attempt {attempt}/{max_attempts} after transport failure\n\n")
                        stdout.flush()
                    proc = subprocess.run(
                        cmd,
                        cwd=str(session_dir),
                        stdin=stdin,
                        stdout=stdout,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                        timeout=float(timeout_sec or self.timeout_seconds),
                    )
            except Exception as exc:
                final_exception = f"{type(exc).__name__}: {exc}"
                shutil.rmtree(isolated_codex_home, ignore_errors=True)
                break
            finally:
                shutil.rmtree(isolated_codex_home, ignore_errors=True)

            stdout_text = stdout_path.read_text(encoding="utf-8", errors="ignore") if stdout_path.exists() else ""
            if self._transport_failed_without_outputs(stdout_text, expected_output_files, output_dir) and attempt < max_attempts:
                logs.append(
                    "Codex transport failure before output materialization; "
                    f"retrying attempt {attempt + 1}/{max_attempts}."
                )
                time.sleep(min(30.0, 5.0 * attempt))
                continue
            break

        runtime = time.time() - t0
        if proc is None:
            final_error = final_exception or "Codex CLI did not start"
            prompt_path.unlink(missing_ok=True)
            return RunResult(
                success=False,
                runtime_sec=runtime,
                error_message=final_error,
                logs=logs + [f"FAILED: {final_error}"],
                native_status="native_error",
                format_status="error",
                recovery_method="none",
                recovery_status="not_applied",
                artifact_origin="none",
                native_error_message=final_error,
            )
        intermediates_manifest = materialize_intermediates_from_files(
            output_dir=output_dir,
            search_roots=[output_dir, session_dir, workspace_dir],
            started_at=t0,
        )
        manifest = recover_outputs_format_only(
            output_dir=output_dir,
            expected_output_files=expected_output_files,
            search_roots=[output_dir, session_dir, workspace_dir],
            started_at=t0,
        )
        missing = list(manifest.get("missing_files") or [])
        native_error_message = None
        if proc.returncode != 0:
            native_error_message = f"Codex CLI exited with code {proc.returncode}"
        if manifest.get("recovered_files"):
            logs.append(
                "Format-only recovery copied outputs: "
                + ", ".join(item["name"] for item in manifest["recovered_files"])
            )
        # ── 6.8 Record cost data ──
        cost_summary = None
        if cost_tracker is not None:
            # Parse token usage from stdout if available
            self._record_cost_from_output(cost_tracker, stdout_path, seed)
            cost_summary = cost_tracker.to_dict()

        prompt_path.unlink(missing_ok=True)
        if not missing:
            if native_error_message:
                logs.append(
                    "Codex CLI exited non-zero after materializing or staging all expected outputs; "
                    "continuing to evaluator."
                )
            return RunResult(
                success=True,
                runtime_sec=runtime,
                logs=logs + [f"Runtime: {runtime:.1f}s"],
                native_status="native_complete" if manifest.get("format_status") == "native_complete" else "native_incomplete",
                format_status=str(manifest.get("format_status") or "unknown"),
                recovery_method="workspace_copy" if manifest.get("format_status") == "recovered_complete" else "none",
                recovery_status="succeeded" if manifest.get("format_status") == "recovered_complete" else "not_applied",
                artifact_origin=str(manifest.get("artifact_origin") or "native"),
                recovery_manifest=manifest.get("recovery_manifest"),
                native_error_message=native_error_message,
                cost_summary=cost_summary,
            )

        # DISABLED: No intermediate conversion recovery.
        # Agent must produce outputs natively or fail.
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="ignore") if stdout_path.exists() else ""
        final_error = f"Missing expected output files: {', '.join(missing)}"
        if proc.returncode != 0:
            final_error = f"Codex CLI exited with code {proc.returncode}; {final_error}"
        if stdout_text:
            tail = "\n".join(stdout_text.strip().splitlines()[-20:])
            final_error += f"; log tail: {tail}"
        logs.append(f"DISABLED: intermediate conversion and deterministic fallback removed. {final_error}")
        return RunResult(
            success=False,
            runtime_sec=runtime,
            error_message=final_error,
            logs=logs + [f"Runtime: {runtime:.1f}s"],
            native_status="native_incomplete",
            format_status=str(manifest.get("format_status") or "incomplete"),
            recovery_method="workspace_copy" if manifest.get("recovered_files") else "none",
            recovery_status="failed" if manifest.get("recovered_files") else "not_applied",
            artifact_origin=str(manifest.get("artifact_origin") or "none"),
            recovery_manifest=manifest.get("recovery_manifest"),
            native_error_message=native_error_message or final_error,
            cost_summary=cost_summary,
        )

    @staticmethod
    def _retry_attempts() -> int:
        raw = os.environ.get("DYNBENCH_CODEX_RETRY_ATTEMPTS", "").strip()
        if not raw:
            return 3
        try:
            return max(1, min(5, int(raw)))
        except ValueError:
            return 3

    @staticmethod
    def _transport_failed_without_outputs(stdout_text: str, expected_output_files: list[str], output_dir: Path) -> bool:
        lower = stdout_text.lower()
        if not any(pattern in lower for pattern in CODEX_TRANSPORT_FAILURE_PATTERNS):
            return False
        return not any((output_dir / name).exists() and (output_dir / name).stat().st_size > 0 for name in expected_output_files)

    @staticmethod
    def _prepare_isolated_codex_home(output_dir: Path) -> Path:
        source_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        target_home = Path(tempfile.mkdtemp(prefix="codex-bench-home-"))
        auth_source = source_home / "auth.json"
        if auth_source.exists():
            shutil.copy2(auth_source, target_home / "auth.json")
        (output_dir / "codex_home_isolation.txt").write_text(
            "Codex benchmark run used an isolated temporary CODEX_HOME with auth.json only; "
            "user config, rules, memories, and prior sessions were not copied.\n",
            encoding="utf-8",
        )
        return target_home

    def _prepare_workspace_links(
        self,
        *,
        workspace_dir: Path,
        session_dir: Path,
        output_dir: Path,
        expected_output_files: list[str],
    ) -> None:
        inter_dir = output_dir / "intermediates"
        inter_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "intermediates").mkdir(parents=True, exist_ok=True)
        (session_dir / "intermediates").mkdir(parents=True, exist_ok=True)

        for name in expected_output_files:
            self._link_into_workspace(workspace_dir / name, output_dir / name)
            self._link_into_workspace(session_dir / name, output_dir / name)

        for name in [
            "holdout_prediction.json",
            "velocity_field.json",
            "growth_rates.json",
            "per_cell_fate.json",
            "perturbation_results.json",
            "driver_genes.json",
            "manifest.json",
        ]:
            self._link_into_workspace(workspace_dir / "intermediates" / name, inter_dir / name)
            self._link_into_workspace(session_dir / "intermediates" / name, inter_dir / name)

    @staticmethod
    def _stage_input_files(*, train_h5ad: Path, session_dir: Path, workspace_dir: Path) -> None:
        source_dir = train_h5ad.parent
        for name in ["train.h5ad", "prediction_targets.json", "fate_classifier.pkl", "TASK.md"]:
            source = source_dir / name
            if not source.exists():
                continue
            for target_dir in (session_dir, workspace_dir):
                target = target_dir / name
                if target.exists() or target.is_symlink():
                    continue
                # Codex runs with a workspace sandbox and is instructed not to inspect
                # external code/data roots, so public task inputs must be real files
                # inside the current working directory rather than symlinks outward.
                shutil.copy2(source, target)

    @staticmethod
    def _link_into_workspace(link_path: Path, target_path: Path) -> None:
        link_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.exists() or link_path.is_symlink():
            return
        # Prevent self-referencing symlinks (e.g. when session_dir == output_dir)
        try:
            if link_path.resolve() == target_path.resolve():
                return
        except OSError:
            return
        try:
            link_path.symlink_to(target_path)
        except OSError:
            # Symlinks are a best-effort convenience layer; recovery still handles fallback.
            return

    def _record_cost_from_output(
        self,
        cost_tracker: CostTracker,
        stdout_path: Path,
        seed: int,
    ) -> None:
        """Parse Codex CLI stdout for token usage and record in tracker."""
        import re as _re

        if not stdout_path.exists():
            return
        try:
            text = stdout_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return
        # Try to find token usage patterns: "tokens: 1234 in, 567 out" or similar
        tokens_in = 0
        tokens_out = 0
        for match in _re.finditer(r"(\d+)\s*(?:tokens?|tok)\s*(?:in|input)", text, _re.IGNORECASE):
            tokens_in = max(tokens_in, int(match.group(1)))
        for match in _re.finditer(r"(\d+)\s*(?:tokens?|tok)\s*(?:out|output)", text, _re.IGNORECASE):
            tokens_out = max(tokens_out, int(match.group(1)))
        # Also try compact patterns like "1234/567 tokens"
        for match in _re.finditer(r"(\d+)/(\d+)\s*tokens", text, _re.IGNORECASE):
            tokens_in = max(tokens_in, int(match.group(1)))
            tokens_out = max(tokens_out, int(match.group(2)))
        if tokens_in or tokens_out:
            cost_tracker.record_call(
                provider="openai",
                model=self.llm_model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        else:
            # Record an empty call just to mark that the agent ran
            cost_tracker.record_call(
                provider="openai",
                model=self.llm_model,
                tokens_in=0,
                tokens_out=0,
            )
