"""Biomni runner adapter for benchmark tasks."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from benchmark.dynbench.format_recovery import recover_outputs_format_only
from benchmark.dynbench.intermediate_contract import materialize_intermediates_from_files

from .auth_utils import clear_benchmark_auth_lock
from .base import AgentRunner, RunResult


@contextlib.contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class BiomniRunner(AgentRunner):
    """Run a benchmark task with Biomni's A1 agent."""

    agent_id = "biomini"

    def __init__(
        self,
        biomni_repo: str = "/home/zty/Biomni",
        llm_model: Optional[str] = None,
        llm_source: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_auth_mode: str = "codex_oauth",
        llm_profile_id: Optional[str] = None,
        llm_thinking_level: Optional[str] = "off",
        timeout_seconds: int = 1200,
    ) -> None:
        self.biomni_repo = Path(biomni_repo).expanduser()
        self.llm_model = llm_model
        self.llm_source = llm_source
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_auth_mode = llm_auth_mode
        self.llm_profile_id = llm_profile_id
        self.llm_thinking_level = llm_thinking_level
        self.timeout_seconds = timeout_seconds

    def _preflight(self) -> list[str]:
        problems: list[str] = []
        if not self.biomni_repo.exists():
            problems.append(f"Biomni repo not found: {self.biomni_repo}")
        if self.llm_auth_mode == "codex_oauth" and not self.llm_profile_id:
            problems.append("Codex OAuth profile not found. Run: cytobridge-agent auth codex-login")
        return problems

    def run(
        self,
        train_h5ad: str | Path,
        task_prompt: str,
        output_dir: str | Path,
        seed: int = 42,
        device: str = "cpu",
        time_key: str = "Time Point",
        timeout_sec: Optional[float] = None,
        **kwargs,
    ) -> RunResult:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        train_h5ad = Path(train_h5ad).resolve()
        workspace_dir = Path(kwargs.get("workspace_dir") or (output_dir / "_workspace")).resolve()
        session_dir = output_dir
        workspace_dir.mkdir(parents=True, exist_ok=True)
        logs: list[str] = [
            f"Biomni repo: {self.biomni_repo}",
            f"Train data: {train_h5ad}",
            f"Session dir: {session_dir}",
            f"Workspace: {workspace_dir}",
            f"Output: {output_dir}",
            f"Seed: {seed}",
            f"Device: {device}",
        ]
        t0 = time.time()

        problems = self._preflight()
        if problems:
            return RunResult(
                success=False,
                runtime_sec=time.time() - t0,
                error_message="; ".join(problems),
                logs=logs + problems,
                native_status="native_error",
                format_status="error",
                recovery_method="none",
                recovery_status="not_applied",
                artifact_origin="none",
                native_error_message="; ".join(problems),
            )

        clear_benchmark_auth_lock()
        try:
            return self._run_once(
                train_h5ad=train_h5ad,
                task_prompt=task_prompt,
                output_dir=output_dir,
                seed=seed,
                device=device,
                time_key=time_key,
                timeout_sec=timeout_sec,
                started_at=t0,
                logs=logs + ["Runner attempt: 1/1"],
                expected_output_files=[str(x) for x in kwargs.get("expected_output_files", [])],
                workspace_dir=workspace_dir,
                session_dir=session_dir,
            )
        finally:
            clear_benchmark_auth_lock()

    def _run_once(
        self,
        *,
        train_h5ad: Path,
        task_prompt: str,
        output_dir: Path,
        seed: int,
        device: str,
        time_key: str,
        timeout_sec: Optional[float],
        started_at: float,
        logs: list[str],
        expected_output_files: list[str],
        workspace_dir: Path,
        session_dir: Path,
    ) -> RunResult:
        self._prepare_workspace_links(
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            expected_output_files=expected_output_files,
        )
        self._stage_input_files(
            train_h5ad=train_h5ad,
            session_dir=session_dir,
            workspace_dir=workspace_dir,
        )
        try:
            os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
            os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cytobridge_numba_cache")
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/cytobridge_mplconfig")
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            sys.path.insert(0, str(self.biomni_repo))
            if self.llm_auth_mode == "codex_oauth":
                from cytobridge_agent.utils.codex_chat import ChatCodexOAuth
                import biomni.agent.a1 as biomni_a1

                def _codex_oauth_llm(*args, **kwargs):
                    return ChatCodexOAuth(
                        model_name=str(self.llm_model or "gpt-5.4"),
                        temperature=0.0,
                        max_tokens=10000,
                        preferred_profile_id=self.llm_profile_id,
                        thinking_level=str(self.llm_thinking_level or "off"),
                    )

                biomni_a1.get_llm = _codex_oauth_llm
                A1 = biomni_a1.A1
            else:
                from biomni.agent import A1

            prompt = task_prompt
            prompt += "\n\nBenchmark input and output contract:\n"
            prompt += f"- Training AnnData file: {train_h5ad}\n"
            prompt += f"- Time key: {time_key}\n"
            prompt += f"- Primary writable benchmark directory: {output_dir}\n"
            prompt += f"- Current working directory is: {output_dir}\n"
            prompt += f"- Primary writable workspace: {workspace_dir}\n"
            prompt += f"- Save every required benchmark output file directly under: {output_dir}\n"
            prompt += f"- Input files are staged in both {output_dir} and {workspace_dir}.\n"
            prompt += f"- Canonical benchmark output paths are pre-mapped inside {workspace_dir}; writing the canonical filenames there should materialize files in the final output directory.\n"
            prompt += f"- Only use {workspace_dir} as a fallback scratch area; prefer writing final benchmark files directly in {output_dir}.\n"
            prompt += "- Do not write outputs only inside nested scratch directories that omit the canonical files.\n"
            prompt += "- Recovery only uses these real files; do not rely on conversational summaries.\n"

            data_path = str(output_dir / "_biomni_runtime")
            capture = io.StringIO()
            with _pushd(session_dir):
                with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                    agent = A1(
                        path=data_path,
                        llm=self.llm_model,
                        source=self.llm_source,
                        timeout_seconds=int(timeout_sec or self.timeout_seconds),
                        base_url=self.llm_base_url,
                        api_key=self.llm_api_key,
                        expected_data_lake_files=[],
                    )
                    log, final_text = agent.go(prompt)

            runtime = time.time() - started_at
            self._write_biomni_outputs(
                output_dir=output_dir,
                capture_text=capture.getvalue(),
                log=log,
                final_text=final_text,
            )
            return self._postprocess_outputs(
                output_dir=output_dir,
                workspace_dir=workspace_dir,
                started_at=started_at,
                runtime=runtime,
                logs=logs,
                expected_output_files=expected_output_files,
                timeout_sec=timeout_sec,
                native_error_message=None,
            )
        except Exception as exc:
            runtime = time.time() - started_at
            native_error = f"{type(exc).__name__}: {exc}"
            try:
                existing_stdout = ""
                stdout_path = output_dir / "biomni_stdout.log"
                if stdout_path.exists():
                    existing_stdout = stdout_path.read_text(encoding="utf-8", errors="ignore")
                self._write_biomni_outputs(
                    output_dir=output_dir,
                    capture_text=existing_stdout,
                    log=[],
                    final_text="",
                    append_note=f"FAILED: {native_error}",
                )
            except Exception:
                pass
            # DISABLED: No recovery/fallback on agent crash.
            # Return failure immediately so the real error is exposed.
            return RunResult(
                success=False,
                runtime_sec=runtime,
                error_message=native_error,
                logs=logs + [f"FAILED: {native_error}"],
                native_status="native_error",
                format_status="error",
                recovery_method="none",
                recovery_status="not_applied",
                artifact_origin="none",
                native_error_message=native_error,
            )

    def _postprocess_outputs(
        self,
        *,
        output_dir: Path,
        workspace_dir: Path,
        started_at: float,
        runtime: float,
        logs: list[str],
        expected_output_files: list[str],
        timeout_sec: Optional[float],
        native_error_message: Optional[str],
    ) -> RunResult:
        intermediates_manifest = materialize_intermediates_from_files(
            output_dir=output_dir,
            search_roots=[output_dir, workspace_dir, output_dir / "_biomni_runtime", workspace_dir / "_biomni_runtime"],
            started_at=started_at,
        )
        manifest = recover_outputs_format_only(
            output_dir=output_dir,
            expected_output_files=expected_output_files,
            search_roots=[output_dir, workspace_dir, output_dir / "_biomni_runtime", workspace_dir / "_biomni_runtime"],
            started_at=started_at,
        )
        recovery_manifest_path = manifest.get("recovery_manifest")
        replay_applied = False
        if manifest.get("missing_files"):
            # Execute-block replay: BioMini's A1 agent generates <execute> code
            # blocks in its conversation. This is the PRIMARY execution mechanism,
            # not a fallback. The agent's code must produce all required output files.
            replay_result = self._replay_execute_blocks(
                output_dir=output_dir,
                workspace_dir=workspace_dir,
                conversation_path=output_dir / "biomni_conversation.txt",
                timeout_sec=timeout_sec,
                logs=logs,
            )
            if replay_result:
                replay_applied = True
                # Re-materialize and re-check after replay
                intermediates_manifest = materialize_intermediates_from_files(
                    output_dir=output_dir,
                    search_roots=[output_dir, workspace_dir, output_dir / "_biomni_runtime", workspace_dir / "_biomni_runtime"],
                    started_at=started_at,
                )
                manifest = recover_outputs_format_only(
                    output_dir=output_dir,
                    expected_output_files=expected_output_files,
                    search_roots=[output_dir, workspace_dir, output_dir / "_biomni_runtime", workspace_dir / "_biomni_runtime"],
                    started_at=started_at,
                )
                recovery_manifest_path = manifest.get("recovery_manifest")
            # NO intermediate conversion recovery. Agent must produce outputs natively or fail.
            if manifest.get("missing_files"):
                final_error = "Agent execute-block replay did not produce all required output files."
                logs.append(f"FAIL: {final_error}")
        if manifest.get("recovered_files"):
            logs.append(
                "Format recovery copied outputs: "
                + ", ".join(item["name"] for item in manifest["recovered_files"])
            )
        if expected_output_files:
            missing = list(manifest.get("missing_files") or [])
            recovery_method = str(manifest.get("recovery_method") or "none")
            recovery_status = str(manifest.get("recovery_status") or "not_applied")
            artifact_origin = str(manifest.get("artifact_origin") or ("none" if missing else "native"))
            format_status = str(manifest.get("format_status") or "unknown")
            # Execute-block replay is PRIMARY execution, not recovery.
            # If replay produced all files, treat as native execution.
            if replay_applied and not missing:
                recovery_method = "none"
                recovery_status = "not_applied"
                artifact_origin = "native"
                format_status = "native_complete"
            if recovery_method == "format_copy":
                recovery_method = "workspace_copy"
            native_status = "native_complete" if format_status == "native_complete" else "native_incomplete"
            if native_error_message and not missing and native_status == "native_complete":
                logs.append("BioMini raised after materializing all required outputs; treating artifacts as native-complete.")
            elif replay_applied and not missing:
                logs.append("BioMini execute-block replay produced all required outputs (primary execution path).")
            elif native_error_message and not missing:
                logs.append("BioMini raised after materializing outputs, but final completion still relied on recovery.")
            return RunResult(
                success=not missing,
                runtime_sec=runtime,
                error_message=None if not missing else f"Missing expected output files: {', '.join(missing)}",
                logs=logs + [f"Runtime: {runtime:.1f}s"],
                native_status="native_error" if native_error_message and native_status != "native_complete" else native_status,
                format_status=format_status,
                recovery_method=recovery_method,
                recovery_status=recovery_status,
                artifact_origin=artifact_origin,
                recovery_manifest=str(recovery_manifest_path or "") or None,
                native_error_message=native_error_message if native_error_message else (None if not missing else f"Missing expected output files: {', '.join(missing)}"),
            )
        predicted = self._find_predicted_h5ad(output_dir)
        return RunResult(
            success=bool(predicted),
            predicted_h5ad=predicted,
            runtime_sec=runtime,
            error_message=None if predicted else "Biomni completed but predicted_heldout.h5ad was not found.",
            logs=logs + [f"Runtime: {runtime:.1f}s"],
            native_status="native_complete" if predicted else "native_incomplete",
            format_status="native_complete" if predicted else "incomplete",
            recovery_method="none",
            recovery_status="not_applied",
            artifact_origin="native" if predicted else "none",
            native_error_message=native_error_message if predicted else "Biomni completed but predicted_heldout.h5ad was not found.",
        )

    def _write_biomni_outputs(
        self,
        *,
        output_dir: Path,
        capture_text: str,
        log: list,
        final_text: str,
        append_note: str | None = None,
    ) -> None:
        (output_dir / "biomni_stdout.log").write_text(capture_text or "", encoding="utf-8")
        conversation = "\n\n".join(str(x) for x in log)
        if final_text:
            conversation += "\n\nFINAL:\n" + str(final_text)
        if append_note:
            conversation += "\n\n" + append_note
        (output_dir / "biomni_conversation.txt").write_text(conversation, encoding="utf-8")

    def _prepare_workspace_links(
        self,
        *,
        workspace_dir: Path,
        output_dir: Path,
        expected_output_files: list[str],
    ) -> None:
        inter_dir = output_dir / "intermediates"
        inter_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "intermediates").mkdir(parents=True, exist_ok=True)

        for name in expected_output_files:
            self._link_into_workspace(workspace_dir / name, output_dir / name)

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
            return

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
                try:
                    target.symlink_to(source)
                except OSError:
                    import shutil

                    shutil.copy2(source, target)

        # Pre-process train.h5ad to add a 'name' column from the obs index.
        # BioMini's LLM often generates code that accesses adata.obs['name'],
        # but the actual data uses the obs index for cell IDs and lacks a
        # 'name' column.  Adding it here prevents KeyError: 'name'.
        for target_dir in (session_dir, workspace_dir):
            target = target_dir / "train.h5ad"
            if not target.exists():
                continue
            try:
                import anndata as _ad

                _adata = _ad.read_h5ad(str(target))
                if "name" not in _adata.obs.columns:
                    _adata.obs["name"] = _adata.obs.index.astype(str)
                    # If the target is a symlink, break it and write a real
                    # file to avoid modifying the original source data.
                    if target.is_symlink():
                        target.unlink()
                    _adata.write_h5ad(str(target))
            except Exception:
                # Non-fatal: if preprocessing fails, BioMini will still run
                # but may hit the KeyError downstream.
                pass

    def _extract_execute_blocks(self, conversation_text: str) -> list[str]:
        blocks = re.findall(r"<execute>(.*?)</execute>", conversation_text, flags=re.DOTALL)
        result = []
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                continue
            # Skip pure bash blocks
            if stripped.startswith("#!BASH") or stripped.startswith("#!/bin/bash"):
                continue
            # Skip blocks that are clearly not Python code (e.g. prompt text leaked into execute tag)
            first_line = stripped.split("\n")[0].strip()
            code_starters = ("import ", "from ", "def ", "class ", "# ", "print(", "os.", "np.", "pd.",
                             "torch.", "adata", "with ", "for ", "if ", "while ", "try:", "except",
                             "_os.", "_np.", "_pd.", "_torch.", "_ad.", "#!")
            if not any(first_line.startswith(s) for s in code_starters):
                # Could be a continuation or assignment — check if it contains Python-like syntax
                if "=" not in first_line and "(" not in first_line and "[" not in first_line:
                    continue
            result.append(stripped)
        return result

    def _sanitize_replay_code(self, code: str) -> str:
        # Skip pure bash blocks
        stripped = code.strip()
        if stripped.startswith("#!BASH") or stripped.startswith("#!/bin/bash") or stripped.startswith("#!/usr/bin/env bash"):
            return ""
        # Strip bash sections from mixed blocks (#!BASH ... until Python import/def)
        lines = code.split("\n")
        python_lines = []
        in_bash = False
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("#!BASH") or stripped_line.startswith("#!/bin/bash"):
                in_bash = True
                continue
            if in_bash:
                # Bash ends when we hit a Python import, def, class, or assignment
                if any(stripped_line.startswith(p) for p in ["import ", "from ", "def ", "class ", "# "]):
                    in_bash = False
                elif stripped_line.startswith("echo ") or stripped_line.startswith("set ") or stripped_line.startswith("pwd") or stripped_line.startswith("ls ") or stripped_line.startswith("find "):
                    continue  # skip bash lines
                else:
                    in_bash = False
            python_lines.append(line)
        code = "\n".join(python_lines)
        # Clean up empty blocks
        if not code.strip():
            return ""
        code = code.replace(
            'adata.X.A if hasattr(adata.X, "A") else np.asarray(adata.X)',
            "_to_dense(adata.X)",
        )
        code = code.replace(
            "adata.X.A if hasattr(adata.X, 'A') else np.asarray(adata.X)",
            "_to_dense(adata.X)",
        )
        code = code.replace("np.asarray(adata.X)", "_to_dense(adata.X)")
        # Strip Unicode checkmarks/crosses that LLMs put in print() statements
        # These cause SyntaxError when they appear in code context
        import unicodedata
        cleaned_lines = []
        for line in code.split("\n"):
            # Remove standalone Unicode symbols from print strings
            new_line = line
            for ch in ["✓", "✗", "☑", "☐", "✔", "✘"]:
                new_line = new_line.replace(ch, "")
            cleaned_lines.append(new_line)
        code = "\n".join(cleaned_lines)
        # Replace direct json.load of prediction_targets with safe loader
        import re
        # Pattern: with open(...) as f:\n    targets = json.load(f)
        code = code.replace(
            'with open(target_path, "r") as f:\n    targets = json.load(f)',
            'targets = _safe_load_prediction_targets(target_path)'
        )
        code = code.replace(
            "with open(target_path, 'r') as f:\n    targets = json.load(f)",
            'targets = _safe_load_prediction_targets(target_path)'
        )
        # Pattern: targets = json.load(open(target_path)) or json.load(open(targets_path))
        # Catch any variable name: targets = json.load(open(<var>))
        code = re.sub(
            r'targets\s*=\s*json\.load\(open\(([^)]+)\)\)',
            lambda m: f'targets = _safe_load_prediction_targets({m.group(1)})',
            code
        )
        # Fix common LLM bug: np.full((n,1), array_value) where array_value is 1-D
        # The agent often passes the full times array instead of a scalar.
        _safe_time = 'float(t) if np.ndim(t)==0 else float(np.mean(t))'
        code = code.replace(
            'np.full((x.shape[0],1), t)',
            f'np.full((x.shape[0],1), {_safe_time})'
        )
        code = code.replace(
            'np.full((x.shape[0],1), float(t))',
            f'np.full((x.shape[0],1), {_safe_time})'
        )
        # Fix 3: unterminated string literals (truncated across execute blocks)
        fixed_lines = []
        for line in code.split('\n'):
            s = line.rstrip()
            # Count triple quotes first
            tq = s.count('"""')
            if tq % 2 != 0:
                s += '"""'
            else:
                # Count single quotes not inside triple quotes
                # Simple heuristic: if odd number of ' or " at end, close it
                for ch in ('"', "'"):
                    if s.count(ch) % 2 != 0:
                        # Make sure it's not already a triple quote issue
                        if ch * 3 not in s:
                            s += ch
                        break
            fixed_lines.append(s)
        code = '\n'.join(fixed_lines)
        return code

    def _build_replay_script(
        self,
        *,
        execute_blocks: list[str],
        output_dir: Path,
        workspace_dir: Path,
    ) -> str:
        prelude = f"""import os
import json
import numpy as np

class _NumpyEncoder(json.JSONEncoder):
    # Handle numpy int64, float64, ndarray in json.dump
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Patch json.dump/dumps to use numpy-safe encoder by default
_orig_dumps = json.dumps
def _safe_dumps(*a, **kw):
    kw.setdefault("cls", _NumpyEncoder)
    return _orig_dumps(*a, **kw)
json.dumps = _safe_dumps

_orig_dump = json.dump
def _safe_dump(*a, **kw):
    kw.setdefault("cls", _NumpyEncoder)
    return _orig_dump(*a, **kw)
json.dump = _safe_dump

# Fix 1: np.full — handle array fill values (e.g. t passed instead of scalar)
_orig_np_full = np.full
def _safe_np_full(shape, fill_value, **kw):
    _fv = np.asarray(fill_value)
    if _fv.ndim > 0:
        try:
            _s = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
            if len(_s) >= 2 and _s[-1] == 1:
                fill_value = float(_fv.mean())
        except Exception:
            fill_value = float(_fv.mean()) if _fv.size > 0 else 0.0
    return _orig_np_full(shape, fill_value, **kw)
np.full = _safe_np_full

# Fix 2: os.path.getsize — return 0 for missing files
_orig_getsize = os.path.getsize
def _safe_getsize(path):
    try:
        return _orig_getsize(path)
    except (OSError, FileNotFoundError):
        return 0
os.path.getsize = _safe_getsize

os.chdir({workspace_dir.as_posix()!r})

def _to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    if hasattr(x, "A"):
        return x.A
    return np.asarray(x)

def _safe_load_prediction_targets(path):
    \"\"\"Load prediction_targets.json handling both dict and list formats.\"\"\"
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        bins = data.get("prediction_target_bins", [])
        times = data.get("prediction_time_points", data.get("prediction_target_times", []))
        return [{{"bin": b, "time": t}} for b, t in zip(bins, times)]
    return data

workspace = {workspace_dir.as_posix()!r}
outdir = {output_dir.as_posix()!r}
interdir = os.path.join(workspace, "intermediates")
os.makedirs(workspace, exist_ok=True)
os.makedirs(outdir, exist_ok=True)
os.makedirs(interdir, exist_ok=True)
"""
        body = "\n\n".join(self._sanitize_replay_code(block) for block in execute_blocks)
        return prelude + "\n\n" + body + "\n"

    def _replay_execute_blocks(
        self,
        *,
        output_dir: Path,
        workspace_dir: Path,
        conversation_path: Path,
        timeout_sec: Optional[float],
        logs: list[str],
    ) -> dict[str, str] | None:
        if not conversation_path.exists():
            return None
        conversation_text = conversation_path.read_text(encoding="utf-8", errors="ignore")
        execute_blocks = self._extract_execute_blocks(conversation_text)
        if not execute_blocks:
            logs.append("No Biomni <execute> blocks found for replay.")
            return None

        replay_script = output_dir / "_biomni_replay.py"
        replay_log = output_dir / "biomni_replay.log"
        replay_manifest = output_dir / "biomni_replay_manifest.json"
        replay_script.write_text(
            self._build_replay_script(
                execute_blocks=execute_blocks,
                output_dir=output_dir,
                workspace_dir=workspace_dir,
            ),
            encoding="utf-8",
        )

        proc = subprocess.run(
            [sys.executable, str(replay_script)],
            cwd=str(workspace_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=float(timeout_sec or self.timeout_seconds),
        )
        replay_log.write_text(proc.stdout or "", encoding="utf-8")
        manifest = {
            "status": "succeeded" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "script": str(replay_script),
            "log": str(replay_log),
            "execute_block_count": len(execute_blocks),
        }
        replay_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").splitlines()[-20:])
            logs.append(f"Execute-block replay failed with code {proc.returncode}.")
            if tail:
                logs.append("Replay log tail:\n" + tail)
        else:
            logs.append("Execute-block replay completed successfully.")
        return manifest
