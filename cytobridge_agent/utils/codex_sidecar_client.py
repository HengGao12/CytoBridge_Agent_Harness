from __future__ import annotations

import atexit
import itertools
import json
import os
import select
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class CodexSidecarError(RuntimeError):
    """Raised when the local Codex sidecar fails."""


class CodexSidecarClient:
    """Persistent JSON-RPC client for the local Node Codex sidecar."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_ids = itertools.count(1)
        self._proc: Optional[subprocess.Popen[str]] = None

    @property
    def sidecar_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "codex_sidecar"

    @property
    def server_path(self) -> Path:
        return self.sidecar_dir / "server.mjs"

    def close(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            finally:
                self._proc = None

    @property
    def rpc_timeout_sec(self) -> float:
        raw = os.environ.get("CYTOBRIDGE_CODEX_SIDECAR_RPC_TIMEOUT_SEC", "").strip()
        if not raw:
            return 420.0
        try:
            value = float(raw)
        except ValueError:
            return 420.0
        return max(5.0, min(value, 1800.0))

    def _read_response(
        self,
        proc: subprocess.Popen[str],
        method: str,
        request_id: int,
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        assert proc.stdout is not None
        rpc_timeout_sec = float(timeout_sec if timeout_sec is not None else self.rpc_timeout_sec)
        deadline = time.monotonic() + rpc_timeout_sec
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CodexSidecarError(
                    f"Codex sidecar timed out during {method} after {rpc_timeout_sec:.0f}s"
                )
            ready, _, _ = select.select([proc.stdout], [], [], remaining)
            if not ready:
                raise CodexSidecarError(
                    f"Codex sidecar timed out during {method} after {rpc_timeout_sec:.0f}s"
                )
            response_line = proc.stdout.readline()
            if response_line:
                try:
                    response = json.loads(response_line)
                except json.JSONDecodeError as exc:
                    raise CodexSidecarError(
                        f"Invalid JSON from Codex sidecar during {method}: {response_line!r}"
                    ) from exc
                if response.get("id") != request_id:
                    continue
                return response
            if proc.poll() is not None:
                raise CodexSidecarError(f"Codex sidecar exited while handling {method}")

    def call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        timeout_sec: Optional[float] = None,
    ) -> Any:
        with self._lock:
            last_error: Optional[Exception] = None
            for attempt in range(2):
                proc = self._ensure_started()
                request_id = next(self._request_ids)
                payload = {
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
                request_line = json.dumps(payload, ensure_ascii=False)
                assert proc.stdin is not None
                assert proc.stdout is not None

                try:
                    proc.stdin.write(request_line + "\n")
                    proc.stdin.flush()
                    response = self._read_response(
                        proc,
                        method,
                        request_id,
                        timeout_sec=timeout_sec,
                    )
                except (BrokenPipeError, CodexSidecarError) as exc:
                    last_error = exc
                    self._restart_locked()
                    if attempt == 0:
                        continue
                    if isinstance(exc, CodexSidecarError):
                        raise exc
                    raise CodexSidecarError(f"Codex sidecar pipe closed during {method}") from exc

                error = response.get("error")
                if error:
                    raise CodexSidecarError(str(error.get("message") or error))
                return response.get("result")

            raise CodexSidecarError(
                f"Codex sidecar request failed during {method}: {last_error or 'unknown error'}"
            )

    def _restart_locked(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.kill()
        except Exception:
            pass
        self._proc = None

    def _ensure_started(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        self._ensure_dependencies()
        node = self._find_executable("node", "CYTOBRIDGE_NODE")
        if not node:
            raise CodexSidecarError("`node` is required for Codex OAuth sidecar support.")
        if not self.server_path.exists():
            raise CodexSidecarError(f"Codex sidecar entrypoint not found: {self.server_path}")

        env = os.environ.copy()
        env = self._env_with_bin_first(env, node)
        if (
            (env.get("HTTP_PROXY") or env.get("http_proxy") or env.get("HTTPS_PROXY") or env.get("https_proxy"))
            and not str(env.get("NODE_USE_ENV_PROXY", "")).strip()
        ):
            env["NODE_USE_ENV_PROXY"] = "1"
        self._proc = subprocess.Popen(
            [node, str(self.server_path)],
            cwd=str(self.sidecar_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        return self._proc

    def _ensure_dependencies(self) -> None:
        dep_marker = self.sidecar_dir / "node_modules" / "@mariozechner" / "pi-ai"
        if dep_marker.exists():
            return
        npm = self._find_executable("npm", "CYTOBRIDGE_NPM")
        if not npm:
            raise CodexSidecarError("`npm` is required to install Codex sidecar dependencies.")
        if (self.sidecar_dir / "package-lock.json").exists():
            npm_args = ["ci", "--no-fund", "--no-audit"]
        else:
            npm_args = ["install", "--no-fund", "--no-audit"]
        subprocess.run(
            [npm, *npm_args],
            cwd=str(self.sidecar_dir),
            check=True,
            env=self._env_with_bin_first(os.environ.copy(), npm),
        )

    def _find_executable(self, name: str, override_env: str) -> Optional[str]:
        candidates: list[str] = []
        override = os.environ.get(override_env, "").strip()
        if override:
            candidates.append(override)
        for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
            candidate = str(Path(prefix) / name)
            if Path(candidate).exists():
                candidates.append(candidate)
        path_match = shutil.which(name)
        if path_match:
            candidates.append(path_match)

        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                proc = subprocess.run(
                    [candidate, "--version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=self._env_with_bin_first(os.environ.copy(), candidate),
                    timeout=8,
                    check=False,
                )
            except Exception:
                continue
            if proc.returncode == 0:
                return candidate
        return None

    @staticmethod
    def _env_with_bin_first(env: Dict[str, str], executable: str) -> Dict[str, str]:
        bin_dir = str(Path(executable).parent)
        current_path = env.get("PATH", "")
        env["PATH"] = f"{bin_dir}{os.pathsep}{current_path}" if current_path else bin_dir
        return env


_CLIENT = CodexSidecarClient()
atexit.register(_CLIENT.close)


def get_codex_sidecar_client() -> CodexSidecarClient:
    return _CLIENT


def reset_codex_sidecar_client() -> None:
    _CLIENT.close()


def codex_sidecar_call(
    method: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timeout_sec: Optional[float] = None,
) -> Any:
    return get_codex_sidecar_client().call(method, params, timeout_sec=timeout_sec)


def get_codex_status_safe() -> Dict[str, Any]:
    try:
        result = codex_sidecar_call(
            "auth.runtime.status",
            {"provider": "openai-codex", "timeout_ms": 5000},
            timeout_sec=5.0,
        )
        return result if isinstance(result, dict) else {"provider": "openai-codex", "profiles": []}
    except Exception as exc:
        return {
            "provider": "openai-codex",
            "profiles": [],
            "available_count": 0,
            "last_good_profile_id": None,
            "error": str(exc),
        }


def get_codex_usage_safe(*, profile_id: Optional[str] = None, all_profiles: bool = True) -> Dict[str, Any]:
    try:
        result = codex_sidecar_call(
            "auth.runtime.usage",
            {
                "provider": "openai-codex",
                "profile_id": profile_id,
                "all_profiles": all_profiles,
                "timeout_ms": 5000,
            },
            timeout_sec=5.0,
        )
        return result if isinstance(result, dict) else {"provider": "openai-codex", "profiles": []}
    except Exception as exc:
        return {
            "provider": "openai-codex",
            "profiles": [],
            "error": str(exc),
        }
