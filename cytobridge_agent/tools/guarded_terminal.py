from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from .web_fetch import _validate_public_url


DEFAULT_TIMEOUT_SEC = 20.0
DEFAULT_MAX_OUTPUT_CHARS = 12000
TERMINAL_TOOL_NAME = "run_terminal_command"

_SIMPLE_READ_COMMANDS = {
    "pwd",
    "ls",
    "find",
    "rg",
    "grep",
    "cat",
    "head",
    "tail",
    "wc",
    "stat",
    "file",
}
_READ_ONLY_GIT_SUBCOMMANDS = {
    "status",
    "diff",
    "log",
    "show",
    "ls-files",
    "branch",
    "rev-parse",
}
_FORBIDDEN_ARG_TOKENS = {"&&", "||", ";", "|", ">", ">>", "<"}
_FORBIDDEN_FIND_ARGS = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprintf", "-fls"}
_ALLOWED_CURL_FLAGS_WITH_VALUE = {"-H", "--header", "--max-time", "-A", "--user-agent"}
_ALLOWED_CURL_STANDALONE_FLAGS = {"-I", "--head", "-L", "--location", "-s", "--silent", "-S", "--show-error", "--fail", "--compressed"}
_FORBIDDEN_CURL_FLAGS = {
    "-o",
    "--output",
    "-O",
    "--remote-name",
    "-T",
    "--upload-file",
    "-F",
    "--form",
    "-d",
    "--data",
    "--data-raw",
    "--data-binary",
    "-X",
    "--request",
    "-K",
    "--config",
    "--proxy",
    "--proxy-header",
    "--trace",
    "--trace-ascii",
    "--stderr",
    "--output-dir",
    "--create-dirs",
}
_ALLOWED_GIT_CLONE_FLAGS_WITH_VALUE = {"--branch", "-b", "--depth"}
_ALLOWED_GIT_CLONE_STANDALONE_FLAGS = {"--single-branch"}

SUPPORTED_COMMAND_GROUPS = {
    "read_only_shell": sorted(_SIMPLE_READ_COMMANDS),
    "git_read_only": sorted(f"git {item}" for item in _READ_ONLY_GIT_SUBCOMMANDS),
    "network_read_only": ["curl <public-url>"],
    "controlled_write_exceptions": ["git clone https://<repo-url> [destination-under-output_dir/external_repos]"],
}
SUPPORTED_COMMAND_SUMMARY = (
    "Allowed commands: "
    + ", ".join(sorted(_SIMPLE_READ_COMMANDS))
    + "; git status/diff/log/show/ls-files/branch/rev-parse; "
      "curl for read-only public HTTP access; "
      "git clone only over https into output_dir/external_repos."
)
USAGE_HINT = (
    "Use `cwd` instead of `cd`. Shell chaining, redirection, command substitution, write/edit commands, "
    "and install commands are blocked."
)


class GuardedTerminalError(ValueError):
    """Raised when a terminal command fails validation before execution."""


def _error_payload(
    message: str,
    *,
    command: str,
    argv: Sequence[str],
    cwd: Optional[str],
) -> Dict[str, Any]:
    return {
        "success": False,
        "error": message,
        "command": command,
        "argv": list(argv),
        "cwd": str(cwd or ""),
        "supported_commands": dict(SUPPORTED_COMMAND_GROUPS),
        "supported_commands_summary": SUPPORTED_COMMAND_SUMMARY,
        "usage_hint": USAGE_HINT,
    }


def _truncate_output(text: str, *, max_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> Tuple[str, bool]:
    normalized = str(text or "")
    if len(normalized) <= max_chars:
        return normalized, False
    marker = f"\n...[truncated {len(normalized) - max_chars} chars]..."
    half = max(0, (max_chars - len(marker)) // 2)
    truncated = normalized[:half].rstrip() + marker + normalized[-half:].lstrip()
    return truncated, True


def _contains_forbidden_shell_syntax(argv: Sequence[str]) -> Optional[str]:
    for token in argv:
        text = str(token or "")
        if not text:
            continue
        if text in _FORBIDDEN_ARG_TOKENS:
            return f"Forbidden shell token '{text}' is not allowed."
        if "`" in text:
            return "Backticks are not allowed."
        if "$(" in text:
            return "Command substitution is not allowed."
        if "\n" in text or "\r" in text:
            return "Multiline commands are not allowed."
    return None


def _resolve_allowed_roots(output_dir: Optional[str]) -> List[Path]:
    roots = {Path.cwd().resolve(), Path.home().resolve()}
    if output_dir:
        try:
            roots.add(Path(output_dir).expanduser().resolve())
        except Exception:
            pass
    return sorted(roots)


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return path == root


def _resolve_cwd(cwd: Optional[str], *, output_dir: Optional[str]) -> Path:
    if cwd:
        candidate = Path(str(cwd).strip()).expanduser().resolve()
    else:
        candidate = Path.cwd().resolve()
    roots = _resolve_allowed_roots(output_dir)
    if not any(_is_within_root(candidate, root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise GuardedTerminalError(
            f"cwd '{candidate}' is outside the allowed roots. Allowed roots: {allowed}"
        )
    if not candidate.exists():
        raise GuardedTerminalError(f"cwd '{candidate}' does not exist.")
    if not candidate.is_dir():
        raise GuardedTerminalError(f"cwd '{candidate}' is not a directory.")
    return candidate


def _repo_name_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name.strip()
    if name.endswith(".git"):
        name = name[:-4]
    return name or "repo"


def _validate_find_args(argv: Sequence[str]) -> None:
    for token in argv[1:]:
        if str(token or "").strip() in _FORBIDDEN_FIND_ARGS:
            raise GuardedTerminalError(f"`find` argument '{token}' is not allowed.")


def _validate_curl_args(argv: Sequence[str]) -> None:
    urls: List[str] = []
    idx = 1
    while idx < len(argv):
        token = str(argv[idx] or "").strip()
        if not token:
            idx += 1
            continue
        if token in _FORBIDDEN_CURL_FLAGS:
            raise GuardedTerminalError(f"`curl` flag '{token}' is not allowed.")
        if token in _ALLOWED_CURL_FLAGS_WITH_VALUE:
            idx += 2
            continue
        if token in _ALLOWED_CURL_STANDALONE_FLAGS:
            idx += 1
            continue
        if token.startswith("-"):
            raise GuardedTerminalError(f"`curl` flag '{token}' is not allowed.")
        urls.append(token)
        idx += 1
    if len(urls) != 1:
        raise GuardedTerminalError("`curl` requires exactly one public http/https URL.")
    validation_error = _validate_public_url(urls[0], allowed_domains=[])
    if validation_error:
        raise GuardedTerminalError(validation_error)


def _normalize_git_clone_args(
    argv: Sequence[str],
    *,
    clone_root: Path,
) -> Tuple[List[str], Optional[Path]]:
    if len(argv) < 3:
        raise GuardedTerminalError("`git clone` requires a repository URL.")

    clone_root.mkdir(parents=True, exist_ok=True)
    idx = 2
    normalized_flags: List[str] = []
    repo_url = ""
    destination_raw = ""
    depth_present = False
    while idx < len(argv):
        token = str(argv[idx] or "").strip()
        if not token:
            idx += 1
            continue
        if token in _ALLOWED_GIT_CLONE_STANDALONE_FLAGS:
            normalized_flags.append(token)
            idx += 1
            continue
        if token in _ALLOWED_GIT_CLONE_FLAGS_WITH_VALUE:
            if idx + 1 >= len(argv):
                raise GuardedTerminalError(f"`git clone` flag '{token}' requires a value.")
            value = str(argv[idx + 1] or "").strip()
            if not value:
                raise GuardedTerminalError(f"`git clone` flag '{token}' requires a value.")
            if token == "--depth":
                depth_present = True
            normalized_flags.extend([token, value])
            idx += 2
            continue
        if token.startswith("-"):
            raise GuardedTerminalError(f"`git clone` flag '{token}' is not allowed.")
        if not repo_url:
            repo_url = token
            idx += 1
            continue
        if not destination_raw:
            destination_raw = token
            idx += 1
            continue
        raise GuardedTerminalError("`git clone` accepts only URL and optional destination.")

    if not repo_url:
        raise GuardedTerminalError("`git clone` requires a repository URL.")
    parsed = urlparse(repo_url)
    if parsed.scheme != "https":
        raise GuardedTerminalError("`git clone` only allows https repository URLs.")

    if destination_raw:
        destination = Path(destination_raw)
        if not destination.is_absolute():
            destination = clone_root / destination
        destination = destination.expanduser().resolve()
    else:
        destination = (clone_root / _repo_name_from_url(repo_url)).resolve()

    if not _is_within_root(destination, clone_root.resolve()):
        raise GuardedTerminalError(
            f"`git clone` destination '{destination}' must stay under '{clone_root.resolve()}'."
        )

    normalized_argv = ["git", "clone"]
    if not depth_present:
        normalized_argv.extend(["--depth", "1"])
    normalized_argv.extend(normalized_flags)
    normalized_argv.extend([repo_url, str(destination)])
    return normalized_argv, destination


def _validate_and_normalize_argv(
    argv: Sequence[str],
    *,
    clone_root: Path,
) -> Tuple[List[str], Optional[Path]]:
    if not argv:
        raise GuardedTerminalError("Command must not be empty.")
    command = str(argv[0] or "").strip()
    if not command:
        raise GuardedTerminalError("Command must not be empty.")

    forbidden = _contains_forbidden_shell_syntax(argv)
    if forbidden:
        raise GuardedTerminalError(forbidden)

    if command in _SIMPLE_READ_COMMANDS:
        normalized = [str(item) for item in argv]
        if command == "find":
            _validate_find_args(normalized)
        return normalized, None

    if command == "curl":
        normalized = [str(item) for item in argv]
        _validate_curl_args(normalized)
        return normalized, None

    if command == "git":
        if len(argv) < 2:
            raise GuardedTerminalError("`git` requires a subcommand.")
        subcommand = str(argv[1] or "").strip()
        if subcommand in _READ_ONLY_GIT_SUBCOMMANDS:
            return [str(item) for item in argv], None
        if subcommand == "clone":
            return _normalize_git_clone_args([str(item) for item in argv], clone_root=clone_root)
        raise GuardedTerminalError(f"`git {subcommand}` is not allowed.")

    raise GuardedTerminalError(f"Command '{command}' is not in the terminal whitelist.")


def execute_guarded_terminal_command(
    command: str,
    *,
    cwd: Optional[str] = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    output_dir: Optional[str] = None,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> Dict[str, Any]:
    normalized_command = str(command or "").strip()
    if not normalized_command:
        return _error_payload("command must not be empty", command="", argv=[], cwd=cwd)

    try:
        argv = shlex.split(normalized_command, posix=True)
    except Exception as exc:
        return _error_payload(
            f"failed to parse command: {exc}",
            command=normalized_command,
            argv=[],
            cwd=cwd,
        )

    try:
        timeout_value = float(timeout_sec)
    except Exception:
        timeout_value = DEFAULT_TIMEOUT_SEC
    if timeout_value <= 0:
        timeout_value = DEFAULT_TIMEOUT_SEC

    try:
        exec_cwd = _resolve_cwd(cwd, output_dir=output_dir)
        clone_root = ((Path(output_dir).expanduser().resolve() if output_dir else Path.cwd().resolve()) / "external_repos").resolve()
        normalized_argv, clone_destination = _validate_and_normalize_argv(argv, clone_root=clone_root)
    except GuardedTerminalError as exc:
        return _error_payload(
            str(exc),
            command=normalized_command,
            argv=argv,
            cwd=cwd,
        )

    env = dict(os.environ)
    env.setdefault("GIT_PAGER", "cat")
    env.setdefault("PAGER", "cat")
    env.setdefault("LESS", "FRX")

    started_at = time.perf_counter()
    proc = subprocess.Popen(
        normalized_argv,
        cwd=str(exec_cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=env,
    )
    timed_out = False
    try:
        stdout_text, stderr_text = proc.communicate(timeout=timeout_value)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        stdout_text, stderr_text = proc.communicate()

    duration_sec = max(0.0, time.perf_counter() - started_at)
    stdout_rendered, stdout_truncated = _truncate_output(stdout_text, max_chars=max_output_chars)
    stderr_rendered, stderr_truncated = _truncate_output(stderr_text, max_chars=max_output_chars)
    exit_code = proc.returncode if proc.returncode is not None else -1
    success = (not timed_out) and exit_code == 0

    payload: Dict[str, Any] = {
        "success": success,
        "command": normalized_command,
        "argv": normalized_argv,
        "cwd": str(exec_cwd),
        "timeout_sec": timeout_value,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "stdout": stdout_rendered,
        "stderr": stderr_rendered,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "duration_sec": round(duration_sec, 3),
        "tool_policy": "guarded_terminal_whitelist",
        "supported_commands": dict(SUPPORTED_COMMAND_GROUPS),
        "supported_commands_summary": SUPPORTED_COMMAND_SUMMARY,
        "usage_hint": USAGE_HINT,
    }
    if clone_destination is not None:
        payload["clone_destination"] = str(clone_destination)
    if timed_out:
        payload["error"] = f"Command timed out after {timeout_value} seconds."
    elif exit_code != 0:
        payload["error"] = f"Command exited with code {exit_code}."
    return payload
