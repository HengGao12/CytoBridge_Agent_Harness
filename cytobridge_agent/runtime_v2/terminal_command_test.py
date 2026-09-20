from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cytobridge_agent.runtime_v2.state import ensure_runtime_v2_state
from cytobridge_agent.runtime_v2.tool_registry import SingleAgentTools
from cytobridge_agent.tools.guarded_terminal import (
    TERMINAL_TOOL_NAME,
    execute_guarded_terminal_command,
)


class DummyLLM:
    pass


class _FakePopen:
    last_argv = None
    last_cwd = None

    def __init__(self, argv, cwd=None, stdout=None, stderr=None, text=None, start_new_session=None, env=None):  # noqa: ANN001
        del stdout, stderr, text, start_new_session, env
        self.__class__.last_argv = list(argv)
        self.__class__.last_cwd = cwd
        self.returncode = 0
        self.pid = 4242

    def communicate(self, timeout=None):  # noqa: ANN001
        del timeout
        return ("cloned ok\n", "")

    def kill(self):
        return None


class TerminalCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state = ensure_runtime_v2_state(
            {
                "session_id": "terminal-command-test",
                "output_dir": self.tmpdir.name,
            }
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_pwd_runs_with_explicit_cwd(self) -> None:
        result = execute_guarded_terminal_command(
            "pwd",
            cwd=self.tmpdir.name,
            output_dir=self.tmpdir.name,
        )
        self.assertTrue(result["success"])
        self.assertEqual(Path(result["cwd"]).resolve(), Path(self.tmpdir.name).resolve())
        self.assertIn(self.tmpdir.name, result["stdout"])

    def test_rejects_non_whitelisted_command(self) -> None:
        result = execute_guarded_terminal_command(
            "rm -rf /tmp/whatever",
            cwd=self.tmpdir.name,
            output_dir=self.tmpdir.name,
        )
        self.assertFalse(result["success"])
        self.assertIn("not in the terminal whitelist", result["error"])
        self.assertIn("read_only_shell", result["supported_commands"])
        self.assertTrue(result["supported_commands_summary"])
        self.assertTrue(result["usage_hint"])

    def test_rejects_find_exec_style_tokens(self) -> None:
        result = execute_guarded_terminal_command(
            "find . -exec ls {} ;",
            cwd=self.tmpdir.name,
            output_dir=self.tmpdir.name,
        )
        self.assertFalse(result["success"])
        self.assertIn("not allowed", result["error"].lower())

    def test_rejects_curl_output_redirection_flags(self) -> None:
        result = execute_guarded_terminal_command(
            "curl -o result.txt https://example.com",
            cwd=self.tmpdir.name,
            output_dir=self.tmpdir.name,
        )
        self.assertFalse(result["success"])
        self.assertIn("curl", result["error"].lower())
        self.assertIn("not allowed", result["error"].lower())

    def test_git_clone_is_redirected_under_controlled_root(self) -> None:
        with patch("cytobridge_agent.tools.guarded_terminal.subprocess.Popen", _FakePopen):
            result = execute_guarded_terminal_command(
                "git clone https://github.com/example/project.git",
                cwd=self.tmpdir.name,
                output_dir=self.tmpdir.name,
            )

        self.assertTrue(result["success"])
        self.assertIn("--depth", _FakePopen.last_argv)
        self.assertEqual(_FakePopen.last_argv[0:2], ["git", "clone"])
        clone_destination = Path(result["clone_destination"]).resolve()
        expected_root = (Path(self.tmpdir.name) / "external_repos").resolve()
        self.assertTrue(str(clone_destination).startswith(str(expected_root)))

    def test_planner_and_general_subagent_get_terminal_tool_but_specialized_evaluators_do_not(self) -> None:
        planner_tools = SingleAgentTools(DummyLLM(), self.state, agent_role="planner", agent_id="planner")
        planner_names = {tool.name for tool in planner_tools.get_tools()}
        self.assertIn(TERMINAL_TOOL_NAME, planner_names)

        general_subagent_tools = SingleAgentTools(
            DummyLLM(),
            dict(self.state),
            agent_role="subagent",
            agent_id="session-test:subagent:1",
            parent_agent_id="planner",
            subagent_type="general",
        )
        general_subagent_names = {tool.name for tool in general_subagent_tools.get_tools()}
        self.assertIn(TERMINAL_TOOL_NAME, general_subagent_names)

        proposal_evaluator_tools = SingleAgentTools(
            DummyLLM(),
            dict(self.state),
            agent_role="subagent",
            agent_id="session-test:subagent:2",
            parent_agent_id="planner",
            subagent_type="proposal_evaluator",
        )
        proposal_evaluator_names = {tool.name for tool in proposal_evaluator_tools.get_tools()}
        self.assertNotIn(TERMINAL_TOOL_NAME, proposal_evaluator_names)

        idea_evaluator_tools = SingleAgentTools(
            DummyLLM(),
            dict(self.state),
            agent_role="subagent",
            agent_id="session-test:subagent:3",
            parent_agent_id="planner",
            subagent_type="idea_evaluator",
        )
        idea_evaluator_names = {tool.name for tool in idea_evaluator_tools.get_tools()}
        self.assertNotIn(TERMINAL_TOOL_NAME, idea_evaluator_names)

    def test_system_prompt_tools_terminal_command_uses_guarded_backend(self) -> None:
        tools = SingleAgentTools(DummyLLM(), self.state, agent_role="planner", agent_id="planner")
        with patch(
            "cytobridge_agent.runtime_v2.tool_registry.execute_guarded_terminal_command",
            return_value={
                "success": True,
                "command": "pwd",
                "argv": ["pwd"],
                "cwd": self.tmpdir.name,
                "timeout_sec": 7.5,
                "timed_out": False,
                "exit_code": 0,
                "stdout": self.tmpdir.name,
                "stderr": "",
                "stdout_truncated": False,
                "stderr_truncated": False,
                "duration_sec": 0.01,
                "tool_policy": "guarded_terminal_whitelist",
            },
        ) as mock_run:
            result = tools.run_terminal_command("pwd", cwd=self.tmpdir.name, timeout_sec=7.5)

        self.assertTrue(result["success"])
        mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
