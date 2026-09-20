from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from cytobridge_agent.cli import _preprocess_argv, _resolve_cli_llm_runtime_args, build_parser


class CLIModesTest(unittest.TestCase):
    def test_empty_invocation_defaults_to_interactive(self) -> None:
        args = build_parser().parse_args(_preprocess_argv([]))
        self.assertEqual(args.command, "interactive")

    def test_prompt_invocation_maps_to_exec(self) -> None:
        args = build_parser().parse_args(_preprocess_argv(["analyze", "this", "dataset"]))
        self.assertEqual(args.command, "exec")
        self.assertEqual(args.prompt, ["analyze", "this", "dataset"])

    def test_global_options_before_prompt_map_to_exec(self) -> None:
        args = build_parser().parse_args(
            _preprocess_argv(["--llm-provider", "xiaomi", "--llm-model", "mimo-v2.5-pro", "run", "analysis"])
        )
        self.assertEqual(args.command, "run")

        args = build_parser().parse_args(
            _preprocess_argv(["--llm-provider", "xiaomi", "--llm-model", "mimo-v2.5-pro", "analyze"])
        )
        self.assertEqual(args.command, "exec")
        self.assertEqual(args.llm_provider, "xiaomi")
        self.assertEqual(args.llm_model, "mimo-v2.5-pro")
        self.assertEqual(args.prompt, ["analyze"])

    def test_session_subcommands_parse(self) -> None:
        args = build_parser().parse_args(_preprocess_argv(["session", "show", "abc123", "--last", "4"]))
        self.assertEqual(args.command, "session")
        self.assertEqual(args.session_command, "show")
        self.assertEqual(args.session_id, "abc123")
        self.assertEqual(args.last, 4)

    def test_doctor_and_perf_parse(self) -> None:
        args = build_parser().parse_args(_preprocess_argv(["doctor", "--json"]))
        self.assertEqual(args.command, "doctor")
        self.assertTrue(args.json)

        args = build_parser().parse_args(_preprocess_argv(["perf", "report", "--session", "abc123"]))
        self.assertEqual(args.command, "perf")
        self.assertEqual(args.perf_command, "report")
        self.assertEqual(args.session, "abc123")

        args = build_parser().parse_args(_preprocess_argv(["jobs", "--active", "--json"]))
        self.assertEqual(args.command, "jobs")
        self.assertTrue(args.active)
        self.assertTrue(args.json)

    def test_resume_without_id_opens_picker_mode(self) -> None:
        args = build_parser().parse_args(_preprocess_argv(["resume"]))
        self.assertEqual(args.command, "resume")
        self.assertIsNone(args.session_id)
        self.assertFalse(args.last)

    def test_continue_alias_parses_as_command(self) -> None:
        args = build_parser().parse_args(_preprocess_argv(["continue"]))
        self.assertEqual(args.command, "continue")

        args = build_parser().parse_args(_preprocess_argv(["c"]))
        self.assertEqual(args.command, "continue")

    def test_tui_parse(self) -> None:
        args = build_parser().parse_args(_preprocess_argv(["tui", "--last", "--grep", "packer"]))
        self.assertEqual(args.command, "tui")
        self.assertTrue(args.last)
        self.assertEqual(args.grep, "packer")

    def test_provider_env_key_overrides_stale_saved_provider_key(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            _preprocess_argv(
                [
                    "--llm-provider",
                    "xiaomi",
                    "--llm-auth-mode",
                    "api_key",
                    "exec",
                    "ping",
                ]
            )
        )
        old_env = os.environ.get("XIAOMI_API_KEY")
        os.environ["XIAOMI_API_KEY"] = "env-xiaomi-key"
        try:
            with patch(
                "cytobridge_agent.cli.get_saved_config",
                return_value={
                    "llm_provider": "xiaomi",
                    "provider_api_keys": {"xiaomi": "stale-saved-key"},
                },
            ):
                runtime = _resolve_cli_llm_runtime_args(args)
            self.assertEqual(runtime["api_key"], "env-xiaomi-key")
        finally:
            if old_env is None:
                os.environ.pop("XIAOMI_API_KEY", None)
            else:
                os.environ["XIAOMI_API_KEY"] = old_env


if __name__ == "__main__":
    unittest.main()
