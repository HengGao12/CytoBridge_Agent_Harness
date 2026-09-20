from __future__ import annotations

from argparse import Namespace

import cytobridge_agent.cli as cli


class _FakeSession:
    session_id = "run123"


class _FakeController:
    def __init__(self) -> None:
        self.opened = None
        self.ran = None

    def new_session(self, **kwargs):
        self.opened = kwargs
        return _FakeSession()

    def run_turn(self, prompt, attachments=None):
        self.ran = prompt
        return Namespace(status="success", session_id="run123", response="ok")


def _args(**overrides):
    base = {
        "input": None,
        "output": None,
        "question": "analyze",
        "time_key": "day",
        "label_key": "cell_type",
        "analyses": ["trajectory_fate"],
        "gene_sets_gmt": None,
        "device": "cpu",
        "max_retries": 3,
        "pilot_epochs": 100,
        "pilot_max_cells": 500,
        "seed": 42,
        "disable_llm_overrides": False,
        "disable_multimodal": False,
        "report_format": "html",
        "checkpoint": False,
        "resume": None,
        "debug": False,
        "verbose": False,
    }
    base.update(overrides)
    return Namespace(**base)


def test_run_uses_unified_runtime_core(monkeypatch, capsys):
    controller = _FakeController()
    monkeypatch.setattr(cli, "_controller_from_args", lambda args: controller)

    rc = cli.cmd_run(_args())

    out = capsys.readouterr().out
    assert rc == 0
    assert controller.opened["user_goal"]["raw_question"].startswith("Run a complete CytoBridge")
    assert "Scientific question: analyze" in controller.ran
    assert "Session: run123" in out


def test_run_rejects_removed_checkpoint_resume(caplog):
    rc = cli.cmd_run(_args(checkpoint=True))

    assert rc == 1
    assert "Legacy pipeline checkpoint/resume has been removed" in caplog.text

