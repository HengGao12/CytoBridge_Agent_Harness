from __future__ import annotations

import copy
from datetime import datetime
import re
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from langchain_core.load import dumpd

from .agent_types import (
    SPAWN_SUBAGENT_TOOL_NAME,
    SUBMIT_IMPLEMENTATION_REVIEW_TOOL_NAME,
    SUBMIT_PROPOSAL_REVIEW_TOOL_NAME,
    SUBMIT_RESEARCH_IDEA_REVIEW_TOOL_NAME,
    SUBMIT_SUBAGENT_RESULT_TOOL_NAME,
    default_subagent_tool_policy,
    resolve_subagent_type,
)
from .paper_review_freshness import build_paper_review_manifest, sync_paper_reviewer_gate_from_review
from .prompt_builder import PromptBuilder
from .resume import default_checkpoint_path


_RETRYABLE_SUBAGENT_ERROR_TERMS = (
    "file is not a database",
    "database disk image is malformed",
    "sqlite",
    "databaseerror",
    "operationalerror",
    "checkpoint",
    "checkpointer",
)


def is_retryable_subagent_infrastructure_error(error: Any) -> bool:
    text = str(error or "").strip().lower()
    if not text:
        return False
    return any(term in text for term in _RETRYABLE_SUBAGENT_ERROR_TERMS)


class SubagentRuntimeManager:
    def __init__(
        self,
        llm: Any,
        state: Dict[str, Any],
        *,
        parent_agent_id: str = "planner",
        event_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        checkpoint_callback: Optional[Callable[[], None]] = None,
        agent_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.llm = llm
        self.state = state
        self.parent_agent_id = str(parent_agent_id or "planner").strip() or "planner"
        self.event_sink = event_sink
        self.checkpoint_callback = checkpoint_callback
        self.agent_factory = agent_factory
        self._agents: Dict[str, Any] = {}
        self._archived_histories: Dict[str, List[Any]] = {}
        self._archived_snapshots: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat()

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.event_sink:
            return
        try:
            self.event_sink(event_type, payload)
        except Exception:
            pass

    def _next_subagent_id(self) -> str:
        session_id = str(self.state.get("session_id") or "runtime-v2-default").strip() or "runtime-v2-default"
        prefix = f"{session_id}:subagent:"
        counters = [int(self.state.get("subagent_counter", 0) or 0)]
        known_ids = set((self._registry() or {}).keys())
        known_ids.update(str((entry or {}).get("subagent_id") or "") for entry in self._run_log())
        known_ids.update(self._archived_histories.keys())
        known_ids.update(self._archived_snapshots.keys())
        known_ids.update(self._agents.keys())
        for subagent_id in known_ids:
            if not str(subagent_id).startswith(prefix):
                continue
            suffix = str(subagent_id)[len(prefix):]
            match = re.match(r"^(\d+)", suffix)
            if match:
                counters.append(int(match.group(1)))
        counter = max(counters) + 1
        self.state["subagent_counter"] = counter
        return f"{session_id}:subagent:{counter}-{uuid4().hex[:8]}"

    def _checkpoint_parent_state(self) -> None:
        if not self.checkpoint_callback:
            return
        try:
            self.checkpoint_callback()
        except Exception:
            pass

    def _registry(self) -> Dict[str, Any]:
        registry = self.state.setdefault("subagent_registry", {})
        if not isinstance(registry, dict):
            registry = {}
            self.state["subagent_registry"] = registry
        return registry

    def _run_log(self) -> List[Dict[str, Any]]:
        log = self.state.setdefault("subagent_run_log", [])
        if not isinstance(log, list):
            log = []
            self.state["subagent_run_log"] = log
        return log

    def _merge_tool_policy(
        self,
        subagent_type: str = "general",
        override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved_type = resolve_subagent_type(subagent_type).name
        policy = default_subagent_tool_policy(resolved_type)
        candidate = dict(override or {})
        disallowed = {
            str(item).strip()
            for item in (policy.get("disallowed_tools") or [])
            if str(item).strip()
        }
        disallowed.update(str(item).strip() for item in (candidate.get("disallowed_tools") or []) if str(item).strip())
        allowed = {
            str(item).strip()
            for item in (policy.get("allowed_tools") or [])
            if str(item).strip()
        }
        allowed.update(str(item).strip() for item in (candidate.get("allowed_tools") or []) if str(item).strip())
        for key, value in candidate.items():
            if key in {"allowed_tools", "disallowed_tools"}:
                continue
            policy[key] = value
        policy["subagent_type"] = str(candidate.get("subagent_type") or resolved_type).strip().lower() or resolved_type
        policy["allowed_tools"] = sorted(allowed)
        policy["disallowed_tools"] = sorted(disallowed)
        return policy

    def _snapshot_payload(self, relevant_paths: List[str]) -> Dict[str, Any]:
        final_cfg = self.state.get("final_config") or {}
        final_model_path = final_cfg.get("path") if isinstance(final_cfg, dict) else ""
        return {
            "workflow_phase": self.state.get("workflow_phase") or "intake",
            "input_path": self.state.get("input_path") or "",
            "converted_path": self.state.get("converted_path") or "",
            "preprocessed_path": self.state.get("preprocessed_path") or "",
            "output_dir": self.state.get("output_dir") or "",
            "time_key": self.state.get("time_key") or "",
            "label_key": self.state.get("label_key") or "",
            "final_model_path": final_model_path or "",
            "relevant_paths": list(relevant_paths),
        }

    def _build_brief(
        self,
        *,
        subagent_type: str,
        task: str,
        success_criteria: List[str],
        context_notes: str,
        relevant_paths: List[str],
    ) -> str:
        return PromptBuilder(self.state).build_subagent_brief(
            subagent_type=subagent_type,
            task=task,
            success_criteria=success_criteria,
            context_notes=context_notes,
            relevant_paths=relevant_paths,
        )

    def _create_subagent(
        self,
        *,
        subagent_id: str,
        run_uid: str,
        subagent_type: str,
        tool_policy: Dict[str, Any],
        require_structured_result: bool,
        event_callback: Callable[[str, Dict[str, Any]], None],
    ) -> Any:
        factory = self.agent_factory
        if factory is None:
            from .agent import CytoBridgeAgent

            factory = CytoBridgeAgent

        sub_state = copy.deepcopy(self.state)
        sub_state["planner_phase"] = "working"
        sub_state["planner_need"] = {}
        sub_state["subagent_registry"] = {}
        sub_state["subagent_run_log"] = []
        sub_state["subagent_counter"] = 0
        checkpoint_identity = f"{subagent_id}:run:{run_uid}"
        safe_checkpoint_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint_identity)[:180] or "subagent"
        parent_checkpoint_path = default_checkpoint_path(self.state)
        sub_state["langgraph_checkpoint_path"] = str(
            parent_checkpoint_path.parent / "subagents" / f"{safe_checkpoint_name}.sqlite"
        )
        return factory(
            self.llm,
            sub_state,
            checkpoint_callback=None,
            event_callback=event_callback,
            agent_role="subagent",
            agent_id=subagent_id,
            parent_agent_id=self.parent_agent_id,
            subagent_type=subagent_type,
            tool_policy=tool_policy,
            require_structured_result=require_structured_result,
            thread_id=checkpoint_identity,
            checkpoint_ns=checkpoint_identity,
        )

    def _relay_subagent_event(self, subagent_id: str) -> Callable[[str, Dict[str, Any]], None]:
        def relay(event_type: str, payload: Dict[str, Any]) -> None:
            data = dict(payload or {})
            data["subagent_id"] = subagent_id
            data.setdefault("parent_agent_id", self.parent_agent_id)
            if event_type == "agent_thought":
                self._emit("subagent_thought", data)
                return
            self._emit(event_type, data)

        return relay

    def _update_registry_entry(self, subagent_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        registry = self._registry()
        merged = dict(registry.get(subagent_id) or {})
        merged.update(dict(updates or {}))
        merged["updated_at"] = self._now_iso()
        registry[subagent_id] = merged
        return merged

    def _archive_agent(
        self,
        subagent_id: str,
        agent: Any,
        submitted_result: Optional[Dict[str, Any]],
    ) -> None:
        history = [dumpd(message) for message in list(getattr(agent, "chat_history", []) or [])]
        snapshot = {
            **dict(agent.export_runtime_snapshot()),
            "agent_role": "subagent",
            "agent_id": subagent_id,
            "parent_agent_id": str((self._registry().get(subagent_id) or {}).get("parent_agent_id") or self.parent_agent_id),
            "tool_policy": dict(getattr(agent, "tool_policy", {}) or {}),
            "require_structured_result": bool(getattr(agent, "require_structured_result", False)),
            "submitted_result": dict(submitted_result or {}),
        }
        self._archived_histories[subagent_id] = history
        self._archived_snapshots[subagent_id] = snapshot

    def _finalize_result(
        self,
        *,
        subagent_id: str,
        subagent_type: str,
        tool_policy: Dict[str, Any],
        run_result: Optional[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        agent = self._agents.get(subagent_id)
        submitted = None
        if agent is not None:
            submitted = agent.tools_handler.peek_subagent_result(copy_result=True)

        if error:
            final = {
                "subagent_id": subagent_id,
                "status": "failed",
                "summary": error,
                "findings": [],
                "artifact_refs": [],
                "proposed_state_updates": {},
                "needs_input_question": "",
            }
        elif submitted:
            final = dict(submitted)
            final["subagent_id"] = subagent_id
        else:
            last_content = str((run_result or {}).get("content") or "").strip()
            final = {
                "subagent_id": subagent_id,
                "status": "failed",
                "summary": "Subagent exited without calling submit_subagent_result(...).",
                "findings": [last_content] if last_content else [],
                "artifact_refs": [],
                "proposed_state_updates": {},
                "needs_input_question": "",
            }

        final_status = str(final.get("status") or "failed").strip().lower()
        if final_status not in {"completed", "needs_input", "failed"}:
            final_status = "failed"
            final["status"] = "failed"

        registry_entry = self._registry().get(subagent_id) or {}
        if subagent_type == "paper_reviewer" and final_status == "completed":
            proposed = final.get("proposed_state_updates")
            review = proposed.get("paper_review") if isinstance(proposed, dict) else None
            if isinstance(review, dict):
                relevant_paths = registry_entry.get("relevant_paths") or []
                manifest = build_paper_review_manifest(relevant_paths)
                if int(manifest.get("file_count") or 0) > 0:
                    review["reviewed_file_hashes"] = manifest
                    gate_sync = sync_paper_reviewer_gate_from_review(review, relevant_paths)
                    review["paper_reviewer_gate_sync"] = gate_sync
                    self._emit(
                        "paper_review_hash_manifest_recorded",
                        {
                            "subagent_id": subagent_id,
                            "file_count": int(manifest.get("file_count") or 0),
                            "digest": str(manifest.get("digest") or ""),
                        },
                    )
                    if gate_sync.get("status") == "synced":
                        self._emit(
                            "paper_reviewer_gate_synced",
                            {
                                "subagent_id": subagent_id,
                                "paths": list(gate_sync.get("paths") or []),
                                "manifest_digest": str(gate_sync.get("manifest_digest") or ""),
                            },
                        )

        history_length = 0
        if agent is not None:
            history_length = len(getattr(agent, "chat_history", []) or [])
            self._archive_agent(subagent_id, agent, submitted)
            self._agents.pop(subagent_id, None)

        self._update_registry_entry(
            subagent_id,
            {
                "status": final_status,
                "subagent_type": subagent_type,
                "summary": str(final.get("summary") or "").strip(),
                "tool_policy": dict(tool_policy),
                "result_schema_submitted": submitted is not None,
                "last_result": dict(final),
                "history_length": history_length,
                "finished_at": self._now_iso(),
            },
        )
        self._run_log().append(
            {
                "subagent_id": subagent_id,
                "subagent_type": subagent_type,
                "status": final_status,
                "summary": str(final.get("summary") or "").strip(),
                "finished_at": self._now_iso(),
            }
        )

        if final_status == "completed":
            self._emit("subagent_completed", dict(final))
        elif final_status == "needs_input":
            self._emit("subagent_needs_input", dict(final))
        else:
            self._emit("subagent_failed", dict(final))

        if self.checkpoint_callback:
            try:
                self.checkpoint_callback()
            except Exception:
                pass
        return final

    def run_sync(
        self,
        *,
        subagent_type: str = "general",
        task: str,
        success_criteria: List[str],
        context_notes: str = "",
        relevant_paths: Optional[List[str]] = None,
        tool_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved_type = resolve_subagent_type(subagent_type)
        success_criteria = [str(item).strip() for item in (success_criteria or []) if str(item).strip()]
        relevant = [str(item).strip() for item in (relevant_paths or []) if str(item).strip()]
        policy = self._merge_tool_policy(resolved_type.name, tool_policy)
        last_failure: Optional[Dict[str, Any]] = None
        for attempt_index in range(2):
            subagent_id = self._next_subagent_id()
            run_uid = uuid4().hex[:12]
            checkpoint_identity = f"{subagent_id}:run:{run_uid}"
            attempt_context_notes = str(context_notes or "").strip()
            if attempt_index > 0 and last_failure:
                attempt_context_notes = (
                    f"{attempt_context_notes}\n\n"
                    "Previous reviewer/subagent attempt failed before producing a review because of runtime "
                    "infrastructure. Retry from a fresh isolated checkpoint and perform the requested review "
                    "normally; do not treat the previous failure as evidence about the artifact."
                ).strip()

            self._update_registry_entry(
                subagent_id,
                {
                    "subagent_id": subagent_id,
                    "run_uid": run_uid,
                    "thread_id": checkpoint_identity,
                    "checkpoint_ns": checkpoint_identity,
                    "subagent_type": resolved_type.name,
                    "parent_agent_id": self.parent_agent_id,
                    "status": "running",
                    "task": str(task or "").strip(),
                    "success_criteria": list(success_criteria),
                    "context_notes": attempt_context_notes,
                    "relevant_paths": list(relevant),
                    "tool_policy": dict(policy),
                    "result_schema_submitted": False,
                    "created_at": self._now_iso(),
                    "workflow_snapshot": self._snapshot_payload(relevant),
                    "retry_of_subagent_id": str(last_failure.get("subagent_id") or "") if last_failure else "",
                    "retry_attempt_index": attempt_index,
                },
            )
            self._checkpoint_parent_state()
            self._emit(
                "subagent_started",
                {
                    "subagent_id": subagent_id,
                    "run_uid": run_uid,
                    "subagent_type": resolved_type.name,
                    "parent_agent_id": self.parent_agent_id,
                    "task": str(task or "").strip(),
                    "retry_of_subagent_id": str(last_failure.get("subagent_id") or "") if last_failure else "",
                    "retry_attempt_index": attempt_index,
                },
            )

            agent = self._create_subagent(
                subagent_id=subagent_id,
                run_uid=run_uid,
                subagent_type=resolved_type.name,
                tool_policy=policy,
                require_structured_result=True,
                event_callback=self._relay_subagent_event(subagent_id),
            )
            agent.state["user_goal"] = {"raw_question": str(task or "").strip()}
            self._agents[subagent_id] = agent

            try:
                run_result = agent.run(
                    user_instruction=self._build_brief(
                        subagent_type=resolved_type.name,
                        task=str(task or "").strip(),
                        success_criteria=success_criteria,
                        context_notes=attempt_context_notes,
                        relevant_paths=relevant,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                error = f"Subagent execution failed: {exc}"
                failed = self._finalize_result(
                    subagent_id=subagent_id,
                    subagent_type=resolved_type.name,
                    tool_policy=policy,
                    run_result=None,
                    error=error,
                )
                if attempt_index == 0 and is_retryable_subagent_infrastructure_error(error):
                    last_failure = dict(failed)
                    self._emit(
                        "subagent_infrastructure_retry",
                        {
                            "failed_subagent_id": subagent_id,
                            "subagent_type": resolved_type.name,
                            "error": error,
                            "next_attempt_index": 1,
                        },
                    )
                    continue
                return failed

            return self._finalize_result(
                subagent_id=subagent_id,
                subagent_type=resolved_type.name,
                tool_policy=policy,
                run_result=run_result,
            )

        if last_failure:
            return last_failure
        return {
            "subagent_id": "",
            "status": "failed",
            "summary": "Subagent failed before starting.",
            "findings": [],
            "artifact_refs": [],
            "proposed_state_updates": {},
            "needs_input_question": "",
        }

    def collect_histories(self) -> Dict[str, List[Any]]:
        histories: Dict[str, List[Any]] = dict(self._archived_histories)
        for subagent_id, agent in self._agents.items():
            histories[subagent_id] = [dumpd(message) for message in list(getattr(agent, "chat_history", []) or [])]
        return histories

    def collect_snapshots(self) -> Dict[str, Dict[str, Any]]:
        snapshots: Dict[str, Dict[str, Any]] = {
            key: dict(value)
            for key, value in self._archived_snapshots.items()
        }
        for subagent_id, agent in self._agents.items():
            snapshots[subagent_id] = {
                **dict(agent.export_runtime_snapshot()),
                "agent_role": "subagent",
                "agent_id": subagent_id,
                "subagent_type": str(getattr(agent, "subagent_type", "") or "general"),
                "parent_agent_id": str((self._registry().get(subagent_id) or {}).get("parent_agent_id") or self.parent_agent_id),
                "tool_policy": dict(getattr(agent, "tool_policy", {}) or {}),
                "require_structured_result": bool(getattr(agent, "require_structured_result", False)),
                "submitted_result": agent.tools_handler.peek_subagent_result(copy_result=True),
            }
        return snapshots

    def restore_from_snapshot(
        self,
        agent_snapshots: Optional[Dict[str, Any]],
        agent_histories: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        restored: Dict[str, str] = {}
        reasons: List[str] = []
        self._agents = {}
        self._archived_histories = {}
        self._archived_snapshots = {}
        snapshots = dict(agent_snapshots or {})
        histories = dict(agent_histories or {})
        for subagent_id, snapshot in snapshots.items():
            if str(subagent_id) == "planner":
                continue
            data = dict(snapshot or {})
            subagent_type = str(
                data.get("subagent_type")
                or (data.get("tool_policy") or {}).get("subagent_type")
                or "general"
            ).strip().lower() or "general"
            tool_policy = self._merge_tool_policy(subagent_type, data.get("tool_policy"))
            try:
                submitted_result = data.get("submitted_result")
                history = list(histories.get(str(subagent_id)) or data.get("chat_history") or [])
                self._archived_histories[str(subagent_id)] = history
                self._archived_snapshots[str(subagent_id)] = {
                    **data,
                    "subagent_type": subagent_type,
                    "tool_policy": dict(tool_policy),
                }
                restored[str(subagent_id)] = "metadata_only"
                self._update_registry_entry(
                    str(subagent_id),
                    {
                        "subagent_id": str(subagent_id),
                        "subagent_type": subagent_type,
                        "parent_agent_id": str(data.get("parent_agent_id") or self.parent_agent_id),
                        "status": str(
                            ((self._registry().get(str(subagent_id)) or {}).get("status"))
                            or ((submitted_result or {}).get("status"))
                            or "completed"
                        ),
                        "tool_policy": dict(tool_policy),
                        "restored_from_snapshot": True,
                        "resume_mode": "metadata_only",
                        "result_schema_submitted": bool(submitted_result),
                        "last_result": dict(submitted_result or {}),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"{subagent_id}: {exc}")
        return {"restored": restored, "degraded": bool(reasons), "reasons": reasons}
