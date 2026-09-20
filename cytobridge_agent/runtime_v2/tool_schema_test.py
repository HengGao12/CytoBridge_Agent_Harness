from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

from langchain_core.utils.function_calling import convert_to_openai_tool

from cytobridge_agent.runtime_v2 import tool_registry


class ToolSchemaTests(unittest.TestCase):
    def test_all_runtime_tools_declare_runtime_contract_metadata(self) -> None:
        tools = tool_registry.SingleAgentTools(llm=None, state={}, agent_role="planner").get_tools()
        self.assertGreater(len(tools), 0)
        required_keys = {
            "cytobridge_contract_version",
            "cytobridge_tool_name",
            "cytobridge_isolation",
            "cytobridge_timeout_sec",
            "cytobridge_isolation_timeout",
            "cytobridge_timeout_enforced",
            "cytobridge_mutates_state",
            "cytobridge_reads_adata",
            "cytobridge_writes_artifacts",
            "cytobridge_may_use_gpu",
        }
        allowed_isolation = {"in_process", "process", "managed_subprocess"}
        missing: list[str] = []
        invalid: list[str] = []
        for tool in tools:
            name = str(getattr(tool, "name", "") or "")
            metadata = dict(getattr(tool, "metadata", {}) or {})
            absent = sorted(required_keys - set(metadata))
            if absent:
                missing.append(f"{name}: {absent}")
                continue
            if metadata["cytobridge_tool_name"] != name:
                invalid.append(f"{name}: metadata name mismatch")
            if metadata["cytobridge_isolation"] not in allowed_isolation:
                invalid.append(f"{name}: invalid isolation={metadata['cytobridge_isolation']}")
            if metadata["cytobridge_isolation"] in {"process", "managed_subprocess"}:
                try:
                    timeout = float(metadata["cytobridge_timeout_sec"])
                except Exception:
                    invalid.append(f"{name}: enforced timeout is not numeric")
                    continue
                if timeout <= 0:
                    invalid.append(f"{name}: enforced timeout must be positive")
                if metadata["cytobridge_timeout_enforced"] is not True:
                    invalid.append(f"{name}: isolated tool must mark timeout enforced")
            else:
                if metadata["cytobridge_timeout_sec"] is not None:
                    invalid.append(f"{name}: in-process tool must not declare an enforced timeout")
                if metadata["cytobridge_timeout_enforced"] is not False:
                    invalid.append(f"{name}: in-process tool must mark timeout_enforced false")
            if metadata["cytobridge_isolation"] == "process":
                if metadata["cytobridge_isolation_timeout"] is None:
                    invalid.append(f"{name}: process-isolated tool missing enforced isolation timeout")
            for flag in (
                "cytobridge_mutates_state",
                "cytobridge_reads_adata",
                "cytobridge_writes_artifacts",
                "cytobridge_may_use_gpu",
            ):
                if not isinstance(metadata[flag], bool):
                    invalid.append(f"{name}: {flag} must be bool")

        self.assertEqual([], missing)
        self.assertEqual([], invalid)

    def test_manual_tool_schemas_match_backing_signatures(self) -> None:
        registry_path = Path(tool_registry.__file__)
        module_ast = ast.parse(registry_path.read_text(encoding="utf-8"))
        checked: list[str] = []
        mismatches: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                if not (isinstance(node.func, ast.Attribute) and node.func.attr == "_build_tool"):
                    self.generic_visit(node)
                    return
                if not node.args:
                    self.generic_visit(node)
                    return
                func_expr = node.args[0]
                if not (
                    isinstance(func_expr, ast.Attribute)
                    and isinstance(func_expr.value, ast.Name)
                    and func_expr.value.id == "self"
                ):
                    self.generic_visit(node)
                    return
                schema_name = ""
                for keyword in node.keywords:
                    if keyword.arg == "args_schema" and isinstance(keyword.value, ast.Name):
                        schema_name = keyword.value.id
                        break
                if not schema_name:
                    self.generic_visit(node)
                    return

                func_name = func_expr.attr
                schema = getattr(tool_registry, schema_name)
                func = getattr(tool_registry.SingleAgentTools, func_name)
                signature = inspect.signature(func)
                params = {
                    name
                    for name, param in signature.parameters.items()
                    if name != "self"
                    and param.kind
                    in {
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    }
                }
                has_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in signature.parameters.values()
                )
                fields = set(schema.model_fields)
                checked.append(f"{func_name}:{schema_name}")
                if has_kwargs:
                    self.generic_visit(node)
                    return
                missing_from_schema = sorted(params - fields)
                unsupported_by_func = sorted(fields - params)
                if missing_from_schema or unsupported_by_func:
                    mismatches.append(
                        f"{func_name} / {schema_name}: "
                        f"missing_from_schema={missing_from_schema}; "
                        f"unsupported_by_func={unsupported_by_func}"
                    )
                self.generic_visit(node)

        Visitor().visit(module_ast)

        self.assertGreater(len(checked), 0)
        self.assertEqual([], mismatches)

    def test_campaign_tool_schemas_document_agent_contracts(self) -> None:
        tools = {
            str(tool.name): tool
            for tool in tool_registry.SingleAgentTools(llm=None, state={}, agent_role="planner").get_tools()
        }

        start_tool = tools["start_algorithm_campaign"]
        start_schema = start_tool.args_schema.model_json_schema()
        start_text = " ".join(
            [
                str(start_tool.description or ""),
                str(start_schema["properties"]["claim_metric_spec"].get("description") or ""),
            ]
        )
        self.assertIn("evaluator_path", start_text)
        self.assertIn("name/direction alone", start_text)

        refresh_tool = tools["refresh_campaign_stage_baselines"]
        refresh_schema = refresh_tool.args_schema.model_json_schema()
        refresh_text = " ".join(
            [
                str(refresh_tool.description or ""),
                str(refresh_schema["properties"]["run_missing"].get("description") or ""),
            ]
        )
        self.assertIn("Reuses existing", refresh_text)
        self.assertIn("truly missing", refresh_text)
        self.assertIn("evaluator_path", refresh_text)

        compute_tool = tools["compute_campaign_claim_metric_for_baselines"]
        compute_schema = compute_tool.args_schema.model_json_schema()
        compute_text = " ".join(
            [
                str(compute_tool.description or ""),
                str(compute_schema["properties"]["regenerate_trajectory_if_missing"].get("description") or ""),
            ]
        )
        self.assertIn("saved", compute_text)
        self.assertIn("structured", compute_text)

        make_config_tool = tools["make_benchmark_dataset_config"]
        make_config_schema = make_config_tool.args_schema.model_json_schema()
        make_config_text = " ".join(
            [
                str(make_config_tool.description or ""),
                str(make_config_schema["properties"]["per_dataset_config_overrides"].get("description") or ""),
            ]
        )
        self.assertIn("per_dataset_config_overrides", make_config_text)
        self.assertIn("stage freezes dataset ids", make_config_text)

        set_panel_tool = tools["set_campaign_stage_panel"]
        set_panel_schema = set_panel_tool.args_schema.model_json_schema()
        set_panel_text = " ".join(
            [
                str(set_panel_tool.description or ""),
                str(set_panel_schema["properties"]["dataset_config_overrides"].get("description") or ""),
            ]
        )
        self.assertIn("make_benchmark_dataset_config", set_panel_text)
        self.assertIn("locks dataset ids", set_panel_text)
        self.assertIn("not per-dataset config values", set_panel_text)

        switch_panel_tool = tools["switch_campaign_stage_panel"]
        switch_panel_schema = switch_panel_tool.args_schema.model_json_schema()
        switch_panel_text = " ".join(
            [
                str(switch_panel_tool.description or ""),
                str(switch_panel_schema["properties"]["dataset_config_overrides"].get("description") or ""),
                str(switch_panel_schema["properties"]["reason"].get("description") or ""),
            ]
        )
        self.assertIn("without resetting", switch_panel_text)
        self.assertIn("trial_count", switch_panel_text)
        self.assertIn("active best", switch_panel_text)
        self.assertIn("baseline", switch_panel_text)

        run_tool = tools["run_campaign_trial"]
        run_schema = run_tool.args_schema.model_json_schema()
        run_text = " ".join(
            [
                str(run_tool.description or ""),
                str(run_schema["properties"]["dataset_config_overrides"].get("description") or ""),
            ]
        )
        self.assertIn("make_benchmark_dataset_config", run_text)
        self.assertIn("frozen stage panel", run_text)
        self.assertIn("dataset-specific config", run_text)
        self.assertIn("does not pass", run_text)

        direct_training_tool = tools["run_training"]
        direct_training_schema = direct_training_tool.args_schema.model_json_schema()
        direct_training_text = str(direct_training_tool.description or "")
        self.assertNotIn("stage", direct_training_schema.get("properties") or {})
        self.assertIn("manual/debug", direct_training_text)
        self.assertIn("final_regression locked release", direct_training_text)

        failed_tool = tools["mark_algorithm_failed"]
        failed_schema = failed_tool.args_schema.model_json_schema()
        failed_text = " ".join(
            [
                str(failed_tool.description or ""),
                str(failed_schema["properties"]["reason"].get("description") or ""),
            ]
        )
        self.assertIn("failed", failed_text)
        self.assertIn("not complete", failed_text)
        self.assertIn("reason", failed_schema.get("required") or [])

        abort_tool = tools["abort_current_campaign_trial"]
        abort_schema = abort_tool.args_schema.model_json_schema()
        abort_text = " ".join(
            [
                str(abort_tool.description or ""),
                str(abort_schema["properties"]["reason"].get("description") or ""),
                str(abort_schema["properties"]["restore_active_best"].get("description") or ""),
            ]
        )
        self.assertIn("stale", abort_text)
        self.assertIn("status=running", abort_text)
        self.assertIn("clears current_trial_id", abort_text)
        self.assertIn("does not change the active-best pointer", abort_text)

        status_tool = tools["get_algorithm_campaign_status"]
        status_schema = status_tool.args_schema.model_json_schema()
        status_text = " ".join(
            [
                str(status_tool.description or ""),
                str(status_schema["properties"]["recent_trials_limit"].get("description") or ""),
            ]
        )
        self.assertNotIn("detail", status_schema["properties"])
        self.assertIn("concise", status_text)
        self.assertIn("does not return the raw campaign registry", status_text)
        self.assertIn("recent trials", status_text)

        list_trials_tool = tools["list_campaign_trials"]
        list_trials_schema = list_trials_tool.args_schema.model_json_schema()
        list_trials_text = " ".join(
            [
                str(list_trials_tool.description or ""),
                str(list_trials_schema["properties"]["limit"].get("description") or ""),
                str(list_trials_schema["properties"]["offset"].get("description") or ""),
            ]
        )
        self.assertEqual(10, list_trials_schema["properties"]["limit"].get("default"))
        self.assertEqual(50, list_trials_schema["properties"]["limit"].get("maximum"))
        self.assertIn("concise", list_trials_text)
        self.assertIn("paginated", list_trials_text)
        self.assertIn("never returns raw full trial payloads", list_trials_text)

        openai_required = {
            name: set(
                convert_to_openai_tool(tools[name])["function"]["parameters"].get("required") or []
            )
            for name in [
                "make_benchmark_dataset_config",
                "set_campaign_stage_panel",
                "switch_campaign_stage_panel",
                "run_campaign_trial",
                "abort_current_campaign_trial",
                "get_algorithm_campaign_status",
                "list_campaign_trials",
                "refresh_campaign_stage_baselines",
                "compute_campaign_claim_metric_for_baselines",
            ]
        }
        self.assertNotIn("dataset_ids", openai_required["make_benchmark_dataset_config"])
        self.assertNotIn("common_config_overrides", openai_required["make_benchmark_dataset_config"])
        self.assertNotIn("per_dataset_config_overrides", openai_required["make_benchmark_dataset_config"])
        self.assertNotIn("dataset_config_overrides", openai_required["set_campaign_stage_panel"])
        self.assertNotIn("dataset_config_overrides", openai_required["switch_campaign_stage_panel"])
        self.assertNotIn("reason", openai_required["switch_campaign_stage_panel"])
        self.assertNotIn("dataset_config_overrides", openai_required["run_campaign_trial"])
        self.assertNotIn("reason", openai_required["abort_current_campaign_trial"])
        self.assertNotIn("restore_active_best", openai_required["abort_current_campaign_trial"])
        self.assertNotIn("detail", openai_required["get_algorithm_campaign_status"])
        self.assertNotIn("recent_trials_limit", openai_required["get_algorithm_campaign_status"])
        self.assertNotIn("limit", openai_required["list_campaign_trials"])
        self.assertNotIn("offset", openai_required["list_campaign_trials"])
        self.assertNotIn("stage", openai_required["list_campaign_trials"])
        self.assertNotIn("baseline_algorithms", openai_required["refresh_campaign_stage_baselines"])
        self.assertNotIn("baseline_algorithms", openai_required["compute_campaign_claim_metric_for_baselines"])

        patch_config_tool = tools["patch_algorithm_config"]
        patch_config_schema = patch_config_tool.args_schema.model_json_schema()
        patch_config_text = " ".join(
            [
                str(patch_config_tool.description or ""),
                str(patch_config_schema["properties"]["updates"].get("description") or ""),
                str(patch_config_schema["properties"]["dry_run"].get("description") or ""),
                str(patch_config_schema["properties"]["allow_list_replace"].get("description") or ""),
            ]
        )
        self.assertIn("typed leaf path", patch_config_text)
        self.assertIn("dry_run", patch_config_text)
        self.assertIn("training.plan", patch_config_text)
        self.assertIn("accidental", patch_config_text)

        artifact_tool = tools["create_algorithm_workspace_artifact"]
        artifact_schema = artifact_tool.args_schema.model_json_schema()
        artifact_text = " ".join(
            [
                str(artifact_tool.description or ""),
                str(artifact_schema["properties"]["relative_path"].get("description") or ""),
                str(artifact_schema["properties"]["kind"].get("description") or ""),
            ]
        )
        self.assertIn("active algorithm workspace", artifact_text)
        self.assertIn("relative", artifact_text)
        self.assertIn("diagnostics", artifact_text)
        self.assertIn("directory", artifact_text)

    def test_campaign_schema_accepts_json_strings_from_weak_tool_call_models(self) -> None:
        make_input = tool_registry._MakeBenchmarkDatasetConfigInput(
            dataset_ids='["simulation_gene_2d", "weinreb_rawrebuild_full_k10mindiff1"]',
            common_config_overrides='{"evaluation": {"trajectory_step": 1.0}}',
        )
        self.assertEqual(
            make_input.dataset_ids,
            ["simulation_gene_2d", "weinreb_rawrebuild_full_k10mindiff1"],
        )
        self.assertEqual(make_input.common_config_overrides["evaluation"]["trajectory_step"], 1.0)

        panel_payload = (
            '{"datasets": [{"dataset_id": "simulation_gene_2d", '
            '"adata_path": "/tmp/sim.h5ad"}], '
            '"target_dataset_ids": ["simulation_gene_2d"]}'
        )
        set_input = tool_registry._SetCampaignStagePanelInput(
            campaign_id="campaign_test",
            dataset_config_overrides=panel_payload,
        )
        run_input = tool_registry._RunCampaignTrialInput(
            campaign_id="campaign_test",
            dataset_config_overrides=panel_payload,
        )
        self.assertEqual(
            set_input.dataset_config_overrides["target_dataset_ids"],
            ["simulation_gene_2d"],
        )
        self.assertEqual(
            run_input.dataset_config_overrides["datasets"][0]["adata_path"],
            "/tmp/sim.h5ad",
        )

    def test_proposal_revision_tool_surface_supports_full_markdown_and_patch(self) -> None:
        tools = {
            str(tool.name): tool
            for tool in tool_registry.SingleAgentTools(llm=None, state={}, agent_role="planner").get_tools()
        }
        self.assertIn("revise_algorithm_proposal", tools)
        revise_tool = tools["revise_algorithm_proposal"]
        revise_text = str(revise_tool.description or "")
        self.assertIn("proposal_markdown", revise_text)
        self.assertIn("proposal_patch", revise_text)
        self.assertIn("review flow", revise_text)

        patch_tool = tools["apply_workspace_patch"]
        patch_text = str(patch_tool.description or "")
        self.assertIn("revise_algorithm_proposal", patch_text)
        self.assertIn("new proposal_id", patch_text)
        self.assertIn("editable_proposal_path", patch_text)
        self.assertIn("must not mix", patch_text)


if __name__ == "__main__":
    unittest.main()
