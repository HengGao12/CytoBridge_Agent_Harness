from __future__ import annotations

import copy
import importlib.util
import json
import numbers
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple

from .training_tools import get_cellcompass_root


MetricHook = Callable[[Any], Dict[str, Any] | None]
DEFAULT_EVALUATION_TRAJECTORY_ARTIFACT_MAX_MB = 1024


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def load_evaluation_trajectory_artifact(path: str | Path) -> Any:
    """
    Load the canonical evaluation trajectory artifact written by
    `TrainingPipeline.evaluate`.

    The artifact is intentionally model-independent: claim metrics can be
    recomputed from a stored rollout without reloading or rerunning a baseline
    model.
    """
    import numpy as np
    from CytoBridge.tl.training_algorithm import EvaluationTrajectory

    artifact_path = Path(path).expanduser()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"evaluation trajectory artifact is missing: {artifact_path}")
    max_mb = _env_float(
        "CYTOBRIDGE_EVALUATION_TRAJECTORY_ARTIFACT_MAX_MB",
        DEFAULT_EVALUATION_TRAJECTORY_ARTIFACT_MAX_MB,
    )
    if max_mb > 0:
        size_mb = float(artifact_path.stat().st_size) / (1024.0 * 1024.0)
        if size_mb > max_mb:
            raise ValueError(
                "evaluation trajectory artifact is too large to load safely: "
                f"compressed_size_mb={size_mb:.1f}, max_mb={max_mb:.1f}, path={artifact_path}"
            )
    with np.load(artifact_path, allow_pickle=False) as payload:
        manifest_raw = payload["manifest_json"] if "manifest_json" in payload.files else None
        if manifest_raw is None:
            raise ValueError(f"evaluation trajectory artifact lacks manifest_json: {artifact_path}")
        manifest = json.loads(str(manifest_raw.item()))
        point_keys = list(manifest.get("point_keys") or [])
        weight_keys = list(manifest.get("weight_keys") or [])
        if len(point_keys) != len(weight_keys):
            raise ValueError("evaluation trajectory artifact point/weight key counts do not match")
        points_by_time = [payload[key].copy() for key in point_keys]
        weights_by_time = [payload[key].copy() for key in weight_keys]
        return EvaluationTrajectory(
            time_points=[float(t) for t in list(manifest.get("time_points") or payload["time_points"].tolist())],
            points_by_time=points_by_time,
            weights_by_time=weights_by_time,
            observed_time_points=[
                float(t)
                for t in list(manifest.get("observed_time_points") or payload["observed_time_points"].tolist())
            ],
            observed_time_indices=[
                int(i)
                for i in list(manifest.get("observed_time_indices") or payload["observed_time_indices"].tolist())
            ],
            source=str(manifest.get("source") or "saved_evaluation_trajectory"),
            step=manifest.get("step"),
            artifacts={"evaluation_trajectory_path": str(artifact_path)},
            metadata=dict(manifest.get("metadata") or {}),
        )


def claim_metric_spec_has_evaluator(spec: Any) -> bool:
    if not isinstance(spec, dict):
        return False
    for key in (
        "evaluator_path",
        "claim_metric_evaluator_path",
        "metric_evaluator_path",
        "campaign_metric_evaluator_path",
    ):
        if str(spec.get(key) or "").strip():
            return True
    return bool(callable(spec.get("evaluator_callable")))


def _claim_metric_name(spec: Dict[str, Any]) -> str:
    return str(
        spec.get("name")
        or spec.get("metric_name")
        or spec.get("primary_metric")
        or "claim_metric"
    ).strip()


def _safe_baseline_key(value: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", str(value or "").strip().lower()).strip("-")


def _raw_evaluator_path(spec: Dict[str, Any]) -> str:
    return str(
        spec.get("evaluator_path")
        or spec.get("claim_metric_evaluator_path")
        or spec.get("metric_evaluator_path")
        or spec.get("campaign_metric_evaluator_path")
        or ""
    ).strip()


def _is_builtin_holdout_time_w1_path(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {
        "builtin:holdout_time_w1",
        "cytobridge:holdout_time_w1",
        "holdout_time_w1",
    }


_CUSTOM_ALGORITHM_EVALUATOR_RE = re.compile(
    r"^custom_algorithm:(?P<algorithm_id>[A-Za-z0-9_.-]+)(?::(?P<function>[A-Za-z_][A-Za-z0-9_]*))?$"
)


def _split_path_function(raw_path: str) -> Tuple[str, str]:
    value = str(raw_path or "").strip()
    custom_match = _CUSTOM_ALGORITHM_EVALUATOR_RE.match(value)
    if custom_match:
        algorithm_id = custom_match.group("algorithm_id")
        function_name = custom_match.group("function") or ""
        return f"custom_algorithm:{algorithm_id}", function_name
    match = re.match(r"^(?P<path>.+\.py):(?P<function>[A-Za-z_][A-Za-z0-9_]*)$", value)
    if not match:
        return value, ""
    return match.group("path"), match.group("function")


def _resolve_evaluator_path(spec: Dict[str, Any]) -> Optional[Path]:
    raw_path, _ = _split_path_function(_raw_evaluator_path(spec))
    if not raw_path:
        return None
    custom_match = re.match(r"^custom_algorithm:(?P<algorithm_id>[A-Za-z0-9_.-]+)$", raw_path)
    if custom_match:
        algorithm_id = custom_match.group("algorithm_id").strip().lower()
        return (get_cellcompass_root() / "training_algorithms" / algorithm_id / "algorithm.py").resolve()
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    algorithm_id = str(spec.get("algorithm_id") or spec.get("training_algorithm_id") or "").strip().lower()
    if algorithm_id:
        candidate = get_cellcompass_root() / "training_algorithms" / algorithm_id / path
        if candidate.exists():
            return candidate.resolve()
    base_dir = str(spec.get("base_dir") or spec.get("workspace_path") or "").strip()
    if base_dir:
        candidate = Path(base_dir).expanduser() / path
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def _load_spec_callable_from_algorithm_path(path: Path, spec: Dict[str, Any], function_name: str) -> Callable[[Any], Any]:
    algorithm_id = str(spec.get("algorithm_id") or spec.get("training_algorithm_id") or path.parent.name).strip().lower()
    if not algorithm_id:
        raise AttributeError(f"cannot resolve TrainingAlgorithmSpec callable from {path}: missing algorithm_id")
    try:
        from CytoBridge.tl.training_algorithm import TrainingAlgorithmContext
        from .training_algorithm_registry import load_training_algorithm, resolve_base_config

        search_roots = [path.parent.parent]
        loaded = load_training_algorithm(
            algorithm_id,
            search_roots=search_roots,
            workspace_root=Path(str(spec.get("workspace_path") or path.parent)).expanduser().resolve().parent,
        )
        resolved_base_config, base_config_name = resolve_base_config(
            loaded.manifest["base_config"],
            config_dir=loaded.root_dir,
        )
        context = TrainingAlgorithmContext(
            algorithm_id=algorithm_id,
            input_adata_path=str(spec.get("input_adata_path") or ""),
            output_dir=str(spec.get("output_dir") or ""),
            stage=str(spec.get("stage") or "claim_metric"),
            base_config_name=base_config_name,
            resolved_base_config=resolved_base_config,
            metadata={
                "purpose": "claim_metric_evaluator_load",
                "source": "TrainingAlgorithmSpec",
            },
        )
        algorithm_spec = loaded.build_spec(context)
    except Exception as exc:
        raise AttributeError(f"cannot load TrainingAlgorithmSpec for claim metric evaluator from {path}: {exc}") from exc
    fn = getattr(algorithm_spec, function_name, None)
    if callable(fn):
        return fn
    raise AttributeError(
        f"TrainingAlgorithmSpec for {algorithm_id} has no callable {function_name!r}; "
        "for nested evaluator hooks, use evaluator_path='algorithm.py:evaluation_metrics_hook'"
    )


def _resolve_baseline_adapter_path(adapter: Dict[str, Any], parent_spec: Dict[str, Any]) -> Optional[Path]:
    raw_path = str(
        adapter.get("adapter_path")
        or adapter.get("adaptor_path")
        or adapter.get("path")
        or ""
    ).strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    algorithm_id = str(parent_spec.get("algorithm_id") or parent_spec.get("training_algorithm_id") or "").strip().lower()
    if algorithm_id:
        candidate = get_cellcompass_root() / "training_algorithms" / algorithm_id / path
        if candidate.exists():
            return candidate.resolve()
    base_dir = str(parent_spec.get("base_dir") or parent_spec.get("workspace_path") or "").strip()
    if base_dir:
        candidate = Path(base_dir).expanduser() / path
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def _resolve_baseline_adapter(spec: Dict[str, Any], baseline_algorithm: str) -> Dict[str, Any]:
    algo = _safe_baseline_key(baseline_algorithm)
    if not algo:
        return {}
    raw_map = (
        spec.get("baseline_metric_adapters")
        or spec.get("baseline_metric_adaptors")
        or spec.get("baseline_adapters")
        or spec.get("baseline_adaptors")
        or {}
    )
    if not isinstance(raw_map, dict):
        return {}
    candidates = [
        baseline_algorithm,
        algo,
        "*",
        "default",
    ]
    for key in candidates:
        if key in raw_map:
            raw = raw_map.get(key)
            if isinstance(raw, str):
                return {"adapter_path": raw, "baseline_algorithm": algo, "matched_key": key}
            if isinstance(raw, dict):
                out = dict(raw)
                out.setdefault("baseline_algorithm", algo)
                out.setdefault("matched_key", key)
                return out
    return {}


def campaign_baseline_metric_adapter_status(spec: Any, baseline_algorithm: str) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        return {"declared": False, "supported": True, "baseline_algorithm": _safe_baseline_key(baseline_algorithm)}
    adapter = _resolve_baseline_adapter(spec, baseline_algorithm)
    if not adapter:
        return {"declared": False, "supported": True, "baseline_algorithm": _safe_baseline_key(baseline_algorithm)}
    adapter_path = _resolve_baseline_adapter_path(adapter, spec)
    return {
        "declared": True,
        "supported": _baseline_adapter_supported(adapter),
        "baseline_algorithm": _safe_baseline_key(baseline_algorithm),
        "matched_key": str(adapter.get("matched_key") or ""),
        "source": str(adapter_path) if adapter_path is not None else "declarative",
        "version": str(adapter.get("version") or adapter.get("adapter_version") or ""),
        "reason": str(adapter.get("reason") or adapter.get("unsupported_reason") or "").strip(),
    }


def _load_callable_from_path(path: Path, function_name: str) -> Callable[[Any], Any]:
    if not path.is_file():
        raise FileNotFoundError(f"claim metric evaluator file does not exist: {path}")
    if path.suffix.lower() != ".py":
        raise ValueError(f"claim metric evaluator must be a Python file, got: {path}")
    module_name = f"_cytobridge_claim_metric_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import claim metric evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, function_name, None)
    if fn is None:
        for fallback in ("evaluate_claim_metric", "claim_metric_evaluator", "evaluate"):
            fn = getattr(module, fallback, None)
            if fn is not None:
                function_name = fallback
                break
    if not callable(fn):
        raise AttributeError(
            f"claim metric evaluator file {path} must define callable {function_name!r} "
            "or one of evaluate_claim_metric/claim_metric_evaluator/evaluate"
        )
    return fn


def _load_adapter_callable_from_path(path: Path, function_name: str) -> Callable[[Any], Any]:
    if not path.is_file():
        raise FileNotFoundError(f"baseline metric adaptor file does not exist: {path}")
    if path.suffix.lower() != ".py":
        raise ValueError(f"baseline metric adaptor must be a Python file, got: {path}")
    module_name = f"_cytobridge_baseline_metric_adapter_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import baseline metric adaptor from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, function_name, None)
    if fn is None:
        for fallback in ("adapt_baseline_metric_context", "baseline_metric_adapter", "adapt", "adaptor"):
            fn = getattr(module, fallback, None)
            if fn is not None:
                break
    if not callable(fn):
        raise AttributeError(
            f"baseline metric adaptor file {path} must define callable {function_name!r} "
            "or one of adapt_baseline_metric_context/baseline_metric_adapter/adapt/adaptor"
        )
    return fn


def _normalize_metric_result(result: Any, metric_name: str) -> Dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, numbers.Real):
        return {metric_name: float(result)}
    if not isinstance(result, dict):
        raise TypeError(
            "claim metric evaluator must return dict[str, Any], numeric scalar, or None; "
            f"got {type(result).__name__}"
        )
    out = dict(result)
    if metric_name and metric_name not in out:
        declared_name = str(out.get("metric_name") or out.get("name") or "").strip()
        if declared_name == metric_name:
            value = out.get("value")
            if isinstance(value, numbers.Real):
                out[metric_name] = float(value)
            else:
                details = out.get("details")
                if isinstance(details, dict) and isinstance(details.get(metric_name), numbers.Real):
                    out[metric_name] = float(details[metric_name])
    return out


def _merge_context_dict(base: Any, updates: Any) -> Dict[str, Any]:
    out = dict(base or {}) if isinstance(base, dict) else {}
    if isinstance(updates, dict):
        out.update(dict(updates))
    return out


def _baseline_adapter_supported(adapter: Dict[str, Any]) -> bool:
    for key in ("supported", "is_supported", "enabled"):
        if key in adapter and isinstance(adapter.get(key), bool):
            return bool(adapter.get(key))
    return True


def _unsupported_payload(metric_name: str, adapter_info: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "claim_metric_support_status": "unsupported",
        "claim_metric_unsupported_reason": reason or "baseline metric adaptor marked this baseline unsupported",
        "claim_metric_name": metric_name,
        "claim_metric_baseline_algorithm": str(adapter_info.get("baseline_algorithm") or ""),
        "claim_metric_adapter_source": str(adapter_info.get("source") or "declarative"),
    }


def _adapt_baseline_metric_context(context: Any, adapter: Dict[str, Any], adapter_info: Dict[str, Any]) -> Tuple[Any, Optional[Dict[str, Any]]]:
    if not adapter:
        return context, None
    if not _baseline_adapter_supported(adapter):
        reason = str(adapter.get("reason") or adapter.get("unsupported_reason") or "").strip()
        return context, {"reason": reason}

    proxy_payload = dict(getattr(context, "__dict__", {}) or {})
    metadata = _merge_context_dict(
        getattr(context, "metadata", {}),
        {
            "baseline_algorithm": str(adapter_info.get("baseline_algorithm") or ""),
            "baseline_metric_adapter": dict(adapter_info),
        },
    )
    metadata = _merge_context_dict(metadata, adapter.get("metadata"))
    metric_params = _merge_context_dict(getattr(context, "metric_params", {}), adapter.get("metric_params"))
    metric_params["baseline_algorithm"] = str(adapter_info.get("baseline_algorithm") or "")
    metric_params["baseline_metric_adapter"] = dict(adapter_info)
    proxy_payload["metadata"] = metadata
    proxy_payload["metric_params"] = metric_params
    adapted_context: Any = SimpleNamespace(**proxy_payload)

    adapter_path = _resolve_baseline_adapter_path(adapter, dict(adapter.get("_parent_spec") or {}))
    if adapter_path is not None:
        function_name = str(
            adapter.get("adapter_function")
            or adapter.get("adaptor_function")
            or adapter.get("function_name")
            or "adapt_baseline_metric_context"
        ).strip()
        fn = _load_adapter_callable_from_path(adapter_path, function_name)
        raw = fn(adapted_context)
        if isinstance(raw, dict):
            if raw.get("supported") is False or raw.get("is_supported") is False:
                return adapted_context, {"reason": str(raw.get("reason") or raw.get("unsupported_reason") or "")}
            if "context" in raw:
                adapted_context = raw["context"]
            else:
                proxy_payload = dict(getattr(adapted_context, "__dict__", {}) or {})
                proxy_payload["metadata"] = _merge_context_dict(proxy_payload.get("metadata"), raw.get("metadata"))
                proxy_payload["metric_params"] = _merge_context_dict(proxy_payload.get("metric_params"), raw.get("metric_params"))
                adapted_context = SimpleNamespace(**proxy_payload)
        elif raw is not None:
            adapted_context = raw
    return adapted_context, None


def load_campaign_claim_metric_evaluator(
    spec: Any,
    *,
    baseline_algorithm: str = "",
) -> Tuple[Optional[MetricHook], Dict[str, Any]]:
    if not isinstance(spec, dict) or not claim_metric_spec_has_evaluator(spec):
        return None, {}
    metric_name = _claim_metric_name(spec)
    raw_evaluator_path = _raw_evaluator_path(spec)
    if _is_builtin_holdout_time_w1_path(raw_evaluator_path):
        source = "builtin:holdout_time_w1"
        function_name = "evaluate_holdout_time_w1"
        info = {
            "name": metric_name,
            "direction": str(spec.get("direction") or "lower"),
            "source": source,
            "function_name": function_name,
            "version": str(spec.get("version") or spec.get("claim_metric_version") or "1"),
            "evaluator_id": str(spec.get("claim_metric_evaluator_id") or spec.get("evaluator_id") or ""),
        }
        stable_payload = {
            "name": info["name"],
            "direction": info["direction"],
            "source": info["source"],
            "function_name": info["function_name"],
            "version": info["version"],
            "evaluator_id": info["evaluator_id"],
            "baseline_metric_adapter": {},
        }
        info["fingerprint_payload"] = stable_payload
        info["fingerprint_json"] = json.dumps(stable_payload, sort_keys=True, ensure_ascii=False)

        def hook(context: Any) -> Dict[str, Any]:
            builtin_metrics = getattr(context, "builtin_metrics", {})
            payload = {}
            if isinstance(builtin_metrics, dict):
                payload = builtin_metrics.get("holdout_time_evaluation") or {}
            if not isinstance(payload, dict):
                payload = {}
            value = payload.get("mean_w1")
            if isinstance(value, numbers.Real):
                return {
                    metric_name: float(value),
                    "claim_metric_evaluator_source": source,
                    "claim_metric_version": info["version"],
                }
            return {}

        return hook, info

    _, inline_function = _split_path_function(raw_evaluator_path)
    function_name = str(
        spec.get("evaluator_function")
        or spec.get("function_name")
        or inline_function
        or "evaluate_claim_metric"
    ).strip()
    evaluator_callable = spec.get("evaluator_callable")
    path = _resolve_evaluator_path(spec)
    if callable(evaluator_callable):
        fn = evaluator_callable
        source = "callable"
    else:
        if path is None:
            raise ValueError("claim metric evaluator spec is missing evaluator_path")
        try:
            fn = _load_callable_from_path(path, function_name)
        except AttributeError:
            if path.name != "algorithm.py" or function_name not in {
                "evaluation_metrics_hook",
                "flow_matching_loss_hook",
                "simulation_hook",
            }:
                raise
            fn = _load_spec_callable_from_algorithm_path(path, spec, function_name)
        source = str(path)

    info = {
        "name": metric_name,
        "direction": str(spec.get("direction") or "greater"),
        "source": source,
        "function_name": function_name,
        "version": str(spec.get("version") or spec.get("claim_metric_version") or ""),
        "evaluator_id": str(spec.get("claim_metric_evaluator_id") or spec.get("evaluator_id") or ""),
    }
    adapter = _resolve_baseline_adapter(spec, baseline_algorithm)
    adapter_info: Dict[str, Any] = {}
    if adapter:
        adapter["_parent_spec"] = dict(spec)
        adapter_path = _resolve_baseline_adapter_path(adapter, spec)
        adapter_info = {
            "baseline_algorithm": _safe_baseline_key(baseline_algorithm),
            "matched_key": str(adapter.get("matched_key") or ""),
            "source": str(adapter_path) if adapter_path is not None else "declarative",
            "function_name": str(
                adapter.get("adapter_function")
                or adapter.get("adaptor_function")
                or adapter.get("function_name")
                or "adapt_baseline_metric_context"
            ),
            "version": str(adapter.get("version") or adapter.get("adapter_version") or ""),
            "supported": _baseline_adapter_supported(adapter),
        }
        if str(adapter.get("reason") or adapter.get("unsupported_reason") or "").strip():
            adapter_info["reason"] = str(adapter.get("reason") or adapter.get("unsupported_reason") or "").strip()
        info["baseline_metric_adapter"] = dict(adapter_info)
    stable_payload = {
        "name": info["name"],
        "direction": info["direction"],
        "source": info["source"],
        "function_name": info["function_name"],
        "version": info["version"],
        "evaluator_id": info["evaluator_id"],
        "baseline_metric_adapter": adapter_info,
    }
    info["fingerprint_payload"] = stable_payload
    info["fingerprint_json"] = json.dumps(stable_payload, sort_keys=True, ensure_ascii=False)

    def hook(context: Any) -> Dict[str, Any]:
        metric_context = context
        if adapter:
            metric_context, unsupported = _adapt_baseline_metric_context(context, adapter, adapter_info)
            if unsupported is not None:
                return _unsupported_payload(metric_name, adapter_info, str(unsupported.get("reason") or ""))
        result = _normalize_metric_result(fn(metric_context), metric_name)
        if result and metric_name not in result and len(result) == 1:
            key = next(iter(result.keys()))
            value = result[key]
            if isinstance(value, numbers.Real):
                result[metric_name] = float(value)
        if result:
            result.setdefault("claim_metric_evaluator_source", info["source"])
            if info["version"]:
                result.setdefault("claim_metric_version", info["version"])
            if adapter_info:
                result.setdefault("claim_metric_baseline_algorithm", adapter_info.get("baseline_algorithm"))
                result.setdefault("claim_metric_adapter_source", adapter_info.get("source"))
                if adapter_info.get("version"):
                    result.setdefault("claim_metric_adapter_version", adapter_info.get("version"))
        return result

    return hook, info


def merge_evaluation_metric_hooks(
    existing_hook: Optional[MetricHook],
    campaign_hook: Optional[MetricHook],
    *,
    claim_metric_name: str,
) -> Optional[MetricHook]:
    if campaign_hook is None:
        return existing_hook
    if existing_hook is None:
        return campaign_hook

    def merged(context: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        existing = existing_hook(context)
        if isinstance(existing, dict):
            out.update(existing)
        campaign_context = context
        if out:
            try:
                campaign_context = copy.copy(context)
                builtin_metrics_raw = getattr(context, "builtin_metrics", None)
                builtin_metrics = dict(builtin_metrics_raw) if isinstance(builtin_metrics_raw, dict) else {}
                custom_metrics_raw = builtin_metrics.get("custom_metrics")
                custom_metrics = dict(custom_metrics_raw) if isinstance(custom_metrics_raw, dict) else {}
                custom_metrics.update(out)
                builtin_metrics["custom_metrics"] = custom_metrics
                setattr(campaign_context, "builtin_metrics", builtin_metrics)
            except Exception:
                campaign_context = context
        campaign = campaign_hook(campaign_context)
        if isinstance(campaign, dict):
            if claim_metric_name in out and claim_metric_name in campaign and out[claim_metric_name] != campaign[claim_metric_name]:
                out["claim_metric_evaluator_conflict_warning"] = (
                    f"campaign evaluator overwrote existing custom metric {claim_metric_name!r}"
                )
            out.update(campaign)
        return out

    return merged
