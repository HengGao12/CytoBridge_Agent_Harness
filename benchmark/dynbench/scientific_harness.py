"""Public-data-only scientific harness for DynBench agent runs.

The harness gives the agent compact, structured context before execution and
audits its first-pass deliverables before the benchmark evaluator is allowed to
see them.  It deliberately has no ground-truth or evaluator input: every
diagnostic is computed from the public task package and the agent's own files.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REQUIRED_OUTPUTS = (
    "velocity_field.csv",
    "growth_rates.csv",
    "holdout_prediction.csv",
    "per_cell_fate.json",
    "perturbation_results.json",
    "driver_genes.json",
)


def _dense(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _finite_float(value)
    return value


def _first_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = {str(column) for column in columns}
    return next((name for name in candidates if name in available), None)


def build_public_data_profile(
    train_h5ad: str | Path,
    prediction_targets_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize public task data without consulting hidden benchmark truth."""
    import anndata as ad

    train_path = Path(train_h5ad)
    adata = ad.read_h5ad(train_path)
    expression = _dense(adata.X).astype(np.float64, copy=False)
    genes = [str(name) for name in adata.var_names]
    time_key = _first_column(
        adata.obs.columns,
        ("time", "time_point_processed", "Time Point", "time_bin"),
    )
    label_key = _first_column(
        adata.obs.columns,
        ("fate_true", "cell_type", "celltype", "label"),
    )

    profile: dict[str, Any] = {
        "source": str(train_path.resolve()),
        "public_inputs_only": True,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "genes": genes,
        "time_key": time_key,
        "label_key": label_key,
        "expression": {
            "finite_fraction": float(np.isfinite(expression).mean()),
            "global_min": _finite_float(np.nanmin(expression)),
            "global_max": _finite_float(np.nanmax(expression)),
            "per_gene_mean": {
                gene: _finite_float(value)
                for gene, value in zip(genes, np.nanmean(expression, axis=0))
            },
            "per_gene_std": {
                gene: _finite_float(value)
                for gene, value in zip(genes, np.nanstd(expression, axis=0))
            },
        },
    }

    if time_key:
        times = pd.to_numeric(adata.obs[time_key], errors="coerce")
        valid_times = times.dropna()
        profile["observed_times"] = sorted(float(value) for value in valid_times.unique())
        profile["counts_by_time"] = {
            str(key): int(value)
            for key, value in valid_times.value_counts().sort_index().items()
        }
        frame = pd.DataFrame(expression, columns=genes)
        frame["__time"] = times.to_numpy()
        means = frame.groupby("__time", observed=True)[genes].mean()
        profile["mean_expression_by_time"] = {
            str(time): {gene: _finite_float(value) for gene, value in row.items()}
            for time, row in means.iterrows()
        }
        if label_key:
            grouped = pd.DataFrame(
                {"time": times.to_numpy(), "label": adata.obs[label_key].astype(str).to_numpy()}
            ).groupby(["time", "label"], observed=True).size()
            profile["counts_by_time_and_label"] = {
                f"{time}|{label}": int(count)
                for (time, label), count in grouped.items()
            }

    targets_path = Path(prediction_targets_path) if prediction_targets_path else None
    if targets_path and targets_path.exists():
        try:
            profile["prediction_targets"] = json.loads(targets_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            profile["prediction_targets_error"] = str(exc)

    return _json_ready(profile)


def write_public_data_profile(
    train_h5ad: str | Path,
    output_path: str | Path,
    prediction_targets_path: str | Path | None = None,
) -> dict[str, Any]:
    profile = build_public_data_profile(train_h5ad, prediction_targets_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return profile


def build_initial_harness_guidance(profile_path: str | Path) -> str:
    """Return metric-oriented instructions grounded only in public evidence."""
    return f"""

## Scientific Harness Protocol

Use the compact public-data profile at `{Path(profile_path)}` before choosing a
model or writing downstream code. This profile contains only information from
the task package; it contains no benchmark answers.

Work through these gates in order:

1. **Contract gate**: identify measured/latent space, observed times, held-out
   targets, source cells, classifier feature order, and all six schemas.
2. **Model-selection gate**: select dynamics and growth models using only public
   evidence. When practical, use leave-one-observed-time-out validation to test
   interpolation and fate stability before generating hidden-time predictions.
3. **Shared-model gate**: velocity, rollout, fate, perturbation, GRN, and growth
   drivers must come from one coherent fitted dynamics model and documented
   projections. Reusing a shared model is more reliable than six unrelated
   estimators.
4. **Scientific calibration gate**:
   - Validate holdout rollout on pseudo-holdouts made from observed time points;
     compare distributional coverage, not only mean trajectories.
   - Rank growth drivers with stability evidence across time strata or bootstrap
     resamples. Check agreement among signed mean gradient, mean absolute
     gradient, growth-head ablation, and expression/growth association rather
     than trusting one global gradient statistic.
   - Treat perturbation as a causal intervention. Use matched source cells and
     matched control rollouts, quantify resampling uncertainty, and distinguish
     upstream regulators from downstream reporters using model-local outgoing
     sensitivity or stable GRN support. Do not report every numerically nonzero
     delta as a biological effect.
   - Estimate GRN edges across time/cell strata and preserve sign only when it is
     stable. Continuous confidence scores should reflect stability as well as
     Jacobian magnitude.
5. **Artifact gate**: run the public schema verifier, then inspect cross-artifact
   consistency: fate totals, perturbation delta arithmetic, driver ranking,
   finite values, output diversity, and provenance.

Spend the execution budget on model fitting, validation, and file correction.
Keep already validated strong artifacts when revising a weak analysis.
"""


def _issue(
    issues: list[dict[str, Any]],
    code: str,
    summary: str,
    evidence: dict[str, Any],
    action: str,
    severity: str = "warning",
) -> None:
    issues.append(
        {
            "code": code,
            "severity": severity,
            "summary": summary,
            "evidence": _json_ready(evidence),
            "recommended_action": action,
        }
    )


def _abs_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) <= 1e-12 or np.std(yr) <= 1e-12:
        return 0.0
    return float(abs(np.corrcoef(xr, yr)[0, 1]))


def _stable_growth_surrogate_scores(
    expression: np.ndarray,
    growth: np.ndarray,
    times: np.ndarray | None,
    labels: np.ndarray | None,
    *,
    n_resamples: int = 8,
    random_state: int = 0,
) -> dict[str, np.ndarray]:
    """Estimate nonlinear growth sensitivity with stratified resampling.

    The target is the fitted model's per-cell growth output, so this remains a
    model-derived interpretation. Time and labels are included as nuisance
    features; only measured-gene importances are returned.
    """
    from sklearn.ensemble import ExtraTreesRegressor

    x = np.asarray(expression, dtype=np.float64)
    y = np.asarray(growth, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or y.shape[0] != x.shape[0] or x.shape[0] < 20:
        raise ValueError("Stable growth surrogate requires at least 20 aligned cells")

    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[valid]
    y = y[valid]
    if x.shape[0] < 20 or float(np.std(y)) <= 1e-12:
        raise ValueError("Growth output has too few finite or varying values")

    nuisance: list[np.ndarray] = []
    if times is not None:
        time_values = np.asarray(times, dtype=float).reshape(-1)[valid]
        fill = float(np.nanmedian(time_values)) if np.isfinite(time_values).any() else 0.0
        nuisance.append(np.nan_to_num(time_values, nan=fill).reshape(-1, 1))
    if labels is not None:
        label_values = np.asarray(labels).reshape(-1)[valid]
        nuisance.append(
            pd.get_dummies(pd.Series(label_values.astype(str)), drop_first=False).to_numpy(dtype=float)
        )
    design = np.column_stack([x, *nuisance]) if nuisance else x

    if times is not None or labels is not None:
        raw_times = np.asarray(times).reshape(-1)[valid] if times is not None else np.zeros(x.shape[0])
        raw_labels = np.asarray(labels).reshape(-1)[valid] if labels is not None else np.repeat("all", x.shape[0])
        stratum = np.asarray(
            [f"{time}|{label}" for time, label in zip(raw_times, raw_labels)],
            dtype=object,
        )
    else:
        stratum = np.repeat("all", x.shape[0])
    strata = [np.flatnonzero(stratum == value) for value in np.unique(stratum)]

    rng = np.random.default_rng(random_state)
    importances = []
    for repeat in range(max(3, int(n_resamples))):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata if len(indices)]
        )
        model = ExtraTreesRegressor(
            n_estimators=96,
            min_samples_leaf=max(5, len(sampled) // 300),
            max_features=1.0,
            random_state=random_state + repeat,
            n_jobs=1,
        )
        model.fit(design[sampled], y[sampled])
        importances.append(model.feature_importances_[: x.shape[1]])

    matrix = np.asarray(importances, dtype=float)
    top_frequency = np.bincount(
        np.argmax(matrix, axis=1), minlength=x.shape[1]
    ).astype(float) / matrix.shape[0]
    return {
        "mean": matrix.mean(axis=0),
        "std": matrix.std(axis=0),
        "top_frequency": top_frequency,
    }


def audit_public_outputs(
    train_h5ad: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Audit scientific plausibility using public inputs and agent outputs only."""
    import anndata as ad

    output = Path(output_dir)
    adata = ad.read_h5ad(train_h5ad)
    expression = _dense(adata.X).astype(np.float64, copy=False)
    genes = [str(name) for name in adata.var_names]
    issues: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}

    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
    if missing:
        _issue(
            issues,
            "missing_outputs",
            "Required deliverables are missing.",
            {"files": missing},
            "Materialize the missing files and rerun the public schema verifier.",
            severity="error",
        )

    growth_path = output / "growth_rates.csv"
    drivers_path = output / "driver_genes.json"
    driver_payload: dict[str, Any] = {}
    if drivers_path.exists():
        try:
            driver_payload = json.loads(drivers_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _issue(
                issues,
                "driver_json_invalid",
                "driver_genes.json is not readable JSON.",
                {"error": str(exc)},
                "Rewrite driver_genes.json with the required public schema.",
                severity="error",
            )

    if growth_path.exists() and driver_payload:
        try:
            growth = pd.read_csv(growth_path)["growth_rate"].to_numpy(dtype=float)
            if len(growth) == adata.n_obs:
                associations = {
                    gene: _abs_spearman(expression[:, idx], growth)
                    for idx, gene in enumerate(genes)
                }
                reported = {
                    str(gene): abs(_finite_float(score))
                    for gene, score in (driver_payload.get("growth_drivers") or {}).items()
                    if str(gene) in genes
                }
                diagnostics["growth_driver_cross_evidence"] = {
                    "abs_spearman_expression_vs_predicted_growth": associations,
                    "reported_driver_scores": reported,
                }
                time_key = _first_column(
                    adata.obs.columns,
                    ("time", "time_point_processed", "Time Point", "time_bin"),
                )
                label_key = _first_column(
                    adata.obs.columns,
                    ("fate_true", "cell_type", "celltype", "label"),
                )
                if time_key:
                    time_values = pd.to_numeric(adata.obs[time_key], errors="coerce").to_numpy(dtype=float)
                    nuisance_columns = [
                        np.ones(adata.n_obs, dtype=float),
                        np.nan_to_num(time_values, nan=float(np.nanmedian(time_values))),
                    ]
                    if label_key:
                        label_design = pd.get_dummies(
                            adata.obs[label_key].astype(str), drop_first=False
                        ).to_numpy(dtype=float)
                        nuisance_columns.extend(label_design.T)
                    nuisance = np.column_stack(nuisance_columns)
                    growth_residual = growth - nuisance @ np.linalg.lstsq(
                        nuisance, growth, rcond=None
                    )[0]
                    expression_residual = expression - nuisance @ np.linalg.lstsq(
                        nuisance, expression, rcond=None
                    )[0]
                    conditional = {
                        gene: _abs_spearman(expression_residual[:, idx], growth_residual)
                        for idx, gene in enumerate(genes)
                    }
                    diagnostics["growth_driver_cross_evidence"][
                        "abs_partial_spearman_controlling_time_and_label"
                    ] = conditional
                time_values = (
                    pd.to_numeric(adata.obs[time_key], errors="coerce").to_numpy(dtype=float)
                    if time_key else None
                )
                label_values = adata.obs[label_key].astype(str).to_numpy() if label_key else None
                stable = _stable_growth_surrogate_scores(
                    expression,
                    growth,
                    time_values,
                    label_values,
                )
                stable_scores = {
                    gene: _finite_float(stable["mean"][idx])
                    for idx, gene in enumerate(genes)
                }
                stable_frequency = {
                    gene: _finite_float(stable["top_frequency"][idx])
                    for idx, gene in enumerate(genes)
                }
                diagnostics["growth_driver_cross_evidence"][
                    "stable_nonlinear_surrogate"
                ] = {
                    "mean_importance": stable_scores,
                    "top_frequency": stable_frequency,
                }
                if reported and stable_scores:
                    top_reported = max(reported, key=reported.get)
                    stable_order = sorted(stable_scores, key=stable_scores.get, reverse=True)
                    stable_top = stable_order[0]
                    if top_reported != stable_top and stable_frequency.get(stable_top, 0.0) >= 0.75:
                        _issue(
                            issues,
                            "growth_driver_evidence_disagreement",
                            "The reported leading growth driver disagrees with a stable nonlinear surrogate of the model growth head.",
                            {
                                "reported_top": top_reported,
                                "stable_surrogate_top": stable_top,
                                "stable_top_frequency": stable_frequency.get(stable_top, 0.0),
                                "stable_order": stable_order,
                            },
                            "Replace the fragile global-gradient ranking with the time/label-adjusted, stratified-resampling surrogate ranking, and preserve both raw and calibrated scores in provenance.",
                        )
        except Exception as exc:
            diagnostics["growth_driver_audit_error"] = str(exc)

    perturb_path = output / "perturbation_results.json"
    effects: dict[str, float] = {}
    if perturb_path.exists():
        try:
            perturbations = json.loads(perturb_path.read_text(encoding="utf-8"))
            inconsistent: list[str] = []
            for row in perturbations if isinstance(perturbations, list) else []:
                gene = str(row.get("gene_name", ""))
                control = row.get("control") or {}
                perturbed = row.get("perturbed") or {}
                delta = row.get("delta") or {}
                fates = set(control) | set(perturbed) | set(delta)
                effects[gene] = max((abs(_finite_float(delta.get(fate))) for fate in fates), default=0.0)
                for fate in fates:
                    expected = _finite_float(perturbed.get(fate)) - _finite_float(control.get(fate))
                    if abs(expected - _finite_float(delta.get(fate))) > 1e-5:
                        inconsistent.append(f"{gene}:{fate}")
            if inconsistent:
                _issue(
                    issues,
                    "perturbation_delta_inconsistent",
                    "Some perturbation deltas do not equal perturbed minus control.",
                    {"pairs": inconsistent[:20]},
                    "Recompute each delta from its matched control and perturbed branches.",
                    severity="error",
                )
            if effects:
                max_effect = max(effects.values())
                activity_floor = max(0.01, 0.05 * max_effect)
                active = [gene for gene, effect in effects.items() if effect >= activity_floor]
                active_fraction = len(active) / max(1, len(effects))
                diagnostics["perturbation_selectivity"] = {
                    "effect_magnitude": effects,
                    "adaptive_activity_floor": activity_floor,
                    "active_genes": active,
                    "active_fraction": active_fraction,
                }
                if len(effects) >= 3 and active_fraction >= 0.8:
                    _issue(
                        issues,
                        "perturbation_unselective",
                        "Nearly every gene is reported as fate-active, which is a common out-of-distribution intervention artifact.",
                        {
                            "active_fraction": active_fraction,
                            "effect_magnitude": effects,
                        },
                        "Calibrate effects with matched bootstrap controls and model-local causal support. Downstream reporters can move the classifier without being upstream fate regulators; unsupported effects should shrink to the empirical null rather than remain automatically active.",
                    )
        except Exception as exc:
            _issue(
                issues,
                "perturbation_json_invalid",
                "perturbation_results.json could not be audited.",
                {"error": str(exc)},
                "Rewrite the file with one valid matched-control record per gene.",
                severity="error",
            )

    edges = driver_payload.get("grn_edges") or []
    if effects and isinstance(edges, list):
        outgoing = {gene: 0.0 for gene in genes}
        for edge in edges:
            source = str(edge.get("source", ""))
            if source in outgoing:
                outgoing[source] += abs(_finite_float(edge.get("score")))
        max_effect = max(effects.values(), default=0.0)
        max_outgoing = max(outgoing.values(), default=0.0)
        joint = {
            gene: (effects.get(gene, 0.0) / (max_effect + 1e-12))
            * (outgoing.get(gene, 0.0) / (max_outgoing + 1e-12))
            for gene in genes
        }
        values = np.asarray(list(joint.values()), dtype=float)
        median = float(np.median(values)) if values.size else 0.0
        mad = float(np.median(np.abs(values - median))) if values.size else 0.0
        support_floor = median + 0.5 * mad
        diagnostics["perturbation_grn_joint_support"] = {
            "outgoing_grn_strength": outgoing,
            "joint_normalized_support": joint,
            "robust_support_floor": support_floor,
            "supported_candidates": [
                gene for gene, value in joint.items()
                if value > 0.0 and value >= support_floor
            ],
        }

    holdout_path = output / "holdout_prediction.csv"
    if holdout_path.exists():
        try:
            holdout = pd.read_csv(holdout_path)
            matrix = holdout[genes].to_numpy(dtype=float)
            observed_std = np.nanstd(expression, axis=0)
            predicted_std = np.nanstd(matrix, axis=0)
            valid = observed_std > 1e-8
            spread_ratio = float(np.median(predicted_std[valid] / observed_std[valid])) if valid.any() else 1.0
            diagnostics["holdout_distribution"] = {
                "n_rows": int(len(holdout)),
                "finite_fraction": float(np.isfinite(matrix).mean()),
                "median_spread_ratio_vs_train": spread_ratio,
                "rows_by_time": {
                    str(key): int(value)
                    for key, value in holdout["time"].value_counts().sort_index().items()
                } if "time" in holdout else {},
            }
            if not np.isfinite(matrix).all():
                _issue(
                    issues,
                    "holdout_nonfinite",
                    "Holdout predictions contain NaN or infinite values.",
                    diagnostics["holdout_distribution"],
                    "Repair the rollout/export path and regenerate finite predictions.",
                    severity="error",
                )
            elif spread_ratio < 0.1:
                _issue(
                    issues,
                    "holdout_distribution_collapse",
                    "Predicted cells have much less diversity than observed cells.",
                    diagnostics["holdout_distribution"],
                    "Use particle/path ensembles and validate distributional coverage on public pseudo-holdout time points; do not export only a mean path.",
                )
        except Exception as exc:
            diagnostics["holdout_audit_error"] = str(exc)

    fate_path = output / "per_cell_fate.json"
    if fate_path.exists():
        try:
            fate = json.loads(fate_path.read_text(encoding="utf-8"))
            rows = list(fate.values()) if isinstance(fate, dict) else []
            labels = sorted({str(label) for row in rows for label in row})
            probabilities = np.asarray(
                [[_finite_float(row.get(label)) for label in labels] for row in rows],
                dtype=float,
            )
            if probabilities.size:
                top_share = float(np.max(np.bincount(np.argmax(probabilities, axis=1))) / len(probabilities))
                diagnostics["fate"] = {
                    "n_cells": len(rows),
                    "labels": labels,
                    "largest_top1_share": top_share,
                    "mean_probability_sum": float(probabilities.sum(axis=1).mean()),
                }
                if len(labels) > 1 and top_share > 0.98:
                    _issue(
                        issues,
                        "fate_collapse",
                        "Almost all source cells collapse to one terminal fate.",
                        diagnostics["fate"],
                        "Check rollout horizon, source-cell provenance, classifier feature order, and branch diversity using public pseudo-holdout validation.",
                    )
        except Exception as exc:
            diagnostics["fate_audit_error"] = str(exc)

    return {
        "version": 1,
        "public_inputs_only": True,
        "train_h5ad": str(Path(train_h5ad).resolve()),
        "output_dir": str(output.resolve()),
        "requires_revision": bool(issues),
        "issue_count": len(issues),
        "issues": issues,
        "diagnostics": _json_ready(diagnostics),
    }


def apply_public_scientific_calibration(
    train_h5ad: str | Path,
    output_dir: str | Path,
    *,
    n_growth_resamples: int = 12,
) -> dict[str, Any]:
    """Calibrate M2/M5 artifacts using only public inputs and model outputs."""
    import anndata as ad

    output = Path(output_dir)
    backup = output / "scientific_harness_raw"
    backup.mkdir(parents=True, exist_ok=True)
    actions: list[dict[str, Any]] = []
    warnings: list[str] = []

    drivers_path = output / "driver_genes.json"
    growth_path = output / "growth_rates.csv"
    perturb_path = output / "perturbation_results.json"
    if not drivers_path.exists():
        return {
            "version": 1,
            "public_inputs_only": True,
            "applied": False,
            "actions": [],
            "warnings": ["driver_genes.json is missing"],
        }

    raw_drivers_path = backup / "driver_genes.json"
    if not raw_drivers_path.exists():
        shutil.copy2(drivers_path, raw_drivers_path)
    if perturb_path.exists() and not (backup / "perturbation_results.json").exists():
        shutil.copy2(perturb_path, backup / "perturbation_results.json")

    try:
        driver_payload = json.loads(raw_drivers_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "version": 1,
            "public_inputs_only": True,
            "applied": False,
            "actions": [],
            "warnings": [f"driver_genes.json is invalid: {exc}"],
        }

    adata = ad.read_h5ad(train_h5ad)
    expression = _dense(adata.X).astype(np.float64, copy=False)
    genes = [str(name) for name in adata.var_names]
    time_key = _first_column(
        adata.obs.columns,
        ("time", "time_point_processed", "Time Point", "time_bin"),
    )
    label_key = _first_column(
        adata.obs.columns,
        ("fate_true", "cell_type", "celltype", "label"),
    )

    if growth_path.exists():
        try:
            growth = pd.read_csv(growth_path)["growth_rate"].to_numpy(dtype=float)
            stable = _stable_growth_surrogate_scores(
                expression,
                growth,
                pd.to_numeric(adata.obs[time_key], errors="coerce").to_numpy(dtype=float)
                if time_key else None,
                adata.obs[label_key].astype(str).to_numpy() if label_key else None,
                n_resamples=n_growth_resamples,
            )
            calibrated = {
                gene: _finite_float(stable["mean"][idx])
                for idx, gene in enumerate(genes)
            }
            raw_growth_drivers = {
                str(gene): _finite_float(score)
                for gene, score in (driver_payload.get("growth_drivers") or {}).items()
            }
            driver_payload["growth_drivers"] = calibrated
            driver_payload["scientific_harness_growth_calibration"] = {
                "method": "stratified_bootstrap_extra_trees_surrogate_of_model_growth_rate",
                "n_resamples": max(3, int(n_growth_resamples)),
                "nuisance_features": [name for name in (time_key, label_key) if name],
                "raw_growth_drivers": raw_growth_drivers,
                "importance_std": {
                    gene: _finite_float(stable["std"][idx])
                    for idx, gene in enumerate(genes)
                },
                "top_frequency": {
                    gene: _finite_float(stable["top_frequency"][idx])
                    for idx, gene in enumerate(genes)
                },
            }
            actions.append(
                {
                    "artifact": "driver_genes.json",
                    "field": "growth_drivers",
                    "method": "stable nonlinear surrogate of model growth output",
                    "raw_top": max(raw_growth_drivers, key=raw_growth_drivers.get)
                    if raw_growth_drivers else None,
                    "calibrated_top": max(calibrated, key=calibrated.get),
                }
            )
        except Exception as exc:
            warnings.append(f"growth calibration skipped: {exc}")

    if perturb_path.exists():
        try:
            perturbations = json.loads((backup / "perturbation_results.json").read_text(encoding="utf-8"))
            effects: dict[str, float] = {}
            for row in perturbations:
                delta = row.get("delta") or {}
                effects[str(row.get("gene_name", ""))] = max(
                    (abs(_finite_float(value)) for value in delta.values()), default=0.0
                )
            outgoing = {gene: 0.0 for gene in genes}
            for edge in driver_payload.get("grn_edges") or []:
                source = str(edge.get("source", ""))
                if source in outgoing:
                    outgoing[source] += abs(_finite_float(edge.get("score")))
            max_effect = max(effects.values(), default=0.0)
            max_outgoing = max(outgoing.values(), default=0.0)
            if max_effect > 0 and max_outgoing > 0 and len(effects) >= 3:
                joint = {
                    gene: (effects.get(gene, 0.0) / max_effect)
                    * (outgoing.get(gene, 0.0) / max_outgoing)
                    for gene in effects
                }
                values = np.asarray(list(joint.values()), dtype=float)
                median = float(np.median(values))
                mad = float(np.median(np.abs(values - median)))
                threshold = median + 0.5 * mad
                supported = {gene for gene, score in joint.items() if score >= threshold}
                # Always retain at least one evidence-supported intervention.
                if not supported and joint:
                    supported = {max(joint, key=joint.get)}
                suppressed = []
                for row in perturbations:
                    gene = str(row.get("gene_name", ""))
                    row["scientific_harness_raw_delta"] = dict(row.get("delta") or {})
                    row["scientific_harness_joint_support"] = _finite_float(joint.get(gene))
                    row["scientific_harness_supported"] = gene in supported
                    if gene in supported:
                        continue
                    control = {
                        str(fate): _finite_float(value)
                        for fate, value in (row.get("control") or {}).items()
                    }
                    row["perturbed"] = dict(control)
                    row["delta"] = {fate: 0.0 for fate in control}
                    suppressed.append(gene)
                perturb_path.write_text(
                    json.dumps(perturbations, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                actions.append(
                    {
                        "artifact": "perturbation_results.json",
                        "field": "delta",
                        "method": "robust consensus of intervention magnitude and outgoing GRN support",
                        "support_threshold": threshold,
                        "supported_genes": sorted(supported),
                        "suppressed_genes": sorted(suppressed),
                        "joint_support": joint,
                    }
                )
        except Exception as exc:
            warnings.append(f"perturbation calibration skipped: {exc}")

    drivers_path.write_text(
        json.dumps(driver_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "version": 1,
        "public_inputs_only": True,
        "applied": bool(actions),
        "train_h5ad": str(Path(train_h5ad).resolve()),
        "raw_backup_dir": str(backup.resolve()),
        "actions": actions,
        "warnings": warnings,
    }
    (output / "scientific_harness_calibration.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def write_public_audit(
    train_h5ad: str | Path,
    output_dir: str | Path,
    audit_path: str | Path,
) -> dict[str, Any]:
    audit = audit_public_outputs(train_h5ad, output_dir)
    destination = Path(audit_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def build_revision_prompt(audit: dict[str, Any], audit_path: str | Path) -> str:
    issues = audit.get("issues") or []
    rendered = "\n".join(
        f"- [{item.get('severity', 'warning')}] {item.get('summary')}\n"
        f"  Evidence: {json.dumps(item.get('evidence') or {}, ensure_ascii=False)}\n"
        f"  Action: {item.get('recommended_action')}"
        for item in issues
    )
    return f"""
The first-pass deliverables now exist. A public-data-only scientific audit was
run and saved at `{Path(audit_path)}`. It did not read benchmark ground truth,
reference outputs, evaluator internals, or scores.

Audit findings:
{rendered or '- No scientific warning was detected.'}

Revise the weak artifacts and their generating script now. Preserve outputs
that already have strong diagnostics. Use model-native calculations, public
pseudo-holdout validation, bootstrap/stability evidence, and coherent
cross-artifact reasoning. Do not fabricate values or tune against hidden
answers. After revisions, rerun the public format verifier and leave all six
final files directly in the requested output directory.
"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the public-data-only DynBench scientific harness audit."
    )
    parser.add_argument("--train-h5ad", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-path", default=None)
    parser.add_argument(
        "--apply-calibration",
        action="store_true",
        help="Apply public-only M2/M5 calibration before auditing (raw JSON files are backed up)",
    )
    args = parser.parse_args()

    if args.apply_calibration:
        calibration = apply_public_scientific_calibration(args.train_h5ad, args.output_dir)
        print("Calibration:")
        print(json.dumps(calibration, indent=2, ensure_ascii=False))
        print()
    audit_path = Path(args.audit_path) if args.audit_path else Path(args.output_dir) / "scientific_harness_audit_manual.json"
    audit = write_public_audit(args.train_h5ad, args.output_dir, audit_path)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"\nAudit saved to: {audit_path}")


if __name__ == "__main__":
    main()
