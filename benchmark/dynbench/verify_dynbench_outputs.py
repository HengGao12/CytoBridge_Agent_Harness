#!/usr/bin/env python
"""Public DynBench output-format verifier.

This script is intended for benchmark agents to run before final delivery. It
uses only public task-package inputs and the submitted output directory. It does
not read hidden answers, does not evaluate scientific correctness, and
does not repair malformed files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_OUTPUT_FILES = (
    "velocity_field.csv",
    "growth_rates.csv",
    "holdout_prediction.csv",
    "per_cell_fate.json",
    "perturbation_results.json",
    "driver_genes.json",
)


def _load_anndata(path: Path):
    try:
        import anndata
    except Exception as exc:
        raise RuntimeError("anndata is required to verify DynBench outputs") from exc
    return anndata.read_h5ad(path)


def _read_csv(output_dir: Path, name: str) -> pd.DataFrame:
    path = output_dir / name
    if not path.exists():
        raise ValueError(f"{name}: missing")
    if path.stat().st_size <= 0:
        raise ValueError(f"{name}: empty file")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"{name}: invalid CSV: {exc}") from exc
    if df.empty:
        raise ValueError(f"{name}: zero rows")
    implicit_index = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if implicit_index:
        raise ValueError(f"{name}: implicit pandas index column present; write with index=False")
    return df


def _read_json(output_dir: Path, name: str) -> Any:
    path = output_dir / name
    if not path.exists():
        raise ValueError(f"{name}: missing")
    if path.stat().st_size <= 0:
        raise ValueError(f"{name}: empty file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{name}: invalid JSON: {exc}") from exc


def _ensure_finite_numeric(df: pd.DataFrame, columns: list[str], *, name: str) -> None:
    try:
        values = df[columns].to_numpy(dtype=float)
    except Exception as exc:
        raise ValueError(f"{name}: required columns are not numeric: {exc}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: contains non-finite numeric values")


def _root_cell_ids(adata) -> set[str]:
    if "time_bin" in adata.obs.columns:
        time_bin = pd.to_numeric(adata.obs["time_bin"], errors="coerce")
        if time_bin.notna().any():
            return set(adata.obs_names[time_bin == time_bin.min()].astype(str).tolist())
    if "time" in adata.obs.columns:
        time = pd.to_numeric(adata.obs["time"], errors="coerce")
        if time.notna().any():
            return set(adata.obs_names[time == time.min()].astype(str).tolist())
    raise ValueError("train.h5ad must contain time_bin or time to identify root cells")


def _validate_velocity(output_dir: Path, gene_names: list[str], n_train: int) -> dict[str, Any]:
    name = "velocity_field.csv"
    df = _read_csv(output_dir, name)
    expected_cols = [f"velocity_{g}" for g in gene_names]
    if list(df.columns) != expected_cols:
        raise ValueError(f"{name}: expected columns {expected_cols}, got {list(df.columns)}")
    if len(df) != n_train:
        raise ValueError(f"{name}: row count {len(df)} != train cells {n_train}")
    _ensure_finite_numeric(df, expected_cols, name=name)
    return {"rows": int(len(df)), "columns": list(df.columns)}


def _validate_growth(output_dir: Path, n_train: int) -> dict[str, Any]:
    name = "growth_rates.csv"
    df = _read_csv(output_dir, name)
    if list(df.columns) != ["growth_rate"]:
        raise ValueError(f"{name}: expected exactly ['growth_rate'], got {list(df.columns)}")
    if len(df) != n_train:
        raise ValueError(f"{name}: row count {len(df)} != train cells {n_train}")
    _ensure_finite_numeric(df, ["growth_rate"], name=name)
    return {"rows": int(len(df)), "columns": list(df.columns)}


def _validate_holdout(output_dir: Path, gene_names: list[str]) -> dict[str, Any]:
    name = "holdout_prediction.csv"
    df = _read_csv(output_dir, name)
    allowed = set(gene_names) | {"time", "weight"}
    missing = [g for g in gene_names if g not in df.columns]
    extra = [c for c in df.columns if c not in allowed]
    if missing or extra:
        raise ValueError(f"{name}: missing gene columns={missing}, extra columns={extra}")
    if "time" not in df.columns:
        raise ValueError(f"{name}: missing required time column")
    numeric_cols = gene_names + ["time"] + (["weight"] if "weight" in df.columns else [])
    _ensure_finite_numeric(df, numeric_cols, name=name)
    if "weight" in df.columns and (df["weight"].astype(float) < 0).any():
        raise ValueError(f"{name}: weight column contains negative values")
    return {"rows": int(len(df)), "columns": list(df.columns)}


def _validate_fate(output_dir: Path, root_ids: set[str]) -> dict[str, Any]:
    name = "per_cell_fate.json"
    payload = _read_json(output_dir, name)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{name}: expected non-empty object mapping cell id to fate probabilities")
    keys = set(payload.keys())
    missing = sorted(root_ids - keys)
    extra = sorted(keys - root_ids)
    if missing or extra:
        raise ValueError(
            f"{name}: cell ids must exactly match root cells; missing={missing[:5]}, extra={extra[:5]}"
        )
    for cell_id, probs in payload.items():
        if not isinstance(probs, dict) or not probs:
            raise ValueError(f"{name}: {cell_id} must map to a non-empty fate probability object")
        total = 0.0
        for fate, prob in probs.items():
            if not isinstance(fate, str):
                raise ValueError(f"{name}: fate names must be strings")
            try:
                value = float(prob)
            except Exception as exc:
                raise ValueError(f"{name}: non-numeric probability for {cell_id}/{fate}") from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name}: invalid probability for {cell_id}/{fate}: {prob}")
            total += value
        if total <= 0 or total > 1.05:
            raise ValueError(f"{name}: probabilities for {cell_id} sum to {total:.6g}, expected (0, 1.05]")
    return {"cells": int(len(payload))}


def _validate_perturbation(output_dir: Path, gene_names: list[str]) -> dict[str, Any]:
    name = "perturbation_results.json"
    payload = _read_json(output_dir, name)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{name}: expected non-empty list")
    by_gene = {}
    for row in payload:
        if not isinstance(row, dict) or "gene_name" not in row or "delta" not in row:
            raise ValueError(f"{name}: each row must contain gene_name and delta")
        gene = row["gene_name"]
        if gene not in gene_names:
            raise ValueError(f"{name}: unknown gene {gene}")
        if gene in by_gene:
            raise ValueError(f"{name}: duplicate gene {gene}")
        if not isinstance(row["delta"], dict) or not row["delta"]:
            raise ValueError(f"{name}: delta for {gene} must be a non-empty object")
        for fate, value in row["delta"].items():
            if not isinstance(fate, str):
                raise ValueError(f"{name}: delta fate names must be strings")
            try:
                numeric = float(value)
            except Exception as exc:
                raise ValueError(f"{name}: non-numeric delta for {gene}/{fate}") from exc
            if not np.isfinite(numeric):
                raise ValueError(f"{name}: non-finite delta for {gene}/{fate}: {value}")
        by_gene[gene] = row
    missing = [g for g in gene_names if g not in by_gene]
    if missing:
        raise ValueError(f"{name}: missing genes {missing}")
    return {"genes": int(len(by_gene))}


def _read_driver_payload(output_dir: Path) -> dict[str, Any]:
    name = "driver_genes.json"
    payload = _read_json(output_dir, name)
    if not isinstance(payload, dict):
        raise ValueError(f"{name}: expected object")
    return payload


def _validate_growth_drivers(output_dir: Path, gene_names: list[str]) -> dict[str, Any]:
    name = "driver_genes.json"
    payload = _read_driver_payload(output_dir)
    growth_drivers = payload.get("growth_drivers")
    if not isinstance(growth_drivers, dict):
        raise ValueError(f"{name}: missing growth_drivers object")
    missing = [g for g in gene_names if g not in growth_drivers]
    if missing:
        raise ValueError(f"{name}: growth_drivers missing genes {missing}")
    for gene, score in growth_drivers.items():
        if gene not in gene_names:
            raise ValueError(f"{name}: growth_drivers contains unknown gene {gene}")
        try:
            numeric = float(score)
        except Exception as exc:
            raise ValueError(f"{name}: growth driver score for {gene} is non-numeric") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"{name}: growth driver score for {gene} is non-finite")
    return {"growth_driver_genes": int(len(growth_drivers))}


def _validate_grn_edges(output_dir: Path, gene_names: list[str]) -> dict[str, Any]:
    name = "driver_genes.json"
    payload = _read_driver_payload(output_dir)
    grn_edges = payload.get("grn_edges")
    if not isinstance(grn_edges, list):
        raise ValueError(f"{name}: missing grn_edges list")
    for edge in grn_edges:
        if not isinstance(edge, dict) or not {"source", "target", "score"} <= set(edge):
            raise ValueError(f"{name}: each grn edge must contain source, target, and score")
        if edge["source"] not in gene_names or edge["target"] not in gene_names:
            raise ValueError(f"{name}: grn edge contains unknown gene: {edge}")
        try:
            numeric = float(edge["score"])
        except Exception as exc:
            raise ValueError(f"{name}: non-numeric grn edge score: {edge}") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"{name}: non-finite grn edge score: {edge}")
    return {"grn_edges": int(len(grn_edges))}


def verify_outputs(output_dir: Path, train_h5ad: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    train_h5ad = train_h5ad.resolve()
    adata = _load_anndata(train_h5ad)
    gene_names = [str(g) for g in adata.var_names.tolist()]
    root_ids = _root_cell_ids(adata)

    checks = {
        "velocity_field.csv": lambda: _validate_velocity(output_dir, gene_names, int(adata.n_obs)),
        "growth_rates.csv": lambda: _validate_growth(output_dir, int(adata.n_obs)),
        "holdout_prediction.csv": lambda: _validate_holdout(output_dir, gene_names),
        "per_cell_fate.json": lambda: _validate_fate(output_dir, root_ids),
        "perturbation_results.json": lambda: _validate_perturbation(output_dir, gene_names),
        "driver_genes.json": lambda: {
            "growth_drivers": _validate_growth_drivers(output_dir, gene_names),
            "grn_edges": _validate_grn_edges(output_dir, gene_names),
        },
    }

    report = {
        "ok": True,
        "output_dir": str(output_dir),
        "train_h5ad": str(train_h5ad),
        "n_train_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_root_cells": int(len(root_ids)),
        "files": {},
    }
    for filename, fn in checks.items():
        try:
            detail = fn()
            report["files"][filename] = {"ok": True, "detail": detail}
        except Exception as exc:
            report["ok"] = False
            report["files"][filename] = {"ok": False, "error": str(exc)}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory containing the six DynBench output files")
    parser.add_argument("--train-h5ad", required=True, help="Public task-package train.h5ad")
    parser.add_argument("--json-report", default=None, help="Optional path to write a machine-readable verification report")
    args = parser.parse_args(argv)

    report = verify_outputs(Path(args.output_dir), Path(args.train_h5ad))
    if args.json_report:
        Path(args.json_report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if report["ok"]:
        print("DynBench output format verification passed.", file=sys.stderr)
        return 0
    print("DynBench output format verification failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
