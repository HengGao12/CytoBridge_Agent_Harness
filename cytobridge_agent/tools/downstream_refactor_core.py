"""Downstream core orchestrator that delegates to package-native downstream APIs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from anndata import AnnData

from .umap_policy import build_umap_policy, compact_policy_signature


@dataclass
class DownstreamResult:
    title: str
    summary: List[str]
    artifacts: Dict[str, str]
    warnings: List[str]
    payload: Dict[str, Any]

    def render_text(self) -> str:
        lines = [
            "=" * 60,
            self.title,
            "=" * 60,
            "",
            "Summary:",
        ]
        lines.extend([f"  - {x}" for x in self.summary] if self.summary else ["  - (none)"])
        if self.warnings:
            lines.extend(["", "Warnings:"])
            lines.extend([f"  - {x}" for x in self.warnings])
        if self.artifacts:
            lines.extend(["", "Generated artifacts:"])
            for _, p in self.artifacts.items():
                lines.append(f" - {p}")
        lines.append("=" * 60)
        return "\n".join(lines)


class DownstreamRefactorCore:
    """Thin orchestrator over ``CytoBridge.tl/pl.downstream`` modules."""

    def __init__(self, output_dir: Path, device: str = "cpu") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def _resolve_save_dir(self, save_dir: Optional[str], subdir: str) -> Path:
        if save_dir and str(save_dir).strip():
            out = Path(save_dir).expanduser().resolve()
        else:
            out = self.output_dir / subdir
        out.mkdir(parents=True, exist_ok=True)
        return out

    @staticmethod
    def _preflight(adata: AnnData, required: Sequence[str], where: str) -> None:
        from CytoBridge.tl.downstream.contracts import ensure_contract

        ensure_contract(adata, required, where=where)

    @staticmethod
    def _merge_artifacts(dst: Dict[str, str], src: Optional[Dict[str, Any]], prefix: str = "") -> None:
        if not isinstance(src, dict):
            return
        for k, v in src.items():
            if isinstance(v, str):
                dst[f"{prefix}{k}" if prefix else k] = v

    @staticmethod
    def _collect_warnings(*parts: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for p in parts:
            ws = p.get("warnings") if isinstance(p, dict) else None
            if isinstance(ws, list):
                out.extend([str(x) for x in ws if str(x).strip()])
        # stable unique order
        seen = set()
        uniq = []
        for w in out:
            if w in seen:
                continue
            seen.add(w)
            uniq.append(w)
        return uniq

    @staticmethod
    def _normalize_basis_name(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        basis = str(name).strip().lower()
        if basis.startswith("x_"):
            basis = basis[2:]
        if basis == "sring":
            basis = "spring"
        return basis or None

    @staticmethod
    def _has_embedding(adata: AnnData, basis: str) -> bool:
        key = f"X_{basis}"
        if key not in adata.obsm:
            return False
        try:
            return int(adata.obsm[key].shape[1]) >= 2
        except Exception:
            return False

    @staticmethod
    def _available_bases(adata: AnnData) -> List[str]:
        bases: List[str] = []
        for key in getattr(adata, "obsm", {}).keys():
            k = str(key)
            if not k.startswith("X_"):
                continue
            basis = k[2:]
            if DownstreamRefactorCore._has_embedding(adata, basis):
                bases.append(basis)
        return sorted(set(bases))

    @staticmethod
    def _select_plot_basis(adata: AnnData, preferred: Optional[str] = None, strict: bool = False) -> str:
        basis = DownstreamRefactorCore._normalize_basis_name(preferred)
        if basis:
            if basis == "umap":
                return "umap"
            if basis == "pca":
                return "pca"
            if basis == "latent":
                return "latent"
            if basis == "fast":
                return "fast"
            if DownstreamRefactorCore._has_embedding(adata, basis):
                return basis
            if strict and basis not in {"umap", "pca", "latent", "fast"}:
                available = DownstreamRefactorCore._available_bases(adata)
                raise ValueError(
                    f"Requested plot basis '{basis}' not found. Available bases: {available or ['(none)']}"
                )

        if "X_umap" in adata.obsm:
            return "umap"
        # Auto-prefer UMAP fallback when latent/PCA exists: _ensure_embedding()
        # will compute X_umap from available representation.
        if DownstreamRefactorCore._has_embedding(adata, "latent"):
            return "umap"
        if DownstreamRefactorCore._has_embedding(adata, "pca"):
            return "umap"
        return "umap"

    @staticmethod
    def _ensure_embedding(adata: AnnData, preferred: Optional[str] = None, strict: bool = False) -> str:
        # Keep the same decision order as DownstreamAnalysisToolkit._ensure_embedding.
        import numpy as np
        import scanpy as sc

        requested_basis = DownstreamRefactorCore._normalize_basis_name(preferred)
        basis = DownstreamRefactorCore._select_plot_basis(adata, preferred=preferred, strict=strict)
        if basis == "latent":
            if not DownstreamRefactorCore._has_embedding(adata, "latent"):
                if strict and requested_basis == "latent":
                    raise ValueError("Requested plot basis 'latent' is unavailable (X_latent missing or <2D).")
                basis = "umap"
        if basis == "fast":
            if not DownstreamRefactorCore._has_embedding(adata, "fast"):
                if DownstreamRefactorCore._has_embedding(adata, "latent"):
                    adata.obsm["X_fast"] = np.asarray(adata.obsm["X_latent"], dtype=np.float32)[:, :2]
                elif DownstreamRefactorCore._has_embedding(adata, "pca"):
                    adata.obsm["X_fast"] = np.asarray(adata.obsm["X_pca"], dtype=np.float32)[:, :2]
                elif DownstreamRefactorCore._has_embedding(adata, "umap"):
                    adata.obsm["X_fast"] = np.asarray(adata.obsm["X_umap"], dtype=np.float32)[:, :2]
                elif strict and requested_basis == "fast":
                    raise ValueError("Requested plot basis 'fast' is unavailable (no latent/pca/umap to derive X_fast).")
                else:
                    basis = "umap"
        if basis == "pca":
            if "X_pca" not in adata.obsm:
                sc.pp.pca(adata)
        if basis == "umap":
            if "X_umap" not in adata.obsm:
                use_rep = None
                if "X_latent" in adata.obsm:
                    use_rep = "X_latent"
                elif "X_pca" in adata.obsm:
                    use_rep = "X_pca"
                policy = build_umap_policy(
                    n_obs=int(adata.n_obs),
                    use_rep=use_rep,
                    quality_preset="publication",
                    viz_goal="publication",
                    mode="auto",
                    overrides=None,
                )
                neighbors_cfg = policy.get("neighbors", {}) if isinstance(policy, dict) else {}
                umap_cfg = policy.get("umap", {}) if isinstance(policy, dict) else {}
                rep = str(neighbors_cfg.get("use_rep") or use_rep or "X")
                n_neighbors = int(neighbors_cfg.get("n_neighbors", 30))
                metric = str(neighbors_cfg.get("metric", "euclidean"))
                conn = adata.obsp.get("connectivities") if hasattr(adata, "obsp") else None
                needs_neighbors = ("neighbors" not in adata.uns or conn is None)
                if not needs_neighbors:
                    params = {}
                    try:
                        params = dict((adata.uns.get("neighbors") or {}).get("params") or {})
                    except Exception:
                        params = {}
                    existing_rep = str(params.get("use_rep") or "X")
                    if existing_rep == "None":
                        existing_rep = "X"
                    existing_metric = str(params.get("metric") or "")
                    try:
                        existing_k = int(params.get("n_neighbors")) if params.get("n_neighbors") is not None else None
                    except Exception:
                        existing_k = None
                    if existing_rep != rep:
                        needs_neighbors = True
                    if existing_k is not None and existing_k != n_neighbors:
                        needs_neighbors = True
                    if existing_metric and existing_metric != metric:
                        needs_neighbors = True
                if needs_neighbors:
                    if rep == "X":
                        sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric=metric)
                    else:
                        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=rep, metric=metric)
                sc.tl.umap(
                    adata,
                    min_dist=float(umap_cfg.get("min_dist", 0.3)),
                    spread=float(umap_cfg.get("spread", 1.0)),
                    random_state=int(umap_cfg.get("random_state", 0)),
                )
                adata.uns["_cytobridge_umap_config"] = compact_policy_signature(policy)
            elif not isinstance(adata.uns.get("_cytobridge_umap_config"), dict):
                adata.uns["_cytobridge_umap_config"] = {
                    "version": 1,
                    "mode": "external",
                    "use_rep": "",
                    "n_neighbors": -1,
                    "metric": "",
                    "min_dist": -1.0,
                    "spread": -1.0,
                    "random_state": -1,
                }
        if basis not in {"umap", "pca", "latent", "fast"} and not DownstreamRefactorCore._has_embedding(adata, basis):
            if strict and requested_basis == basis:
                available = DownstreamRefactorCore._available_bases(adata)
                raise ValueError(
                    f"Requested plot basis '{basis}' is unavailable after embedding checks. "
                    f"Available bases: {available or ['(none)']}"
                )
            basis = "umap"
        return basis

    def analyze_trajectory_fate(
        self,
        adata: AnnData,
        label_key: Optional[str] = None,
        mode: str = "fast",
        include_gene_projection: bool = False,
        include_sde: bool = False,
        save_dir: Optional[str] = None,
        n_pcs: int = 50,
        preferred_basis: Optional[str] = None,
        strict_basis: bool = False,
    ) -> DownstreamResult:
        from CytoBridge.tl.downstream.velocity import (
            build_velocity_graph_bundle,
            compute_velocity_bundle,
            summarize_velocity_drivers_bundle,
        )
        from CytoBridge.tl.downstream.trajectory import generate_sde_trajectory_bundle
        from CytoBridge.pl.downstream.velocity_plot import plot_velocity_stream_bundle
        from CytoBridge.tl.downstream.contracts import LATENT_KEY, TIME_KEY

        outdir = self._resolve_save_dir(save_dir, "trajectory")
        artifacts: Dict[str, str] = {}
        self._preflight(adata, [LATENT_KEY, TIME_KEY], where="downstream_refactor_core.analyze_trajectory_fate")

        # Prefer reusing existing latent velocity when present (retina fast path).
        if "velocity_latent" in adata.obsm and adata.obsm["velocity_latent"] is not None:
            vel = {
                "space_used": "latent",
                "projection_backend": "none",
                "artifacts": {},
                "warnings": ["reuse existing velocity_latent"],
            }
        else:
            vel = compute_velocity_bundle(
                adata=adata,
                model=None,
                device=self.device,
                output_dir=str(outdir / "velocity"),
            )
        plot_basis = self._ensure_embedding(
            adata,
            preferred=preferred_basis,
            strict=bool(strict_basis),
        )
        # velocity_plot expects basis names in {umap,pca,fast}; latent maps to fast.
        graph_basis = "fast" if plot_basis == "latent" else plot_basis
        graph = build_velocity_graph_bundle(
            adata=adata,
            output_dir=str(outdir / "velocity_graph"),
            n_pcs=int(max(2, n_pcs)),
            n_neighbors=30,
            reuse_neighbors=True,
            preferred_basis=graph_basis,
            strict_preferred=bool(strict_basis),
        )
        plot = plot_velocity_stream_bundle(
            adata=adata,
            model=None,
            output_dir=str(self.figures_dir),
            dim_reduction=str(graph.get("basis", graph_basis)),
            device=self.device,
            color_key=label_key,
            vkey="velocity",
        )

        self._merge_artifacts(artifacts, vel.get("artifacts"), prefix="velocity_")
        self._merge_artifacts(artifacts, graph.get("artifacts"), prefix="graph_")
        self._merge_artifacts(artifacts, plot.get("artifacts"), prefix="plot_")

        summary = [
            f"velocity space={vel.get('space_used', 'latent')} projection={vel.get('projection_backend', 'none')}",
            f"graph basis={graph.get('basis', 'unknown')} rep={graph.get('representation', 'unknown')}",
        ]
        if preferred_basis:
            summary.append(
                f"requested basis={self._normalize_basis_name(preferred_basis)} strict={bool(strict_basis)}"
            )

        if include_gene_projection:
            vd = summarize_velocity_drivers_bundle(
                adata=adata,
                output_dir=str(outdir / "gene_velocity_summary"),
                analysis_space="gene",
                top_n=50,
            )
            self._merge_artifacts(artifacts, vd.get("artifacts"), prefix="gene_summary_")
            top = vd.get("top") or []
            if isinstance(top, list) and len(top) > 0:
                t0 = top[0]
                summary.append(
                    f"gene summary top1={t0.get('feature_name', 'unknown')} mean_abs_velocity={float(t0.get('mean_abs_velocity', 0.0)):.5f}"
                )

        mode_norm = str(mode or "fast").strip().lower()
        if include_sde or mode_norm == "full":
            sde = generate_sde_trajectory_bundle(
                adata=adata,
                output_dir=str(outdir / "sde"),
                n_time_steps=40,
                sample_traj_num=400,
                init_time=0,
                device=self.device,
            )
            self._merge_artifacts(artifacts, sde.get("artifacts"), prefix="sde_")
            summary.append("SDE trajectories generated")
        else:
            sde = {}

        warnings = self._collect_warnings(vel, graph, plot, sde)
        payload = {
            "space_used": vel.get("space_used", "latent"),
            "projection_backend": vel.get("projection_backend", "none"),
            "graph_basis": graph.get("basis", "unknown"),
            "graph_representation": graph.get("representation", "unknown"),
            "requested_basis": self._normalize_basis_name(preferred_basis) if preferred_basis else None,
            "strict_basis": bool(strict_basis),
            "artifacts": artifacts,
            "warnings": warnings,
        }
        return DownstreamResult(
            title="Trajectory/Fate Analysis",
            summary=summary,
            artifacts=artifacts,
            warnings=warnings,
            payload=payload,
        )

    def analyze_growth_mass(self, adata: AnnData, save_dir: Optional[str] = None) -> DownstreamResult:
        from CytoBridge.tl.downstream.growth import summarize_growth_bundle
        from CytoBridge.pl.downstream.growth_plot import plot_growth_bundle
        from CytoBridge.tl.downstream.contracts import LATENT_KEY, TIME_KEY

        outdir = self._resolve_save_dir(save_dir, "growth")
        artifacts: Dict[str, str] = {}
        self._preflight(adata, [LATENT_KEY, TIME_KEY], where="downstream_refactor_core.analyze_growth_mass")

        stats = summarize_growth_bundle(
            adata=adata,
            key="growth_rate",
            output_dir=str(outdir),
        )
        plot = plot_growth_bundle(
            adata=adata,
            output_dir=str(self.figures_dir),
            key="growth_rate",
        )

        self._merge_artifacts(artifacts, stats.get("artifacts"), prefix="stats_")
        self._merge_artifacts(artifacts, plot.get("artifacts"), prefix="plot_")
        warnings = self._collect_warnings(stats, plot)

        s = stats.get("stats", {}) if isinstance(stats, dict) else {}
        summary = [
            f"mass_cv={float(s.get('mass_cv', 0.0)):.4f}",
            f"time_points={len(s.get('time_points', []))}",
        ]
        if "growth_mean" in s:
            summary.append(f"growth_mean={float(s.get('growth_mean', 0.0)):.5f}, growth_std={float(s.get('growth_std', 0.0)):.5f}")

        payload = {
            "space_used": stats.get("space_used", "latent"),
            "projection_backend": stats.get("projection_backend", "none"),
            "artifacts": artifacts,
            "warnings": warnings,
            "stats": s,
        }
        return DownstreamResult(
            title="Growth/Mass Analysis",
            summary=summary,
            artifacts=artifacts,
            warnings=warnings,
            payload=payload,
        )

    def analyze_growth_driver_genes(
        self,
        adata: AnnData,
        top_n: int = 20,
        max_cells: int = 20000,
        batch_size: int = 2048,
        random_state: int = 0,
        save_dir: Optional[str] = None,
    ) -> DownstreamResult:
        from CytoBridge.tl.downstream.growth import summarize_growth_drivers_bundle
        from CytoBridge.pl.downstream.growth_plot import plot_driver_scores_bundle
        from CytoBridge.tl.downstream.contracts import LATENT_KEY, TIME_KEY

        outdir = self._resolve_save_dir(save_dir, "growth_drivers")
        artifacts: Dict[str, str] = {}
        self._preflight(adata, [LATENT_KEY, TIME_KEY], where="downstream_refactor_core.analyze_growth_driver_genes")

        res = summarize_growth_drivers_bundle(
            adata=adata,
            output_dir=str(outdir),
            top_n=int(top_n),
            max_cells=int(max_cells),
            batch_size=int(batch_size),
            random_state=int(random_state),
            device=self.device,
        )
        rows = res.get("top") or []
        plot = plot_driver_scores_bundle(
            rows=rows,
            output_dir=str(self.figures_dir),
            file_name="growth_driver_top.png",
            value_key="driver_score",
            title=f"Top {len(rows)} growth drivers",
            top_n=len(rows) if len(rows) > 0 else 1,
        )

        self._merge_artifacts(artifacts, res.get("artifacts"), prefix="drivers_")
        self._merge_artifacts(artifacts, plot.get("artifacts"), prefix="plot_")
        warnings = self._collect_warnings(res, plot)

        summary = [
            f"space_used={res.get('space_used', 'latent')} projection={res.get('projection_backend', 'none')}",
            f"top_count={len(rows)}",
        ]
        if len(rows) > 0:
            summary.append(f"top1={rows[0].get('feature_name', 'unknown')} score={float(rows[0].get('driver_score', 0.0)):.5f}")

        payload = {
            "space_used": res.get("space_used", "latent"),
            "projection_backend": res.get("projection_backend", "none"),
            "artifacts": artifacts,
            "warnings": warnings,
            "top": rows,
        }
        return DownstreamResult(
            title="Growth Driver Analysis",
            summary=summary,
            artifacts=artifacts,
            warnings=warnings,
            payload=payload,
        )

    def analyze_velocity_driver_genes(
        self,
        adata: AnnData,
        top_n: int = 20,
        analysis_space: str = "gene",
        save_dir: Optional[str] = None,
    ) -> DownstreamResult:
        from CytoBridge.tl.downstream.velocity import summarize_velocity_drivers_bundle
        from CytoBridge.pl.downstream.growth_plot import plot_driver_scores_bundle
        from CytoBridge.tl.downstream.contracts import LATENT_KEY, TIME_KEY, VELOCITY_LATENT_KEY

        outdir = self._resolve_save_dir(save_dir, "velocity_drivers")
        artifacts: Dict[str, str] = {}
        self._preflight(
            adata,
            [LATENT_KEY, TIME_KEY, VELOCITY_LATENT_KEY],
            where="downstream_refactor_core.analyze_velocity_driver_genes",
        )

        res = summarize_velocity_drivers_bundle(
            adata=adata,
            output_dir=str(outdir),
            analysis_space=analysis_space,
            top_n=int(top_n),
        )
        rows = res.get("top") or []
        plot = plot_driver_scores_bundle(
            rows=rows,
            output_dir=str(self.figures_dir),
            file_name="velocity_driver_top.png",
            value_key="mean_abs_velocity",
            title=f"Top {len(rows)} velocity drivers",
            top_n=len(rows) if len(rows) > 0 else 1,
        )

        self._merge_artifacts(artifacts, res.get("artifacts"), prefix="drivers_")
        self._merge_artifacts(artifacts, plot.get("artifacts"), prefix="plot_")
        warnings = self._collect_warnings(res, plot)

        summary = [
            f"space_used={res.get('space_used', 'latent')} projection={res.get('projection_backend', 'none')}",
            f"top_count={len(rows)}",
        ]
        if len(rows) > 0:
            summary.append(
                f"top1={rows[0].get('feature_name', 'unknown')} mean_abs_velocity={float(rows[0].get('mean_abs_velocity', 0.0)):.5f}"
            )

        payload = {
            "space_used": res.get("space_used", "latent"),
            "projection_backend": res.get("projection_backend", "none"),
            "artifacts": artifacts,
            "warnings": warnings,
            "top": rows,
        }
        return DownstreamResult(
            title="Velocity Driver Analysis",
            summary=summary,
            artifacts=artifacts,
            warnings=warnings,
            payload=payload,
        )

    def analyze_grn(
        self,
        adata: AnnData,
        max_genes: int = 10,
        max_time_points: int = 5,
        genes: Optional[Sequence[str]] = None,
        save_dir: Optional[str] = None,
    ) -> DownstreamResult:
        from CytoBridge.tl.downstream.grn import analyze_grn_bundle
        from CytoBridge.pl.downstream.grn_plot import plot_grn_timeseries_bundle
        from CytoBridge.tl.downstream.contracts import LATENT_KEY, TIME_KEY

        outdir = self._resolve_save_dir(save_dir, "grn")
        artifacts: Dict[str, str] = {}
        self._preflight(adata, [LATENT_KEY, TIME_KEY], where="downstream_refactor_core.analyze_grn")

        res = analyze_grn_bundle(
            adata=adata,
            output_dir=str(outdir),
            max_genes=int(max_genes),
            max_time_points=int(max_time_points),
            genes=genes,
            device=self.device,
        )
        plot = plot_grn_timeseries_bundle(
            grn_payload=res.get("grn", {}),
            output_dir=str(self.figures_dir),
            prefix="grn",
        )

        self._merge_artifacts(artifacts, res.get("artifacts"), prefix="grn_")
        self._merge_artifacts(artifacts, plot.get("artifacts"), prefix="plot_")
        warnings = self._collect_warnings(res, plot)

        summary = [
            f"space_used={res.get('space_used', 'latent')} projection={res.get('projection_backend', 'none')}",
            f"time_points={len(res.get('grn', {}))}",
            f"features={len(res.get('features', []))}",
        ]

        payload = {
            "space_used": res.get("space_used", "latent"),
            "projection_backend": res.get("projection_backend", "none"),
            "artifacts": artifacts,
            "warnings": warnings,
            "grn": res.get("grn", {}),
            "features": res.get("features", []),
        }
        return DownstreamResult(
            title="GRN Analysis",
            summary=summary,
            artifacts=artifacts,
            warnings=warnings,
            payload=payload,
        )
