"""Centralized plotting style and export helpers for downstream figures."""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

COLORBLIND_SAFE_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


def normalize_preset(preset: Optional[str]) -> str:
    s = str(preset or "publication").strip().lower()
    if s not in {"publication", "balanced", "fast"}:
        return "publication"
    return s


def get_preset_config(preset: Optional[str]) -> Dict[str, Any]:
    selected = normalize_preset(preset)
    if selected == "fast":
        return {
            "name": selected,
            "dpi": 150,
            "formats": ("png",),
            "font_family": "Helvetica, Arial, sans-serif",
            "font_size": 11,
            "line_width": 1.0,
            "scale": 1.5,
        }
    if selected == "balanced":
        return {
            "name": selected,
            "dpi": 220,
            "formats": ("png", "svg"),
            "font_family": "Helvetica, Arial, sans-serif",
            "font_size": 12,
            "line_width": 1.2,
            "scale": 1.75,
        }
    return {
        "name": "publication",
        "dpi": 300,
        "formats": ("png", "svg"),
        "font_family": "Helvetica, Arial, sans-serif",
        "font_size": 12,
        "line_width": 1.4,
        "scale": 2.0,
    }


def apply_matplotlib_style(preset: Optional[str] = None) -> Dict[str, Any]:
    cfg = get_preset_config(preset)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#111111",
            "axes.labelcolor": "#111111",
            "text.color": "#111111",
            "axes.grid": False,
            "grid.color": "#DDDDDD",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": float(cfg["font_size"]),
            "axes.titlesize": float(cfg["font_size"]) + 1,
            "axes.labelsize": float(cfg["font_size"]),
            "xtick.labelsize": float(cfg["font_size"]) - 1,
            "ytick.labelsize": float(cfg["font_size"]) - 1,
            "legend.fontsize": float(cfg["font_size"]) - 1,
            "axes.linewidth": float(cfg["line_width"]),
            "lines.linewidth": float(cfg["line_width"]),
            "patch.edgecolor": "none",
            "image.cmap": "viridis",
            "axes.prop_cycle": plt.cycler(color=COLORBLIND_SAFE_PALETTE),
        }
    )
    return cfg


def _normalize_formats(formats: Optional[Sequence[str]], fallback: Sequence[str]) -> List[str]:
    if not formats:
        return [str(x).lower() for x in fallback]
    out: List[str] = []
    for raw in formats:
        s = str(raw).strip().lower()
        if not s:
            continue
        if s == "both":
            for item in ("png", "svg"):
                if item not in out:
                    out.append(item)
            continue
        if s == "all":
            for item in ("png", "svg", "pdf"):
                if item not in out:
                    out.append(item)
            continue
        if s in {"png", "svg", "pdf", "html"} and s not in out:
            out.append(s)
    return out or [str(x).lower() for x in fallback]


def build_figure_stem(analysis: str, plot_type: str, variant: str = "default") -> str:
    """Build normalized stem: <analysis>__<plot_type>__<variant>."""
    def _norm(token: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9]+", "_", str(token or "").strip().lower())
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "unknown"
    return f"{_norm(analysis)}__{_norm(plot_type)}__{_norm(variant)}"


def save_figure_bundle(
    fig: Any,
    base_name: str,
    output_dir: Path,
    *,
    preset: Optional[str] = None,
    formats: Optional[Sequence[str]] = None,
    dpi: Optional[int] = None,
    bbox_inches: str = "tight",
) -> Dict[str, str]:
    cfg = get_preset_config(preset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = str(base_name or "figure").strip().replace(" ", "_")
    export_formats = _normalize_formats(formats, cfg["formats"])
    export_dpi = int(dpi or cfg["dpi"])

    artifacts: Dict[str, str] = {}
    for fmt in export_formats:
        if fmt == "html":
            continue
        out_path = output_dir / f"{base}.{fmt}"
        fig.savefig(out_path, dpi=export_dpi, bbox_inches=bbox_inches)
        artifacts[fmt] = str(out_path)
    return artifacts


def save_plotly_bundle(
    fig: Any,
    base_name: str,
    output_dir: Path,
    *,
    preset: Optional[str] = None,
    formats: Optional[Sequence[str]] = None,
    width: int = 1600,
    height: int = 1000,
    scale: Optional[float] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    cfg = get_preset_config(preset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = str(base_name or "figure").strip().replace(" ", "_")
    export_formats = _normalize_formats(formats, ("html", *cfg["formats"]))
    export_scale = float(scale or cfg["scale"])

    artifacts: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    if "html" in export_formats:
        html_path = output_dir / f"{base}.html"
        fig.write_html(str(html_path))
        artifacts["html"] = str(html_path)
    for fmt in export_formats:
        if fmt == "html":
            continue
        out_path = output_dir / f"{base}.{fmt}"
        try:
            fig.write_image(
                str(out_path),
                format=fmt,
                width=int(width),
                height=int(height),
                scale=export_scale,
            )
            artifacts[fmt] = str(out_path)
        except Exception as exc:
            errors[fmt] = str(exc)
    return artifacts, errors


def load_result_table(path_or_glob: str, *, base_dir: Optional[Path] = None) -> Any:
    """Load result table/object from csv/tsv/json/npz via exact path or glob pattern."""
    pattern = str(path_or_glob or "").strip()
    if not pattern:
        raise ValueError("path_or_glob is empty")
    root = Path(base_dir or Path.cwd())
    candidates: List[Path] = []
    raw = Path(pattern)
    if raw.is_absolute():
        candidates = [raw]
    else:
        direct = root / pattern
        if direct.exists():
            candidates = [direct]
        else:
            candidates = [Path(x) for x in glob.glob(str(root / pattern))]
    existing = [p for p in candidates if p.exists() and p.is_file()]
    if not existing:
        raise FileNotFoundError(f"No files matched: {pattern}")
    target = sorted(existing)[0]
    suffix = target.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        import pandas as pd
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(target, sep=sep)
    if suffix == ".json":
        with target.open("r", encoding="utf-8") as f:
            return json.load(f)
    if suffix == ".npz":
        import numpy as np
        with np.load(target, allow_pickle=True) as npz:
            return {k: npz[k] for k in npz.files}
    raise ValueError(f"Unsupported file type for load_result_table: {suffix}")


def pick_top_figures(metrics_json: Any, k: int = 6) -> List[Dict[str, Any]]:
    """Pick top-K figure candidates from metrics payload/path with score priority."""
    payload: Any = metrics_json
    if isinstance(metrics_json, (str, Path)):
        p = Path(str(metrics_json))
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = {}
    figures: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        raw = payload.get("figures", payload.get("items", []))
        if isinstance(raw, list):
            figures = [x for x in raw if isinstance(x, dict)]
    elif isinstance(payload, list):
        figures = [x for x in payload if isinstance(x, dict)]
    if not figures:
        return []
    topk = max(1, int(k))
    ranked = sorted(
        figures,
        key=lambda item: (
            -float(item.get("priority", item.get("score", 0.0)) or 0.0),
            str(item.get("path", item.get("name", ""))),
        ),
    )
    return ranked[:topk]


def build_appendix_gallery(
    figures: List[Dict[str, Any]],
    *,
    title: str = "All Figures Appendix",
    description: str = "Auto-generated gallery for comprehensive figure coverage.",
) -> str:
    """Build a lightweight HTML gallery block for appendix insertion."""
    blocks: List[str] = [
        '<section id="all-figures-appendix">',
        f"<h2>{title}</h2>",
        f"<p>{description}</p>",
    ]
    for idx, fig in enumerate(figures, start=1):
        src = str(fig.get("path", "")).strip()
        if not src:
            continue
        caption = str(fig.get("caption", "")).strip() or f"Figure {idx}"
        analysis = str(fig.get("analysis", "downstream")).strip() or "downstream"
        summary = str(fig.get("summary", "")).strip()
        blocks.extend(
            [
                f'<figure data-analysis="{analysis}">',
                f'  <img src="{src}" alt="{caption}" />',
                f"  <figcaption><strong>Figure {idx}.</strong> {caption} ({analysis})</figcaption>",
            ]
        )
        if summary:
            blocks.append(f"  <p>{summary}</p>")
        blocks.append("</figure>")
    blocks.append("</section>")
    return "\n".join(blocks)


def _convert_raster_to_format(src_path: Path, dst_path: Path, dpi: int) -> None:
    import matplotlib.image as mpimg
    import numpy as np

    img = mpimg.imread(str(src_path))
    if isinstance(img, np.ndarray) and img.ndim == 2:
        cmap = "gray"
    else:
        cmap = None
    fig_h = max(2.0, float(img.shape[0]) / float(max(1, dpi)))
    fig_w = max(2.0, float(img.shape[1]) / float(max(1, dpi)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img, cmap=cmap)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(dst_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def ensure_publication_bundle_for_artifacts(
    artifacts: Dict[str, str],
    *,
    preset: Optional[str] = None,
    keep_original: bool = True,
) -> Dict[str, str]:
    """Ensure image artifacts include publication-friendly png/svg outputs.

    For raster inputs, missing formats are generated by deterministic conversion.
    """
    cfg = get_preset_config(preset)
    required_formats = _normalize_formats(cfg.get("formats"), ("png", "svg"))
    out: Dict[str, str] = dict(artifacts) if keep_original else {}
    if not artifacts:
        return out

    for key, raw_path in artifacts.items():
        src = Path(str(raw_path)).expanduser()
        if not src.exists() or not src.is_file():
            continue
        src_ext = src.suffix.lower().lstrip(".")
        if src_ext not in {"png", "jpg", "jpeg", "webp", "svg", "pdf"}:
            continue
        for fmt in required_formats:
            bundle_key = f"{key}_{fmt}"
            target = src.with_suffix(f".{fmt}")
            if target.exists():
                out[bundle_key] = str(target)
                continue
            if src_ext == fmt:
                out[bundle_key] = str(src)
                continue
            if src_ext in {"png", "jpg", "jpeg", "webp"} and fmt in {"png", "svg", "pdf"}:
                try:
                    _convert_raster_to_format(src, target, int(cfg["dpi"]))
                    out[bundle_key] = str(target)
                except Exception:
                    continue
    return out
