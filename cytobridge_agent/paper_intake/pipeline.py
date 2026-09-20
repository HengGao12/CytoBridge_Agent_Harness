from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .converters import PaperIntakeError, convert_asset_to_anndata, standardize_adata, write_conversion_report
from .llm_flow import (
    apply_llm_asset_preferences,
    enrich_inspection_with_llm,
    materialize_with_llm_script,
    resolve_paper_intake_download_root,
    resolve_paper_intake_llm,
)
from .manuscript import inspect_manuscript
from .models import DataAsset
from .providers import asset_looks_processed_candidate, asset_looks_raw_candidate, resolve_manuscript_assets, write_asset_report


FORMAT_PREFERENCE_BONUS = {
    "h5ad": 4.0,
    "loom": 3.0,
    "10x_h5": 2.0,
    "10x_mtx": 1.0,
}


def _asset_name_adjustment(asset: DataAsset) -> float:
    lowered = (asset.filename or "").lower()
    score = 0.0
    if any(token in lowered for token in ("normed_count", "normed_counts", "expression", "expr")):
        score += 8.0
    if "counts" in lowered and "clone" not in lowered:
        score += 4.0
    if "series_matrix" in lowered:
        score -= 18.0
    if any(token in lowered for token in ("filelist", "manifest", "checksum", "checksums", "md5", "sha256")):
        score -= 32.0
    if any(
        token in lowered
        for token in (
            "cell_barcodes",
            "barcodes.tsv",
            "gene_names",
            "feature_names",
            "features.tsv",
            "genes.tsv",
            "library_name",
            "library_names",
        )
    ):
        score -= 16.0
    if any(token in lowered for token in ("clone_matrix", "lineage", "barcode_matrix")):
        score -= 18.0
    return score


def _asset_matches_hint(asset: DataAsset, dataset_hint: str) -> bool:
    hint = (dataset_hint or "").strip().lower()
    if not hint:
        return True
    haystack = " ".join(
        [
            asset.url,
            asset.filename,
            asset.provider,
            asset.description,
            json.dumps(asset.metadata, ensure_ascii=False),
        ]
    ).lower()
    return hint in haystack


def _apply_dataset_hint_preference(assets: List[DataAsset], dataset_hint: str) -> List[DataAsset]:
    hint = (dataset_hint or "").strip()
    if not hint:
        return assets

    ranked: List[DataAsset] = []
    for asset in assets:
        score = float(asset.score)
        if _asset_matches_hint(asset, hint):
            score += 30.0
        clone = DataAsset(
            url=asset.url,
            provider=asset.provider,
            asset_kind=asset.asset_kind,
            file_format=asset.file_format,
            filename=asset.filename,
            score=score,
            source=asset.source,
            description=asset.description,
            metadata=dict(asset.metadata),
        )
        ranked.append(clone)
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def _rank_assets(assets: List[DataAsset], prefer_processed: bool = True, allow_raw: bool = True) -> List[DataAsset]:
    ranked: List[DataAsset] = []
    for asset in assets:
        score = float(asset.score)
        score += _asset_name_adjustment(asset)
        if prefer_processed and asset_looks_processed_candidate(asset):
            score += 4.0
            score += FORMAT_PREFERENCE_BONUS.get(asset.file_format, 0.0)
        if asset.asset_kind == "archive" and not asset_looks_processed_candidate(asset):
            score -= 6.0
        if not allow_raw and (asset.asset_kind == "raw" or asset_looks_raw_candidate(asset)):
            score -= 100.0
        clone = DataAsset(
            url=asset.url,
            provider=asset.provider,
            asset_kind=asset.asset_kind,
            file_format=asset.file_format,
            filename=asset.filename,
            score=score,
            source=asset.source,
            description=asset.description,
            metadata=dict(asset.metadata),
        )
        ranked.append(clone)
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def _select_ranked_asset(ranked: List[DataAsset], selected_asset_url: Optional[str] = None) -> DataAsset:
    if not ranked:
        raise PaperIntakeError("No downloadable data assets could be resolved from the manuscript.")

    requested = str(selected_asset_url or "").strip()
    if not requested:
        return ranked[0]

    exact = next((asset for asset in ranked if asset.url == requested), None)
    if exact is not None:
        return exact

    preview = "\n".join(
        f"- {asset.filename or '(unnamed)'} | {asset.file_format} | {asset.url}"
        for asset in ranked[:10]
    )
    raise PaperIntakeError(
        "Requested selected_asset_url was not found among candidate assets.\n"
        f"Requested: {requested}\n"
        f"Top candidates:\n{preview}"
    )


def _select_asset_with_llm_priority(
    ranked: List[DataAsset],
    *,
    selected_asset_url: Optional[str] = None,
    llm_selected_urls: Optional[List[str]] = None,
) -> DataAsset:
    requested = str(selected_asset_url or "").strip()
    if requested:
        return _select_ranked_asset(ranked, selected_asset_url=requested)

    by_url = {asset.url: asset for asset in ranked}
    for url in llm_selected_urls or []:
        value = str(url or "").strip()
        if value and value in by_url:
            return by_url[value]

    return _select_ranked_asset(ranked)


def inspect_paper_source(
    input_source: str,
    output_dir: str,
    fetch_online_if_needed: bool = True,
    article_url: Optional[str] = None,
    dataset_hint: Optional[str] = None,
    target_dataset_prompt: Optional[str] = None,
    llm: Any = None,
    use_llm: Optional[bool] = None,
    trace: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    inspection = inspect_manuscript(
        input_source=input_source,
        output_dir=output_dir,
        fetch_online_if_needed=fetch_online_if_needed,
        article_url=article_url,
    )
    assets = resolve_manuscript_assets(inspection, trace=trace)
    llm_requested = bool(use_llm) if use_llm is not None else resolve_paper_intake_llm(llm) is not None
    llm_payload: Dict[str, Any] = {}
    if llm_requested:
        llm_payload = enrich_inspection_with_llm(
            inspection,
            output_dir=output_dir,
            llm=llm,
            seed_assets=assets,
            dataset_hint=dataset_hint,
            target_dataset_prompt=target_dataset_prompt,
            trace=trace,
        )
        candidate_payload = llm_payload.get("candidate_assets") or []
        if candidate_payload:
            assets = [DataAsset(**item) for item in candidate_payload]
    asset_report = write_asset_report(assets, output_dir=output_dir)
    payload = inspection.to_dict()
    payload["candidate_assets"] = [asset.to_dict() for asset in assets]
    payload["asset_report_path"] = asset_report
    payload.update(
        {
            "llm_enabled": bool(llm_payload.get("llm_enabled")),
            "selected_asset_urls": list(llm_payload.get("selected_asset_urls") or []),
            "supporting_asset_urls": list(llm_payload.get("supporting_asset_urls") or []),
            "fallback_asset_urls": list(llm_payload.get("fallback_asset_urls") or []),
            "resolved_dataset_hint": str(llm_payload.get("dataset_hint") or ""),
            "target_dataset_prompt": str(target_dataset_prompt or llm_payload.get("target_dataset_prompt") or ""),
            "llm_pdf_analysis_path": str(llm_payload.get("pdf_analysis_path") or ""),
            "llm_web_analysis_path": str(llm_payload.get("web_analysis_path") or ""),
            "llm_discovery_trace_path": str(llm_payload.get("discovery_trace_path") or ""),
            "fetched_pages": list(llm_payload.get("fetched_pages") or []),
            "discovery_observations": list(llm_payload.get("discovery_observations") or []),
        }
    )
    manifest_path = Path(output_dir).expanduser().resolve() / "paper_intake" / "inspection_bundle.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["inspection_bundle_path"] = str(manifest_path)
    return payload


def prepare_anndata_from_paper(
    input_path: str,
    output_dir: str,
    dataset_hint: Optional[str] = None,
    target_dataset_prompt: Optional[str] = None,
    selected_asset_url: Optional[str] = None,
    fetch_online_if_needed: bool = True,
    article_url: Optional[str] = None,
    prefer_processed: bool = True,
    allow_raw: bool = True,
    raw_reference_path: Optional[str] = None,
    llm: Any = None,
    use_llm: Optional[bool] = None,
    require_downstream_ready: Optional[bool] = None,
    max_materialization_attempts: int = 3,
    trace: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    payload = inspect_paper_source(
        input_source=input_path,
        output_dir=output_dir,
        fetch_online_if_needed=fetch_online_if_needed,
        article_url=article_url,
        dataset_hint=dataset_hint,
        target_dataset_prompt=target_dataset_prompt,
        llm=llm,
        use_llm=use_llm,
        trace=trace,
    )
    assets = [DataAsset(**item) for item in payload.get("candidate_assets", [])]
    ranked = _rank_assets(assets, prefer_processed=prefer_processed, allow_raw=allow_raw)
    effective_hint = str(dataset_hint or payload.get("resolved_dataset_hint") or "").strip()
    if payload.get("llm_enabled"):
        ranked = apply_llm_asset_preferences(
            ranked,
            selected_asset_urls=payload.get("selected_asset_urls") or [],
            fallback_asset_urls=payload.get("fallback_asset_urls") or [],
            dataset_hint=effective_hint,
        )
    if effective_hint:
        ranked = _apply_dataset_hint_preference(ranked, effective_hint)
    selected = _select_asset_with_llm_priority(
        ranked,
        selected_asset_url=selected_asset_url,
        llm_selected_urls=list(payload.get("selected_asset_urls") or []),
    )
    supporting_url_set = {str(url or "").strip() for url in list(payload.get("supporting_asset_urls") or [])}
    supporting_assets = [asset for asset in ranked if asset.url in supporting_url_set and asset.url != selected.url]
    selected_asset_locked = bool(str(selected_asset_url or "").strip())
    llm_enabled = bool(payload.get("llm_enabled"))
    strict_validation = require_downstream_ready if require_downstream_ready is not None else llm_enabled
    manuscript_summary = {
        key: value
        for key, value in payload.items()
        if key not in {"candidate_assets"}
    }

    materialization_attempts_path = ""
    if llm_enabled:
        materialized = materialize_with_llm_script(
            selected_asset=selected,
            ranked_assets=ranked,
            manuscript_summary=manuscript_summary,
            output_dir=output_dir,
            llm=llm,
            raw_reference_path=raw_reference_path,
            require_downstream_ready=bool(strict_validation),
            max_attempts=max_materialization_attempts,
            allow_asset_switch=not selected_asset_locked,
            supporting_assets=supporting_assets,
            trace=trace,
        )
        adata = materialized["adata"]
        selected = DataAsset(**dict(materialized["selected_asset"]))
        details = dict(materialized.get("conversion_details") or {})
        materialization_attempts_path = str(materialized.get("materialization_attempts_path") or "")
    else:
        adata, details = convert_asset_to_anndata(
            selected,
            download_root=str(resolve_paper_intake_download_root(output_dir, trace=trace)),
            raw_reference_path=raw_reference_path,
        )
        adata = standardize_adata(
            adata,
            selected,
            manuscript_summary=manuscript_summary,
        )

    save_path = Path(output_dir).expanduser().resolve() / "paper_intake" / "paper_input.h5ad"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(save_path)

    report_payload: Dict[str, object] = {
        "input_path": input_path,
        "dataset_hint": effective_hint,
        "target_dataset_prompt": str(target_dataset_prompt or payload.get("target_dataset_prompt") or ""),
        "requested_asset_url": str(selected_asset_url or ""),
        "selected_asset_locked": selected_asset_locked,
        "selected_asset": selected.to_dict(),
        "llm_selected_asset_urls": list(payload.get("selected_asset_urls") or []),
        "llm_supporting_asset_urls": list(payload.get("supporting_asset_urls") or []),
        "llm_fallback_asset_urls": list(payload.get("fallback_asset_urls") or []),
        "supporting_assets": [asset.to_dict() for asset in supporting_assets],
        "all_ranked_assets": [asset.to_dict() for asset in ranked[:25]],
        "conversion_details": details,
        "paper_intake_validation": dict(adata.uns.get("paper_intake", {}).get("validation", {})),
        "saved_adata_path": str(save_path),
        "inspection_bundle_path": payload.get("inspection_bundle_path") or "",
        "asset_report_path": payload.get("asset_report_path") or "",
        "llm_enabled": llm_enabled,
        "llm_pdf_analysis_path": payload.get("llm_pdf_analysis_path") or "",
        "llm_web_analysis_path": payload.get("llm_web_analysis_path") or "",
        "llm_discovery_trace_path": payload.get("llm_discovery_trace_path") or "",
        "materialization_attempts_path": materialization_attempts_path,
    }
    report_path = write_conversion_report(report_payload, output_dir=output_dir)
    return {
        "adata": adata,
        "adata_path": str(save_path),
        "selected_asset": selected.to_dict(),
        "report_path": report_path,
        "llm_enabled": llm_enabled,
        "inspection_bundle_path": payload.get("inspection_bundle_path") or "",
        "asset_report_path": payload.get("asset_report_path") or "",
        "llm_discovery_trace_path": payload.get("llm_discovery_trace_path") or "",
        "target_dataset_prompt": str(target_dataset_prompt or payload.get("target_dataset_prompt") or ""),
        "conversion_details": details,
        "materialization_attempts_path": materialization_attempts_path,
    }
