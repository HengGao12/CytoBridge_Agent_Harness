from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from ..utils.config_manager import CONFIG_FILE, get_saved_config
from ..utils.llm_factory import instantiate_llm, normalize_auth_mode
from ..utils.llm_runtime import invoke_with_retry
from .converters import (
    PaperIntakeError,
    convert_local_path_to_anndata,
    describe_local_inputs,
    download_asset,
    read_sidecar_table,
    standardize_adata,
    try_merge_obs_table,
)
from .manuscript import DATA_KEYWORDS, extract_accessions, fetch_html_document
from .models import DataAsset, ManuscriptInspection
from .providers import (
    DATA_HOST_HINTS,
    classify_reference,
    crawl_download_page,
    resolve_arrayexpress_accession,
    resolve_ena_accession,
    resolve_geo_accession,
    resolve_zenodo_identifier,
)

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:  # pragma: no cover - optional runtime dependency
    HumanMessage = None
    SystemMessage = None


logger = logging.getLogger(__name__)
TraceCallback = Optional[Callable[[str], None]]


def _paper_intake_slug(output_dir: Path) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(output_dir).strip("/"))
    digest = hashlib.sha1(str(output_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{normalized[-96:]}__{digest}"


def resolve_paper_intake_download_root(
    output_dir: str,
    *,
    paper_dir: Optional[Path] = None,
    trace: TraceCallback = None,
) -> Path:
    """Return the download root for paper intake.

    When CYTOBRIDGE_RAW_DATA_POOL is configured, new downloads go into the
    shared raw-data pool and the run-local paper_intake/downloads path is kept
    as a symlink. Existing non-symlink run-local download directories are left
    in place to avoid breaking resumed campaigns.
    """

    output_path = Path(output_dir).expanduser().resolve()
    paper_root = paper_dir or output_path / "paper_intake"
    paper_root.mkdir(parents=True, exist_ok=True)
    local_root = paper_root / "downloads"

    pool_text = os.environ.get("CYTOBRIDGE_RAW_DATA_POOL") or os.environ.get("CYTOBRIDGE_RAW_DATA_ROOT")
    if not pool_text:
        local_root.mkdir(parents=True, exist_ok=True)
        return local_root

    pool_root = Path(pool_text).expanduser().resolve()
    pool_download_root = pool_root / "incoming" / "paper_intake" / _paper_intake_slug(output_path) / "downloads"
    pool_download_root.mkdir(parents=True, exist_ok=True)

    pointer_path = paper_root / "downloads_path.txt"
    pointer_path.write_text(str(pool_download_root) + "\n", encoding="utf-8")

    if local_root.exists() or local_root.is_symlink():
        if local_root.is_symlink():
            try:
                if local_root.resolve() != pool_download_root:
                    local_root.unlink()
                    local_root.symlink_to(pool_download_root, target_is_directory=True)
            except OSError as exc:
                _trace(trace, f"[paper_intake] could not refresh downloads symlink: {exc}")
        return local_root if not local_root.is_symlink() else pool_download_root

    try:
        local_root.symlink_to(pool_download_root, target_is_directory=True)
    except OSError as exc:
        _trace(trace, f"[paper_intake] could not create downloads symlink, using pool path directly: {exc}")
    return pool_download_root


def _clean_optional_text(value: Any) -> str:
    return str(value or "").strip()


def _saved_config_sections(saved_config: Dict[str, Any]) -> List[Mapping[str, Any]]:
    sections: List[Mapping[str, Any]] = []
    if isinstance(saved_config, Mapping):
        sections.append(saved_config)
        for key in ("llm", "llm_config", "openai", "model_config"):
            nested = saved_config.get(key)
            if isinstance(nested, Mapping):
                sections.append(nested)
    return sections


def _lookup_saved_text(saved_config: Dict[str, Any], *keys: str) -> str:
    for section in _saved_config_sections(saved_config):
        for key in keys:
            for candidate_key in {key, key.replace("_", "-"), key.replace("-", "_")}:
                if candidate_key not in section:
                    continue
                text = _clean_optional_text(section.get(candidate_key))
                if text:
                    return text
    return ""


def _inspect_saved_config_debug() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "saved_config_path": str(CONFIG_FILE),
        "saved_config_present": CONFIG_FILE.exists(),
        "saved_config_parse_ok": False,
        "saved_config_error": "",
        "saved_config_top_level_keys": [],
        "saved_config_section_keys": {},
    }
    if not CONFIG_FILE.exists():
        return info

    try:
        raw = CONFIG_FILE.read_text(encoding="utf-8-sig")
        parsed = json.loads(raw)
    except Exception as exc:
        info["saved_config_error"] = str(exc)
        return info

    if not isinstance(parsed, Mapping):
        info["saved_config_error"] = "Top-level JSON value is not an object."
        return info

    info["saved_config_parse_ok"] = True
    info["saved_config_top_level_keys"] = sorted(str(key) for key in parsed.keys())
    for key in ("llm", "llm_config", "openai", "model_config"):
        nested = parsed.get(key)
        if isinstance(nested, Mapping):
            info["saved_config_section_keys"][key] = sorted(str(item) for item in nested.keys())
    return info


def _resolve_paper_intake_llm_config() -> Dict[str, Any]:
    saved_config = get_saved_config()
    config_debug = _inspect_saved_config_debug()

    env_model = _clean_optional_text(
        os.getenv("CYTOBRIDGE_PAPER_INTAKE_LLM_MODEL")
        or os.getenv("CYTOBRIDGE_LLM_MODEL")
    )
    saved_model = _lookup_saved_text(saved_config, "llm_model", "model", "model_name")

    env_base_url = _clean_optional_text(
        os.getenv("CYTOBRIDGE_PAPER_INTAKE_LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
    )
    saved_base_url = _lookup_saved_text(
        saved_config,
        "llm_base_url",
        "base_url",
        "api_base",
        "api_base_url",
        "openai_base_url",
        "openai_api_base",
    )

    env_api_key = _clean_optional_text(
        os.getenv("CYTOBRIDGE_PAPER_INTAKE_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    saved_api_key = _lookup_saved_text(saved_config, "openai_api_key", "llm_api_key", "api_key")

    env_auth_mode = _clean_optional_text(os.getenv("CYTOBRIDGE_PAPER_INTAKE_LLM_AUTH_MODE"))
    saved_auth_mode = _lookup_saved_text(saved_config, "llm_auth_mode", "auth_mode")

    env_profile_id = _clean_optional_text(os.getenv("CYTOBRIDGE_PAPER_INTAKE_LLM_PROFILE_ID"))
    saved_profile_id = _lookup_saved_text(saved_config, "llm_profile_id", "profile_id", "preferred_profile_id")

    env_thinking_level = _clean_optional_text(os.getenv("CYTOBRIDGE_PAPER_INTAKE_LLM_THINKING_LEVEL"))
    saved_thinking_level = _lookup_saved_text(saved_config, "llm_thinking_level", "thinking_level", "reasoning")

    has_saved_llm_config = any(
        (
            saved_model,
            saved_base_url,
            saved_api_key,
            saved_auth_mode,
            saved_profile_id,
            saved_thinking_level,
        )
    )
    has_env_llm_config = any(
        (
            env_model,
            env_base_url,
            env_api_key,
            env_auth_mode,
            env_profile_id,
            env_thinking_level,
        )
    )
    auth_mode = normalize_auth_mode(env_auth_mode or saved_auth_mode or "auto")
    resolved_model = env_model or saved_model
    if not resolved_model:
        if auth_mode == "codex_oauth":
            resolved_model = "gpt-5.4"
        else:
            resolved_model = "gpt-4o"

    return {
        "enabled": bool(has_saved_llm_config or has_env_llm_config),
        "model": resolved_model,
        "base_url": env_base_url or saved_base_url or "",
        "api_key": env_api_key or saved_api_key or "",
        "auth_mode": auth_mode,
        "profile_id": env_profile_id or saved_profile_id or None,
        "thinking_level": env_thinking_level or saved_thinking_level or None,
        "used_saved_config": bool(has_saved_llm_config and not has_env_llm_config),
        "used_env_overrides": bool(has_env_llm_config),
        **config_debug,
    }


def _trace(callback: TraceCallback, message: str) -> None:
    if callback is None:
        return
    try:
        callback(message)
    except Exception:
        return


def _dedupe(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _asset_matches_hint(asset: DataAsset, dataset_hint: str) -> bool:
    hint = (dataset_hint or "").strip().lower()
    if not hint:
        return False
    haystack = " ".join(
        [
            asset.url,
            asset.filename,
            asset.provider,
            asset.description,
            json.dumps(asset.metadata or {}, ensure_ascii=False),
        ]
    ).lower()
    return hint in haystack


def _target_scope_text(dataset_hint: Optional[str] = None, target_dataset_prompt: Optional[str] = None) -> str:
    target = str(target_dataset_prompt or "").strip()
    hint = str(dataset_hint or "").strip()
    if target and hint and hint != target:
        return f"{target}\nAdditional structured hint: {hint}"
    return target or hint


def _target_scope_prompt_block(dataset_hint: Optional[str] = None, target_dataset_prompt: Optional[str] = None) -> str:
    target = str(target_dataset_prompt or "").strip()
    hint = str(dataset_hint or "").strip()
    if not target:
        hint_text = f" Optional dataset hint: {hint}" if hint else ""
        return (
            "Target dataset prompt: (none). Use the default behavior: select the best original analyzable "
            "single-cell dataset from the paper, preferring processed expression data with metadata."
            f"{hint_text}"
        )
    scope = _target_scope_text(dataset_hint, target_dataset_prompt)
    return (
        "Binding target dataset prompt from the user/main agent:\n"
        f"{scope}\n"
        "This is a hard scope constraint. Select and download only assets needed for that requested subset. "
        "Exclude other manuscript datasets, conditions, tissues, time courses, or raw/full-study archives unless "
        "they are the only way to obtain the requested subset; if so, state the residual risk clearly. "
        "Supporting assets must be sidecars for the target subset, not unrelated samples."
    )


def _render_llm_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def resolve_paper_intake_llm(llm: Any = None) -> Any:
    if llm is not None:
        return llm

    config = _resolve_paper_intake_llm_config()
    if not bool(config.get("enabled")):
        return None

    try:
        resolved, _ = instantiate_llm(
            model=config.get("model"),
            base_url=(str(config.get("base_url") or "").strip() or None),
            api_key=(str(config.get("api_key") or "").strip() or None),
            auth_mode=config.get("auth_mode"),
            preferred_profile_id=config.get("profile_id"),
            thinking_level=config.get("thinking_level"),
        )
        return resolved
    except Exception as exc:  # pragma: no cover - runtime-only fallback
        logger.warning("paper intake LLM bootstrap failed: %s", exc)
        return None


def _call_llm_text(llm: Any, system_prompt: str, user_prompt: str, *, agent_name: str) -> Optional[str]:
    if llm is None:
        return None
    if hasattr(llm, "invoke") and HumanMessage is not None and SystemMessage is not None:
        response = invoke_with_retry(
            llm,
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            logger,
            agent_name,
        )
        return _render_llm_content(response)
    if hasattr(llm, "chat"):
        return llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2500,
        )
    raise ValueError("Unsupported LLM object passed to paper intake flow.")


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    body = str(text or "").strip()
    if not body:
        return None

    candidates = [body]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", body, flags=re.DOTALL)
    fenced += re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", body, flags=re.DOTALL)
    candidates.extend(fenced)

    first_obj = body.find("{")
    last_obj = body.rfind("}")
    if first_obj >= 0 and last_obj > first_obj:
        candidates.append(body[first_obj : last_obj + 1])
    first_arr = body.find("[")
    last_arr = body.rfind("]")
    if first_arr >= 0 and last_arr > first_arr:
        candidates.append(body[first_arr : last_arr + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, list):
            return {"items": parsed}
        if isinstance(parsed, dict):
            return parsed
    return None


def _write_json(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _read_text(path: str) -> str:
    if not str(path or "").strip():
        return ""
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        return ""
    if candidate.is_dir():
        return ""
    try:
        return candidate.read_text(encoding="utf-8")
    except OSError:
        return ""


def _truncate(text: str, max_chars: int) -> str:
    body = str(text or "")
    if len(body) <= max_chars:
        return body
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return body[:head] + "\n\n[... truncated ...]\n\n" + body[-tail:]


def _collect_pdf_focus_snippets(inspection: ManuscriptInspection, limit: int = 8) -> List[Dict[str, Any]]:
    if inspection.input_type != "file":
        return []
    source_path = Path(inspection.input_source).expanduser().resolve()
    if source_path.suffix.lower() != ".pdf" or not source_path.exists():
        return []

    try:
        from ..rag.rag_tools import extract_pdf_pages
    except Exception:
        return []

    try:
        pages = extract_pdf_pages(source_path)
    except Exception:
        return []

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for page_number, text in pages:
        lowered = str(text or "").lower()
        keyword_score = sum(lowered.count(keyword) for keyword in DATA_KEYWORDS)
        url_score = lowered.count("http")
        accession_score = sum(len(values) for values in extract_accessions(text).values() if isinstance(values, list))
        total = keyword_score * 2 + url_score * 4 + accession_score * 3
        if total <= 0:
            continue
        scored.append(
            (
                float(total),
                {
                    "page": int(page_number),
                    "score": float(total),
                    "text": _truncate(re.sub(r"\s+", " ", text), 2400),
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[:limit]]


def _seed_page_urls(inspection: ManuscriptInspection) -> List[str]:
    urls: List[str] = []
    for url in inspection.candidate_article_urls + inspection.discovered_urls:
        lowered = str(url).lower()
        if not lowered.startswith(("http://", "https://")):
            continue
        if lowered.endswith(".pdf"):
            continue
        urls.append(url)
    return _dedupe(urls)[:6]


def _summarize_page(page: Dict[str, Any], max_text_chars: int = 20000, max_links: int = 25) -> Dict[str, Any]:
    links = _dedupe(page.get("links") or [])
    return {
        "url": str(page.get("url") or ""),
        "title": str(page.get("title") or ""),
        "content_type": str(page.get("content_type") or ""),
        "text_excerpt": _truncate(str(page.get("text") or ""), max_text_chars),
        "links": links[:max_links],
    }


def _resolve_accessions_from_strings(strings: Sequence[str], trace: TraceCallback = None) -> List[DataAsset]:
    assets: List[DataAsset] = []
    for item in strings:
        accessions = extract_accessions(str(item))
        for accession in accessions.get("geo_series", []):
            assets.extend(resolve_geo_accession(accession, trace=trace))
        for accession in accessions.get("geo_sample", []):
            assets.extend(resolve_geo_accession(accession, trace=trace))
        for accession in accessions.get("sra_study", []):
            try:
                assets.extend(resolve_ena_accession(accession, trace=trace))
            except Exception:
                continue
        for accession in accessions.get("bioproject", []):
            try:
                assets.extend(resolve_ena_accession(accession, trace=trace))
            except Exception:
                continue
        for accession in accessions.get("arrayexpress", []):
            try:
                assets.extend(resolve_arrayexpress_accession(accession, trace=trace))
            except Exception:
                continue
        for doi in accessions.get("zenodo_doi", []):
            try:
                assets.extend(resolve_zenodo_identifier(doi, trace=trace))
            except Exception:
                continue
    return assets


def _resolve_url_hints(urls: Sequence[str], trace: TraceCallback = None) -> Tuple[List[DataAsset], List[Dict[str, Any]]]:
    assets: List[DataAsset] = []
    pages: List[Dict[str, Any]] = []

    for url in _dedupe(list(urls)):
        asset = classify_reference(url, source="llm")
        if asset.asset_kind != "page":
            assets.append(asset)
            continue

        try:
            page = fetch_html_document(url)
        except Exception as exc:
            _trace(trace, f"[paper_intake.llm] failed to fetch hinted page {url}: {exc}")
            continue

        pages.append(_summarize_page(page))
        page_text = str(page.get("text") or "")
        assets.extend(_resolve_accessions_from_strings([page_text], trace=trace))

        lowered = url.lower()
        if any(host in lowered for host in DATA_HOST_HINTS) or any(
            token in lowered for token in ("download", "dataset", "supp", "files", "matrix", "record", "atlas", "arrayexpress")
        ):
            try:
                assets.extend(crawl_download_page(url, trace=trace))
            except Exception:
                continue

    return assets, pages


def _dedupe_assets(assets: Sequence[DataAsset]) -> List[DataAsset]:
    best: Dict[Tuple[str, str, str], DataAsset] = {}
    for asset in assets:
        key = (asset.url, asset.asset_kind, asset.file_format)
        previous = best.get(key)
        if previous is None or float(asset.score) > float(previous.score):
            best[key] = asset
    return sorted(best.values(), key=lambda item: item.score, reverse=True)


def _select_string_list(payload: Dict[str, Any], key: str) -> List[str]:
    out: List[str] = []
    for item in payload.get(key, []) or []:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            value = item.get("url") or item.get("value") or item.get("accession")
            if value:
                out.append(str(value))
    return _dedupe(out)


def _select_accession_strings(payload: Dict[str, Any], key: str) -> List[str]:
    raw = payload.get(key) or []
    values: List[Any] = []
    if isinstance(raw, Mapping):
        for item in raw.values():
            if isinstance(item, (list, tuple, set)):
                values.extend(item)
            else:
                values.append(item)
    elif isinstance(raw, (list, tuple, set)):
        values.extend(raw)
    else:
        values.append(raw)
    return _dedupe(str(item) for item in values if str(item or "").strip())


def _paper_intake_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _error_payload(exc: Exception) -> Dict[str, str]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }


def _asset_observation(asset: DataAsset) -> Dict[str, Any]:
    return {
        "url": asset.url,
        "provider": asset.provider,
        "asset_kind": asset.asset_kind,
        "file_format": asset.file_format,
        "filename": asset.filename,
        "score": asset.score,
        "source": asset.source,
        "description": asset.description,
        "metadata": dict(asset.metadata or {}),
    }


def _assets_observation(assets: Sequence[DataAsset], limit: int = 25) -> List[Dict[str, Any]]:
    return [_asset_observation(asset) for asset in list(assets)[:limit]]


def _page_link_values(page: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for item in page.get("links") or []:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, (list, tuple)) and item:
            values.append(str(item[0]))
        elif isinstance(item, Mapping):
            value = item.get("url") or item.get("href")
            if value:
                values.append(str(value))
    for item in page.get("raw_links") or []:
        if item:
            values.append(str(item))
    return _dedupe(values)


def _resolve_accession_text_with_observations(
    text: str,
    *,
    source: str = "llm_discovery",
    trace: TraceCallback = None,
) -> Tuple[List[DataAsset], List[Dict[str, Any]]]:
    assets: List[DataAsset] = []
    observations: List[Dict[str, Any]] = []
    accessions = extract_accessions(str(text or ""))

    def record(kind: str, accession: str, resolver: Callable[..., List[DataAsset]]) -> None:
        obs: Dict[str, Any] = {
            "action": "resolve_accession",
            "kind": kind,
            "accession": accession,
            "source": source,
            "ok": False,
            "assets": [],
        }
        try:
            resolved = resolver(accession, trace=trace)
            assets.extend(resolved)
            obs.update(
                {
                    "ok": True,
                    "asset_count": len(resolved),
                    "assets": _assets_observation(resolved, limit=12),
                }
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to LLM as recoverable evidence
            obs["error"] = _error_payload(exc)
            _trace(trace, f"[paper_intake.llm] accession resolve failed accession={accession}: {exc}")
        observations.append(obs)

    for accession in accessions.get("geo_series", [])[:8]:
        record("geo_series", accession, resolve_geo_accession)
    for accession in accessions.get("geo_sample", [])[:8]:
        record("geo_sample", accession, resolve_geo_accession)
    for accession in accessions.get("sra_study", [])[:5]:
        record("sra_study", accession, resolve_ena_accession)
    for accession in accessions.get("bioproject", [])[:5]:
        record("bioproject", accession, resolve_ena_accession)
    for accession in accessions.get("arrayexpress", [])[:5]:
        record("arrayexpress", accession, resolve_arrayexpress_accession)
    for doi in accessions.get("zenodo_doi", [])[:5]:
        record("zenodo_doi", doi, resolve_zenodo_identifier)

    return assets, observations


def _observe_url_candidate(
    url: str,
    *,
    source: str = "llm_discovery",
    force_crawl: bool = False,
    trace: TraceCallback = None,
) -> Tuple[List[DataAsset], List[Dict[str, Any]], List[Dict[str, Any]]]:
    url = str(url or "").strip()
    if not url:
        return [], [], []

    assets: List[DataAsset] = []
    pages: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    asset = classify_reference(url, source=source)
    obs: Dict[str, Any] = {
        "action": "fetch_url" if not force_crawl else "crawl_page",
        "url": url,
        "source": source,
        "ok": False,
        "assets": [],
        "pages": [],
    }

    if asset.asset_kind != "page":
        logger.info("[paper_intake_discovery] treating URL as direct asset url=%s kind=%s", url, asset.asset_kind)
        assets.append(asset)
        obs.update(
            {
                "ok": True,
                "event": "direct_asset",
                "assets": [_asset_observation(asset)],
            }
        )
        observations.append(obs)
        return assets, pages, observations

    try:
        logger.info("[paper_intake_discovery] fetching page url=%s", url)
        page = fetch_html_document(url)
    except Exception as exc:  # noqa: BLE001 - recoverable; feed it back to the LLM
        obs["error"] = _error_payload(exc)
        observations.append(obs)
        _trace(trace, f"[paper_intake.llm] failed to fetch page url={url}: {exc}")
        logger.info("[paper_intake_discovery] page fetch failed url=%s error=%s", url, exc)
        return assets, pages, observations

    page_summary = _summarize_page(page, max_text_chars=16000, max_links=60)
    pages.append(page_summary)
    obs.update(
        {
            "ok": True,
            "event": "page_fetched",
            "pages": [page_summary],
        }
    )

    direct_assets: List[DataAsset] = []
    for link in _page_link_values(page):
        linked_asset = classify_reference(link, source=url)
        if linked_asset.asset_kind != "page":
            direct_assets.append(linked_asset)
    if direct_assets:
        assets.extend(direct_assets)
        obs["direct_asset_count"] = len(direct_assets)

    page_text = str(page.get("text") or "")
    accession_assets, accession_obs = _resolve_accession_text_with_observations(
        page_text,
        source=url,
        trace=trace,
    )
    assets.extend(accession_assets)
    observations.extend(accession_obs)

    lowered = url.lower()
    should_crawl = force_crawl or any(host in lowered for host in DATA_HOST_HINTS) or any(
        token in lowered for token in ("download", "dataset", "supp", "files", "matrix", "record", "atlas", "arrayexpress")
    )
    if should_crawl:
        try:
            logger.info("[paper_intake_discovery] crawling page url=%s", url)
            crawled = crawl_download_page(url, trace=trace)
            assets.extend(crawled)
            obs["crawl_asset_count"] = len(crawled)
        except Exception as exc:  # noqa: BLE001 - recoverable; feed it back to the LLM
            obs.setdefault("warnings", []).append({"crawl_error": _error_payload(exc)})
            _trace(trace, f"[paper_intake.llm] crawl failed url={url}: {exc}")

    obs["assets"] = _assets_observation(_dedupe_assets(assets), limit=20)
    observations.insert(0, obs)
    return assets, pages, observations


def _coerce_discovery_actions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_actions = payload.get("actions") or payload.get("next_actions") or []
    if isinstance(raw_actions, Mapping):
        raw_actions = [raw_actions]
    if isinstance(raw_actions, (str, bytes)):
        raw_actions = [raw_actions]
    if not isinstance(raw_actions, Sequence):
        return []

    actions: List[Dict[str, Any]] = []
    for item in raw_actions:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            action_type = "fetch_url" if text.startswith(("http://", "https://")) else "resolve_accession"
            actions.append({"action": action_type, "url": text, "accession": text})
            continue
        if not isinstance(item, Mapping):
            continue
        action_type = str(item.get("action") or item.get("type") or item.get("tool") or "").strip().lower()
        if not action_type:
            continue
        normalized = {
            "action": action_type,
            "url": str(item.get("url") or item.get("href") or "").strip(),
            "accession": str(item.get("accession") or item.get("identifier") or item.get("value") or "").strip(),
            "reason": str(item.get("reason") or item.get("rationale") or "").strip(),
            "description": str(item.get("description") or "").strip(),
        }
        if normalized["action"] in {"fetch", "open_url", "read_url", "read_page"}:
            normalized["action"] = "fetch_url"
        elif normalized["action"] in {"crawl", "crawl_url"}:
            normalized["action"] = "crawl_page"
        elif normalized["action"] in {"resolve", "resolve_accessions", "resolve_identifier"}:
            normalized["action"] = "resolve_accession"
        elif normalized["action"] in {"asset", "add_url", "add_asset_url"}:
            normalized["action"] = "add_asset"
        actions.append(normalized)
    return actions


def _execute_discovery_action(
    action: Dict[str, Any],
    *,
    trace: TraceCallback = None,
) -> Tuple[List[DataAsset], List[Dict[str, Any]], List[Dict[str, Any]]]:
    action_type = str(action.get("action") or "").strip().lower()
    url = str(action.get("url") or "").strip()
    accession = str(action.get("accession") or "").strip()

    if action_type in {"fetch_url", "crawl_page"} and url:
        return _observe_url_candidate(
            url,
            source="llm_action",
            force_crawl=action_type == "crawl_page",
            trace=trace,
        )

    if action_type == "resolve_accession" and accession:
        assets, observations = _resolve_accession_text_with_observations(
            accession,
            source="llm_action",
            trace=trace,
        )
        return assets, [], observations

    if action_type == "add_asset" and url:
        asset = classify_reference(url, source="llm_action", description=str(action.get("description") or ""))
        observation = {
            "action": "add_asset",
            "url": url,
            "ok": True,
            "assets": [_asset_observation(asset)],
            "reason": action.get("reason") or "",
        }
        return [asset], [], [observation]

    observation = {
        "action": action_type or "unknown",
        "ok": False,
        "error": {"type": "InvalidAction", "message": f"Unsupported or incomplete discovery action: {action}"},
    }
    return [], [], [observation]


def _discovery_round_prompt(
    *,
    inspection: ManuscriptInspection,
    parsed_pdf: Dict[str, Any],
    dataset_hint: Optional[str],
    target_dataset_prompt: Optional[str],
    assets: Sequence[DataAsset],
    fetched_pages: Sequence[Dict[str, Any]],
    observations: Sequence[Dict[str, Any]],
    round_index: int,
    max_actions: int,
) -> Tuple[str, str]:
    system_prompt = (
        "You are steering a paper-to-AnnData discovery loop. The program provides weak heuristic evidence, "
        "but you decide what to inspect next. Treat failed URL fetches, 404s, empty pages, and provider errors "
        "as observations to reason from, not as final failures. If a target dataset prompt is provided, it is "
        "binding and narrower than the paper as a whole. Return JSON only."
    )
    manuscript_brief = {
        "input_source": inspection.input_source,
        "title": inspection.title,
        "doi": inspection.doi,
        "accessions": inspection.accessions,
        "candidate_article_urls": inspection.candidate_article_urls[:20],
        "relevant_sentences": inspection.relevant_sentences[:20],
        "notes": inspection.notes,
    }
    asset_preview = _assets_observation(_dedupe_assets(list(assets)), limit=30)
    recent_observations = list(observations)[-18:]
    page_preview = list(fetched_pages)[-6:]
    user_prompt = (
        "Choose the next small set of evidence-gathering actions and optionally name the best assets so far.\n"
        "Return a JSON object with keys:\n"
        "- actions: list of at most "
        f"{max_actions} items. Each item must be one of: "
        "{\"action\":\"fetch_url\",\"url\":\"...\"}, "
        "{\"action\":\"crawl_page\",\"url\":\"...\"}, "
        "{\"action\":\"resolve_accession\",\"accession\":\"...\"}, "
        "{\"action\":\"add_asset\",\"url\":\"...\",\"description\":\"...\"}.\n"
        "- selected_asset_urls: best analyzable asset URLs from the candidate inventory, in priority order.\n"
        "- fallback_asset_urls: backup analyzable asset URLs.\n"
        "- dataset_hint: refined hint if the manuscript has multiple datasets.\n"
        "- stop_discovery: true when no more pages need to be read.\n"
        "- reasoning: concise explanation.\n\n"
        "Do not request the same failed URL again unless you have a concrete reason. Prefer processed expression "
        "matrices with usable metadata/time or stage information. Heuristic scores are weak priors only. "
        "When a target prompt is present, do not select unrelated datasets merely because they are easier or higher-scored.\n\n"
        f"Round: {round_index}\n"
        f"User dataset hint: {dataset_hint or ''}\n"
        f"{_target_scope_prompt_block(dataset_hint, target_dataset_prompt)}\n"
        f"Manuscript brief: {json.dumps(manuscript_brief, ensure_ascii=False)}\n"
        f"PDF analysis: {json.dumps(parsed_pdf, ensure_ascii=False)}\n"
        f"Fetched pages: {json.dumps(page_preview, ensure_ascii=False)}\n"
        f"Candidate assets: {json.dumps(asset_preview, ensure_ascii=False)}\n"
        f"Recent observations including errors: {json.dumps(recent_observations, ensure_ascii=False)}"
    )
    return system_prompt, user_prompt


def _run_llm_discovery_loop(
    *,
    inspection: ManuscriptInspection,
    output_dir: str,
    llm: Any,
    parsed_pdf: Dict[str, Any],
    seed_assets: Sequence[DataAsset],
    hinted_urls: Sequence[str],
    hinted_accessions: Sequence[str],
    dataset_hint: Optional[str],
    target_dataset_prompt: Optional[str],
    trace: TraceCallback = None,
) -> Dict[str, Any]:
    paper_dir = Path(output_dir).expanduser().resolve() / "paper_intake"
    paper_dir.mkdir(parents=True, exist_ok=True)
    max_rounds = _paper_intake_env_int("CYTOBRIDGE_PAPER_INTAKE_DISCOVERY_ROUNDS", 2, 0, 8)
    max_actions = _paper_intake_env_int("CYTOBRIDGE_PAPER_INTAKE_DISCOVERY_ACTIONS", 4, 1, 8)

    assets: List[DataAsset] = list(seed_assets or [])
    fetched_pages: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    llm_rounds: List[Dict[str, Any]] = []
    trace_file = paper_dir / "llm_discovery_trace.json"
    last_payload: Dict[str, Any] = {}

    def write_trace_snapshot(status: str, current: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "status": status,
            "current": current or {},
            "target_dataset_prompt": str(target_dataset_prompt or ""),
            "dataset_hint": str(dataset_hint or ""),
            "seed_asset_count": len(seed_assets or []),
            "candidate_assets": [_asset_observation(asset) for asset in _dedupe_assets(assets)],
            "fetched_pages": fetched_pages,
            "observations": observations,
            "llm_rounds": llm_rounds,
            "last_selection_payload": last_payload,
        }
        return _write_json(trace_file, payload)

    trace_path = write_trace_snapshot("started")

    for url in _dedupe(list(hinted_urls))[:8]:
        trace_path = write_trace_snapshot("executing_seed_url", {"url": url})
        new_assets, new_pages, new_obs = _observe_url_candidate(url, source="llm_pdf_hint", trace=trace)
        assets.extend(new_assets)
        fetched_pages.extend(new_pages)
        observations.extend(new_obs)
        trace_path = write_trace_snapshot("after_seed_url", {"url": url})

    trace_path = write_trace_snapshot("resolving_seed_accessions")
    accession_assets, accession_obs = _resolve_accession_text_with_observations(
        "\n".join(str(item) for item in hinted_accessions),
        source="llm_pdf_hint",
        trace=trace,
    )
    assets.extend(accession_assets)
    observations.extend(accession_obs)
    trace_path = write_trace_snapshot("after_seed_accessions")

    if not fetched_pages:
        for url in _seed_page_urls(inspection)[:3]:
            trace_path = write_trace_snapshot("executing_fallback_seed_url", {"url": url})
            new_assets, new_pages, new_obs = _observe_url_candidate(url, source="fallback_seed", trace=trace)
            assets.extend(new_assets)
            fetched_pages.extend(new_pages)
            observations.extend(new_obs)
            trace_path = write_trace_snapshot("after_fallback_seed_url", {"url": url})

    for round_index in range(1, max_rounds + 1):
        trace_path = write_trace_snapshot("waiting_for_llm_round", {"round": round_index})
        system_prompt, user_prompt = _discovery_round_prompt(
            inspection=inspection,
            parsed_pdf=parsed_pdf,
            dataset_hint=dataset_hint,
            target_dataset_prompt=target_dataset_prompt,
            assets=assets,
            fetched_pages=fetched_pages,
            observations=observations,
            round_index=round_index,
            max_actions=max_actions,
        )
        raw = _call_llm_text(
            llm,
            system_prompt,
            user_prompt,
            agent_name=f"paper_intake_discovery_round_{round_index}",
        ) or ""
        payload = _extract_json_from_text(raw) or {}
        last_payload = payload
        actions = _coerce_discovery_actions(payload)[:max_actions]
        round_record: Dict[str, Any] = {
            "round": round_index,
            "parsed": payload,
            "raw_response": raw,
            "actions": actions,
        }
        llm_rounds.append(round_record)
        trace_path = write_trace_snapshot("after_llm_round", {"round": round_index, "action_count": len(actions)})
        if not actions or bool(payload.get("stop_discovery")):
            break
        for action in actions:
            trace_path = write_trace_snapshot("executing_llm_action", {"round": round_index, "action": action})
            new_assets, new_pages, new_obs = _execute_discovery_action(action, trace=trace)
            assets.extend(new_assets)
            fetched_pages.extend(new_pages)
            observations.extend(new_obs)
            round_record.setdefault("action_observations", []).extend(new_obs)
            trace_path = write_trace_snapshot("after_llm_action", {"round": round_index, "action": action})

    deduped_assets = _dedupe_assets(assets)
    trace_path = _write_json(
        trace_file,
        {
            "status": "completed",
            "current": {},
            "target_dataset_prompt": str(target_dataset_prompt or ""),
            "dataset_hint": str(dataset_hint or ""),
            "seed_asset_count": len(seed_assets or []),
            "candidate_assets": [_asset_observation(asset) for asset in deduped_assets],
            "fetched_pages": fetched_pages,
            "observations": observations,
            "llm_rounds": llm_rounds,
            "last_selection_payload": last_payload,
        },
    )
    _trace(trace, f"[paper_intake.llm] wrote discovery trace to {trace_path}")
    return {
        "assets": deduped_assets,
        "fetched_pages": fetched_pages,
        "observations": observations,
        "last_payload": last_payload,
        "trace_path": trace_path,
    }


def enrich_inspection_with_llm(
    inspection: ManuscriptInspection,
    output_dir: str,
    *,
    llm: Any = None,
    seed_assets: Optional[List[DataAsset]] = None,
    dataset_hint: Optional[str] = None,
    target_dataset_prompt: Optional[str] = None,
    trace: TraceCallback = None,
) -> Dict[str, Any]:
    paper_llm = resolve_paper_intake_llm(llm)
    paper_dir = Path(output_dir).expanduser().resolve() / "paper_intake"
    paper_dir.mkdir(parents=True, exist_ok=True)

    if paper_llm is None:
        return {
            "llm_enabled": False,
            "candidate_assets": [asset.to_dict() for asset in (seed_assets or [])],
            "selected_asset_urls": [],
            "fallback_asset_urls": [],
            "supporting_asset_urls": [],
            "dataset_hint": dataset_hint or "",
            "target_dataset_prompt": target_dataset_prompt or "",
            "pdf_analysis_path": "",
            "web_analysis_path": "",
            "discovery_trace_path": "",
            "fetched_pages": [],
            "discovery_observations": [],
        }

    primary_text = _read_text(inspection.primary_text_path)
    article_text = _read_text(inspection.article_text_path)
    focus_snippets = _collect_pdf_focus_snippets(inspection)
    relevant_sentences = inspection.relevant_sentences[:20]
    target_scope_block = _target_scope_prompt_block(dataset_hint, target_dataset_prompt)

    pdf_system = (
        "You are helping build a paper-to-AnnData workflow for single-cell sequencing papers. "
        "Read the manuscript carefully and identify the most likely places where original data resources are exposed. "
        "If the user/main agent supplied a target dataset prompt, focus links/accessions and dataset hints on that requested subset. "
        "Return JSON only."
    )
    pdf_user = (
        "Return a JSON object with keys: summary, likely_article_urls, likely_data_links, "
        "data_accessions, dataset_hints, search_focus, notes.\n\n"
        f"Paper title: {inspection.title or '(unknown)'}\n"
        f"DOI: {inspection.doi or '(not detected)'}\n"
        f"Existing discovered URLs: {json.dumps(inspection.discovered_urls[:40], ensure_ascii=False)}\n"
        f"Existing accessions: {json.dumps(inspection.accessions, ensure_ascii=False)}\n"
        f"{target_scope_block}\n"
        f"Program-relevant sentences: {json.dumps(relevant_sentences, ensure_ascii=False)}\n"
        f"Program PDF focus snippets: {json.dumps(focus_snippets, ensure_ascii=False)}\n\n"
        "Full manuscript text follows.\n"
        f"{_truncate(primary_text, 50000)}\n\n"
        "Fetched article text follows when available.\n"
        f"{_truncate(article_text, 20000)}"
    )
    raw_pdf = _call_llm_text(paper_llm, pdf_system, pdf_user, agent_name="paper_intake_pdf_round")
    parsed_pdf = _extract_json_from_text(raw_pdf or "") or {}
    pdf_analysis_path = _write_json(
        paper_dir / "llm_pdf_analysis.json",
        {
            "parsed": parsed_pdf,
            "raw_response": raw_pdf or "",
        },
    )
    _trace(trace, f"[paper_intake.llm] wrote PDF round analysis to {pdf_analysis_path}")

    hinted_urls = _select_string_list(parsed_pdf, "likely_article_urls") + _select_string_list(parsed_pdf, "likely_data_links")
    hinted_accessions = _select_accession_strings(parsed_pdf, "data_accessions")
    hinted_focus = [str(item) for item in parsed_pdf.get("search_focus", []) or []]

    discovery = _run_llm_discovery_loop(
        inspection=inspection,
        output_dir=output_dir,
        llm=paper_llm,
        parsed_pdf=parsed_pdf,
        seed_assets=list(seed_assets or []),
        hinted_urls=hinted_urls,
        hinted_accessions=hinted_accessions,
        dataset_hint=dataset_hint,
        target_dataset_prompt=target_dataset_prompt,
        trace=trace,
    )
    fetched_pages = list(discovery.get("fetched_pages") or [])
    discovery_observations = list(discovery.get("observations") or [])
    discovery_payload = dict(discovery.get("last_payload") or {})
    combined_assets = _dedupe_assets(list(discovery.get("assets") or []))
    asset_preview = [
        {
            "url": asset.url,
            "provider": asset.provider,
            "asset_kind": asset.asset_kind,
            "file_format": asset.file_format,
            "filename": asset.filename,
            "score": asset.score,
            "description": asset.description,
        }
        for asset in combined_assets[:20]
    ]

    web_system = (
        "You are making the final dataset-selection decision for a paper-to-AnnData workflow. "
        "The heuristic score is only a weak prior; prioritize the manuscript evidence, fetched page content, "
        "recoverable access errors, and whether the asset can become a usable AnnData with cell metadata/time context. "
        "If a target dataset prompt is present, it is a hard scope constraint and overrides broad full-paper coverage. "
        "Return JSON only."
    )
    web_user = (
        "Return a JSON object with keys: selected_asset_urls, supporting_asset_urls, fallback_asset_urls, "
        "dataset_hint, reasoning, notes.\n\n"
        f"User dataset hint: {dataset_hint or ''}\n"
        f"{target_scope_block}\n"
        f"LLM search focus from PDF round: {json.dumps(hinted_focus[:12], ensure_ascii=False)}\n"
        f"Discovery loop selection so far: {json.dumps(discovery_payload, ensure_ascii=False)}\n"
        f"Fetched web pages: {json.dumps(fetched_pages[-6:], ensure_ascii=False)}\n"
        f"Discovery observations including failed URLs/provider errors: {json.dumps(discovery_observations[-24:], ensure_ascii=False)}\n"
        f"Candidate assets: {json.dumps(asset_preview, ensure_ascii=False)}\n"
        "Select one or more candidate asset URLs from the provided inventory. Put the primary expression/count "
        "asset first. Use supporting_asset_urls for metadata sidecars or files that should be downloaded together. "
        "Do not put assets from other paper arms/samples into supporting_asset_urls. Do not select full-study archives "
        "when subset-specific processed files match the target. "
        "If none look perfect, still pick the best analyzable fallback and explain the residual risk."
    )
    raw_web = _call_llm_text(paper_llm, web_system, web_user, agent_name="paper_intake_web_round")
    parsed_web = _extract_json_from_text(raw_web or "") or {}
    web_analysis_path = _write_json(
        paper_dir / "llm_web_analysis.json",
        {
            "parsed": parsed_web,
            "raw_response": raw_web or "",
            "target_dataset_prompt": str(target_dataset_prompt or ""),
            "fetched_pages": fetched_pages,
            "discovery_observations": discovery_observations,
            "candidate_assets": asset_preview,
        },
    )
    _trace(trace, f"[paper_intake.llm] wrote web round analysis to {web_analysis_path}")

    resolved_hint = str(parsed_web.get("dataset_hint") or "").strip() or str(dataset_hint or "").strip()
    selected_urls = _select_string_list(parsed_web, "selected_asset_urls")
    if not selected_urls:
        selected_urls = _select_string_list(discovery_payload, "selected_asset_urls")
    fallback_urls = _select_string_list(parsed_web, "fallback_asset_urls")
    if not fallback_urls:
        fallback_urls = _select_string_list(discovery_payload, "fallback_asset_urls")
    return {
        "llm_enabled": True,
        "candidate_assets": [asset.to_dict() for asset in combined_assets],
        "selected_asset_urls": selected_urls,
        "supporting_asset_urls": _select_string_list(parsed_web, "supporting_asset_urls"),
        "fallback_asset_urls": fallback_urls,
        "dataset_hint": resolved_hint,
        "target_dataset_prompt": str(target_dataset_prompt or ""),
        "pdf_analysis_path": pdf_analysis_path,
        "web_analysis_path": web_analysis_path,
        "discovery_trace_path": str(discovery.get("trace_path") or ""),
        "fetched_pages": fetched_pages,
        "discovery_observations": discovery_observations,
    }


def apply_llm_asset_preferences(
    assets: Sequence[DataAsset],
    *,
    selected_asset_urls: Optional[Sequence[str]] = None,
    fallback_asset_urls: Optional[Sequence[str]] = None,
    dataset_hint: str = "",
) -> List[DataAsset]:
    selected = {str(item).strip() for item in (selected_asset_urls or []) if str(item).strip()}
    fallback = {str(item).strip() for item in (fallback_asset_urls or []) if str(item).strip()}

    ranked: List[DataAsset] = []
    for asset in assets:
        score = float(asset.score)
        if asset.url in selected:
            score += 120.0
        elif asset.url in fallback:
            score += 40.0
        if dataset_hint and _asset_matches_hint(asset, dataset_hint):
            score += 22.0
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


def _default_materialization_script() -> str:
    return """# Available globals:
# - asset, staged_path, work_root, downloaded_files, raw_reference_path
# - convert_local_path_to_anndata, try_merge_obs_table, read_sidecar_table
# - Path, pd, np, ad, json

adata, details = convert_local_path_to_anndata(
    asset,
    str(staged_path),
    str(work_root),
    raw_reference_path=raw_reference_path,
)

paper_specific_notes = []
merge_reports = []

table_candidates = [
    Path(item["path"])
    for item in downloaded_files
    if item.get("metadata_like") or item.get("file_format") == "table"
]

for table_path in table_candidates:
    report = try_merge_obs_table(
        adata,
        table_path,
        barcode_candidates=[
            "barcode",
            "barcodes",
            "cell",
            "cell_id",
            "cellid",
            "cell_barcode",
            "cellbarcode",
            "obs_names",
        ],
    )
    if report.get("merged_columns"):
        merge_reports.append(report)

if merge_reports:
    details["merge_reports"] = merge_reports

if "time_point_raw" not in adata.obs.columns:
    for candidate in list(adata.obs.columns):
        lowered = candidate.lower()
        if any(token in lowered for token in ("time", "day", "stage", "age", "week")):
            adata.obs["time_point_raw"] = adata.obs[candidate].astype(str)
            paper_specific_notes.append(f"Created time_point_raw from {candidate}")
            break

details["paper_specific_notes"] = paper_specific_notes
"""


def _materialization_prompt(
    *,
    selected_asset: DataAsset,
    downloaded_files: List[Dict[str, object]],
    manuscript_summary: Dict[str, Any],
    base_script: str,
    previous_script: str = "",
    previous_error: str = "",
    previous_validation: Optional[Dict[str, Any]] = None,
    ranked_assets: Optional[List[DataAsset]] = None,
) -> Tuple[str, str]:
    system_prompt = (
        "You are editing a paper-specific Python materialization script for a scientific workflow. "
        "Your goal is to produce a usable AnnData input for CytoBridge preprocessing. "
        "Use only the provided helper functions and local downloaded files. "
        "Do not invent unavailable files or network calls. If manuscript_summary includes target_dataset_prompt, "
        "preserve that subset scope in the AnnData and do not merge unrelated paper arms. Return JSON only."
    )
    asset_preview = []
    for asset in list(ranked_assets or [])[:8]:
        asset_preview.append(
            {
                "url": asset.url,
                "provider": asset.provider,
                "asset_kind": asset.asset_kind,
                "file_format": asset.file_format,
                "filename": asset.filename,
                "score": asset.score,
            }
        )
    user_prompt = (
        "Return a JSON object with keys: reasoning, selected_asset_url, script, notes.\n\n"
        "Rules for the script:\n"
        "- It must assign `adata` and `details`.\n"
        "- It may call `convert_local_path_to_anndata(...)` once and then refine `adata.obs`.\n"
        "- It may use `try_merge_obs_table(...)` and `read_sidecar_table(...)` on local files.\n"
        "- It should try to preserve or create a time-like column when manuscript evidence suggests one exists.\n"
        "- If target_dataset_prompt is non-empty, keep only the requested paper subset when local files contain multiple arms/samples.\n"
        "- It must not download additional data or call external services.\n\n"
        f"Selected asset right now: {json.dumps(selected_asset.to_dict(), ensure_ascii=False)}\n"
        f"Top ranked fallback assets: {json.dumps(asset_preview, ensure_ascii=False)}\n"
        f"Downloaded local files: {json.dumps(downloaded_files[:40], ensure_ascii=False)}\n"
        f"Manuscript summary: {json.dumps(manuscript_summary, ensure_ascii=False)}\n"
        f"Base script template:\n{base_script}\n\n"
        f"Previous script (if any):\n{previous_script}\n\n"
        f"Previous execution error: {previous_error}\n"
        f"Previous validation: {json.dumps(previous_validation or {}, ensure_ascii=False)}\n"
    )
    return system_prompt, user_prompt


def _coerce_materialization_payload(raw_text: str, fallback_script: str) -> Dict[str, Any]:
    parsed = _extract_json_from_text(raw_text) or {}
    script = str(parsed.get("script") or "").strip()
    if not script:
        fenced = re.findall(r"```(?:python)?\s*(.*?)```", raw_text or "", flags=re.DOTALL)
        if fenced:
            script = str(fenced[0]).strip()
    if not script:
        script = fallback_script
    return {
        "reasoning": str(parsed.get("reasoning") or "").strip(),
        "selected_asset_url": str(parsed.get("selected_asset_url") or "").strip(),
        "script": script,
        "notes": list(parsed.get("notes") or []),
    }


def _should_rotate_asset_after_error(error_text: str) -> bool:
    lowered = str(error_text or "").lower()
    return any(
        token in lowered
        for token in (
            "missing features/barcodes",
            "mex bundle",
            "unsupported processed data format",
            "could not parse numeric matrix",
            "dimensions do not match",
        )
    )


def draft_materialization_script(
    *,
    selected_asset: DataAsset,
    ranked_assets: List[DataAsset],
    manuscript_summary: Dict[str, Any],
    output_dir: str,
    staged_path: str | Path,
    llm: Any = None,
) -> Dict[str, Any]:
    paper_llm = resolve_paper_intake_llm(llm)
    paper_dir = Path(output_dir).expanduser().resolve() / "paper_intake"
    paper_dir.mkdir(parents=True, exist_ok=True)

    staged = Path(staged_path).expanduser().resolve()
    downloaded_files = describe_local_inputs(staged)
    base_script = _default_materialization_script()
    payload = {
        "reasoning": "LLM disabled; using default materialization script.",
        "selected_asset_url": selected_asset.url,
        "script": base_script,
        "notes": [],
    }
    raw_response = ""

    if paper_llm is not None:
        system_prompt, user_prompt = _materialization_prompt(
            selected_asset=selected_asset,
            downloaded_files=downloaded_files,
            manuscript_summary=manuscript_summary,
            base_script=base_script,
            ranked_assets=ranked_assets,
        )
        raw_response = _call_llm_text(
            paper_llm,
            system_prompt,
            user_prompt,
            agent_name="paper_intake_materialize_prefetch_plan",
        ) or ""
        payload = _coerce_materialization_payload(raw_response, base_script)
        if payload.get("selected_asset_url") and payload["selected_asset_url"] != selected_asset.url:
            notes = list(payload.get("notes") or [])
            notes.append(
                "Selected asset override from the planning round was ignored because the login-node prefetch "
                "stage has already staged a specific local asset for compute-node materialization."
            )
            payload["notes"] = notes
            payload["selected_asset_url"] = selected_asset.url

    script_path = paper_dir / "paper_materialize_prefetch_plan.py"
    script_path.write_text(str(payload.get("script") or base_script), encoding="utf-8")
    plan_path = _write_json(
        paper_dir / "paper_materialize_prefetch_plan.json",
        {
            "selected_asset": selected_asset.to_dict(),
            "selected_asset_url": str(payload.get("selected_asset_url") or selected_asset.url),
            "reasoning": str(payload.get("reasoning") or ""),
            "notes": list(payload.get("notes") or []),
            "script_path": str(script_path),
            "downloaded_files": downloaded_files[:80],
            "raw_response": raw_response,
        },
    )
    return {
        "selected_asset_url": str(payload.get("selected_asset_url") or selected_asset.url),
        "reasoning": str(payload.get("reasoning") or ""),
        "notes": list(payload.get("notes") or []),
        "script_path": str(script_path),
        "plan_path": plan_path,
        "downloaded_files": downloaded_files,
    }


def execute_materialization_script_path(
    script_path: str | Path,
    *,
    asset: DataAsset,
    staged_path: str | Path,
    work_root: str | Path,
    raw_reference_path: Optional[str] = None,
) -> Dict[str, Any]:
    script_file = Path(script_path).expanduser().resolve()
    staged = Path(staged_path).expanduser().resolve()
    work_dir = Path(work_root).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = describe_local_inputs(staged)
    adata, details = _execute_materialization_script(
        script_file,
        asset=asset,
        staged_path=staged,
        work_root=work_dir,
        downloaded_files=downloaded_files,
        raw_reference_path=raw_reference_path,
    )
    return {
        "adata": adata,
        "details": details,
        "downloaded_files": downloaded_files,
        "script_path": str(script_file),
    }


def _execute_materialization_script(
    script_path: Path,
    *,
    asset: DataAsset,
    staged_path: Path,
    work_root: Path,
    downloaded_files: List[Dict[str, object]],
    raw_reference_path: Optional[str],
) -> Tuple[ad.AnnData, Dict[str, Any]]:
    namespace: Dict[str, Any] = {
        "__name__": "__paper_intake_materialize__",
        "asset": asset,
        "staged_path": staged_path,
        "work_root": work_root,
        "downloaded_files": downloaded_files,
        "raw_reference_path": raw_reference_path,
        "convert_local_path_to_anndata": convert_local_path_to_anndata,
        "try_merge_obs_table": try_merge_obs_table,
        "read_sidecar_table": read_sidecar_table,
        "Path": Path,
        "pd": pd,
        "np": np,
        "ad": ad,
        "json": json,
    }
    code = script_path.read_text(encoding="utf-8")
    exec(compile(code, str(script_path), "exec"), namespace, namespace)
    adata = namespace.get("adata")
    details = namespace.get("details", {})
    if not isinstance(adata, ad.AnnData):
        raise PaperIntakeError(
            f"Paper-specific script did not create an AnnData object at {script_path}"
        )
    if not isinstance(details, dict):
        details = {"details_rendered": str(details)}
    return adata, details


def _select_recovery_asset_after_error(
    *,
    error_text: str,
    current_asset: DataAsset,
    ranked_assets: List[DataAsset],
    attempted_asset_urls: Sequence[str],
    manuscript_summary: Dict[str, Any],
    paper_llm: Any,
    allow_asset_switch: bool,
) -> Optional[DataAsset]:
    if not allow_asset_switch:
        return None

    asset_by_url = {asset.url: asset for asset in ranked_assets}
    attempted = {str(url) for url in attempted_asset_urls}
    candidate_preview = [
        {
            "url": asset.url,
            "provider": asset.provider,
            "asset_kind": asset.asset_kind,
            "file_format": asset.file_format,
            "filename": asset.filename,
            "score": asset.score,
            "already_attempted": asset.url in attempted,
        }
        for asset in ranked_assets[:20]
    ]

    if paper_llm is not None:
        system_prompt = (
            "You are recovering from a failed paper-intake download or materialization attempt. "
            "Choose the next asset only from the provided candidate inventory. Return JSON only."
        )
        user_prompt = (
            "Return a JSON object with keys: selected_asset_url, reasoning, stop.\n"
            "If the current error means the asset is missing, inaccessible, or structurally wrong, choose a different "
            "unattempted fallback. If no candidate can work, set stop=true and explain why.\n\n"
            f"Current asset: {json.dumps(current_asset.to_dict(), ensure_ascii=False)}\n"
            f"Attempted asset URLs: {json.dumps(list(attempted), ensure_ascii=False)}\n"
            f"Error: {error_text}\n"
            f"Candidate inventory: {json.dumps(candidate_preview, ensure_ascii=False)}\n"
            f"Manuscript summary: {json.dumps(manuscript_summary, ensure_ascii=False)}"
        )
        try:
            raw = _call_llm_text(
                paper_llm,
                system_prompt,
                user_prompt,
                agent_name="paper_intake_recover_asset_after_error",
            ) or ""
            parsed = _extract_json_from_text(raw) or {}
            if not bool(parsed.get("stop")):
                requested = str(parsed.get("selected_asset_url") or parsed.get("next_asset_url") or "").strip()
                if requested in asset_by_url and requested not in attempted:
                    return asset_by_url[requested]
        except Exception:
            pass

    return next((candidate for candidate in ranked_assets if candidate.url not in attempted), None)


def _download_supporting_assets(
    *,
    supporting_assets: Sequence[DataAsset],
    current_asset: DataAsset,
    download_root: Path,
    staged_cache: Dict[str, Path],
    trace: TraceCallback = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for support in supporting_assets:
        if support.url == current_asset.url:
            continue
        record: Dict[str, Any] = {
            "asset": support.to_dict(),
            "ok": False,
            "path": "",
        }
        staged_path = staged_cache.get(support.url)
        if staged_path is not None:
            record.update({"ok": True, "path": str(staged_path), "cached": True})
            records.append(record)
            continue
        try:
            _trace(trace, f"[paper_intake.llm] downloading supporting asset url={support.url}")
            staged_path = download_asset(support, str(download_root), progress_callback=trace)
            staged_cache[support.url] = staged_path
            record.update({"ok": True, "path": str(staged_path)})
        except Exception as exc:  # noqa: BLE001 - supporting files are helpful but not always mandatory
            record["error"] = _error_payload(exc)
            _trace(trace, f"[paper_intake.llm] supporting asset download failed url={support.url}: {exc}")
        records.append(record)
    return records


def materialize_with_llm_script(
    *,
    selected_asset: DataAsset,
    ranked_assets: List[DataAsset],
    manuscript_summary: Dict[str, Any],
    output_dir: str,
    llm: Any = None,
    raw_reference_path: Optional[str] = None,
    require_downstream_ready: bool = True,
    max_attempts: int = 3,
    allow_asset_switch: bool = True,
    supporting_assets: Optional[List[DataAsset]] = None,
    trace: TraceCallback = None,
) -> Dict[str, Any]:
    paper_llm = resolve_paper_intake_llm(llm)
    paper_dir = Path(output_dir).expanduser().resolve() / "paper_intake"
    paper_dir.mkdir(parents=True, exist_ok=True)

    supporting_assets = list(supporting_assets or [])
    asset_by_url = {asset.url: asset for asset in list(ranked_assets) + supporting_assets}
    download_root = resolve_paper_intake_download_root(output_dir, paper_dir=paper_dir, trace=trace)
    materialize_root = paper_dir / "materialize"
    materialize_root.mkdir(parents=True, exist_ok=True)

    staged_cache: Dict[str, Path] = {}
    current_asset = selected_asset
    previous_script = ""
    previous_error = ""
    previous_validation: Dict[str, Any] = {}
    attempts: List[Dict[str, Any]] = []
    best_success: Optional[Dict[str, Any]] = None
    attempted_asset_urls: List[str] = []

    for attempt_index in range(1, max(1, int(max_attempts)) + 1):
        if current_asset.url not in attempted_asset_urls:
            attempted_asset_urls.append(current_asset.url)
        staged_path = staged_cache.get(current_asset.url)
        if staged_path is None:
            _trace(trace, f"[paper_intake.llm] downloading asset attempt={attempt_index} url={current_asset.url}")
            try:
                staged_path = download_asset(current_asset, str(download_root), progress_callback=trace)
                staged_cache[current_asset.url] = staged_path
            except Exception as exc:  # noqa: BLE001 - recoverable; feed it back to the LLM
                previous_error = f"Download failed for {current_asset.url}: {exc}"
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "stage": "download",
                        "success": False,
                        "selected_asset": current_asset.to_dict(),
                        "error": previous_error,
                        "traceback": traceback.format_exc(),
                    }
                )
                next_asset = _select_recovery_asset_after_error(
                    error_text=previous_error,
                    current_asset=current_asset,
                    ranked_assets=ranked_assets,
                    attempted_asset_urls=attempted_asset_urls,
                    manuscript_summary=manuscript_summary,
                    paper_llm=paper_llm,
                    allow_asset_switch=allow_asset_switch,
                )
                if paper_llm is None or next_asset is None:
                    break
                current_asset = next_asset
                _trace(
                    trace,
                    f"[paper_intake.llm] switching asset after download failure attempt={attempt_index} url={current_asset.url}",
                )
                continue

        supporting_downloads = _download_supporting_assets(
            supporting_assets=supporting_assets,
            current_asset=current_asset,
            download_root=download_root,
            staged_cache=staged_cache,
            trace=trace,
        )
        downloaded_files = describe_local_inputs(staged_path)
        for support_record in supporting_downloads:
            if support_record.get("ok"):
                continue
            downloaded_files.append(
                {
                    "path": "",
                    "relative_path": "",
                    "name": support_record.get("asset", {}).get("filename", ""),
                    "asset_kind": support_record.get("asset", {}).get("asset_kind", ""),
                    "file_format": support_record.get("asset", {}).get("file_format", ""),
                    "metadata_like": True,
                    "download_error": support_record.get("error", {}),
                    "source_url": support_record.get("asset", {}).get("url", ""),
                }
            )
        base_script = _default_materialization_script()
        payload = {
            "reasoning": "LLM disabled; using default materialization script.",
            "selected_asset_url": current_asset.url,
            "script": base_script,
            "notes": [],
        }

        if paper_llm is not None:
            system_prompt, user_prompt = _materialization_prompt(
                selected_asset=current_asset,
                downloaded_files=downloaded_files,
                manuscript_summary=manuscript_summary,
                base_script=base_script,
                previous_script=previous_script,
                previous_error=previous_error,
                previous_validation=previous_validation,
                ranked_assets=ranked_assets,
            )
            raw = _call_llm_text(
                paper_llm,
                system_prompt,
                user_prompt,
                agent_name=f"paper_intake_materialize_attempt_{attempt_index}",
            ) or ""
            payload = _coerce_materialization_payload(raw, base_script)
            if not raw.strip():
                _trace(
                    trace,
                    f"[paper_intake.llm] empty materialization response attempt={attempt_index} asset={current_asset.url}",
                )

            new_asset_url = payload.get("selected_asset_url") or ""
            if new_asset_url and new_asset_url in asset_by_url:
                if allow_asset_switch:
                    current_asset = asset_by_url[new_asset_url]
                    if current_asset.url not in attempted_asset_urls:
                        attempted_asset_urls.append(current_asset.url)
                    staged_path = staged_cache.get(current_asset.url)
                    if staged_path is None:
                        _trace(
                            trace,
                            f"[paper_intake.llm] switching asset attempt={attempt_index} url={current_asset.url}",
                        )
                        try:
                            staged_path = download_asset(current_asset, str(download_root), progress_callback=trace)
                            staged_cache[current_asset.url] = staged_path
                        except Exception as exc:  # noqa: BLE001 - recoverable; feed it back to the LLM
                            previous_script = str(payload.get("script") or "")
                            previous_error = f"Download failed for switched asset {current_asset.url}: {exc}"
                            attempts.append(
                                {
                                    "attempt": attempt_index,
                                    "stage": "download",
                                    "success": False,
                                    "selected_asset": current_asset.to_dict(),
                                    "llm_reasoning": str(payload.get("reasoning") or ""),
                                    "llm_notes": list(payload.get("notes") or []),
                                    "error": previous_error,
                                    "traceback": traceback.format_exc(),
                                }
                            )
                            next_asset = _select_recovery_asset_after_error(
                                error_text=previous_error,
                                current_asset=current_asset,
                                ranked_assets=ranked_assets,
                                attempted_asset_urls=attempted_asset_urls,
                                manuscript_summary=manuscript_summary,
                                paper_llm=paper_llm,
                                allow_asset_switch=allow_asset_switch,
                            )
                            if next_asset is None:
                                break
                            current_asset = next_asset
                            _trace(
                                trace,
                                f"[paper_intake.llm] switching asset after switched download failure attempt={attempt_index} url={current_asset.url}",
                            )
                            continue
                    supporting_downloads = _download_supporting_assets(
                        supporting_assets=supporting_assets,
                        current_asset=current_asset,
                        download_root=download_root,
                        staged_cache=staged_cache,
                        trace=trace,
                    )
                    downloaded_files = describe_local_inputs(staged_path)
                    for support_record in supporting_downloads:
                        if support_record.get("ok"):
                            continue
                        downloaded_files.append(
                            {
                                "path": "",
                                "relative_path": "",
                                "name": support_record.get("asset", {}).get("filename", ""),
                                "asset_kind": support_record.get("asset", {}).get("asset_kind", ""),
                                "file_format": support_record.get("asset", {}).get("file_format", ""),
                                "metadata_like": True,
                                "download_error": support_record.get("error", {}),
                                "source_url": support_record.get("asset", {}).get("url", ""),
                            }
                        )
                else:
                    _trace(
                        trace,
                        f"[paper_intake.llm] ignored asset switch attempt={attempt_index} locked_url={current_asset.url} requested_url={new_asset_url}",
                    )

        script_path = paper_dir / f"paper_materialize_attempt_{attempt_index:02d}.py"
        script_path.write_text(str(payload.get("script") or base_script), encoding="utf-8")

        attempt_record: Dict[str, Any] = {
            "attempt": attempt_index,
            "selected_asset": current_asset.to_dict(),
            "script_path": str(script_path),
            "llm_reasoning": str(payload.get("reasoning") or ""),
            "llm_notes": list(payload.get("notes") or []),
            "downloaded_files": downloaded_files[:80],
            "supporting_downloads": supporting_downloads,
        }

        try:
            adata, details = _execute_materialization_script(
                script_path,
                asset=current_asset,
                staged_path=staged_path,
                work_root=materialize_root / f"attempt_{attempt_index:02d}",
                downloaded_files=downloaded_files,
                raw_reference_path=raw_reference_path,
            )
            adata = standardize_adata(
                adata,
                current_asset,
                manuscript_summary=manuscript_summary,
            )
            validation = dict(adata.uns.get("paper_intake", {}).get("validation", {}))
            attempt_record.update(
                {
                    "success": True,
                    "conversion_details": details,
                    "validation": validation,
                }
            )
            best_success = {
                "adata": adata,
                "selected_asset": current_asset.to_dict(),
                "conversion_details": details,
                "validation": validation,
                "script_path": str(script_path),
            }
            ready = bool(validation.get("ready_for_preprocessing")) or not require_downstream_ready
            attempts.append(attempt_record)
            previous_script = str(payload.get("script") or "")
            previous_error = ""
            previous_validation = validation
            if ready:
                break
        except Exception as exc:
            attempt_record.update(
                {
                    "success": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            previous_script = str(payload.get("script") or "")
            previous_error = str(exc)
            previous_validation = {}
            attempts.append(attempt_record)
            if paper_llm is None:
                break
            if allow_asset_switch and _should_rotate_asset_after_error(previous_error):
                next_asset = next(
                    (candidate for candidate in ranked_assets if candidate.url not in attempted_asset_urls),
                    None,
                )
                if next_asset is not None:
                    current_asset = next_asset
                    _trace(
                        trace,
                        f"[paper_intake.llm] rotating asset after failure attempt={attempt_index} next_url={current_asset.url}",
                    )
            continue

        if paper_llm is None:
            break

    attempts_path = _write_json(
        paper_dir / "materialization_attempts.json",
        {"attempts": attempts},
    )

    if best_success is None:
        raise PaperIntakeError(
            "LLM-assisted materialization did not produce a usable AnnData object. "
            f"See {attempts_path}"
        )

    validation = dict(best_success.get("validation") or {})
    if require_downstream_ready and not bool(validation.get("ready_for_preprocessing")):
        raise PaperIntakeError(
            "Materialization finished, but the resulting AnnData is still not ready for "
            f"CytoBridge preprocessing. See {attempts_path}"
        )

    best_success["materialization_attempts_path"] = attempts_path
    best_success["attempts"] = attempts
    return best_success
