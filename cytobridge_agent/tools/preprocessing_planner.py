"""
Preprocessing planning tools for CytoBridge Agent.

Provides intelligent preprocessing strategy planning using LLM analysis
of data characteristics and user intent.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import scanpy as sc
import anndata as ad
import scipy
from scipy import sparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Analysis Functions
# ---------------------------------------------------------------------------

def analyze_normalization_status(adata: ad.AnnData) -> Dict[str, Any]:
    """
    Analyze whether data is normalized/log-transformed or already preprocessed (e.g., PCA).
    
    Returns:
        Dictionary with normalization analysis results
    """
    # Check if data is already dimensionality-reduced
    n_features = adata.n_vars
    has_pca = 'X_pca' in adata.obsm
    has_latent = 'X_latent' in adata.obsm
    has_raw = adata.raw is not None
    
    # If X has very few features (< 100), it's likely already PCA-reduced
    is_likely_reduced = n_features < 100
    
    # Analyze X matrix
    if sparse.issparse(adata.X):
        x_data = adata.X.data
        if len(x_data) == 0:
            return {
                "is_normalized": False, 
                "is_log": False, 
                "max_val": 0, 
                "mean_val": 0,
                "is_reduced": True,
                "n_features": n_features,
            }
        max_val = float(x_data.max())
        mean_val = float(x_data.mean())
        median_val = float(np.median(x_data))
    else:
        max_val = float(adata.X.max())
        mean_val = float(adata.X.mean())
        median_val = float(np.median(adata.X[adata.X > 0]))
    
    # Heuristics for normalization status
    is_counts = max_val > 100 and mean_val > 10
    is_normalized = 1 < max_val < 100 and mean_val < 20
    is_log = max_val < 15 and mean_val < 5
    
    # Determine if data is already preprocessed
    is_preprocessed = (
        is_likely_reduced or  # Few features suggest PCA
        has_pca or  # Has PCA results
        has_latent or  # Has latent representation
        (has_raw and n_features < 1000)  # Has raw but X is small
    )
    
    # Recommendation
    if is_preprocessed:
        recommendation = "already_preprocessed"
    elif is_log:
        recommendation = "skip_both"
    elif is_normalized:
        recommendation = "log_only"
    else:
        recommendation = "normalize_and_log"
    
    return {
        "is_counts": is_counts,
        "is_normalized": is_normalized,
        "is_log_transformed": is_log,
        "is_preprocessed": is_preprocessed,
        "is_likely_reduced": is_likely_reduced,
        "n_features": n_features,
        "has_pca": has_pca,
        "has_latent": has_latent,
        "has_raw": has_raw,
        "max_value": max_val,
        "mean_value": mean_val,
        "median_nonzero": median_val,
        "recommendation": recommendation,
    }


def suggest_hvg_count(adata: ad.AnnData) -> int:
    """Suggest number of HVGs based on dataset size."""
    n_genes = adata.n_vars
    n_cells = adata.n_obs
    
    if n_genes < 2000:
        return min(1000, n_genes - 1)
    elif n_cells < 5000:
        return 1500
    elif n_cells < 20000:
        return 2000
    elif n_cells < 50000:
        return 2500
    else:
        return 3000


def suggest_pca_dims(adata: ad.AnnData) -> int:
    """Suggest number of PCA dimensions based on dataset."""
    n_cells = adata.n_obs
    
    if n_cells < 1000:
        return 30
    elif n_cells < 10000:
        return 50
    elif n_cells < 50000:
        return 50
    else:
        return 100


def compare_time_candidates(
    adata: ad.AnnData,
    candidates: List[str]
) -> List[Dict[str, Any]]:
    """
    Analyze and rank time column candidates.
    
    Returns:
        List of candidate analyses sorted by suitability
    """
    results = []
    
    for col in candidates:
        if col not in adata.obs.columns:
            continue
            
        vals = adata.obs[col]
        
        try:
            # Get unique values
            unique_vals = vals.dropna().unique()
            n_unique = len(unique_vals)
            
            # Check if numeric
            is_numeric = np.issubdtype(vals.dtype, np.number)
            
            # Calculate distribution
            val_counts = vals.value_counts()
            min_count = val_counts.min()
            max_count = val_counts.max()
            balance_ratio = min_count / max_count if max_count > 0 else 0
            
            # Score the candidate
            score = 0
            if 2 <= n_unique <= 20:
                score += 50  # Good number of time points
            elif n_unique <= 50:
                score += 30
            
            if is_numeric:
                score += 20
            
            if balance_ratio > 0.1:
                score += 20  # Reasonably balanced
            
            # Bonus for time-related keywords
            if any(kw in col.lower() for kw in ['time', 'day', 'stage', 'tp', 'hour']):
                score += 30
            
            results.append({
                "column": col,
                "n_unique": n_unique,
                "is_numeric": is_numeric,
                "balance_ratio": float(balance_ratio),
                "min_cells_per_tp": int(min_count),
                "max_cells_per_tp": int(max_count),
                "score": score,
                "sample_values": list(unique_vals[:5])
            })
        except Exception as e:
            logger.warning(f"Error analyzing column {col}: {e}")
    
    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Preprocessing Strategy
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingStrategy:
    """Preprocessing strategy determined by LLM."""
    time_key: str
    label_key: Optional[str]
    do_normalize: bool
    do_log: bool
    n_hvg: int
    n_pcs: int
    reasoning: str


def plan_preprocessing_strategy(
    inspection_result: Dict[str, Any],
    adata: ad.AnnData,
    user_question: Optional[str] = None,
) -> PreprocessingStrategy:
    """
    Plan preprocessing strategy using heuristics (fallback when LLM unavailable).
    
    Args:
        inspection_result: Result from inspect_adata
        adata: The AnnData object
        user_question: Optional user research question
        
    Returns:
        PreprocessingStrategy with recommended parameters
    """
    # Analyze normalization status
    norm_status = analyze_normalization_status(adata)
    
    # Select time key
    time_candidates = inspection_result.get("time_candidates", [])
    if time_candidates:
        time_analysis = compare_time_candidates(adata, time_candidates)
        time_key = time_analysis[0]["column"] if time_analysis else time_candidates[0]
    else:
        # Fallback: look for common names
        for col in adata.obs.columns:
            if any(kw in col.lower() for kw in ['time', 'day', 'stage']):
                time_key = col
                break
        else:
            time_key = adata.obs.columns[0]  # Last resort
    
    # Select label key
    label_candidates = inspection_result.get("label_candidates", [])
    label_key = None
    if label_candidates:
        # Prefer columns with 'cell', 'type', 'cluster' in name
        for col in label_candidates:
            if any(kw in col.lower() for kw in ['cell_type', 'celltype', 'cluster', 'annotation']):
                label_key = col
                break
        if label_key is None:
            label_key = label_candidates[0]
    
    # Determine normalization needs
    do_normalize = norm_status["recommendation"] == "normalize_and_log"
    do_log = norm_status["recommendation"] in ["normalize_and_log", "log_only"]
    
    # Suggest HVG and PCA
    n_hvg = suggest_hvg_count(adata)
    n_pcs = suggest_pca_dims(adata)
    
    reasoning = f"""Preprocessing strategy based on data analysis:
- Time key: {time_key} (from {len(time_candidates)} candidates)
- Label key: {label_key}
- Normalization: {'yes' if do_normalize else 'no'} (data {'looks like counts' if norm_status.get('is_counts') else 'appears normalized'})
- Log transform: {'yes' if do_log else 'no'}
- HVGs: {n_hvg} (based on {adata.n_obs} cells, {adata.n_vars} genes)
- PCs: {n_pcs}"""
    
    return PreprocessingStrategy(
        time_key=time_key,
        label_key=label_key,
        do_normalize=do_normalize,
        do_log=do_log,
        n_hvg=n_hvg,
        n_pcs=n_pcs,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Robust Preprocessing Application
# ---------------------------------------------------------------------------

def _clean_inf_values(adata: ad.AnnData) -> ad.AnnData:
    """Replace inf/nan values with 0 in adata.X."""
    if sparse.issparse(adata.X):
        data = adata.X.data
        mask = ~np.isfinite(data)
        if mask.any():
            n_bad = mask.sum()
            logger.warning(f"Cleaning {n_bad} inf/nan values from sparse matrix")
            data[mask] = 0.0
            adata.X.eliminate_zeros()
    else:
        mask = ~np.isfinite(adata.X)
        if mask.any():
            n_bad = mask.sum()
            logger.warning(f"Cleaning {n_bad} inf/nan values from dense matrix")
            adata.X[mask] = 0.0
    return adata


def _filter_zero_cells(adata: ad.AnnData, min_total_counts: float = 1.0) -> ad.AnnData:
    """
    Remove cells with zero or very low total counts.
    
    Args:
        adata: AnnData object
        min_total_counts: Minimum total counts per cell (default: 1.0)
    
    Returns:
        Filtered AnnData
    """
    if sparse.issparse(adata.X):
        cell_counts = np.asarray(adata.X.sum(axis=1)).flatten()
    else:
        cell_counts = np.sum(adata.X, axis=1)
    
    # Log statistics before filtering
    n_cells_before = adata.n_obs
    n_zero = (cell_counts == 0).sum()
    n_very_low = (cell_counts > 0) & (cell_counts < min_total_counts)
    n_very_low_count = n_very_low.sum()
    
    if n_zero > 0 or n_very_low_count > 0:
        logger.info(f"Cell count statistics:")
        logger.info(f"  Total cells: {n_cells_before}")
        logger.info(f"  Zero counts: {n_zero} ({n_zero/n_cells_before*100:.1f}%)")
        logger.info(f"  Very low counts (<{min_total_counts}): {n_very_low_count} ({n_very_low_count/n_cells_before*100:.1f}%)")
        logger.info(f"  Mean counts per cell: {cell_counts.mean():.2f}")
        logger.info(f"  Median counts per cell: {np.median(cell_counts):.2f}")
        
        # Filter cells with counts below threshold
        valid_mask = cell_counts >= min_total_counts
        n_filtered = (~valid_mask).sum()
        
        if n_filtered > n_cells_before * 0.5:
            logger.warning(f"⚠️  Filtering {n_filtered} cells ({n_filtered/n_cells_before*100:.1f}%) - this is a large fraction!")
            logger.warning(f"   This may indicate data quality issues. Remaining cells: {(valid_mask).sum()}")
        else:
            logger.info(f"Filtering {n_filtered} cells with counts < {min_total_counts}")
        
        adata = adata[valid_mask].copy()
        
        logger.info(f"After filtering: {adata.n_obs} cells ({adata.n_obs/n_cells_before*100:.1f}% retained)")
    
    return adata


def _clip_before_log(adata: ad.AnnData, max_val: float = 100.0) -> ad.AnnData:
    """Clip values before log1p to prevent overflow."""
    if sparse.issparse(adata.X):
        np.clip(adata.X.data, 0, max_val, out=adata.X.data)
    else:
        adata.X = np.clip(adata.X, 0, max_val)
    return adata


def apply_custom_preprocessing(
    adata: ad.AnnData,
    strategy,
) -> ad.AnnData:
    """
    Apply preprocessing based on strategy with robust error handling.
    
    Args:
        adata: AnnData object
        strategy: Preprocessing strategy - can be PreprocessingStrategy or dict
        
    Returns:
        Preprocessed AnnData
    """
    # Handle both dict and PreprocessingStrategy
    if isinstance(strategy, dict):
        time_key = strategy.get("time_key", "time")
        label_key = strategy.get("label_key")
        do_normalize = strategy.get("normalize", strategy.get("do_normalize", True))
        do_log = strategy.get("log1p", strategy.get("do_log", True))
        n_hvg = strategy.get("n_top_genes", strategy.get("n_hvg", 2000))
        n_pcs = strategy.get("n_pcs", 50)
    else:
        time_key = strategy.time_key
        label_key = strategy.label_key
        do_normalize = strategy.do_normalize
        do_log = strategy.do_log
        n_hvg = strategy.n_hvg
        n_pcs = strategy.n_pcs
    
    logger.info(f"Applying preprocessing strategy...")
    logger.info(f"  Time key: {time_key}")
    logger.info(f"  Label key: {label_key}")
    logger.info(f"  Normalize: {do_normalize}, Log: {do_log}")
    logger.info(f"  HVGs: {n_hvg}, PCs: {n_pcs}")
    
    # Step 0: Analyze data status BEFORE any processing
    logger.info("Step 0: Analyzing data status...")
    norm_status = analyze_normalization_status(adata)
    logger.info(f"  Data status: counts={norm_status['is_counts']}, "
                f"normalized={norm_status['is_normalized']}, "
                f"log={norm_status['is_log_transformed']}, "
                f"preprocessed={norm_status.get('is_preprocessed', False)}")
    logger.info(f"  Features: {norm_status['n_features']}, "
                f"Has PCA: {norm_status.get('has_pca', False)}, "
                f"Has latent: {norm_status.get('has_latent', False)}")
    logger.info(f"  Value range: max={norm_status['max_value']:.2f}, "
                f"mean={norm_status['mean_value']:.2f}")
    
    # Check if data is already preprocessed (e.g., PCA-reduced)
    is_preprocessed = norm_status.get('is_preprocessed', False)
    
    if is_preprocessed:
        logger.info("⚠️  数据已经预处理过（可能是PCA降维后的），跳过大部分预处理步骤")
        logger.info("  只进行必要的清理和time_point设置")
        
        # Step 1: Clean inf/nan only
        logger.info("Step 1/2: Cleaning inf/nan values...")
        adata = _clean_inf_values(adata)
        
        # Step 2: Ensure X_latent exists (use X if no X_latent)
        if 'X_latent' not in adata.obsm.keys():
            logger.info("Step 2/2: Creating X_latent from X...")
            if sparse.issparse(adata.X):
                adata.obsm['X_latent'] = adata.X.toarray()
            else:
                adata.obsm['X_latent'] = adata.X.copy()
            logger.info(f"  Created X_latent: {adata.obsm['X_latent'].shape}")
        else:
            logger.info("Step 2/2: X_latent already exists, using it")
        
        # Skip normalization, log, HVG, PCA
        logger.info("✅ 预处理完成（数据已预处理，跳过归一化/HVG/PCA）")
        
    else:
        # Full preprocessing pipeline for raw data
        logger.info("📊 数据是原始数据，执行完整预处理流程")
        
        # Step 1: Initial cleaning
        logger.info("Step 1/6: Cleaning inf/nan values...")
        adata = _clean_inf_values(adata)
        
        # Step 2: Filter zero-count cells (adjust threshold based on data status)
        logger.info("Step 2/6: Filtering zero/low-count cells...")
        # If data is already normalized/log, use a much lower threshold
        if norm_status['is_log_transformed']:
            min_threshold = 0.0  # Only filter completely zero cells
            logger.info("  Data appears log-transformed, using threshold=0.0")
        elif norm_status['is_normalized']:
            min_threshold = 0.1  # Very low threshold for normalized data
            logger.info("  Data appears normalized, using threshold=0.1")
        else:
            min_threshold = 1.0
            logger.info("  Data appears to be raw counts, using threshold=1.0")
        
        adata = _filter_zero_cells(adata, min_total_counts=min_threshold)
        
        # Step 3: Normalize
        if do_normalize:
            logger.info("Step 3/6: Normalizing counts...")
            sc.pp.normalize_total(adata, target_sum=1e4)
            adata = _clean_inf_values(adata)
        else:
            logger.info("Step 3/6: Skipping normalization...")
        
        # Step 4: Log transform with overflow protection
        if do_log:
            logger.info("Step 4/6: Log-transforming (with overflow protection)...")
            adata = _clip_before_log(adata, max_val=100.0)
            sc.pp.log1p(adata)
            adata = _clean_inf_values(adata)
        else:
            logger.info("Step 4/6: Skipping log transform...")

        # Preserve gene-level matrix before HVG filtering for downstream analysis
        if adata.raw is None:
            adata.raw = adata.copy()
            logger.info("Stored normalized/logged gene matrix in adata.raw for downstream gene-level analysis.")
        
        # Step 5: HVG selection
        n_hvg_final = min(n_hvg, adata.n_vars - 1)
        logger.info(f"Step 5/6: Selecting {n_hvg_final} highly variable genes...")
        
        # Final safety check
        if sparse.issparse(adata.X):
            bad_vals = ~np.isfinite(adata.X.data)
            if bad_vals.any():
                logger.error(f"Still have {bad_vals.sum()} bad values before HVG!")
                adata.X.data[bad_vals] = 0.0
        else:
            bad_vals = ~np.isfinite(adata.X)
            if bad_vals.any():
                logger.error(f"Still have {bad_vals.sum()} bad values before HVG!")
                adata.X[bad_vals] = 0.0
        
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg_final)
        adata = adata[:, adata.var['highly_variable']].copy()
        
        # Step 6: PCA
        n_pcs_final = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
        logger.info(f"Step 6/6: Computing {n_pcs_final} PCs...")
        
        # Convert sparse to dense before scaling to avoid warning
        if sparse.issparse(adata.X):
            logger.info("Converting sparse matrix to dense for PCA scaling...")
            adata.X = adata.X.toarray()
        
        sc.pp.scale(adata, max_value=10)
        adata = _clean_inf_values(adata)
        sc.tl.pca(adata, n_comps=n_pcs_final)
        
        # Store latent representation - check if PCA was successful
        if 'X_pca' in adata.obsm.keys():
            adata.obsm['X_latent'] = adata.obsm['X_pca'][:, :n_pcs_final]
            logger.info(f"Created X_latent from X_pca: {adata.obsm['X_latent'].shape}")
        else:
            # PCA might have failed or data structure issue, use X directly
            logger.warning("X_pca not found after PCA, using X as X_latent")
            if sparse.issparse(adata.X):
                adata.obsm['X_latent'] = adata.X.toarray()
            else:
                adata.obsm['X_latent'] = adata.X.copy()
    
    # Store time point mapping
    if time_key in adata.obs.columns:
        time_vals = adata.obs[time_key]
        unique_times = sorted(time_vals.unique())
        
        # Filter out time points with too few cells (need at least 2 for OT)
        min_cells_per_tp = 2
        time_counts = time_vals.value_counts()
        valid_times = time_counts[time_counts >= min_cells_per_tp].index.tolist()
        
        if len(valid_times) < 2:
            logger.warning(f"Only {len(valid_times)} time points with ≥{min_cells_per_tp} cells. "
                          f"May cause training issues.")
        elif len(valid_times) < len(unique_times):
            logger.warning(f"Filtering {len(unique_times) - len(valid_times)} time points with <{min_cells_per_tp} cells")
            adata = adata[time_vals.isin(valid_times)].copy()
            time_vals = adata.obs[time_key]
            unique_times = sorted(valid_times)
        
        # Map to consecutive integers starting from 0
        time_map = {t: i for i, t in enumerate(unique_times)}
        adata.obs['time_point_processed'] = time_vals.map(time_map).astype(int)
        
        # Convert keys to strings for h5ad compatibility
        adata.uns['time_point_mapping'] = {str(k): int(v) for k, v in time_map.items()}
        
        # Log final time point distribution
        final_counts = adata.obs['time_point_processed'].value_counts().sort_index()
        logger.info(f"Time points mapped: {len(unique_times)} time points")
        logger.info(f"  Cell counts per TP: min={final_counts.min()}, max={final_counts.max()}, mean={final_counts.mean():.1f}")
    
    # Final validation: check for NaN/inf in X_latent (should already exist from above)
    if 'X_latent' not in adata.obsm.keys():
        logger.warning("X_latent not found, creating from X...")
        if sparse.issparse(adata.X):
            adata.obsm['X_latent'] = adata.X.toarray()
        else:
            adata.obsm['X_latent'] = adata.X.copy()
    
    # Final validation: check for NaN/inf in X_latent
    if sparse.issparse(adata.obsm['X_latent']):
        X_data = adata.obsm['X_latent'].data
        if np.any(~np.isfinite(X_data)):
            n_bad = np.sum(~np.isfinite(X_data))
            logger.warning(f"Found {n_bad} NaN/inf in X_latent, fixing...")
            X_data[~np.isfinite(X_data)] = 0.0
            adata.obsm['X_latent'].eliminate_zeros()
    else:
        if np.any(~np.isfinite(adata.obsm['X_latent'])):
            n_bad = np.sum(~np.isfinite(adata.obsm['X_latent']))
            logger.warning(f"Found {n_bad} NaN/inf in X_latent, fixing...")
            adata.obsm['X_latent'][~np.isfinite(adata.obsm['X_latent'])] = 0.0
    
    logger.info(f"Preprocessing complete: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata
