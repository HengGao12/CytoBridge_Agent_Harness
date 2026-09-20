"""
Data inspection and preprocessing tools for CytoBridge Agent.

Provides LangChain-compatible tools for loading, inspecting, and preprocessing
AnnData objects for use with CytoBridge dynamical modeling.
"""
import logging
from typing import Any, Dict, Optional, List
from pathlib import Path

import numpy as np
import scanpy as sc
import anndata as ad
import scipy
from scipy import sparse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions for robust data cleaning
# ---------------------------------------------------------------------------

def _clean_inf_values(adata: ad.AnnData) -> ad.AnnData:
    """
    Replace inf/nan values with 0 in adata.X.
    Handles both sparse and dense matrices efficiently.
    """
    if sparse.issparse(adata.X):
        # For sparse matrices, operate directly on .data array
        data = adata.X.data
        mask = ~np.isfinite(data)
        if mask.any():
            n_bad = mask.sum()
            logger.warning(f"Cleaning {n_bad} inf/nan values from sparse matrix")
            data[mask] = 0.0
            adata.X.eliminate_zeros()
    else:
        # For dense matrices
        mask = ~np.isfinite(adata.X)
        if mask.any():
            n_bad = mask.sum()
            logger.warning(f"Cleaning {n_bad} inf/nan values from dense matrix")
            adata.X[mask] = 0.0
    return adata


def _filter_zero_cells(adata: ad.AnnData, min_total_counts: float = 1.0) -> ad.AnnData:
    """
    Remove cells with zero or very low total counts to prevent division by zero in normalization.
    
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
    """
    Clip values before log1p to prevent overflow in scipy.sparse.expm1.
    The default max_val=100 ensures log1p(100)~4.6 which is safe.
    """
    if sparse.issparse(adata.X):
        # Clip sparse matrix data in place
        np.clip(adata.X.data, 0, max_val, out=adata.X.data)
    else:
        adata.X = np.clip(adata.X, 0, max_val)
    return adata


# ---------------------------------------------------------------------------
# Pydantic models for tool inputs/outputs
# ---------------------------------------------------------------------------

class DataInspectionResult(BaseModel):
    """Result of data inspection."""
    n_cells: int = Field(description="Number of cells")
    n_genes: int = Field(description="Number of genes")
    obs_columns: List[str] = Field(description="Observation column names")
    var_columns: List[str] = Field(description="Variable column names")
    obsm_keys: List[str] = Field(description="Keys in obsm")
    uns_keys: List[str] = Field(description="Keys in uns")
    time_candidates: List[str] = Field(description="Potential time columns")
    label_candidates: List[str] = Field(description="Potential label columns")
    has_raw: bool = Field(description="Whether raw layer exists")
    is_normalized: bool = Field(description="Whether data appears normalized")
    is_log_transformed: bool = Field(description="Whether data appears log-transformed")
    sparsity: float = Field(description="Fraction of zero values")
    summary: str = Field(description="Human-readable summary")


# ---------------------------------------------------------------------------
# Main tool functions
# ---------------------------------------------------------------------------

def inspect_adata(adata_input) -> Dict[str, Any]:
    """
    Inspect an AnnData object and return detailed metadata.
    
    Args:
        adata_input: Path to .h5ad file (str) or AnnData object
        
    Returns:
        Dictionary with inspection results
    """
    # Handle both file path and AnnData object
    if isinstance(adata_input, str):
        logger.info(f"Inspecting AnnData from file: {adata_input}")
        adata = sc.read_h5ad(adata_input)
    elif isinstance(adata_input, ad.AnnData):
        logger.info("Inspecting AnnData object")
        adata = adata_input
    else:
        raise TypeError(f"Expected str or AnnData, got {type(adata_input)}")
    
    # Identify time candidates (numeric or ordered categorical columns)
    time_candidates = []
    time_keywords = ['time', 'day', 'hour', 'stage', 'tp', 'age', 'embryo', 'lineage', 'pseudotime', 'dpt']
    
    for col in adata.obs.columns:
        try:
            vals = adata.obs[col]
            unique_vals = vals.nunique()
            
            # Check for time-related keywords in column name (high priority)
            has_time_keyword = any(kw in col.lower() for kw in time_keywords)
            
            # Check if numeric with reasonable number of unique values
            if np.issubdtype(vals.dtype, np.number):
                # Numeric columns: 2-100 unique values, or has time keyword
                if 2 <= unique_vals <= 100 or has_time_keyword:
                    time_candidates.append(col)
            elif has_time_keyword:
                # Non-numeric but has time keyword
                if 2 <= unique_vals <= 200:
                    time_candidates.append(col)
            # Check for categorical with few levels (potential discrete time points)
            elif hasattr(vals, 'cat') and 2 <= unique_vals <= 50:
                time_candidates.append(col)
        except Exception:
            pass
    
    # Identify label candidates (categorical columns)
    label_candidates = []
    for col in adata.obs.columns:
        try:
            vals = adata.obs[col]
            if hasattr(vals, 'cat') or vals.dtype == object:
                n_unique = vals.nunique()
                if 2 <= n_unique <= 500:  # Reasonable number of cell types
                    label_candidates.append(col)
            elif any(kw in col.lower() for kw in ['cell', 'type', 'cluster', 'label', 'annot']):
                label_candidates.append(col)
        except Exception:
            pass
    
    # Check normalization status
    if sparse.issparse(adata.X):
        x_data = adata.X.data
        max_val = x_data.max() if len(x_data) > 0 else 0
        mean_val = x_data.mean() if len(x_data) > 0 else 0
    else:
        max_val = adata.X.max()
        mean_val = adata.X.mean()
    
    # Heuristics for normalization detection
    is_normalized = max_val < 100 or (1 < mean_val < 20)
    is_log_transformed = max_val < 15 and mean_val < 5
    
    # Calculate sparsity
    if sparse.issparse(adata.X):
        n_nonzero = adata.X.nnz
        total = adata.X.shape[0] * adata.X.shape[1]
        sparsity = 1 - (n_nonzero / total)
    else:
        sparsity = (adata.X == 0).sum() / adata.X.size
    
    # Build summary
    summary = f"""AnnData: {adata.n_obs} cells × {adata.n_vars} genes
Sparsity: {sparsity:.1%}
Time candidates: {time_candidates[:5]}
Label candidates: {label_candidates[:5]}
Normalized: {is_normalized}, Log-transformed: {is_log_transformed}
Has raw: {adata.raw is not None}"""
    
    result = {
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "obs_columns": list(adata.obs.columns),
        "var_columns": list(adata.var.columns),
        "obsm_keys": list(adata.obsm.keys()),
        "uns_keys": list(adata.uns.keys()),
        "time_candidates": time_candidates,
        "label_candidates": label_candidates,
        "has_raw": adata.raw is not None,
        "is_normalized": is_normalized,
        "is_log_transformed": is_log_transformed,
        "sparsity": float(sparsity),
        "summary": summary,
    }
    
    logger.info(f"Inspection complete: {adata.n_obs} cells, {adata.n_vars} genes")
    return result


def preprocess_adata(
    adata: ad.AnnData,
    time_key: str,
    label_key: Optional[str] = None,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    do_normalize: bool = True,
    do_log: bool = True,
) -> ad.AnnData:
    """
    Preprocess AnnData for CytoBridge training with robust inf/nan handling.
    
    Args:
        adata: AnnData object
        time_key: Column name for time points
        label_key: Optional column name for cell labels
        n_hvg: Number of highly variable genes to select
        n_pcs: Number of principal components
        do_normalize: Whether to normalize counts
        do_log: Whether to log-transform
        
    Returns:
        Preprocessed AnnData
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Step 0: Analyze data status BEFORE any processing
    from .preprocessing_planner import analyze_normalization_status
    logger.info("Step 0: Analyzing data status...")
    norm_status = analyze_normalization_status(adata)
    logger.info(f"  Data status: counts={norm_status['is_counts']}, "
                f"normalized={norm_status['is_normalized']}, "
                f"log={norm_status['is_log_transformed']}, "
                f"preprocessed={norm_status.get('is_preprocessed', False)}")
    logger.info(f"  Features: {norm_status['n_features']}, "
                f"Has PCA: {norm_status.get('has_pca', False)}, "
                f"Has latent: {norm_status.get('has_latent', False)}")
    
    # Check if data is already preprocessed
    is_preprocessed = norm_status.get('is_preprocessed', False)
    
    if is_preprocessed:
        logger.info("⚠️  数据已经预处理过（可能是PCA降维后的），跳过大部分预处理步骤")
        logger.info("  只进行必要的清理和time_point设置")
        
        # Step 1: Clean inf/nan only
        logger.info("Step 1/2: Cleaning inf/nan values...")
        adata = _clean_inf_values(adata)
        
        # Step 2: Ensure X_latent exists
        if 'X_latent' not in adata.obsm.keys():
            logger.info("Step 2/2: Creating X_latent from X...")
            if sparse.issparse(adata.X):
                adata.obsm['X_latent'] = adata.X.toarray()
            else:
                adata.obsm['X_latent'] = adata.X.copy()
        
        # Skip normalization, log, HVG, PCA - go directly to time mapping
        logger.info("✅ 预处理完成（数据已预处理，跳过归一化/HVG/PCA）")
        
    else:
        # Full preprocessing pipeline for raw data
        logger.info("📊 数据是原始数据，执行完整预处理流程")
        
        # Step 1: Initial cleaning
        logger.info("Step 1/6: Cleaning inf/nan values...")
        adata = _clean_inf_values(adata)
        
        # Step 2: Filter zero-count cells (adjust threshold based on data status)
        logger.info("Step 2/6: Filtering zero/low-count cells...")
        if norm_status['is_log_transformed']:
            min_threshold = 0.0
            logger.info("  Data appears log-transformed, using threshold=0.0")
        elif norm_status['is_normalized']:
            min_threshold = 0.1
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
        
        # Step 4: Log transform with overflow protection
        if do_log:
            logger.info("Step 4/6: Log-transforming (with overflow protection)...")
            adata = _clip_before_log(adata, max_val=100.0)
            sc.pp.log1p(adata)
            adata = _clean_inf_values(adata)  # Clean after log1p

        # Preserve gene-level matrix before HVG filtering for downstream analysis
        if adata.raw is None:
            adata.raw = adata.copy()
            logger.info("Stored normalized/logged gene matrix in adata.raw for downstream gene-level analysis.")
        
        # Step 5: HVG selection
        logger.info(f"Step 5/6: Selecting {n_hvg} highly variable genes...")
        
        # Final safety check before HVG
        if sparse.issparse(adata.X):
            bad_vals = ~np.isfinite(adata.X.data)
            if bad_vals.any():
                logger.warning(f"Found {bad_vals.sum()} bad values before HVG, fixing...")
                adata.X.data[bad_vals] = 0.0
        else:
            bad_vals = ~np.isfinite(adata.X)
            if bad_vals.any():
                logger.warning(f"Found {bad_vals.sum()} bad values before HVG, fixing...")
                adata.X[bad_vals] = 0.0
        
        sc.pp.highly_variable_genes(adata, n_top_genes=min(n_hvg, adata.n_vars))
        adata = adata[:, adata.var['highly_variable']].copy()
        
        # Step 6: PCA
        logger.info(f"Step 6/6: Computing {n_pcs} PCs...")
        
        # Convert sparse to dense before scaling to avoid warning
        if sparse.issparse(adata.X):
            logger.info("Converting sparse matrix to dense for PCA scaling...")
            adata.X = adata.X.toarray()
        
        sc.pp.scale(adata, max_value=10)
        adata = _clean_inf_values(adata)  # Clean after scaling
        sc.tl.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1))
        sc.pp.neighbors(adata, n_pcs=n_pcs)
        sc.tl.umap(adata)
        
        # Store latent representation - check if PCA was successful
        if 'X_pca' in adata.obsm.keys():
            adata.obsm['X_latent'] = adata.obsm['X_pca'][:, :min(n_pcs, adata.n_vars - 1)]
            logger.info(f"Created X_latent from X_pca: {adata.obsm['X_latent'].shape}")
        else:
            # PCA might have failed, use X directly
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
    
    # Store latent representation
    adata.obsm['X_latent'] = adata.obsm['X_pca'][:, :n_pcs]
    
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


def _fallback_preprocess(
    adata: ad.AnnData,
    time_key: str,
    label_key: Optional[str] = None,
) -> ad.AnnData:
    """
    Fallback preprocessing when LLM planning fails.
    Uses conservative defaults with robust cleaning.
    """
    logger.info("Using fallback preprocessing with defaults...")
    return preprocess_adata(
        adata=adata,
        time_key=time_key,
        label_key=label_key,
        n_hvg=2000,
        n_pcs=50,
        do_normalize=True,
        do_log=True,
    )


# ---------------------------------------------------------------------------
# LangChain Tool wrappers
# ---------------------------------------------------------------------------

class InspectDataTool:
    """LangChain-compatible tool for data inspection."""
    name = "inspect_data"
    description = "Inspect an AnnData file and return metadata about cells, genes, and potential time/label columns."
    
    def __call__(self, adata_path: str) -> Dict[str, Any]:
        return inspect_adata(adata_path)


class PreprocessTool:
    """LangChain-compatible tool for data preprocessing."""
    name = "preprocess_data"
    description = "Preprocess AnnData for CytoBridge training including normalization, HVG selection, and PCA."
    
    def __call__(
        self,
        adata: ad.AnnData,
        time_key: str,
        label_key: Optional[str] = None,
        n_hvg: int = 2000,
        n_pcs: int = 50,
    ) -> ad.AnnData:
        return preprocess_adata(adata, time_key, label_key, n_hvg, n_pcs)
