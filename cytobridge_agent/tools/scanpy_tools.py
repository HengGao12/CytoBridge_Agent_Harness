"""
Scanpy analysis tools for LLM to inspect and analyze AnnData.

These tools allow the LLM to autonomously explore data and make decisions
about preprocessing parameters, time keys, and label keys.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

logger = logging.getLogger(__name__)


def analyze_obs_column(adata: ad.AnnData, column: str) -> Dict[str, Any]:
    """
    Analyze an observation column to determine if it's suitable as time_key or label_key.
    
    Args:
        adata: AnnData object
        column: Column name in adata.obs
        
    Returns:
        Dictionary with analysis results
    """
    if column not in adata.obs.columns:
        return {"error": f"Column '{column}' not found"}
    
    vals = adata.obs[column]
    
    # Basic statistics
    n_unique = vals.nunique()
    n_total = len(vals)
    dtype = str(vals.dtype)
    
    # Check if numeric (handle categorical dtype safely)
    try:
        is_numeric = np.issubdtype(vals.dtype, np.number)
    except TypeError:
        # CategoricalDtype throws TypeError
        is_numeric = False
    
    # Get value distribution
    if is_numeric:
        mean_val = float(vals.mean())
        std_val = float(vals.std())
        min_val = float(vals.min())
        max_val = float(vals.max())
        sample_values = sorted(vals.unique())[:10]
    else:
        mean_val = None
        std_val = None
        min_val = None
        max_val = None
        sample_values = vals.unique().tolist()[:10]
    
    # Value counts
    value_counts = vals.value_counts()
    counts_dict = {str(k): int(v) for k, v in value_counts.head(20).items()}
    
    # Assess suitability
    # Time key criteria: numeric or ordered, reasonable number of unique values (2-100)
    time_score = 0
    if 2 <= n_unique <= 100:
        time_score += 50
    if is_numeric:
        time_score += 30
    if any(kw in column.lower() for kw in ['time', 'day', 'stage', 'tp', 'hour', 'age', 'embryo']):
        time_score += 20
    
    # Label key criteria: categorical, reasonable number of unique values (2-500)
    label_score = 0
    if 2 <= n_unique <= 500:
        label_score += 50
    if not is_numeric or n_unique <= 50:
        label_score += 30
    if any(kw in column.lower() for kw in ['cell', 'type', 'cluster', 'label', 'annot', 'lineage']):
        label_score += 20
    
    return {
        "column": column,
        "n_unique": n_unique,
        "n_total": n_total,
        "dtype": dtype,
        "is_numeric": is_numeric,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "sample_values": [str(v) for v in sample_values],
        "value_counts": counts_dict,
        "time_key_score": time_score,
        "label_key_score": label_score,
        "suitable_as_time": time_score >= 50,
        "suitable_as_label": label_score >= 50,
    }


def list_obs_columns(adata: ad.AnnData) -> List[str]:
    """List all observation column names."""
    return list(adata.obs.columns)


def get_column_summary(adata: ad.AnnData, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get summary statistics for multiple columns.
    
    Args:
        adata: AnnData object
        columns: List of column names (if None, analyzes all columns)
        
    Returns:
        Dictionary with analysis for each column
    """
    if columns is None:
        columns = list(adata.obs.columns)
    
    results = {}
    for col in columns:
        try:
            results[col] = analyze_obs_column(adata, col)
        except Exception as e:
            results[col] = {"error": str(e)}
    
    return results


def check_data_status(adata: ad.AnnData) -> Dict[str, Any]:
    """
    Check if data is already normalized/log-transformed or preprocessed (e.g., PCA-reduced).
    
    Returns:
        Dictionary with data status information
    """
    from scipy import sparse
    
    # Check if data is already dimensionality-reduced
    n_features = adata.n_vars
    has_pca = 'X_pca' in adata.obsm
    has_latent = 'X_latent' in adata.obsm
    has_raw = adata.raw is not None
    
    # If X has very few features (< 100), it's likely already PCA-reduced
    is_likely_reduced = n_features < 100
    
    if sparse.issparse(adata.X):
        x_data = adata.X.data
        if len(x_data) == 0:
            return {"error": "Empty data matrix"}
        max_val = float(x_data.max())
        mean_val = float(x_data.mean())
        median_val = float(np.median(x_data))
        nonzero_fraction = len(x_data) / (adata.n_obs * adata.n_vars)
    else:
        max_val = float(adata.X.max())
        mean_val = float(adata.X.mean())
        median_val = float(np.median(adata.X[adata.X > 0]))
        nonzero_fraction = (adata.X > 0).sum() / adata.X.size
    
    # Heuristics
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
        "n_cells": adata.n_obs,
        "n_genes": n_features,
        "max_value": max_val,
        "mean_value": mean_val,
        "median_nonzero": median_val,
        "nonzero_fraction": nonzero_fraction,
        "is_counts": is_counts,
        "is_normalized": is_normalized,
        "is_log_transformed": is_log,
        "is_preprocessed": is_preprocessed,
        "is_likely_reduced": is_likely_reduced,
        "has_pca": has_pca,
        "has_latent": has_latent,
        "has_raw_layer": has_raw,
        "recommendation": recommendation,
    }


def get_time_candidates(adata: ad.AnnData, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Get top candidates for time_key sorted by suitability score.
    
    Args:
        adata: AnnData object
        top_n: Number of top candidates to return
        
    Returns:
        List of candidate analyses sorted by time_key_score
    """
    all_columns = list_obs_columns(adata)
    candidates = []
    
    for col in all_columns:
        try:
            analysis = analyze_obs_column(adata, col)
            if analysis.get("time_key_score", 0) > 0:
                candidates.append(analysis)
        except Exception:
            pass
    
    # Sort by score
    candidates.sort(key=lambda x: x.get("time_key_score", 0), reverse=True)
    
    return candidates[:top_n]


def get_label_candidates(adata: ad.AnnData, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Get top candidates for label_key sorted by suitability score.
    
    Excludes columns that look like time columns.
    
    Args:
        adata: AnnData object
        top_n: Number of top candidates to return
        
    Returns:
        List of candidate analyses sorted by label_key_score
    """
    all_columns = list_obs_columns(adata)
    
    # Keywords that indicate time columns (should be excluded from label candidates)
    time_keywords = ['time', 'day', 'stage', 'tp', 'hour', 'age', 'embryo', 'week', 'month']
    
    candidates = []
    
    for col in all_columns:
        # Skip columns that look like time columns
        if any(kw in col.lower() for kw in time_keywords):
            continue
            
        try:
            analysis = analyze_obs_column(adata, col)
            if analysis.get("label_key_score", 0) > 0:
                candidates.append(analysis)
        except Exception:
            pass
    
    # Sort by score
    candidates.sort(key=lambda x: x.get("label_key_score", 0), reverse=True)
    
    return candidates[:top_n]

