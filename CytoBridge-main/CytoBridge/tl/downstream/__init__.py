"""Structured downstream analysis APIs."""

from .velocity import (
    compute_velocity_bundle,
    build_velocity_graph_bundle,
    summarize_velocity_drivers_bundle,
    summarize_velocity_jacobian_drivers_bundle,
)
from .trajectory import generate_ode_trajectory_bundle, generate_sde_trajectory_bundle
from .growth import (
    summarize_growth_bundle,
    summarize_growth_drivers_bundle,
    summarize_growth_jacobian_drivers_bundle,
)
from .grn import estimate_velocity_jacobian_bundle, analyze_grn_bundle
from .contracts import (
    TIME_KEY,
    LATENT_KEY,
    VELOCITY_LATENT_KEY,
    GROWTH_KEY,
    PCA_LOADINGS_KEY,
    ensure_contract,
)

__all__ = [
    "compute_velocity_bundle",
    "build_velocity_graph_bundle",
    "summarize_velocity_drivers_bundle",
    "summarize_velocity_jacobian_drivers_bundle",
    "generate_ode_trajectory_bundle",
    "generate_sde_trajectory_bundle",
    "summarize_growth_bundle",
    "summarize_growth_drivers_bundle",
    "summarize_growth_jacobian_drivers_bundle",
    "estimate_velocity_jacobian_bundle",
    "analyze_grn_bundle",
    "TIME_KEY",
    "LATENT_KEY",
    "VELOCITY_LATENT_KEY",
    "GROWTH_KEY",
    "PCA_LOADINGS_KEY",
    "ensure_contract",
]
