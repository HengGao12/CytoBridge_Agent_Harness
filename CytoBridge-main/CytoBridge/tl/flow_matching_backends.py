from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import ot as pot
import torch

from CytoBridge.tl.flow_matching import (
    ConditionalFlowMatcher,
    WFRTravelingGaussianPath,
    calculate_auto_regularization,
    sample_from_ot_plan,
)
from CytoBridge.tl.training_algorithm import FlowMatchingBuildContext


_WFR_EPS = 1e-8
_WFR_ANGLE_MARGIN = 1e-4


def _notify_path_interval_context(
    path: ConditionalFlowMatcher,
    *,
    t0: torch.Tensor,
    t1: torch.Tensor,
    time_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """
    Give custom conditional paths the current adjacent-time interval.

    The backend still passes local times `s in [0, 1]` into path methods, but
    learned/shared paths may need global interval context to distinguish
    biological gaps. Custom paths can implement either:

    - set_interval_context(t0=..., t1=..., delta_t=..., time_idx=..., ...)
    - set_interval(delta_t)
    """

    t0_t = torch.as_tensor(t0, device=device, dtype=dtype)
    t1_t = torch.as_tensor(t1, device=device, dtype=dtype)
    delta_t = t1_t - t0_t

    set_context = getattr(path, "set_interval_context", None)
    if callable(set_context):
        context_kwargs = {
            "t0": t0_t,
            "t1": t1_t,
            "delta_t": delta_t,
            "time_idx": int(time_idx),
            "device": device,
            "dtype": dtype,
        }
        signature = inspect.signature(set_context)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        if accepts_kwargs:
            set_context(**context_kwargs)
        else:
            set_context(
                **{
                    key: value
                    for key, value in context_kwargs.items()
                    if key in signature.parameters
                }
            )
        return

    set_interval = getattr(path, "set_interval", None)
    if callable(set_interval):
        try:
            set_interval(delta_t)
        except TypeError:
            set_interval(float(delta_t.detach().cpu().item()))


def _as_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class CouplingPlanStore:
    """
    Unified storage for one adjacent-time coupling.

    Dense and chunked couplings expose the same high-level operations:
    sampling pairs and retrieving per-sample terminal mass. Callers should not
    reason about dummy plans or sampling-info side channels.
    """

    source_n: int
    target_n: int
    dense_plan: Optional[np.ndarray] = None
    sub_plans: list[np.ndarray] = field(default_factory=list)
    source_groups: list[np.ndarray] = field(default_factory=list)
    target_groups: list[np.ndarray] = field(default_factory=list)
    terminal_mass_rows: Optional[list[np.ndarray]] = None
    terminal_mass_subplans: Optional[list[np.ndarray]] = None
    edge_src: Optional[np.ndarray] = None
    edge_tgt: Optional[np.ndarray] = None
    edge_weight: Optional[np.ndarray] = None
    edge_cdf: Optional[np.ndarray] = None
    edge_terminal_mass: Optional[np.ndarray] = None
    edge_row_mass: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(
        cls,
        plan: np.ndarray,
        sampling_info: Optional[dict[str, Any]] = None,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "CouplingPlanStore":
        plan_arr = np.asarray(plan, dtype=np.float32)
        if sampling_info is None:
            return cls(
                source_n=int(plan_arr.shape[0]),
                target_n=int(plan_arr.shape[1]),
                dense_plan=plan_arr,
                metadata=dict(metadata or {}),
            )
        sampling_info = _coerce_aux_data_to_cpu(sampling_info)
        sub_plans = [np.asarray(p, dtype=np.float32) for p in sampling_info.get("sub_plans", [])]
        source_groups = [np.asarray(g, dtype=np.int64) for g in sampling_info.get("source_groups", [])]
        target_groups = [np.asarray(g, dtype=np.int64) for g in sampling_info.get("target_groups", [])]
        source_n = int(plan_arr.shape[0])
        target_n = int(plan_arr.shape[1])
        dense_plan = None if plan_arr.shape in {(0, 0), (1, 1)} else plan_arr
        if dense_plan is None:
            source_n = max((int(g.max()) + 1 for g in source_groups if g.size), default=source_n)
            target_n = max((int(g.max()) + 1 for g in target_groups if g.size), default=target_n)
        return cls(
            source_n=source_n,
            target_n=target_n,
            dense_plan=dense_plan,
            sub_plans=sub_plans,
            source_groups=source_groups,
            target_groups=target_groups,
            terminal_mass_rows=(
                [np.asarray(v, dtype=np.float32) for v in sampling_info.get("terminal_mass_rows", [])]
                if sampling_info.get("terminal_mass_rows") is not None
                else None
            ),
            terminal_mass_subplans=(
                [np.asarray(v, dtype=np.float32) for v in sampling_info.get("terminal_mass_subplans", [])]
                if sampling_info.get("terminal_mass_subplans") is not None
                else None
            ),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_chunked(
        cls,
        *,
        source_n: int,
        target_n: int,
        sub_plans: list[np.ndarray],
        source_groups: list[np.ndarray],
        target_groups: list[np.ndarray],
        dense_plan: Optional[np.ndarray] = None,
        terminal_mass_rows: Optional[list[np.ndarray]] = None,
        terminal_mass_subplans: Optional[list[np.ndarray]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "CouplingPlanStore":
        return cls(
            source_n=int(source_n),
            target_n=int(target_n),
            dense_plan=None if dense_plan is None else np.asarray(dense_plan, dtype=np.float32),
            sub_plans=[np.asarray(p, dtype=np.float32) for p in sub_plans],
            source_groups=[np.asarray(g, dtype=np.int64) for g in source_groups],
            target_groups=[np.asarray(g, dtype=np.int64) for g in target_groups],
            terminal_mass_rows=(
                [np.asarray(v, dtype=np.float32) for v in terminal_mass_rows]
                if terminal_mass_rows is not None
                else None
            ),
            terminal_mass_subplans=(
                [np.asarray(v, dtype=np.float32) for v in terminal_mass_subplans]
                if terminal_mass_subplans is not None
                else None
            ),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_edges(
        cls,
        *,
        source_n: int,
        target_n: int,
        edge_src: np.ndarray | torch.Tensor | list[int],
        edge_tgt: np.ndarray | torch.Tensor | list[int],
        edge_weight: np.ndarray | torch.Tensor | list[float],
        terminal_mass_rows: Optional[np.ndarray | torch.Tensor | list[float]] = None,
        terminal_mass_edges: Optional[np.ndarray | torch.Tensor | list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "CouplingPlanStore":
        """
        Build a sparse edge-list coupling store.

        This is the canonical representation for sparse-support transport:
        proposal-defined source/target index pairs plus non-negative edge
        weights. The store owns pair sampling and terminal source-mass lookup
        so custom algorithms do not need private sampling_info side channels.
        """
        src = _as_numpy_array(edge_src).astype(np.int64, copy=False).reshape(-1)
        tgt = _as_numpy_array(edge_tgt).astype(np.int64, copy=False).reshape(-1)
        weight = _as_numpy_array(edge_weight).astype(np.float32, copy=False).reshape(-1)
        source_n = int(source_n)
        target_n = int(target_n)
        if source_n <= 0 or target_n <= 0:
            raise ValueError(f"source_n and target_n must be positive, got {source_n}, {target_n}")
        if not (src.shape[0] == tgt.shape[0] == weight.shape[0]):
            raise ValueError(
                "edge_src, edge_tgt, and edge_weight must have the same length, "
                f"got {src.shape[0]}, {tgt.shape[0]}, {weight.shape[0]}"
            )
        if src.shape[0] == 0:
            raise ValueError("edge-list coupling requires at least one edge")
        if src.min(initial=0) < 0 or src.max(initial=-1) >= source_n:
            raise ValueError("edge_src contains indices outside [0, source_n)")
        if tgt.min(initial=0) < 0 or tgt.max(initial=-1) >= target_n:
            raise ValueError("edge_tgt contains indices outside [0, target_n)")
        if not np.isfinite(weight).all():
            raise ValueError("edge_weight contains NaN/inf")
        if (weight < 0).any():
            raise ValueError("edge_weight contains negative entries")
        total = float(weight.sum())
        if total <= 0:
            raise ValueError("edge_weight must have positive total mass")

        if terminal_mass_rows is None:
            row_mass = np.bincount(src, weights=weight.astype(np.float64), minlength=source_n).astype(
                np.float32,
                copy=False,
            )
        else:
            row_mass = _as_numpy_array(terminal_mass_rows).astype(np.float32, copy=False).reshape(-1)
            if row_mass.shape[0] != source_n:
                raise ValueError(
                    f"terminal_mass_rows must have length source_n={source_n}, got {row_mass.shape[0]}"
                )
            if not np.isfinite(row_mass).all() or (row_mass < 0).any():
                raise ValueError("terminal_mass_rows must be finite and non-negative")

        edge_terminal = None
        if terminal_mass_edges is not None:
            edge_terminal = _as_numpy_array(terminal_mass_edges).astype(np.float32, copy=False).reshape(-1)
            if edge_terminal.shape[0] != src.shape[0]:
                raise ValueError(
                    f"terminal_mass_edges must have length n_edges={src.shape[0]}, got {edge_terminal.shape[0]}"
                )
            if not np.isfinite(edge_terminal).all() or (edge_terminal < 0).any():
                raise ValueError("terminal_mass_edges must be finite and non-negative")

        cdf = np.cumsum(weight.astype(np.float64, copy=False))
        cdf /= float(cdf[-1])
        cdf[-1] = 1.0
        meta = dict(metadata or {})
        meta.setdefault("storage", "sparse_edges")
        meta.setdefault("n_edges", int(src.shape[0]))
        meta.setdefault("edge_weight_sum", float(total))
        return cls(
            source_n=source_n,
            target_n=target_n,
            edge_src=src,
            edge_tgt=tgt,
            edge_weight=weight,
            edge_cdf=cdf.astype(np.float64, copy=False),
            edge_terminal_mass=edge_terminal,
            edge_row_mass=row_mass,
            metadata=meta,
        )

    @property
    def is_dense(self) -> bool:
        return self.dense_plan is not None

    @property
    def is_chunked(self) -> bool:
        return bool(self.sub_plans)

    @property
    def is_sparse_edges(self) -> bool:
        return self.edge_src is not None

    @property
    def has_terminal_mass_payload(self) -> bool:
        return (
            self.terminal_mass_rows is not None
            or self.terminal_mass_subplans is not None
            or self.edge_terminal_mass is not None
            or self.edge_row_mass is not None
        )

    def to_legacy_plan(self) -> np.ndarray:
        if self.dense_plan is not None:
            return self.dense_plan
        return np.empty((0, 0), dtype=np.float32)

    def to_sampling_info(self) -> Optional[dict[str, Any]]:
        if not self.sub_plans:
            return None
        sampling_info: dict[str, Any] = {
            "sub_plans": self.sub_plans,
            "source_groups": self.source_groups,
            "target_groups": self.target_groups,
        }
        if self.terminal_mass_subplans is not None:
            sampling_info["terminal_mass_subplans"] = self.terminal_mass_subplans
        elif self.terminal_mass_rows is not None:
            sampling_info["terminal_mass_rows"] = self.terminal_mass_rows
        return sampling_info

    def sample_pairs(
        self,
        *,
        X: list[np.ndarray],
        time_idx: int,
        batch_size: int,
        device: torch.device,
        epsilon: float = 1e-9,
    ) -> Optional["PairBatch"]:
        if self.edge_src is not None:
            if self.edge_cdf is None or self.edge_tgt is None or self.edge_weight is None:
                raise RuntimeError("Sparse edge CouplingPlanStore is missing edge sampling arrays.")
            draws = np.random.random(size=int(batch_size))
            edge_idx = np.searchsorted(self.edge_cdf, draws, side="left")
            edge_idx = np.clip(edge_idx, 0, self.edge_src.shape[0] - 1)
            sampled_i = self.edge_src[edge_idx].astype(np.int64, copy=False)
            sampled_j = self.edge_tgt[edge_idx].astype(np.int64, copy=False)
            sampled_x0 = X[time_idx][sampled_i]
            sampled_x1 = X[time_idx + 1][sampled_j]
            sampled_terminal_mass = None
            if self.edge_terminal_mass is not None:
                sampled_terminal_mass = self.edge_terminal_mass[edge_idx].astype(np.float32, copy=False)
            elif self.edge_row_mass is not None:
                sampled_terminal_mass = self.edge_row_mass[sampled_i].astype(np.float32, copy=False)
            return PairBatch(
                x0=torch.from_numpy(sampled_x0).float().to(device),
                x1=torch.from_numpy(sampled_x1).float().to(device),
                idx0=sampled_i,
                idx1=sampled_j,
                time_idx=time_idx,
                sampled_terminal_mass=(
                    None if sampled_terminal_mass is None else sampled_terminal_mass.reshape(-1, 1)
                ),
            )

        sampling_info = self.to_sampling_info()
        if sampling_info is not None:
            return _sample_pair_batch_from_chunked_sampling_info(
                sampling_info=sampling_info,
                X=X,
                time_idx=time_idx,
                batch_size=batch_size,
                device=device,
                epsilon=epsilon,
            )
        if self.dense_plan is None:
            raise RuntimeError(f"CouplingPlanStore for time_idx={time_idx} has neither dense nor chunked state.")
        sampled_x0, sampled_x1, sampled_idx0, sampled_idx1 = sample_from_ot_plan(
            ot_plan=self.dense_plan,
            x0=X[time_idx],
            x1=X[time_idx + 1],
            batch_size=batch_size,
            sampling_info=None,
            device=device,
        )
        if sampled_x0.size == 0:
            return None
        return PairBatch(
            x0=torch.from_numpy(sampled_x0).float().to(device),
            x1=torch.from_numpy(sampled_x1).float().to(device),
            idx0=sampled_idx0,
            idx1=sampled_idx1,
            time_idx=time_idx,
        )

    def terminal_mass_for_batch(self, pair_batch: "PairBatch") -> np.ndarray:
        if pair_batch.sampled_terminal_mass is not None:
            return np.asarray(pair_batch.sampled_terminal_mass, dtype=np.float32).reshape(-1, 1)
        if self.edge_row_mass is not None:
            return self.edge_row_mass[pair_batch.idx0].reshape(-1, 1).astype(np.float32, copy=False)
        if self.dense_plan is None:
            raise RuntimeError(
                "Non-dense coupling did not attach sampled_terminal_mass or edge_row_mass. "
                "Use CouplingPlanStore.sample_pairs(...) for chunked/sparse stores."
            )
        selected_plan = self.dense_plan[pair_batch.idx0]
        return selected_plan.sum(axis=-1, keepdims=True).astype(np.float32, copy=False)


@dataclass(init=False)
class CouplingState:
    plan_stores: list[CouplingPlanStore]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        plan_stores: Optional[list[CouplingPlanStore]] = None,
        plans: Optional[list[np.ndarray]] = None,
        sampling_info_plans: Optional[list[Optional[dict[str, Any]]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if plan_stores is None:
            if plans is None:
                raise ValueError("CouplingState requires plan_stores or legacy plans.")
            sampling_info_plans = sampling_info_plans or [None] * len(plans)
            if len(plans) != len(sampling_info_plans):
                raise ValueError(
                    f"plans and sampling_info_plans length mismatch: {len(plans)} vs {len(sampling_info_plans)}"
                )
            plan_stores = [
                CouplingPlanStore.from_legacy(plan, sampling_info)
                for plan, sampling_info in zip(plans, sampling_info_plans)
            ]
        self.plan_stores = list(plan_stores)
        self.metadata = dict(metadata or {})

    def plan_store(self, time_idx: int) -> CouplingPlanStore:
        return self.plan_stores[time_idx]

    @property
    def plans(self) -> list[np.ndarray]:
        """Legacy read-only view. New code should use plan_stores."""
        return [store.to_legacy_plan() for store in self.plan_stores]

    @property
    def sampling_info_plans(self) -> list[Optional[dict[str, Any]]]:
        """Legacy read-only view. New code should use plan_stores."""
        return [store.to_sampling_info() for store in self.plan_stores]


@dataclass
class PairBatch:
    x0: torch.Tensor
    x1: torch.Tensor
    idx0: np.ndarray
    idx1: np.ndarray
    time_idx: int
    sampled_terminal_mass: Optional[np.ndarray] = None


@dataclass
class CouplingResult:
    """Pairwise coupling output for one adjacent time interval."""

    plan: np.ndarray
    sampling_info: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PairwiseCost:
    """Pairwise transport cost specification for one adjacent time interval."""

    cost_matrix: np.ndarray | torch.Tensor
    source_mass: Optional[np.ndarray] = None
    target_mass: Optional[np.ndarray] = None
    normalize_cost: Optional[bool] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransportSolverConfig:
    """
    Solver settings for converting a pairwise cost into a plan store.

    Custom algorithms should prefer this configuration object over hand-written
    OT/UOT solver calls when they only need to change the cost/support. It keeps
    dense, chunked, and sparse-support implementations on the same sampling
    contract.
    """

    solver_mode: str = "uot"
    use_mini_batch: bool = True
    chunk_size: int = 1000
    alpha_regm: float = 1.0
    reg: float | None = None
    reg_m: float | None = None
    normalize_cost: bool = False
    balanced_method: str = "exact"
    balanced_reg: float = 0.05
    wfr_epsilon: float = _WFR_EPS
    warn: bool = True


class TransportPlanBuilder:
    """
    Public helper for the common custom-coupling path:

        cost/support -> OT/UOT/WFR-style solver -> CouplingPlanStore

    Use `solve_cost_to_store(...)` when the algorithm defines a pairwise cost
    matrix. Use `edges_to_store(...)` when the algorithm defines sparse
    source/target edges and edge weights directly. Both outputs expose the same
    `sample_pairs(...)` and `terminal_mass_for_batch(...)` interface.
    """

    def __init__(self, solver_config: Optional[TransportSolverConfig] = None) -> None:
        self.solver_config = solver_config or TransportSolverConfig()

    def solve_cost_to_store(
        self,
        pairwise_cost: PairwiseCost | np.ndarray | torch.Tensor,
        *,
        time_idx: int = 0,
        device: torch.device | None = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CouplingPlanStore:
        """Solve a pairwise cost and wrap the result as a `CouplingPlanStore`."""
        cfg = self.solver_config
        cost = _coerce_pairwise_cost(pairwise_cost, time_idx=time_idx)
        solver_mode = str(cfg.solver_mode).lower()
        if solver_mode in {"uot", "unbalanced", "unbalanced_ot"}:
            result = _solve_uot_from_pairwise_cost(
                cost,
                use_mini_batch=cfg.use_mini_batch,
                chunk_size=cfg.chunk_size,
                alpha_regm=cfg.alpha_regm,
                reg=cfg.reg,
                reg_m=cfg.reg_m,
                normalize_cost=cfg.normalize_cost,
                device=device,
            )
        elif solver_mode in {"balanced", "balanced_ot", "ot"}:
            result = _solve_balanced_ot_from_pairwise_cost(
                cost,
                use_mini_batch=cfg.use_mini_batch,
                chunk_size=cfg.chunk_size,
                method=cfg.balanced_method,
                reg=float(cfg.reg if cfg.reg is not None else cfg.balanced_reg),
                normalize_cost=cfg.normalize_cost,
                warn=cfg.warn,
            )
        elif solver_mode in {"wfr", "wfr_oet", "wfr-oet"}:
            result = _solve_wfr_oet_from_pairwise_cost(
                cost,
                use_mini_batch=cfg.use_mini_batch,
                chunk_size=cfg.chunk_size,
                epsilon=cfg.wfr_epsilon,
                device=device,
            )
        else:
            raise ValueError(
                "Unsupported solver_mode for TransportPlanBuilder: "
                f"{cfg.solver_mode!r}. Expected 'uot', 'balanced', or 'wfr'."
            )
        merged_metadata = dict(result.metadata)
        merged_metadata.update(metadata or {})
        if result.sampling_info is None and "terminal_mass_matrix" in result.metadata:
            plan_arr = np.asarray(result.plan, dtype=np.float32)
            return CouplingPlanStore.from_chunked(
                source_n=plan_arr.shape[0],
                target_n=plan_arr.shape[1],
                sub_plans=[plan_arr],
                source_groups=[np.arange(plan_arr.shape[0], dtype=np.int64)],
                target_groups=[np.arange(plan_arr.shape[1], dtype=np.int64)],
                terminal_mass_subplans=[
                    np.asarray(result.metadata["terminal_mass_matrix"], dtype=np.float32)
                ],
                metadata=merged_metadata,
            )
        return CouplingPlanStore.from_legacy(
            result.plan,
            result.sampling_info,
            metadata=merged_metadata,
        )

    @staticmethod
    def edges_to_store(
        *,
        source_n: int,
        target_n: int,
        edge_src: np.ndarray | torch.Tensor | list[int],
        edge_tgt: np.ndarray | torch.Tensor | list[int],
        edge_weight: np.ndarray | torch.Tensor | list[float],
        terminal_mass_rows: Optional[np.ndarray | torch.Tensor | list[float]] = None,
        terminal_mass_edges: Optional[np.ndarray | torch.Tensor | list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CouplingPlanStore:
        """Create a sparse-support store from edge-list transport mass."""
        return CouplingPlanStore.from_edges(
            source_n=source_n,
            target_n=target_n,
            edge_src=edge_src,
            edge_tgt=edge_tgt,
            edge_weight=edge_weight,
            terminal_mass_rows=terminal_mass_rows,
            terminal_mass_edges=terminal_mass_edges,
            metadata=metadata,
        )


@dataclass
class FlowMatchingBatch:
    """
    One sampled flow-matching batch.

    Notes:
    - `gt` is the growth-rate regression target.
    - `loss_weights` stores per-sample training loss weights.
      It is not the current path mass `w(t)`.
    """
    t_local: torch.Tensor
    t_global: torch.Tensor
    xt: torch.Tensor
    ut: torch.Tensor
    gt: torch.Tensor
    loss_weights: torch.Tensor
    eps: torch.Tensor
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class MassTargets:
    """
    Terminal mass targets for one local interval.

    `terminal_mass` answers the question:
    if a sampled source particle has unit mass at local time `t=0`, what mass
    should it have at local time `t=1`?
    """

    terminal_mass: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class CouplingStrategy(ABC):
    """Developer hook for transport/coupling computation and pair sampling."""

    @abstractmethod
    def build_state(
        self,
        X: list[np.ndarray],
        t_train: torch.Tensor,
        device: torch.device,
    ) -> CouplingState:
        raise NotImplementedError

    @abstractmethod
    def sample_pairs(
        self,
        state: CouplingState,
        X: list[np.ndarray],
        time_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[PairBatch]:
        raise NotImplementedError


def _sample_pair_batch_from_state(
    *,
    state: CouplingState,
    X: list[np.ndarray],
    time_idx: int,
    batch_size: int,
    device: torch.device,
) -> Optional[PairBatch]:
    return state.plan_store(time_idx).sample_pairs(
        X=X,
        time_idx=time_idx,
        batch_size=batch_size,
        device=device,
    )


def _sample_pair_batch_from_chunked_sampling_info(
    *,
    sampling_info: dict[str, Any],
    X: list[np.ndarray],
    time_idx: int,
    batch_size: int,
    device: torch.device,
    epsilon: float = 1e-9,
) -> Optional[PairBatch]:
    """Sample pairs from cached chunk plans without requiring a full dense plan."""

    sub_plans = [np.asarray(plan, dtype=np.float32) for plan in sampling_info.get("sub_plans", [])]
    source_groups = [np.asarray(idx, dtype=np.int64) for idx in sampling_info.get("source_groups", [])]
    target_groups = [np.asarray(idx, dtype=np.int64) for idx in sampling_info.get("target_groups", [])]
    terminal_mass_rows = sampling_info.get("terminal_mass_rows")
    terminal_mass_subplans = sampling_info.get("terminal_mass_subplans")

    if not sub_plans:
        return None
    if len(sub_plans) != len(source_groups) or len(sub_plans) != len(target_groups):
        raise ValueError(
            f"Chunked sampling info for time_idx={time_idx} has inconsistent "
            f"sub_plans/source_groups/target_groups lengths."
        )

    chunk_masses = np.asarray([float(plan.sum()) for plan in sub_plans], dtype=np.float64)
    total_mass = float(chunk_masses.sum())
    if total_mass < epsilon:
        return None
    chunk_probs = torch.tensor(chunk_masses / total_mass, dtype=torch.float32, device=device)
    sampled_chunk_indices = torch.multinomial(chunk_probs, num_samples=batch_size, replacement=True)
    sampled_i_t = torch.empty(batch_size, dtype=torch.long, device=device)
    sampled_j_t = torch.empty(batch_size, dtype=torch.long, device=device)

    sub_plans_t = [torch.from_numpy(plan).float().to(device) for plan in sub_plans]
    source_groups_t = [torch.from_numpy(group).long().to(device) for group in source_groups]
    target_groups_t = [torch.from_numpy(group).long().to(device) for group in target_groups]
    for chunk_idx, count in zip(*torch.unique(sampled_chunk_indices, return_counts=True)):
        chunk_plan = sub_plans_t[int(chunk_idx)]
        row_sums = chunk_plan.sum(dim=1)
        chunk_total = row_sums.sum()
        if float(chunk_total.detach().cpu()) < epsilon:
            continue
        local_i = torch.multinomial(row_sums / chunk_total, num_samples=int(count.item()), replacement=True)
        selected_rows = chunk_plan[local_i]
        selected_row_sums = row_sums[local_i]
        local_j = torch.multinomial(selected_rows / (selected_row_sums.unsqueeze(1) + epsilon), num_samples=1).squeeze(1)
        mask = sampled_chunk_indices == chunk_idx
        sampled_i_t[mask] = source_groups_t[int(chunk_idx)][local_i]
        sampled_j_t[mask] = target_groups_t[int(chunk_idx)][local_j]

    sampled_i = sampled_i_t.detach().cpu().numpy()
    sampled_j = sampled_j_t.detach().cpu().numpy()
    sampled_x0 = X[time_idx][sampled_i]
    sampled_x1 = X[time_idx + 1][sampled_j]

    sampled_terminal_mass: np.ndarray | None = None
    if terminal_mass_rows is not None or terminal_mass_subplans is not None:
        sampled_terminal_mass = np.empty(sampled_i.shape[0], dtype=np.float32)
        source_lookup: dict[int, tuple[int, int]] = {}
        target_lookup: dict[tuple[int, int], int] = {}
        for chunk_id, (src_group, tgt_group) in enumerate(zip(source_groups, target_groups)):
            for local_i, global_i in enumerate(src_group):
                source_lookup[int(global_i)] = (chunk_id, local_i)
            for local_j, global_j in enumerate(tgt_group):
                target_lookup[(chunk_id, int(global_j))] = local_j
        for row, (global_i, global_j) in enumerate(zip(sampled_i, sampled_j)):
            chunk_id, local_i = source_lookup[int(global_i)]
            local_j = target_lookup[(chunk_id, int(global_j))]
            if terminal_mass_subplans is not None:
                terminal_chunk = np.asarray(terminal_mass_subplans[chunk_id], dtype=np.float32)
                sampled_terminal_mass[row] = terminal_chunk[local_i, local_j]
            else:
                terminal_rows = np.asarray(terminal_mass_rows[chunk_id], dtype=np.float32)
                sampled_terminal_mass[row] = terminal_rows[local_i]

    return PairBatch(
        x0=torch.from_numpy(sampled_x0).float().to(device),
        x1=torch.from_numpy(sampled_x1).float().to(device),
        idx0=sampled_i,
        idx1=sampled_j,
        time_idx=time_idx,
        sampled_terminal_mass=sampled_terminal_mass,
    )


class PairwiseOTCouplingStrategy(CouplingStrategy):
    """
    Preferred high-level developer API for custom pairwise OT/UOT coupling.

    In the common case, custom algorithms should override `compute_ot_coupling(...)`
    for one adjacent time pair and reuse the default state assembly/sampling logic.
    Override `build_state(...)` only when the coupling requires global cross-time
    coordination beyond independent adjacent-pair construction.
    """

    @abstractmethod
    def compute_ot_coupling(
        self,
        x0: np.ndarray,
        x1: np.ndarray,
        *,
        time_idx: int,
        t0: float,
        t1: float,
        device: torch.device,
    ) -> CouplingResult:
        raise NotImplementedError

    def describe_metadata(self) -> dict[str, Any]:
        return {}

    def _coerce_coupling_result(
        self,
        result: CouplingResult | np.ndarray | torch.Tensor,
        *,
        time_idx: int,
    ) -> CouplingResult:
        if isinstance(result, CouplingResult):
            coerced = result
        elif isinstance(result, np.ndarray):
            coerced = CouplingResult(plan=result)
        elif isinstance(result, torch.Tensor):
            coerced = CouplingResult(plan=result)
        else:
            raise TypeError(
                f"compute_ot_coupling(...) must return CouplingResult, np.ndarray, or torch.Tensor, "
                f"got {type(result).__name__} at time_idx={time_idx}"
            )

        plan = _coerce_plan_array(coerced.plan)
        if plan.ndim != 2:
            raise ValueError(
                f"Coupling plan for time_idx={time_idx} must be 2D, got shape={plan.shape}"
            )
        return CouplingResult(
            plan=plan,
            sampling_info=_coerce_aux_data_to_cpu(coerced.sampling_info),
            metadata=_coerce_metadata_dict(coerced.metadata),
        )

    def build_state(
        self,
        X: list[np.ndarray],
        t_train: torch.Tensor,
        device: torch.device,
    ) -> CouplingState:
        if len(X) != len(t_train):
            raise ValueError(f"len(X) must equal len(t_train), got {len(X)} and {len(t_train)}")

        plan_stores: list[CouplingPlanStore] = []
        pair_metadata: list[dict[str, Any]] = []

        for time_idx in range(len(X) - 1):
            result = self._coerce_coupling_result(
                self.compute_ot_coupling(
                    X[time_idx],
                    X[time_idx + 1],
                    time_idx=time_idx,
                    t0=float(torch.as_tensor(t_train[time_idx]).item()),
                    t1=float(torch.as_tensor(t_train[time_idx + 1]).item()),
                    device=device,
                ),
                time_idx=time_idx,
            )
            expected_shape = (X[time_idx].shape[0], X[time_idx + 1].shape[0])
            if result.plan.shape != expected_shape:
                raise ValueError(
                    f"Coupling plan for time_idx={time_idx} has shape={result.plan.shape}, "
                    f"expected={expected_shape}"
                )
            plan_stores.append(
                CouplingPlanStore.from_legacy(
                    result.plan,
                    result.sampling_info,
                    metadata=result.metadata,
                )
            )
            pair_metadata.append(result.metadata)

        metadata = dict(self.describe_metadata())
        metadata["pair_metadata"] = pair_metadata
        return CouplingState(plan_stores=plan_stores, metadata=metadata)

    def sample_pairs(
        self,
        state: CouplingState,
        X: list[np.ndarray],
        time_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[PairBatch]:
        return _sample_pair_batch_from_state(
            state=state,
            X=X,
            time_idx=time_idx,
            batch_size=batch_size,
            device=device,
        )


class CostBasedPairwiseOTCouplingStrategy(PairwiseOTCouplingStrategy):
    """
    Cost-matrix custom API for algorithms that only change pairwise transport cost.

    Developers override `build_pairwise_cost(...)`; the package handles OT/UOT
    solving and sampling-info assembly. This API still requires a full
    adjacent-pair `PairwiseCost.cost_matrix`. `use_mini_batch=True` chunks
    solver/sampling work, but it is not a true streaming memory contract.
    Large-data methods that cannot materialize full pairwise state should
    implement a custom `CouplingStrategy.build_state(...)` / `sample_pairs(...)`
    path instead.
    """

    def __init__(
        self,
        *,
        solver_mode: str = "uot",
        chunk_size: int = 1000,
        use_mini_batch: bool = True,
        alpha_regm: float = 1.0,
        reg_strategy: str = "per_time",
        reg: float | None = None,
        reg_m: float | None = None,
        auto_reg_device: str = "cpu",
        balanced_method: str = "exact",
        balanced_reg: float = 0.05,
        normalize_cost: bool = False,
        warn: bool = True,
    ) -> None:
        if solver_mode not in {"uot", "balanced"}:
            raise ValueError(f"solver_mode must be 'uot' or 'balanced', got {solver_mode}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        self.solver_mode = str(solver_mode)
        self.chunk_size = int(chunk_size)
        self.use_mini_batch = bool(use_mini_batch)
        self.alpha_regm = float(alpha_regm)
        self.reg_strategy = str(reg_strategy)
        self.fixed_reg = None if reg is None else float(reg)
        self.fixed_reg_m = None if reg_m is None else float(reg_m)
        self.auto_reg_device = str(auto_reg_device or "cpu").lower()
        self.balanced_method = str(balanced_method)
        self.balanced_reg = float(balanced_reg)
        self.normalize_cost = bool(normalize_cost)
        self.warn = bool(warn)
        self.supports_minibatch = True
        self.use_mini_batch_uot = self.solver_mode == "uot" and self.use_mini_batch
        self.use_mini_batch_balanced = self.solver_mode == "balanced" and self.use_mini_batch

    @abstractmethod
    def build_pairwise_cost(
        self,
        x0: np.ndarray,
        x1: np.ndarray,
        *,
        time_idx: int,
        t0: float,
        t1: float,
        device: torch.device,
    ) -> PairwiseCost:
        raise NotImplementedError

    def describe_metadata(self) -> dict[str, Any]:
        return {
            "kind": "cost_based_pairwise_ot",
            "solver_mode": self.solver_mode,
            "chunk_size": self.chunk_size,
            "use_mini_batch": self.use_mini_batch,
            "alpha_regm": self.alpha_regm,
            "reg_strategy": self.reg_strategy,
            "fixed_reg": self.fixed_reg,
            "fixed_reg_m": self.fixed_reg_m,
            "auto_reg_device": self.auto_reg_device,
            "balanced_method": self.balanced_method,
            "balanced_reg": self.balanced_reg,
            "normalize_cost": self.normalize_cost,
        }

    def _solve_cost(
        self,
        pairwise_cost: PairwiseCost,
        *,
        reg: float | None = None,
        reg_m: float | None = None,
        device: torch.device | None = None,
    ) -> CouplingResult:
        if self.solver_mode == "uot":
            effective_reg = self.fixed_reg if reg is None else reg
            effective_reg_m = self.fixed_reg_m if reg_m is None else reg_m
            return _solve_uot_from_pairwise_cost(
                pairwise_cost,
                use_mini_batch=self.use_mini_batch,
                chunk_size=self.chunk_size,
                alpha_regm=self.alpha_regm,
                reg=effective_reg,
                reg_m=effective_reg_m,
                normalize_cost=self.normalize_cost,
                device=device,
            )
        return _solve_balanced_ot_from_pairwise_cost(
            pairwise_cost,
            use_mini_batch=self.use_mini_batch,
            chunk_size=self.chunk_size,
            method=self.balanced_method,
            reg=self.balanced_reg,
            normalize_cost=self.normalize_cost,
            warn=self.warn,
        )

    def _auto_regularization_device(self, runtime_device: torch.device) -> torch.device:
        if self.auto_reg_device in {"cuda", "gpu"} and torch.cuda.is_available():
            return torch.device("cuda")
        if self.auto_reg_device in {"runtime", "device"} and torch.device(runtime_device).type == "cuda":
            return runtime_device
        return torch.device("cpu")

    def compute_ot_coupling(
        self,
        x0: np.ndarray,
        x1: np.ndarray,
        *,
        time_idx: int,
        t0: float,
        t1: float,
        device: torch.device,
    ) -> CouplingResult:
        pairwise_cost = _coerce_pairwise_cost(
            self.build_pairwise_cost(
                x0,
                x1,
                time_idx=time_idx,
                t0=t0,
                t1=t1,
                device=device,
            ),
            time_idx=time_idx,
        )
        result = self._solve_cost(pairwise_cost, device=device)
        result.metadata = {**self.describe_metadata(), **pairwise_cost.metadata, **result.metadata}
        return result

    def build_state(
        self,
        X: list[np.ndarray],
        t_train: torch.Tensor,
        device: torch.device,
    ) -> CouplingState:
        if len(X) != len(t_train):
            raise ValueError(f"len(X) must equal len(t_train), got {len(X)} and {len(t_train)}")

        pairwise_costs: list[PairwiseCost] = []
        for time_idx in range(len(X) - 1):
            pairwise_costs.append(
                _coerce_pairwise_cost(
                    self.build_pairwise_cost(
                        X[time_idx],
                        X[time_idx + 1],
                        time_idx=time_idx,
                        t0=float(torch.as_tensor(t_train[time_idx]).item()),
                        t1=float(torch.as_tensor(t_train[time_idx + 1]).item()),
                        device=device,
                    ),
                    time_idx=time_idx,
                )
            )

        global_reg = None
        global_reg_m = None
        needs_auto_reg = self.solver_mode == "uot" and (self.fixed_reg is None or self.fixed_reg_m is None)
        if self.solver_mode == "uot" and self.reg_strategy == "max_over_time" and needs_auto_reg:
            reg_values: list[float] = []
            reg_m_values: list[float] = []
            for pairwise_cost in pairwise_costs:
                reg_i, reg_m_i = _auto_regularization_for_pairwise_cost(
                    pairwise_cost,
                    use_mini_batch=self.use_mini_batch,
                    chunk_size=self.chunk_size,
                    normalize_cost=self.normalize_cost,
                    device=self._auto_regularization_device(device),
                )
                reg_values.append(self.fixed_reg if self.fixed_reg is not None else reg_i)
                reg_m_values.append(self.fixed_reg_m if self.fixed_reg_m is not None else reg_m_i)
            global_reg = max(reg_values)
            global_reg_m = max(reg_m_values)
        elif self.solver_mode == "uot" and not needs_auto_reg:
            global_reg = self.fixed_reg
            global_reg_m = self.fixed_reg_m

        plan_stores: list[CouplingPlanStore] = []
        pair_metadata: list[dict[str, Any]] = []
        for time_idx, pairwise_cost in enumerate(pairwise_costs):
            result = self._solve_cost(
                pairwise_cost,
                reg=global_reg,
                reg_m=global_reg_m,
                device=device,
            )
            expected_shape = (X[time_idx].shape[0], X[time_idx + 1].shape[0])
            if result.plan.shape != expected_shape:
                raise ValueError(
                    f"Coupling plan for time_idx={time_idx} has shape={result.plan.shape}, "
                    f"expected={expected_shape}"
                )
            result.metadata = {
                **self.describe_metadata(),
                **pairwise_cost.metadata,
                **result.metadata,
            }
            plan_stores.append(
                CouplingPlanStore.from_legacy(
                    result.plan,
                    result.sampling_info,
                    metadata=result.metadata,
                )
            )
            pair_metadata.append(result.metadata)

        metadata = dict(self.describe_metadata())
        metadata["pair_metadata"] = pair_metadata
        return CouplingState(plan_stores=plan_stores, metadata=metadata)


class ChunkedTransportCouplingStrategy(CouplingStrategy):
    """
    Chunked pairwise-cost OT/UOT API for large datasets.

    Use this when the algorithm changes the pairwise cost but a full
    `n_source x n_target` cost matrix is too large. Developers only implement
    `build_pairwise_cost_block(...)`; the package splits adjacent time pairs into
    bounded source/target blocks, solves each block once in `build_state(...)`,
    caches sub-plans, and samples from those cached plans during training.

    This is the preferred scalability path before writing a custom
    `build_state(...)` / `sample_pairs(...)` implementation from scratch.
    """

    def __init__(
        self,
        *,
        solver_mode: str = "uot",
        chunk_size: int = 1000,
        use_mini_batch: bool = True,
        alpha_regm: float = 1.0,
        reg_strategy: str = "per_time",
        reg: float | None = None,
        reg_m: float | None = None,
        auto_reg_device: str = "cpu",
        balanced_method: str = "exact",
        balanced_reg: float = 0.05,
        normalize_cost: bool = False,
        auto_normalize_large_cost: bool = False,
        regularization_block_policy: str = "all_blocks",
        warn: bool = True,
        epsilon: float = 1e-9,
    ) -> None:
        if solver_mode not in {"uot", "balanced"}:
            raise ValueError(f"solver_mode must be 'uot' or 'balanced', got {solver_mode}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if regularization_block_policy not in {"all_blocks", "first_block_per_time"}:
            raise ValueError(
                "regularization_block_policy must be 'all_blocks' or 'first_block_per_time', "
                f"got {regularization_block_policy}"
            )
        self.solver_mode = str(solver_mode)
        self.chunk_size = int(chunk_size)
        self.use_mini_batch = bool(use_mini_batch)
        self.alpha_regm = float(alpha_regm)
        self.reg_strategy = str(reg_strategy)
        self.fixed_reg = None if reg is None else float(reg)
        self.fixed_reg_m = None if reg_m is None else float(reg_m)
        self.auto_reg_device = str(auto_reg_device or "cpu").lower()
        self.balanced_method = str(balanced_method)
        self.balanced_reg = float(balanced_reg)
        self.normalize_cost = bool(normalize_cost)
        self.auto_normalize_large_cost = bool(auto_normalize_large_cost)
        self.regularization_block_policy = str(regularization_block_policy)
        self.warn = bool(warn)
        self.epsilon = float(epsilon)
        self.supports_minibatch = True
        self.uses_chunked_cost_blocks = True
        self.precomputes_coupling_in_build_state = True

    @abstractmethod
    def build_pairwise_cost_block(
        self,
        x0_block: np.ndarray,
        x1_block: np.ndarray,
        *,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        time_idx: int,
        t0: float,
        t1: float,
        device: torch.device,
    ) -> PairwiseCost | np.ndarray | torch.Tensor:
        raise NotImplementedError

    def describe_metadata(self) -> dict[str, Any]:
        return {
            "kind": "chunked_cost_pairwise_ot",
            "solver_mode": self.solver_mode,
            "chunk_size": self.chunk_size,
            "use_mini_batch": self.use_mini_batch,
            "alpha_regm": self.alpha_regm,
            "reg_strategy": self.reg_strategy,
            "fixed_reg": self.fixed_reg,
            "fixed_reg_m": self.fixed_reg_m,
            "auto_reg_device": self.auto_reg_device,
            "balanced_method": self.balanced_method,
            "balanced_reg": self.balanced_reg,
            "normalize_cost": self.normalize_cost,
            "auto_normalize_large_cost": self.auto_normalize_large_cost,
            "regularization_block_policy": self.regularization_block_policy,
        }

    def _solve_block(
        self,
        pairwise_cost: PairwiseCost,
        *,
        reg: float | None,
        reg_m: float | None,
        device: torch.device,
    ) -> CouplingResult:
        if self.solver_mode == "uot":
            effective_reg = self.fixed_reg if reg is None else reg
            effective_reg_m = self.fixed_reg_m if reg_m is None else reg_m
            return _solve_uot_from_pairwise_cost(
                pairwise_cost,
                use_mini_batch=False,
                chunk_size=self.chunk_size,
                alpha_regm=self.alpha_regm,
                reg=effective_reg,
                reg_m=effective_reg_m,
                normalize_cost=self.normalize_cost,
                device=device,
            )
        return _solve_balanced_ot_from_pairwise_cost(
            pairwise_cost,
            use_mini_batch=False,
            chunk_size=self.chunk_size,
            method=self.balanced_method,
            reg=self.balanced_reg,
            normalize_cost=self.normalize_cost,
            warn=self.warn,
        )

    def _split_blocks(self, n_source: int, n_target: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not self.use_mini_batch:
            return [np.arange(n_source)], [np.arange(n_target)]
        return _split_transport_chunks(n_source, n_target, self.chunk_size)

    def _needs_auto_regularization(self) -> bool:
        return self.solver_mode == "uot" and (self.fixed_reg is None or self.fixed_reg_m is None)

    def _auto_regularization_device(self, runtime_device: torch.device) -> torch.device:
        if self.auto_reg_device in {"cuda", "gpu"} and torch.cuda.is_available():
            return torch.device("cuda")
        if self.auto_reg_device in {"runtime", "device"} and torch.device(runtime_device).type == "cuda":
            return runtime_device
        return torch.device("cpu")

    def build_state(
        self,
        X: list[np.ndarray],
        t_train: torch.Tensor,
        device: torch.device,
    ) -> CouplingState:
        if len(X) != len(t_train):
            raise ValueError(f"len(X) must equal len(t_train), got {len(X)} and {len(t_train)}")

        time_blocks: list[list[dict[str, Any]]] = []
        all_reg_values: list[float] = []
        all_reg_m_values: list[float] = []

        for time_idx in range(len(X) - 1):
            t0 = float(torch.as_tensor(t_train[time_idx]).item())
            t1 = float(torch.as_tensor(t_train[time_idx + 1]).item())
            source_chunks, target_chunks = self._split_blocks(
                X[time_idx].shape[0],
                X[time_idx + 1].shape[0],
            )

            block_specs: list[dict[str, Any]] = []
            local_reg_values: list[float] = []
            local_reg_m_values: list[float] = []
            for src_chunk, tgt_chunk in zip(source_chunks, target_chunks):
                if len(src_chunk) == 0 or len(tgt_chunk) == 0:
                    continue
                pairwise_cost = _coerce_pairwise_cost(
                    self.build_pairwise_cost_block(
                        X[time_idx][src_chunk],
                        X[time_idx + 1][tgt_chunk],
                        source_indices=src_chunk,
                        target_indices=tgt_chunk,
                        time_idx=time_idx,
                        t0=t0,
                        t1=t1,
                        device=device,
                    ),
                    time_idx=time_idx,
                )
                expected_shape = (len(src_chunk), len(tgt_chunk))
                if pairwise_cost.cost_matrix.shape != expected_shape:
                    raise ValueError(
                        f"Chunked cost block for time_idx={time_idx} has shape="
                        f"{pairwise_cost.cost_matrix.shape}, expected={expected_shape}"
                    )
                block_specs.append(
                    {
                        "source_indices": np.asarray(src_chunk, dtype=np.int64),
                        "target_indices": np.asarray(tgt_chunk, dtype=np.int64),
                        "pairwise_cost": pairwise_cost,
                    }
                )

            if self.auto_normalize_large_cost and block_specs:
                total_sum = 0.0
                total_count = 0
                max_value = 0.0
                for spec in block_specs:
                    block_sum, block_count, block_max = _cost_matrix_sum_count_max(spec["pairwise_cost"].cost_matrix)
                    total_sum += block_sum
                    total_count += block_count
                    max_value = max(max_value, block_max)
                mean_value = total_sum / max(total_count, 1)
                if mean_value > 100.0 and max_value > 0.0:
                    for spec in block_specs:
                        pairwise_cost = spec["pairwise_cost"]
                        spec["pairwise_cost"] = PairwiseCost(
                            cost_matrix=_scale_cost_matrix(pairwise_cost.cost_matrix, max_value),
                            source_mass=pairwise_cost.source_mass,
                            target_mass=pairwise_cost.target_mass,
                            normalize_cost=False,
                            metadata={
                                **pairwise_cost.metadata,
                                "auto_normalized_large_cost": True,
                                "auto_normalization_scale": float(max_value),
                            },
                        )

            if self._needs_auto_regularization():
                tune_specs = (
                    block_specs[:1]
                    if self.regularization_block_policy == "first_block_per_time"
                    else block_specs
                )
                for spec in tune_specs:
                    reg_i, reg_m_i = _auto_regularization_for_pairwise_cost(
                        spec["pairwise_cost"],
                        use_mini_batch=False,
                        chunk_size=self.chunk_size,
                        normalize_cost=self.normalize_cost,
                        device=self._auto_regularization_device(device),
                    )
                    reg_i = self.fixed_reg if self.fixed_reg is not None else reg_i
                    reg_m_i = self.fixed_reg_m if self.fixed_reg_m is not None else reg_m_i
                    local_reg_values.append(reg_i)
                    local_reg_m_values.append(reg_m_i)
                    all_reg_values.append(reg_i)
                    all_reg_m_values.append(reg_m_i)

            if self._needs_auto_regularization():
                if self.reg_strategy == "max_over_time":
                    reg = reg_m = None
                else:
                    reg = max(local_reg_values) if local_reg_values else None
                    reg_m = max(local_reg_m_values) if local_reg_m_values else None
                for spec in block_specs:
                    spec["reg"] = reg
                    spec["reg_m"] = reg_m
            elif self.solver_mode == "uot":
                for spec in block_specs:
                    spec["reg"] = self.fixed_reg
                    spec["reg_m"] = self.fixed_reg_m
            time_blocks.append(block_specs)

        global_reg = None
        global_reg_m = None
        if self.solver_mode == "uot" and self.reg_strategy == "max_over_time":
            global_reg = max(all_reg_values) if all_reg_values else self.fixed_reg
            global_reg_m = max(all_reg_m_values) if all_reg_m_values else self.fixed_reg_m

        plan_stores: list[CouplingPlanStore] = []
        pair_metadata: list[dict[str, Any]] = []
        for time_idx, block_specs in enumerate(time_blocks):
            sub_plans: list[np.ndarray] = []
            source_groups: list[np.ndarray] = []
            target_groups: list[np.ndarray] = []
            terminal_mass_rows: list[np.ndarray] = []
            terminal_mass_subplans: list[np.ndarray] = []
            has_terminal_mass_subplans = False
            block_metadata: list[dict[str, Any]] = []

            for spec in block_specs:
                reg = global_reg if self.reg_strategy == "max_over_time" else spec.get("reg")
                reg_m = global_reg_m if self.reg_strategy == "max_over_time" else spec.get("reg_m")
                result = self._solve_block(spec["pairwise_cost"], reg=reg, reg_m=reg_m, device=device)
                sub_plan = np.asarray(result.plan, dtype=np.float32)
                sub_plans.append(sub_plan)
                source_groups.append(spec["source_indices"])
                target_groups.append(spec["target_indices"])
                terminal_payload = self._terminal_mass_payload_for_block(result)
                if terminal_payload.ndim == 2:
                    has_terminal_mass_subplans = True
                    terminal_mass_subplans.append(terminal_payload.astype(np.float32, copy=False))
                elif terminal_payload.ndim == 1:
                    terminal_mass_rows.append(terminal_payload.astype(np.float32, copy=False))
                else:
                    raise ValueError(
                        f"terminal mass payload for time_idx={time_idx} must be 1D or 2D, got {terminal_payload.shape}"
                    )
                block_metadata.append(
                    {
                        **spec["pairwise_cost"].metadata,
                        **_lightweight_metadata(result.metadata),
                        "source_n": int(len(spec["source_indices"])),
                        "target_n": int(len(spec["target_indices"])),
                    }
                )

            source_n = X[time_idx].shape[0]
            target_n = X[time_idx + 1].shape[0]
            has_single_complete_block = (
                len(sub_plans) == 1
                and source_groups[0].shape[0] == source_n
                and target_groups[0].shape[0] == target_n
            )
            if has_single_complete_block:
                # Keep exact dense plans for small/single-block intervals. Larger
                # chunked intervals still avoid materializing full n_source x n_target.
                dense_plan = np.zeros((source_n, target_n), dtype=np.float32)
                dense_plan[np.ix_(source_groups[0], target_groups[0])] = sub_plans[0]
            else:
                dense_plan = None
            metadata_i = {
                **self.describe_metadata(),
                "time_idx": int(time_idx),
                "chunked": not (has_single_complete_block and not has_terminal_mass_subplans),
                "n_chunks": int(len(sub_plans)),
                "reg": float(global_reg) if global_reg is not None else (
                    float(block_specs[0]["reg"]) if block_specs and block_specs[0].get("reg") is not None else None
                ),
                "reg_m": float(global_reg_m) if global_reg_m is not None else (
                    float(block_specs[0]["reg_m"]) if block_specs and block_specs[0].get("reg_m") is not None else None
                ),
                "block_metadata": block_metadata,
            }
            pair_metadata.append(metadata_i)
            plan_stores.append(
                CouplingPlanStore.from_chunked(
                    source_n=source_n,
                    target_n=target_n,
                    sub_plans=[] if has_single_complete_block and not has_terminal_mass_subplans else sub_plans,
                    source_groups=[] if has_single_complete_block and not has_terminal_mass_subplans else source_groups,
                    target_groups=[] if has_single_complete_block and not has_terminal_mass_subplans else target_groups,
                    dense_plan=dense_plan,
                    terminal_mass_rows=(
                        None
                        if has_single_complete_block and not has_terminal_mass_subplans
                        else (None if has_terminal_mass_subplans or not terminal_mass_rows else terminal_mass_rows)
                    ),
                    terminal_mass_subplans=terminal_mass_subplans if has_terminal_mass_subplans else None,
                    metadata=metadata_i,
                )
            )

        metadata = dict(self.describe_metadata())
        metadata["pair_metadata"] = pair_metadata
        return CouplingState(plan_stores=plan_stores, metadata=metadata)

    def _terminal_mass_payload_for_block(self, result: CouplingResult) -> np.ndarray:
        if "terminal_mass_matrix" in result.metadata:
            return np.asarray(result.metadata["terminal_mass_matrix"], dtype=np.float32)
        if "terminal_mass_rows" in result.metadata:
            return np.asarray(result.metadata["terminal_mass_rows"], dtype=np.float32).reshape(-1)
        return np.asarray(result.plan, dtype=np.float32).sum(axis=1)

    def sample_pairs(
        self,
        state: CouplingState,
        X: list[np.ndarray],
        time_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[PairBatch]:
        return state.plan_store(time_idx).sample_pairs(
            X=X,
            time_idx=time_idx,
            batch_size=batch_size,
            device=device,
            epsilon=self.epsilon,
        )


class ChunkedCostPairwiseOTCouplingStrategy(ChunkedTransportCouplingStrategy):
    """Backward-compatible name for chunked pairwise-cost transport couplings."""


class MassStrategy(ABC):
    """
    Developer hook for per-particle terminal mass targets.

    Contract:
    - return `MassTargets`
    - let the conditional path convert terminal mass into local growth targets
      and training weights
    """

    @abstractmethod
    def compute(
        self,
        state: CouplingState,
        pair_batch: PairBatch,
        t_local: torch.Tensor,
        device: torch.device,
    ) -> MassTargets:
        raise NotImplementedError


class RegularizedUnbalancedConditionalPath(ConditionalFlowMatcher):
    """
    Default RUOT path used by the package.

    This only defines the conditional path and noise schedule. It intentionally
    does not know how couplings or mass targets are computed.

    `sigma=0` is allowed and gives the deterministic VGFM limit: the sampled
    intermediate state is exactly the linear interpolation `mu_t`, so the
    stochastic correction term in `compute_conditional_flow` vanishes.
    """

    def __init__(self, sigma: float = 1.0):
        if sigma < 0:
            raise ValueError(f"Sigma must be non-negative, got {sigma}.")
        super().__init__(sigma=sigma)

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t)
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.as_tensor(t, device=x0.device, dtype=x0.dtype).reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        return sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0


class LinearDeterministicConditionalPath(ConditionalFlowMatcher):
    """Linear CFM path with no diffusion by default."""

    def __init__(self, sigma: float = 0.0):
        if sigma < 0:
            raise ValueError(f"Sigma must be non-negative, got {sigma}.")
        super().__init__(sigma=sigma)


class SchrodingerBridgeConditionalPath(RegularizedUnbalancedConditionalPath):
    """
    Stochastic bridge path used by SF2M-style balanced OT flow matching.

    The path is mass-balanced by construction when paired with NullMassStrategy;
    the class name keeps this semantic separate from CRUFM's unbalanced mass
    coupling even though the stochastic path equation is the same bridge family.
    """

    def __init__(self, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError(f"SchrodingerBridgeConditionalPath requires sigma > 0, got {sigma}.")
        super().__init__(sigma=sigma)


class UnbalancedOTCouplingStrategy(ChunkedTransportCouplingStrategy):
    """Current package default: precompute UOT plans between adjacent time points."""

    def __init__(
        self,
        *,
        use_mini_batch_uot: bool = True,
        chunk_size: int = 1000,
        alpha_regm: float = 1.0,
        reg_strategy: str = "max_over_time",
        reg: float | None = None,
        reg_m: float | None = None,
        auto_reg_device: str = "cpu",
    ) -> None:
        super().__init__(
            solver_mode="uot",
            chunk_size=int(chunk_size),
            use_mini_batch=bool(use_mini_batch_uot),
            alpha_regm=float(alpha_regm),
            reg_strategy=str(reg_strategy),
            reg=reg,
            reg_m=reg_m,
            auto_reg_device=auto_reg_device,
            auto_normalize_large_cost=True,
            regularization_block_policy="first_block_per_time",
        )
        self.use_mini_batch_uot = self.use_mini_batch

    def describe_metadata(self) -> dict[str, Any]:
        return {
            "kind": "unbalanced_ot",
            "use_mini_batch_uot": self.use_mini_batch_uot,
            "chunk_size": self.chunk_size,
            "alpha_regm": self.alpha_regm,
            "reg_strategy": self.reg_strategy,
            "fixed_reg": self.fixed_reg,
            "fixed_reg_m": self.fixed_reg_m,
            "auto_reg_device": self.auto_reg_device,
        }

    def build_pairwise_cost_block(
        self,
        x0_block: np.ndarray,
        x1_block: np.ndarray,
        *,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        time_idx: int,
        t0: float,
        t1: float,
        device: torch.device,
    ) -> PairwiseCost:
        del source_indices, target_indices, t0, t1
        x0_t = torch.as_tensor(x0_block, dtype=torch.float32, device=device)
        x1_t = torch.as_tensor(x1_block, dtype=torch.float32, device=device)
        return PairwiseCost(
            cost_matrix=torch.cdist(x0_t, x1_t).pow(2),
            source_mass=np.ones(x0_block.shape[0], dtype=np.float32),
            target_mass=np.ones(x1_block.shape[0], dtype=np.float32),
            metadata={"time_idx": int(time_idx)},
        )


class BalancedOTCouplingStrategy(ChunkedTransportCouplingStrategy):
    """
    Balanced OT alternative for developers who do not want unbalanced coupling.

    `method` can be `exact` or `sinkhorn` because those are the stable balanced
    paths already supported by `OTPlanSampler`.
    """

    def __init__(
        self,
        *,
        method: str = "exact",
        reg: float = 0.05,
        normalize_cost: bool = False,
        warn: bool = True,
        use_mini_batch_balanced: bool = False,
        chunk_size: int = 1000,
    ) -> None:
        super().__init__(
            solver_mode="balanced",
            chunk_size=int(chunk_size),
            use_mini_batch=bool(use_mini_batch_balanced),
            balanced_method=str(method),
            balanced_reg=float(reg),
            normalize_cost=bool(normalize_cost),
            warn=bool(warn),
        )
        self.method = self.balanced_method
        self.reg = self.balanced_reg
        self.use_mini_batch_balanced = self.use_mini_batch

    def describe_metadata(self) -> dict[str, Any]:
        return {
            "kind": "balanced_ot",
            "method": self.method,
            "reg": self.reg,
            "normalize_cost": self.normalize_cost,
            "use_mini_batch_balanced": self.use_mini_batch_balanced,
            "chunk_size": self.chunk_size,
        }

    def build_pairwise_cost_block(
        self,
        x0_block: np.ndarray,
        x1_block: np.ndarray,
        *,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        time_idx: int,
        t0: float,
        t1: float,
        device: torch.device,
    ) -> PairwiseCost:
        del source_indices, target_indices, t0, t1
        x0_t = torch.as_tensor(x0_block, dtype=torch.float32, device=device)
        x1_t = torch.as_tensor(x1_block, dtype=torch.float32, device=device)
        return PairwiseCost(
            cost_matrix=torch.cdist(x0_t, x1_t).pow(2),
            source_mass=np.ones(x0_block.shape[0], dtype=np.float32),
            target_mass=np.ones(x1_block.shape[0], dtype=np.float32),
            metadata={"time_idx": int(time_idx)},
        )


class UOTMassStrategy(MassStrategy):
    """
    Default package semantics for terminal mass.

    For each sampled source particle, this defines the target mass ratio at the
    end of the local interval under the convention that source mass is
    normalized to `1` at local time `t=0`.
    """

    def compute(
        self,
        state: CouplingState,
        pair_batch: PairBatch,
        t_local: torch.Tensor,
        device: torch.device,
    ) -> MassTargets:
        del t_local
        store = state.plan_store(pair_batch.time_idx)
        terminal_mass_np = store.terminal_mass_for_batch(pair_batch)
        terminal_mass = torch.tensor(
            terminal_mass_np,
            dtype=torch.float32,
            device=device,
        )
        return MassTargets(terminal_mass=terminal_mass)


class NullMassStrategy(MassStrategy):
    """Balanced/no-growth fallback: terminal mass stays at 1."""

    def compute(
        self,
        state: CouplingState,
        pair_batch: PairBatch,
        t_local: torch.Tensor,
        device: torch.device,
    ) -> MassTargets:
        del state, pair_batch
        batch = t_local.shape[0]
        return MassTargets(
            terminal_mass=torch.ones(batch, 1, dtype=torch.float32, device=device),
        )


class WFROETCouplingStrategy(ChunkedTransportCouplingStrategy):
    """
    WFR-FM coupling based on the OET form of WFR.

    Semantics:
    - solve the WFR optimal entropy-transport problem with marginal KL penalties;
    - sample from the source-side semi-coupling gamma0;
    - attach the pair-specific terminal mass gamma1 / gamma0 for the WFR path.

    The mini-batch path stores only subplans and samples from them directly, so
    package `wfrfm` can run as a scalable builtin example instead of forcing a
    dense full coupling on biological-scale data.
    """

    def __init__(
        self,
        *,
        delta: float,
        chunk_size: int = 1000,
        use_mini_batch: bool = True,
        alpha_regm: float = 1.0,
        reg: float | None = None,
        reg_m: float | None = None,
        epsilon: float = _WFR_EPS,
        angle_margin: float = _WFR_ANGLE_MARGIN,
        delta_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            solver_mode="uot",
            chunk_size=int(chunk_size),
            alpha_regm=float(alpha_regm),
            reg_strategy="per_time",
            normalize_cost=False,
            use_mini_batch=bool(use_mini_batch),
        )
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if abs(float(alpha_regm) - 1.0) > 1e-12:
            raise ValueError(
                "Strict WFR-OET reproduction requires alpha_regm=1.0 because WFR-FM uses unit "
                "coefficients on both marginal KL penalties."
            )
        if reg is not None and abs(float(reg)) > 1e-12:
            raise ValueError("Strict WFR-OET reproduction requires reg=None or reg=0.0.")
        if reg_m is not None and abs(float(reg_m) - 1.0) > 1e-12:
            raise ValueError("Strict WFR-OET reproduction requires reg_m=None or reg_m=1.0.")
        self.delta = float(delta)
        self.fixed_reg = 0.0
        self.fixed_reg_m = 1.0
        self.epsilon = float(epsilon)
        self.angle_margin = float(angle_margin)
        self.delta_metadata = dict(delta_metadata or {})

    def describe_metadata(self) -> dict[str, Any]:
        return {
            "kind": "wfr_oet_semicoupling",
            "delta": self.delta,
            "delta_metadata": dict(self.delta_metadata),
            "chunk_size": self.chunk_size,
            "use_mini_batch": self.use_mini_batch,
            "alpha_regm": self.alpha_regm,
            "fixed_reg": self.fixed_reg,
            "fixed_reg_m": self.fixed_reg_m,
        }

    def build_pairwise_cost_block(
        self,
        x0_block: np.ndarray,
        x1_block: np.ndarray,
        *,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        time_idx: int,
        t0: float,
        t1: float,
        device: torch.device,
    ) -> PairwiseCost:
        del source_indices, target_indices, t0, t1
        cost_matrix = self._compute_wfr_cost_matrix(x0_block, x1_block, device=device)
        n0, n1 = cost_matrix.shape
        return PairwiseCost(
            cost_matrix=cost_matrix,
            source_mass=np.ones(n0, dtype=np.float32),
            target_mass=np.ones(n1, dtype=np.float32),
            metadata={"time_idx": int(time_idx), "wfr_delta": self.delta},
        )

    def _compute_wfr_cost_matrix(self, x0: np.ndarray, x1: np.ndarray, *, device: torch.device) -> torch.Tensor:
        x0_t = torch.as_tensor(x0, dtype=torch.float32, device=device)
        x1_t = torch.as_tensor(x1, dtype=torch.float32, device=device)
        distance = torch.cdist(x0_t, x1_t)
        angles = torch.clamp(
            distance / (2.0 * self.delta),
            min=0.0,
            max=(math.pi / 2.0) - self.angle_margin,
        )
        return -2.0 * torch.log(torch.cos(angles))

    def _solve_gamma(self, pairwise_cost: PairwiseCost, *, device: torch.device) -> tuple[np.ndarray, float, float]:
        result = _solve_wfr_oet_block(pairwise_cost, epsilon=self.epsilon, device=device)
        return np.asarray(result.metadata["gamma"], dtype=np.float32), self.fixed_reg, self.fixed_reg_m

    def _convert_to_semicouplings(
        self,
        gamma: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return _convert_wfr_gamma_to_semicouplings(gamma, a, b, epsilon=self.epsilon)

    def _solve_block(
        self,
        pairwise_cost: PairwiseCost,
        *,
        reg: float | None = None,
        reg_m: float | None = None,
        device: torch.device,
    ) -> CouplingResult:
        del reg, reg_m
        result = _solve_wfr_oet_block(pairwise_cost, epsilon=self.epsilon, device=device)
        result.metadata = {**self.describe_metadata(), **result.metadata}
        return result

    def _needs_auto_regularization(self) -> bool:
        return False


class WFRTerminalMassStrategy(MassStrategy):
    """Return pair-specific WFR terminal mass ratio gamma1 / gamma0."""

    def __init__(self, epsilon: float = _WFR_EPS) -> None:
        self.epsilon = float(epsilon)

    def compute(
        self,
        state: CouplingState,
        pair_batch: PairBatch,
        t_local: torch.Tensor,
        device: torch.device,
    ) -> MassTargets:
        del t_local
        if pair_batch.sampled_terminal_mass is not None:
            selected = np.asarray(pair_batch.sampled_terminal_mass, dtype=np.float32).reshape(-1, 1)
        else:
            pair_metadata = state.metadata.get("pair_metadata", [])
            if pair_batch.time_idx >= len(pair_metadata):
                raise IndexError(f"Missing pair metadata for time_idx={pair_batch.time_idx}")
            terminal_mass_matrix = np.asarray(pair_metadata[pair_batch.time_idx]["terminal_mass_matrix"], dtype=np.float32)
            selected = terminal_mass_matrix[pair_batch.idx0, pair_batch.idx1].reshape(-1, 1)
        terminal_mass = torch.tensor(
            np.maximum(selected, self.epsilon),
            dtype=torch.float32,
            device=device,
        )
        return MassTargets(terminal_mass=terminal_mass, metadata={"kind": "wfr_terminal_mass_ratio"})


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _canonical_policy_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _bool_option(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _get_first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _flow_matching_options(stage_params: dict[str, Any]) -> dict[str, Any]:
    options = dict(_as_dict(stage_params.get("flow_matching")))
    if stage_params.get("flow_matching_backend") is not None:
        options["backend"] = stage_params.get("flow_matching_backend")
    return options


def _path_sigma(
    stage_params: dict[str, Any],
    path_config: dict[str, Any],
    *,
    default: float,
) -> float:
    return float(_get_first(path_config, "sigma", default=_get_first(stage_params, "sigma", default=default)))


def _build_balanced_coupling(coupling_config: dict[str, Any]) -> BalancedOTCouplingStrategy:
    return BalancedOTCouplingStrategy(
        method=str(_get_first(coupling_config, "method", "balanced_method", default="exact")),
        reg=float(_get_first(coupling_config, "reg", "balanced_reg", default=0.05)),
        normalize_cost=_bool_option(coupling_config.get("normalize_cost"), False),
        warn=_bool_option(coupling_config.get("warn"), True),
        use_mini_batch_balanced=_bool_option(
            _get_first(coupling_config, "use_mini_batch", "use_mini_batch_balanced"),
            True,
        ),
        chunk_size=int(_get_first(coupling_config, "chunk_size", default=1000)),
    )


def _build_sf2m_coupling(coupling_config: dict[str, Any], *, sigma: float) -> BalancedOTCouplingStrategy:
    """
    Build the SF2M/SB-CFM coupling.

    Use exact OT by default, matching the practical reference implementation.
    Entropic/Sinkhorn OT remains configurable for experiments, but small
    regularization can be numerically fragile on CellCompass benchmark data.
    """
    return BalancedOTCouplingStrategy(
        method=str(_get_first(coupling_config, "method", "balanced_method", default="exact")),
        reg=float(_get_first(coupling_config, "reg", "balanced_reg", default=2.0 * sigma * sigma)),
        normalize_cost=_bool_option(coupling_config.get("normalize_cost"), False),
        warn=_bool_option(coupling_config.get("warn"), True),
        use_mini_batch_balanced=_bool_option(
            _get_first(coupling_config, "use_mini_batch", "use_mini_batch_balanced"),
            True,
        ),
        chunk_size=int(_get_first(coupling_config, "chunk_size", default=1000)),
    )


def _build_unbalanced_coupling(coupling_config: dict[str, Any]) -> UnbalancedOTCouplingStrategy:
    return UnbalancedOTCouplingStrategy(
        use_mini_batch_uot=_bool_option(
            _get_first(coupling_config, "use_mini_batch", "use_mini_batch_uot"),
            True,
        ),
        chunk_size=int(_get_first(coupling_config, "chunk_size", default=1000)),
        alpha_regm=float(_get_first(coupling_config, "alpha_regm", default=1.0)),
        reg_strategy=str(_get_first(coupling_config, "reg_strategy", default="max_over_time")),
        reg=_float_or_none(_get_first(coupling_config, "reg", "uot_reg", "ot_reg")),
        reg_m=_float_or_none(_get_first(coupling_config, "reg_m", "uot_reg_m", "ot_reg_m")),
        auto_reg_device=str(_get_first(coupling_config, "auto_reg_device", default="cpu")),
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(value)


def _is_auto_delta_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"auto", "adaptive", "data", "dataset", "true", "yes", "on"}


def _wfr_delta_setting(
    coupling_config: dict[str, Any],
    path_config: dict[str, Any],
    stage_params: dict[str, Any],
) -> Any:
    return _get_first(
        coupling_config,
        "delta",
        default=_get_first(
            path_config,
            "delta",
            default=_get_first(stage_params, "delta", default="auto"),
        ),
    )


def _wfr_auto_delta_requested(
    coupling_config: dict[str, Any],
    path_config: dict[str, Any],
    stage_params: dict[str, Any],
) -> bool:
    explicit = _get_first(
        coupling_config,
        "auto_delta",
        "delta_auto",
        "delta_strategy",
        default=_get_first(
            path_config,
            "auto_delta",
            "delta_auto",
            "delta_strategy",
            default=_get_first(stage_params, "auto_delta", "delta_auto", "delta_strategy"),
        ),
    )
    return _is_auto_delta_value(explicit) or _is_auto_delta_value(
        _wfr_delta_setting(coupling_config, path_config, stage_params)
    )


def _sample_adjacent_pair_distances(
    training_data: Any,
    *,
    sample_size: int,
    pair_samples: int,
    seed: int,
) -> np.ndarray:
    latent_by_time = list(getattr(training_data, "latent_by_time", []) or [])
    if len(latent_by_time) < 2:
        return np.empty(0, dtype=np.float32)

    rng = np.random.default_rng(int(seed))
    sampled: list[np.ndarray] = []
    for x0_raw, x1_raw in zip(latent_by_time[:-1], latent_by_time[1:]):
        x0 = _as_numpy_array(x0_raw).astype(np.float32, copy=False)
        x1 = _as_numpy_array(x1_raw).astype(np.float32, copy=False)
        if x0.ndim != 2 or x1.ndim != 2 or x0.shape[0] == 0 or x1.shape[0] == 0:
            continue
        n0 = min(int(sample_size), int(x0.shape[0]))
        n1 = min(int(sample_size), int(x1.shape[0]))
        if n0 <= 0 or n1 <= 0:
            continue
        src = x0[rng.choice(x0.shape[0], size=n0, replace=False)]
        tgt = x1[rng.choice(x1.shape[0], size=n1, replace=False)]
        pairs = max(1, int(pair_samples))
        idx0 = rng.integers(0, n0, size=pairs)
        idx1 = rng.integers(0, n1, size=pairs)
        distances = np.linalg.norm(src[idx0] - tgt[idx1], axis=1)
        distances = distances[np.isfinite(distances)]
        if distances.size:
            sampled.append(distances.astype(np.float32, copy=False))
    if not sampled:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(sampled).astype(np.float32, copy=False)


def _resolve_wfr_delta(
    coupling_config: dict[str, Any],
    path_config: dict[str, Any],
    stage_params: dict[str, Any],
    *,
    training_data: Any | None,
) -> tuple[float, dict[str, Any]]:
    raw_delta = _wfr_delta_setting(coupling_config, path_config, stage_params)
    if not _wfr_auto_delta_requested(coupling_config, path_config, stage_params):
        return float(raw_delta), {"strategy": "fixed", "configured_delta": float(raw_delta)}

    if training_data is None:
        fallback = float(_get_first(coupling_config, "fallback_delta", default=_get_first(stage_params, "fallback_delta", default=1.5)))
        return fallback, {"strategy": "auto_adjacent_distance_q", "fallback_reason": "missing_training_data", "fallback_delta": fallback}

    quantile = float(_get_first(coupling_config, "delta_quantile", default=_get_first(stage_params, "delta_quantile", default=0.9)))
    target_angle = float(_get_first(coupling_config, "delta_target_angle", default=_get_first(stage_params, "delta_target_angle", default=1.0)))
    sample_size = int(_get_first(coupling_config, "delta_sample_size", default=_get_first(stage_params, "delta_sample_size", default=512)))
    pair_samples = int(_get_first(coupling_config, "delta_pair_samples", default=_get_first(stage_params, "delta_pair_samples", default=20000)))
    seed = int(_get_first(coupling_config, "delta_seed", default=_get_first(stage_params, "delta_seed", default=0)))
    min_delta = float(_get_first(coupling_config, "delta_min", default=_get_first(stage_params, "delta_min", default=1e-3)))
    max_delta_raw = _get_first(coupling_config, "delta_max", default=_get_first(stage_params, "delta_max"))
    max_delta = None if max_delta_raw is None else float(max_delta_raw)

    if not (0.0 < quantile < 1.0):
        raise ValueError(f"WFR auto delta requires 0 < delta_quantile < 1, got {quantile}")
    if target_angle <= 0:
        raise ValueError(f"WFR auto delta requires delta_target_angle > 0, got {target_angle}")

    distances = _sample_adjacent_pair_distances(
        training_data,
        sample_size=sample_size,
        pair_samples=pair_samples,
        seed=seed,
    )
    if distances.size == 0:
        fallback = float(_get_first(coupling_config, "fallback_delta", default=_get_first(stage_params, "fallback_delta", default=1.5)))
        return fallback, {"strategy": "auto_adjacent_distance_q", "fallback_reason": "no_valid_distances", "fallback_delta": fallback}

    distance_q = float(np.quantile(distances, quantile))
    resolved = max(float(min_delta), distance_q / (2.0 * target_angle))
    if max_delta is not None:
        resolved = min(max_delta, resolved)

    saturation_threshold = (math.pi - (2.0 * _WFR_ANGLE_MARGIN)) * resolved
    metadata = {
        "strategy": "auto_adjacent_distance_q",
        "resolved_delta": float(resolved),
        "distance_quantile": quantile,
        "distance_quantile_value": distance_q,
        "target_angle": target_angle,
        "sample_size_per_timepoint": sample_size,
        "pair_samples_per_gap": pair_samples,
        "seed": seed,
        "min_delta": min_delta,
        "max_delta": max_delta,
        "sampled_distance_count": int(distances.size),
        "angle_at_quantile": float(distance_q / (2.0 * resolved)),
        "saturation_fraction": float(np.mean(distances >= saturation_threshold)),
    }
    return float(resolved), metadata


def _write_resolved_wfr_delta(
    stage_params: dict[str, Any],
    coupling_config: dict[str, Any],
    path_config: dict[str, Any],
    *,
    delta: float,
    metadata: dict[str, Any],
) -> None:
    stage_params["delta"] = float(delta)
    stage_params["wfr_delta_metadata"] = dict(metadata)
    flow_matching = stage_params.get("flow_matching")
    if isinstance(flow_matching, dict):
        coupling = flow_matching.setdefault("coupling", {})
        if isinstance(coupling, dict):
            coupling["delta"] = float(delta)
            coupling["wfr_delta_metadata"] = dict(metadata)
        path = flow_matching.setdefault("path", {})
        if isinstance(path, dict):
            path["delta"] = float(delta)
            path["wfr_delta_metadata"] = dict(metadata)
    coupling_config["delta"] = float(delta)
    coupling_config["wfr_delta_metadata"] = dict(metadata)
    path_config["delta"] = float(delta)
    path_config["wfr_delta_metadata"] = dict(metadata)


def _build_wfr_coupling(
    coupling_config: dict[str, Any],
    *,
    stage_params: dict[str, Any],
    path_config: dict[str, Any],
    training_data: Any | None = None,
) -> WFROETCouplingStrategy:
    delta, delta_metadata = _resolve_wfr_delta(
        coupling_config,
        path_config,
        stage_params,
        training_data=training_data,
    )
    _write_resolved_wfr_delta(
        stage_params,
        coupling_config,
        path_config,
        delta=delta,
        metadata=delta_metadata,
    )
    epsilon = float(_get_first(coupling_config, "epsilon", default=_get_first(path_config, "epsilon", default=_get_first(stage_params, "epsilon", default=_WFR_EPS))))
    return WFROETCouplingStrategy(
        delta=delta,
        chunk_size=int(_get_first(coupling_config, "chunk_size", default=_get_first(stage_params, "chunk_size", default=1000))),
        use_mini_batch=_bool_option(_get_first(coupling_config, "use_mini_batch", "use_mini_batch_wfr", default=_get_first(stage_params, "use_mini_batch_wfr", default=True)), True),
        alpha_regm=float(_get_first(coupling_config, "alpha_regm", default=_get_first(stage_params, "alpha_regm", default=1.0))),
        reg=_float_or_none(_get_first(coupling_config, "reg", "wfr_ot_reg", default=_get_first(stage_params, "wfr_ot_reg"))),
        reg_m=_float_or_none(_get_first(coupling_config, "reg_m", "wfr_ot_reg_m", default=_get_first(stage_params, "wfr_ot_reg_m"))),
        epsilon=epsilon,
        delta_metadata=delta_metadata,
    )


def _configured_backend_for_stage(
    stage_params: dict[str, Any],
    *,
    regress_score: bool,
    training_data: Any | None = None,
) -> Optional["FlowMatchingBackend"]:
    options = _flow_matching_options(stage_params)
    policy = _canonical_policy_name(_get_first(options, "backend", "policy", "name"))
    if not policy:
        return None

    coupling_config = _as_dict(options.get("coupling"))
    path_config = _as_dict(options.get("path"))
    mass_config = _as_dict(options.get("mass"))

    if policy in {"balanced_ot_cfm", "ot_cfm", "balanced_cfm"}:
        path = LinearDeterministicConditionalPath(
            sigma=_path_sigma(stage_params, path_config, default=0.0),
        )
        if regress_score and path.sigma <= 0:
            raise ValueError("balanced_ot_cfm score regression requires a positive path sigma.")
        return FlowMatchingBackend(
            path=path,
            coupling=_build_balanced_coupling(coupling_config),
            mass=NullMassStrategy(),
        )

    if policy in {"sf2m", "sb_cfm", "balanced_stochastic_cfm"}:
        sigma = _path_sigma(stage_params, path_config, default=0.05)
        return FlowMatchingBackend(
            path=SchrodingerBridgeConditionalPath(sigma=sigma),
            coupling=_build_sf2m_coupling(coupling_config, sigma=sigma),
            mass=NullMassStrategy(),
        )

    if policy in {
        "vgfm",
        "unbalanced_ot_cfm",
        "unbalanced_deterministic_cfm",
    }:
        path = LinearDeterministicConditionalPath(
            sigma=_path_sigma(stage_params, path_config, default=0.0),
        )
        if regress_score and path.sigma <= 0:
            raise ValueError("vgfm/deterministic unbalanced FM score regression requires a positive path sigma.")
        mass_kind = _canonical_policy_name(_get_first(mass_config, "kind", "mode", default="uot"))
        mass_strategy: MassStrategy = NullMassStrategy() if mass_kind in {"null", "none", "balanced"} else UOTMassStrategy()
        return FlowMatchingBackend(
            path=path,
            coupling=_build_unbalanced_coupling(coupling_config),
            mass=mass_strategy,
        )

    if policy in {"crufm", "regularized_unbalanced", "stochastic_unbalanced_cfm"}:
        sigma = _path_sigma(stage_params, path_config, default=0.05)
        return FlowMatchingBackend(
            path=RegularizedUnbalancedConditionalPath(sigma=sigma),
            coupling=_build_unbalanced_coupling(coupling_config),
            mass=UOTMassStrategy(),
        )

    if policy in {"wfrfm", "wfr_fm", "wfr_ot_fm", "wfr"}:
        sigma = _path_sigma(stage_params, path_config, default=0.0)
        coupling = _build_wfr_coupling(
            coupling_config,
            stage_params=stage_params,
            path_config=path_config,
            training_data=training_data,
        )
        delta = float(coupling.delta)
        epsilon = float(_get_first(path_config, "epsilon", default=_get_first(coupling_config, "epsilon", default=_get_first(stage_params, "epsilon", default=_WFR_EPS))))
        path = WFRTravelingGaussianPath(delta=delta, sigma=sigma, epsilon=epsilon)
        return WFRFlowMatchingBackend(
            path=path,
            coupling=coupling,
            mass=WFRTerminalMassStrategy(epsilon=epsilon),
        )

    raise ValueError(
        f"Unsupported flow_matching.backend policy '{policy}'. "
        "Supported policies: balanced_ot_cfm, sf2m, vgfm, crufm, wfrfm."
    )


class FlowMatchingBackend:
    """
    Composes coupling, path, and mass strategies into a single training backend.

    This is the package's developer-facing extension point for flow matching.

    Runtime split:
    - `coupling` defines which source-target pairs are sampled
    - `path` defines local interpolation / flow / noise / local mass dynamics
    - `mass` defines terminal per-particle mass targets

    This backend does not decide which network heads exist. Model components are
    defined separately by `config['model']['components']`, and stage-level
    regression flags are handled separately by the trainer.
    """

    def __init__(
        self,
        *,
        path: ConditionalFlowMatcher,
        coupling: CouplingStrategy,
        mass: Optional[MassStrategy] = None,
    ) -> None:
        self.path = path
        self.coupling = coupling
        self.mass = mass if mass is not None else NullMassStrategy()
        self.state: Optional[CouplingState] = None

    def prepare(
        self,
        X: list[np.ndarray],
        t_train: torch.Tensor,
        device: torch.device,
    ) -> CouplingState:
        self.state = self.coupling.build_state(X, t_train, device)
        return self.state

    def sample_batch(
        self,
        X: list[np.ndarray],
        t_train: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> FlowMatchingBatch:
        if self.state is None:
            raise RuntimeError("FlowMatchingBackend.prepare(...) must be called before sample_batch(...).")

        local_times = []
        global_times = []
        xts = []
        uts = []
        gts = []
        loss_weights_list = []
        noises = []

        for time_idx in range(len(t_train) - 1):
            pair_batch = self.coupling.sample_pairs(self.state, X, time_idx, batch_size, device)
            if pair_batch is None:
                continue

            _notify_path_interval_context(
                self.path,
                t0=t_train[time_idx],
                t1=t_train[time_idx + 1],
                time_idx=time_idx,
                device=device,
                dtype=pair_batch.x0.dtype,
            )
            t_local = self.path.sample_time(pair_batch.x0, pair_batch.x1).to(
                device=device,
                dtype=pair_batch.x0.dtype,
            )
            eps = self.path.sample_noise_like(pair_batch.x0)
            xt = self.path.sample_xt(pair_batch.x0, pair_batch.x1, t_local, eps)
            ut = self.path.compute_conditional_flow(pair_batch.x0, pair_batch.x1, t_local, xt)
            mass_result = self.mass.compute(self.state, pair_batch, t_local, device)
            gt, loss_weights = self.path.compute_conditional_mass(
                mass_result.terminal_mass,
                t_local,
            )

            time_interval = t_train[time_idx + 1] - t_train[time_idx]
            local_times.append(t_local)
            global_times.append(t_local * time_interval + t_train[time_idx])
            xts.append(xt)
            uts.append(ut / time_interval)
            gts.append(gt / time_interval)
            loss_weights_list.append(loss_weights)
            noises.append(eps)

        if not global_times:
            empty = torch.empty(0, device=device)
            empty_latent = torch.empty(0, *X[0].shape[1:], device=device)
            empty_col = torch.empty(0, 1, device=device)
            return FlowMatchingBatch(
                t_local=empty,
                t_global=empty,
                xt=empty_latent,
                ut=empty_latent,
                gt=empty_col,
                loss_weights=empty_col,
                eps=empty_latent,
            )

        return FlowMatchingBatch(
            t_local=torch.cat(local_times),
            t_global=torch.cat(global_times),
            xt=torch.cat(xts),
            ut=torch.cat(uts),
            gt=torch.cat(gts),
            loss_weights=torch.cat(loss_weights_list),
            eps=torch.cat(noises),
        )

    def compute_lambda(self, t_local: torch.Tensor) -> torch.Tensor:
        return self.path.compute_lambda(t_local)

    @classmethod
    def default_for_stage(
        cls,
        stage_params: dict[str, Any],
        *,
        regress_v: bool,
        regress_g: bool,
        regress_score: bool,
        training_data: Any | None = None,
    ) -> "FlowMatchingBackend":
        """
        Choose package-default backend pieces for one training stage.

        Important semantic note:
        - `regress_v/regress_g/regress_score` may influence which default
          backend settings are convenient for this stage
        - but they do not mean the conditional path itself is selecting which
          model components exist
        - model components come from `config['model']['components']`
        - these flags only describe which heads are trained in this stage
        """
        configured_backend = _configured_backend_for_stage(
            stage_params,
            regress_score=regress_score,
            training_data=training_data,
        )
        if configured_backend is not None:
            return configured_backend

        alpha_regm = stage_params.get("alpha_regm", 1.0)
        if regress_g or regress_v:
            coupling = UnbalancedOTCouplingStrategy(
                use_mini_batch_uot=True,
                chunk_size=1000,
                alpha_regm=alpha_regm,
                reg_strategy="max_over_time",
            )
        else:
            coupling = UnbalancedOTCouplingStrategy(
                use_mini_batch_uot=True,
                chunk_size=2000,
                alpha_regm=alpha_regm,
                reg_strategy="per_time",
            )
        return cls(
            path=RegularizedUnbalancedConditionalPath(sigma=stage_params["sigma"]),
            coupling=coupling,
            mass=UOTMassStrategy(),
        )


class WFRFlowMatchingBackend(FlowMatchingBackend):
    """Backend that constructs WFR path targets from pair-specific terminal mass before sampling xt."""

    path: WFRTravelingGaussianPath
    mass: WFRTerminalMassStrategy
    coupling: WFROETCouplingStrategy

    def sample_batch(
        self,
        X: list[np.ndarray],
        t_train: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> FlowMatchingBatch:
        if self.state is None:
            raise RuntimeError("WFRFlowMatchingBackend.prepare(...) must be called before sample_batch(...).")

        local_times = []
        global_times = []
        xts = []
        uts = []
        gts = []
        loss_weights_list = []
        noises = []

        for time_idx in range(len(t_train) - 1):
            pair_batch = self.coupling.sample_pairs(self.state, X, time_idx, batch_size, device)
            if pair_batch is None:
                continue

            _notify_path_interval_context(
                self.path,
                t0=t_train[time_idx],
                t1=t_train[time_idx + 1],
                time_idx=time_idx,
                device=device,
                dtype=pair_batch.x0.dtype,
            )
            t_local = self.path.sample_time(pair_batch.x0, pair_batch.x1).to(device=device, dtype=pair_batch.x0.dtype)
            terminal_mass = self.mass.compute(self.state, pair_batch, t_local, device).terminal_mass
            eps = self.path.sample_noise_like(pair_batch.x0)
            xt = self.path.sample_xt_wfr(pair_batch.x0, pair_batch.x1, terminal_mass, t_local, eps)
            ut = self.path.compute_conditional_flow_wfr(pair_batch.x0, pair_batch.x1, terminal_mass, t_local, xt)
            gt, loss_weights = self.path.compute_conditional_mass_wfr(pair_batch.x0, pair_batch.x1, terminal_mass, t_local)

            time_interval = t_train[time_idx + 1] - t_train[time_idx]
            local_times.append(t_local)
            global_times.append(t_local * time_interval + t_train[time_idx])
            xts.append(xt)
            uts.append(ut / time_interval)
            gts.append(gt / time_interval)
            loss_weights_list.append(loss_weights)
            noises.append(eps)

        if not global_times:
            empty = torch.empty(0, device=device)
            empty_latent = torch.empty(0, *X[0].shape[1:], device=device)
            empty_col = torch.empty(0, 1, device=device)
            return FlowMatchingBatch(
                t_local=empty,
                t_global=empty,
                xt=empty_latent,
                ut=empty_latent,
                gt=empty_col,
                loss_weights=empty_col,
                eps=empty_latent,
            )

        return FlowMatchingBatch(
            t_local=torch.cat(local_times),
            t_global=torch.cat(global_times),
            xt=torch.cat(xts),
            ut=torch.cat(uts),
            gt=torch.cat(gts),
            loss_weights=torch.cat(loss_weights_list),
            eps=torch.cat(noises),
        )


FlowMatchingBackendBuilder = Callable[
    [FlowMatchingBuildContext],
    FlowMatchingBackend,
]


def _coerce_aux_data_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: _coerce_aux_data_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_aux_data_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_coerce_aux_data_to_cpu(v) for v in value)
    return value


def _coerce_metadata_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError(f"CouplingResult.metadata must be a dict or None, got {type(metadata).__name__}")
    return {str(k): _coerce_aux_data_to_cpu(v) for k, v in metadata.items()}


def _lightweight_metadata(metadata: Any) -> dict[str, Any]:
    """Keep registry/debug metadata small; large arrays stay in sampling payloads."""

    coerced = _coerce_metadata_dict(metadata)
    lightweight: dict[str, Any] = {}
    for key, value in coerced.items():
        if isinstance(value, np.ndarray):
            lightweight[f"{key}_shape"] = list(value.shape)
            if value.size:
                lightweight[f"{key}_sum"] = float(np.asarray(value, dtype=np.float64).sum())
            continue
        lightweight[key] = value
    return lightweight


def _coerce_plan_array(plan: Any) -> np.ndarray:
    if isinstance(plan, torch.Tensor):
        return plan.detach().float().cpu().numpy()
    return np.asarray(plan, dtype=np.float32)


def _cost_matrix_to_numpy(cost_matrix: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(cost_matrix, torch.Tensor):
        return cost_matrix.detach().float().cpu().numpy()
    return np.asarray(cost_matrix, dtype=np.float32)


def _cost_matrix_to_tensor(cost_matrix: np.ndarray | torch.Tensor, device: torch.device | None) -> torch.Tensor:
    target_device = device or torch.device("cpu")
    if isinstance(cost_matrix, torch.Tensor):
        return cost_matrix.detach().float().to(target_device)
    return torch.as_tensor(cost_matrix, dtype=torch.float32, device=target_device)


def _cost_matrix_sum_count_max(cost_matrix: np.ndarray | torch.Tensor) -> tuple[float, int, float]:
    if isinstance(cost_matrix, torch.Tensor):
        cost = cost_matrix.detach().float()
        return float(cost.sum().cpu()), int(cost.numel()), float(cost.max().cpu()) if cost.numel() else 0.0
    cost = np.asarray(cost_matrix, dtype=np.float32)
    return float(cost.sum()), int(cost.size), float(cost.max(initial=0.0))


def _scale_cost_matrix(cost_matrix: np.ndarray | torch.Tensor, scale: float) -> np.ndarray | torch.Tensor:
    if isinstance(cost_matrix, torch.Tensor):
        return (cost_matrix / float(scale)).to(dtype=torch.float32)
    return (np.asarray(cost_matrix, dtype=np.float32) / float(scale)).astype(np.float32, copy=False)


def _slice_cost_matrix(
    cost_matrix: np.ndarray | torch.Tensor,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray | torch.Tensor:
    if isinstance(cost_matrix, torch.Tensor):
        src_t = torch.as_tensor(source_indices, dtype=torch.long, device=cost_matrix.device)
        tgt_t = torch.as_tensor(target_indices, dtype=torch.long, device=cost_matrix.device)
        return cost_matrix.index_select(0, src_t).index_select(1, tgt_t)
    return np.asarray(cost_matrix, dtype=np.float32)[np.ix_(source_indices, target_indices)]


def _coerce_mass_array(mass: Any, *, name: str, expected_len: int) -> np.ndarray | None:
    if mass is None:
        return None
    arr = _coerce_plan_array(mass).reshape(-1)
    if arr.shape[0] != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/inf")
    if (arr < 0).any():
        raise ValueError(f"{name} contains negative entries")
    if float(arr.sum()) <= 0:
        raise ValueError(f"{name} must have positive total mass")
    return arr.astype(np.float32, copy=False)


def _sanitize_cost_matrix(cost_matrix: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(cost_matrix, torch.Tensor):
        cost = cost_matrix.detach().float()
        if cost.ndim != 2:
            raise ValueError(f"cost_matrix must be 2D, got shape={tuple(cost.shape)}")
        if bool(torch.isnan(cost).any().cpu()):
            raise ValueError("cost_matrix contains NaN")
        finite_mask = torch.isfinite(cost)
        if not bool(finite_mask.any().cpu()):
            raise ValueError("cost_matrix contains no finite entries")
        if not bool(finite_mask.all().cpu()):
            finite_vals = cost[finite_mask]
            finite_max = float(finite_vals.max().cpu())
            fill_value = finite_max + max(1.0, abs(finite_max)) * 100.0
            cost = cost.clone()
            cost[~finite_mask] = fill_value
        return cost.to(dtype=torch.float32)

    cost = np.asarray(cost_matrix, dtype=np.float32)
    if cost.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D, got shape={cost.shape}")
    if np.isnan(cost).any():
        raise ValueError("cost_matrix contains NaN")
    finite_mask = np.isfinite(cost)
    if not finite_mask.any():
        raise ValueError("cost_matrix contains no finite entries")
    if not finite_mask.all():
        finite_vals = cost[finite_mask]
        finite_max = float(np.max(finite_vals))
        fill_value = finite_max + max(1.0, abs(finite_max)) * 100.0
        cost = cost.copy()
        cost[~finite_mask] = fill_value
    return cost.astype(np.float32, copy=False)


def _normalize_transport_mass(mass: np.ndarray | None, length: int) -> np.ndarray:
    if mass is None:
        return pot.unif(length).astype(np.float32)
    total = float(mass.sum())
    return (mass / total).astype(np.float32, copy=False)


def _split_transport_chunks(n_source: int, n_target: int, chunk_size: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n_chunks = max(1, n_source // chunk_size + 1)
    source_perm = np.arange(n_source)
    target_perm = np.arange(n_target)
    np.random.shuffle(source_perm)
    np.random.shuffle(target_perm)
    return list(np.array_split(source_perm, n_chunks)), list(np.array_split(target_perm, n_chunks))


def _coerce_pairwise_cost(
    result: PairwiseCost | np.ndarray | torch.Tensor,
    *,
    time_idx: int,
) -> PairwiseCost:
    if isinstance(result, PairwiseCost):
        pairwise_cost = result
    elif isinstance(result, (np.ndarray, torch.Tensor)):
        pairwise_cost = PairwiseCost(cost_matrix=result)
    else:
        raise TypeError(
            f"build_pairwise_cost(...) must return PairwiseCost, np.ndarray, or torch.Tensor, "
            f"got {type(result).__name__} at time_idx={time_idx}"
        )

    cost_matrix = _sanitize_cost_matrix(pairwise_cost.cost_matrix)
    source_mass = _coerce_mass_array(
        pairwise_cost.source_mass,
        name=f"source_mass(time_idx={time_idx})",
        expected_len=cost_matrix.shape[0],
    )
    target_mass = _coerce_mass_array(
        pairwise_cost.target_mass,
        name=f"target_mass(time_idx={time_idx})",
        expected_len=cost_matrix.shape[1],
    )
    return PairwiseCost(
        cost_matrix=cost_matrix,
        source_mass=source_mass,
        target_mass=target_mass,
        normalize_cost=pairwise_cost.normalize_cost,
        metadata=_coerce_metadata_dict(pairwise_cost.metadata),
    )


def _normalize_cost_if_needed(cost_matrix: np.ndarray | torch.Tensor, normalize_cost: bool) -> np.ndarray | torch.Tensor:
    if not normalize_cost:
        return cost_matrix
    if isinstance(cost_matrix, torch.Tensor):
        max_value = float(cost_matrix.detach().float().max().cpu())
        if max_value <= 0:
            return cost_matrix
        return (cost_matrix / max_value).to(dtype=torch.float32)
    max_value = float(np.max(cost_matrix))
    if max_value <= 0:
        return cost_matrix
    return (cost_matrix / max_value).astype(np.float32, copy=False)


def _calculate_auto_regularization_device(
    a: np.ndarray,
    b: np.ndarray,
    cost_matrix: np.ndarray | torch.Tensor,
    *,
    device: torch.device | None,
) -> tuple[float, float]:
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    if target_device.type != "cuda":
        return calculate_auto_regularization(a, b, _cost_matrix_to_numpy(cost_matrix))

    if int(cost_matrix.shape[0]) == 0 or int(cost_matrix.shape[1]) == 0:
        print("Cost matrix is empty – falling back to default parameters.")
        return 1e-5, 1.0

    # Use float64 here to preserve legacy CPU auto-reg choices. Float32 GPU
    # Sinkhorn can shift the elbow point enough to change downstream plans.
    a_t = torch.as_tensor(a, dtype=torch.float64, device=target_device)
    b_t = torch.as_tensor(b, dtype=torch.float64, device=target_device)
    cost_t = _cost_matrix_to_tensor(cost_matrix, target_device).to(dtype=torch.float64)

    def _stable(plan: Any, *, min_sum: float) -> bool:
        if isinstance(plan, torch.Tensor):
            return (
                bool(torch.isfinite(plan).all().detach().cpu())
                and float(plan.sum().detach().cpu()) > min_sum
                and float(plan.min().detach().cpu()) >= 0.0
            )
        plan_np = np.asarray(plan)
        return bool(np.all(np.isfinite(plan_np)) and plan_np.sum() > min_sum and plan_np.min() >= 0.0)

    fixed_reg_m = 50.0
    reg = 10.0
    eps_min, eps_step, eps_max = 5e-2, 2e-2, 10.0
    current_eps = eps_min
    first_valid_eps = None

    while current_eps <= eps_max:
        try:
            plan = pot.unbalanced.sinkhorn_unbalanced(
                a=a_t,
                b=b_t,
                M=cost_t,
                reg=current_eps,
                reg_m=[fixed_reg_m, np.inf],
            )
            if _stable(plan, min_sum=1e-8):
                first_valid_eps = current_eps
                break
        except Exception as exc:
            print(f"[GPU Round-1 eps={current_eps:.3e}] Failed: {type(exc).__name__}: {exc}")
        current_eps += eps_step

    if first_valid_eps is None:
        print("No stable eps found in coarse GPU search – keeping default reg =", reg)
    else:
        current_eps = first_valid_eps + 1e-3
        best_eps = None
        while current_eps <= eps_max:
            try:
                plan = pot.unbalanced.sinkhorn_unbalanced(
                    a=a_t,
                    b=b_t,
                    M=cost_t,
                    reg=current_eps,
                    reg_m=[fixed_reg_m, np.inf],
                )
                if _stable(plan, min_sum=1e-8):
                    best_eps = current_eps
                    break
            except Exception as exc:
                print(f"[GPU Round-2 eps={current_eps:.3e}] Failed: {type(exc).__name__}: {exc}")
            current_eps += 1e-3
        reg = best_eps if best_eps is not None else first_valid_eps
        print(f"Final entropic reg selected: {reg}")

    reg_m_candidates = np.logspace(-2, 1.2, 40)
    reg_m_list: list[float] = []
    transport_loss_list: list[float] = []
    for reg_m in reg_m_candidates:
        try:
            plan = pot.unbalanced.sinkhorn_unbalanced(
                a=a_t,
                b=b_t,
                M=cost_t,
                reg=reg,
                reg_m=[float(reg_m), np.inf],
            )
            if not _stable(plan, min_sum=1e-6):
                continue
            if isinstance(plan, torch.Tensor):
                transport_loss = float((plan * cost_t).sum().detach().cpu())
            else:
                transport_loss = float((np.asarray(plan) * _cost_matrix_to_numpy(cost_t)).sum())
            reg_m_list.append(float(reg_m))
            transport_loss_list.append(transport_loss)
        except Exception:
            continue

    if len(reg_m_list) < 4:
        best_reg_m = reg_m_list[int(np.argmin(transport_loss_list))] if reg_m_list else 1.0
        print(f"Insufficient valid reg_m candidates – using min-loss reg_m: {best_reg_m}")
    else:
        x = np.array(reg_m_list, dtype=float)
        y = np.array(transport_loss_list, dtype=float)
        x_norm = (x - x[0]) / (x[-1] - x[0] + 1e-12)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
        y0, y1 = y_norm[0], y_norm[-1]
        line_vec = np.array([1.0, y1 - y0])
        line_length = np.linalg.norm(line_vec)
        distances = np.abs(
            np.cross(
                line_vec,
                np.column_stack([x_norm - x_norm[0], y_norm - y_norm[0]]),
            )
        ) / line_length
        best_reg_m = reg_m_list[int(np.argmax(distances))]
        print(f"Elbow rule selected reg_m: {best_reg_m:.6f}")

    return float(reg), float(best_reg_m)


def _auto_regularization_for_pairwise_cost(
    pairwise_cost: PairwiseCost,
    *,
    use_mini_batch: bool,
    chunk_size: int,
    normalize_cost: bool,
    device: torch.device | None = None,
) -> tuple[float, float]:
    cost_matrix = _normalize_cost_if_needed(pairwise_cost.cost_matrix, normalize_cost)
    # The legacy UOT path calls calculate_auto_regularization with float64
    # np.ones marginals. POT's unbalanced Sinkhorn stability warnings can differ
    # for float32 marginals, which changes the selected entropic regularization.
    a = (
        np.asarray(pairwise_cost.source_mass, dtype=np.float64)
        if pairwise_cost.source_mass is not None
        else np.ones(cost_matrix.shape[0], dtype=np.float64)
    )
    b = (
        np.asarray(pairwise_cost.target_mass, dtype=np.float64)
        if pairwise_cost.target_mass is not None
        else np.ones(cost_matrix.shape[1], dtype=np.float64)
    )
    if not use_mini_batch:
        return _calculate_auto_regularization_device(a, b, cost_matrix, device=device)
    source_chunks, target_chunks = _split_transport_chunks(cost_matrix.shape[0], cost_matrix.shape[1], chunk_size)
    first_src = source_chunks[0]
    first_tgt = target_chunks[0]
    sub_cost = _slice_cost_matrix(cost_matrix, first_src, first_tgt)
    sub_a = a[first_src]
    sub_b = b[first_tgt]
    return _calculate_auto_regularization_device(sub_a, sub_b, sub_cost, device=device)


def _solve_uot_from_pairwise_cost(
    pairwise_cost: PairwiseCost,
    *,
    use_mini_batch: bool,
    chunk_size: int,
    alpha_regm: float,
    reg: float | None,
    reg_m: float | None,
    normalize_cost: bool,
    device: torch.device | None,
) -> CouplingResult:
    cost_matrix = _normalize_cost_if_needed(pairwise_cost.cost_matrix, normalize_cost or bool(pairwise_cost.normalize_cost))
    a = pairwise_cost.source_mass if pairwise_cost.source_mass is not None else np.ones(cost_matrix.shape[0], dtype=np.float32)
    b = pairwise_cost.target_mass if pairwise_cost.target_mass is not None else np.ones(cost_matrix.shape[1], dtype=np.float32)

    local_reg = reg
    local_reg_m = reg_m
    if local_reg is None or local_reg_m is None:
        auto_reg, auto_reg_m = _auto_regularization_for_pairwise_cost(
            PairwiseCost(cost_matrix=cost_matrix, source_mass=a, target_mass=b),
            use_mini_batch=use_mini_batch,
            chunk_size=chunk_size,
            normalize_cost=False,
            device=device,
        )
        if local_reg is None:
            local_reg = auto_reg
        if local_reg_m is None:
            local_reg_m = auto_reg_m

    scaled_reg_m = float(local_reg_m) * float(alpha_regm)

    if not use_mini_batch:
        target_device = device or torch.device("cpu")
        a_t = torch.as_tensor(a, dtype=torch.float32, device=target_device)
        b_t = torch.as_tensor(b, dtype=torch.float32, device=target_device)
        cost_t = _cost_matrix_to_tensor(cost_matrix, target_device)
        plan = pot.unbalanced.sinkhorn_unbalanced(a_t, b_t, cost_t, local_reg, [scaled_reg_m, np.inf])
        return CouplingResult(
            plan=plan.detach().cpu().numpy().astype(np.float32, copy=False),
            sampling_info=None,
            metadata={"solver_mode": "uot", "chunked": False, "reg": float(local_reg), "reg_m": float(scaled_reg_m)},
        )

    source_chunks, target_chunks = _split_transport_chunks(cost_matrix.shape[0], cost_matrix.shape[1], chunk_size)
    plan = np.zeros(cost_matrix.shape, dtype=np.float32)
    sub_plans: list[np.ndarray] = []
    valid_source_chunks: list[np.ndarray] = []
    valid_target_chunks: list[np.ndarray] = []
    for src_chunk, tgt_chunk in zip(source_chunks, target_chunks):
        if len(src_chunk) == 0 or len(tgt_chunk) == 0:
            continue
        sub_cost = _slice_cost_matrix(cost_matrix, src_chunk, tgt_chunk)
        sub_a = a[src_chunk]
        sub_b = b[tgt_chunk]
        target_device = device or torch.device("cpu")
        sub_plan = pot.unbalanced.sinkhorn_unbalanced(
            torch.from_numpy(sub_a).float().to(target_device),
            torch.from_numpy(sub_b).float().to(target_device),
            _cost_matrix_to_tensor(sub_cost, target_device),
            local_reg,
            [scaled_reg_m, np.inf],
        )
        sub_plan_np = sub_plan.detach().cpu().numpy().astype(np.float32, copy=False)
        plan[np.ix_(src_chunk, tgt_chunk)] = sub_plan_np
        sub_plans.append(sub_plan_np)
        valid_source_chunks.append(src_chunk)
        valid_target_chunks.append(tgt_chunk)
    sampling_info = {
        "sub_plans": sub_plans,
        "source_groups": valid_source_chunks,
        "target_groups": valid_target_chunks,
    }
    return CouplingResult(
        plan=plan,
        sampling_info=sampling_info,
        metadata={"solver_mode": "uot", "chunked": True, "reg": float(local_reg), "reg_m": float(scaled_reg_m)},
    )


def _solve_balanced_ot_from_pairwise_cost(
    pairwise_cost: PairwiseCost,
    *,
    use_mini_batch: bool,
    chunk_size: int,
    method: str,
    reg: float,
    normalize_cost: bool,
    warn: bool,
) -> CouplingResult:
    cost_matrix = _normalize_cost_if_needed(pairwise_cost.cost_matrix, normalize_cost or bool(pairwise_cost.normalize_cost))
    a = _normalize_transport_mass(pairwise_cost.source_mass, cost_matrix.shape[0])
    b = _normalize_transport_mass(pairwise_cost.target_mass, cost_matrix.shape[1])

    if not use_mini_batch:
        plan = _balanced_plan_from_cost_matrix(
            cost_matrix,
            a=a,
            b=b,
            method=method,
            reg=reg,
            warn=warn,
        )
        return CouplingResult(
            plan=plan,
            sampling_info=None,
            metadata={"solver_mode": "balanced", "chunked": False, "method": method, "reg": float(reg)},
        )

    source_chunks, target_chunks = _split_transport_chunks(cost_matrix.shape[0], cost_matrix.shape[1], chunk_size)
    plan = np.zeros(cost_matrix.shape, dtype=np.float32)
    sub_plans: list[np.ndarray] = []
    valid_source_chunks: list[np.ndarray] = []
    valid_target_chunks: list[np.ndarray] = []
    for src_chunk, tgt_chunk in zip(source_chunks, target_chunks):
        if len(src_chunk) == 0 or len(tgt_chunk) == 0:
            continue
        sub_plan = _balanced_plan_from_cost_matrix(
            _slice_cost_matrix(cost_matrix, src_chunk, tgt_chunk),
            a=_normalize_transport_mass(a[src_chunk], len(src_chunk)),
            b=_normalize_transport_mass(b[tgt_chunk], len(tgt_chunk)),
            method=method,
            reg=reg,
            warn=warn,
        )
        plan[np.ix_(src_chunk, tgt_chunk)] = sub_plan
        sub_plans.append(sub_plan)
        valid_source_chunks.append(src_chunk)
        valid_target_chunks.append(tgt_chunk)
    return CouplingResult(
        plan=plan,
        sampling_info={
            "sub_plans": sub_plans,
            "source_groups": valid_source_chunks,
            "target_groups": valid_target_chunks,
        },
        metadata={"solver_mode": "balanced", "chunked": True, "method": method, "reg": float(reg)},
    )


def _convert_wfr_gamma_to_semicouplings(
    gamma: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_mass = gamma.sum(axis=1, keepdims=True)
    col_mass = gamma.sum(axis=0, keepdims=True)
    safe_row = np.maximum(row_mass, epsilon)
    safe_col = np.maximum(col_mass, epsilon)
    gamma0 = gamma * (a[:, None] / safe_row)
    gamma1 = gamma * (b[None, :] / safe_col)
    terminal_mass = (b[None, :] * safe_row) / (np.maximum(a[:, None], epsilon) * safe_col)
    terminal_mass = np.maximum(terminal_mass.astype(np.float32, copy=False), epsilon)
    return (
        gamma0.astype(np.float32, copy=False),
        gamma1.astype(np.float32, copy=False),
        row_mass.astype(np.float32, copy=False),
        terminal_mass,
    )


def _solve_wfr_oet_block(
    pairwise_cost: PairwiseCost,
    *,
    epsilon: float,
    device: torch.device | None,
) -> CouplingResult:
    a = (
        np.asarray(pairwise_cost.source_mass, dtype=np.float64)
        if pairwise_cost.source_mass is not None
        else np.ones(pairwise_cost.cost_matrix.shape[0], dtype=np.float64)
    )
    b = (
        np.asarray(pairwise_cost.target_mass, dtype=np.float64)
        if pairwise_cost.target_mass is not None
        else np.ones(pairwise_cost.cost_matrix.shape[1], dtype=np.float64)
    )
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    if target_device.type == "cuda":
        gamma = pot.unbalanced.mm_unbalanced(
            torch.as_tensor(a, dtype=torch.float64, device=target_device),
            torch.as_tensor(b, dtype=torch.float64, device=target_device),
            _cost_matrix_to_tensor(pairwise_cost.cost_matrix, target_device).to(dtype=torch.float64),
            reg_m=[1.0, 1.0],
            reg=0.0,
            div="kl",
            numItermax=5000,
            stopThr=1e-12,
        )
    else:
        gamma = pot.unbalanced.mm_unbalanced(
            a,
            b,
            _cost_matrix_to_numpy(pairwise_cost.cost_matrix).astype(np.float64, copy=False),
            reg_m=[1.0, 1.0],
            reg=0.0,
            div="kl",
            numItermax=5000,
            stopThr=1e-12,
        )
    gamma_np = _as_numpy_array(gamma).astype(np.float64, copy=False)
    gamma_np = np.nan_to_num(gamma_np, nan=0.0, posinf=0.0, neginf=0.0)
    gamma_np = np.maximum(gamma_np, 0.0).astype(np.float32, copy=False)
    gamma0, gamma1, row_mass, terminal_mass = _convert_wfr_gamma_to_semicouplings(
        gamma_np,
        a.astype(np.float32, copy=False),
        b.astype(np.float32, copy=False),
        epsilon=epsilon,
    )
    return CouplingResult(
        plan=gamma0,
        sampling_info=None,
        metadata={
            **pairwise_cost.metadata,
            "solver_mode": "wfr_oet",
            "reg": 0.0,
            "reg_m": 1.0,
            "gamma": gamma_np,
            "gamma0": gamma0,
            "gamma1": gamma1,
            "row_mass": row_mass.reshape(-1),
            "terminal_mass_matrix": terminal_mass,
        },
    )


def _solve_wfr_oet_from_pairwise_cost(
    pairwise_cost: PairwiseCost,
    *,
    use_mini_batch: bool,
    chunk_size: int,
    epsilon: float,
    device: torch.device | None,
) -> CouplingResult:
    cost_matrix = _normalize_cost_if_needed(pairwise_cost.cost_matrix, bool(pairwise_cost.normalize_cost))
    source_mass = (
        np.asarray(pairwise_cost.source_mass, dtype=np.float32)
        if pairwise_cost.source_mass is not None
        else np.ones(cost_matrix.shape[0], dtype=np.float32)
    )
    target_mass = (
        np.asarray(pairwise_cost.target_mass, dtype=np.float32)
        if pairwise_cost.target_mass is not None
        else np.ones(cost_matrix.shape[1], dtype=np.float32)
    )
    if not use_mini_batch:
        return _solve_wfr_oet_block(
            PairwiseCost(
                cost_matrix=cost_matrix,
                source_mass=source_mass,
                target_mass=target_mass,
                normalize_cost=False,
                metadata=dict(pairwise_cost.metadata),
            ),
            epsilon=epsilon,
            device=device,
        )

    source_chunks, target_chunks = _split_transport_chunks(cost_matrix.shape[0], cost_matrix.shape[1], chunk_size)
    sub_plans: list[np.ndarray] = []
    valid_source_chunks: list[np.ndarray] = []
    valid_target_chunks: list[np.ndarray] = []
    terminal_mass_subplans: list[np.ndarray] = []
    block_metadata: list[dict[str, Any]] = []
    for src_chunk, tgt_chunk in zip(source_chunks, target_chunks):
        if len(src_chunk) == 0 or len(tgt_chunk) == 0:
            continue
        sub_result = _solve_wfr_oet_block(
            PairwiseCost(
                cost_matrix=_slice_cost_matrix(cost_matrix, src_chunk, tgt_chunk),
                source_mass=source_mass[src_chunk],
                target_mass=target_mass[tgt_chunk],
                normalize_cost=False,
                metadata=dict(pairwise_cost.metadata),
            ),
            epsilon=epsilon,
            device=device,
        )
        sub_plans.append(np.asarray(sub_result.plan, dtype=np.float32))
        valid_source_chunks.append(src_chunk)
        valid_target_chunks.append(tgt_chunk)
        terminal_mass_subplans.append(np.asarray(sub_result.metadata["terminal_mass_matrix"], dtype=np.float32))
        block_metadata.append(_lightweight_metadata(sub_result.metadata))

    return CouplingResult(
        plan=np.empty((0, 0), dtype=np.float32),
        sampling_info={
            "sub_plans": sub_plans,
            "source_groups": valid_source_chunks,
            "target_groups": valid_target_chunks,
            "terminal_mass_subplans": terminal_mass_subplans,
        },
        metadata={
            "solver_mode": "wfr_oet",
            "chunked": True,
            "reg": 0.0,
            "reg_m": 1.0,
            "block_metadata": block_metadata,
        },
    )


def _balanced_plan_from_cost_matrix(
    cost_matrix: np.ndarray | torch.Tensor,
    *,
    a: np.ndarray,
    b: np.ndarray,
    method: str,
    reg: float,
    warn: bool,
) -> np.ndarray:
    if method == "exact":
        # POT accepts torch arrays for ot.emd, but its network-simplex EMD
        # implementation uses the C++ CPU backend. Keep exact EMD explicit on
        # CPU to avoid pretending GPU arrays accelerate this path.
        plan = pot.emd(a, b, _cost_matrix_to_numpy(cost_matrix))
    elif method == "sinkhorn":
        if isinstance(cost_matrix, torch.Tensor):
            target_device = cost_matrix.device
            plan = pot.sinkhorn(
                torch.as_tensor(a, dtype=torch.float32, device=target_device),
                torch.as_tensor(b, dtype=torch.float32, device=target_device),
                cost_matrix.detach().float().to(target_device),
                reg=reg,
                warn=warn,
            )
        else:
            plan = pot.sinkhorn(a, b, cost_matrix, reg=reg, warn=warn)
    else:
        raise ValueError(f"Unsupported balanced solver method for cost-based coupling: {method}")
    plan_np = _as_numpy_array(plan).astype(np.float32, copy=False)
    plan_sum = float(np.nansum(plan_np)) if plan_np.size else 0.0
    if (not np.isfinite(plan_np).all()) or plan_sum <= 1e-8:
        if warn:
            import warnings

            warnings.warn(
                "Numerical errors in balanced OT plan; reverting to outer-product uniform coupling.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Match the legacy OTPlanSampler fallback: when Sinkhorn underflows or
        # returns a zero plan, keep training/evaluation alive with a valid
        # balanced independent coupling instead of producing empty samples.
        plan_np = np.outer(
            np.asarray(a, dtype=np.float32),
            np.asarray(b, dtype=np.float32),
        ).astype(np.float32, copy=False)
    return plan_np


def build_flow_matching_backend(
    builder: FlowMatchingBackendBuilder,
    *,
    build_context: FlowMatchingBuildContext,
) -> FlowMatchingBackend:
    """
    Build a flow-matching backend from the standard builder context.
    """
    return builder(build_context)


def default_flow_matching_backend_builder(
    build_context: FlowMatchingBuildContext,
) -> FlowMatchingBackend:
    return FlowMatchingBackend.default_for_stage(
        build_context.stage_params,
        regress_v=build_context.regress_v,
        regress_g=build_context.regress_g,
        regress_score=build_context.regress_score,
        training_data=build_context.training_data,
    )
