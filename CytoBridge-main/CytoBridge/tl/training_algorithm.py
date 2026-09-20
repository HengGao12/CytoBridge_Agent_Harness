from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch


@dataclass
class TrainingAlgorithmContext:
    """Developer-facing context passed into custom training algorithm entrypoints."""

    algorithm_id: str
    input_adata_path: str
    output_dir: str
    stage: str  # "pilot" | "final"
    base_config_name: Optional[str]
    resolved_base_config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingDataBuilderContext:
    """Context passed to custom training-data builders."""

    adata: Any
    resolved_config: dict[str, Any]
    stage: str
    device: str
    output_dir: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingDataBundle:
    """
    Canonical training inputs for the standard runtime.

    `latent_by_time` must remain aligned to `time_points`, and each
    `obs_indices_by_time[i]` must index the rows in `adata` corresponding to
    `latent_by_time[i]`.
    """

    adata: Any
    time_points: list[float]
    latent_by_time: list[torch.Tensor]
    obs_indices_by_time: list[Any]
    extra_modalities_by_time: dict[str, list[Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowMatchingBuildContext:
    """
    Runtime context passed into flow-matching backend builders.

    Semantic note:
    - `regress_v/regress_g/regress_score` are stage-level training flags
    - they tell the builder which losses are active in this stage
    - they do not redefine model components, mass semantics, or path semantics
      by themselves
    - builders may use them to choose different backend defaults, but that is a
      builder policy choice rather than a statement that the conditional path
      "contains" `v/g/s`
    """

    stage_params: dict[str, Any]
    training_data: TrainingDataBundle
    device: torch.device
    regress_v: bool
    regress_g: bool
    regress_score: bool
    model: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowMatchingLossContext:
    """Runtime context passed into additive flow-matching loss hooks."""

    stage_params: dict[str, Any]
    training_data: TrainingDataBundle
    backend: Any
    batch: Any
    model: Any
    net_input: torch.Tensor
    lambda_t: torch.Tensor
    regress_v: bool
    regress_g: bool
    regress_score: bool
    x_input: torch.Tensor | None = None
    t_input: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowMatchingLossResult:
    """Differentiable loss adjustment returned by a custom flow-matching hook.

    `extra_loss` is additive. The `replace_*_loss` fields explicitly replace
    the corresponding builtin flow-matching component loss when that component
    is active, which avoids fragile `custom_loss - builtin_loss` arithmetic in
    custom algorithms.
    """

    extra_loss: torch.Tensor | None = None
    replace_velocity_loss: torch.Tensor | None = None
    replace_growth_loss: torch.Tensor | None = None
    replace_score_loss: torch.Tensor | None = None
    logs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelBuildContext:
    """Runtime context passed into custom model builders."""

    algorithm_id: str
    resolved_config: dict[str, Any]
    latent_dim: int
    training_data: TrainingDataBundle
    device: torch.device
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSerializeContext:
    """Runtime context passed into custom model serializers."""

    algorithm_id: str
    model: torch.nn.Module
    resolved_config: dict[str, Any]
    latent_dim: int
    training_data: TrainingDataBundle
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelLoadContext:
    """Runtime context passed into custom model loaders."""

    algorithm_id: str
    adata: Any
    resolved_config: dict[str, Any]
    latent_dim: int
    device: torch.device
    model_payload: dict[str, Any] = field(default_factory=dict)
    model_state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationTimepointResult:
    """
    Reusable inference/evaluation artifact for one predicted future time point.

    This lets custom metrics reuse the same simulated trajectory outputs that
    builtin `W1/TMV` already consumed, instead of re-running inference.
    """

    time_index: int
    time_point: float
    observed_points: Any
    predicted_points: Any
    predicted_weights: Any
    normalized_predicted_weights: Any
    observed_uniform_weights: Any
    w1: float
    tmv: float
    trajectory_time_index: int | None = None
    trajectory_time_point: float | None = None


@dataclass
class EvaluationTrajectory:
    """
    Canonical model-generated evaluation trajectory.

    The trajectory is generated once from t0 particles to the final requested
    time point. Builtin metrics and additive custom metrics must read from this
    artifact rather than running separate prediction or metric-specific logic.

    `points_by_time` may contain a different number of particles at each time
    slice. This supports both weighted-particle rollouts and explicit
    birth/death/splitting simulators. `weights_by_time` must match each slice's
    particle count. TMV compares predicted total mass relative to the first
    observed trajectory slice, so unit-weight explicit splitting is allowed when
    the initial slice also uses unit weights.
    """

    time_points: list[float]
    points_by_time: list[Any]
    weights_by_time: list[Any]
    observed_time_points: list[float]
    observed_time_indices: list[int]
    source: str = "model_rollout"
    step: float | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def observed_points_by_time(self) -> list[Any]:
        return [self.points_by_time[idx] for idx in self.observed_time_indices]

    def observed_weights_by_time(self) -> list[Any]:
        return [self.weights_by_time[idx] for idx in self.observed_time_indices]


@dataclass
class EvaluationMetricsContext:
    """
    Runtime context passed into additive evaluation-metrics hooks.

    Builtin evaluation metrics (`w1_scores`, `tmv_scores`) are computed by the
    package first and must remain immutable. Custom hooks may only append
    additional metrics and diagnostics.
    """

    adata: Any
    training_data: TrainingDataBundle
    model: Any
    config: dict[str, Any]
    data: list[torch.Tensor]
    time_points: list[float]
    builtin_metrics: dict[str, Any]
    metric_params: dict[str, Any]
    simulated_points_by_time: list[Any]
    simulated_weights_by_time: list[Any]
    timepoint_results: list[EvaluationTimepointResult]
    device: torch.device
    evaluation_trajectory: EvaluationTrajectory | None = None
    trajectory_time_points: list[float] = field(default_factory=list)
    trajectory_points_by_time: list[Any] = field(default_factory=list)
    trajectory_weights_by_time: list[Any] = field(default_factory=list)
    observed_time_indices: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceContextBuilderContext:
    """
    Context for constructing evaluation-time inference inputs.

    The builder is intentionally flexible: algorithms may place any required
    inference payload into `InferenceContext.payload`, but the source should be
    documented in provenance so reviewers can verify that rollout inputs come
    from t=0 cells, known exogenous conditions, constants, or model state.
    """

    t0_adata: Any
    initial_data: Any
    initial_obs_indices: Any
    config: dict[str, Any]
    time_points: list[float]
    device: torch.device | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceContext:
    """
    Free-form evaluation-time input envelope.

    Fixed fields are minimal by design:
    - `payload` can contain arbitrary algorithm-specific inputs
    - `visibility` declares allowed source class per payload entry
    - `provenance` records how the payload was built
    """

    payload: dict[str, Any] = field(default_factory=dict)
    visibility: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class SimulationContext:
    """Runtime context passed into custom evaluation-time simulators/predictors."""

    adata: Any
    model: torch.nn.Module
    training_data: TrainingDataBundle
    config: dict[str, Any]
    time_points: list[float]
    observed_time_points: list[float] = field(default_factory=list)
    trajectory_time_points: list[float] = field(default_factory=list)
    trajectory_step: float | None = None
    x0_override: Any = None
    device: torch.device | None = None
    inference_context: InferenceContext | dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Prediction payload returned by a custom simulator/predictor.

    Return a full t0-to-final trajectory. Each time slice may have its own
    particle count as long as its weights have the same length. Weighted
    rollouts usually start with weights summing to 1. Explicit birth/death or
    splitting simulators may return unit weights; builtin TMV normalizes by the
    first predicted total mass before comparing to observed cell-count ratios.
    """

    predicted_points_by_time: list[Any]
    predicted_weights_by_time: list[Any]
    trajectory_time_points: list[float] = field(default_factory=list)
    aux_state_by_time: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageRunnerContext:
    """Runtime context passed into custom stage runners."""

    model: torch.nn.Module
    config: dict[str, Any]
    stage_params: dict[str, Any]
    stage_index: int
    training_data: TrainingDataBundle
    device: torch.device
    batch_size: int
    output_dir: str
    progress_callback: Optional[Callable[[str, float], None]] = None
    should_stop: Optional[Callable[[], bool]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    preview_only: bool = False


@dataclass
class StageRunnerResult:
    """Standardized result returned by a custom stage runner."""

    stage_summary: dict[str, Any] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    model_outputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingAlgorithmSpec:
    """
    Developer-facing training algorithm contract returned by custom entrypoints.

    `evaluation_metrics_hook` is additive only:
    - builtin `w1_scores` and `tmv_scores` remain package-defined
    - custom algorithms may append extra metrics, but may not replace builtin
      evaluation semantics
    - `evaluation_metrics_params` is the explicit channel for evaluation-only
      thresholds, priors, and other custom metric parameters
    """

    algorithm_id: str
    base_config: str | dict[str, Any]
    config_overrides: dict[str, Any] = field(default_factory=dict)
    training_data_builder: Callable[[TrainingDataBuilderContext], TrainingDataBundle] | None = None
    flow_matching_backend_builder: Callable[[FlowMatchingBuildContext], Any] | None = None
    flow_matching_loss_hook: Callable[[FlowMatchingLossContext], Any] | None = None
    evaluation_metrics_hook: Callable[[EvaluationMetricsContext], dict[str, Any] | None] | None = None
    model_builder: Callable[[ModelBuildContext], torch.nn.Module] | None = None
    serialize_model: Callable[[ModelSerializeContext], dict[str, Any]] | None = None
    deserialize_model: Callable[[ModelLoadContext], torch.nn.Module] | None = None
    stage_runner: Callable[[StageRunnerContext], StageRunnerResult | None] | None = None
    inference_context_builder: Callable[[InferenceContextBuilderContext], InferenceContext | dict[str, Any] | None] | None = None
    simulation_hook: Callable[[SimulationContext], SimulationResult | dict[str, Any]] | None = None
    evaluation_metrics_params: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
