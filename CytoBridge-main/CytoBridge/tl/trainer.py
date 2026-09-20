import torch
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from CytoBridge.tl.methods import neural_ode_step, ODEFunc
from CytoBridge.utils.utils import sample,trace_df_dz,compute_integral
from CytoBridge.tl.losses import calc_ot_loss, calc_mass_loss, calc_score_matching_loss,Density_loss,calc_pinn_loss
from CytoBridge.tl import methods
from CytoBridge.tl.models import DynamicalModel
from CytoBridge.tl.flow_matching_backends import (
    FlowMatchingBackend,
    build_flow_matching_backend,
    default_flow_matching_backend_builder,
)
from CytoBridge.tl.w1_backend import (
    compute_w1_distance,
    select_w1_backend_policy,
    w1_backend_metric_metadata,
)
from CytoBridge.tl.training_algorithm import (
    EvaluationTimepointResult,
    EvaluationTrajectory,
    EvaluationMetricsContext,
    FlowMatchingBuildContext,
    FlowMatchingLossContext,
    FlowMatchingLossResult,
    InferenceContext,
    InferenceContextBuilderContext,
    SimulationContext,
    SimulationResult,
    StageRunnerContext,
    StageRunnerResult,
    TrainingDataBundle,
)
from CytoBridge.pl.plot import plot_interaction_potential_epoch
from CytoBridge.tl.analysis import simulate_trajectory
import math
import os
import json
import signal
import ctypes
import threading
import time
import ot
from pathlib import Path
from torchdiffeq import odeint
from torch.optim.lr_scheduler import StepLR  # Import StepLR scheduler

TRAINING_TIME_BUDGET_CELL_BLOCK = 5000
TRAINING_TIME_BUDGET_SECONDS_PER_CELL_BLOCK = 90
TRAINING_TIME_BUDGET_BASE_SECONDS = 60
TRAINING_TIME_BUDGET_GAP_OVERHEAD_SECONDS = 64
TRAINING_TIME_BUDGET_SAFETY_MULTIPLIER = 1.33
TRAINING_TIME_BUDGET_ROUNDING_SECONDS = 30
TRAINING_TIME_BUDGET_MIN_SECONDS = 120
INFERENCE_TIME_BUDGET_CELL_BLOCK = 5000
INFERENCE_TIME_BUDGET_SECONDS_PER_CELL_BLOCK = 25
INFERENCE_TIME_BUDGET_BASE_SECONDS = 10
INFERENCE_TIME_BUDGET_GAP_OVERHEAD_SECONDS = 5
INFERENCE_TIME_BUDGET_SAFETY_MULTIPLIER = 1.15
INFERENCE_TIME_BUDGET_ROUNDING_SECONDS = 5
INFERENCE_TIME_BUDGET_MIN_SECONDS = 15
INFERENCE_TIME_BUDGET_TINY_EFFECTIVE_CELL_MAX = 5000
INFERENCE_TIME_BUDGET_TINY_MAX_SECONDS = 30
INFERENCE_TIME_BUDGET_MAX_SECONDS = 150
EVALUATION_TRAJECTORY_MAX_SLICES = 501
EVALUATION_TRAJECTORY_MAX_UNCOMPRESSED_MB = 1024


def _clone_state_dict_tensors(state_dict):
    """Deep-copy a PyTorch state dict so later optimizer steps cannot mutate it."""

    return {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in dict(state_dict).items()
    }


def _env_float(name, default):
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name, default):
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


class InferenceTimeoutError(TimeoutError):
    """Raised when evaluation/inference exceeds its wall-clock budget."""


def _raise_in_thread(thread_id, exc_type):
    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(int(thread_id)),
        ctypes.py_object(exc_type),
    )
    if result > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(int(thread_id)), None)


class _InferenceTimeoutGuard:
    def __init__(self, timeout_sec):
        self.timeout_sec = float(timeout_sec or 0.0)
        self._old_handler = None
        self._timer = None
        self._use_signal = False
        self._thread_id = None

    def __enter__(self):
        if self.timeout_sec <= 0:
            return self
        self._thread_id = threading.get_ident()
        self._use_signal = (
            threading.current_thread() is threading.main_thread()
            and hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
        )
        if self._use_signal:
            self._old_handler = signal.getsignal(signal.SIGALRM)

            def _handler(signum, frame):
                del signum, frame
                raise InferenceTimeoutError(f"Inference timed out after {self.timeout_sec:.3g}s")

            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, self.timeout_sec)
        else:
            self._timer = threading.Timer(
                self.timeout_sec,
                _raise_in_thread,
                args=(self._thread_id, InferenceTimeoutError),
            )
            self._timer.daemon = True
            self._timer.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._use_signal:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
        return False


class TrainingPipeline:
    def __init__(
        self,
        model,
        config,
        batch_size,
        device,
        training_data: TrainingDataBundle,
        progress_callback=None,
        flow_matching_backend_builder=None,
        flow_matching_loss_hook=None,
        evaluation_metrics_hook=None,
        evaluation_metrics_params=None,
        stage_runner=None,
        inference_context_builder=None,
        simulation_hook=None,
        prepared_flow_matching_backend=None,
        prepared_flow_matching_stage_name=None,
        prepared_flow_matching_stage_params=None,
    ):
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.optimizer = None
        self.scheduler = None  # Initialize scheduler variable
        self.device = device
        self.model.to(device)
        self.progress_callback = progress_callback  # Callback for progress updates
        # Determine if mass component is used based on model configuration
        self.use_mass = 'growth' in self.config['model']['components']
        # Determine if score component is used based on model configuration
        self.use_score = 'score' in self.config['model']['components']
        # Determine if interaction component is used based on model configuration
        self.use_interaction = 'interaction' in self.config['model']['components']

        # Initialize ODE function (unified gradient calculation entry)
        self.ode_func = ODEFunc(
            model=self.model,
            sigma=config['training']['defaults'].get('sigma', 0.05),
            use_mass=self.use_mass,
            score_use=self.use_score,
            interaction_use=self.use_interaction
        )

        # New: Initialize variables required for train_score_model
        self.logger = self._setup_logger()  # Simple logger implementation
        # Get experiment directory from configuration (default to './results' if not specified)
        self.exp_dir = self.config.get('ckpt_dir', './results')
        os.makedirs(self.exp_dir, exist_ok=True)
        self.training_data = training_data
        self.adata = training_data.adata
        # Construct DataFrame from input data to fit the format required by train_score_model
        self.df = self._prepare_df(training_data.latent_by_time)
        # Get sorted list of unique time points (grouped by 'samples' column)
        self.groups = sorted(self.df.samples.unique())
        self.flow_matching_backend_builder = flow_matching_backend_builder
        self.flow_matching_loss_hook = flow_matching_loss_hook
        self.evaluation_metrics_hook = evaluation_metrics_hook
        self.evaluation_metrics_params = dict(evaluation_metrics_params or {})
        self.stage_runner = stage_runner
        self.inference_context_builder = inference_context_builder
        self.simulation_hook = simulation_hook
        self.prepared_flow_matching_backend = prepared_flow_matching_backend
        self.prepared_flow_matching_stage_name = prepared_flow_matching_stage_name
        self.prepared_flow_matching_stage_params = prepared_flow_matching_stage_params
        self.training_summary = {"stages": []}
        self._training_started_at = None
        self._inference_started_at = None
        self.training_time_budget = self._build_training_time_budget()
        self.inference_time_budget = self._build_inference_time_budget()
        self._refresh_time_budget_summary()
        self._refresh_inference_time_budget_summary()

    def _timepoint_cell_counts(self, data=None):
        counts = []
        source = data if data is not None else getattr(self.training_data, "latent_by_time", [])
        for tensor in list(source or []):
            try:
                counts.append(int(tensor.shape[0]))
            except Exception:
                counts.append(0)
        return counts

    def _build_training_time_budget(self):
        cell_count = int(getattr(self.adata, "n_obs", 0) or 0)
        timepoint_counts = self._timepoint_cell_counts()
        if timepoint_counts:
            cell_count = int(sum(timepoint_counts))
        gap_effective_counts = [
            max(1, min(int(timepoint_counts[i]), int(timepoint_counts[i + 1])))
            for i in range(max(0, len(timepoint_counts) - 1))
        ]
        gap_count = len(gap_effective_counts)
        effective_cell_count = int(sum(gap_effective_counts)) if gap_effective_counts else cell_count
        gap_cell_units = [
            float(count) / float(TRAINING_TIME_BUDGET_CELL_BLOCK)
            for count in gap_effective_counts
        ]
        linear_cell_budget_sec = sum(
            unit * float(TRAINING_TIME_BUDGET_SECONDS_PER_CELL_BLOCK)
            for unit in gap_cell_units
        )
        gap_overhead_sec = sum(
            math.sqrt(max(unit, 0.0)) * float(TRAINING_TIME_BUDGET_GAP_OVERHEAD_SECONDS)
            for unit in gap_cell_units
        )
        raw_budget_sec = (
            float(TRAINING_TIME_BUDGET_BASE_SECONDS)
            + linear_cell_budget_sec
            + gap_overhead_sec
        ) * float(TRAINING_TIME_BUDGET_SAFETY_MULTIPLIER)
        rounding = max(1, int(TRAINING_TIME_BUDGET_ROUNDING_SECONDS))
        budget_sec = int(max(
            TRAINING_TIME_BUDGET_MIN_SECONDS,
            math.ceil(raw_budget_sec / rounding) * rounding,
        ))
        return {
            "enabled": True,
            "budget_rule": "continuous_adjacent_gap_bottleneck_cells",
            "cell_count": cell_count,
            "timepoint_count": len(timepoint_counts),
            "gap_count": gap_count,
            "timepoint_cell_counts": timepoint_counts,
            "gap_effective_cell_counts": gap_effective_counts,
            "gap_cell_units": gap_cell_units,
            "effective_cell_count": effective_cell_count,
            "cell_block_size": TRAINING_TIME_BUDGET_CELL_BLOCK,
            "seconds_per_cell_block": TRAINING_TIME_BUDGET_SECONDS_PER_CELL_BLOCK,
            "base_seconds": TRAINING_TIME_BUDGET_BASE_SECONDS,
            "gap_overhead_seconds": TRAINING_TIME_BUDGET_GAP_OVERHEAD_SECONDS,
            "linear_cell_budget_sec": round(float(linear_cell_budget_sec), 3),
            "gap_overhead_sec": round(float(gap_overhead_sec), 3),
            "safety_multiplier": TRAINING_TIME_BUDGET_SAFETY_MULTIPLIER,
            "raw_budget_sec": round(float(raw_budget_sec), 3),
            "rounding_seconds": rounding,
            "min_budget_sec": TRAINING_TIME_BUDGET_MIN_SECONDS,
            "budget_sec": budget_sec,
            "elapsed_sec": 0.0,
            "timed_out": False,
            "completed_epochs": 0,
            "last_stage_name": "",
            "last_stage_mode": "",
            "last_epoch": 0,
            "last_total_epochs": 0,
            "message": "",
        }

    def _build_inference_time_budget(self, data=None):
        cell_count = int(getattr(self.adata, "n_obs", 0) or 0)
        timepoint_counts = self._timepoint_cell_counts(data=data)
        if timepoint_counts:
            cell_count = int(sum(timepoint_counts))
        gap_effective_counts = [
            max(1, min(int(timepoint_counts[i]), int(timepoint_counts[i + 1])))
            for i in range(max(0, len(timepoint_counts) - 1))
        ]
        gap_count = len(gap_effective_counts)
        effective_cell_count = int(sum(gap_effective_counts)) if gap_effective_counts else cell_count
        cell_units = float(effective_cell_count) / float(INFERENCE_TIME_BUDGET_CELL_BLOCK)
        linear_cell_budget_sec = cell_units * float(INFERENCE_TIME_BUDGET_SECONDS_PER_CELL_BLOCK)
        gap_overhead_sec = float(gap_count) * float(INFERENCE_TIME_BUDGET_GAP_OVERHEAD_SECONDS)
        raw_budget_sec = (
            float(INFERENCE_TIME_BUDGET_BASE_SECONDS)
            + linear_cell_budget_sec
            + gap_overhead_sec
        )
        scaled_budget_sec = raw_budget_sec * float(INFERENCE_TIME_BUDGET_SAFETY_MULTIPLIER)
        rounding = max(1, int(INFERENCE_TIME_BUDGET_ROUNDING_SECONDS))
        budget_sec = float(
            max(
                INFERENCE_TIME_BUDGET_MIN_SECONDS,
                math.ceil(scaled_budget_sec / rounding) * rounding,
            )
        )
        cap_sec = float(INFERENCE_TIME_BUDGET_MAX_SECONDS)
        if 0 < effective_cell_count <= int(INFERENCE_TIME_BUDGET_TINY_EFFECTIVE_CELL_MAX):
            cap_sec = min(cap_sec, float(INFERENCE_TIME_BUDGET_TINY_MAX_SECONDS))
        budget_sec = min(budget_sec, cap_sec)
        configured = (
            (self.config.get("evaluation") or {}).get("inference_timeout_sec")
            if isinstance(self.config.get("evaluation"), dict)
            else None
        )
        if configured is not None:
            try:
                configured_value = float(configured)
            except Exception:
                configured_value = 0.0
            if configured_value > 0:
                budget_sec = min(budget_sec, configured_value)
        return {
            "enabled": True,
            "budget_rule": "strict_adjacent_gap_bottleneck_cells",
            "cell_count": cell_count,
            "timepoint_count": len(timepoint_counts),
            "gap_count": gap_count,
            "timepoint_cell_counts": timepoint_counts,
            "gap_effective_cell_counts": gap_effective_counts,
            "effective_cell_count": effective_cell_count,
            "cell_block_size": INFERENCE_TIME_BUDGET_CELL_BLOCK,
            "seconds_per_cell_block": INFERENCE_TIME_BUDGET_SECONDS_PER_CELL_BLOCK,
            "base_seconds": INFERENCE_TIME_BUDGET_BASE_SECONDS,
            "gap_overhead_seconds": INFERENCE_TIME_BUDGET_GAP_OVERHEAD_SECONDS,
            "linear_cell_budget_sec": round(float(linear_cell_budget_sec), 3),
            "gap_overhead_sec": round(float(gap_overhead_sec), 3),
            "raw_budget_sec": round(float(raw_budget_sec), 3),
            "safety_multiplier": INFERENCE_TIME_BUDGET_SAFETY_MULTIPLIER,
            "scaled_budget_sec": round(float(scaled_budget_sec), 3),
            "rounding_seconds": rounding,
            "min_budget_sec": INFERENCE_TIME_BUDGET_MIN_SECONDS,
            "tiny_effective_cell_max": INFERENCE_TIME_BUDGET_TINY_EFFECTIVE_CELL_MAX,
            "tiny_max_budget_sec": INFERENCE_TIME_BUDGET_TINY_MAX_SECONDS,
            "max_budget_sec": INFERENCE_TIME_BUDGET_MAX_SECONDS,
            "budget_sec": round(float(budget_sec), 3),
            "elapsed_sec": 0.0,
            "timed_out": False,
            "message": "",
        }

    def _start_training_timer(self):
        if self._training_started_at is None:
            self._training_started_at = time.monotonic()
            self._refresh_time_budget_summary()

    def _start_inference_timer(self):
        self._inference_started_at = time.monotonic()
        self._refresh_inference_time_budget_summary()

    def _refresh_time_budget_summary(self):
        if self._training_started_at is not None:
            elapsed = max(0.0, time.monotonic() - self._training_started_at)
            self.training_time_budget["elapsed_sec"] = round(float(elapsed), 3)
        self.training_summary["time_budget"] = dict(self.training_time_budget)

    def _refresh_inference_time_budget_summary(self):
        if self._inference_started_at is not None:
            elapsed = max(0.0, time.monotonic() - self._inference_started_at)
            self.inference_time_budget["elapsed_sec"] = round(float(elapsed), 3)
        self.training_summary["inference_time_budget"] = dict(self.inference_time_budget)

    def _record_completed_epoch(self, stage_params, epoch_index, total_epochs):
        self.training_time_budget["completed_epochs"] = int(self.training_time_budget.get("completed_epochs", 0)) + 1
        self.training_time_budget["last_stage_name"] = str(stage_params.get("name", ""))
        self.training_time_budget["last_stage_mode"] = str(stage_params.get("mode", ""))
        self.training_time_budget["last_epoch"] = int(epoch_index) + 1
        self.training_time_budget["last_total_epochs"] = int(total_epochs)
        self._refresh_time_budget_summary()

    def _time_budget_exceeded(self):
        self._refresh_time_budget_summary()
        if self.training_time_budget.get("timed_out"):
            return True
        budget_sec = float(self.training_time_budget.get("budget_sec") or 0.0)
        elapsed_sec = float(self.training_time_budget.get("elapsed_sec") or 0.0)
        if budget_sec > 0 and elapsed_sec >= budget_sec:
            message = (
                "Training time budget reached "
                f"after {elapsed_sec:.1f}s/{budget_sec:.1f}s "
                f"for {self.training_time_budget.get('effective_cell_count')} effective gap cells "
                f"across {self.training_time_budget.get('gap_count')} gaps "
                f"({self.training_time_budget.get('cell_count')} total cells); "
                f"stopped after epoch {self.training_time_budget.get('last_epoch')}/"
                f"{self.training_time_budget.get('last_total_epochs')} "
                f"in stage '{self.training_time_budget.get('last_stage_name')}'."
            )
            self.training_time_budget["timed_out"] = True
            self.training_time_budget["message"] = message
            print(f"[WARN] {message}")
            if self.progress_callback:
                self.progress_callback(message, 1.0)
            self._refresh_time_budget_summary()
            return True
        return False

    def _mark_inference_timeout(self):
        self._refresh_inference_time_budget_summary()
        elapsed_sec = float(self.inference_time_budget.get("elapsed_sec") or 0.0)
        budget_sec = float(self.inference_time_budget.get("budget_sec") or 0.0)
        if not self.inference_time_budget.get("message"):
            self.inference_time_budget["message"] = (
                "Inference time budget reached "
                f"after {elapsed_sec:.1f}s/{budget_sec:.1f}s "
                f"for {self.inference_time_budget.get('effective_cell_count')} effective gap cells "
                f"across {self.inference_time_budget.get('gap_count')} gaps "
                f"({self.inference_time_budget.get('cell_count')} total cells); "
                "stopped evaluation before producing trusted metrics."
            )
        self.inference_time_budget["timed_out"] = True
        self._refresh_inference_time_budget_summary()
        message = str(self.inference_time_budget.get("message") or "")
        print(f"[WARN] {message}")
        if self.progress_callback:
            self.progress_callback(message, 1.0)
        return message

    def _raise_if_inference_time_budget_exceeded(self):
        self._refresh_inference_time_budget_summary()
        budget_sec = float(self.inference_time_budget.get("budget_sec") or 0.0)
        elapsed_sec = float(self.inference_time_budget.get("elapsed_sec") or 0.0)
        if budget_sec > 0 and elapsed_sec >= budget_sec:
            raise InferenceTimeoutError(f"Inference timed out after {budget_sec:.3g}s")


    def _setup_logger(self):
        """Simple logger implementation to replace the original logger"""

        class SimpleLogger:
            @staticmethod
            def info(msg):
                print(f"[INFO] {msg}")

        return SimpleLogger()

    def _prepare_df(self, data):
        """Construct DataFrame from input data to fit the format required by train_score_model
        
        Args:
            data: List of tensors where each element represents samples at a specific time point (shape: n_samples×2)
        
        Returns:
            pd.DataFrame: Combined DataFrame with columns 'x1', 'x2', and 'samples' (time point)
        """
        all_samples = []
        for t_idx, x in enumerate(data):
            x_np = x.cpu().detach().numpy()  # Convert tensor to numpy array
            # Construct DataFrame for current time point: columns = [x1, x2, samples (time point)]
            df_t = pd.DataFrame({
                'x1': x_np[:, 0],
                'x2': x_np[:, 1],
                # Assign current time point to all samples (dtype: float64 for consistency)
                'samples': np.full(x_np.shape[0], t_idx, dtype=np.float64)
            })
            all_samples.append(df_t)
        # Concatenate DataFrames from all time points and reset index
        return pd.concat(all_samples, ignore_index=True)

    def _get_module_by_path(self, path):
        current = self.model
        for part in str(path).split("."):
            if not part:
                continue
            if isinstance(current, (torch.nn.Sequential, torch.nn.ModuleList)) and part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        if not isinstance(current, torch.nn.Module):
            raise TypeError(f"Trainable module path {path!r} did not resolve to torch.nn.Module")
        return current

    @staticmethod
    def _as_module_spec_list(value):
        if value is None:
            return []
        if isinstance(value, (str, torch.nn.Module, torch.nn.Parameter)):
            return [value]
        if isinstance(value, dict):
            specs = []
            for item in value.values():
                specs.extend(TrainingPipeline._as_module_spec_list(item))
            return specs
        try:
            return list(value)
        except TypeError:
            return [value]

    @staticmethod
    def _component_selected(key, flags):
        normalized = str(key).strip().lower()
        aliases = {
            "v": "velocity",
            "vel": "velocity",
            "velocity_net": "velocity",
            "g": "growth",
            "mass": "growth",
            "growth_net": "growth",
            "s": "score",
            "score_net": "score",
            "i": "interaction",
            "interaction_net": "interaction",
            "all": "always",
            "always": "always",
            "extra": "always",
        }
        component = aliases.get(normalized, normalized)
        if component == "always":
            return True
        return bool(flags.get(component, False))

    def _collect_component_module_specs(self, flags, stage_params):
        specs = []
        defaults = {
            "velocity": ["velocity_net"],
            "growth": ["growth_net"],
            "score": ["score_net"],
            "interaction": ["interaction_net"],
        }
        for component, names in defaults.items():
            if flags.get(component):
                specs.extend(name for name in names if hasattr(self.model, name))

        model_map = getattr(self.model, "cytobridge_component_modules", None)
        if model_map is None:
            model_map = getattr(self.model, "component_trainable_modules", None)
        if callable(model_map):
            model_map = model_map(stage_params=dict(stage_params), train_flags=dict(flags))
        if isinstance(model_map, dict):
            for component, value in model_map.items():
                if self._component_selected(component, flags):
                    specs.extend(self._as_module_spec_list(value))

        stage_map = (
            stage_params.get("component_trainable_modules")
            or stage_params.get("cytobridge_component_modules")
            or {}
        )
        if isinstance(stage_map, dict):
            for component, value in stage_map.items():
                if self._component_selected(component, flags):
                    specs.extend(self._as_module_spec_list(value))

        specs.extend(self._as_module_spec_list(stage_params.get("trainable_modules")))
        specs.extend(self._as_module_spec_list(stage_params.get("extra_trainable_modules")))
        return specs

    def _coerce_trainable_selection(self, selection):
        if selection is None:
            return [], []
        if isinstance(selection, dict):
            modules = self._as_module_spec_list(selection.get("modules"))
            params = self._as_module_spec_list(selection.get("parameters"))
            if "modules" not in selection and "parameters" not in selection:
                modules = self._as_module_spec_list(selection)
            return modules, params
        specs = self._as_module_spec_list(selection)
        modules = [item for item in specs if not isinstance(item, torch.nn.Parameter)]
        params = [item for item in specs if isinstance(item, torch.nn.Parameter)]
        return modules, params

    def _setup_trainable_parameters(self, stage_params, *, use_v, train_g, train_s, use_i):
        flags = {
            "velocity": bool(use_v),
            "growth": bool(train_g),
            "score": bool(train_s),
            "interaction": bool(use_i),
        }
        for param in self.model.parameters():
            param.requires_grad = False

        selected_params = []
        seen = set()

        def add_param(param):
            if not isinstance(param, torch.nn.Parameter):
                raise TypeError(f"Expected torch.nn.Parameter in trainable selection, got {type(param).__name__}")
            ident = id(param)
            if ident not in seen:
                param.requires_grad = True
                selected_params.append(param)
                seen.add(ident)

        if bool(stage_params.get("train_all_parameters", False)):
            for param in self.model.parameters():
                add_param(param)
        else:
            selector = getattr(self.model, "cytobridge_trainable_parameters", None)
            if callable(selector):
                modules, params = self._coerce_trainable_selection(
                    selector(stage_params=dict(stage_params), train_flags=dict(flags))
                )
            else:
                modules = self._collect_component_module_specs(flags, stage_params)
                params = []

            for item in params:
                add_param(item)
            for item in modules:
                module = self._get_module_by_path(item) if isinstance(item, str) else item
                if not isinstance(module, torch.nn.Module):
                    raise TypeError(
                        f"Trainable module spec {item!r} did not resolve to torch.nn.Module"
                    )
                for param in module.parameters():
                    add_param(param)

        if not selected_params:
            raise RuntimeError(
                "No trainable parameters selected for this stage. If this is a custom model, "
                "attach extra heads to velocity_net/growth_net/score_net/interaction_net, set "
                "model.cytobridge_component_modules, implement cytobridge_trainable_parameters(...), "
                "or set stage trainable_modules / train_all_parameters."
            )
        return selected_params

    # --------------------------
    # Main Modifications: Optimizer and Scheduler Setup
    # --------------------------
    def _setup_stage(self, stage_params):
        lr = stage_params['lr']
        print(f"\n====  {stage_params['name']}  ====")

        # Get flags for score network training from stage parameters
        train_strategy = str(stage_params.get('train_strategy', '')).lower()



        if not train_strategy or train_strategy == 'none':
            use_v = train_g = use_s = use_i = True          # 缺省策略：全训练
        else:
            use_v, train_g, use_s, use_i = 'v' in train_strategy, 'g' in train_strategy, 's' in train_strategy, 'i' in train_strategy

        if stage_params.get('mode') ==  "neural_ode":
            train_s = False
            if use_v and use_s and use_i:
                train_s = True
            self.model.use_growth_in_ode_inter = stage_params.get('use_growth_in_ode_inter', True)
            self.ode_func.use_mass = self.model.use_growth_in_ode_inter
            self.ode_func.score_use = train_strategy is not None and 's' in train_strategy
            self.ode_func.interaction_use = train_strategy is not None and 'i' in train_strategy

        elif stage_params.get('mode') ==  "flow_matching":
            train_s = use_s
        else:
            raise ValueError(f"Unknown training mode: {stage_params['mode']}")

        params = self._setup_trainable_parameters(
            stage_params,
            use_v=use_v,
            train_g=train_g,
            train_s=train_s,
            use_i=use_i,
        )
        # Initialize Adam optimizer with only trainable parameters
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # Reset scheduler before setting up new one
        self.scheduler = None
        if 'scheduler_type' in stage_params:
            if stage_params['scheduler_type'] == 'cosine':
                # Use Cosine Annealing scheduler if specified
                cosine_epochs = stage_params.get('cosine_epochs', 1000)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=cosine_epochs, 
                    eta_min=1e-5  # Minimum learning rate
                )
            elif stage_params['scheduler_type'] == 'steplr':
                # Use StepLR scheduler if specified
                self.scheduler = StepLR(
                    optimizer=self.optimizer,
                    step_size=stage_params['scheduler_step_size'],
                    gamma=stage_params['scheduler_gamma']  # Learning rate decay factor
                )
                print(f"  Enabled learning rate scheduler: step_size={stage_params['scheduler_step_size']}, gamma={stage_params['scheduler_gamma']}")
        else:
            print("  No scheduler parameters configured - keeping learning rate constant")

        # Print gradient status (trainable/non-trainable) for each module
        for n, m in self.model.named_children():
            flag = any(p.requires_grad for p in m.parameters())
            print(f"  {n:<15}  grad={flag}")
        # Print shapes of parameters in optimizer
        print("  Optimizer parameters (shapes):", [p.shape for g in self.optimizer.param_groups for p in g['params']])

    def train(self, data=None, time_points=None):
        """Main training loop that executes multiple training stages based on configuration
        
        Args:
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        
        Returns:
            DynamicalModel: Trained model
        """
        if data is None:
            data = self.training_data.latent_by_time
        if time_points is None:
            time_points = self.training_data.time_points

        # Get training plan and base default parameters from configuration
        training_plan = self.config['training']['plan']
        base_defaults = self.config['training']['defaults']
        self._start_training_timer()

        # Execute each stage in the training plan
        for stage_index, stage_config in enumerate(training_plan):
            if self._time_budget_exceeded():
                break
            # Merge base defaults with stage-specific config (stage config takes priority)
            stage_params = base_defaults.copy()
            stage_params.update(stage_config)
            stage_name = stage_params['name']


            print(f"\n--- Starting Stage: {stage_name} ---")
            print(
                f"  Mode: {stage_params['mode']}, Epochs: {stage_params['epochs']}, Use Score: {stage_params.get('score_use', False)}")
            train_strategy = stage_params.get('train_strategy', None)


            if self.stage_runner is not None:
                custom_result = self.run_custom_stage(stage_params, stage_index)
                if custom_result is not None:
                    if self._time_budget_exceeded():
                        break
                    continue
            if stage_params['mode'] == 'neural_ode':
                self._setup_stage(stage_params)
                self.run_neural_ode_stage(stage_params, data, time_points)
            elif stage_params['mode'] == 'flow_matching':
                self._setup_stage(stage_params)
                self.run_flow_matching_stage(stage_params, data, time_points)
            else:
                raise ValueError(f"Unknown training mode: {stage_params['mode']}")
            if self._time_budget_exceeded():
                break

        self._refresh_time_budget_summary()
        return self.model

    def _stage_output_dir(self, stage_params):
        ckpt_dir = os.path.join(self.config.get('ckpt_dir', '.'), str(stage_params.get('name', 'stage')))
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    def run_custom_stage(self, stage_params, stage_index, *, preview_only: bool = False):
        if self.stage_runner is None:
            raise ValueError("stage_runner is not configured")
        context = StageRunnerContext(
            model=self.model,
            config=self.config,
            stage_params=dict(stage_params),
            stage_index=int(stage_index),
            training_data=self.training_data,
            device=self.device,
            batch_size=int(self.batch_size),
            output_dir=self._stage_output_dir(stage_params),
            progress_callback=self.progress_callback,
            should_stop=self._time_budget_exceeded,
            metadata={
                "ckpt_dir": self.config.get("ckpt_dir"),
                "exp_dir": self.exp_dir,
                "source": "TrainingPipeline.run_custom_stage",
                "time_budget": dict(self.training_time_budget),
            },
            preview_only=bool(preview_only),
        )
        result = self.stage_runner(context)
        if result is None:
            return None
        elif isinstance(result, dict):
            result = StageRunnerResult(**result)
        elif not isinstance(result, StageRunnerResult):
            raise TypeError(
                f"stage_runner must return StageRunnerResult, dict, or None, got {type(result).__name__}"
            )
        if not (result.stage_summary or result.logs or result.artifacts or result.model_outputs):
            return None
        if preview_only:
            return result
        stage_summary = dict(result.stage_summary or {})
        stage_summary.setdefault("name", stage_params.get("name"))
        stage_summary.setdefault("mode", stage_params.get("mode"))
        stage_summary.setdefault("epochs", int(stage_params.get("epochs", 0)))
        if result.logs:
            stage_summary.setdefault("logs", result.logs)
        if result.artifacts:
            stage_summary.setdefault("artifacts", result.artifacts)
        if result.model_outputs:
            stage_summary.setdefault("model_outputs", result.model_outputs)
        if self._time_budget_exceeded():
            stage_summary.setdefault("stopped_early_timeout", True)
            stage_summary.setdefault("completed_epochs", stage_summary.get("completed_epochs", 0))
        self.training_summary["stages"].append(stage_summary)
        return result

    def run_neural_ode_stage(self, stage_params, data, time_points):
        """Execute training stage using Neural ODE mode
        
        Args:
            stage_params: Dictionary of parameters for current stage (epochs, loss weights, etc.)
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        """
        epochs = stage_params['epochs']
        # Get model saving strategy (default to 'best' if not specified)
        save_strategy = stage_params.get('save_strategy', 'best')


        # Initialize variables for tracking best model
        best_loss = float('inf')
        best_state = _clone_state_dict_tensors(self.model.state_dict())
        train_strategy = stage_params.get('train_strategy', None)
        train_name=stage_params["name"]

        last_epoch_loss = None
        completed_epochs = 0
        # Training loop over epochs
        for epoch in range(epochs):
            # Calculate loss for one epoch of Neural ODE training
            loss = self.train_neural_ode_epoch(stage_params, data, time_points, self.ode_func)
            last_epoch_loss = float(loss)
            completed_epochs = epoch + 1

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"  Stage '{stage_params['name']}', Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
                # Emit progress callback for Web UI
                if self.progress_callback:
                    progress = (epoch + 1) / epochs
                    self.progress_callback(f"Neural ODE - {stage_params['name']} Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}", progress)

            # if epoch % 10 == 0 and self.use_interaction:
            #     plot_interaction_potential_epoch(self.model,d=1,num_points=40,output_path=self.config["ckpt_dir"]+f"/interfigures/{train_name}_epoch_{epoch}_inter",device="cuda")
            #     if epoch < 15:
            #         print(f"{train_name} plot_interaction_potential_epoch {epoch} has done")

            # if "i" in train_strategy:
            #     if epoch % 10 == 0:
            #         plot_interaction_potential_epoch(self.model,d=1,num_points=21,output_path=self.config["ckpt_dir"]+f"/interfigures/{train_name}_epoch_{epoch}_inter",device="cuda")
            #         print(f"{train_name} plot_interaction_potential_epoch {epoch} has done")
            # Update best model if current loss is lower than previous best
            if loss < best_loss:
                best_loss = loss
                self.logger.info(f"Epoch {epoch:3d} has a lower loss| all_loss {best_loss:.4f}")
                best_state = _clone_state_dict_tensors(self.model.state_dict())
            self._record_completed_epoch(stage_params, epoch, epochs)
            if self._time_budget_exceeded():
                break

        # Determine which model state to save (best or last)
        if save_strategy == 'best':
            save_state = best_state
            save_loss = best_loss
        else:  # 'last' strategy
            save_state = _clone_state_dict_tensors(self.model.state_dict())
            save_loss = last_epoch_loss if last_epoch_loss is not None else float('nan')

        # Load saved state (best or last) back to model
        self.model.load_state_dict(save_state)
        # Create checkpoint directory for current stage
        ckpt_dir = os.path.join(self.config.get('ckpt_dir', '.'), stage_params['name'])
        os.makedirs(ckpt_dir, exist_ok=True)
        # Define checkpoint filename based on save strategy
        ckpt_filename = 'best_model.pth' if save_strategy == 'best' else 'last_model.pth'
        torch.save(save_state, os.path.join(ckpt_dir, ckpt_filename))
        print(f"  {save_strategy.capitalize()} model (loss={save_loss:.4f}) saved → {ckpt_dir}/{ckpt_filename}")
        self.training_summary["stages"].append({
            "name": stage_params["name"],
            "mode": stage_params.get("mode", "neural_ode"),
            "save_strategy": save_strategy,
            "saved_loss": float(save_loss),
            "best_loss": float(best_loss),
            "last_epoch_loss": float(last_epoch_loss) if last_epoch_loss is not None else None,
            "epochs": int(epochs),
            "completed_epochs": int(completed_epochs),
            "stopped_early_timeout": bool(self.training_time_budget.get("timed_out")),
        })

    def train_neural_ode_epoch(self, stage_params, data, time_points, ode_func):
        """Calculate loss for one epoch of Neural ODE training
        
        Args:
            stage_params: Dictionary of parameters for current stage (loss weights, etc.)
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
            ode_func: ODEFunc instance for computing ODE updates
        
        Returns:
            float: Average loss over all time intervals
        """
        # Get loss weights and configuration from stage parameters
        lambda_ot = stage_params['lambda_ot']
        lambda_mass = stage_params['lambda_mass']
        lambda_energy = stage_params['lambda_energy']
        
        OT_loss_type = stage_params['OT_loss']
        use_density_loss = stage_params.get('use_density_loss', False)
        use_pinn_loss = stage_params.get('use_pinn_loss', False)

        global_mass = stage_params.get('global_mass', False)
        if use_density_loss:
            if 'density_top_k' not in stage_params or 'lambda_density' not in stage_params or 'density_hinge_value' not in stage_params:
                raise ValueError(
                    "When use_density_loss=True, all 'density_top_k','lambda_density' and 'density_hinge_value' "
                    "must be provided in stage_params.(Default recommended ( 5 , 10 and  0.01))" 
                )            
            top_k = stage_params['density_top_k']
            hinge_value = stage_params['density_hinge_value']
            lambda_density = stage_params['lambda_density']
            density_fn = Density_loss(hinge_value)


        # Initialize with sampled data from the first time point
        x0 = sample(data[0], self.batch_size).to(self.device)
        # Initialize log-weights (uniform distribution)
        lnw0 = torch.log(torch.ones(self.batch_size, 1) / self.batch_size).to(self.device)
        # Total number of samples at the first time point
        mass_0 = data[0].shape[0]

        total_loss = 0.0
        # Iterate over all time intervals (from t_{i-1} to t_i)
        for idx in range(1, len(time_points)):
            # Reset gradients before each time interval update
            self.optimizer.zero_grad()

            # Get current time interval and target data
            t0, t1 = time_points[idx - 1], time_points[idx]
            data_t1 = sample(data[idx], self.batch_size).to(self.device)
            # Total number of samples at the target time point
            mass_1 = data[idx].shape[0]
            # Calculate relative mass ratio between target and initial time points
            relative_mass = mass_1 / mass_0

            # Perform one Neural ODE step to predict state at t1
            x1, lnw1, e1 = neural_ode_step(ode_func, x0, lnw0, t0, t1, self.device)

            # Calculate individual loss components
            loss_ot = calc_ot_loss(x1, data_t1, lnw1, OT_loss_type)
            # Calculate mass loss only if mass component is enabled
            loss_mass = calc_mass_loss(x1, data_t1, lnw1, relative_mass, global_mass) if self.use_mass else 0.0
            # Energy loss (average of energy term from ODE step)
            loss_energy = e1.mean()

            # Combine losses with respective weights
            loss = (lambda_ot * loss_ot) + (lambda_mass * loss_mass) + (lambda_energy * loss_energy)

            if use_density_loss:          
                density_loss = density_fn(x1, data_t1, top_k=top_k)
                density_loss = density_loss.to(loss.device)
                loss += lambda_density * density_loss
                # print('density loss')
                # print(density_loss)
            if use_pinn_loss: 
                if 'lambda_pinn'  not in stage_params:
                    raise ValueError(
                        "When use_pinn_loss=True, 'lambda_pinn' must be provided in stage_params.(Default recommended (100))" 
                    )            
                lambda_pinn = stage_params['lambda_pinn'] 

                loss_pinn = calc_pinn_loss(self, t1, data_t1,sigma=stage_params['sigma'], use_mass=self.use_mass,trace_df_dz=trace_df_dz,device=self.device)
                # print("loss_pinn",loss_pinn)
                # print("loss",loss)
                loss += lambda_pinn * loss_pinn
            # print(f"OT Loss: {loss_ot:.4f} (λ={lambda_ot}), Mass Loss: {loss_mass:.4f} (λ={lambda_mass}), Energy Loss: {loss_energy:.4f} (λ={lambda_energy}), Density Loss: {density_loss:.4f} (λ={lambda_density})" if use_density_loss else f"OT Loss: {loss_ot:.4f} (λ={lambda_ot}), Mass Loss: {loss_mass:.4f} (λ={lambda_mass}), Energy Loss: {loss_energy:.4f} (λ={lambda_energy})", end="")
            # if use_pinn_loss:
            #     print(f", PINN Loss: {loss_pinn:.4f} (λ={lambda_pinn})")
            # Backpropagate gradients and update optimizer
            loss.backward()
            self.optimizer.step()

            # Update initial state for next time interval (detach to avoid gradient accumulation)
            x0 = x1.clone().detach()
            lnw0 = lnw1.clone().detach()

            # Accumulate total loss over all time intervals
            total_loss += loss.item()

        # Return average loss per time interval
        return total_loss / (len(time_points) - 1)


    def run_flow_matching_stage(self, stage_params, data, time_points):
        """Execute training stage using Flow Matching mode
        
        Args:
            stage_params: Dictionary of parameters for current stage (epochs, sigma, etc.)
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        """
        # Create checkpoint directory for current stage
        ckpt_dir = os.path.join(self.config.get('ckpt_dir', '.'), stage_params['name'])
        os.makedirs(ckpt_dir, exist_ok=True)

        # Convert time points to tensor (device-compatible)
        time = torch.tensor(time_points, device=self.device, dtype=torch.float32)
        # Get alpha regularization parameter (default to 1.0 if not specified)
        alpha_regm = stage_params.get('alpha_regm', 1.0)
        print("alpha_regm :", alpha_regm)
        self.sigma = stage_params['sigma']
        # Convert data to list of numpy arrays (required for compute_uot_plans)
        X = [data[i].float().cpu().detach().numpy() for i in range(len(time_points))]
        
        # Stage-level loss selection only.
        # This does not redefine model components or path semantics.
        train_strategy = str(stage_params.get('train_strategy', 's')).lower()
        regress_v, regress_g, regress_score = 'v' in train_strategy, 'g' in train_strategy, 's' in train_strategy
        
        backend = self._resolve_flow_matching_backend(
            stage_params=stage_params,
            regress_v=regress_v,
            regress_g=regress_g,
            regress_score=regress_score,
        )
        use_prepared_backend = (
            backend is self.prepared_flow_matching_backend
            and getattr(backend, "state", None) is not None
            and stage_params.get("name") == self.prepared_flow_matching_stage_name
        )
        if not use_prepared_backend:
            backend.prepare(X, time, self.device)
        else:
            self.prepared_flow_matching_backend = None
            self.prepared_flow_matching_stage_name = None
            self.prepared_flow_matching_stage_params = None
        # Get model saving strategy (default to 'best' if not specified)
        save_strategy = stage_params.get('save_strategy', 'best')
        # Initialize variables for tracking best model
        best_loss = float('inf')
        best_state_dict = None
        
        # Get batch size from stage parameters
        batch_size = stage_params['batch_size']



        # Training loop over epochs (with tqdm progress bar)
        total_epochs = stage_params['epochs']
        last_epoch_loss = None
        last_epoch_penalty = None
        completed_epochs = 0
        loss_hook_logs = []
        for epoch in tqdm(range(total_epochs), desc='Flow matching'):
            # Calculate loss for one epoch of Flow Matching training
            loss, penalty, hook_logs = self.train_flow_matching_epoch(
                backend, X, time,
                self.optimizer,
                stage_params,
                stage_params['flow_matching']['lambda_penalty'],
                batch_size,
                regress_v, regress_g, regress_score,
            )
            last_epoch_loss = float(loss.item() if hasattr(loss, "item") else loss)
            last_epoch_penalty = float(penalty.item() if hasattr(penalty, "item") else penalty)
            if hook_logs:
                loss_hook_logs.append({"epoch": int(epoch + 1), **hook_logs})

            # Stop training if loss becomes NaN (numerical instability)
            if torch.isnan(loss):
                self.logger.info("Training stopped due to NaN loss")
                # Load best model state before NaN occurred
                if best_state_dict is not None:
                    self.model.load_state_dict(best_state_dict)
                break

            # Update best model if current loss is lower than previous best
            if loss < best_loss:
                best_loss = float(loss.item() if hasattr(loss, "item") else loss)
                best_state_dict = _clone_state_dict_tensors(self.model.state_dict())

            # Emit progress callback for Web UI every 10 epochs
            if epoch % 10 == 0 and self.progress_callback:
                progress = (epoch + 1) / total_epochs
                loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
                self.progress_callback(f"Flow Matching Epoch {epoch + 1}/{total_epochs}, Loss: {loss_val:.4f}", progress)

            # Combine loss and penalty for backpropagation
            total_loss = loss + penalty
            # print("score_loss",loss)
            # print("penalty",penalty)

            total_loss.backward()
            # Update optimizer
            self.optimizer.step()
            # Update scheduler if initialized
            if self.scheduler is not None:
                self.scheduler.step()
            completed_epochs = epoch + 1
            self._record_completed_epoch(stage_params, epoch, total_epochs)
            if self._time_budget_exceeded():
                break


        # Determine which model state to save (best or last)
        if save_strategy == 'best':
            save_state = best_state_dict if best_state_dict is not None else _clone_state_dict_tensors(self.model.state_dict())
            save_loss = best_loss
        else:  # 'last' strategy
            save_state = _clone_state_dict_tensors(self.model.state_dict())
            if last_epoch_loss is not None and last_epoch_penalty is not None:
                save_loss = float(last_epoch_loss) + float(last_epoch_penalty)
            else:
                save_loss = float('nan')

        # Load saved state (best or last) back to model
        self.model.load_state_dict(save_state)
        # Define checkpoint filename based on save strategy
        ckpt_filename = 'best_model.pth' if save_strategy == 'best' else 'last_model.pth'
        torch.save(save_state, os.path.join(ckpt_dir, ckpt_filename))
        print(f"  {save_strategy.capitalize()} model (loss={save_loss:.4f}) "
              f"saved → {ckpt_dir}/{save_strategy}_model.pth")
        self.training_summary["stages"].append({
            "name": stage_params["name"],
            "mode": stage_params.get("mode", "flow_matching"),
            "save_strategy": save_strategy,
            "saved_loss": float(save_loss),
            "best_loss": float(best_loss),
            "last_epoch_loss": last_epoch_loss,
            "last_epoch_penalty": last_epoch_penalty,
            "epochs": int(total_epochs),
            "completed_epochs": int(completed_epochs),
            "stopped_early_timeout": bool(self.training_time_budget.get("timed_out")),
            "loss_hook_logs": loss_hook_logs,
        })

    def _resolve_flow_matching_backend(self, stage_params, regress_v, regress_g, regress_score):
        if (
            self.prepared_flow_matching_backend is not None
            and stage_params.get("name") == self.prepared_flow_matching_stage_name
        ):
            return self.prepared_flow_matching_backend
        builder = self.flow_matching_backend_builder or default_flow_matching_backend_builder
        build_context = FlowMatchingBuildContext(
            stage_params=stage_params,
            training_data=self.training_data,
            device=self.device,
            regress_v=regress_v,
            regress_g=regress_g,
            regress_score=regress_score,
            model=self.model,
            metadata={
                "training_config": self.config.get("training", {}),
                "model_config": self.config.get("model", {}),
                "ckpt_dir": self.config.get("ckpt_dir"),
            },
        )
        return build_flow_matching_backend(builder, build_context=build_context)

    def _sanitize_loss_hook_logs(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return {
                "type": "tensor",
                "shape": list(value.shape),
            }
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            return {
                "type": "ndarray",
                "shape": list(value.shape),
            }
        if isinstance(value, dict):
            return {str(k): self._sanitize_loss_hook_logs(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_loss_hook_logs(v) for v in value]
        if isinstance(value, tuple):
            return [self._sanitize_loss_hook_logs(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _validate_loss_tensor(self, value, name: str) -> None:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar tensor, got shape {tuple(value.shape)}.")
        if not torch.isfinite(value.detach()).all():
            raise ValueError(f"{name} must be finite.")

    def _apply_flow_matching_loss_hook(
        self,
        *,
        stage_params,
        backend,
        batch,
        net_input,
        lambda_t,
        regress_v,
        regress_g,
        regress_score,
        builtin_losses=None,
    ):
        if self.flow_matching_loss_hook is None:
            return None

        metadata = {
            "device": str(self.device),
            "ckpt_dir": self.config.get("ckpt_dir"),
            "net_input_order": "[x, t]",
        }
        if builtin_losses:
            metadata["builtin_losses"] = builtin_losses
        context = FlowMatchingLossContext(
            stage_params=stage_params,
            training_data=self.training_data,
            backend=backend,
            batch=batch,
            model=self.model,
            net_input=net_input,
            lambda_t=lambda_t,
            regress_v=regress_v,
            regress_g=regress_g,
            regress_score=regress_score,
            x_input=net_input[:, :-1],
            t_input=net_input[:, -1:],
            metadata=metadata,
        )
        result = self.flow_matching_loss_hook(context)
        if result is None:
            return None
        if isinstance(result, FlowMatchingLossResult):
            return result
        if isinstance(result, torch.Tensor):
            return FlowMatchingLossResult(extra_loss=result)
        raise TypeError(
            "flow_matching_loss_hook must return None, torch.Tensor, or FlowMatchingLossResult"
        )

    def train_flow_matching_epoch(self, backend, X, time,
                                  optimizer, stage_params, lambda_pen, batch_size, regress_v, regress_g, regress_score):
        """Calculate loss for one epoch of Flow Matching training
        
        Args:
            backend: Prepared FlowMatchingBackend instance
            X: List of numpy arrays where each element represents samples at a specific time point
            time: Tensor of time points (device-compatible)
            optimizer: Torch optimizer instance
            lambda_pen: Penalty weight for score network training
            batch_size: Batch size for sampling
            regress_v: Flag to train velocity network (v)
            regress_g: Flag to train growth network (g)
            regress_score: Flag to train score network
        
        Returns:
            tuple: (total_loss, penalty, hook_logs) where losses are torch tensors
        """
        # Reset gradients before each batch
        optimizer.zero_grad()
        # Sample batch data for Flow Matching (time, positions, velocities, growth values, weights, noise)
        batch = backend.sample_batch(X, time, batch_size, device=self.device)
        if batch.xt.shape[0] == 0:
            raise RuntimeError(
                "Flow-matching batch is empty. Coupling likely produced near-zero/invalid transport "
                "plans for all adjacent time pairs. Please revise coupling regularization, constraints, "
                "or mini-batch strategy."
            )

        # Reshape time tensor to (batch_size, 1) for concatenation with position data
        t = torch.unsqueeze(batch.t_global, 1).to(self.device)

        # Compute lambda(t) from the backend's local path clock
        lambda_t = backend.compute_lambda(batch.t_local)

        # Enable gradient computation for position data (required for score calculation via autograd)
        xt = batch.xt.requires_grad_(True)
        # Get references to model components
        v_net = getattr(self.model, "velocity_net", None)
        g_net = getattr(self.model, "growth_net", None)
        score_net = getattr(self.model, "score_net", None)
        # Concatenate position and time data for network input as [x, t].
        # This matches DynamicalModel.forward(...) and every builtin velocity/growth head.
        net_input = torch.cat([xt, t], dim=1)

        # Initialize component losses and penalty. Keep component losses separate
        # so custom hooks can explicitly replace one builtin objective without
        # brittle "custom - builtin" arithmetic.
        score_loss = None
        velocity_loss = None
        growth_loss = None
        penalty = torch.as_tensor(0.0, device=self.device)
        # Train score network if enabled
        if regress_score:
            if score_net is None:
                raise AttributeError("Flow-matching stage requested score regression but model has no score_net.")
            # Predict score potential (value_st) from score network
            value_st = score_net(net_input)
            # Compute score via automatic differentiation (gradient of value_st w.r.t. xt)
            st = torch.autograd.grad(
                outputs=value_st,
                inputs=xt,
                grad_outputs=torch.ones_like(value_st),
                create_graph=True  # Required for second-order gradients (if needed)
            )[0]
            # `batch.loss_weights` are path-defined per-sample loss weights, not the
            # current path masses.
            score_loss = torch.mean(batch.loss_weights * ((lambda_t[:, None] * st + batch.eps) ** 2))
            # Handle NaN loss (set to 0 to avoid training instability)
            if torch.isnan(score_loss):
                score_loss = torch.zeros_like(score_loss)
            self._validate_loss_tensor(score_loss, "score_loss")
            # Add penalty term to regularize score potential (prevents exploding values)
            penalty += lambda_pen * torch.max(torch.relu(value_st))
        
        # Train velocity network (v) if enabled
        if regress_v:
            if v_net is None:
                raise AttributeError("Flow-matching stage requested velocity regression but model has no velocity_net.")
            # Predict velocity from velocity network
            v_predict = v_net(net_input)
            # Add weighted MSE loss between predicted and target velocities.
            # The weights are optimization-time loss weights.
            velocity_loss = torch.mean(batch.loss_weights * (v_predict - batch.ut) ** 2)
            self._validate_loss_tensor(velocity_loss, "velocity_loss")

        # Train growth network (g) if enabled
        if regress_g:
            if g_net is None:
                raise AttributeError("Flow-matching stage requested growth regression but model has no growth_net.")
            # Predict growth values from growth network
            g_predict = g_net(net_input)
            # Add weighted MSE loss between predicted and target growth-rate values.
            # `batch.gt` is the rate target; `batch.loss_weights` is a separate loss
            # weighting term.
            growth_loss = torch.mean(batch.loss_weights * (g_predict - batch.gt) ** 2)
            self._validate_loss_tensor(growth_loss, "growth_loss")

        hook_result = self._apply_flow_matching_loss_hook(
            stage_params=stage_params,
            backend=backend,
            batch=batch,
            net_input=net_input,
            lambda_t=lambda_t,
            regress_v=regress_v,
            regress_g=regress_g,
            regress_score=regress_score,
            builtin_losses={
                "score_loss": score_loss,
                "velocity_loss": velocity_loss,
                "growth_loss": growth_loss,
            },
        )
        hook_logs = None
        extra_loss = None
        if hook_result is not None:
            hook_logs = self._sanitize_loss_hook_logs(hook_result.logs)
            if hook_result.replace_score_loss is not None:
                if not regress_score or score_loss is None:
                    raise RuntimeError(
                        "flow_matching_loss_hook returned replace_score_loss, but score regression is not active."
                    )
                score_loss = hook_result.replace_score_loss
                self._validate_loss_tensor(score_loss, "replace_score_loss")
            if hook_result.replace_velocity_loss is not None:
                if not regress_v or velocity_loss is None:
                    raise RuntimeError(
                        "flow_matching_loss_hook returned replace_velocity_loss, but velocity regression is not active."
                    )
                velocity_loss = hook_result.replace_velocity_loss
                self._validate_loss_tensor(velocity_loss, "replace_velocity_loss")
            if hook_result.replace_growth_loss is not None:
                if not regress_g or growth_loss is None:
                    raise RuntimeError(
                        "flow_matching_loss_hook returned replace_growth_loss, but growth regression is not active."
                    )
                growth_loss = hook_result.replace_growth_loss
                self._validate_loss_tensor(growth_loss, "replace_growth_loss")
            extra_loss = hook_result.extra_loss
            if extra_loss is not None:
                self._validate_loss_tensor(extra_loss, "extra_loss")

        loss = torch.as_tensor(0.0, device=self.device)
        for component_loss in (score_loss, velocity_loss, growth_loss):
            if component_loss is not None:
                loss = loss + component_loss
        if extra_loss is not None:
            loss = loss + extra_loss

        return (
            torch.as_tensor(loss, device=self.device),
            torch.as_tensor(penalty, device=self.device),
            hook_logs,
        )

    def evaluate(self, adata=None, data=None, time_points=None):
        """Evaluate trained model using builtin Wasserstein-1 distance and TMV.

        Builtin metrics remain package-defined. Custom algorithms may append
        additive metrics through `evaluation_metrics_hook`, but may not replace
        `w1_scores` or `tmv_scores`.
        
        Args:
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        
        Returns:
            dict: Builtin metrics plus optional additive `custom_metrics`
        """
        if adata is None:
            adata = self.training_data.adata
        if data is None:
            data = self.training_data.latent_by_time
        if time_points is None:
            time_points = self.training_data.time_points

        print(f"\n--- Starting Evaluation ---")
        self.inference_time_budget = self._build_inference_time_budget(data=data)
        self._start_inference_timer()
        try:
            with _InferenceTimeoutGuard(self.inference_time_budget.get("budget_sec")):
                trajectory = self._predict_evaluation_trajectory(
                    adata=adata,
                    data=data,
                    time_points=time_points,
                )
                self._raise_if_inference_time_budget_exceeded()
                point = trajectory.observed_points_by_time()
                weight = trajectory.observed_weights_by_time()

                # Calculate Wasserstein-1 distance for each time point (excluding initial time)
                metrics, timepoint_results = self._compute_builtin_metrics(
                    data=data,
                    time_points=time_points,
                    evaluation_trajectory=trajectory,
                )
                try:
                    trajectory_path = self._save_evaluation_trajectory_artifact(trajectory)
                    if trajectory_path:
                        trajectory.artifacts["evaluation_trajectory_path"] = trajectory_path
                        metrics["evaluation_trajectory_path"] = trajectory_path
                        metrics["trajectory_path"] = trajectory_path
                except Exception as exc:
                    metrics["evaluation_trajectory_warning"] = str(exc)
                self._raise_if_inference_time_budget_exceeded()

                try:
                    custom_metrics = self._apply_evaluation_metrics_hook(
                        adata=adata,
                        data=data,
                        time_points=time_points,
                        builtin_metrics=metrics,
                        simulated_points_by_time=point,
                        simulated_weights_by_time=weight,
                        evaluation_trajectory=trajectory,
                        timepoint_results=timepoint_results,
                    )
                    if custom_metrics:
                        metrics["custom_metrics"] = custom_metrics
                except InferenceTimeoutError:
                    raise
                except Exception as exc:
                    metrics["custom_metrics"] = {}
                    metrics["custom_metrics_warning"] = str(exc)
        except InferenceTimeoutError:
            message = self._mark_inference_timeout()
            metrics = {
                "w1_scores": [],
                "tmv_scores": [],
                "custom_metrics": {},
                "evaluation_warning": message,
                "inference_timed_out": True,
                "inference_timeout_message": message,
                "inference_time_budget": dict(self.inference_time_budget),
            }
            try:
                custom_metrics = self._apply_evaluation_metrics_hook(
                    adata=adata,
                    data=data,
                    time_points=time_points,
                    builtin_metrics=metrics,
                    simulated_points_by_time=[],
                    simulated_weights_by_time=[],
                    evaluation_trajectory=None,
                    timepoint_results=[],
                    metadata={
                        "source": "TrainingPipeline.evaluate.timeout",
                        "prediction_contract": "builtin_inference_timed_out",
                        "inference_timed_out": True,
                        "inference_timeout_message": message,
                    },
                )
                if custom_metrics:
                    metrics["custom_metrics"] = custom_metrics
            except Exception as exc:
                metrics["custom_metrics"] = {}
                metrics["custom_metrics_warning"] = (
                    "evaluation_metrics_hook_after_timeout failed: " + str(exc)
                )
            return metrics
        finally:
            self._refresh_inference_time_budget_summary()

        metrics["inference_timed_out"] = bool(self.inference_time_budget.get("timed_out"))
        metrics["inference_timeout_message"] = str(self.inference_time_budget.get("message") or "")
        metrics["inference_time_budget"] = dict(self.inference_time_budget)
        return metrics

    def _apply_evaluation_metrics_hook(
        self,
        *,
        adata,
        data,
        time_points,
        builtin_metrics,
        simulated_points_by_time,
        simulated_weights_by_time,
        evaluation_trajectory,
        timepoint_results,
        metadata=None,
    ):
        if self.evaluation_metrics_hook is None:
            return {}
        hook_metadata = {
            "source": "TrainingPipeline.evaluate",
            "prediction_contract": "full_t0_to_final_trajectory",
        }
        if metadata:
            hook_metadata.update(dict(metadata))
        if evaluation_trajectory is None:
            hook_metadata.setdefault("trajectory_source", "unavailable")
            hook_metadata.setdefault("trajectory_unavailable", True)
            trajectory_time_points = []
            trajectory_points_by_time = []
            trajectory_weights_by_time = []
            observed_time_indices = []
        else:
            hook_metadata.setdefault("trajectory_source", evaluation_trajectory.source)
            trajectory_time_points = list(evaluation_trajectory.time_points)
            trajectory_points_by_time = list(evaluation_trajectory.points_by_time)
            trajectory_weights_by_time = list(evaluation_trajectory.weights_by_time)
            observed_time_indices = list(evaluation_trajectory.observed_time_indices)
        context = EvaluationMetricsContext(
            adata=adata,
            training_data=self.training_data,
            model=self.model,
            config=self.config,
            data=list(data),
            time_points=[float(t) for t in time_points],
            builtin_metrics=dict(builtin_metrics),
            metric_params=dict(self.evaluation_metrics_params),
            simulated_points_by_time=list(simulated_points_by_time),
            simulated_weights_by_time=list(simulated_weights_by_time),
            evaluation_trajectory=evaluation_trajectory,
            trajectory_time_points=trajectory_time_points,
            trajectory_points_by_time=trajectory_points_by_time,
            trajectory_weights_by_time=trajectory_weights_by_time,
            observed_time_indices=observed_time_indices,
            timepoint_results=list(timepoint_results),
            device=self.device,
            metadata=hook_metadata,
        )
        result = self.evaluation_metrics_hook(context)
        if result is None:
            return {}
        if not isinstance(result, dict):
            raise TypeError(
                "evaluation_metrics_hook must return dict[str, Any] or None"
            )
        forbidden = {"w1_scores", "tmv_scores", "evaluation_warning", "custom_metrics"} & set(result.keys())
        if forbidden:
            raise ValueError(
                "evaluation_metrics_hook may not overwrite builtin evaluation keys: "
                + ", ".join(sorted(forbidden))
            )
        return dict(result)

    def _evaluation_trajectory_artifact_path(self):
        evaluation_config = self.config.get("evaluation") if isinstance(self.config, dict) else {}
        if not isinstance(evaluation_config, dict):
            evaluation_config = {}
        configured = (
            evaluation_config.get("trajectory_artifact_path")
            or evaluation_config.get("evaluation_trajectory_path")
        )
        if configured:
            return Path(str(configured)).expanduser().resolve()
        ckpt_dir = Path(str(self.config.get("ckpt_dir") or "./cytobridge_output")).expanduser().resolve()
        artifact_dir = ckpt_dir.parent if ckpt_dir.name == "checkpoints" else ckpt_dir
        return artifact_dir / "evaluation_trajectory.npz"

    def _save_evaluation_trajectory_artifact(self, trajectory):
        path = self._evaluation_trajectory_artifact_path()
        estimated_mb = self._estimate_evaluation_trajectory_uncompressed_mb(trajectory)
        max_mb = self._evaluation_trajectory_max_uncompressed_mb()
        if max_mb > 0 and estimated_mb > max_mb:
            raise ValueError(
                "Evaluation trajectory artifact is too large to save safely: "
                f"estimated_uncompressed_mb={estimated_mb:.1f}, max_mb={max_mb:.1f}. "
                "Increase evaluation.max_trajectory_uncompressed_mb only if this is intentional."
            )
        arrays = {
            "time_points": np.asarray(trajectory.time_points, dtype=np.float64),
            "observed_time_points": np.asarray(trajectory.observed_time_points, dtype=np.float64),
            "observed_time_indices": np.asarray(trajectory.observed_time_indices, dtype=np.int64),
        }
        point_keys = []
        weight_keys = []
        for idx, (points, weights) in enumerate(zip(trajectory.points_by_time, trajectory.weights_by_time)):
            point_key = f"points_{idx:06d}"
            weight_key = f"weights_{idx:06d}"
            arrays[point_key] = np.asarray(points)
            arrays[weight_key] = np.asarray(weights).reshape(-1)
            point_keys.append(point_key)
            weight_keys.append(weight_key)
        manifest = {
            "schema_version": 1,
            "format": "cytobridge_evaluation_trajectory_npz",
            "source": str(trajectory.source),
            "step": trajectory.step,
            "time_points": [float(t) for t in trajectory.time_points],
            "observed_time_points": [float(t) for t in trajectory.observed_time_points],
            "observed_time_indices": [int(i) for i in trajectory.observed_time_indices],
            "point_keys": point_keys,
            "weight_keys": weight_keys,
            "estimated_uncompressed_mb": round(float(estimated_mb), 3),
            "max_uncompressed_mb": round(float(max_mb), 3),
            "metadata": dict(trajectory.metadata or {}),
        }
        arrays["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=False))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return str(path)

    @staticmethod
    def _array_nbytes(value):
        try:
            if torch.is_tensor(value):
                return int(value.numel()) * int(value.element_size())
        except Exception:
            pass
        try:
            return int(np.asarray(value).nbytes)
        except Exception:
            return 0

    def _estimate_evaluation_trajectory_uncompressed_mb(self, trajectory):
        total = 0
        for points in list(getattr(trajectory, "points_by_time", []) or []):
            total += self._array_nbytes(points)
        for weights in list(getattr(trajectory, "weights_by_time", []) or []):
            total += self._array_nbytes(weights)
        return float(total) / (1024.0 * 1024.0)

    def _evaluation_trajectory_step(self):
        evaluation_config = self.config.get("evaluation") if isinstance(self.config, dict) else {}
        if not isinstance(evaluation_config, dict):
            evaluation_config = {}
        raw_step = (
            evaluation_config.get("trajectory_step")
            or evaluation_config.get("trajectory_dt")
            or evaluation_config.get("prediction_trajectory_step")
            or 0.1
        )
        try:
            step = float(raw_step)
        except Exception:
            step = 0.1
        return step if step > 0 else 0.1

    def _evaluation_trajectory_max_slices(self):
        evaluation_config = self.config.get("evaluation") if isinstance(self.config, dict) else {}
        if not isinstance(evaluation_config, dict):
            evaluation_config = {}
        raw = (
            evaluation_config.get("max_trajectory_slices")
            or evaluation_config.get("trajectory_max_slices")
            or _env_int("CYTOBRIDGE_EVALUATION_TRAJECTORY_MAX_SLICES", EVALUATION_TRAJECTORY_MAX_SLICES)
        )
        try:
            value = int(raw)
        except Exception:
            value = EVALUATION_TRAJECTORY_MAX_SLICES
        return max(0, value)

    def _evaluation_trajectory_max_uncompressed_mb(self):
        evaluation_config = self.config.get("evaluation") if isinstance(self.config, dict) else {}
        if not isinstance(evaluation_config, dict):
            evaluation_config = {}
        raw = (
            evaluation_config.get("max_trajectory_uncompressed_mb")
            or evaluation_config.get("trajectory_max_uncompressed_mb")
            or _env_float(
                "CYTOBRIDGE_EVALUATION_TRAJECTORY_MAX_UNCOMPRESSED_MB",
                EVALUATION_TRAJECTORY_MAX_UNCOMPRESSED_MB,
            )
        )
        try:
            value = float(raw)
        except Exception:
            value = float(EVALUATION_TRAJECTORY_MAX_UNCOMPRESSED_MB)
        return max(0.0, value)

    def _evaluation_integration_dt(self, trajectory_step):
        evaluation_config = self.config.get("evaluation") if isinstance(self.config, dict) else {}
        if not isinstance(evaluation_config, dict):
            evaluation_config = {}
        raw_dt = (
            evaluation_config.get("trajectory_integration_dt")
            or evaluation_config.get("integration_dt")
            or evaluation_config.get("simulation_dt")
            or 0.01
        )
        try:
            dt = float(raw_dt)
        except Exception:
            dt = 0.01
        if dt <= 0:
            dt = float(trajectory_step or 0.1)
        return min(dt, float(trajectory_step or dt))

    def _evaluation_sigma(self):
        sigma = getattr(self, "sigma", None)
        if sigma is not None:
            try:
                return float(sigma)
            except Exception:
                pass
        training_config = self.config.get("training") if isinstance(self.config, dict) else {}
        if not isinstance(training_config, dict):
            training_config = {}
        plan = training_config.get("plan")
        if isinstance(plan, list):
            for stage in reversed(plan):
                if isinstance(stage, dict) and stage.get("sigma") is not None:
                    try:
                        return float(stage.get("sigma"))
                    except Exception:
                        break
        defaults = training_config.get("defaults")
        if isinstance(defaults, dict) and defaults.get("sigma") is not None:
            try:
                return float(defaults.get("sigma"))
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _unique_sorted_time_points(values, *, tolerance=1e-8):
        out = []
        for value in sorted(float(v) for v in values):
            if not out or abs(value - out[-1]) > tolerance:
                out.append(float(round(value, 10)))
        return out

    def _build_evaluation_trajectory_time_points(self, time_points):
        observed = [float(t) for t in time_points]
        if len(observed) <= 1:
            return observed
        step = self._evaluation_trajectory_step()
        start = float(observed[0])
        end = float(observed[-1])
        if end < start:
            raise ValueError("Evaluation time_points must be increasing.")
        grid = []
        current = start
        # Output a dense trajectory grid while always keeping exact observed
        # time points for metric slicing and future holdout-time evaluation.
        while current < end - 1e-9:
            grid.append(float(round(current, 10)))
            current += step
        grid.append(float(round(end, 10)))
        trajectory_time_points = self._unique_sorted_time_points([*grid, *observed])
        max_slices = self._evaluation_trajectory_max_slices()
        if max_slices > 0 and len(trajectory_time_points) > max_slices:
            raise ValueError(
                "Evaluation trajectory grid is too dense: "
                f"{len(trajectory_time_points)} slices exceeds max_trajectory_slices={max_slices}. "
                "Increase evaluation.trajectory_step or explicitly raise evaluation.max_trajectory_slices."
            )
        return trajectory_time_points

    @staticmethod
    def _as_numpy_array(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _match_observed_time_indices(self, trajectory_time_points, observed_time_points):
        trajectory = [float(t) for t in trajectory_time_points]
        indices = []
        for target in [float(t) for t in observed_time_points]:
            if not trajectory:
                raise ValueError("Evaluation trajectory has no time points.")
            best_idx = min(range(len(trajectory)), key=lambda idx: abs(trajectory[idx] - target))
            if abs(trajectory[best_idx] - target) > 1e-6:
                raise ValueError(
                    f"Evaluation trajectory does not contain observed time point {target}; "
                    f"nearest available time is {trajectory[best_idx]}."
                )
            indices.append(best_idx)
        return indices

    def _coerce_evaluation_trajectory(
        self,
        *,
        predicted_points_by_time,
        predicted_weights_by_time,
        trajectory_time_points,
        observed_time_points,
        source,
        artifacts=None,
        metadata=None,
    ):
        raw_points = [] if predicted_points_by_time is None else predicted_points_by_time
        raw_weights = [] if predicted_weights_by_time is None else predicted_weights_by_time
        raw_times = [] if trajectory_time_points is None else trajectory_time_points
        points = [self._as_numpy_array(x) for x in list(raw_points)]
        weights = [self._as_numpy_array(w).reshape(-1) for w in list(raw_weights)]
        actual_times = [float(t) for t in list(raw_times)]
        expected_step = self._evaluation_trajectory_step()
        if not points:
            raise ValueError("Evaluation trajectory is empty.")
        if len(points) != len(weights):
            raise ValueError(
                "Evaluation trajectory points and weights must have the same number of time slices "
                f"(got {len(points)} points vs {len(weights)} weights)."
            )
        if len(points) != len(actual_times):
            raise ValueError(
                "Evaluation prediction must return a full trajectory from t0 to the final time point. "
                f"Expected {len(actual_times)} trajectory slices for times {actual_times[:5]}...{actual_times[-5:]}, "
                f"got {len(points)}. Do not return only observed future-time predictions."
            )
        if any(actual_times[i] > actual_times[i + 1] for i in range(len(actual_times) - 1)):
            raise ValueError("Evaluation trajectory time points must be increasing.")
        if len(actual_times) >= 2:
            max_gap = max(actual_times[i + 1] - actual_times[i] for i in range(len(actual_times) - 1))
            if max_gap > expected_step * 1.5 + 1e-8:
                raise ValueError(
                    "Evaluation trajectory is too sparse for trusted metrics: "
                    f"max gap {max_gap:.6g} exceeds configured trajectory_step {expected_step:.6g}."
                )
        observed = [float(t) for t in observed_time_points]
        if observed:
            if abs(actual_times[0] - observed[0]) > 1e-6 or abs(actual_times[-1] - observed[-1]) > 1e-6:
                raise ValueError(
                    "Evaluation trajectory must start at the first observed time point and end at the final observed time point."
                )
        observed_indices = self._match_observed_time_indices(actual_times, observed)
        for idx, (point, weight) in enumerate(zip(points, weights)):
            if point.shape[0] != weight.shape[0]:
                raise ValueError(
                    f"Evaluation trajectory slice {idx} has {point.shape[0]} particles but {weight.shape[0]} weights."
                )
            if not np.isfinite(point).all() or not np.isfinite(weight).all():
                raise ValueError(f"Evaluation trajectory slice {idx} contains NaN or infinite values.")
            if weight.sum() <= 0:
                raise ValueError(f"Evaluation trajectory slice {idx} has non-positive total weight.")
        return EvaluationTrajectory(
            time_points=actual_times,
            points_by_time=points,
            weights_by_time=weights,
            observed_time_points=observed,
            observed_time_indices=observed_indices,
            source=str(source),
            step=expected_step,
            artifacts=dict(artifacts or {}),
            metadata=dict(metadata or {}),
        )

    def _predict_evaluation_trajectory(self, *, adata, data, time_points):
        device = self.device
        for param in self.model.parameters():
            param.requires_grad = False
        observed_time_points = [float(t) for t in time_points]
        trajectory_time_points = self._build_evaluation_trajectory_time_points(observed_time_points)
        trajectory_step = self._evaluation_trajectory_step()

        if self.simulation_hook is not None:
            inference_context = self._build_inference_context(
                adata=adata,
                data=data,
                time_points=time_points,
            )
            safe_adata = adata
            safe_training_data = self.training_data
            if self.inference_context_builder is not None:
                safe_adata = self._t0_adata_view(adata)
                safe_training_data = self._t0_training_data_bundle(adata=adata, data=data, time_points=time_points)
            context = SimulationContext(
                adata=safe_adata,
                model=self.model,
                training_data=safe_training_data,
                config=self.config,
                time_points=[float(t) for t in time_points],
                observed_time_points=observed_time_points,
                trajectory_time_points=trajectory_time_points,
                trajectory_step=trajectory_step,
                x0_override=data[0],
                device=device,
                inference_context=inference_context,
                metadata={
                    "source": "TrainingPipeline.evaluate",
                    "inference_context_builder_active": self.inference_context_builder is not None,
                    "simulation_data_scope": "t0_only" if self.inference_context_builder is not None else "legacy_full_context",
                    "prediction_contract": "full_t0_to_final_trajectory",
                    "trajectory_step": trajectory_step,
                },
            )
            result = self.simulation_hook(context)
            if isinstance(result, dict):
                result = SimulationResult(**result)
            if not isinstance(result, SimulationResult):
                raise TypeError(
                    f"simulation_hook must return SimulationResult or dict, got {type(result).__name__}"
                )
            actual_times = result.trajectory_time_points or trajectory_time_points
            return self._coerce_evaluation_trajectory(
                predicted_points_by_time=result.predicted_points_by_time,
                predicted_weights_by_time=result.predicted_weights_by_time,
                trajectory_time_points=actual_times,
                observed_time_points=observed_time_points,
                source="simulation_hook",
                artifacts=result.artifacts,
                metadata={
                    "aux_state_keys": sorted((result.aux_state_by_time or {}).keys()),
                    "simulation_hook": True,
                },
            )

        x0 = data[0].to(device)
        sigma = self._evaluation_sigma()
        point, weight = simulate_trajectory(
            adata,
            self.model,
            x0,
            sigma,
            trajectory_time_points,
            dt=self._evaluation_integration_dt(trajectory_step),
            device=x0.device
        )
        return self._coerce_evaluation_trajectory(
            predicted_points_by_time=point,
            predicted_weights_by_time=weight,
            trajectory_time_points=trajectory_time_points,
            observed_time_points=observed_time_points,
            source="builtin_simulate_trajectory",
            metadata={"simulation_hook": False, "sigma": float(sigma)},
        )

    def _predict_for_evaluation(self, *, adata, data, time_points):
        trajectory = self._predict_evaluation_trajectory(adata=adata, data=data, time_points=time_points)
        return trajectory.observed_points_by_time(), trajectory.observed_weights_by_time()

    def _t0_adata_view(self, adata):
        obs_indices = []
        try:
            obs_indices = list(self.training_data.obs_indices_by_time[0])
        except Exception:
            obs_indices = []
        if obs_indices:
            try:
                return adata[obs_indices].copy()
            except Exception:
                pass
        try:
            time_key = "time_point_processed"
            first_time = self.training_data.time_points[0]
            mask = np.asarray(adata.obs[time_key].values, dtype=np.float32) == float(first_time)
            return adata[mask].copy()
        except Exception:
            return adata

    def _t0_training_data_bundle(self, *, adata, data, time_points):
        obs_indices = []
        try:
            obs_indices = list(self.training_data.obs_indices_by_time[0])
        except Exception:
            obs_indices = []
        extra_modalities = {}
        for key, values in dict(getattr(self.training_data, "extra_modalities_by_time", {}) or {}).items():
            if isinstance(values, (list, tuple)) and values:
                extra_modalities[key] = [values[0]]
        metadata = dict(getattr(self.training_data, "metadata", {}) or {})
        metadata.update(
            {
                "source": "TrainingPipeline._t0_training_data_bundle",
                "scope": "t0_only",
                "full_time_points": [float(t) for t in time_points],
            }
        )
        return TrainingDataBundle(
            adata=self._t0_adata_view(adata),
            time_points=[float(time_points[0])],
            latent_by_time=[data[0]],
            obs_indices_by_time=[obs_indices],
            extra_modalities_by_time=extra_modalities,
            metadata=metadata,
        )

    def _build_inference_context(self, *, adata, data, time_points):
        if self.inference_context_builder is None:
            return None
        t0_adata = self._t0_adata_view(adata)
        try:
            obs_indices = list(self.training_data.obs_indices_by_time[0])
        except Exception:
            obs_indices = []
        context = InferenceContextBuilderContext(
            t0_adata=t0_adata,
            initial_data=data[0],
            initial_obs_indices=obs_indices,
            config=self.config,
            time_points=[float(t) for t in time_points],
            device=self.device,
            metadata={
                "source": "TrainingPipeline._build_inference_context",
                "scope": "t0_inference_builder",
                "allowed_sources": [
                    "t0_inference",
                    "exogenous_inference",
                    "constant_prior",
                    "model_state",
                ],
            },
        )
        result = self.inference_context_builder(context)
        if result is None:
            result = InferenceContext(
                payload={},
                visibility={},
                provenance={
                    "builder_returned": "None",
                    "future_rows_used": False,
                },
                notes="Empty inference context returned by builder.",
            )
        elif isinstance(result, dict):
            if any(key in result for key in ("payload", "visibility", "provenance", "notes")):
                result = InferenceContext(**result)
            else:
                result = InferenceContext(
                    payload=dict(result),
                    visibility={},
                    provenance={
                        "builder_returned": "raw_payload_dict",
                        "future_rows_used": False,
                    },
                )
        elif not isinstance(result, InferenceContext):
            raise TypeError(
                "inference_context_builder must return InferenceContext, dict, or None, "
                f"got {type(result).__name__}"
            )
        if not isinstance(result.payload, dict) or not isinstance(result.provenance, dict):
            raise TypeError("InferenceContext.payload and InferenceContext.provenance must be dictionaries.")
        return result

    def _compute_builtin_metrics(
        self,
        *,
        data,
        time_points,
        evaluation_trajectory,
    ):
        wasserstein_scores = []
        tmv_scores = []
        timepoint_results = []
        initial_trajectory_idx = evaluation_trajectory.observed_time_indices[0] if evaluation_trajectory.observed_time_indices else 0
        predicted_initial_mass = float(
            np.asarray(evaluation_trajectory.weights_by_time[initial_trajectory_idx]).reshape(-1).sum()
        )
        if not np.isfinite(predicted_initial_mass) or predicted_initial_mass <= 0:
            raise ValueError("Evaluation trajectory has invalid initial predicted mass.")
        w1_pair_sizes = []
        for idx in range(1, len(time_points)):
            trajectory_idx = evaluation_trajectory.observed_time_indices[idx]
            predicted_count = int(np.asarray(evaluation_trajectory.points_by_time[trajectory_idx]).shape[0])
            observed_count = int(data[idx].shape[0])
            w1_pair_sizes.append((observed_count, predicted_count))
        w1_backend_policy = select_w1_backend_policy(w1_pair_sizes, selection_scope="evaluation_panel")
        for idx in range(1, len(time_points)):
            self._raise_if_inference_time_budget_exceeded()
            t1 = time_points[idx]
            data_t1 = data[idx].detach().cpu().numpy()
            trajectory_idx = evaluation_trajectory.observed_time_indices[idx]
            x1 = evaluation_trajectory.points_by_time[trajectory_idx]
            raw_m1 = evaluation_trajectory.weights_by_time[trajectory_idx]

            predicted_relative_mass = float(raw_m1.sum()) / predicted_initial_mass
            observed_relative_mass = data[idx].shape[0] / data[0].shape[0]
            tmv = np.abs(predicted_relative_mass - observed_relative_mass)
            m1 = raw_m1 / raw_m1.sum()
            m2 = np.ones(data_t1.shape[0]) / data_t1.shape[0]

            w1 = compute_w1_distance(
                data_t1,
                x1,
                observed_weights=m2,
                predicted_weights=m1.reshape(-1),
                backend_policy=w1_backend_policy,
                device=self.device,
            )

            wasserstein_scores.append(float(w1))
            tmv_scores.append(float(tmv))
            timepoint_results.append(
                EvaluationTimepointResult(
                    time_index=idx,
                    time_point=float(t1),
                    observed_points=data_t1,
                    predicted_points=x1,
                    predicted_weights=raw_m1.reshape(-1),
                    normalized_predicted_weights=m1.reshape(-1),
                    observed_uniform_weights=m2.reshape(-1),
                    w1=float(w1),
                    tmv=float(tmv),
                    trajectory_time_index=int(trajectory_idx),
                    trajectory_time_point=float(evaluation_trajectory.time_points[trajectory_idx]),
                )
            )
            print(f"  Time Point {t1}: Wasserstein-1 Distance = {w1:.4f} ({w1_backend_policy['w1_backend']})")
            print(f"  Time Point {t1}: TMV = {tmv:.4f}")
            self._raise_if_inference_time_budget_exceeded()

        metrics = {
            'w1_scores': wasserstein_scores,
            'tmv_scores': tmv_scores,
            'evaluation_prediction_contract': 'full_t0_to_final_trajectory',
            'evaluation_trajectory_source': evaluation_trajectory.source,
            'evaluation_trajectory_step': evaluation_trajectory.step,
            'evaluation_sigma': float(evaluation_trajectory.metadata.get("sigma", 0.0)),
            'evaluation_trajectory_time_points': [float(t) for t in evaluation_trajectory.time_points],
            'evaluation_observed_time_indices': [int(i) for i in evaluation_trajectory.observed_time_indices],
        }
        metrics.update(w1_backend_metric_metadata(w1_backend_policy))
        if wasserstein_scores:
            metrics["w1_mean"] = float(np.mean(wasserstein_scores))
        if tmv_scores:
            metrics["tmv_mean"] = float(np.mean(tmv_scores))
            metrics["tmv_max"] = float(np.max(tmv_scores))
        return metrics, timepoint_results

    
    # def generate_state_trajectory(self, data, time_points):
    #     """Generate reference trajectory without score guidance (using only velocity and growth components)
        
    #     Args:
    #         data: List of tensors where each element represents samples at a specific time point
    #         time_points: List of time values corresponding to each element in 'data'
        
    #     Returns:
    #         list: List of tensors representing predicted positions at each time point (detached from graph)
    #     """
    #     # Get initial time point data (t=0)
    #     x0 = data[0].to(self.device)
    #     n_samples = x0.shape[0]

    #     # Initialize ODE function (force disable score component)
    #     ode_func = ODEFunc(
    #         model=self.model,
    #         sigma=self.config['training']['defaults'].get('sigma', 0.1),
    #         use_mass=self.use_mass,
    #         score_use=False,
    #         score_flow_matching_use=False
    #     ).to(self.device)

    #     # ODE initial state: (positions, log-weights, mass)
    #     init_lnw = torch.log(torch.ones(n_samples, 1) / n_samples).to(self.device)
    #     init_m = torch.zeros_like(init_lnw).to(self.device)
    #     initial_state = (x0, init_lnw, init_m)

    #     # Solve ODE to get full trajectory
    #     t_eval = torch.tensor(time_points, device=self.device, dtype=torch.float32)
    #     traj_x, _, _ = odeint(
    #         func=ode_func,
    #         y0=initial_state,
    #         t=t_eval,
    #         method='euler'  # Euler method for ODE solving (fast but less accurate)
    #     )

    #     # Split trajectory by time point and detach from computation graph (avoid memory leaks)
    #     return [traj_x[i].detach() for i in range(len(time_points))]


    # def generate_state_trajectory1(self, data, time_points, reg=None, reg_m=None, method='sinkhorn', numItermax=1000,
    #                                stopThr=1e-6, **kwargs):
    #     """Generate trajectory with Unbalanced Sinkhorn matching (fixed: ensure valid trajectory output + add error handling)
        
    #     Args:
    #         data: List of tensors where each element represents samples at a specific time point
    #         time_points: List of time values corresponding to each element in 'data'
    #         reg: Regularization parameter for Sinkhorn (auto-calculated if None)
    #         reg_m: Mass regularization parameter for Unbalanced Sinkhorn (auto-calculated if None)
    #         method: Matching method (default: 'sinkhorn')
    #         numItermax: Maximum number of iterations for Sinkhorn
    #         stopThr: Convergence threshold for Sinkhorn
    #         **kwargs: Additional keyword arguments
        
    #     Returns:
    #         list: List of tensors representing matched trajectory (detached from graph)
    #     """
    #     try:
    #         # 1. Get number of samples at each time point
    #         max_iter = numItermax
    #         tol = stopThr
    #         data_sizes = [d.shape[0] for d in data]
    #         raw_masses = data_sizes

    #         # 2. Determine sample sizes for each time point (balance between speed and accuracy)
    #         min_size = min(data_sizes)
    #         max_size = max(data_sizes)
    #         print("min_size", min_size, "max_size", max_size)
            
    #         if min_size >= 1024:
    #             # Scale sample sizes proportionally if minimum size is ≥1024
    #             sample_sizes = [max(1, int(round(1024 * s / min_size))) for s in data_sizes]
    #         elif max_size >= 1024:
    #             # Cap sample sizes at max_size if maximum size is ≥1024 (avoid oversampling)
    #             sample_sizes = []
    #             for s in data_sizes:
    #                 target = max(1, int(round(1024 * s / min_size)))
    #                 target = min(target, s)
    #                 sample_sizes.append(target)
    #         else:
    #             # Use original sample sizes if all are <1024
    #             sample_sizes = data_sizes

    #         # 3. Sample data for each time point (ensure consistent batch size)
    #         sampled_data = []
    #         for t_idx in range(len(time_points)):
    #             size = sample_sizes[t_idx]
    #             data_t = data[t_idx].to(self.device)
    #             # Oversample if current time point has fewer samples than target size
    #             if data_t.shape[0] < size:
    #                 indices = torch.randint(0, data_t.shape[0], (size,), device=self.device)
    #             else:
    #                 # Undersample if current time point has more samples than target size
    #                 indices = torch.randperm(data_t.shape[0], device=self.device)[:size]
    #             sampled = data_t[indices]
    #             sampled_data.append(sampled)

    #         # 4. Unbalanced Sinkhorn matching to connect time points
    #         matched_trajectories = [sampled_data[0]]  # Initialize trajectory with first time point

    #         # Iterate over time points to match consecutive time steps
    #         for t_idx in range(1, len(time_points)):
    #             t_prev = time_points[t_idx - 1]
    #             t_curr = time_points[t_idx]
    #             prev_points = matched_trajectories[-1]  # Points from previous time point
    #             curr_points = sampled_data[t_idx]  # Points from current time point
    #             n, m = prev_points.shape[0], curr_points.shape[0]

    #             # Convert tensors to numpy arrays (required for OT library)
    #             prev_np = prev_points.cpu().detach().numpy()
    #             curr_np = curr_points.cpu().detach().numpy()

    #             # Get true mass values for previous and current time points
    #             prev_mass = raw_masses[t_idx - 1]
    #             curr_mass = raw_masses[t_idx]

    #             # Auto-calculate regularization parameters if not provided
    #             if reg is None or reg_m is None:
    #                 auto_reg, auto_reg_m = self.calculate_auto_regularization(prev_np, curr_np, prev_mass, curr_mass)
    #                 reg = auto_reg if reg is None else reg
    #                 reg_m = auto_reg_m if reg_m is None else reg_m
    #                 print(f"Auto-calculated regularization: reg={reg:.4f}, reg_m={reg_m:.4f}")
    #             else:
    #                 print(f"User-specified regularization: reg={reg:.4f}, reg_m={reg_m:.4f}")

    #             # Predict source weights (a) using growth network (fixed: ensure correct weight calculation)
    #             # a. Initialize log-weights for previous time point (uniform distribution)
    #             lnw_prev_init = torch.log(torch.ones(n, 1, device=self.device) / n)
    #             # b. Define time interval for ODE solving (from previous to current time point)
    #             t_interval = torch.tensor([t_prev, t_curr], device=self.device, dtype=torch.float32)
    #             # c. Initialize ODE state (positions, log-weights, mass)
    #             initial_state = (prev_points, lnw_prev_init, torch.zeros_like(lnw_prev_init, device=self.device))
    #             # d. Solve ODE to get predicted log-weights at current time point
    #             traj_x, traj_lnw, _ = odeint(
    #                 func=self.ode_func,
    #                 y0=initial_state,
    #                 t=t_interval,
    #                 method='euler'
    #             )
    #             # e. Convert log-weights to probabilities (normalize to sum to 1)
    #             lnw_prev_pred = traj_lnw[-1]
    #             mu_prev = torch.exp(lnw_prev_pred)
    #             mu_prev = mu_prev / mu_prev.sum()
    #             a = mu_prev.cpu().detach().numpy().squeeze()  # Source weights (1D array)

    #             # Target weights (b): uniform distribution over current time point samples
    #             nu_curr = torch.ones(m, 1, device=self.device) / m
    #             b = nu_curr.cpu().detach().numpy().squeeze()  # Target weights (1D array)

    #             # Compute Euclidean distance matrix between previous and current points
    #             M = ot.dist(prev_np, curr_np)
    #             # Solve Unbalanced Sinkhorn to get transport matrix
    #             transport_matrix = ot.unbalanced.sinkhorn_unbalanced(
    #                 a, b, M, reg, reg_m,
    #                 numItermax=max_iter, stopThr=tol
    #             )

    #             # Match current time point points to previous time point (max weight in transport matrix)
    #             sinkhorn_result = torch.tensor(transport_matrix, device=self.device)
    #             matched_indices = torch.argmax(sinkhorn_result, dim=1)  # For each previous point, find best current point
    #             matched_points = curr_points[matched_indices]
    #             matched_trajectories.append(matched_points)

    #         # Ensure trajectory is not empty (raise error if no points were generated)
    #         if not matched_trajectories:
    #             raise ValueError("Trajectory generation failed: no points were generated")

    #         # Detach all points from computation graph and return trajectory
    #         trajectory = [points.detach() for points in matched_trajectories]
    #         return trajectory

    #     except Exception as e:
    #         # Print error message and return original data (detached) as fallback
    #         print(f"Trajectory generation encountered an error: {str(e)}")
    #         return [data[t_idx].detach() for t_idx in range(len(time_points))]


    # def visualize_trajectory(self, trajectory, trajectory_times):
    #     """Visualize trajectory with scatter plots (time-colored) and connecting lines for each trajectory chain
        
    #     Args:
    #         trajectory: List of tensors where each element represents predicted positions at a specific time point
    #         trajectory_times: List of time values corresponding to each element in 'trajectory'
    #     """
    #     import matplotlib.pyplot as plt
    #     import numpy as np

    #     # 1. Prepare data for scatter plot (combine all time points)
    #     all_data = np.concatenate([x.cpu().detach().numpy() for x in trajectory], axis=0)
    #     # Create time labels for color coding (each sample gets its corresponding time point)
    #     time_labels = np.concatenate([np.full(x.shape[0], t.item()) for x, t in zip(trajectory, trajectory_times)])

    #     # 2. Create plot and scatter plot (time-colored points)
    #     plt.figure(figsize=(8, 6))
    #     # Scatter plot: color by time point, semi-transparent, higher z-order (on top of lines)
    #     scatter = plt.scatter(
    #         all_data[:, 0],
    #         all_data[:, 1],
    #         c=time_labels,
    #         cmap='viridis',
    #         alpha=0.6,
    #         zorder=2
    #     )
    #     # Add color bar to indicate time point mapping
    #     plt.colorbar(scatter, label='Time Point')

    #     # 3. Add connecting lines for each trajectory chain (same sample across time points)
    #     # Reshape trajectory to (num_time_points, num_trajectories, 2) for easy indexing
    #     traj_matrix = np.concatenate([
    #         pts.cpu().detach().numpy()[:, :2][None, ...]  # Shape: (1, num_trajectories, 2)
    #         for pts in trajectory
    #     ], axis=0)  # Final shape: (num_time_points, num_trajectories, 2)
    #     T, n_traj = traj_matrix.shape[:2]

    #     # Plot line for each trajectory chain (low alpha + low z-order to not obscure scatter points)
    #     for traj_id in range(n_traj):
    #         plt.plot(
    #             traj_matrix[:, traj_id, 0],  # X-coordinates across time
    #             traj_matrix[:, traj_id, 1],  # Y-coordinates across time
    #             color='black',
    #             linewidth=0.8,
    #             alpha=0.4,
    #             zorder=1
    #         )

    #     # Add plot labels and title
    #     plt.xlabel('Latent Dimension 1')
    #     plt.ylabel('Latent Dimension 2')
    #     plt.title('Trajectory Visualization (lines connect same chain)')
    #     # Save plot (high DPI for clarity, tight layout to avoid label cutoff)
    #     plt.savefig("/home/sjt/workspace2/CytoBridge_test_main/figures/tra_test.png", dpi=300, bbox_inches='tight')


    # def _plot_snapshot(self, epoch, stage_params, data, time_points, exp_fig_dir):
    #     """Plot SDE trajectory and score field at specific time points for current epoch
        
    #     Args:
    #         epoch: Current training epoch (for filename labeling)
    #         stage_params: Stage-specific parameters (not used directly but kept for consistency)
    #         data: Input data (not used directly but kept for consistency)
    #         time_points: List of time points (not used directly but kept for consistency)
    #         exp_fig_dir: Directory to save plot files
    #     """
    #     # Create directory for figures if it doesn't exist
    #     os.makedirs(exp_fig_dir, exist_ok=True)

    #     # 3. Plot score field for time points t=0,1,2,3,4
    #     for t in [0, 1, 2, 3, 4]:
    #         # Define save path with epoch and time point labels
    #         save_score = os.path.join(exp_fig_dir, f"score_epoch{epoch}_t{t}.png")
    #         # Generate and save score field plot
    #         plot_score_and_gradient(
    #             dynamical_model=self.model,
    #             device=self.device,
    #             t_value=float(t),  # Time point to visualize
    #             x_range=(0, 2.5),  # X-axis range for grid
    #             y_range=(0, 2.5),  # Y-axis range for grid
    #             save_path=save_score,
    #             cmap='rainbow'  # Color map for score visualization
    #         )
