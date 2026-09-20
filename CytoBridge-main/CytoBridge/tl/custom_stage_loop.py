from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import os

import torch

from CytoBridge.tl.training_algorithm import StageRunnerContext, StageRunnerResult


@dataclass
class CustomStageLossResult:
    """Loss payload returned by a custom-stage objective."""

    loss: torch.Tensor
    logs: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, torch.nn.Module, torch.nn.Parameter)):
        return [value]
    if isinstance(value, dict):
        items: list[Any] = []
        for item in value.values():
            items.extend(_as_list(item))
        return items
    try:
        return list(value)
    except TypeError:
        return [value]


def _module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    current: Any = model
    for part in str(path).split("."):
        if not part:
            continue
        current = getattr(current, part)
    if not isinstance(current, torch.nn.Module):
        raise TypeError(f"Trainable module path {path!r} did not resolve to torch.nn.Module")
    return current


def _component_selected(key: str, flags: dict[str, bool]) -> bool:
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
    component = aliases.get(str(key).strip().lower(), str(key).strip().lower())
    return component == "always" or bool(flags.get(component, False))


def _coerce_trainable_selection(selection: Any) -> tuple[list[Any], list[torch.nn.Parameter]]:
    if selection is None:
        return [], []
    if isinstance(selection, dict):
        modules = _as_list(selection.get("modules"))
        params = _as_list(selection.get("parameters"))
        if "modules" not in selection and "parameters" not in selection:
            modules = _as_list(selection)
        return modules, [p for p in params if isinstance(p, torch.nn.Parameter)]
    specs = _as_list(selection)
    modules = [item for item in specs if not isinstance(item, torch.nn.Parameter)]
    params = [item for item in specs if isinstance(item, torch.nn.Parameter)]
    return modules, params


def select_custom_stage_parameters(
    model: torch.nn.Module,
    stage_params: dict[str, Any],
    *,
    objective: Any = None,
    state: Any = None,
) -> list[torch.nn.Parameter]:
    """Select trainable parameters for a custom stage using package conventions.

    This mirrors the standard trainer conventions so custom stages do not need
    to hand-roll optimizer ownership. The objective may override selection with
    `trainable_parameters(model=..., stage_params=..., state=...)`.
    """

    train_strategy = str(stage_params.get("train_strategy", "")).lower()
    if not train_strategy or train_strategy == "none":
        flags = {"velocity": True, "growth": True, "score": True, "interaction": True}
    else:
        flags = {
            "velocity": "v" in train_strategy,
            "growth": "g" in train_strategy,
            "score": "s" in train_strategy,
            "interaction": "i" in train_strategy,
        }

    for param in model.parameters():
        param.requires_grad = False

    selected: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    def add_param(param: torch.nn.Parameter) -> None:
        if not isinstance(param, torch.nn.Parameter):
            raise TypeError(f"Expected torch.nn.Parameter, got {type(param).__name__}")
        if id(param) not in seen:
            param.requires_grad = True
            selected.append(param)
            seen.add(id(param))

    if bool(stage_params.get("train_all_parameters", False)):
        for param in model.parameters():
            add_param(param)
    else:
        modules: list[Any] = []
        params: list[torch.nn.Parameter] = []
        selector = getattr(objective, "trainable_parameters", None)
        if callable(selector):
            try:
                selection = selector(model=model, stage_params=dict(stage_params), state=state)
            except TypeError:
                selection = selector(model, dict(stage_params), state)
            modules, params = _coerce_trainable_selection(selection)
        else:
            model_selector = getattr(model, "cytobridge_trainable_parameters", None)
            if callable(model_selector):
                modules, params = _coerce_trainable_selection(
                    model_selector(stage_params=dict(stage_params), train_flags=dict(flags))
                )
            else:
                defaults = {
                    "velocity": ["velocity_net"],
                    "growth": ["growth_net"],
                    "score": ["score_net"],
                    "interaction": ["interaction_net"],
                }
                for component, names in defaults.items():
                    if flags.get(component):
                        modules.extend(name for name in names if hasattr(model, name))
                model_map = getattr(model, "cytobridge_component_modules", None)
                if model_map is None:
                    model_map = getattr(model, "component_trainable_modules", None)
                if callable(model_map):
                    model_map = model_map(stage_params=dict(stage_params), train_flags=dict(flags))
                if isinstance(model_map, dict):
                    for component, value in model_map.items():
                        if _component_selected(component, flags):
                            modules.extend(_as_list(value))

        stage_map = (
            stage_params.get("component_trainable_modules")
            or stage_params.get("cytobridge_component_modules")
            or {}
        )
        if isinstance(stage_map, dict):
            for component, value in stage_map.items():
                if _component_selected(component, flags):
                    modules.extend(_as_list(value))
        modules.extend(_as_list(stage_params.get("trainable_modules")))
        modules.extend(_as_list(stage_params.get("extra_trainable_modules")))

        for param in params:
            add_param(param)
        for item in modules:
            module = _module_by_path(model, item) if isinstance(item, str) else item
            if not isinstance(module, torch.nn.Module):
                raise TypeError(f"Trainable module spec {item!r} did not resolve to torch.nn.Module")
            for param in module.parameters():
                add_param(param)

    if not selected:
        raise RuntimeError(
            "No trainable parameters selected for custom stage. Set train_strategy, "
            "trainable_modules, train_all_parameters, model.cytobridge_component_modules, "
            "model.cytobridge_trainable_parameters(...), or objective.trainable_parameters(...)."
        )
    return selected


def _coerce_loss_result(value: Any) -> CustomStageLossResult:
    if isinstance(value, CustomStageLossResult):
        return value
    if torch.is_tensor(value):
        return CustomStageLossResult(loss=value)
    if isinstance(value, dict):
        if "loss" not in value:
            raise ValueError("Custom-stage loss dict must contain a 'loss' key.")
        return CustomStageLossResult(
            loss=value["loss"],
            logs=dict(value.get("logs") or {}),
            artifacts=dict(value.get("artifacts") or {}),
        )
    raise TypeError(
        "Custom-stage objective.compute_loss(...) must return a tensor, dict, "
        f"or CustomStageLossResult, got {type(value).__name__}."
    )


def _scalar_loss(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"Custom-stage loss must be a torch.Tensor, got {type(value).__name__}.")
    if value.numel() != 1:
        raise ValueError(f"Custom-stage loss must be scalar, got shape {tuple(value.shape)}.")
    if not torch.isfinite(value.detach()).all():
        raise ValueError("Custom-stage loss must be finite.")
    return value


def run_custom_stage_loop(context: StageRunnerContext, objective: Any) -> StageRunnerResult:
    """Run a custom objective under the package-owned training loop.

    The objective supplies algorithm-specific state, batches, and loss:
    - `build_state(context)` optional
    - `sample_batch(context, state, epoch)` required for training
    - `compute_loss(context, state, batch, epoch)` required for training
    - `evaluate_preview(context, state)` optional for preview-only calls

    The package owns optimizer setup, train/eval mode, device, checkpoint,
    timeout checks, progress callback, gradient clipping, scheduler stepping,
    and standardized `StageRunnerResult` formatting.
    """

    stage_params = dict(context.stage_params)
    stage_name = str(stage_params.get("name", f"custom_stage_{context.stage_index}"))
    total_epochs = int(stage_params.get("epochs", 0))
    save_strategy = str(stage_params.get("save_strategy", "best")).lower()
    lr = float(stage_params.get("lr", 1e-3))
    os.makedirs(context.output_dir, exist_ok=True)

    build_state = getattr(objective, "build_state", None)
    state = build_state(context) if callable(build_state) else None
    if context.preview_only:
        evaluate_preview = getattr(objective, "evaluate_preview", None)
        preview = evaluate_preview(context, state) if callable(evaluate_preview) else None
        if isinstance(preview, StageRunnerResult):
            return preview
        if isinstance(preview, dict):
            return StageRunnerResult(
                stage_summary={
                    "name": stage_name,
                    "mode": stage_params.get("mode", "custom"),
                    "preview_only": True,
                },
                logs=dict(preview.get("logs") or preview),
                artifacts=dict(preview.get("artifacts") or {}),
                model_outputs=dict(preview.get("model_outputs") or {}),
            )
        return StageRunnerResult(
            stage_summary={
                "name": stage_name,
                "mode": stage_params.get("mode", "custom"),
                "preview_only": True,
                "completed_epochs": 0,
            }
        )

    params = select_custom_stage_parameters(
        context.model,
        stage_params,
        objective=objective,
        state=state,
    )
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = None
    if stage_params.get("scheduler_type") == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(stage_params.get("scheduler_step_size", 100)),
            gamma=float(stage_params.get("scheduler_gamma", 0.5)),
        )
    elif stage_params.get("scheduler_type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(stage_params.get("cosine_epochs", max(1, total_epochs))),
            eta_min=float(stage_params.get("eta_min", 1e-5)),
        )

    best_loss = float("inf")
    best_state = None
    last_loss = float("nan")
    completed_epochs = 0
    logs: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    max_grad_norm = stage_params.get("max_grad_norm", stage_params.get("grad_clip_norm"))

    context.model.train()
    for epoch in range(total_epochs):
        if context.should_stop is not None and context.should_stop():
            break
        sample_batch = getattr(objective, "sample_batch", None)
        compute_loss = getattr(objective, "compute_loss", None)
        if not callable(sample_batch) or not callable(compute_loss):
            raise AttributeError(
                "Custom-stage objective must define sample_batch(context, state, epoch) "
                "and compute_loss(context, state, batch, epoch)."
            )
        batch = sample_batch(context, state, epoch)
        raw_loss = compute_loss(context, state, batch, epoch)
        loss_result = _coerce_loss_result(raw_loss)
        loss = _scalar_loss(loss_result.loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_value = float(loss.detach().cpu().item())
        last_loss = loss_value
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                key: value.detach().clone() if torch.is_tensor(value) else value
                for key, value in context.model.state_dict().items()
            }
        completed_epochs = epoch + 1
        if loss_result.logs:
            logs.append({"epoch": completed_epochs, **dict(loss_result.logs)})
        if loss_result.artifacts:
            artifacts.update(dict(loss_result.artifacts))
        if context.progress_callback and (epoch == 0 or completed_epochs % 10 == 0):
            context.progress_callback(
                f"Custom Stage {stage_name} Epoch {completed_epochs}/{total_epochs}, Loss: {loss_value:.4f}",
                completed_epochs / max(1, total_epochs),
            )

    save_state = best_state if save_strategy == "best" and best_state is not None else {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in context.model.state_dict().items()
    }
    if save_state is not None:
        context.model.load_state_dict(save_state)
    ckpt_filename = "best_model.pth" if save_strategy == "best" else "last_model.pth"
    ckpt_path = os.path.join(context.output_dir, ckpt_filename)
    torch.save(save_state, ckpt_path)
    artifacts.setdefault("checkpoint", ckpt_path)

    return StageRunnerResult(
        stage_summary={
            "name": stage_name,
            "mode": stage_params.get("mode", "custom"),
            "save_strategy": save_strategy,
            "epochs": total_epochs,
            "completed_epochs": completed_epochs,
            "best_loss": float(best_loss),
            "last_epoch_loss": float(last_loss),
            "saved_loss": float(best_loss if save_strategy == "best" else last_loss),
            "stopped_early_timeout": bool(context.should_stop() if context.should_stop else False),
            "runner": "run_custom_stage_loop",
        },
        logs={"loss": logs},
        artifacts=artifacts,
    )
