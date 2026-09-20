import random
import torch
import numpy as np
import math
import os
import json
from copy import deepcopy
from anndata import AnnData
from torchdiffeq import odeint
from sklearn.decomposition import PCA

from ..tl.models import DynamicalModel

def trace_df_dz(f, z):
    sum_diag = 0.0
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0][:, i]
    return sum_diag

def compute_integral(x, time, V, rho_net, num_samples=100, sigma=0.1, sigmaa=1.0):
    batch_size, dim = x.shape
    noise = torch.randn(batch_size, num_samples, dim, device=x.device)
    y = sigma * noise
    perterbed_x = x.unsqueeze(1) + sigma * noise
    perterbed_x_flat = perterbed_x.reshape(-1, dim)
    time_repeated = time.repeat(num_samples, 1)
    ss = rho_net(time_repeated, perterbed_x_flat)
    rrho = torch.exp(ss * 2 / (sigmaa ** 2))
    rrho = rrho.reshape(batch_size, num_samples, 1)
    y_flat = y.reshape(-1, dim)
    y_flat.requires_grad_(True)

    v = V(y_flat)
    if v.dim() > 1:
        v = v.squeeze(-1)
    grad_y = torch.autograd.grad(v.sum(), y_flat, create_graph=False)[0]
    F = -grad_y
    F = F.view(batch_size, num_samples, dim)
    norm_constant = (2 * math.pi) ** (dim / 2) * (sigma ** dim)
    q = torch.exp(-0.5 * (noise ** 2).sum(dim=-1)) / norm_constant
    contributions = (rrho * F) / q.unsqueeze(-1)
    integral_estimate = contributions.mean(dim=1)
    return integral_estimate

# --------------------------
def set_seed(seed=42):
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    os.environ['PYTHONHASHSEED'] = str(seed)

def sample(data, size):
    N = data.shape[0]
    indices = torch.tensor(random.sample(range(N), size))
    sampled = data[indices]
    return sampled

def check_submodules(model):
    growth_net_exists = hasattr(model, 'growth_net')
    score_net_exists = hasattr(model, 'score_net')
    interaction_net_exists = hasattr(model, 'interaction_net')

    return growth_net_exists, score_net_exists, interaction_net_exists


def _coerce_training_config(training_config: dict | None) -> dict:
    config = deepcopy(training_config or {})
    if isinstance(config.get('plan'), str):
        config['plan'] = json.loads(config['plan'])
    return config


def _coerce_state_dict(state_dict_np: dict[str, np.ndarray] | None) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v) for k, v in (state_dict_np or {}).items()}


def _resolve_all_model_record(adata: AnnData) -> dict:
    if 'all_model' not in adata.uns or not isinstance(adata.uns['all_model'], dict):
        raise ValueError("CytoBridge model information not found. Please fit the model first.")
    return dict(adata.uns['all_model'])


def _resolved_config_from_record(record: dict) -> dict:
    resolved_yaml = record.get("resolved_config_yaml")
    if isinstance(resolved_yaml, str) and resolved_yaml.strip():
        try:
            import yaml

            loaded = yaml.safe_load(resolved_yaml) or {}
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
    return {
        "model": deepcopy(record.get("model_config") or {}),
        "training": _coerce_training_config(record.get("training_config")),
    }


def resolved_config_from_adata(adata: AnnData) -> dict:
    """Return the resolved training config serialized with a fitted model."""
    return _resolved_config_from_record(_resolve_all_model_record(adata))


def load_model_from_adata(adata: AnnData) -> torch.nn.Module:
    record = _resolve_all_model_record(adata)
    print("Reconstructing model...")

    provider = str(record.get("provider") or "").strip()
    schema_version = int(record.get("schema_version") or 0)
    dim = adata.obsm['X_latent'].shape[1]

    # Legacy builtin format.
    if not provider and 'model_config' in record:
        training_config = _coerce_training_config(record.get('training_config'))
        model_config = record['model_config']
        model = DynamicalModel(dim, model_config)
        state_dict_torch = _coerce_state_dict(record.get('model_state_dict'))
        model.load_state_dict(state_dict_torch)
        model.eval()
        print("Model loaded successfully.")
        return model

    if schema_version >= 2 and provider == "builtin":
        model_config = record.get("model_config") or (_resolved_config_from_record(record).get("model") or {})
        model = DynamicalModel(dim, model_config)
        model.load_state_dict(_coerce_state_dict(record.get("model_state_dict")))
        model.eval()
        print("Model loaded successfully.")
        return model

    if schema_version >= 2 and provider == "training_algorithm":
        algorithm_id = str(record.get("algorithm_id") or "").strip()
        if not algorithm_id:
            raise ValueError("Custom model record is missing algorithm_id.")

        from CytoBridge.tl.fit import resolve_training_data_bundle
        from CytoBridge.tl.training_algorithm import (
            ModelBuildContext,
            ModelLoadContext,
            TrainingAlgorithmContext,
        )
        from CytoBridge.tl.training_algorithm_registry import (
            default_training_algorithm_roots,
            load_training_algorithm,
            resolve_base_config,
        )

        loaded = load_training_algorithm(algorithm_id, default_training_algorithm_roots())
        resolved_base_config, base_config_name = resolve_base_config(
            loaded.manifest["base_config"],
            config_dir=loaded.root_dir,
        )
        stage = str(record.get("stage") or "final")
        spec = loaded.build_spec(
            TrainingAlgorithmContext(
                algorithm_id=algorithm_id,
                input_adata_path="",
                output_dir="",
                stage=stage,
                base_config_name=base_config_name,
                resolved_base_config=deepcopy(resolved_base_config),
                metadata={"source": "load_model_from_adata"},
            )
        )
        resolved_config = _resolved_config_from_record(record)
        state_dict_torch = _coerce_state_dict(record.get("model_state_dict"))
        if spec.deserialize_model is not None:
            model = spec.deserialize_model(
                ModelLoadContext(
                    algorithm_id=algorithm_id,
                    adata=adata,
                    resolved_config=deepcopy(resolved_config),
                    latent_dim=int(dim),
                    device=torch.device("cpu"),
                    model_payload=deepcopy(record.get("model_payload") or {}),
                    model_state_dict=state_dict_torch,
                    training_config=_coerce_training_config(record.get("training_config")),
                    metadata={"source": "load_model_from_adata"},
                )
            )
            if not isinstance(model, torch.nn.Module):
                raise TypeError(
                    f"deserialize_model must return torch.nn.Module, got {type(model).__name__}"
                )
        else:
            if spec.model_builder is None:
                raise ValueError(
                    f"Custom algorithm '{algorithm_id}' does not define model_builder or deserialize_model."
                )
            training_data = resolve_training_data_bundle(
                adata,
                resolved_config,
                stage=stage,
                device="cpu",
                output_dir=str(record.get("model_payload", {}).get("output_dir") or ""),
                training_data_builder=spec.training_data_builder,
                metadata={
                    "source": "load_model_from_adata",
                    "algorithm_id": algorithm_id,
                },
            )
            model = spec.model_builder(
                ModelBuildContext(
                    algorithm_id=algorithm_id,
                    resolved_config=deepcopy(resolved_config),
                    latent_dim=int(dim),
                    training_data=training_data,
                    device=torch.device("cpu"),
                    stage=stage,
                    metadata={"source": "load_model_from_adata"},
                )
            )
            if not isinstance(model, torch.nn.Module):
                raise TypeError(
                    f"model_builder must return torch.nn.Module, got {type(model).__name__}"
                )
            model.load_state_dict(state_dict_torch)

        model.eval()
        print("Model loaded successfully.")
        return model

    raise ValueError(f"Unsupported all_model provider/schema: provider={provider!r}, schema_version={schema_version}")


#%%
import torch
from anndata import AnnData
from CytoBridge.tl.models import DynamicalModel
import os

def save_model_to_adata(adata: AnnData, model: DynamicalModel) -> None:
    """
    Save the trained DynamicalModel to the AnnData object

    Parameters:
        adata: AnnData object used to store the model
        model: Trained DynamicalModel instance
    """
    if 'dynamic_model' not in adata.uns:
        adata.uns['dynamic_model'] = {}

    # Save model configuration
    adata.uns['dynamic_model']['model_config'] = model.config

    # Save model weights (convert to numpy array for h5ad compatibility)
    state_dict = model.state_dict()
    state_dict_cpu_numpy = {k: v.cpu().numpy() for k, v in state_dict.items()}
    adata.uns['dynamic_model']['model_state_dict'] = state_dict_cpu_numpy

    print("Model has been successfully saved to adata.uns['dynamic_model']")

def save_model_to_file(model: DynamicalModel, save_path: str) -> None:
    """
    Save the model directly as a pth file (recommended for separate backup)

    Parameters:
        model: Trained DynamicalModel instance
        save_path: Save path (e.g., 'models/trained_model.pth')
    """
    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save model weights and configuration
    torch.save({
        'model_config': model.config,
        'state_dict': model.state_dict()
    }, save_path)

    print(f"Model has been successfully saved to file: {save_path}")

def load_model_from_file(load_path: str, latent_dim: int) -> DynamicalModel:
    """
    Load model from pth file

    Parameters:
        load_path: Model file path
        latent_dim: Latent space dimension (must match the one used during training)
    Returns:
        DynamicalModel instance with loaded weights
    """
    checkpoint = torch.load(load_path)
    model = DynamicalModel(latent_dim, checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"Model has been loaded from file: {load_path}")
    return model
