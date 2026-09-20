import logging
import math
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from anndata import AnnData
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn import Module
from torchdiffeq import odeint

from CytoBridge.tl.interaction import cal_interaction
from CytoBridge.utils import check_submodules, load_model_from_adata
from ..tl.methods import ODEFunc

LOGGER = logging.getLogger(__name__)
TIME_KEY = "time_point_processed"
LATENT_KEY = "X_latent"


def _resolve_device(device: str | torch.device) -> torch.device:
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return target


def _require_standard_inputs(adata) -> None:
    if TIME_KEY not in adata.obs:
        raise ValueError(f"adata.obs['{TIME_KEY}'] is required.")
    if LATENT_KEY not in adata.obsm:
        raise ValueError(f"adata.obsm['{LATENT_KEY}'] is required.")


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(value)


def _pad_latent_sequence(states: list[torch.Tensor]) -> torch.Tensor:
    if not states:
        raise ValueError("State sequence is empty.")
    latent_dim = states[0].shape[1]
    max_cells = max(state.shape[0] for state in states)
    padded = torch.full(
        (len(states), max_cells, latent_dim),
        float("nan"),
        dtype=states[0].dtype,
        device=states[0].device,
    )
    for i, state in enumerate(states):
        if state.shape[0] > 0:
            padded[i, : state.shape[0], :] = state
    return padded


def _pad_lnw_sequence(states: list[torch.Tensor]) -> torch.Tensor:
    if not states:
        raise ValueError("Log-weight sequence is empty.")
    max_cells = max(state.shape[0] for state in states)
    padded = torch.full(
        (len(states), max_cells, 1),
        float("nan"),
        dtype=states[0].dtype,
        device=states[0].device,
    )
    for i, state in enumerate(states):
        if state.shape[0] > 0:
            padded[i, : state.shape[0], :] = state
    return padded

def compute_velocity(adata, model, device='cuda'):
    '''
    Calculate velocity based on the trained model 
    '''
    _require_standard_inputs(adata)
    device = _resolve_device(device)
    model.to(device)
    # Calculate velocity and growth on the dataset for downstream analysis
    all_times = torch.as_tensor(adata.obs[TIME_KEY].to_numpy(), dtype=torch.float32, device=device).unsqueeze(1)
    all_data = torch.as_tensor(adata.obsm[LATENT_KEY], dtype=torch.float32, device=device)
    net_input = torch.cat([all_data, all_times], dim = 1)
    with torch.no_grad():
        velocity = model.velocity_net(net_input)
    # Store in obsm instead of layers to avoid dimension mismatch
    # layers requires shape (n_obs, n_vars) but velocity is (n_obs, latent_dim)
    adata.obsm['velocity_latent'] = velocity.detach().cpu().numpy()

    return adata

def compute_interaction_force(adata, device='cuda'):
    '''
    Calculate interaction force based on the trained model
    '''
    _require_standard_inputs(adata)
    device = _resolve_device(device)
    model = load_model_from_adata(adata)
    model.to(device)
    
    # Check if interaction component exists in the model
    if 'interaction' not in model.components:
        raise ValueError("Model does not contain interaction component. Please ensure the model was trained with interaction.")
    
    # Prepare input data: latent features + time
    all_times = torch.as_tensor(adata.obs[TIME_KEY].to_numpy(), dtype=torch.float32, device=device).unsqueeze(1)
    all_data = torch.as_tensor(adata.obsm[LATENT_KEY], dtype=torch.float32, device=device)
    
    # Initialize log weights (required for interaction calculation).
    #
    # Important semantic note:
    # - `growth_rate` stores g(t, x) = d/dt log w, i.e. a rate field
    # - `cal_interaction(...)` expects lnw = log w, i.e. the current log-mass state
    # These are not interchangeable. If an explicit log-mass state is unavailable
    # in `adata`, fall back to unit mass and warn instead of misusing growth rate.
    LOGGER.info("interaction_force uses mass coupling: %s", model.use_growth_in_ode_inter)
    lnw = torch.zeros((all_data.shape[0], 1), dtype=torch.float32, device=device)
    if model.use_growth_in_ode_inter:
        if 'log_mass' in adata.obsm:
            lnw = torch.as_tensor(adata.obsm['log_mass'], dtype=torch.float32, device=device)
        elif 'lnw' in adata.obsm:
            lnw = torch.as_tensor(adata.obsm['lnw'], dtype=torch.float32, device=device)
        else:
            LOGGER.warning(
                "compute_interaction_force requested mass-aware interaction, but no "
                "explicit log-mass state (`adata.obsm['log_mass']` or `adata.obsm['lnw']`) "
                "was found. `growth_rate` is a log-mass rate, not log-mass itself, so "
                "interaction_force falls back to unit mass."
            )

    interaction_force = cal_interaction(
        z=all_data,
        lnw=lnw,
        interaction_potential=model.interaction_net,
        cutoff=model.interaction_net.cutoff,
        use_mass=model.use_growth_in_ode_inter
    ).float()
    
    adata.layers['interaction_force'] = interaction_force.detach().cpu().numpy()
    
    return adata

# If CytoSDE / euler_sdeint / load_model_from_adata are defined in other files,
# import them here according to the actual file structure
# from .model import CytoSDE, load_model_from_adata
# from .integrator import euler_sdeint

# ------------------------------------------------------------------------------
# SDE Definition Dependencies: PyTorch Neural Network Module


class CytoSDE(Module):
    """
    SDE definition for CytoBridge models (Ito-type with diagonal noise).
    Computes drift (f) and diffusion (g) terms for latent state (z) and log-weight (lnw).
    
    Attributes
    ----------
    noise_type : str
        Type of noise ("diagonal" = independent noise per dimension).
    sde_type : str
        Type of SDE ("ito" = Ito calculus).
    model : torch.nn.Module
        Pre-trained CytoBridge model (must include velocity/score/growth networks as components).
    sigma : float
        Global diffusion coefficient that scales the magnitude of noise.
    """
    noise_type = "diagonal"  # Diagonal noise (independent across dimensions)
    sde_type = "ito"         # Ito-type SDE

    def __init__(self, model, sigma=1.0):
        super().__init__()
        self.model = model    # Trained CytoBridge model (contains velocity/score/growth networks)
        self.sigma = sigma    # Diffusion coefficient for noise scaling

    def f(self, t, y):
        """
        Compute drift term f(t, y) for the SDE.
        
        Parameters
        ----------
        t : torch.Tensor
            Current time (shape: [1]).
        y : tuple[torch.Tensor, torch.Tensor]
            Current state (z, lnw), where:
            - z: Latent state (shape: [batch_size, latent_dim])
            - lnw: Log-weight for particles (shape: [batch_size, 1])
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Drift terms for z and lnw (same shapes as input y).
        """
        z, lnw = y
        # Expand time to match the batch size of z
        t_expand = t.expand(z.shape[0], 1).to(dtype=z.dtype)
        # Input to velocity network: latent features + time
        vel_in = torch.cat([z, t_expand], 1)
        
        # Compute drift for z (base velocity + score correction if available)
        drift_z = self.model.velocity_net(vel_in)
        if "score" in self.model.components:
            _, gradients = self.model.compute_score(t, z)  # Assume compute_score returns (score, gradients)
            drift_z += gradients
        
        # Compute drift for lnw (growth term if available, else zero)
        if "growth" in self.model.components:
            drift_lnw = self.model.growth_net(vel_in)
        else:
            drift_lnw = torch.zeros_like(lnw)
        if "interaction" in self.model.components:
            interaction_force = cal_interaction(z=z,lnw=lnw,interaction_potential=self.model.interaction_net,cutoff=self.model.interaction_net.cutoff,use_mass=self.model.use_growth_in_ode_inter)
            drift_z += interaction_force
        return (drift_z, drift_lnw)

    def g(self, t, y):
        """
        Compute diffusion term g(t, y) for the SDE.
        Noise is applied to z (latent state) but not to lnw (log-weight).
        
        Parameters
        ----------
        t : torch.Tensor
            Current time (shape: [1]).
        y : tuple[torch.Tensor, torch.Tensor]
            Current state (z, lnw) (shapes: [batch_size, latent_dim], [batch_size, 1]).
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Diffusion terms for z and lnw (same shapes as input y).
        """
        z, lnw = y
        return (torch.ones_like(z) * self.sigma,  # Diffusion for z
                torch.ones_like(lnw) * self.sigma * 0.0)  # No diffusion for lnw


def euler_sdeint_split(sde, initial_state, dt, ts, noise_std = 0.01):
    device = initial_state[0].device
    t0 = ts[0].item()
    tf = ts[-1].item()
    current_state = initial_state
    current_time = t0
    output_states = []
    ts_list = ts.tolist()
    next_output_idx = 0
    w_prev = torch.exp(current_state[1])
    while current_time <= tf + 1e-8:
        t_tensor = torch.tensor([current_time], dtype=torch.float32, device=device)
        f_z, f_lnw = sde.f(t_tensor, current_state)
        noise_z = torch.randn_like(current_state[0]) * math.sqrt(dt)
        g_z ,g_lnw = sde.g(t_tensor, current_state)
        noise_lnw = torch.randn_like(current_state[1]) * math.sqrt(dt)
        new_z = current_state[0] + f_z * dt + g_z * noise_z
        new_lnw = current_state[1] + f_lnw * dt+ g_lnw * noise_lnw

        current_time += dt
        if current_time >= ts_list[next_output_idx] - 1e-8:
            w_next = torch.exp(new_lnw)
            r = w_next / w_prev
            next_z = []
            next_lnw = []
            for j in range(current_state[0].shape[0]):
                if r[j] >= 1:
                    r_floor = torch.floor(r[j])
                    m_j = int(r_floor) + (1 if torch.rand(1, device=device) < (r[j] - r_floor) else 0)
                    for _ in range(m_j):
                        noise = torch.normal(0, noise_std, size=new_z[j].shape, device=device)
                        perturbed_x = new_z[j] + noise
                        next_z.append(perturbed_x.unsqueeze(0))
                        next_lnw.append(new_lnw[j].unsqueeze(0))
                else:
                    if torch.rand(1, device=device) < r[j]:
                        next_z.append(new_z[j].unsqueeze(0))
                        next_lnw.append(new_lnw[j].unsqueeze(0))
            if next_z:
                new_z = torch.cat(next_z, dim=0)
                new_lnw = torch.log(torch.ones(new_z.shape[0], 1) / initial_state[0].shape[0]).to(device)
            else:
                new_z = torch.empty(0, current_state[0].shape[1], device=device)
                new_lnw = torch.empty(0, 1, device=device)
            current_state = (new_z, new_lnw)
            output_states.append(current_state)
            next_output_idx += 1
            w_prev = torch.exp(new_lnw)
            if next_output_idx >= len(ts_list):
                break
        else:
            current_state = (new_z, new_lnw)
    while len(output_states) < len(ts_list):
        output_states.append(current_state)
    traj_z = [state[0] for state in output_states]
    traj_lnw = [state[1] for state in output_states]
    return traj_z, traj_lnw

def euler_sdeint(sde, y0, dt, ts):
    """
    Numerical integration of SDE using Euler-Maruyama method (for Ito-type SDEs).
    
    Parameters
    ----------
    sde : CytoSDE
        SDE object with f (drift) and g (diffusion) methods.
    y0 : tuple[torch.Tensor, torch.Tensor]
        Initial state (z0, lnw0):
        - z0: Initial latent state (shape: [batch_size, latent_dim])
        - lnw0: Initial log-weight (shape: [batch_size, 1])
    dt : float
        Integration time step (fixed).
    ts : torch.Tensor
        Target time points to record (shape: [n_time_points], sorted in ascending order).
    
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - z_traj: Latent state trajectories (shape: [n_time_points, batch_size, latent_dim])
        - lnw_traj: Log-weight trajectories (shape: [n_time_points, batch_size, 1])
    """
    # Initialize state and current time
    y, t = y0, ts[0].item()
    out = []  # Store states at target time points
    ts_lst = ts.tolist()  # Convert target times to list for iteration
    idx = 0  # Index for target time points

    # Iterate until exceeding the maximum target time
    while t <= ts_lst[-1] + 1e-8:
        # Record state if current time matches target time (within tolerance)
        if t >= ts_lst[idx] - 1e-8:
            out.append(y)
            idx += 1
            if idx >= len(ts_lst):
                break  # Exit if all target times are recorded
        
        # Convert current time to tensor (matches model input format)
        t_tensor = torch.tensor([t], dtype=torch.float32, device=y[0].device)
        
        # Compute drift and diffusion terms
        f_z, f_lnw = sde.f(t_tensor, y)
        g_z, g_lnw = sde.g(t_tensor, y)
        
        # Generate Gaussian noise (scaled by sqrt(dt) for Ito calculus)
        noise_z = torch.randn_like(y[0]) * math.sqrt(dt)
        noise_lnw = torch.randn_like(y[1]) * math.sqrt(dt)
        
        # Euler-Maruyama update: y_{t+dt} = y_t + f*dt + g*noise
        y = (y[0] + f_z * dt + g_z * noise_z,
             y[1] + f_lnw * dt + g_lnw * noise_lnw)
        
        # Step forward in time
        t += dt

    # Fill missing target times with the last recorded state (due to numerical tolerance)
    while len(out) < len(ts_lst):
        out.append(out[-1])

    # Reshape output to [n_time_points, batch_size, ...]
    z_traj = torch.stack([o[0] for o in out])
    lnw_traj = torch.stack([o[1] for o in out])
    return z_traj, lnw_traj

def generate_sde_trajectories(
    adata,
    exp_dir: str,
    device: str | torch.device,
    sigma: float,
    n_time_steps: int,
    sample_traj_num: int,
    init_time: int,
    split_true: bool =False

) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encapsulates the complete workflow from the parent function (model loading → SDE integration → sampling/filtering → saving) into a standalone function.

    Returns
    -------
    sde_traj : np.ndarray
        Shape (n_steps, n_sample_cells, latent_dim)
    sde_point : np.ndarray
        Shape (n_time_points, n_cells_used, latent_dim)
    w_point : np.ndarray
        Shape (n_time_points, n_cells_used, 1) - Normalized weights
    """
    _require_standard_inputs(adata)
    device = _resolve_device(device)
    split_true = _to_bool(split_true)
    os.makedirs(exp_dir, exist_ok=True)
    # ---------- Load Pre-trained Model ----------
    LOGGER.info("Loading model from adata for SDE trajectories.")
    model = load_model_from_adata(adata).to(device).float()  # Assume load_model_from_adata is predefined
    model.eval()
    LOGGER.info("Model components: %s", model.components)

    # ---------- 2. Time Axis ----------
    all_times = sorted(adata.obs[TIME_KEY].unique())
    if len(all_times) < 2:
        raise ValueError(f"Need at least 2 unique '{TIME_KEY}' values for trajectory integration.")
    if n_time_steps < 2:
        raise ValueError("n_time_steps must be >= 2.")
    ts_points = torch.tensor(all_times, dtype=torch.float32, device=device)
    ts_continuous = torch.linspace(
        all_times[0], all_times[-1], n_time_steps,
        dtype=torch.float32, device=device
    )

    # ---------- 3. Initial State ----------
    init_mask = adata.obs[TIME_KEY] == init_time
    if not init_mask.any():
        raise ValueError(f"No data found for init_time={init_time}")
    x0 = torch.as_tensor(adata[init_mask].obsm[LATENT_KEY], dtype=torch.float32, device=device)
    x0 = x0.requires_grad_(True)

    length = x0.shape[0]
    lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device) / length)
    initial_state = (x0, lnw0)

    # ---------- 4. SDE Integration ----------
    sde = CytoSDE(model, sigma).to(device)
    dt = (all_times[-1] - all_times[0]) / (n_time_steps - 1)
    if split_true:
        traj_states, traj_lnw_states = euler_sdeint_split(sde, initial_state, dt, ts_continuous)
        point_states, point_lnw_states = euler_sdeint_split(sde, initial_state, dt, ts_points)
        sde_traj = _pad_latent_sequence(traj_states)
        traj_lnw = _pad_lnw_sequence(traj_lnw_states)
        sde_point = _pad_latent_sequence(point_states)
        point_lnw = _pad_lnw_sequence(point_lnw_states)
    else:
        sde_traj, traj_lnw = euler_sdeint(sde, initial_state, dt, ts_continuous)
        sde_point, point_lnw = euler_sdeint(sde, initial_state, dt, ts_points)
    # ---------- 5. Sampling / Consistency Filtering ----------
    sample_traj_num = max(1, min(sample_traj_num, x0.shape[0]))
    traj_cell_indices = torch.randperm(x0.shape[0], device=device)[:sample_traj_num]


    sde_traj = sde_traj[:, traj_cell_indices]
    traj_lnw = traj_lnw[:, traj_cell_indices]


    # ---------- 6. Convert to NumPy / Weight Normalization ----------
    sde_traj = sde_traj.detach().cpu().numpy()
    sde_point = sde_point.detach().cpu().numpy()
    w_point_t = torch.exp(point_lnw)
    w_point_t = torch.nan_to_num(w_point_t, nan=0.0, posinf=0.0, neginf=0.0)
    w_norm = w_point_t.sum(dim=1, keepdim=True).clamp_min(1e-12)
    w_point = (w_point_t / w_norm).detach().cpu().numpy()

    # ---------- 7. Saving ----------
    np.save(os.path.join(exp_dir, "sde_trajec.npy"), sde_traj)
    np.save(os.path.join(exp_dir, "sde_point.npy"), sde_point)
    np.save(os.path.join(exp_dir, "sde_weight.npy"), w_point)

    return sde_traj, sde_point, w_point

def euler_odeint_split(ode_func, initial_state, ts, noise_std=0.01):
    device = initial_state[0].device
    t0, tf = ts[0].item(), ts[-1].item()
    current_t = t0
    current_z, current_lnw, current_m = initial_state
    w_prev = torch.exp(current_lnw)  # Initial weights
    
    # Save initial shape for later reference
    initial_size = current_z.shape[0]
    feature_size = current_z.shape[1]  # Feature dimension

    ts_list = ts.tolist()
    # -------------------------- Key Modification 1: Save historical data for all time steps (for backtracking and copying) --------------------------
    # Historical data format: [time_step0_data, time_step1_data, ...], each element is (z, lnw)
    history_z = [current_z.clone()]  # Particle feature history at initial step (t0)
    history_lnw = [current_lnw.clone()]  # Log-weight history at initial step (t0)
    idx_next = 1  # Start iteration from step 1 (t0 is already stored in history)

    while current_t <= tf + 1e-8 and idx_next < len(ts_list):
        t_tensor = torch.tensor([current_t], device=device)
        f_z, f_lnw, f_m = ode_func(t_tensor, (current_z, current_lnw, current_m))

        # 1. Euler integration to update current step particle state
        dt_step = ts_list[idx_next] - current_t
        new_z   = current_z   + f_z   * dt_step
        new_lnw = current_lnw + f_lnw * dt_step
        new_m   = current_m   + f_m   * dt_step
        current_t = ts_list[idx_next]

        w_next = torch.exp(new_lnw)
        ratio = w_next / w_prev
        current_size = new_z.shape[0]  # Current number of particles

        # -------------------------- 2. Splitting Handling: Add logic for "backtracking and copying historical data" --------------------------
        mask_split = (ratio >= 1).squeeze(-1)
        # Data after splitting: includes "original splitting particles + newly split particles"
        split_z = new_z[mask_split].clone()  # Keep original splitting particles (not discarded)
        split_lnw = new_lnw[mask_split].clone()
        split_m = new_m[mask_split].clone()
        # Record indices of original splitting particles (for subsequent backtracking and copying of history)
        original_split_indices = torch.where(mask_split)[0]
        
        if mask_split.any():
            r_split = ratio[mask_split].squeeze(-1)
            z_split_original = new_z[mask_split]  # Original splitting particles (to split based on)
            lnw_split_original = new_lnw[mask_split]
            m_split_original = new_m[mask_split]

            # Calculate the number of splits for each original splitting particle (consistent with original logic)
            floor_r = torch.floor(r_split)
            frac_r = r_split - floor_r
            rand_f = torch.rand_like(frac_r)
            mj = floor_r.int() + (rand_f < frac_r).int()  # Number of new particles split from each original particle
            valid = mj > 0

            if valid.any():
                # Filter indices of original particles that need splitting
                valid_mask = valid
                valid_mj = mj[valid_mask]  # Valid number of splits
                valid_original_indices = original_split_indices[valid_mask]  # Global indices of valid original splitting particles
                valid_z = z_split_original[valid_mask]
                valid_lnw = lnw_split_original[valid_mask]
                valid_m = m_split_original[valid_mask]

                # Generate newly split particles (current step data)
                new_split_z_list = []
                new_split_lnw_list = []
                for i in range(len(valid_mj)):
                    mj_i = valid_mj[i].item()  # Number of new particles split from the i-th original particle
                    original_idx_i = valid_original_indices[i].item()  # Global index of the i-th original particle

                    # a. Generate current-step data for newly split particles (with noise)
                    rep_z = valid_z[i].unsqueeze(0).repeat(mj_i, 1)  # Repeat mj_i times
                    rep_lnw = valid_lnw[i].unsqueeze(0).repeat(mj_i, 1)
                    noise = torch.normal(0, noise_std, size=rep_z.shape, device=device)
                    new_z_i = rep_z + noise
                    new_lnw_i = rep_lnw

                    # b. Key step: Backtrack and copy historical data of the original particle (fill historical time steps for new particles)
                    # Iterate through all historical time steps (from t0 to the step before current step) and copy the original particle's history
                    for hist_idx in range(len(history_z)):
                        # Historical data of the original particle at the historical time step
                        hist_z_original = history_z[hist_idx][original_idx_i].unsqueeze(0)
                        hist_lnw_original = history_lnw[hist_idx][original_idx_i].unsqueeze(0)
                        # Copy to the new particle's history (history of each new particle matches the original particle)
                        history_z[hist_idx] = torch.cat([history_z[hist_idx], hist_z_original.repeat(mj_i, 1)], dim=0)
                        history_lnw[hist_idx] = torch.cat([history_lnw[hist_idx], hist_lnw_original.repeat(mj_i, 1)], dim=0)

                    # c. Collect current-step data of newly split particles
                    new_split_z_list.append(new_z_i)
                    new_split_lnw_list.append(new_lnw_i)

                # Merge current-step data of all newly split particles
                if new_split_z_list:
                    new_split_z = torch.cat(new_split_z_list, dim=0)
                    new_split_lnw = torch.cat(new_split_lnw_list, dim=0)
                    new_split_m = valid_m.unsqueeze(0).repeat_interleave(valid_mj, dim=1).squeeze(0)
                    # Add newly split particles to the current-step splitting results (original splitting particles + newly split particles)
                    split_z = torch.cat([split_z, new_split_z], dim=0)
                    split_lnw = torch.cat([split_lnw, new_split_lnw], dim=0)
                    split_m = torch.cat([split_m, new_split_m], dim=0)

        # -------------------------- 3. Extinction Handling: Modify to "set to NaN instead of discarding" --------------------------
        mask_extinct = ~mask_split
        # Keep indices of all extinct particles; set dead particles to NaN (not discarded to ensure consistent time-step indices)
        extinct_z = new_z[mask_extinct].clone()
        extinct_lnw = new_lnw[mask_extinct].clone()
        extinct_m = new_m[mask_extinct].clone()
        
        if mask_extinct.any():
            r_ext = ratio[mask_extinct].squeeze(-1)
            # Generate survival mask: True = survive, False = dead (set to NaN)
            keep_mask = torch.rand_like(r_ext) < r_ext
            # Set dead particles to NaN (keep indices; no further updates in subsequent steps)
            extinct_z[~keep_mask] = torch.nan
            extinct_lnw[~keep_mask] = torch.nan
            extinct_m[~keep_mask] = torch.nan

        # -------------------------- 4. Merge: Keep all particle indices (splitting + extinction) --------------------------
        current_z = torch.cat([split_z, extinct_z], dim=0)
        current_lnw = torch.cat([split_lnw, extinct_lnw], dim=0)
        current_m = torch.cat([split_m, extinct_m], dim=0)
        w_prev = torch.exp(current_lnw)  # Update weights

        # -------------------------- 5. Save current-step data to history (for subsequent splitting backtracking) --------------------------
        history_z.append(current_z.clone())
        history_lnw.append(current_lnw.clone())
        idx_next += 1

    # -------------------------- 6. Unify shapes of all time steps (historical data is complete; directly concatenate) --------------------------
    # Historical data already contains complete time-series of all particles (original + split); directly stack into tensors
    out_z = torch.stack(history_z, dim=0)  # shape: (time_steps, num_particles, feature_size)
    out_lnw = torch.stack(history_lnw, dim=0)  # shape: (time_steps, num_particles, 1)
    
    return out_z, out_lnw


# ---------- Subfunction: Only responsible for integration ----------
def generate_ode_trajectories(
        model,
        adata,
        n_trajectories: int = 50,
        n_bins: int = 100,
        device: str = "cuda",
        samples_key: str = 'time_point_processed',
        exp_dir: str = None,          
        split_true: bool = False
) -> tuple[np.ndarray, np.ndarray]:


    _require_standard_inputs(adata)
    device = _resolve_device(device)
    split_true = _to_bool(split_true)
    if samples_key != TIME_KEY:
        LOGGER.warning(
            "samples_key=%s ignored; package contract requires %s.",
            samples_key,
            TIME_KEY,
        )
    samples_key = TIME_KEY
    model.to(device).eval()
    X_raw = adata.obsm[LATENT_KEY]

    unique_times = np.sort(adata.obs[samples_key].unique())
    if len(unique_times) < 2:
        raise ValueError(f"Need at least 2 unique '{samples_key}' values for ODE integration.")

    init_time = unique_times[0]
    init_mask = adata.obs[samples_key] == init_time
    init_idx = np.where(init_mask)[0]
    replace = len(init_idx) < n_trajectories

    chosen_init_idx = np.random.choice(init_idx, size=n_trajectories, replace=replace)
    init_x = torch.tensor(X_raw[chosen_init_idx], dtype=torch.float32, device=device)

    use_mass, score_use, interaction_use = check_submodules(model)

    ode_func = ODEFunc(model,use_mass=use_mass,score_use=score_use,interaction_use=interaction_use).to(device)

    t_min, t_max = float(unique_times[0]), float(unique_times[-1])
    t_bins  = torch.linspace(t_min, t_max, n_bins, device=device)
    t_point = torch.tensor(unique_times, dtype=torch.float32, device=device)
    init_lnw = torch.log(torch.ones(n_trajectories, 1, device=device) / n_trajectories)
    init_m   = torch.zeros_like(init_lnw)
    initial_state = (init_x, init_lnw, init_m)
    if split_true:
        traj_x, traj_lnw  = euler_odeint_split(ode_func, initial_state, t_bins)
        point_x, point_lnw = euler_odeint_split(ode_func, initial_state, t_point)
    else:
        traj_x, traj_lnw, _ = odeint(ode_func, initial_state, t_bins,  method='euler')
        point_x, point_lnw, _ = odeint(ode_func, initial_state, t_point, method='euler', options=dict(step_size=0.2))
    traj_array  = traj_x.detach().cpu().numpy()          # (T_bins, M, D)
    traj_lnw    = traj_lnw.detach().cpu().numpy().squeeze(-1)  # (T, M)
    point_array = point_x.detach().cpu().numpy()         # (T, M, D)
    point_lnw   = point_lnw.detach().cpu().numpy().squeeze(-1)  # (T, M)
    # Saving
    if exp_dir is not None:
        os.makedirs(exp_dir, exist_ok=True)
        np.save(os.path.join(exp_dir, "ode_traj.npy"), traj_array)
        np.save(os.path.join(exp_dir, "ode_traj_lnw.npy"), traj_lnw)
        np.save(os.path.join(exp_dir, "ode_point.npy"),  point_array)
        np.save(os.path.join(exp_dir, "ode_point_lnw.npy"), point_lnw)
        LOGGER.info("[generate_ode_trajectories] trajectories saved to %s", exp_dir)

    return point_array, traj_array


def simulate_trajectory(adata,model, x0, sigma, time, dt, device):
    x0 = x0.detach().to(device)
    lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device, dtype=torch.float32) / x0.shape[0])
    initial_state = (x0, lnw0)
    was_training = bool(getattr(model, "training", False))
    model.eval()

    class CytoSDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"
        def __init__(self, model, sigma=1.0):
            super().__init__()
            self.model = model
            self.sigma = sigma
        def f(self, t, y):
            z, lnw = y
            z_base = z.detach()
            lnw_base = lnw.detach()
            t_expand = t.expand(z_base.shape[0], 1).to(dtype=z_base.dtype)
            vel_in = torch.cat([z_base, t_expand], 1)
            with torch.no_grad():
                drift_z = self.model.velocity_net(vel_in)
            if "score" in self.model.components:
                with torch.enable_grad():
                    _, gradients = self.model.compute_score(
                        t,
                        z_base.detach().requires_grad_(True),
                        create_graph=False,
                    )
                drift_z = drift_z + gradients.detach()
            if "growth" in self.model.components:
                with torch.no_grad():
                    drift_lnw = self.model.growth_net(vel_in)
            else:
                drift_lnw = torch.zeros_like(lnw_base)
            if "interaction" in self.model.components:
                with torch.enable_grad():
                    interaction_force = cal_interaction(
                        z=z_base.detach().requires_grad_(True),
                        lnw=lnw_base,
                        interaction_potential=self.model.interaction_net,
                        cutoff=self.model.interaction_net.cutoff,
                        use_mass=self.model.use_growth_in_ode_inter,
                        create_graph=False,
                    )
                drift_z = drift_z + interaction_force.detach()

            return (drift_z.detach(), drift_lnw.detach())
        def g(self, t, y):
            z, lnw = y
            return (torch.ones_like(z) * self.sigma,
                    torch.ones_like(lnw) * 0.0)

    sde = CytoSDE(model, sigma).to(device)

    def euler_sdeint(sde, y0, dt, ts_lst):
        y, t = y0, ts_lst[0]
        out_x = []
        out_lnw = []
        idx = 0
        def record(state):
            out_x.append(state[0].detach().cpu().numpy().copy())
            out_lnw.append(state[1].detach().cpu().numpy().copy())
        while t <= ts_lst[-1] + 1e-8:
            if t >= ts_lst[idx] - 1e-8:
                record(y)
                idx += 1
                if idx >= len(ts_lst):
                    break
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
            f_z, f_lnw = sde.f(t_tensor, y)
            g_z, g_lnw = sde.g(t_tensor, y)
            noise_z = torch.randn_like(y[0]) * math.sqrt(dt)
            noise_lnw = torch.randn_like(y[1]) * math.sqrt(dt)
            y = (
                (y[0] + f_z * dt + g_z * noise_z).detach(),
                (y[1] + f_lnw * dt + g_lnw * noise_lnw).detach(),
            )
            t += dt
        while len(out_x) < len(ts_lst):
            out_x.append(out_x[-1].copy())
            out_lnw.append(out_lnw[-1].copy())
        return np.stack(out_x), np.stack(out_lnw)

    try:
        sde_point, point_lnw = euler_sdeint(sde, initial_state, dt, time)
    finally:
        if was_training:
            model.train()

    w_point = np.exp(point_lnw)

    return sde_point, w_point

class MLPClassifier(nn.Module):
    """MLP model for cell type classification"""
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_mlp_classifier(
    adata: AnnData,
    classifyed_type: str = 'cluster',
    hidden_size: int = 128,
    train_mlp_classifier_epoches: int = 400,
    lr: float = 0.001,
    device: str = 'cuda'
) -> Tuple[MLPClassifier, LabelEncoder]:
    """
    Train MLP classifier using real data
    
    Parameters:
        adata: AnnData object containing cell data
        classifyed_type: Key of classification target in obs (default 'cluster')
        hidden_size: MLP hidden layer dimension
        train_mlp_classifier_epoches: Number of training epochs
        lr: Learning rate
        device: Training device
    
    Returns:
        Trained MLP model and label encoder
    """
    device = _resolve_device(device)
    # Extract features and labels
    X = adata.X  # Assume X is the feature matrix (n_obs × n_vars)
    if hasattr(X, "toarray"):
        X = X.toarray()
    y = adata.obs[classifyed_type].values
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # Initialize model
    input_size = X.shape[1]
    num_classes = len(label_encoder.classes_)
    model = MLPClassifier(input_size, hidden_size, num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    for epoch in range(train_mlp_classifier_epoches):
        model.train()
        X_train.requires_grad_(True)
        
        # Forward propagation
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Gradient regularization (to prevent overfitting)
        grad_outputs = torch.ones_like(outputs)
        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=X_train,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grad_norm = grads.norm(dim=1).mean()
        reg_lambda = 1e-2
        
        # L2 regularization
        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            l2_reg += torch.norm(param, 2) ** 2
        weight_decay = 1e-4
        
        # Total loss
        total_loss = loss + reg_lambda * grad_norm + weight_decay * l2_reg
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Print results every 100 epochs
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                _, predicted = torch.max(test_outputs, 1)
                accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
            LOGGER.info(
                "Epoch [%s/%s], Loss: %.4f, Test Acc: %.4f",
                epoch + 1,
                train_mlp_classifier_epoches,
                total_loss.item(),
                accuracy,
            )
    
    return model, label_encoder


# ------------------------------------------------------------------------------
# Structured downstream wrappers (compatibility layer)
# ------------------------------------------------------------------------------
def compute_velocity_bundle(adata, model=None, device="cuda", output_dir=None):
    from .downstream.velocity import compute_velocity_bundle as _impl

    return _impl(adata=adata, model=model, device=device, output_dir=output_dir)


def build_velocity_graph_bundle(
    adata,
    output_dir=None,
    n_pcs=50,
    n_neighbors=30,
    n_jobs=None,
    vkey="velocity",
    preferred_basis=None,
    strict_preferred=False,
):
    from .downstream.velocity import build_velocity_graph_bundle as _impl

    return _impl(
        adata=adata,
        output_dir=output_dir,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
        vkey=vkey,
        preferred_basis=preferred_basis,
        strict_preferred=strict_preferred,
    )


def summarize_velocity_drivers_bundle(
    adata,
    output_dir=None,
    analysis_space="gene",
    top_n=20,
):
    from .downstream.velocity import summarize_velocity_drivers_bundle as _impl

    return _impl(
        adata=adata,
        output_dir=output_dir,
        analysis_space=analysis_space,
        top_n=top_n,
    )


def summarize_velocity_jacobian_drivers_bundle(
    adata,
    model=None,
    output_dir=None,
    analysis_space="gene",
    top_n=20,
    n_cells=512,
    random_state=0,
    device="cuda",
):
    from .downstream.velocity import summarize_velocity_jacobian_drivers_bundle as _impl

    return _impl(
        adata=adata,
        model=model,
        output_dir=output_dir,
        analysis_space=analysis_space,
        top_n=top_n,
        n_cells=n_cells,
        random_state=random_state,
        device=device,
    )


def summarize_growth_drivers_bundle(
    adata,
    output_dir=None,
    analysis_space="auto",
    top_n=20,
    max_cells=20000,
    batch_size=2048,
    random_state=0,
    device="cuda",
):
    from .downstream.growth import summarize_growth_drivers_bundle as _impl

    return _impl(
        adata=adata,
        output_dir=output_dir,
        analysis_space=analysis_space,
        top_n=top_n,
        max_cells=max_cells,
        batch_size=batch_size,
        random_state=random_state,
        device=device,
    )


def summarize_growth_jacobian_drivers_bundle(
    adata,
    output_dir=None,
    top_n=20,
    max_cells=20000,
    batch_size=2048,
    random_state=0,
    device="cuda",
):
    from .downstream.growth import summarize_growth_jacobian_drivers_bundle as _impl

    return _impl(
        adata=adata,
        output_dir=output_dir,
        top_n=top_n,
        max_cells=max_cells,
        batch_size=batch_size,
        random_state=random_state,
        device=device,
    )


def analyze_grn_bundle(
    adata,
    output_dir=None,
    max_genes=10,
    max_time_points=5,
    genes=None,
    device="cuda",
):
    from .downstream.grn import analyze_grn_bundle as _impl

    return _impl(
        adata=adata,
        output_dir=output_dir,
        max_genes=max_genes,
        max_time_points=max_time_points,
        genes=genes,
        device=device,
    )


def generate_ode_trajectory_bundle(
    model,
    adata,
    output_dir,
    n_trajectories=50,
    n_bins=100,
    device="cuda",
    split_true=False,
):
    from .downstream.trajectory import generate_ode_trajectory_bundle as _impl

    return _impl(
        model=model,
        adata=adata,
        output_dir=output_dir,
        n_trajectories=n_trajectories,
        n_bins=n_bins,
        device=device,
        split_true=split_true,
    )


def generate_sde_trajectory_bundle(
    adata,
    output_dir,
    n_time_steps=100,
    sample_traj_num=1000,
    init_time=0,
    device="cuda",
    split_true=False,
):
    from .downstream.trajectory import generate_sde_trajectory_bundle as _impl

    return _impl(
        adata=adata,
        output_dir=output_dir,
        n_time_steps=n_time_steps,
        sample_traj_num=sample_traj_num,
        init_time=init_time,
        device=device,
        split_true=split_true,
    )
