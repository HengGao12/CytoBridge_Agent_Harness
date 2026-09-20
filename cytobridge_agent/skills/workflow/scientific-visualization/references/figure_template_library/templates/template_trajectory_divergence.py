"""
Template: Trajectory Divergence Panel
=====================================
Show how generated trajectories separate into different fates over time on a
shared embedding (e.g., UMAP).  Produces a row of time-sliced sub-panels
(Early / Mid / Late) with a uniform grey background and colored foreground
scatter + KDE contours for each fate group.

Inspired by: CREST main figure Panel D (lineage tracing project).

Required inputs
---------------
- bg_coords : ndarray (N, 2) — background cell coordinates (all cells).
- traj_coords : ndarray (T, M, 2) — trajectory coordinates over T time steps
  for M cells.
- fate_labels : ndarray (M,) — fate group label per trajectory cell
  (e.g., "BM", "AL").
- time_indices : list[int] — which time steps to display (e.g., [0, 10, 19]
  for Early/Mid/Late).
- time_titles : list[str] — titles for each time slice (same length as
  time_indices).

Optional inputs
---------------
- fate_colors : dict[str, str] — color per fate group.  Defaults to red/blue.
- fate_threshold : float — fate probability threshold to select committed cells.
  Caller should pre-filter; this template plots what it receives.
- bg_color : str — background point color.  Default "#d0d0d0".
- bg_size : float — background point size.  Default 1.
- fg_size : float — foreground (trajectory) point size.  Default 3.
- contour_levels : int — number of KDE contour levels.  Default 2.
- figsize : tuple — overall figure size.  Default (12, 4).

Style rules (from style_playbook.md)
-------------------------------------
- Grey background, colored overlay.
- All sub-panels share exact UMAP limits and aspect.
- KDE contours thin (1-2 levels), same hue as scatter.
- No axes, no frame, no ticks.
- Small title per sub-panel.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_trajectory_divergence(
    bg_coords,
    traj_coords,
    fate_labels,
    time_indices,
    time_titles,
    fate_colors=None,
    bg_color="#d0d0d0",
    bg_size=1,
    fg_size=3,
    contour_levels=2,
    figsize=(12, 4),
    dpi=300,
    save_path=None,
):
    """
    Parameters
    ----------
    bg_coords : ndarray (N, 2)
    traj_coords : ndarray (T, M, 2)
    fate_labels : ndarray (M,)
    time_indices : list[int]
    time_titles : list[str]
    fate_colors : dict or None
    bg_color, bg_size, fg_size, contour_levels, figsize, dpi, save_path
    """
    if fate_colors is None:
        unique = sorted(set(fate_labels))
        default_cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
        fate_colors = {f: default_cols[i % len(default_cols)] for i, f in enumerate(unique)}

    n_panels = len(time_indices)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, dpi=dpi)
    if n_panels == 1:
        axes = [axes]

    x_all = np.concatenate([bg_coords[:, 0]] + [traj_coords[t, :, 0] for t in time_indices])
    y_all = np.concatenate([bg_coords[:, 1]] + [traj_coords[t, :, 1] for t in time_indices])
    pad = 0.03
    xmin, xmax = x_all.min(), x_all.max()
    ymin, ymax = y_all.min(), y_all.max()
    xr = xmax - xmin
    yr = ymax - ymin
    xlim = (xmin - pad * xr, xmax + pad * xr)
    ylim = (ymin - pad * yr, ymax + pad * yr)

    for ax, t_idx, title in zip(axes, time_indices, time_titles):
        ax.scatter(bg_coords[:, 0], bg_coords[:, 1], c=bg_color, s=bg_size,
                   alpha=0.4, edgecolors="none")

        for fate, color in fate_colors.items():
            mask = fate_labels == fate
            pts = traj_coords[t_idx, mask]
            if pts.shape[0] < 5:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], c=color, s=fg_size, alpha=0.6,
                       edgecolors="none", zorder=3)
            try:
                kde = gaussian_kde(pts.T, bw_method=0.3)
                xx, yy = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=contour_levels, colors=[color],
                           linewidths=0.8, alpha=0.7, zorder=4)
            except np.linalg.LinAlgError:
                pass

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.axis("off")

    fig.tight_layout(pad=1.0)

    if save_path:
        for ext in (".png", ".pdf"):
            fig.savefig(save_path + ext, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig, axes
