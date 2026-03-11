from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from movie_edge_sim.simulation import SimulationConfig, SimulationResult


def _pick_indices(n_total: int, max_count: int) -> np.ndarray:
    if n_total <= max_count:
        return np.arange(n_total, dtype=int)
    return np.linspace(0, n_total - 1, num=max_count, dtype=int)


def plot_trajectories(
    result: SimulationResult,
    cfg: SimulationConfig,
    output_dir: str | Path,
    max_ues_to_plot: int = 120,
) -> Path:
    """Create a static plot showing UE and SBS trajectories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ue = result.ue_positions_over_time
    sbs = result.sbs_positions_over_time

    ue_idx = _pick_indices(ue.shape[1], max_ues_to_plot)

    fig, ax = plt.subplots(figsize=(10, 10))

    # UE trajectories in light blue to show movement density.
    for i in ue_idx:
        ax.plot(ue[:, i, 0], ue[:, i, 1], color="#0a0d10", alpha=0.18, linewidth=0.8)

    # SBS trajectories and their final points.
    for j in range(sbs.shape[1]):
        ax.plot(sbs[:, j, 0], sbs[:, j, 1], color="#e15759", alpha=0.9, linewidth=1.8)
        ax.scatter(sbs[-1, j, 0], sbs[-1, j, 1], color="#d62728", s=38, edgecolors="black", linewidths=0.4)

    ax.set_xlim(0, cfg.grid_size)
    ax.set_ylim(0, cfg.grid_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"UE/SBS trajectories (UEs shown: {len(ue_idx)}/{cfg.n_ues}, updates: {len(result.update_times)})"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    out_path = output_dir / "trajectories.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path
