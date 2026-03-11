from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SimulationConfig:
    grid_size: float = 300.0
    n_ues: int = 300
    n_sbs: int = 8
    total_time: float = 300.0
    dt: float = 1.0
    t_update: float = 10.0
    max_speed: float = 1.5
    sbs_max_speed: float = 0.5
    random_seed: int = 42
    kmeans_iters: int = 30


@dataclass
class SimulationResult:
    ue_positions_over_time: np.ndarray  # (n_steps+1, n_ues, 2)
    sbs_positions_over_time: np.ndarray  # (n_updates, n_sbs, 2)
    update_times: np.ndarray  # (n_updates,)


def _random_initial_positions(rng: np.random.Generator, n_points: int, grid_size: float) -> np.ndarray:
    return rng.uniform(0.0, grid_size, size=(n_points, 2))


def _slow_random_walk_step(
    rng: np.random.Generator, positions: np.ndarray, dt: float, max_speed: float, grid_size: float
) -> np.ndarray:
    # Random direction and slow speed at each step.
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(positions.shape[0],))
    speeds = rng.uniform(0.0, max_speed, size=(positions.shape[0],))
    deltas = np.column_stack((np.cos(angles), np.sin(angles))) * (speeds * dt)[:, None]
    new_pos = positions + deltas

    # Reflect on boundaries.
    for dim in (0, 1):
        lower_mask = new_pos[:, dim] < 0.0
        new_pos[lower_mask, dim] = -new_pos[lower_mask, dim]

        upper_mask = new_pos[:, dim] > grid_size
        new_pos[upper_mask, dim] = 2.0 * grid_size - new_pos[upper_mask, dim]

        new_pos[:, dim] = np.clip(new_pos[:, dim], 0.0, grid_size)

    return new_pos


def kmeans(points: np.ndarray, k: int, rng: np.random.Generator, iters: int = 30) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be > 0")
    if points.shape[0] < k:
        raise ValueError("k cannot exceed number of points")

    init_idx = rng.choice(points.shape[0], size=k, replace=False)
    centroids = points[init_idx].copy()

    for _ in range(iters):
        d2 = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)

        new_centroids = np.empty_like(centroids)
        for i in range(k):
            cluster_points = points[labels == i]
            if len(cluster_points) == 0:
                new_centroids[i] = points[rng.integers(0, points.shape[0])]
            else:
                new_centroids[i] = cluster_points.mean(axis=0)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < 1e-6:
            break

    return centroids


def _greedy_match_targets(current: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Reorder target centroids to keep SBS identities stable across updates."""
    k = current.shape[0]
    dist = ((current[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2)

    matched_targets = np.empty_like(targets)
    used_current: set[int] = set()
    used_target: set[int] = set()

    for _ in range(k):
        best_i = -1
        best_j = -1
        best_d = np.inf
        for i in range(k):
            if i in used_current:
                continue
            for j in range(k):
                if j in used_target:
                    continue
                d = dist[i, j]
                if d < best_d:
                    best_d = d
                    best_i = i
                    best_j = j
        used_current.add(best_i)
        used_target.add(best_j)
        matched_targets[best_i] = targets[best_j]

    return matched_targets


def _move_points_toward_targets(points: np.ndarray, targets: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0:
        return points.copy()

    diff = targets - points
    dist = np.linalg.norm(diff, axis=1)
    scale = np.ones_like(dist)
    mask = dist > max_step
    scale[mask] = max_step / dist[mask]
    return points + diff * scale[:, None]


def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    if cfg.t_update < cfg.dt:
        raise ValueError("t_update must be >= dt")

    rng = np.random.default_rng(cfg.random_seed)

    n_steps = int(cfg.total_time / cfg.dt)
    update_stride = int(round(cfg.t_update / cfg.dt))
    if not np.isclose(update_stride * cfg.dt, cfg.t_update):
        raise ValueError("t_update must be an integer multiple of dt")

    ue_positions = _random_initial_positions(rng, cfg.n_ues, cfg.grid_size)
    ue_history = np.empty((n_steps + 1, cfg.n_ues, 2), dtype=np.float64)
    ue_history[0] = ue_positions

    sbs_history = []
    update_times = []

    # Update SBS locations at t=0 and then every t_update.
    sbs_positions = kmeans(ue_positions, cfg.n_sbs, rng, cfg.kmeans_iters)
    sbs_history.append(sbs_positions.copy())
    update_times.append(0.0)

    for step in range(1, n_steps + 1):
        ue_positions = _slow_random_walk_step(rng, ue_positions, cfg.dt, cfg.max_speed, cfg.grid_size)
        ue_history[step] = ue_positions

        if step % update_stride == 0:
            target_centroids = kmeans(ue_positions, cfg.n_sbs, rng, cfg.kmeans_iters)
            matched_targets = _greedy_match_targets(sbs_positions, target_centroids)
            max_sbs_step = cfg.sbs_max_speed * cfg.t_update
            sbs_positions = _move_points_toward_targets(sbs_positions, matched_targets, max_sbs_step)
            sbs_history.append(sbs_positions.copy())
            update_times.append(step * cfg.dt)

    return SimulationResult(
        ue_positions_over_time=ue_history,
        sbs_positions_over_time=np.asarray(sbs_history),
        update_times=np.asarray(update_times),
    )


def write_trajectories(result: SimulationResult, output_dir: str | Path, dt: float) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ue_csv = output_dir / "ue_trajectories.csv"
    sbs_csv = output_dir / "sbs_positions.csv"

    with ue_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "ue_id", "x", "y"])
        for step, frame in enumerate(result.ue_positions_over_time):
            t = step * dt
            for ue_id, (x, y) in enumerate(frame):
                writer.writerow([f"{t:.3f}", ue_id, f"{x:.6f}", f"{y:.6f}"])

    with sbs_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "sbs_id", "x", "y"])
        for idx, frame in enumerate(result.sbs_positions_over_time):
            t = result.update_times[idx]
            for sbs_id, (x, y) in enumerate(frame):
                writer.writerow([f"{t:.3f}", sbs_id, f"{x:.6f}", f"{y:.6f}"])

    return ue_csv, sbs_csv
