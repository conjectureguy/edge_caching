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
    prediction_horizon_factor: float = 1.0
    n_hotspots: int = 0
    hotspot_speed: float = 0.0
    hotspot_weight: float = 0.0
    random_seed: int = 42
    kmeans_iters: int = 30


@dataclass
class SimulationResult:
    ue_positions_over_time: np.ndarray  # (n_steps+1, n_ues, 2)
    sbs_positions_over_time: np.ndarray  # (n_updates, n_sbs, 2)
    update_times: np.ndarray  # (n_updates,)


@dataclass
class LatencyResult:
    time_s: np.ndarray  # (n_steps+1,)
    mean_latency_ms: np.ndarray  # (n_steps+1,)
    p95_latency_ms: np.ndarray  # (n_steps+1,)
    mean_distance: np.ndarray  # (n_steps+1,)


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


def _biased_random_walk_step(
    rng: np.random.Generator,
    positions: np.ndarray,
    targets: np.ndarray,
    dt: float,
    max_speed: float,
    grid_size: float,
    attraction_weight: float,
) -> np.ndarray:
    attraction_weight = float(np.clip(attraction_weight, 0.0, 1.0))
    if attraction_weight <= 0.0:
        return _slow_random_walk_step(rng, positions, dt, max_speed, grid_size)

    angles = rng.uniform(0.0, 2.0 * np.pi, size=(positions.shape[0],))
    random_dir = np.column_stack((np.cos(angles), np.sin(angles)))
    toward = targets - positions
    toward_norm = np.linalg.norm(toward, axis=1, keepdims=True)
    toward_dir = np.divide(toward, np.maximum(toward_norm, 1e-8), out=np.zeros_like(toward), where=toward_norm > 1e-8)

    direction = (1.0 - attraction_weight) * random_dir + attraction_weight * toward_dir
    direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = np.divide(direction, np.maximum(direction_norm, 1e-8), out=np.zeros_like(direction), where=direction_norm > 1e-8)

    speeds = rng.uniform(0.6 * max_speed, max_speed, size=(positions.shape[0],))
    deltas = direction * (speeds * dt)[:, None]
    new_pos = positions + deltas
    return _reflect_positions_in_grid(new_pos, grid_size)


def _reflect_positions_in_grid(positions: np.ndarray, grid_size: float) -> np.ndarray:
    new_pos = positions.copy()
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

    hotspot_positions = None
    hotspot_assignments = None
    if cfg.n_hotspots > 0 and cfg.hotspot_weight > 0.0:
        hotspot_positions = _random_initial_positions(rng, cfg.n_hotspots, cfg.grid_size)
        hotspot_assignments = rng.integers(0, cfg.n_hotspots, size=(cfg.n_ues,))

    sbs_history = []
    update_times = []

    # Update SBS locations at t=0 and then every t_update.
    sbs_positions = kmeans(ue_positions, cfg.n_sbs, rng, cfg.kmeans_iters)
    sbs_history.append(sbs_positions.copy())
    update_times.append(0.0)

    for step in range(1, n_steps + 1):
        if hotspot_positions is not None and hotspot_assignments is not None:
            hotspot_positions = _slow_random_walk_step(rng, hotspot_positions, cfg.dt, cfg.hotspot_speed, cfg.grid_size)
            ue_targets = hotspot_positions[hotspot_assignments]
            ue_positions = _biased_random_walk_step(
                rng,
                ue_positions,
                ue_targets,
                cfg.dt,
                cfg.max_speed,
                cfg.grid_size,
                cfg.hotspot_weight,
            )
        else:
            ue_positions = _slow_random_walk_step(rng, ue_positions, cfg.dt, cfg.max_speed, cfg.grid_size)
        ue_history[step] = ue_positions

        if step % update_stride == 0:
            target_points = ue_positions
            if cfg.prediction_horizon_factor > 0.0 and step > 0:
                recent_delta = ue_positions - ue_history[step - 1]
                horizon_steps = cfg.prediction_horizon_factor * update_stride
                predicted_ues = ue_positions + recent_delta * horizon_steps
                target_points = _reflect_positions_in_grid(predicted_ues, cfg.grid_size)
            target_centroids = kmeans(target_points, cfg.n_sbs, rng, cfg.kmeans_iters)
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


def expand_sbs_positions(result: SimulationResult, cfg: SimulationConfig) -> np.ndarray:
    """Expand piecewise-constant SBS positions to every simulation step."""
    n_steps = result.ue_positions_over_time.shape[0] - 1
    update_indices = np.rint(result.update_times / cfg.dt).astype(int)
    expanded = np.empty((n_steps + 1, cfg.n_sbs, 2), dtype=np.float64)
    current = result.sbs_positions_over_time[0]
    update_ptr = 0
    for step in range(n_steps + 1):
        while update_ptr + 1 < len(update_indices) and step >= update_indices[update_ptr + 1]:
            update_ptr += 1
            current = result.sbs_positions_over_time[update_ptr]
        expanded[step] = current
    return expanded


def compute_latency_series(
    result: SimulationResult,
    cfg: SimulationConfig,
    base_latency_ms: float = 5.0,
    distance_factor_ms: float = 0.05,
    distance_exponent: float = 1.0,
    handover_penalty_ms: float = 1.5,
    association_hysteresis: float = 0.0,
) -> LatencyResult:
    """Estimate latency from UE-to-serving-SBS distance and handover events.

    If ``association_hysteresis`` is positive, a UE keeps its previous serving
    SBS unless a new SBS is closer by at least that many grid units. This
    reduces ping-pong handovers and better reflects practical cell reselection.
    """
    ue = result.ue_positions_over_time
    sbs = expand_sbs_positions(result, cfg)
    n_steps = ue.shape[0]
    prev_assoc: np.ndarray | None = None

    mean_latency = np.empty((n_steps,), dtype=np.float64)
    p95_latency = np.empty((n_steps,), dtype=np.float64)
    mean_distance = np.empty((n_steps,), dtype=np.float64)

    for step in range(n_steps):
        d = np.sqrt(((ue[step, :, None, :] - sbs[step, None, :, :]) ** 2).sum(axis=2))
        nearest_assoc = np.argmin(d, axis=1)
        if prev_assoc is None or association_hysteresis <= 0.0:
            assoc = nearest_assoc
        else:
            assoc = prev_assoc.copy()
            prev_dist = d[np.arange(d.shape[0]), prev_assoc]
            nearest_dist = d[np.arange(d.shape[0]), nearest_assoc]
            switch = nearest_dist + association_hysteresis < prev_dist
            assoc[switch] = nearest_assoc[switch]
        min_dist = d[np.arange(d.shape[0]), assoc]
        latency = base_latency_ms + distance_factor_ms * np.power(min_dist, distance_exponent)
        if prev_assoc is not None:
            handover = assoc != prev_assoc
            latency = latency + handover_penalty_ms * handover.astype(np.float64)
        mean_latency[step] = float(np.mean(latency))
        p95_latency[step] = float(np.percentile(latency, 95))
        mean_distance[step] = float(np.mean(min_dist))
        prev_assoc = assoc

    time_s = np.arange(n_steps, dtype=np.float64) * cfg.dt
    return LatencyResult(
        time_s=time_s,
        mean_latency_ms=mean_latency,
        p95_latency_ms=p95_latency,
        mean_distance=mean_distance,
    )


def write_latency_csv(latency: LatencyResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "mean_latency_ms", "p95_latency_ms", "mean_distance"])
        for t, mean_lat, p95_lat, mean_dist in zip(
            latency.time_s,
            latency.mean_latency_ms,
            latency.p95_latency_ms,
            latency.mean_distance,
        ):
            writer.writerow([f"{t:.3f}", f"{mean_lat:.6f}", f"{p95_lat:.6f}", f"{mean_dist:.6f}"])
    return output_path


def freeze_sbs_positions(result: SimulationResult) -> SimulationResult:
    """Create a paired baseline where SBSs stay at their initial positions.

    UE trajectories and update times are kept identical to the source run so the
    only changed factor is SBS mobility.
    """
    fixed_sbs = np.repeat(result.sbs_positions_over_time[:1], repeats=result.sbs_positions_over_time.shape[0], axis=0)
    return SimulationResult(
        ue_positions_over_time=result.ue_positions_over_time.copy(),
        sbs_positions_over_time=fixed_sbs,
        update_times=result.update_times.copy(),
    )
