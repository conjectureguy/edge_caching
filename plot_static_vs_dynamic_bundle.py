from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from movie_edge_sim.simulation import SimulationConfig, expand_sbs_positions, freeze_sbs_positions, run_simulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate static-vs-dynamic mobility plots under clustered UE demand.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/static_vs_dynamic_bundle"))
    p.add_argument("--grid-size", type=float, default=300.0)
    p.add_argument("--n-ues", type=int, default=300)
    p.add_argument("--n-sbs", type=int, default=8)
    p.add_argument("--total-time", type=float, default=300.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--t-update", type=float, default=5.0)
    p.add_argument("--max-speed", type=float, default=4.0)
    p.add_argument("--sbs-max-speed", type=float, default=1.5)
    p.add_argument("--prediction-horizon-factor", type=float, default=1.0)
    p.add_argument("--n-hotspots", type=int, default=6)
    p.add_argument("--hotspot-speed", type=float, default=1.2)
    p.add_argument("--hotspot-weight", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kmeans-iters", type=int, default=30)
    p.add_argument("--coverage-radius", type=float, default=35.0)
    p.add_argument("--handover-penalty", type=float, default=0.2)
    return p.parse_args()


def _step_metrics(result, cfg: SimulationConfig, coverage_radius: float, handover_penalty: float) -> dict[str, np.ndarray]:
    ue = result.ue_positions_over_time
    sbs = expand_sbs_positions(result, cfg)
    n_steps = ue.shape[0]
    prev_assoc = None

    mean_distance = np.empty((n_steps,), dtype=np.float64)
    p95_distance = np.empty((n_steps,), dtype=np.float64)
    coverage_fraction = np.empty((n_steps,), dtype=np.float64)
    effective_cost = np.empty((n_steps,), dtype=np.float64)
    load_cv = np.empty((n_steps,), dtype=np.float64)
    hotspot_tracking_error = np.empty((n_steps,), dtype=np.float64)
    cluster_capture_rate = np.empty((n_steps,), dtype=np.float64)
    handover_rate = np.zeros((n_steps,), dtype=np.float64)

    hotspot_history = result.hotspot_positions_over_time
    hotspot_assignments = result.hotspot_assignments

    for step in range(n_steps):
        d = np.sqrt(((ue[step, :, None, :] - sbs[step, None, :, :]) ** 2).sum(axis=2))
        assoc = np.argmin(d, axis=1)
        min_dist = d[np.arange(d.shape[0]), assoc]

        if prev_assoc is not None:
            handover = assoc != prev_assoc
            handover_rate[step] = float(np.mean(handover.astype(np.float64)))
        else:
            handover = np.zeros((ue.shape[1],), dtype=bool)

        mean_distance[step] = float(np.mean(min_dist))
        p95_distance[step] = float(np.percentile(min_dist, 95))
        coverage_fraction[step] = float(np.mean((min_dist <= coverage_radius).astype(np.float64)))
        effective_cost[step] = float(np.mean(np.square(min_dist) + handover_penalty * handover.astype(np.float64)))

        counts = np.bincount(assoc, minlength=cfg.n_sbs).astype(np.float64)
        load_cv[step] = float(np.std(counts) / max(np.mean(counts), 1e-8))

        if hotspot_history is not None and hotspot_assignments is not None:
            hotspots = hotspot_history[step]
            hotspot_to_sbs = np.argmin(np.sqrt(((hotspots[:, None, :] - sbs[step, None, :, :]) ** 2).sum(axis=2)), axis=1)
            hotspot_error = np.sqrt(((hotspots - sbs[step, hotspot_to_sbs, :]) ** 2).sum(axis=1))
            hotspot_tracking_error[step] = float(np.mean(hotspot_error))

            capture = []
            for hotspot_id in range(hotspots.shape[0]):
                ue_mask = hotspot_assignments == hotspot_id
                if not np.any(ue_mask):
                    continue
                capture.append(float(np.mean((assoc[ue_mask] == hotspot_to_sbs[hotspot_id]).astype(np.float64))))
            cluster_capture_rate[step] = float(np.mean(capture)) if capture else 0.0
        else:
            hotspot_tracking_error[step] = 0.0
            cluster_capture_rate[step] = 0.0

        prev_assoc = assoc

    time_s = np.arange(n_steps, dtype=np.float64) * cfg.dt
    cumulative_distance = np.cumsum(mean_distance) / np.arange(1, n_steps + 1)

    return {
        "time_s": time_s,
        "mean_distance": mean_distance,
        "cumulative_distance": cumulative_distance,
        "p95_distance": p95_distance,
        "coverage_fraction": coverage_fraction,
        "effective_cost": effective_cost,
        "load_cv": load_cv,
        "hotspot_tracking_error": hotspot_tracking_error,
        "cluster_capture_rate": cluster_capture_rate,
        "handover_rate": handover_rate,
    }


def _save_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _line_plot(time_s, fixed, moving, title: str, ylabel: str, out_path: Path, lower_is_better: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(time_s, fixed, color="#4c78a8", linewidth=2.0, label="UE moving, SBS fixed")
    ax.plot(time_s, moving, color="#d62828", linewidth=2.0, label="UE moving, SBS moving")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend()
    ax.text(0.98, 0.95, "Lower is better" if lower_is_better else "Higher is better", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _cdf_plot(fixed_values, moving_values, title: str, xlabel: str, out_path: Path) -> None:
    fixed = np.sort(np.asarray(fixed_values))
    moving = np.sort(np.asarray(moving_values))
    fy = np.arange(1, fixed.shape[0] + 1) / fixed.shape[0]
    my = np.arange(1, moving.shape[0] + 1) / moving.shape[0]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(fixed, fy, color="#4c78a8", linewidth=2.0, label="UE moving, SBS fixed")
    ax.plot(moving, my, color="#d62828", linewidth=2.0, label="UE moving, SBS moving")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _gain_plot(time_s, fixed, moving, title: str, ylabel: str, out_path: Path) -> None:
    gain = 100.0 * (np.asarray(fixed) - np.asarray(moving)) / np.maximum(np.asarray(fixed), 1e-8)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(time_s, gain, color="#d62828", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimulationConfig(
        grid_size=args.grid_size,
        n_ues=args.n_ues,
        n_sbs=args.n_sbs,
        total_time=args.total_time,
        dt=args.dt,
        t_update=args.t_update,
        max_speed=args.max_speed,
        sbs_max_speed=args.sbs_max_speed,
        prediction_horizon_factor=args.prediction_horizon_factor,
        n_hotspots=args.n_hotspots,
        hotspot_speed=args.hotspot_speed,
        hotspot_weight=args.hotspot_weight,
        random_seed=args.seed,
        kmeans_iters=args.kmeans_iters,
    )

    print("Running clustered mobility simulation...")
    moving_result = run_simulation(cfg)
    fixed_result = freeze_sbs_positions(moving_result)

    fixed = _step_metrics(fixed_result, cfg, args.coverage_radius, args.handover_penalty)
    moving = _step_metrics(moving_result, cfg, args.coverage_radius, args.handover_penalty)

    rows = []
    for idx, t in enumerate(fixed["time_s"]):
        rows.append(
            {
                "time_s": float(t),
                "fixed_mean_distance": float(fixed["mean_distance"][idx]),
                "moving_mean_distance": float(moving["mean_distance"][idx]),
                "fixed_coverage_fraction": float(fixed["coverage_fraction"][idx]),
                "moving_coverage_fraction": float(moving["coverage_fraction"][idx]),
                "fixed_effective_cost": float(fixed["effective_cost"][idx]),
                "moving_effective_cost": float(moving["effective_cost"][idx]),
                "fixed_hotspot_tracking_error": float(fixed["hotspot_tracking_error"][idx]),
                "moving_hotspot_tracking_error": float(moving["hotspot_tracking_error"][idx]),
                "fixed_cluster_capture_rate": float(fixed["cluster_capture_rate"][idx]),
                "moving_cluster_capture_rate": float(moving["cluster_capture_rate"][idx]),
                "fixed_load_cv": float(fixed["load_cv"][idx]),
                "moving_load_cv": float(moving["load_cv"][idx]),
                "fixed_handover_rate": float(fixed["handover_rate"][idx]),
                "moving_handover_rate": float(moving["handover_rate"][idx]),
            }
        )
    _save_csv(args.output_dir / "static_vs_dynamic_metrics.csv", rows)

    _line_plot(fixed["time_s"], fixed["mean_distance"], moving["mean_distance"], "Mean UE-to-Serving-SBS Distance", "Distance", args.output_dir / "mean_distance_over_time.png")
    _line_plot(fixed["time_s"], fixed["cumulative_distance"], moving["cumulative_distance"], "Cumulative Mean Distance", "Cumulative Mean Distance", args.output_dir / "cumulative_mean_distance.png")
    _line_plot(fixed["time_s"], fixed["effective_cost"], moving["effective_cost"], "Handover-Aware Effective Distance Cost", "Mean Effective Cost", args.output_dir / "effective_distance_cost.png")
    _line_plot(fixed["time_s"], fixed["coverage_fraction"], moving["coverage_fraction"], "UE Fraction Within Coverage Radius", "Coverage Fraction", args.output_dir / "coverage_fraction_over_time.png", lower_is_better=False)
    _cdf_plot(fixed["mean_distance"], moving["mean_distance"], "CDF of Mean UE-to-SBS Distance", "Mean Distance", args.output_dir / "distance_cdf.png")
    _line_plot(fixed["time_s"], fixed["p95_distance"], moving["p95_distance"], "95th Percentile UE-to-SBS Distance", "P95 Distance", args.output_dir / "p95_distance_over_time.png")
    _line_plot(fixed["time_s"], fixed["hotspot_tracking_error"], moving["hotspot_tracking_error"], "Hotspot Tracking Error", "Distance to Nearest SBS", args.output_dir / "hotspot_tracking_error.png")
    _line_plot(fixed["time_s"], fixed["cluster_capture_rate"], moving["cluster_capture_rate"], "Cluster Capture Rate", "Capture Rate", args.output_dir / "cluster_capture_rate.png", lower_is_better=False)
    _line_plot(fixed["time_s"], fixed["load_cv"], moving["load_cv"], "SBS Load Imbalance (CV)", "Load CV", args.output_dir / "load_imbalance_cv.png")
    _line_plot(fixed["time_s"], fixed["handover_rate"], moving["handover_rate"], "Handover Rate", "Handover Rate", args.output_dir / "handover_rate_over_time.png")
    _gain_plot(fixed["time_s"], fixed["mean_distance"], moving["mean_distance"], "Distance Reduction from SBS Mobility", "Distance Reduction (%)", args.output_dir / "distance_reduction_gain.png")
    _gain_plot(fixed["time_s"], fixed["hotspot_tracking_error"], moving["hotspot_tracking_error"], "Hotspot-Tracking Improvement from SBS Mobility", "Tracking Improvement (%)", args.output_dir / "hotspot_tracking_gain.png")

    print(f"Saved static-vs-dynamic plot bundle to: {args.output_dir}")


if __name__ == "__main__":
    main()
