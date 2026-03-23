from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from movie_edge_sim.simulation import (
    SimulationConfig,
    compute_latency_series,
    freeze_sbs_positions,
    run_simulation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a paired fixed-vs-moving SBS latency study under clustered UE mobility."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/clustered_latency_study"))
    parser.add_argument("--grid-size", type=float, default=300.0)
    parser.add_argument("--n-ues", type=int, default=300)
    parser.add_argument("--n-sbs", type=int, default=8)
    parser.add_argument("--total-time", type=float, default=180.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--t-update", type=float, default=5.0)
    parser.add_argument("--max-speed", type=float, default=4.0)
    parser.add_argument("--sbs-max-speed", type=float, default=1.5)
    parser.add_argument("--prediction-horizon-factor", type=float, default=1.0)
    parser.add_argument("--n-hotspots", type=int, default=6)
    parser.add_argument("--hotspot-speed", type=float, default=1.2)
    parser.add_argument("--hotspot-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-iters", type=int, default=30)
    parser.add_argument("--latency-base-ms", type=float, default=5.0)
    parser.add_argument("--latency-distance-factor-ms", type=float, default=0.01)
    parser.add_argument("--latency-distance-exponent", type=float, default=2.0)
    parser.add_argument("--latency-handover-penalty-ms", type=float, default=0.2)
    parser.add_argument("--latency-hysteresis", type=float, default=0.0)
    return parser.parse_args()


def evaluate_pair(cfg: SimulationConfig, args: argparse.Namespace) -> dict[str, np.ndarray | float]:
    result = run_simulation(cfg)
    fixed_result = freeze_sbs_positions(result)

    latency_kwargs = dict(
        base_latency_ms=args.latency_base_ms,
        distance_factor_ms=args.latency_distance_factor_ms,
        distance_exponent=args.latency_distance_exponent,
        handover_penalty_ms=args.latency_handover_penalty_ms,
        association_hysteresis=args.latency_hysteresis,
    )
    moving = compute_latency_series(result, cfg, **latency_kwargs)
    fixed = compute_latency_series(fixed_result, cfg, **latency_kwargs)

    return {
        "time_s": moving.time_s,
        "fixed_latency": fixed.mean_latency_ms,
        "moving_latency": moving.mean_latency_ms,
        "fixed_mean": float(np.mean(fixed.mean_latency_ms)),
        "moving_mean": float(np.mean(moving.mean_latency_ms)),
    }


def save_sweep_csv(output_path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_latency_over_time(base: dict[str, np.ndarray | float], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(base["time_s"], base["fixed_latency"], color="#4c78a8", linewidth=2.0, label="UE moving, SBS fixed")
    ax.plot(base["time_s"], base["moving_latency"], color="#e45756", linewidth=2.0, label="UE moving, SBS moving")
    ax.set_title("Mean Latency Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Latency (ms)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "latency_over_time.png", dpi=180)
    plt.close(fig)


def plot_cumulative_latency(base: dict[str, np.ndarray | float], output_dir: Path) -> None:
    fixed = np.asarray(base["fixed_latency"])
    moving = np.asarray(base["moving_latency"])
    steps = np.arange(1, fixed.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(base["time_s"], np.cumsum(fixed) / steps, color="#4c78a8", linewidth=2.0, label="UE moving, SBS fixed")
    ax.plot(base["time_s"], np.cumsum(moving) / steps, color="#e45756", linewidth=2.0, label="UE moving, SBS moving")
    ax.set_title("Cumulative Average Latency")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Mean Latency (ms)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_average_latency.png", dpi=180)
    plt.close(fig)


def plot_latency_cdf(base: dict[str, np.ndarray | float], output_dir: Path) -> None:
    fixed = np.sort(np.asarray(base["fixed_latency"]))
    moving = np.sort(np.asarray(base["moving_latency"]))
    fixed_y = np.arange(1, fixed.shape[0] + 1) / fixed.shape[0]
    moving_y = np.arange(1, moving.shape[0] + 1) / moving.shape[0]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(fixed, fixed_y, color="#4c78a8", linewidth=2.0, label="UE moving, SBS fixed")
    ax.plot(moving, moving_y, color="#e45756", linewidth=2.0, label="UE moving, SBS moving")
    ax.set_title("Latency CDF")
    ax.set_xlabel("Mean Step Latency (ms)")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "latency_cdf.png", dpi=180)
    plt.close(fig)


def plot_sweep(rows: list[dict[str, float]], x_key: str, title: str, xlabel: str, output_path: Path) -> None:
    x = [row[x_key] for row in rows]
    fixed = [row["fixed_mean_latency_ms"] for row in rows]
    moving = [row["moving_mean_latency_ms"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(x, fixed, marker="o", color="#4c78a8", linewidth=2.0, label="UE moving, SBS fixed")
    ax.plot(x, moving, marker="o", color="#e45756", linewidth=2.0, label="UE moving, SBS moving")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean Latency (ms)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_improvement(rows: list[dict[str, float]], x_key: str, title: str, xlabel: str, output_path: Path) -> None:
    x = [row[x_key] for row in rows]
    improvement = [100.0 * (row["fixed_mean_latency_ms"] - row["moving_mean_latency_ms"]) / row["fixed_mean_latency_ms"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(x, improvement, color="#e45756", edgecolor="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Latency Reduction (%)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = SimulationConfig(
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

    print("Running base paired simulation...")
    base = evaluate_pair(base_cfg, args)
    plot_latency_over_time(base, output_dir)
    plot_cumulative_latency(base, output_dir)
    plot_latency_cdf(base, output_dir)

    t_update_rows: list[dict[str, float]] = []
    print("Sweeping t_update...")
    for value in [3.0, 5.0, 8.0, 12.0]:
        result = evaluate_pair(replace(base_cfg, t_update=value), args)
        t_update_rows.append(
            {
                "t_update_s": value,
                "fixed_mean_latency_ms": result["fixed_mean"],
                "moving_mean_latency_ms": result["moving_mean"],
            }
        )
    save_sweep_csv(output_dir / "latency_vs_t_update.csv", t_update_rows)
    plot_sweep(t_update_rows, "t_update_s", "Mean Latency vs SBS Update Interval", "SBS Update Interval (s)", output_dir / "latency_vs_t_update.png")

    sbs_speed_rows: list[dict[str, float]] = []
    print("Sweeping sbs_max_speed...")
    for value in [0.5, 1.0, 1.5, 2.0]:
        result = evaluate_pair(replace(base_cfg, sbs_max_speed=value), args)
        sbs_speed_rows.append(
            {
                "sbs_max_speed": value,
                "fixed_mean_latency_ms": result["fixed_mean"],
                "moving_mean_latency_ms": result["moving_mean"],
            }
        )
    save_sweep_csv(output_dir / "latency_vs_sbs_speed.csv", sbs_speed_rows)
    plot_sweep(sbs_speed_rows, "sbs_max_speed", "Mean Latency vs SBS Speed", "Max SBS Speed (grid units/s)", output_dir / "latency_vs_sbs_speed.png")

    hotspot_rows: list[dict[str, float]] = []
    print("Sweeping hotspot_weight...")
    for value in [0.25, 0.5, 0.75, 0.9]:
        result = evaluate_pair(replace(base_cfg, hotspot_weight=value), args)
        hotspot_rows.append(
            {
                "hotspot_weight": value,
                "fixed_mean_latency_ms": result["fixed_mean"],
                "moving_mean_latency_ms": result["moving_mean"],
            }
        )
    save_sweep_csv(output_dir / "latency_vs_hotspot_weight.csv", hotspot_rows)
    plot_sweep(hotspot_rows, "hotspot_weight", "Mean Latency vs UE Cluster Strength", "Hotspot Attraction Weight", output_dir / "latency_vs_hotspot_weight.png")

    n_sbs_rows: list[dict[str, float]] = []
    print("Sweeping n_sbs...")
    for value in [4, 6, 8, 10, 12]:
        result = evaluate_pair(replace(base_cfg, n_sbs=value), args)
        n_sbs_rows.append(
            {
                "n_sbs": float(value),
                "fixed_mean_latency_ms": result["fixed_mean"],
                "moving_mean_latency_ms": result["moving_mean"],
            }
        )
    save_sweep_csv(output_dir / "latency_vs_n_sbs.csv", n_sbs_rows)
    plot_sweep(n_sbs_rows, "n_sbs", "Mean Latency vs Number of SBSs", "Number of SBSs", output_dir / "latency_vs_n_sbs.png")
    plot_improvement(
        n_sbs_rows,
        "n_sbs",
        "Latency Reduction from SBS Mobility",
        "Number of SBSs",
        output_dir / "latency_reduction_vs_n_sbs.png",
    )

    print(f"Saved clustered latency study to: {output_dir}")


if __name__ == "__main__":
    main()
