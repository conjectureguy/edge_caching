from __future__ import annotations

import argparse
from pathlib import Path

from movie_edge_sim.data import download_movielens_100k, load_ratings
from movie_edge_sim.plotting import plot_trajectories
from movie_edge_sim.simulation import SimulationConfig, run_simulation, write_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MovieLens 100K and simulate UE mobility with SBS clustering updates."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Directory for MovieLens dataset.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Directory for generated trajectory CSV files."
    )
    parser.add_argument("--grid-size", type=float, default=300.0)
    parser.add_argument("--n-ues", type=int, default=300)
    parser.add_argument("--n-sbs", type=int, default=8)
    parser.add_argument("--total-time", type=float, default=300.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--t-update", type=float, default=10.0)
    parser.add_argument("--max-speed", type=float, default=1.5, help="Max UE speed in grid-units/second.")
    parser.add_argument("--sbs-max-speed", type=float, default=0.5, help="Max SBS speed in grid-units/second.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-iters", type=int, default=30)
    parser.add_argument(
        "--skip-dataset-download",
        action="store_true",
        help="Skip MovieLens download/load (useful when running mobility plotting only).",
    )
    parser.add_argument("--plot", action="store_true", help="Generate trajectory plot image.")
    parser.add_argument(
        "--plot-max-ues",
        type=int,
        default=120,
        help="Maximum number of UE trajectories to draw in the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.skip_dataset_download:
        dataset_dir = None
        ratings = []
    else:
        dataset_dir = download_movielens_100k(args.data_root)
        ratings = load_ratings(dataset_dir)

    cfg = SimulationConfig(
        grid_size=args.grid_size,
        n_ues=args.n_ues,
        n_sbs=args.n_sbs,
        total_time=args.total_time,
        dt=args.dt,
        t_update=args.t_update,
        max_speed=args.max_speed,
        sbs_max_speed=args.sbs_max_speed,
        random_seed=args.seed,
        kmeans_iters=args.kmeans_iters,
    )

    result = run_simulation(cfg)
    ue_csv, sbs_csv = write_trajectories(result, args.output_dir, cfg.dt)

    if args.plot:
        plot_path = plot_trajectories(result, cfg, args.output_dir, max_ues_to_plot=args.plot_max_ues)
    else:
        plot_path = None

    if dataset_dir is not None:
        print(f"MovieLens dir: {dataset_dir}")
        print(f"Ratings loaded: {len(ratings)}")
    else:
        print("MovieLens download/load: skipped")

    print(f"UE trajectory CSV: {ue_csv}")
    print(f"SBS position CSV: {sbs_csv}")
    if plot_path is not None:
        print(f"Trajectory plot: {plot_path}")
    print(f"SBS updates: {len(result.update_times)}")


if __name__ == "__main__":
    main()
