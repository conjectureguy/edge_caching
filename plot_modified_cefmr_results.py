from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot training results for modified CEFMR runs. "
            "Each run dir should contain temporal_training.csv and rl_training.csv."
        )
    )
    p.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        default=[Path("outputs/modified_cefmr")],
        help="One or more run directories to plot/compare.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/plots_modified_cefmr"),
        help="Directory for generated plots.",
    )
    p.add_argument("--ma-window", type=int, default=5, help="Moving-average window for RL curves.")
    return p.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size < w:
        return x.copy()
    kernel = np.ones((w,), dtype=np.float64) / float(w)
    y = np.convolve(x, kernel, mode="valid")
    pad = np.full((w - 1,), np.nan, dtype=np.float64)
    return np.concatenate([pad, y], axis=0)


def load_run(run_dir: Path) -> dict[str, np.ndarray]:
    temporal_path = run_dir / "temporal_training.csv"
    rl_path = run_dir / "rl_training.csv"
    if not temporal_path.exists():
        raise FileNotFoundError(f"Missing: {temporal_path}")
    if not rl_path.exists():
        raise FileNotFoundError(f"Missing: {rl_path}")

    temporal_rows = _read_csv_rows(temporal_path)
    rl_rows = _read_csv_rows(rl_path)

    t_round = np.asarray([int(r["round"]) for r in temporal_rows], dtype=np.int64)
    t_train = np.asarray([float(r["train_loss"]) for r in temporal_rows], dtype=np.float64)
    t_val = np.asarray([float(r["val_loss"]) for r in temporal_rows], dtype=np.float64)

    e = np.asarray([int(r["episode"]) for r in rl_rows], dtype=np.int64)
    reward = np.asarray([float(r["reward"]) for r in rl_rows], dtype=np.float64)
    local = np.asarray([float(r["local_hit_rate"]) for r in rl_rows], dtype=np.float64)
    neighbor = np.asarray([float(r["neighbor_fetch_rate"]) for r in rl_rows], dtype=np.float64)
    cloud = np.asarray([float(r["cloud_fetch_rate"]) for r in rl_rows], dtype=np.float64)
    paper_hit = local + neighbor

    return {
        "name": run_dir.name,
        "run_dir": np.asarray([str(run_dir)]),
        "t_round": t_round,
        "t_train": t_train,
        "t_val": t_val,
        "e": e,
        "reward": reward,
        "local": local,
        "neighbor": neighbor,
        "cloud": cloud,
        "paper_hit": paper_hit,
    }


def save_summary(run: dict[str, np.ndarray], out_path: Path) -> None:
    reward = run["reward"]
    local = run["local"]
    neighbor = run["neighbor"]
    cloud = run["cloud"]
    paper_hit = run["paper_hit"]
    t_val = run["t_val"]

    with out_path.open("w") as f:
        f.write(f"run: {run['name']}\n")
        f.write(f"episodes: {reward.size}\n")
        f.write(f"reward_mean: {reward.mean():.6f}\n")
        f.write(f"reward_min: {reward.min():.6f}\n")
        f.write(f"reward_max: {reward.max():.6f}\n")
        f.write(f"local_hit_mean: {local.mean():.6f}\n")
        f.write(f"local_hit_min: {local.min():.6f}\n")
        f.write(f"local_hit_max: {local.max():.6f}\n")
        f.write(f"neighbor_fetch_mean: {neighbor.mean():.6f}\n")
        f.write(f"cloud_fetch_mean: {cloud.mean():.6f}\n")
        f.write(f"paper_style_hit_mean(local+neighbor): {paper_hit.mean():.6f}\n")
        if t_val.size > 0:
            f.write(f"temporal_val_start: {t_val[0]:.6f}\n")
            f.write(f"temporal_val_end: {t_val[-1]:.6f}\n")
        else:
            f.write("temporal_val_start: N/A\n")
            f.write("temporal_val_end: N/A\n")


def plot_single_run(run: dict[str, np.ndarray], out_dir: Path, ma_window: int) -> None:
    name = run["name"]
    t_round = run["t_round"]
    t_train = run["t_train"]
    t_val = run["t_val"]
    e = run["e"]
    reward = run["reward"]
    local = run["local"]
    neighbor = run["neighbor"]
    cloud = run["cloud"]
    paper_hit = run["paper_hit"]

    if t_round.size > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_round, t_train, label="train_loss", color="#4e79a7")
        ax.plot(t_round, t_val, label="val_loss", color="#e15759")
        ax.set_title(f"Temporal Training Loss - {name}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Cross-entropy")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_temporal_loss.png", dpi=170)
        plt.close(fig)

    reward_ma = _moving_avg(reward, ma_window)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(e, reward, label="reward", color="#59a14f", alpha=0.35)
    ax.plot(e, reward_ma, label=f"reward_ma{ma_window}", color="#2f7d32", linewidth=2.0)
    ax.set_title(f"RL Reward - {name}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_reward.png", dpi=170)
    plt.close(fig)

    local_ma = _moving_avg(local, ma_window)
    neighbor_ma = _moving_avg(neighbor, ma_window)
    cloud_ma = _moving_avg(cloud, ma_window)
    paper_ma = _moving_avg(paper_hit, ma_window)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(e, local, color="#f28e2b", alpha=0.25, label="local_hit_rate")
    ax.plot(e, local_ma, color="#e67e22", linewidth=2.0, label=f"local_ma{ma_window}")
    ax.plot(e, neighbor_ma, color="#4e79a7", linewidth=1.8, label=f"neighbor_ma{ma_window}")
    ax.plot(e, cloud_ma, color="#e15759", linewidth=1.8, label=f"cloud_ma{ma_window}")
    ax.plot(e, paper_ma, color="#76b7b2", linewidth=2.0, label=f"(local+neighbor)_ma{ma_window}")
    ax.set_title(f"RL Request-Service Rates - {name}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_rates.png", dpi=170)
    plt.close(fig)


def plot_comparison(runs: list[dict[str, np.ndarray]], out_dir: Path, ma_window: int) -> None:
    if len(runs) < 2:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for run in runs:
        e = run["e"]
        local = run["local"]
        local_ma = _moving_avg(local, ma_window)
        ax.plot(e, local_ma, linewidth=2.0, label=f"{run['name']} local_ma{ma_window}")
    ax.set_title("Local Hit Rate Comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Local hit rate")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_local_hit.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for run in runs:
        e = run["e"]
        reward = run["reward"]
        reward_ma = _moving_avg(reward, ma_window)
        ax.plot(e, reward_ma, linewidth=2.0, label=f"{run['name']} reward_ma{ma_window}")
    ax.set_title("Reward Comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_reward.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for run in runs:
        e = run["e"]
        paper_hit = run["paper_hit"]
        paper_ma = _moving_avg(paper_hit, ma_window)
        ax.plot(e, paper_ma, linewidth=2.0, label=f"{run['name']} (local+neighbor)_ma{ma_window}")
    ax.set_title("Paper-Style Hit Rate Comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_paper_style_hit.png", dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = [load_run(d) for d in args.run_dirs]
    for run in runs:
        plot_single_run(run, args.out_dir, args.ma_window)
        save_summary(run, args.out_dir / f"{run['name']}_summary.txt")

    plot_comparison(runs, args.out_dir, args.ma_window)
    print(f"Saved plots in: {args.out_dir}")


if __name__ == "__main__":
    main()
