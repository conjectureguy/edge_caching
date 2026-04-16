#!/usr/bin/env python3
"""
Plot comparison of selected algorithms from train.log and related_work.log.
Filters to only: TemporalGraph, MAAFDRL, C-epsilon-greedy, BSG, Random
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import matplotlib

mpl_dir = Path("outputs/.mplconfig")
mpl_dir.mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate comparison plots for selected algorithms from log files."
    )
    p.add_argument(
        "--train-log",
        type=Path,
        default=Path("outputs/research_benchmark_runs/temporalgraph_20260415_041120/train.log"),
        help="Path to train.log file.",
    )
    p.add_argument(
        "--related-work-log",
        type=Path,
        default=Path("outputs/research_benchmark_runs/temporalgraph_20260415_041120/related_work.log"),
        help="Path to related_work.log file.",
    )
    p.add_argument(
        "--capacity-csv",
        type=Path,
        default=Path("outputs/research_benchmark_runs/temporalgraph_20260415_041120/novel_comparison_bundle/capacity_sweep.csv"),
        help="Path to capacity_sweep.csv for baseline algorithm data.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/filtered_comparison_plots"),
        help="Directory to save generated plots.",
    )
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["TemporalGraph", "MAAFDRL", "C-epsilon-greedy", "BSG-like", "Random"],
        help="Algorithms to include in plots.",
    )
    return p.parse_args()


def parse_related_work_log(log_path: Path) -> dict[str, Any]:
    """Parse related_work.log to extract episode data and summaries."""
    episode_pattern = re.compile(
        r"(?P<name>[\w\-]+) episode (?P<ep>\d+)/\d+ \| "
        r"reward=(?P<reward>[\d\.]+) local=(?P<local>[\d\.]+) "
        r"neighbor=(?P<neighbor>[\d\.]+) cloud=(?P<cloud>[\d\.]+) "
        r"paper_hit=(?P<paper_hit>[\d\.]+)"
    )
    summary_pattern = re.compile(
        r"(?P<name>[\w\-]+): reward_mean=(?P<reward>[\d\.]+) "
        r"local_hit_mean=(?P<local>[\d\.]+) paper_hit_mean=(?P<paper_hit>[\d\.]+) "
        r"cloud_fetch_mean=(?P<cloud>[\d\.]+)"
    )

    episodes: dict[str, list[dict[str, float]]] = {}
    summaries: dict[str, dict[str, float]] = {}

    if not log_path.exists():
        return {"episodes": episodes, "summaries": summaries}

    text = log_path.read_text()

    # Parse episode-level data
    for match in episode_pattern.finditer(text):
        name = match.group("name")
        if name not in episodes:
            episodes[name] = []
        episodes[name].append({
            "episode": int(match.group("ep")),
            "reward": float(match.group("reward")),
            "local_hit_rate": float(match.group("local")),
            "neighbor_fetch_rate": float(match.group("neighbor")),
            "cloud_fetch_rate": float(match.group("cloud")),
            "paper_hit_rate": float(match.group("paper_hit")),
        })

    # Parse summary lines
    for match in summary_pattern.finditer(text):
        name = match.group("name")
        summaries[name] = {
            "reward_mean": float(match.group("reward")),
            "local_hit_mean": float(match.group("local")),
            "paper_hit_mean": float(match.group("paper_hit")),
            "cloud_fetch_mean": float(match.group("cloud")),
        }

    return {"episodes": episodes, "summaries": summaries}


def parse_capacity_csv(csv_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Parse capacity_sweep.csv to extract baseline algorithm data by cache capacity."""
    results: dict[str, list[dict[str, Any]]] = {}

    if not csv_path.exists():
        return results

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            if model not in results:
                results[model] = []
            results[model].append({
                "cache_capacity": int(float(row["cache_capacity"])),
                "reward_mean": float(row["reward_mean"]),
                "paper_hit_mean": float(row["paper_hit_mean"]),
                "local_hit_mean": float(row["local_hit_mean"]),
                "neighbor_mean": float(row["neighbor_mean"]),
                "cloud_mean": float(row["cloud_mean"]),
            })

    return results


def parse_train_log(log_path: Path) -> dict[str, Any]:
    """Parse train.log to extract TemporalGraph training data."""
    temporal_pattern = re.compile(
        r"Real-world temporal round (?P<round>\d+)/\d+ \| "
        r"train_loss=(?P<train>[\d\.]+) val_loss=(?P<val>[\d\.]+)"
    )
    imitation_pattern = re.compile(
        r"Imitation epoch (?P<epoch>\d+)/\d+ summary \| "
        r"loss=(?P<loss>[\d\.]+) reward=(?P<reward>[\d\.]+) "
        r"local_hit=(?P<local>[\d\.]+) paper_hit=(?P<paper>[\d\.]+)"
    )
    reinforce_pattern = re.compile(
        r"Reinforce epoch (?P<epoch>\d+)/\d+ summary \| "
        r"loss=(?P<loss>[\-\d\.]+) reward=(?P<reward>[\-\d\.]+) "
        r"local_hit=(?P<local>[\d\.]+) paper_hit=(?P<paper>[\d\.]+)"
    )

    temporal_rounds: list[dict[str, float]] = []
    imitation_epochs: list[dict[str, float]] = []
    reinforce_epochs: list[dict[str, float]] = []

    if not log_path.exists():
        return {
            "temporal_rounds": temporal_rounds,
            "imitation_epochs": imitation_epochs,
            "reinforce_epochs": reinforce_epochs,
        }

    text = log_path.read_text()

    for match in temporal_pattern.finditer(text):
        temporal_rounds.append({
            "round": int(match.group("round")),
            "train_loss": float(match.group("train")),
            "val_loss": float(match.group("val")),
        })

    for match in imitation_pattern.finditer(text):
        imitation_epochs.append({
            "epoch": int(match.group("epoch")),
            "loss": float(match.group("loss")),
            "reward": float(match.group("reward")),
            "local_hit": float(match.group("local")),
            "paper_hit": float(match.group("paper")),
        })

    for match in reinforce_pattern.finditer(text):
        reinforce_epochs.append({
            "epoch": int(match.group("epoch")),
            "loss": float(match.group("loss")),
            "reward": float(match.group("reward")),
            "local_hit": float(match.group("local")),
            "paper_hit": float(match.group("paper")),
        })

    return {
        "temporal_rounds": temporal_rounds,
        "imitation_epochs": imitation_epochs,
        "reinforce_epochs": reinforce_epochs,
    }


def plot_reward_comparison(
    data: dict[str, Any],
    algorithms: list[str],
    out_dir: Path,
) -> None:
    """Plot reward comparison bar chart."""
    summaries = data["related_work"]["summaries"]

    names = []
    rewards = []
    colors = ["#e15759", "#f28e2b", "#4e79a7", "#59a14f", "#76b7b2"]

    for alg in algorithms:
        if alg in summaries:
            names.append(alg)
            rewards.append(summaries[alg]["reward_mean"])

    if not names:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, rewards, color=colors[: len(names)])
    ax.set_title("Reward Comparison (Filtered Algorithms)")
    ax.set_ylabel("Mean Reward")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=15)

    for bar, val in zip(bars, rewards):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_dir / "reward_comparison.png", dpi=180)
    plt.close(fig)


def plot_paper_hit_comparison(
    data: dict[str, Any],
    algorithms: list[str],
    out_dir: Path,
) -> None:
    """Plot paper hit rate comparison bar chart."""
    summaries = data["related_work"]["summaries"]

    names = []
    hits = []
    colors = ["#e15759", "#f28e2b", "#4e79a7", "#59a14f", "#76b7b2"]

    for alg in algorithms:
        if alg in summaries:
            names.append(alg)
            hits.append(summaries[alg]["paper_hit_mean"])

    if not names:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, hits, color=colors[: len(names)])
    ax.set_title("Paper-Style Hit Rate Comparison (Local + Neighbor)")
    ax.set_ylabel("Hit Rate")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=15)

    for bar, val in zip(bars, hits):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_dir / "paper_hit_comparison.png", dpi=180)
    plt.close(fig)


def plot_local_hit_comparison(
    data: dict[str, Any],
    algorithms: list[str],
    out_dir: Path,
) -> None:
    """Plot local hit rate comparison bar chart."""
    summaries = data["related_work"]["summaries"]

    names = []
    hits = []
    colors = ["#e15759", "#f28e2b", "#4e79a7", "#59a14f", "#76b7b2"]

    for alg in algorithms:
        if alg in summaries:
            names.append(alg)
            hits.append(summaries[alg]["local_hit_mean"])

    if not names:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, hits, color=colors[: len(names)])
    ax.set_title("Local Hit Rate Comparison")
    ax.set_ylabel("Local Hit Rate")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=15)

    for bar, val in zip(bars, hits):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_dir / "local_hit_comparison.png", dpi=180)
    plt.close(fig)


def plot_cloud_fetch_comparison(
    data: dict[str, Any],
    algorithms: list[str],
    out_dir: Path,
) -> None:
    """Plot cloud fetch rate comparison bar chart."""
    summaries = data["related_work"]["summaries"]

    names = []
    clouds = []
    colors = ["#e15759", "#f28e2b", "#4e79a7", "#59a14f", "#76b7b2"]

    for alg in algorithms:
        if alg in summaries:
            names.append(alg)
            clouds.append(summaries[alg]["cloud_fetch_mean"])

    if not names:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, clouds, color=colors[: len(names)])
    ax.set_title("Cloud Fetch Rate Comparison")
    ax.set_ylabel("Cloud Fetch Rate")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=15)

    for bar, val in zip(bars, clouds):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_dir / "cloud_fetch_comparison.png", dpi=180)
    plt.close(fig)


def plot_episode_comparison(
    data: dict[str, Any],
    algorithms: list[str],
    out_dir: Path,
) -> None:
    """Plot episode-wise comparison curves."""
    episodes = data["related_work"]["episodes"]

    # Reward vs Episode
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"TemporalGraph": "#1f77b4", "MAAFDRL": "#ff7f0e", "C-epsilon-greedy": "#2ca02c", "BSG-like": "#d62728", "Random": "#9467bd"}

    for alg in algorithms:
        if alg in episodes:
            eps = episodes[alg]
            xs = [e["episode"] for e in eps]
            ys = [e["reward"] for e in eps]
            ax.plot(xs, ys, marker="o", linewidth=2, label=alg, color=colors.get(alg, None))

    ax.set_title("Reward vs Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reward_vs_episode.png", dpi=180)
    plt.close(fig)

    # Paper hit rate vs Episode
    fig, ax = plt.subplots(figsize=(12, 6))
    for alg in algorithms:
        if alg in episodes:
            eps = episodes[alg]
            xs = [e["episode"] for e in eps]
            ys = [e["paper_hit_rate"] for e in eps]
            ax.plot(xs, ys, marker="o", linewidth=2, label=alg, color=colors.get(alg, None))

    ax.set_title("Paper Hit Rate vs Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Paper Hit Rate (local + neighbor)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "paper_hit_vs_episode.png", dpi=180)
    plt.close(fig)


def plot_temporal_training(
    data: dict[str, Any],
    out_dir: Path,
) -> None:
    """Plot temporal encoder training curves from train.log."""
    rounds = data["train"]["temporal_rounds"]
    if not rounds:
        return

    xs = [r["round"] for r in rounds]
    train_loss = [r["train_loss"] for r in rounds]
    val_loss = [r["val_loss"] for r in rounds]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, train_loss, label="Train Loss", marker="o", color="#4e79a7")
    ax.plot(xs, val_loss, label="Val Loss", marker="o", color="#e15759")
    ax.set_title("Temporal Encoder Training Loss")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cross-entropy Loss")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_training_loss.png", dpi=180)
    plt.close(fig)


def plot_rl_training_curves(
    data: dict[str, Any],
    out_dir: Path,
) -> None:
    """Plot RL training curves from train.log."""
    imitation = data["train"]["imitation_epochs"]
    reinforce = data["train"]["reinforce_epochs"]

    if not imitation and not reinforce:
        return

    # Reward curves
    fig, ax = plt.subplots(figsize=(12, 6))

    if imitation:
        xs = [e["epoch"] for e in imitation]
        rewards = [e["reward"] for e in imitation]
        ax.plot(xs, rewards, marker="o", linewidth=2, label="Imitation Learning", color="#4e79a7")

    if reinforce:
        offset = len(imitation) if imitation else 0
        xs = [e["epoch"] + offset for e in reinforce]
        rewards = [e["reward"] for e in reinforce]
        ax.plot(xs, rewards, marker="s", linewidth=2, label="REINFORCE", color="#e15759")

    ax.set_title("TemporalGraph RL Training: Reward per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rl_training_reward.png", dpi=180)
    plt.close(fig)

    # Paper hit rate curves
    fig, ax = plt.subplots(figsize=(12, 6))

    if imitation:
        xs = [e["epoch"] for e in imitation]
        hits = [e["paper_hit"] for e in imitation]
        ax.plot(xs, hits, marker="o", linewidth=2, label="Imitation Learning", color="#4e79a7")

    if reinforce:
        offset = len(imitation) if imitation else 0
        xs = [e["epoch"] + offset for e in reinforce]
        hits = [e["paper_hit"] for e in reinforce]
        ax.plot(xs, hits, marker="s", linewidth=2, label="REINFORCE", color="#e15759")

    ax.set_title("TemporalGraph RL Training: Paper Hit Rate per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Paper Hit Rate")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rl_training_paper_hit.png", dpi=180)
    plt.close(fig)


def save_summary_csv(
    data: dict[str, Any],
    algorithms: list[str],
    out_dir: Path,
) -> None:
    """Save summary statistics to CSV."""
    summaries = data["related_work"]["summaries"]

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm",
            "reward_mean",
            "local_hit_mean",
            "paper_hit_mean",
            "cloud_fetch_mean",
        ])
        for alg in algorithms:
            if alg in summaries:
                s = summaries[alg]
                writer.writerow([
                    alg,
                    f"{s['reward_mean']:.8f}",
                    f"{s['local_hit_mean']:.8f}",
                    f"{s['paper_hit_mean']:.8f}",
                    f"{s['cloud_fetch_mean']:.8f}",
                ])


def plot_capacity_sweep(
    capacity_data: dict[str, list[dict[str, Any]]],
    algorithms: list[str],
    out_dir: Path,
) -> None:
    """Plot capacity sweep comparison (reward and hit rate vs cache capacity)."""
    # Filter to requested algorithms that exist in data
    available_algs = [a for a in algorithms if a in capacity_data]
    if not available_algs:
        return

    colors = {"TemporalGraph": "#1f77b4", "MAAFDRL": "#ff7f0e", "C-epsilon-greedy": "#2ca02c",
              "BSG-like": "#d62728", "Random": "#9467bd", "AWFDRL": "#8c564b"}

    # Reward vs Cache Capacity
    fig, ax = plt.subplots(figsize=(10, 6))
    for alg in available_algs:
        data = capacity_data[alg]
        capacities = [d["cache_capacity"] for d in sorted(data, key=lambda x: x["cache_capacity"])]
        rewards = [d["reward_mean"] for d in sorted(data, key=lambda x: x["cache_capacity"])]
        ax.plot(capacities, rewards, marker="o", linewidth=2, label=alg, color=colors.get(alg, None))

    ax.set_title("Reward vs Cache Capacity")
    ax.set_xlabel("Cache Capacity (C)")
    ax.set_ylabel("Mean Reward")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reward_vs_capacity.png", dpi=180)
    plt.close(fig)

    # Paper Hit vs Cache Capacity
    fig, ax = plt.subplots(figsize=(10, 6))
    for alg in available_algs:
        data = capacity_data[alg]
        capacities = [d["cache_capacity"] for d in sorted(data, key=lambda x: x["cache_capacity"])]
        hits = [d["paper_hit_mean"] for d in sorted(data, key=lambda x: x["cache_capacity"])]
        ax.plot(capacities, hits, marker="o", linewidth=2, label=alg, color=colors.get(alg, None))

    ax.set_title("Paper Hit Rate vs Cache Capacity")
    ax.set_xlabel("Cache Capacity (C)")
    ax.set_ylabel("Paper Hit Rate")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "paper_hit_vs_capacity.png", dpi=180)
    plt.close(fig)

    # Local Hit vs Cache Capacity
    fig, ax = plt.subplots(figsize=(10, 6))
    for alg in available_algs:
        data = capacity_data[alg]
        capacities = [d["cache_capacity"] for d in sorted(data, key=lambda x: x["cache_capacity"])]
        hits = [d["local_hit_mean"] for d in sorted(data, key=lambda x: x["cache_capacity"])]
        ax.plot(capacities, hits, marker="o", linewidth=2, label=alg, color=colors.get(alg, None))

    ax.set_title("Local Hit Rate vs Cache Capacity")
    ax.set_xlabel("Cache Capacity (C)")
    ax.set_ylabel("Local Hit Rate")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "local_hit_vs_capacity.png", dpi=180)
    plt.close(fig)

    # Cloud Fetch vs Cache Capacity
    fig, ax = plt.subplots(figsize=(10, 6))
    for alg in available_algs:
        data = capacity_data[alg]
        capacities = [d["cache_capacity"] for d in sorted(data, key=lambda x: x["cache_capacity"])]
        clouds = [d["cloud_mean"] for d in sorted(data, key=lambda x: x["cache_capacity"])]
        ax.plot(capacities, clouds, marker="o", linewidth=2, label=alg, color=colors.get(alg, None))

    ax.set_title("Cloud Fetch Rate vs Cache Capacity")
    ax.set_xlabel("Cache Capacity (C)")
    ax.set_ylabel("Cloud Fetch Rate")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cloud_fetch_vs_capacity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {args.train_log}...")
    train_data = parse_train_log(args.train_log)

    print(f"Parsing {args.related_work_log}...")
    related_work_data = parse_related_work_log(args.related_work_log)

    print(f"Parsing {args.capacity_csv}...")
    capacity_data = parse_capacity_csv(args.capacity_csv)

    data = {"train": train_data, "related_work": related_work_data, "capacity": capacity_data}

    print("Generating plots...")

    # Plot related work comparisons (from log)
    plot_reward_comparison(data, args.algorithms, args.output_dir)
    plot_paper_hit_comparison(data, args.algorithms, args.output_dir)
    plot_local_hit_comparison(data, args.algorithms, args.output_dir)
    plot_cloud_fetch_comparison(data, args.algorithms, args.output_dir)
    plot_episode_comparison(data, args.algorithms, args.output_dir)

    # Plot capacity sweep comparisons (from CSV)
    plot_capacity_sweep(capacity_data, args.algorithms, args.output_dir)

    # Plot training curves from train.log
    plot_temporal_training(data, args.output_dir)
    plot_rl_training_curves(data, args.output_dir)

    # Save summary CSV
    save_summary_csv(data, args.algorithms, args.output_dir)

    print(f"\nDone! Plots saved to: {args.output_dir}")
    print(f"  - reward_comparison.png (from related_work.log)")
    print(f"  - paper_hit_comparison.png (from related_work.log)")
    print(f"  - local_hit_comparison.png (from related_work.log)")
    print(f"  - cloud_fetch_comparison.png (from related_work.log)")
    print(f"  - reward_vs_episode.png (from related_work.log)")
    print(f"  - paper_hit_vs_episode.png (from related_work.log)")
    print(f"  - reward_vs_capacity.png (from capacity_sweep.csv)")
    print(f"  - paper_hit_vs_capacity.png (from capacity_sweep.csv)")
    print(f"  - local_hit_vs_capacity.png (from capacity_sweep.csv)")
    print(f"  - cloud_fetch_vs_capacity.png (from capacity_sweep.csv)")
    print(f"  - temporal_training_loss.png (from train.log)")
    print(f"  - rl_training_reward.png (from train.log)")
    print(f"  - rl_training_paper_hit.png (from train.log)")
    print(f"  - summary.csv")


if __name__ == "__main__":
    main()
