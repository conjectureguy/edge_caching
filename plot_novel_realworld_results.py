from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot results for the novel real-world cooperative caching pipeline.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def maybe_plot_temporal_training(input_dir: Path, output_dir: Path) -> None:
    path = input_dir / "temporal_training.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    rounds = [int(r["round"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds, train_loss, marker="o", label="Train loss")
    plt.plot(rounds, val_loss, marker="s", label="Val loss")
    plt.xlabel("Federated round")
    plt.ylabel("Cross-entropy loss")
    plt.title("Temporal Encoder Training")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_training.png", dpi=180)
    plt.close()


def maybe_plot_policy_imitation(input_dir: Path, output_dir: Path) -> None:
    path = input_dir / "policy_imitation.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    epochs = [int(r["epoch"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    reward = [float(r["reward"]) for r in rows]
    local = [float(r["local_hit_rate"]) for r in rows]
    paper = [float(r["paper_hit_rate"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(epochs, loss, marker="o", color="#b23a48")
    axes[0].set_title("Imitation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, reward, marker="o", color="#2a6f97")
    axes[1].set_title("Reward During Imitation")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)

    axes[2].plot(epochs, local, marker="o", label="Local hit", color="#3a7d44")
    axes[2].plot(epochs, paper, marker="s", label="Paper hit", color="#f4a259")
    axes[2].set_title("Hit Rate During Imitation")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "policy_imitation.png", dpi=180)
    plt.close(fig)


def plot_eval_bars(input_dir: Path, output_dir: Path) -> None:
    eval_files = [
        ("random_eval.csv", "Random"),
        ("bsg_like_eval.csv", "BSG-like"),
        ("c_epsilon_greedy_eval.csv", "C-epsilon-greedy"),
        ("teacher_eval.csv", "Teacher"),
        ("temporal_graph_eval.csv", "TemporalGraph"),
    ]
    summary = []
    for filename, label in eval_files:
        path = input_dir / filename
        if not path.exists():
            continue
        rows = read_csv(path)
        summary.append(
            {
                "label": label,
                "reward": float(np.mean([float(r["reward"]) for r in rows])),
                "local": float(np.mean([float(r["local_hit_rate"]) for r in rows])),
                "paper": float(np.mean([float(r["paper_hit_rate"]) for r in rows])),
                "cloud": float(np.mean([float(r["cloud_fetch_rate"]) for r in rows])),
            }
        )

    if not summary:
        return

    labels = [x["label"] for x in summary]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(x, [s["reward"] for s in summary], color=["#8d99ae", "#577590", "#90be6d", "#f8961e", "#d62828"][: len(summary)])
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_title("Reward Comparison")
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.35
    axes[1].bar(x - width / 2, [s["local"] for s in summary], width=width, label="Local hit", color="#3a7d44")
    axes[1].bar(x + width / 2, [s["paper"] for s in summary], width=width, label="Paper hit", color="#f4a259")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_title("Hit Rate Comparison")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    axes[2].bar(x, [s["cloud"] for s in summary], color="#457b9d")
    axes[2].set_xticks(x, labels, rotation=20, ha="right")
    axes[2].set_title("Cloud Fetch Rate")
    axes[2].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / "baseline_comparison.png", dpi=180)
    plt.close(fig)


def plot_eval_episode_curves(input_dir: Path, output_dir: Path) -> None:
    eval_files = [
        ("random_eval.csv", "Random", "#8d99ae"),
        ("bsg_like_eval.csv", "BSG-like", "#577590"),
        ("c_epsilon_greedy_eval.csv", "C-epsilon-greedy", "#90be6d"),
        ("teacher_eval.csv", "Teacher", "#f8961e"),
        ("temporal_graph_eval.csv", "TemporalGraph", "#d62828"),
    ]
    available = []
    for filename, label, color in eval_files:
        path = input_dir / filename
        if path.exists():
            available.append((read_csv(path), label, color))
    if not available:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for rows, label, color in available:
        episodes = [int(r["episode"]) for r in rows]
        reward = [float(r["reward"]) for r in rows]
        paper = [float(r["paper_hit_rate"]) for r in rows]
        axes[0].plot(episodes, reward, marker="o", label=label, color=color)
        axes[1].plot(episodes, paper, marker="o", label=label, color=color)

    axes[0].set_title("Reward by Episode")
    axes[0].set_xlabel("Episode")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Paper Hit by Episode")
    axes[1].set_xlabel("Episode")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "episode_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    maybe_plot_temporal_training(args.input_dir, output_dir)
    maybe_plot_policy_imitation(args.input_dir, output_dir)
    plot_eval_bars(args.input_dir, output_dir)
    plot_eval_episode_curves(args.input_dir, output_dir)
    print(f"Saved plots under: {output_dir}")


if __name__ == "__main__":
    main()
