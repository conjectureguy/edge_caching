from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate episode/epoch-only curves from a training run.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def load_csv(path: Path) -> list[dict[str, float]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, float]] = []
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
        return rows


def plot_single(x, y, title: str, xlabel: str, ylabel: str, out_path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(x, y, marker="o", linewidth=2.0, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or args.input_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    imitation_rows = load_csv(args.input_dir / "policy_imitation.csv")
    eval_rows = load_csv(args.input_dir / "temporal_graph_eval.csv")

    epochs = [int(r["epoch"]) for r in imitation_rows]
    local_epoch = [r["local_hit_rate"] for r in imitation_rows]
    paper_epoch = [r["paper_hit_rate"] for r in imitation_rows]

    episodes = [int(r["episode"]) for r in eval_rows]
    reward_ep = [r["reward"] for r in eval_rows]
    paper_ep = [r["paper_hit_rate"] for r in eval_rows]
    cloud_ep = [r["cloud_fetch_rate"] for r in eval_rows]

    plot_single(
        epochs,
        local_epoch,
        "Local Hit vs Epoch",
        "Epoch",
        "Local Hit Rate",
        out_dir / "local_hit_vs_epoch.png",
        "#f4a261",
    )
    plot_single(
        epochs,
        paper_epoch,
        "Paper Hit vs Epoch",
        "Epoch",
        "Paper Hit Rate",
        out_dir / "paper_hit_vs_epoch.png",
        "#d62828",
    )
    plot_single(
        episodes,
        reward_ep,
        "Reward vs Episode",
        "Evaluation Episode",
        "Reward",
        out_dir / "reward_vs_episode_only.png",
        "#d62828",
    )
    plot_single(
        episodes,
        paper_ep,
        "Paper Hit vs Episode",
        "Evaluation Episode",
        "Paper Hit Rate",
        out_dir / "paper_hit_vs_episode_only.png",
        "#e76f51",
    )
    plot_single(
        episodes,
        cloud_ep,
        "Cloud Fetch vs Episode",
        "Evaluation Episode",
        "Cloud Fetch Rate",
        out_dir / "cloud_fetch_vs_episode_only.png",
        "#577590",
    )

    print(f"Saved episode/epoch curves to: {out_dir}")


if __name__ == "__main__":
    main()
