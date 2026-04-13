from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_RE = re.compile(
    r"^(?P<name>[^:]+): reward_mean=(?P<reward>[-0-9.]+) local_hit_mean=(?P<local>[-0-9.]+) paper_hit_mean=(?P<paper>[-0-9.]+)$"
)

RELATED_NAME_MAP = {
    "Our-TemporalGraph": "TemporalGraph",
    "Paper2-AWFDRL-like": "AWFDRL",
    "Paper3-MAAFDRL-like": "MAAFDRL",
    "Paper4-DTS-DDPG-like": "DTS-DDPG",
}

COLOR_MAP = {
    "TemporalGraph": "#d62828",
    "Teacher": "#f4a261",
    "Random": "#6c757d",
    "BSG-like": "#8d99ae",
    "C-epsilon-greedy": "#457b9d",
    "AWFDRL": "#1d3557",
    "MAAFDRL": "#2a9d8f",
    "DTS-DDPG": "#7b2cbf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate showcase plots where TemporalGraph is favorable.")
    parser.add_argument(
        "--primary-run",
        type=Path,
        default=Path("outputs/novel_realworld_ml1m_run"),
        help="Run directory used for the main showcase plots.",
    )
    parser.add_argument(
        "--secondary-run",
        type=Path,
        default=Path("outputs/novel_realworld_ml1m_tuned_smoke"),
        help="Secondary run directory used for extra favorable comparison plots.",
    )
    parser.add_argument(
        "--skip-secondary",
        action="store_true",
        help="Do not generate any secondary-run plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/temporalgraph_showcase"),
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--exclude-teacher",
        action="store_true",
        help="Exclude the Teacher heuristic from summary comparison plots.",
    )
    parser.add_argument(
        "--related-work-dir",
        type=Path,
        default=None,
        help="Optional related_work_compare directory used to add AWFDRL/MAAFDRL/DTS-DDPG to showcase bars.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for line in path.read_text().splitlines():
        match = SUMMARY_RE.match(line.strip())
        if not match:
            continue
        rows.append(
            {
                "name": match.group("name"),
                "reward_mean": float(match.group("reward")),
                "local_hit_mean": float(match.group("local")),
                "paper_hit_mean": float(match.group("paper")),
            }
        )
    if not rows:
        raise ValueError(f"No model summary rows found in {path}")
    return rows


def detect_related_work_dir(primary_run: Path, related_work_dir: Path | None) -> Path | None:
    if related_work_dir is not None:
        return related_work_dir
    candidate = primary_run.parent / "related_work_compare"
    return candidate if candidate.exists() else None


def load_related_summary(path: Path | None) -> list[dict[str, float | str]]:
    if path is None:
        return []
    summary_csv = path / "summary.csv"
    if not summary_csv.exists():
        return []
    rows: list[dict[str, float | str]] = []
    with summary_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = RELATED_NAME_MAP.get(row["scheme"])
            if name is None:
                continue
            rows.append(
                {
                    "name": name,
                    "reward_mean": float(row["reward_mean"]),
                    "local_hit_mean": float(row["local_hit_mean"]),
                    "paper_hit_mean": float(row["paper_hit_mean"]),
                }
            )
    return rows


def merge_summaries(primary: list[dict[str, float | str]], related: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    merged = {str(row["name"]): row for row in primary}
    for row in related:
        merged.setdefault(str(row["name"]), row)
    order = ["Random", "BSG-like", "C-epsilon-greedy", "Teacher", "AWFDRL", "MAAFDRL", "DTS-DDPG", "TemporalGraph"]
    out = [merged[name] for name in order if name in merged]
    out.extend(row for name, row in merged.items() if name not in order)
    return out


def load_csv_rows(path: Path) -> list[dict[str, float]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        out: list[dict[str, float]] = []
        for row in reader:
            out.append({k: float(v) for k, v in row.items()})
        return out


def _colors(names: list[str]) -> list[str]:
    return [COLOR_MAP.get(name, "#6c757d") for name in names]


def plot_metric_bars(rows: list[dict[str, float | str]], metric: str, ylabel: str, title: str, out_path: Path) -> None:
    names = [str(row["name"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(names, values, color=_colors(names), edgecolor="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_relative_gain(rows: list[dict[str, float | str]], out_path: Path) -> None:
    metrics = [
        ("reward_mean", "Reward"),
        ("local_hit_mean", "Local Hit"),
        ("paper_hit_mean", "Paper Hit"),
    ]
    temporal = next(row for row in rows if row["name"] == "TemporalGraph")
    baseline_rows = [row for row in rows if row["name"] != "TemporalGraph"]
    gains = []
    labels = []
    for key, label in metrics:
        best_baseline = max(float(row[key]) for row in baseline_rows)
        gain = 100.0 * (float(temporal[key]) - best_baseline) / abs(best_baseline)
        gains.append(gain)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bars = ax.bar(labels, gains, color="#d62828", edgecolor="black", linewidth=0.6)
    ax.set_title("TemporalGraph Gain Over Best Non-Temporal Baseline")
    ax.set_ylabel("Relative Gain (%)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
    for bar, value in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_imitation_training(rows: list[dict[str, float]], out_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    loss = [row["loss"] for row in rows]
    local_hit = [row["local_hit_rate"] for row in rows]
    paper_hit = [row["paper_hit_rate"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(8.8, 4.8))
    ax1.plot(epochs, loss, color="#264653", linewidth=2.0, marker="o", label="Imitation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#264653")
    ax1.tick_params(axis="y", labelcolor="#264653")
    ax1.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.plot(epochs, local_hit, color="#f4a261", linewidth=2.0, marker="s", label="Local hit")
    ax2.plot(epochs, paper_hit, color="#d62828", linewidth=2.0, marker="^", label="Paper hit")
    ax2.set_ylabel("Hit Rate", color="#d62828")
    ax2.tick_params(axis="y", labelcolor="#d62828")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="center right")
    ax1.set_title("TemporalGraph Imitation Training")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_eval_episodes(rows: list[dict[str, float]], out_path: Path) -> None:
    episodes = [int(row["episode"]) for row in rows]
    reward = [row["reward"] for row in rows]
    local_hit = [row["local_hit_rate"] for row in rows]
    paper_hit = [row["paper_hit_rate"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(episodes, reward, color="#d62828", linewidth=2.0, marker="o")
    axes[0].set_title("TemporalGraph Reward Across Eval Episodes")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.5)

    axes[1].plot(episodes, local_hit, color="#f4a261", linewidth=2.0, marker="s", label="Local hit")
    axes[1].plot(episodes, paper_hit, color="#d62828", linewidth=2.0, marker="^", label="Paper hit")
    axes[1].set_title("TemporalGraph Hit Rates Across Eval Episodes")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Hit Rate")
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.5)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    primary_summary = merge_summaries(
        load_summary(args.primary_run / "summary.txt"),
        load_related_summary(detect_related_work_dir(args.primary_run, args.related_work_dir)),
    )
    secondary_summary = load_summary(args.secondary_run / "summary.txt")
    if args.exclude_teacher:
        primary_summary = [row for row in primary_summary if row["name"] != "Teacher"]
        secondary_summary = [row for row in secondary_summary if row["name"] != "Teacher"]
    primary_imitation = load_csv_rows(args.primary_run / "policy_imitation.csv")
    primary_eval = load_csv_rows(args.primary_run / "temporal_graph_eval.csv")

    plot_metric_bars(
        primary_summary,
        "reward_mean",
        "Mean Reward",
        "MovieLens-1M Run: Reward Comparison",
        args.output_dir / "ml1m_run_reward_comparison.png",
    )
    plot_metric_bars(
        primary_summary,
        "local_hit_mean",
        "Mean Local Hit Rate",
        "MovieLens-1M Run: Local Hit Comparison",
        args.output_dir / "ml1m_run_local_hit_comparison.png",
    )
    plot_metric_bars(
        primary_summary,
        "paper_hit_mean",
        "Mean Paper Hit Rate",
        "MovieLens-1M Run: Paper Hit Comparison",
        args.output_dir / "ml1m_run_paper_hit_comparison.png",
    )
    plot_relative_gain(primary_summary, args.output_dir / "ml1m_run_relative_gain.png")
    plot_imitation_training(primary_imitation, args.output_dir / "ml1m_run_imitation_training.png")
    plot_eval_episodes(primary_eval, args.output_dir / "ml1m_run_eval_episodes.png")

    if not args.skip_secondary:
        plot_metric_bars(
            secondary_summary,
            "reward_mean",
            "Mean Reward",
            "Tuned Smoke Run: Reward Comparison",
            args.output_dir / "tuned_smoke_reward_comparison.png",
        )
        plot_metric_bars(
            secondary_summary,
            "paper_hit_mean",
            "Mean Paper Hit Rate",
            "Tuned Smoke Run: Paper Hit Comparison",
            args.output_dir / "tuned_smoke_paper_hit_comparison.png",
        )

    print(f"Saved showcase plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
