from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "random": "Random",
    "bsg_like": "BSG-like",
    "c_epsilon_greedy": "C-epsilon-greedy",
    "temporal_graph": "TemporalGraph",
    "awfdrl": "AWFDRL",
    "maafdrl": "MAAFDRL",
    "dts_ddpg": "DTS-DDPG",
}

COLORS = {
    "random": "#6c757d",
    "bsg_like": "#8d99ae",
    "c_epsilon_greedy": "#457b9d",
    "temporal_graph": "#d62828",
    "awfdrl": "#1d3557",
    "maafdrl": "#2a9d8f",
    "dts_ddpg": "#7b2cbf",
}

RELATED_NAME_MAP = {
    "Paper2-AWFDRL-like": "awfdrl",
    "Paper3-MAAFDRL-like": "maafdrl",
    "Paper4-DTS-DDPG-like": "dts_ddpg",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate no-teacher comparison plots for the final TemporalGraph run.")
    p.add_argument("--input-dir", type=Path, default=Path("outputs/novel_realworld_ml1m_final"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/final_no_teacher_bundle"))
    p.add_argument("--related-work-dir", type=Path, default=None)
    return p.parse_args()


def load_eval_csv(path: Path) -> list[dict[str, float]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return [{k: float(v) for k, v in row.items()} for row in reader]


def detect_related_work_dir(input_dir: Path, related_work_dir: Path | None) -> Path | None:
    if related_work_dir is not None:
        return related_work_dir
    candidate = input_dir.parent / "related_work_compare"
    return candidate if candidate.exists() else None


def model_rows(input_dir: Path, related_work_dir: Path | None) -> dict[str, list[dict[str, float]]]:
    rows_by_model: dict[str, list[dict[str, float]]] = {
        "random": load_eval_csv(input_dir / "random_eval.csv"),
        "bsg_like": load_eval_csv(input_dir / "bsg_like_eval.csv"),
        "c_epsilon_greedy": load_eval_csv(input_dir / "c_epsilon_greedy_eval.csv"),
        "temporal_graph": load_eval_csv(input_dir / "temporal_graph_eval.csv"),
    }
    if related_work_dir is None:
        return rows_by_model
    path = related_work_dir / "episode_metrics.csv"
    if not path.exists():
        return rows_by_model
    with path.open() as f:
        reader = csv.DictReader(f)
        grouped: dict[str, list[dict[str, float]]] = {}
        for row in reader:
            key = RELATED_NAME_MAP.get(row["scheme"])
            if key is None:
                continue
            grouped.setdefault(key, []).append({k: float(v) for k, v in row.items() if k != "scheme"})
        rows_by_model.update(grouped)
    return rows_by_model


def metric_means(rows_by_model: dict[str, list[dict[str, float]]], metric: str) -> dict[str, float]:
    return {
        model: float(np.mean([row[metric] for row in rows]))
        for model, rows in rows_by_model.items()
    }


def bar_metric(rows_by_model: dict[str, list[dict[str, float]]], metric: str, title: str, ylabel: str, out_path: Path, lower_is_better: bool = False) -> None:
    names = list(rows_by_model.keys())
    values = [metric_means(rows_by_model, metric)[name] for name in names]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(
        [DISPLAY_NAMES[n] for n in names],
        values,
        color=[COLORS[n] for n in names],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
    ax.tick_params(axis="x", rotation=18)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    if lower_is_better:
        ax.text(0.98, 0.95, "Lower is better", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def line_metric(rows_by_model: dict[str, list[dict[str, float]]], metric: str, title: str, ylabel: str, out_path: Path, lower_is_better: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for name, rows in rows_by_model.items():
        ax.plot(
            [int(r["episode"]) for r in rows],
            [r[metric] for r in rows],
            marker="o",
            linewidth=2.0,
            color=COLORS[name],
            label=DISPLAY_NAMES[name],
        )
    ax.set_title(title)
    ax.set_xlabel("Evaluation Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend()
    if lower_is_better:
        ax.text(0.98, 0.95, "Lower is better", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def stacked_service(rows_by_model: dict[str, list[dict[str, float]]], out_path: Path) -> None:
    names = list(rows_by_model.keys())
    local = [metric_means(rows_by_model, "local_hit_rate")[n] for n in names]
    neighbor = [metric_means(rows_by_model, "neighbor_fetch_rate")[n] for n in names]
    cloud = [metric_means(rows_by_model, "cloud_fetch_rate")[n] for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(x, local, color="#2a9d8f", edgecolor="black", linewidth=0.6, label="Local served")
    ax.bar(x, neighbor, bottom=local, color="#e9c46a", edgecolor="black", linewidth=0.6, label="Neighbor served")
    ax.bar(x, cloud, bottom=np.array(local) + np.array(neighbor), color="#adb5bd", edgecolor="black", linewidth=0.6, label="Cloud served")
    ax.set_xticks(x, [DISPLAY_NAMES[n] for n in names], rotation=18)
    ax.set_ylabel("Request Fraction")
    ax.set_title("Service Source Composition")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def edge_offload_gain(rows_by_model: dict[str, list[dict[str, float]]], out_path: Path) -> None:
    temporal = metric_means(rows_by_model, "paper_hit_rate")["temporal_graph"]
    gains = []
    labels = []
    for name in rows_by_model:
        if name == "temporal_graph":
            continue
        base = metric_means(rows_by_model, "paper_hit_rate")[name]
        gains.append(100.0 * (temporal - base) / max(base, 1e-8))
        labels.append(DISPLAY_NAMES[name])

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.bar(labels, gains, color="#d62828", edgecolor="black", linewidth=0.6)
    ax.set_title("TemporalGraph Paper-Hit Gain Over Baselines")
    ax.set_ylabel("Gain (%)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
    for bar, val in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def win_count_plot(rows_by_model: dict[str, list[dict[str, float]]], out_path: Path) -> None:
    metrics = {
        "Reward wins": ("reward", "max"),
        "Paper-hit wins": ("paper_hit_rate", "max"),
        "Cloud-min wins": ("cloud_fetch_rate", "min"),
        "Neighbor-use wins": ("neighbor_fetch_rate", "max"),
    }
    counts = []
    labels = []
    episode_sets = [
        {int(row["episode"]) for row in rows}
        for rows in rows_by_model.values()
        if rows
    ]
    shared_episodes = sorted(set.intersection(*episode_sets)) if episode_sets else []
    for label, (metric, mode) in metrics.items():
        wins = 0
        for episode in shared_episodes:
            vals = {
                name: next(row[metric] for row in rows if int(row["episode"]) == episode)
                for name, rows in rows_by_model.items()
                if any(int(row["episode"]) == episode for row in rows)
            }
            best = max(vals.values()) if mode == "max" else min(vals.values())
            if abs(vals["temporal_graph"] - best) < 1e-10:
                wins += 1
        labels.append(label)
        counts.append(wins)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(labels, counts, color="#d62828", edgecolor="black", linewidth=0.6)
    ax.set_title("TemporalGraph Episode-Level Win Count")
    ax.set_ylabel("Winning Episodes")
    ax.set_ylim(0, len(shared_episodes) + 0.5)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val}/{len(shared_episodes)}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_model = model_rows(args.input_dir, detect_related_work_dir(args.input_dir, args.related_work_dir))

    bar_metric(rows_by_model, "reward", "Mean Reward Comparison", "Mean Episode Reward", args.output_dir / "reward_mean_no_teacher.png")
    bar_metric(rows_by_model, "paper_hit_rate", "Mean Paper-Hit Comparison", "Mean Paper-Hit Rate", args.output_dir / "paper_hit_mean_no_teacher.png")
    bar_metric(rows_by_model, "cloud_fetch_rate", "Mean Cloud-Fetch Comparison", "Mean Cloud-Fetch Rate", args.output_dir / "cloud_fetch_mean_no_teacher.png", lower_is_better=True)
    bar_metric(rows_by_model, "neighbor_fetch_rate", "Mean Neighbor-Fetch Comparison", "Mean Neighbor-Fetch Rate", args.output_dir / "neighbor_fetch_mean_no_teacher.png")

    line_metric(rows_by_model, "reward", "Reward Across Evaluation Episodes", "Episode Reward", args.output_dir / "reward_vs_episode_no_teacher.png")
    line_metric(rows_by_model, "paper_hit_rate", "Paper-Hit Across Evaluation Episodes", "Paper-Hit Rate", args.output_dir / "paper_hit_vs_episode_no_teacher.png")
    line_metric(rows_by_model, "cloud_fetch_rate", "Cloud-Fetch Across Evaluation Episodes", "Cloud-Fetch Rate", args.output_dir / "cloud_fetch_vs_episode_no_teacher.png", lower_is_better=True)
    line_metric(rows_by_model, "neighbor_fetch_rate", "Neighbor-Fetch Across Evaluation Episodes", "Neighbor-Fetch Rate", args.output_dir / "neighbor_fetch_vs_episode_no_teacher.png")

    stacked_service(rows_by_model, args.output_dir / "service_source_composition_no_teacher.png")
    edge_offload_gain(rows_by_model, args.output_dir / "paper_hit_gain_no_teacher.png")
    win_count_plot(rows_by_model, args.output_dir / "temporalgraph_win_count_no_teacher.png")

    print(f"Saved final no-teacher plot bundle to: {args.output_dir}")


if __name__ == "__main__":
    main()
