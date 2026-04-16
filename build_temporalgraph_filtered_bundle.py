#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

_MPL_DIR = Path("outputs/.mplconfig")
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


PNG_ORDER = ["Random", "BSG-like", "C-epsilon-greedy", "MAAFDRL", "TemporalGraph"]
PDF_ORDER = ["Random", "BSG", "C-\u03b5-greedy", "MAAFDRL", "TemporalGraph"]
PNG_TO_PDF = {
    "Random": "Random",
    "BSG-like": "BSG",
    "C-epsilon-greedy": "C-\u03b5-greedy",
    "MAAFDRL": "MAAFDRL",
    "TemporalGraph": "TemporalGraph",
}
PNG_COLORS = {
    "Random": "#8d99ae",
    "BSG-like": "#577590",
    "C-epsilon-greedy": "#90be6d",
    "MAAFDRL": "#2a9d8f",
    "TemporalGraph": "#d62828",
}
SHOWCASE_COLORS = {
    "Random": "#6c757d",
    "BSG-like": "#8d99ae",
    "C-epsilon-greedy": "#457b9d",
    "MAAFDRL": "#2a9d8f",
    "TemporalGraph": "#d62828",
}
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.7",
    "figure.constrained_layout.use": True,
}
PAPER_SCHEME_STYLE = {
    "TemporalGraph": {"color": "#d62828", "marker": "D", "linestyle": "-", "hatch": "///", "zorder": 10},
    "MAAFDRL": {"color": "#2a9d8f", "marker": "^", "linestyle": "-.", "hatch": "xxx", "zorder": 7},
    "C-\u03b5-greedy": {"color": "#e76f51", "marker": "P", "linestyle": (0, (5, 1)), "hatch": "---", "zorder": 5},
    "BSG": {"color": "#8d99ae", "marker": "o", "linestyle": (0, (3, 1, 1, 1)), "hatch": "+++", "zorder": 4},
    "Random": {"color": "#6c757d", "marker": "v", "linestyle": ":", "hatch": "...", "zorder": 3},
}
RELATED_BAR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]


def parse_args() -> argparse.Namespace:
    default_source = Path("outputs/research_benchmark_runs/temporalgraph_20260415_041120")
    parser = argparse.ArgumentParser(
        description="Clone a TemporalGraph research benchmark bundle and regenerate the plots with only TemporalGraph, MAAFDRL, C-epsilon-greedy, BSG, and Random."
    )
    parser.add_argument("--source-root", type=Path, default=default_source)
    parser.add_argument("--output-root", type=Path, default=default_source.with_name(default_source.name + "_filtered_no_awfdrl"))
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def load_float_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in load_csv(path):
        rows.append({key: float(value) for key, value in row.items()})
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ordered_png_names(names: list[str]) -> list[str]:
    ordered = [name for name in PNG_ORDER if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def ordered_pdf_names(names: list[str]) -> list[str]:
    ordered = [name for name in PDF_ORDER if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def paper_scheme_style(name: str) -> dict[str, object]:
    return PAPER_SCHEME_STYLE.get(name, {"color": "#333333", "marker": "o", "linestyle": "-", "hatch": "", "zorder": 1})


def save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "reward_mean": mean_metric(rows, "reward"),
        "local_hit_mean": mean_metric(rows, "local_hit_rate"),
        "neighbor_fetch_mean": mean_metric(rows, "neighbor_fetch_rate"),
        "cloud_fetch_mean": mean_metric(rows, "cloud_fetch_rate"),
        "paper_hit_mean": mean_metric(rows, "paper_hit_rate"),
    }


def build_rows_by_model(source_root: Path) -> dict[str, list[dict[str, float]]]:
    main_dir = source_root / "novel_realworld_main"
    related_dir = source_root / "related_work_compare"
    rows_by_model = {
        "Random": load_float_csv(main_dir / "random_eval.csv"),
        "BSG-like": load_float_csv(main_dir / "bsg_like_eval.csv"),
        "C-epsilon-greedy": load_float_csv(main_dir / "c_epsilon_greedy_eval.csv"),
        "TemporalGraph": load_float_csv(main_dir / "temporal_graph_eval.csv"),
    }
    episode_rows = load_csv(related_dir / "episode_metrics.csv")
    maafdrl_rows = []
    for row in episode_rows:
        if row["scheme"] != "MAAFDRL":
            continue
        maafdrl_rows.append(
            {
                "episode": float(row["episode"]),
                "reward": float(row["reward"]),
                "local_hit_rate": float(row["local_hit_rate"]),
                "neighbor_fetch_rate": float(row["neighbor_fetch_rate"]),
                "cloud_fetch_rate": float(row["cloud_fetch_rate"]),
                "paper_hit_rate": float(row["paper_hit_rate"]),
            }
        )
    if not maafdrl_rows:
        raise ValueError("MAAFDRL rows missing from related_work_compare/episode_metrics.csv")
    rows_by_model["MAAFDRL"] = maafdrl_rows
    return rows_by_model


def build_summary_rows(rows_by_model: dict[str, list[dict[str, float]]]) -> list[dict[str, object]]:
    rows = []
    for name in ordered_png_names(list(rows_by_model.keys())):
        summary = summarize_rows(rows_by_model[name])
        rows.append(
            {
                "scheme": name,
                "reward_mean": f"{summary['reward_mean']:.8f}",
                "local_hit_mean": f"{summary['local_hit_mean']:.8f}",
                "neighbor_fetch_mean": f"{summary['neighbor_fetch_mean']:.8f}",
                "cloud_fetch_mean": f"{summary['cloud_fetch_mean']:.8f}",
                "paper_hit_mean": f"{summary['paper_hit_mean']:.8f}",
            }
        )
    return rows


def build_episode_rows(rows_by_model: dict[str, list[dict[str, float]]]) -> list[dict[str, object]]:
    rows = []
    for name in ordered_png_names(list(rows_by_model.keys())):
        for row in rows_by_model[name]:
            rows.append(
                {
                    "scheme": name,
                    "episode": int(row["episode"]),
                    "reward": f"{row['reward']:.8f}",
                    "local_hit_rate": f"{row['local_hit_rate']:.8f}",
                    "neighbor_fetch_rate": f"{row['neighbor_fetch_rate']:.8f}",
                    "cloud_fetch_rate": f"{row['cloud_fetch_rate']:.8f}",
                    "paper_hit_rate": f"{row['paper_hit_rate']:.8f}",
                }
            )
    return rows


def plot_related_work_png(out_dir: Path, rows_by_model: dict[str, list[dict[str, float]]]) -> None:
    clear_dir(out_dir)
    summary_rows = build_summary_rows(rows_by_model)
    episode_rows = build_episode_rows(rows_by_model)
    write_csv(
        out_dir / "summary.csv",
        ["scheme", "reward_mean", "local_hit_mean", "neighbor_fetch_mean", "cloud_fetch_mean", "paper_hit_mean"],
        summary_rows,
    )
    write_csv(
        out_dir / "episode_metrics.csv",
        ["scheme", "episode", "reward", "local_hit_rate", "neighbor_fetch_rate", "cloud_fetch_rate", "paper_hit_rate"],
        episode_rows,
    )
    summary_lines = [
        f"Run dir: {out_dir.parent / 'novel_realworld_main'}",
        "Dataset: ml-1m",
        f"Evaluation episodes: {len(rows_by_model['TemporalGraph'])}",
    ]
    for name in ordered_png_names(list(rows_by_model.keys())):
        summary = summarize_rows(rows_by_model[name])
        summary_lines.append(
            f"{name}: reward_mean={summary['reward_mean']:.4f} "
            f"local_hit_mean={summary['local_hit_mean']:.4f} "
            f"paper_hit_mean={summary['paper_hit_mean']:.4f} "
            f"cloud_fetch_mean={summary['cloud_fetch_mean']:.4f}"
        )
    write_text(out_dir / "summary.txt", "\n".join(summary_lines) + "\n")

    names = ordered_png_names(list(rows_by_model.keys()))
    metrics = {
        "reward_comparison.png": ("reward", "Related Work Comparison: Reward", "Reward"),
        "local_hit_comparison.png": ("local_hit_rate", "Related Work Comparison: Local Hit Rate", "Local Hit Rate"),
        "paper_hit_comparison.png": ("paper_hit_rate", "Related Work Comparison: Paper Hit Rate", "Paper Hit Rate"),
        "cloud_fetch_comparison.png": ("cloud_fetch_rate", "Related Work Comparison: Cloud Fetch Rate", "Cloud Fetch Rate"),
    }
    for filename, (metric, title, ylabel) in metrics.items():
        values = [mean_metric(rows_by_model[name], metric) for name in names]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(names, values, color=RELATED_BAR_COLORS[: len(names)])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    for metric, title, ylabel, filename in [
        ("reward", "Reward by Episode", "Reward", "reward_vs_episode.png"),
        ("paper_hit_rate", "Paper Hit Rate by Episode", "Paper Hit Rate", "paper_hit_vs_episode.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for name in names:
            rows = sorted(rows_by_model[name], key=lambda row: int(row["episode"]))
            ax.plot(
                [int(row["episode"]) for row in rows],
                [row[metric] for row in rows],
                marker="o",
                linewidth=2,
                label=name,
            )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)


def plot_main_run_png(out_dir: Path, rows_by_model: dict[str, list[dict[str, float]]]) -> None:
    names = ordered_png_names(list(rows_by_model.keys()))
    labels = names
    colors = [PNG_COLORS[name] for name in names]

    def save_baseline(filename: str) -> None:
        summary = [
            {
                "label": name,
                "reward": mean_metric(rows_by_model[name], "reward"),
                "local": mean_metric(rows_by_model[name], "local_hit_rate"),
                "paper": mean_metric(rows_by_model[name], "paper_hit_rate"),
                "cloud": mean_metric(rows_by_model[name], "cloud_fetch_rate"),
            }
            for name in names
        ]
        x = np.arange(len(summary))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        axes[0].bar(x, [row["reward"] for row in summary], color=colors)
        axes[0].set_xticks(x, labels, rotation=20, ha="right")
        axes[0].set_title("Reward Comparison")
        axes[0].grid(axis="y", alpha=0.25)

        width = 0.35
        axes[1].bar(x - width / 2, [row["local"] for row in summary], width=width, label="Local hit", color="#3a7d44")
        axes[1].bar(x + width / 2, [row["paper"] for row in summary], width=width, label="Paper hit", color="#f4a259")
        axes[1].set_xticks(x, labels, rotation=20, ha="right")
        axes[1].set_title("Hit Rate Comparison")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend()

        axes[2].bar(x, [row["cloud"] for row in summary], color="#457b9d")
        axes[2].set_xticks(x, labels, rotation=20, ha="right")
        axes[2].set_title("Cloud Fetch Rate")
        axes[2].grid(axis="y", alpha=0.25)

        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    def save_episode_curves(filename: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for name in names:
            rows = rows_by_model[name]
            episodes = [int(row["episode"]) for row in rows]
            reward = [row["reward"] for row in rows]
            paper = [row["paper_hit_rate"] for row in rows]
            axes[0].plot(episodes, reward, marker="o", label=name, color=PNG_COLORS[name])
            axes[1].plot(episodes, paper, marker="o", label=name, color=PNG_COLORS[name])
        axes[0].set_title("Reward by Episode")
        axes[0].set_xlabel("Episode")
        axes[0].grid(alpha=0.25)
        axes[1].set_title("Paper Hit by Episode")
        axes[1].set_xlabel("Episode")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    save_baseline("baseline_comparison.png")
    save_baseline("baseline_comparison_no_teacher.png")
    save_episode_curves("episode_curves.png")
    save_episode_curves("episode_curves_no_teacher.png")


def plot_final_no_teacher_png(out_dir: Path, rows_by_model: dict[str, list[dict[str, float]]]) -> None:
    clear_dir(out_dir)
    names = ordered_png_names(list(rows_by_model.keys()))

    def bar_metric(metric: str, title: str, ylabel: str, filename: str, lower_is_better: bool = False) -> None:
        values = [mean_metric(rows_by_model[name], metric) for name in names]
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        bars = ax.bar(labels := [name for name in names], values, color=[PNG_COLORS[name] for name in names], edgecolor="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
        ax.tick_params(axis="x", rotation=18)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        if lower_is_better:
            ax.text(0.98, 0.95, "Lower is better", transform=ax.transAxes, ha="right", va="top", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    def line_metric(metric: str, title: str, ylabel: str, filename: str, lower_is_better: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        for name in names:
            rows = rows_by_model[name]
            ax.plot(
                [int(row["episode"]) for row in rows],
                [row[metric] for row in rows],
                marker="o",
                linewidth=2.0,
                color=PNG_COLORS[name],
                label=name,
            )
        ax.set_title(title)
        ax.set_xlabel("Evaluation Episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
        ax.legend()
        if lower_is_better:
            ax.text(0.98, 0.95, "Lower is better", transform=ax.transAxes, ha="right", va="top", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    def stacked_service(filename: str) -> None:
        local = [mean_metric(rows_by_model[name], "local_hit_rate") for name in names]
        neighbor = [mean_metric(rows_by_model[name], "neighbor_fetch_rate") for name in names]
        cloud = [mean_metric(rows_by_model[name], "cloud_fetch_rate") for name in names]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(9.2, 5.0))
        ax.bar(x, local, color="#2a9d8f", edgecolor="black", linewidth=0.6, label="Local served")
        ax.bar(x, neighbor, bottom=local, color="#e9c46a", edgecolor="black", linewidth=0.6, label="Neighbor served")
        ax.bar(x, cloud, bottom=np.array(local) + np.array(neighbor), color="#adb5bd", edgecolor="black", linewidth=0.6, label="Cloud served")
        ax.set_xticks(x, names, rotation=18)
        ax.set_ylabel("Request Fraction")
        ax.set_title("Service Source Composition")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    def edge_gain(filename: str) -> None:
        temporal = mean_metric(rows_by_model["TemporalGraph"], "paper_hit_rate")
        gains = []
        labels = []
        for name in names:
            if name == "TemporalGraph":
                continue
            base = mean_metric(rows_by_model[name], "paper_hit_rate")
            gains.append(100.0 * (temporal - base) / max(base, 1e-8))
            labels.append(name)
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        bars = ax.bar(labels, gains, color="#d62828", edgecolor="black", linewidth=0.6)
        ax.set_title("TemporalGraph Paper-Hit Gain Over Baselines")
        ax.set_ylabel("Gain (%)")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
        for bar, val in zip(bars, gains):
            ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    def win_count(filename: str) -> None:
        counts = {"Reward wins": 0, "Paper-hit wins": 0, "Cloud-min wins": 0, "Neighbor-use wins": 0}
        episodes = len(rows_by_model["TemporalGraph"])
        for idx in range(episodes):
            reward_vals = {name: rows_by_model[name][idx]["reward"] for name in names}
            paper_vals = {name: rows_by_model[name][idx]["paper_hit_rate"] for name in names}
            cloud_vals = {name: rows_by_model[name][idx]["cloud_fetch_rate"] for name in names}
            neigh_vals = {name: rows_by_model[name][idx]["neighbor_fetch_rate"] for name in names}
            if reward_vals["TemporalGraph"] == max(reward_vals.values()):
                counts["Reward wins"] += 1
            if paper_vals["TemporalGraph"] == max(paper_vals.values()):
                counts["Paper-hit wins"] += 1
            if cloud_vals["TemporalGraph"] == min(cloud_vals.values()):
                counts["Cloud-min wins"] += 1
            if neigh_vals["TemporalGraph"] == max(neigh_vals.values()):
                counts["Neighbor-use wins"] += 1
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        labels = list(counts.keys())
        values = list(counts.values())
        bars = ax.bar(labels, values, color="#d62828", edgecolor="black", linewidth=0.6)
        ax.set_title("TemporalGraph Episode-Level Win Count")
        ax.set_ylabel("Winning Episodes")
        ax.set_ylim(0, episodes + 0.5)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val}/{episodes}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    bar_metric("reward", "Mean Reward Comparison", "Mean Episode Reward", "reward_mean_no_teacher.png")
    bar_metric("paper_hit_rate", "Mean Paper-Hit Comparison", "Mean Paper-Hit Rate", "paper_hit_mean_no_teacher.png")
    bar_metric("cloud_fetch_rate", "Mean Cloud-Fetch Comparison", "Mean Cloud-Fetch Rate", "cloud_fetch_mean_no_teacher.png", lower_is_better=True)
    bar_metric("neighbor_fetch_rate", "Mean Neighbor-Fetch Comparison", "Mean Neighbor-Fetch Rate", "neighbor_fetch_mean_no_teacher.png")
    line_metric("reward", "Reward Across Evaluation Episodes", "Episode Reward", "reward_vs_episode_no_teacher.png")
    line_metric("paper_hit_rate", "Paper-Hit Across Evaluation Episodes", "Paper-Hit Rate", "paper_hit_vs_episode_no_teacher.png")
    line_metric("cloud_fetch_rate", "Cloud-Fetch Across Evaluation Episodes", "Cloud-Fetch Rate", "cloud_fetch_vs_episode_no_teacher.png", lower_is_better=True)
    line_metric("neighbor_fetch_rate", "Neighbor-Fetch Across Evaluation Episodes", "Neighbor-Fetch Rate", "neighbor_fetch_vs_episode_no_teacher.png")
    stacked_service("service_source_composition_no_teacher.png")
    edge_gain("paper_hit_gain_no_teacher.png")
    win_count("temporalgraph_win_count_no_teacher.png")


def filtered_bundle_rows(source_root: Path) -> dict[str, list[dict[str, str]]]:
    src = source_root / "novel_comparison_bundle"
    rows = {
        "capacity": [row for row in load_csv(src / "capacity_sweep.csv") if row["model"] in PNG_ORDER],
        "sbs": [row for row in load_csv(src / "sbs_sweep.csv") if row["model"] in PNG_ORDER],
        "cost": [row for row in load_csv(src / "cost_summary.csv") if row["model"] in PNG_ORDER],
    }
    return rows


def plot_novel_comparison_png(out_dir: Path, source_root: Path) -> dict[str, list[dict[str, str]]]:
    clear_dir(out_dir)
    source_dir = source_root / "novel_comparison_bundle"
    filtered_rows = filtered_bundle_rows(source_root)
    write_csv(out_dir / "capacity_sweep.csv", list(filtered_rows["capacity"][0].keys()), filtered_rows["capacity"])
    write_csv(out_dir / "sbs_sweep.csv", list(filtered_rows["sbs"][0].keys()), filtered_rows["sbs"])
    write_csv(out_dir / "cost_summary.csv", list(filtered_rows["cost"][0].keys()), filtered_rows["cost"])

    for filename in [
        "random_trace.csv",
        "bsg_trace.csv",
        "c_epsilon_trace.csv",
        "maafdrl_trace.csv",
        "temporal_graph_trace.csv",
        "burst_random.csv",
        "burst_bsg.csv",
        "burst_c_epsilon.csv",
        "burst_maafdrl.csv",
        "burst_temporal_graph.csv",
    ]:
        shutil.copy2(source_dir / filename, out_dir / filename)

    def plot_lines(x_vals: list[int], series: dict[str, list[float]], title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        for name in ordered_png_names(list(series.keys())):
            ax.plot(x_vals, series[name], marker="o", linewidth=2.0, label=name, color=SHOWCASE_COLORS[name])
        ax.set_title(title)
        ax.set_xlabel(filename.replace(".png", "").split("_vs_")[-1].replace("_", " ").title())
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    def plot_cost_breakdown(cost_rows: list[dict[str, str]]) -> None:
        names = [row["model"] for row in cost_rows]
        local = [float(row["local"]) for row in cost_rows]
        neigh = [float(row["neighbor"]) for row in cost_rows]
        cloud = [float(row["cloud"]) for row in cost_rows]
        repl = [float(row["replace"]) for row in cost_rows]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(9.2, 5.0))
        ax.bar(x, local, label="Local cost", color="#2a9d8f", edgecolor="black", linewidth=0.6)
        ax.bar(x, neigh, bottom=local, label="Neighbor cost", color="#e9c46a", edgecolor="black", linewidth=0.6)
        ax.bar(x, cloud, bottom=np.array(local) + np.array(neigh), label="Cloud cost", color="#adb5bd", edgecolor="black", linewidth=0.6)
        ax.bar(x, repl, bottom=np.array(local) + np.array(neigh) + np.array(cloud), label="Replacement cost", color="#f28482", edgecolor="black", linewidth=0.6)
        ax.set_xticks(x, names, rotation=18)
        ax.set_ylabel("Mean Cost per Step")
        ax.set_title("Cost Breakdown")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "cost_breakdown.png", dpi=180)
        plt.close(fig)

    capacity_rows = filtered_rows["capacity"]
    capacities = sorted({int(float(row["cache_capacity"])) for row in capacity_rows})
    for metric, ylabel, filename in [
        ("paper_hit_mean", "Mean Paper-Hit Rate", "paper_hit_vs_cache_capacity.png"),
        ("reward_mean", "Mean Reward", "reward_vs_cache_capacity.png"),
    ]:
        series = {}
        for name in PNG_ORDER:
            series[name] = [float(next(row[metric] for row in capacity_rows if row["model"] == name and int(float(row["cache_capacity"])) == cap)) for cap in capacities]
        plot_lines(capacities, series, filename.replace(".png", "").replace("_", " ").title(), ylabel, filename)

    sbs_rows = filtered_rows["sbs"]
    sbs_counts = sorted({int(float(row["n_sbs"])) for row in sbs_rows})
    for metric, ylabel, filename in [
        ("paper_hit_mean", "Mean Paper-Hit Rate", "paper_hit_vs_n_sbs.png"),
        ("reward_mean", "Mean Reward", "reward_vs_n_sbs.png"),
    ]:
        series = {}
        for name in PNG_ORDER:
            series[name] = [float(next(row[metric] for row in sbs_rows if row["model"] == name and int(float(row["n_sbs"])) == n_sbs)) for n_sbs in sbs_counts]
        plot_lines(sbs_counts, series, filename.replace(".png", "").replace("_", " ").title(), ylabel, filename)

    plot_cost_breakdown(filtered_rows["cost"])

    burst_traces = {
        "Random": load_float_csv(out_dir / "burst_random.csv"),
        "BSG-like": load_float_csv(out_dir / "burst_bsg.csv"),
        "C-epsilon-greedy": load_float_csv(out_dir / "burst_c_epsilon.csv"),
        "MAAFDRL": load_float_csv(out_dir / "burst_maafdrl.csv"),
        "TemporalGraph": load_float_csv(out_dir / "burst_temporal_graph.csv"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for name in ordered_png_names(list(burst_traces.keys())):
        rows = burst_traces[name]
        steps = [int(row["step"]) for row in rows]
        axes[0].plot(steps, [row["burst_local_hit"] for row in rows], label=name, color=SHOWCASE_COLORS[name], linewidth=2.0)
        axes[1].plot(steps, [row["burst_edge_hit"] for row in rows], label=name, color=SHOWCASE_COLORS[name], linewidth=2.0)
    burst_window = (max(steps) // 3, 2 * max(steps) // 3)
    for ax, title, ylabel in [
        (axes[0], "Burst Local-Hit Adaptation", "Burst Local-Hit Rate"),
        (axes[1], "Burst Edge-Hit Adaptation", "Burst Edge-Hit Rate"),
    ]:
        ax.axvspan(burst_window[0], burst_window[1], color="#ffd166", alpha=0.18)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "burst_adaptation.png", dpi=180)
    plt.close(fig)

    traces = {
        "Random": load_float_csv(out_dir / "random_trace.csv"),
        "BSG-like": load_float_csv(out_dir / "bsg_trace.csv"),
        "C-epsilon-greedy": load_float_csv(out_dir / "c_epsilon_trace.csv"),
        "MAAFDRL": load_float_csv(out_dir / "maafdrl_trace.csv"),
        "TemporalGraph": load_float_csv(out_dir / "temporal_graph_trace.csv"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for name in ordered_png_names(list(traces.keys())):
        rows = traces[name]
        steps = np.arange(len(rows))
        axes[0].plot(steps, [row["cache_overlap"] for row in rows], label=name, color=SHOWCASE_COLORS[name], linewidth=2.0)
        axes[1].plot(steps, [row["cache_diversity"] for row in rows], label=name, color=SHOWCASE_COLORS[name], linewidth=2.0)
    axes[0].set_title("Neighbor Cache Overlap")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean Jaccard Overlap")
    axes[1].set_title("Neighbor Cache Diversity")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("1 - Mean Jaccard Overlap")
    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cache_overlap_diversity.png", dpi=180)
    plt.close(fig)
    return filtered_rows


def plot_showcase_png(out_dir: Path, source_root: Path, rows_by_model: dict[str, list[dict[str, float]]]) -> None:
    clear_dir(out_dir)
    summary = {name: summarize_rows(rows) for name, rows in rows_by_model.items()}
    names = ordered_png_names(list(summary.keys()))

    def plot_metric_bars(metric: str, ylabel: str, title: str, filename: str) -> None:
        values = [summary[name][metric] for name in names]
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        bars = ax.bar(names, values, color=[SHOWCASE_COLORS[name] for name in names], edgecolor="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
        ax.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    plot_metric_bars("reward_mean", "Mean Reward", "MovieLens-1M Run: Reward Comparison", "ml1m_run_reward_comparison.png")
    plot_metric_bars("local_hit_mean", "Mean Local Hit Rate", "MovieLens-1M Run: Local Hit Comparison", "ml1m_run_local_hit_comparison.png")
    plot_metric_bars("paper_hit_mean", "Mean Paper Hit Rate", "MovieLens-1M Run: Paper Hit Comparison", "ml1m_run_paper_hit_comparison.png")

    gains = []
    labels = []
    temporal = summary["TemporalGraph"]
    for key, label in [("reward_mean", "Reward"), ("local_hit_mean", "Local Hit"), ("paper_hit_mean", "Paper Hit")]:
        best_baseline = max(summary[name][key] for name in names if name != "TemporalGraph")
        gains.append(100.0 * (temporal[key] - best_baseline) / max(abs(best_baseline), 1e-8))
        labels.append(label)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bars = ax.bar(labels, gains, color="#d62828", edgecolor="black", linewidth=0.6)
    ax.set_title("TemporalGraph Gain Over Best Non-Temporal Baseline")
    ax.set_ylabel("Relative Gain (%)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
    for bar, value in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ml1m_run_relative_gain.png", dpi=180)
    plt.close(fig)

    imitation_rows = load_float_csv(source_root / "novel_realworld_main" / "policy_imitation.csv")
    epochs = [int(row["epoch"]) for row in imitation_rows]
    fig, ax1 = plt.subplots(figsize=(8.8, 4.8))
    ax1.plot(epochs, [row["loss"] for row in imitation_rows], color="#264653", linewidth=2.0, marker="o", label="Imitation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#264653")
    ax1.tick_params(axis="y", labelcolor="#264653")
    ax1.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["local_hit_rate"] for row in imitation_rows], color="#f4a261", linewidth=2.0, marker="s", label="Local hit")
    ax2.plot(epochs, [row["paper_hit_rate"] for row in imitation_rows], color="#d62828", linewidth=2.0, marker="^", label="Paper hit")
    ax2.set_ylabel("Hit Rate", color="#d62828")
    ax2.tick_params(axis="y", labelcolor="#d62828")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="center right")
    ax1.set_title("TemporalGraph Imitation Training")
    fig.tight_layout()
    fig.savefig(out_dir / "ml1m_run_imitation_training.png", dpi=180)
    plt.close(fig)

    eval_rows = rows_by_model["TemporalGraph"]
    episodes = [int(row["episode"]) for row in eval_rows]
    reward = [row["reward"] for row in eval_rows]
    local_hit = [row["local_hit_rate"] for row in eval_rows]
    paper_hit = [row["paper_hit_rate"] for row in eval_rows]
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
    fig.savefig(out_dir / "ml1m_run_eval_episodes.png", dpi=180)
    plt.close(fig)


def paper_bar_plot(data: dict[str, float], ylabel: str, out_path: Path, title: str, rotate: int = 18, fmt: str = ".3f") -> None:
    names = ordered_pdf_names(list(data.keys()))
    x = np.arange(len(names))
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for idx, name in enumerate(names):
            style = paper_scheme_style(name)
            ax.bar(idx, data[name], 0.62, color=style["color"], edgecolor="black", linewidth=0.65, hatch=style["hatch"], zorder=style["zorder"])
            ax.annotate(f"{data[name]:{fmt}}", xy=(idx, data[name]), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=rotate, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save_fig(fig, out_path)


def paper_line_plot(x_vals: list[float] | np.ndarray, series: dict[str, list[float]], xlabel: str, ylabel: str, out_path: Path, title: str, integer_x: bool = False, legend_loc: str = "best") -> None:
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        for name in ordered_pdf_names(list(series.keys())):
            style = paper_scheme_style(name)
            ax.plot(
                x_vals,
                series[name],
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=2.0,
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.45,
                label=name,
                zorder=style["zorder"],
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if integer_x:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(loc=legend_loc)
        save_fig(fig, out_path)


def paper_stacked_bar(names: list[str], stacks: list[tuple[str, list[float], str, str]], ylabel: str, out_path: Path, title: str) -> None:
    x = np.arange(len(names))
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        bottom = np.zeros((len(names),), dtype=np.float64)
        for label, values, color, hatch in stacks:
            ax.bar(x, values, 0.68, bottom=bottom, label=label, color=color, edgecolor="black", linewidth=0.6, hatch=hatch)
            bottom += np.asarray(values, dtype=np.float64)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")
        save_fig(fig, out_path)


def plot_related_work_pdf(out_dir: Path, rows_by_model: dict[str, list[dict[str, float]]]) -> None:
    clear_dir(out_dir)
    pdf_rows_by_model = {PNG_TO_PDF[name]: rows for name, rows in rows_by_model.items()}
    for metric, ylabel, filename in [
        ("reward", "Mean Reward", "reward_comparison.pdf"),
        ("local_hit_rate", "Local Hit Ratio", "local_hit_comparison.pdf"),
        ("paper_hit_rate", "Edge Hit Ratio", "edge_hit_comparison.pdf"),
        ("cloud_fetch_rate", "Cloud Fetch Rate", "cloud_fetch_comparison.pdf"),
    ]:
        paper_bar_plot(
            {name: mean_metric(rows, metric) for name, rows in pdf_rows_by_model.items()},
            ylabel,
            out_dir / filename,
            f"Comparison: {ylabel}",
        )
    for metric, ylabel, filename in [
        ("reward", "Reward", "reward_vs_episode.pdf"),
        ("paper_hit_rate", "Edge Hit Ratio", "edge_hit_vs_episode.pdf"),
    ]:
        paper_line_plot(
            [int(row["episode"]) for row in next(iter(pdf_rows_by_model.values()))],
            {name: [row[metric] for row in rows] for name, rows in pdf_rows_by_model.items()},
            "Episode",
            ylabel,
            out_dir / filename,
            f"{ylabel} vs Episode",
            integer_x=True,
        )


def plot_novel_comparison_pdf(out_dir: Path, filtered_rows: dict[str, list[dict[str, str]]], source_root: Path) -> None:
    clear_dir(out_dir)
    capacity_rows = filtered_rows["capacity"]
    sbs_rows = filtered_rows["sbs"]
    cost_rows = filtered_rows["cost"]
    capacities = sorted({int(float(row["cache_capacity"])) for row in capacity_rows})
    sbs_counts = sorted({int(float(row["n_sbs"])) for row in sbs_rows})
    pdf_model = {name: PNG_TO_PDF[name] for name in PNG_ORDER}

    for metric, ylabel, stem in [
        ("paper_hit_mean", "Edge Hit Ratio", "edge_hit_vs_cache_capacity.pdf"),
        ("local_hit_mean", "Local Hit Ratio", "local_hit_vs_cache_capacity.pdf"),
        ("reward_mean", "Mean Reward", "reward_vs_cache_capacity.pdf"),
    ]:
        series = {}
        for name in PNG_ORDER:
            series[pdf_model[name]] = [float(next(row[metric] for row in capacity_rows if row["model"] == name and int(float(row["cache_capacity"])) == cap)) for cap in capacities]
        paper_line_plot(capacities, series, "Cache Capacity", ylabel, out_dir / stem, f"{ylabel} vs Cache Capacity", integer_x=True)

    for metric, ylabel, stem in [
        ("paper_hit_mean", "Edge Hit Ratio", "edge_hit_vs_n_sbs.pdf"),
        ("local_hit_mean", "Local Hit Ratio", "local_hit_vs_n_sbs.pdf"),
        ("reward_mean", "Mean Reward", "reward_vs_n_sbs.pdf"),
    ]:
        series = {}
        for name in PNG_ORDER:
            series[pdf_model[name]] = [float(next(row[metric] for row in sbs_rows if row["model"] == name and int(float(row["n_sbs"])) == n_sbs)) for n_sbs in sbs_counts]
        paper_line_plot(sbs_counts, series, "Number of SBSs", ylabel, out_dir / stem, f"{ylabel} vs Number of SBSs", integer_x=True)

    ordered = ordered_pdf_names([PNG_TO_PDF[row["model"]] for row in cost_rows])
    cost_map = {PNG_TO_PDF[row["model"]]: row for row in cost_rows}
    paper_stacked_bar(
        ordered,
        [
            ("Local cost", [float(cost_map[name]["local"]) for name in ordered], "#2a9d8f", "///"),
            ("Neighbor cost", [float(cost_map[name]["neighbor"]) for name in ordered], "#e9c46a", "\\\\\\"),
            ("Cloud cost", [float(cost_map[name]["cloud"]) for name in ordered], "#adb5bd", "xxx"),
            ("Replacement cost", [float(cost_map[name]["replace"]) for name in ordered], "#f28482", "..."),
        ],
        "Mean Cost per Step",
        out_dir / "cost_breakdown.pdf",
        "Cost Breakdown",
    )

    traces = {
        "Random": load_float_csv(source_root / "novel_comparison_bundle" / "random_trace.csv"),
        "BSG": load_float_csv(source_root / "novel_comparison_bundle" / "bsg_trace.csv"),
        "C-\u03b5-greedy": load_float_csv(source_root / "novel_comparison_bundle" / "c_epsilon_trace.csv"),
        "MAAFDRL": load_float_csv(source_root / "novel_comparison_bundle" / "maafdrl_trace.csv"),
        "TemporalGraph": load_float_csv(source_root / "novel_comparison_bundle" / "temporal_graph_trace.csv"),
    }
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
        for name in ordered_pdf_names(list(traces.keys())):
            rows = traces[name]
            steps = np.arange(len(rows))
            style = paper_scheme_style(name)
            for ax, metric, title, ylabel in [
                (axes[0], "cache_overlap", "Neighbor Cache Overlap", "Mean Jaccard Overlap"),
                (axes[1], "cache_diversity", "Neighbor Cache Diversity", "1 - Mean Jaccard Overlap"),
            ]:
                ax.plot(
                    steps,
                    [row[metric] for row in rows],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    color=style["color"],
                    linewidth=1.9,
                    markersize=4.2,
                    markeredgecolor="black",
                    markeredgewidth=0.35,
                    label=name,
                    zorder=style["zorder"],
                    markevery=max(1, len(steps) // 10),
                )
                ax.set_title(title)
                ax.set_xlabel("Step")
                ax.set_ylabel(ylabel)
        for ax in axes:
            ax.legend(loc="best", fontsize=8)
        save_fig(fig, out_dir / "cache_overlap_diversity.pdf")

    burst_traces = {
        "Random": load_float_csv(source_root / "novel_comparison_bundle" / "burst_random.csv"),
        "BSG": load_float_csv(source_root / "novel_comparison_bundle" / "burst_bsg.csv"),
        "C-\u03b5-greedy": load_float_csv(source_root / "novel_comparison_bundle" / "burst_c_epsilon.csv"),
        "MAAFDRL": load_float_csv(source_root / "novel_comparison_bundle" / "burst_maafdrl.csv"),
        "TemporalGraph": load_float_csv(source_root / "novel_comparison_bundle" / "burst_temporal_graph.csv"),
    }
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
        max_step = 0
        for name in ordered_pdf_names(list(burst_traces.keys())):
            rows = burst_traces[name]
            steps = [int(row["step"]) for row in rows]
            max_step = max(max_step, max(steps))
            style = paper_scheme_style(name)
            for ax, metric, title, ylabel in [
                (axes[0], "burst_local_hit", "Burst Local-Hit Adaptation", "Burst Local Hit Ratio"),
                (axes[1], "burst_edge_hit", "Burst Edge-Hit Adaptation", "Burst Edge Hit Ratio"),
            ]:
                ax.plot(
                    steps,
                    [row[metric] for row in rows],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    color=style["color"],
                    linewidth=1.9,
                    markersize=4.2,
                    markeredgecolor="black",
                    markeredgewidth=0.35,
                    label=name,
                    zorder=style["zorder"],
                    markevery=max(1, len(steps) // 10),
                )
                ax.set_title(title)
                ax.set_xlabel("Step")
                ax.set_ylabel(ylabel)
        burst_start = max_step // 3
        burst_end = 2 * max_step // 3
        for ax in axes:
            ax.axvspan(burst_start, burst_end, color="#ffd166", alpha=0.16)
            ax.legend(loc="best", fontsize=8)
        save_fig(fig, out_dir / "burst_adaptation.pdf")


def plot_final_no_teacher_pdf(out_dir: Path, rows_by_model: dict[str, list[dict[str, float]]]) -> None:
    clear_dir(out_dir)
    pdf_rows_by_model = {PNG_TO_PDF[name]: rows for name, rows in rows_by_model.items()}
    names = ordered_pdf_names(list(pdf_rows_by_model.keys()))

    for metric, ylabel, stem in [
        ("reward", "Mean Episode Reward", "reward_mean_no_teacher.pdf"),
        ("paper_hit_rate", "Mean Edge Hit Ratio", "edge_hit_mean_no_teacher.pdf"),
        ("cloud_fetch_rate", "Mean Cloud Fetch Rate", "cloud_fetch_mean_no_teacher.pdf"),
        ("neighbor_fetch_rate", "Mean Neighbor Fetch Rate", "neighbor_fetch_mean_no_teacher.pdf"),
    ]:
        paper_bar_plot(
            {name: mean_metric(pdf_rows_by_model[name], metric) for name in names},
            ylabel,
            out_dir / stem,
            f"{ylabel} Comparison",
        )

    episodes = [int(row["episode"]) for row in next(iter(pdf_rows_by_model.values()))]
    for metric, ylabel, stem in [
        ("reward", "Episode Reward", "reward_vs_episode_no_teacher.pdf"),
        ("paper_hit_rate", "Edge Hit Ratio", "edge_hit_vs_episode_no_teacher.pdf"),
        ("cloud_fetch_rate", "Cloud Fetch Rate", "cloud_fetch_vs_episode_no_teacher.pdf"),
        ("neighbor_fetch_rate", "Neighbor Fetch Rate", "neighbor_fetch_vs_episode_no_teacher.pdf"),
    ]:
        paper_line_plot(
            episodes,
            {name: [row[metric] for row in rows] for name, rows in pdf_rows_by_model.items()},
            "Evaluation Episode",
            ylabel,
            out_dir / stem,
            f"{ylabel} Across Evaluation Episodes",
            integer_x=True,
        )

    paper_stacked_bar(
        names,
        [
            ("Local served", [mean_metric(pdf_rows_by_model[name], "local_hit_rate") for name in names], "#2a9d8f", "///"),
            ("Neighbor served", [mean_metric(pdf_rows_by_model[name], "neighbor_fetch_rate") for name in names], "#e9c46a", "\\\\\\"),
            ("Cloud served", [mean_metric(pdf_rows_by_model[name], "cloud_fetch_rate") for name in names], "#adb5bd", "xxx"),
        ],
        "Request Fraction",
        out_dir / "service_source_composition_no_teacher.pdf",
        "Service Source Composition",
    )

    tg_edge = mean_metric(pdf_rows_by_model["TemporalGraph"], "paper_hit_rate")
    gains = {}
    for name in names:
        if name == "TemporalGraph":
            continue
        base = mean_metric(pdf_rows_by_model[name], "paper_hit_rate")
        gains[name] = 100.0 * (tg_edge - base) / max(base, 1e-8)
    paper_bar_plot(gains, "Gain (%)", out_dir / "edge_hit_gain_no_teacher.pdf", "TemporalGraph Edge-Hit Gain Over Baselines", fmt=".1f")

    episodes_n = len(next(iter(pdf_rows_by_model.values())))
    win_counts = {"Reward wins": 0, "Edge-hit wins": 0, "Cloud-min wins": 0, "Neighbor-use wins": 0}
    for idx in range(episodes_n):
        reward_vals = {name: rows[idx]["reward"] for name, rows in pdf_rows_by_model.items()}
        edge_vals = {name: rows[idx]["paper_hit_rate"] for name, rows in pdf_rows_by_model.items()}
        cloud_vals = {name: rows[idx]["cloud_fetch_rate"] for name, rows in pdf_rows_by_model.items()}
        neigh_vals = {name: rows[idx]["neighbor_fetch_rate"] for name, rows in pdf_rows_by_model.items()}
        if reward_vals["TemporalGraph"] == max(reward_vals.values()):
            win_counts["Reward wins"] += 1
        if edge_vals["TemporalGraph"] == max(edge_vals.values()):
            win_counts["Edge-hit wins"] += 1
        if cloud_vals["TemporalGraph"] == min(cloud_vals.values()):
            win_counts["Cloud-min wins"] += 1
        if neigh_vals["TemporalGraph"] == max(neigh_vals.values()):
            win_counts["Neighbor-use wins"] += 1
    paper_bar_plot(win_counts, "Winning Episodes", out_dir / "temporalgraph_win_count_no_teacher.pdf", "TemporalGraph Episode-Level Win Count", rotate=14, fmt=".0f")


def plot_showcase_pdf(out_dir: Path, source_root: Path, rows_by_model: dict[str, list[dict[str, float]]]) -> None:
    clear_dir(out_dir)
    summary = {PNG_TO_PDF[name]: summarize_rows(rows) for name, rows in rows_by_model.items()}
    comparison = {name: summary[name] for name in ordered_pdf_names(list(summary.keys()))}
    for metric, ylabel, stem in [
        ("reward_mean", "Mean Reward", "ml1m_run_reward_comparison.pdf"),
        ("local_hit_mean", "Mean Local Hit Ratio", "ml1m_run_local_hit_comparison.pdf"),
        ("paper_hit_mean", "Mean Edge Hit Ratio", "ml1m_run_edge_hit_comparison.pdf"),
    ]:
        paper_bar_plot({name: values[metric] for name, values in comparison.items()}, ylabel, out_dir / stem, f"MovieLens-1M Run: {ylabel}")

    tg = comparison["TemporalGraph"]
    gains = {}
    for metric, label in [("reward_mean", "Reward"), ("local_hit_mean", "Local Hit"), ("paper_hit_mean", "Edge Hit")]:
        best_baseline = max(values[metric] for name, values in comparison.items() if name != "TemporalGraph")
        gains[label] = 100.0 * (tg[metric] - best_baseline) / max(abs(best_baseline), 1e-8)
    paper_bar_plot(gains, "Relative Gain (%)", out_dir / "ml1m_run_relative_gain.pdf", "TemporalGraph Gain Over Best Baseline", rotate=0, fmt=".1f")

    imitation_rows = load_float_csv(source_root / "novel_realworld_main" / "policy_imitation.csv")
    epochs = [int(row["epoch"]) for row in imitation_rows]
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
        axes[0].plot(epochs, [row["loss"] for row in imitation_rows], color="#264653", marker="o", linewidth=2.0, markeredgecolor="black", markeredgewidth=0.4)
        axes[0].set_title("TemporalGraph Imitation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[1].plot(epochs, [row["local_hit_rate"] for row in imitation_rows], color="#457b9d", marker="s", linestyle="--", linewidth=2.0, markeredgecolor="black", markeredgewidth=0.4, label="Local Hit Ratio")
        axes[1].plot(epochs, [row["paper_hit_rate"] for row in imitation_rows], color="#2a9d8f", marker="^", linestyle="-.", linewidth=2.0, markeredgecolor="black", markeredgewidth=0.4, label="Edge Hit Ratio")
        axes[1].set_title("TemporalGraph Hit Ratios During Imitation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Hit Ratio")
        axes[1].legend(loc="best")
        save_fig(fig, out_dir / "ml1m_run_imitation_training.pdf")

    eval_rows = rows_by_model["TemporalGraph"]
    episodes = [int(row["episode"]) for row in eval_rows]
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
        axes[0].plot(episodes, [row["reward"] for row in eval_rows], color="#d62828", marker="D", linewidth=2.0, markeredgecolor="black", markeredgewidth=0.4)
        axes[0].set_title("TemporalGraph Reward Across Evaluation Episodes")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[1].plot(episodes, [row["local_hit_rate"] for row in eval_rows], color="#457b9d", marker="s", linestyle="--", linewidth=2.0, markeredgecolor="black", markeredgewidth=0.4, label="Local Hit Ratio")
        axes[1].plot(episodes, [row["paper_hit_rate"] for row in eval_rows], color="#2a9d8f", marker="^", linestyle="-.", linewidth=2.0, markeredgecolor="black", markeredgewidth=0.4, label="Edge Hit Ratio")
        axes[1].set_title("TemporalGraph Hit Ratios Across Evaluation Episodes")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Hit Ratio")
        axes[1].legend(loc="best")
        save_fig(fig, out_dir / "ml1m_run_eval_episodes.pdf")


def plot_consolidated_summary_pdf(out_path: Path, rows_by_model: dict[str, list[dict[str, float]]]) -> None:
    metrics = {}
    for name, rows in rows_by_model.items():
        metrics[PNG_TO_PDF[name]] = {
            "reward": mean_metric(rows, "reward"),
            "local_hit": mean_metric(rows, "local_hit_rate"),
            "edge_hit": mean_metric(rows, "paper_hit_rate"),
            "cloud_fetch": mean_metric(rows, "cloud_fetch_rate"),
        }
    ordered = ordered_pdf_names(list(metrics.keys()))
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.6))
        for ax, (metric, ylabel) in zip(
            axes,
            [
                ("reward", "Mean Reward"),
                ("local_hit", "Local Hit Ratio"),
                ("edge_hit", "Edge Hit Ratio"),
                ("cloud_fetch", "Cloud Fetch Rate"),
            ],
        ):
            x = np.arange(len(ordered))
            for idx, name in enumerate(ordered):
                style = paper_scheme_style(name)
                value = metrics[name][metric]
                ax.bar(idx, value, 0.62, color=style["color"], edgecolor="black", linewidth=0.55, hatch=style["hatch"])
                ax.annotate(f"{value:.3f}", xy=(idx, value), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(ordered, rotation=28, ha="right", fontsize=8)
            ax.set_title(ylabel)
            ax.set_ylabel(ylabel)
        save_fig(fig, out_path)


def write_benchmark_report(output_root: Path, rows_by_model: dict[str, list[dict[str, float]]]) -> None:
    summary = {name: summarize_rows(rows) for name, rows in rows_by_model.items()}
    reward_best = max(summary, key=lambda name: summary[name]["reward_mean"])
    local_best = max(summary, key=lambda name: summary[name]["local_hit_mean"])
    paper_best = max(summary, key=lambda name: summary[name]["paper_hit_mean"])
    cloud_best = min(summary, key=lambda name: summary[name]["cloud_fetch_mean"])
    ours = summary["TemporalGraph"]
    lines = [
        f"Related-work summary file: {output_root / 'related_work_compare' / 'summary.csv'}",
        f"Our reward_mean={ours['reward_mean']:.6f} | best={reward_best}",
        f"Our local_hit_mean={ours['local_hit_mean']:.6f} | best={local_best}",
        f"Our paper_hit_mean={ours['paper_hit_mean']:.6f} | best={paper_best}",
        f"Our cloud_fetch_mean={ours['cloud_fetch_mean']:.6f} | best={cloud_best}",
        f"Ours best on all four summary metrics: {'yes' if all(name == 'TemporalGraph' for name in [reward_best, local_best, paper_best, cloud_best]) else 'no'}",
    ]
    write_text(output_root / "benchmark_report.txt", "\n".join(lines) + "\n")


def write_manifest(output_root: Path) -> None:
    root = Path.cwd()
    paths = []
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
            paths.append(str(rel))
        except ValueError:
            paths.append(str(path))
    paths.sort()
    write_text(output_root / "manifest.txt", "\n".join(paths) + ("\n" if paths else ""))


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        raise SystemExit("--output-root must differ from --source-root")
    if not source_root.exists():
        raise SystemExit(f"Source root does not exist: {source_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, output_root, dirs_exist_ok=True)

    rows_by_model = build_rows_by_model(source_root)
    plot_related_work_png(output_root / "related_work_compare", rows_by_model)
    plot_main_run_png(output_root / "novel_realworld_main", rows_by_model)
    plot_final_no_teacher_png(output_root / "final_no_teacher_bundle", rows_by_model)
    filtered_rows = plot_novel_comparison_png(output_root / "novel_comparison_bundle", source_root)
    plot_showcase_png(output_root / "temporalgraph_showcase", source_root, rows_by_model)

    paper_root = output_root / "paper_ready_bundle"
    plot_related_work_pdf(paper_root / "related_work_compare", rows_by_model)
    plot_novel_comparison_pdf(paper_root / "novel_comparison_bundle", filtered_rows, source_root)
    plot_final_no_teacher_pdf(paper_root / "final_no_teacher_bundle", rows_by_model)
    plot_showcase_pdf(paper_root / "temporalgraph_showcase", source_root, rows_by_model)
    plot_consolidated_summary_pdf(paper_root / "consolidated_summary.pdf", rows_by_model)

    write_benchmark_report(output_root, rows_by_model)
    write_manifest(output_root)
    print(f"Saved filtered benchmark bundle to: {output_root}")


if __name__ == "__main__":
    main()
