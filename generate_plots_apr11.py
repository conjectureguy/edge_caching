#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_mpl_dir = Path("outputs/.mplconfig")
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from movie_edge_sim.simulation import SimulationConfig, freeze_sbs_positions, run_simulation
from plot_clustered_latency_study import evaluate_pair
from plot_static_vs_dynamic_bundle import _step_metrics as compute_static_dynamic_metrics


plt.rcParams.update(
    {
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
)


SCHEME_STYLE: dict[str, dict[str, Any]] = {
    "TemporalGraph": {"color": "#d62828", "marker": "D", "linestyle": "-", "hatch": "///", "zorder": 10},
    "AWFDRL": {"color": "#1d3557", "marker": "s", "linestyle": "--", "hatch": "\\\\\\", "zorder": 7},
    "MAAFDRL": {"color": "#2a9d8f", "marker": "^", "linestyle": "-.", "hatch": "xxx", "zorder": 7},
    "C-ε-greedy": {"color": "#e76f51", "marker": "P", "linestyle": (0, (5, 1)), "hatch": "---", "zorder": 5},
    "BSG": {"color": "#8d99ae", "marker": "o", "linestyle": (0, (3, 1, 1, 1)), "hatch": "+++", "zorder": 4},
    "Random": {"color": "#6c757d", "marker": "v", "linestyle": ":", "hatch": "...", "zorder": 3},
    "Teacher": {"color": "#f4a261", "marker": "X", "linestyle": (0, (1, 1)), "hatch": "ooo", "zorder": 6},
}

PAIR_STYLE = {
    "UE moving, SBS fixed": {"color": "#4c78a8", "marker": "o", "linestyle": "--"},
    "UE moving, SBS moving": {"color": "#d62828", "marker": "D", "linestyle": "-"},
}

BAR_ORDER = ["TemporalGraph", "AWFDRL", "MAAFDRL", "C-ε-greedy", "BSG", "Random", "Teacher"]
LINE_ORDER = BAR_ORDER[:]

_NAME_MAP = {
    "Our-TemporalGraph": "TemporalGraph",
    "TemporalGraph": "TemporalGraph",
    "Paper2-AWFDRL-like": "AWFDRL",
    "Paper3-MAAFDRL-like": "MAAFDRL",
    "Paper4-DTS-DDPG-like": None,
    "DTS-DDPG": None,
    "AWFDRL-like": "AWFDRL",
    "MAAFDRL-like": "MAAFDRL",
    "BSG-like": "BSG",
    "C-epsilon-greedy": "C-ε-greedy",
    "Random": "Random",
    "Teacher": "Teacher",
}

SUMMARY_RE = re.compile(r"^(?P<name>[^:]+): (?P<body>.+)$")


def map_name(raw: str) -> str | None:
    if raw in _NAME_MAP:
        return _NAME_MAP[raw]
    stripped = raw.replace("-like", "")
    if stripped in _NAME_MAP:
        return _NAME_MAP[stripped]
    return raw


def scheme_style(name: str) -> dict[str, Any]:
    return SCHEME_STYLE.get(name, {"color": "#333333", "marker": "o", "linestyle": "-", "hatch": "", "zorder": 1})


def ordered_names(names: list[str], include_teacher: bool = True) -> list[str]:
    order = BAR_ORDER if include_teacher else [name for name in BAR_ORDER if name != "Teacher"]
    ordered = [name for name in order if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def ext_path(directory: Path, stem: str, fmt: str) -> Path:
    return directory / f"{stem}.{fmt}"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def load_float_csv(path: Path) -> list[dict[str, float]]:
    rows = []
    for row in load_csv(path):
        rows.append({k: float(v) for k, v in row.items()})
    return rows


def parse_summary_txt(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        match = SUMMARY_RE.match(line.strip())
        if not match:
            continue
        name = map_name(match.group("name").strip())
        if name is None:
            continue
        metrics: dict[str, float] = {}
        for part in match.group("body").split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
        if metrics:
            out[name] = metrics
    return out


def save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path}")


def line_plot(
    x_vals: list[float] | np.ndarray,
    series: dict[str, list[float] | np.ndarray],
    xlabel: str,
    ylabel: str,
    out_path: Path,
    title: str,
    integer_x: bool = False,
    legend_loc: str = "best",
    markevery: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for name in ordered_names(list(series.keys()), include_teacher="Teacher" in series):
        values = series[name]
        style = scheme_style(name)
        ax.plot(
            x_vals,
            values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=2.0,
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.45,
            label=name,
            zorder=style["zorder"],
            markevery=markevery,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if integer_x:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc=legend_loc)
    save_fig(fig, out_path)


def pair_line_plot(
    x_vals: list[float] | np.ndarray,
    fixed_vals: list[float] | np.ndarray,
    moving_vals: list[float] | np.ndarray,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    title: str,
    lower_is_better: bool | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for label, vals in [
        ("UE moving, SBS fixed", fixed_vals),
        ("UE moving, SBS moving", moving_vals),
    ]:
        style = PAIR_STYLE[label]
        ax.plot(
            x_vals,
            vals,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=2.0,
            markersize=5.5,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    if lower_is_better is not None:
        note = "Lower is better" if lower_is_better else "Higher is better"
        ax.text(0.98, 0.96, note, transform=ax.transAxes, ha="right", va="top", fontsize=9)
    save_fig(fig, out_path)


def bar_plot(
    data: dict[str, float],
    ylabel: str,
    out_path: Path,
    title: str,
    rotate: int = 18,
    include_teacher: bool = True,
    fmt: str = ".3f",
) -> None:
    names = ordered_names(list(data.keys()), include_teacher=include_teacher)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for idx, name in enumerate(names):
        style = scheme_style(name)
        ax.bar(
            x[idx],
            data[name],
            0.62,
            color=style["color"],
            edgecolor="black",
            linewidth=0.65,
            hatch=style["hatch"],
            zorder=style["zorder"],
        )
        ax.annotate(
            f"{data[name]:{fmt}}",
            xy=(x[idx], data[name]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=rotate, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_fig(fig, out_path)


def stacked_bar_plot(
    names: list[str],
    stacks: list[tuple[str, list[float], str, str]],
    ylabel: str,
    out_path: Path,
    title: str,
) -> None:
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bottom = np.zeros((len(names),), dtype=np.float64)
    for label, values, color, hatch in stacks:
        ax.bar(
            x,
            values,
            0.68,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch,
        )
        bottom += np.asarray(values, dtype=np.float64)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    save_fig(fig, out_path)


def load_episode_series(main_dir: Path) -> dict[str, list[dict[str, float]]]:
    files = [
        ("random_eval.csv", "Random"),
        ("bsg_like_eval.csv", "BSG"),
        ("c_epsilon_greedy_eval.csv", "C-ε-greedy"),
        ("teacher_eval.csv", "Teacher"),
        ("temporal_graph_eval.csv", "TemporalGraph"),
    ]
    out: dict[str, list[dict[str, float]]] = {}
    for filename, scheme in files:
        path = main_dir / filename
        if path.exists():
            out[scheme] = load_float_csv(path)
    return out


def bundle_has_extended_baselines(bundle_dir: Path) -> bool:
    if not bundle_dir.exists():
        return False
    required_files = [
        "capacity_sweep.csv",
        "sbs_sweep.csv",
        "cost_summary.csv",
        "awfdrl_trace.csv",
        "maafdrl_trace.csv",
        "burst_awfdrl.csv",
        "burst_maafdrl.csv",
    ]
    for filename in required_files:
        if not (bundle_dir / filename).exists():
            return False
    required = {"AWFDRL", "MAAFDRL"}
    for filename in ["capacity_sweep.csv", "sbs_sweep.csv", "cost_summary.csv"]:
        models = {row["model"] for row in load_csv(bundle_dir / filename)}
        if not required.issubset(models):
            return False
    return True


def ensure_extended_bundle(data_dir: Path, output_dir: Path, run_dir: Path, python_bin: str) -> Path:
    source_bundle = data_dir / "novel_comparison_bundle"
    if bundle_has_extended_baselines(source_bundle):
        print("Novel comparison bundle already includes AWFDRL and MAAFDRL.")
        return source_bundle

    generated_bundle = output_dir / "_generated_data" / "novel_comparison_bundle"
    if bundle_has_extended_baselines(generated_bundle):
        print(f"Using cached augmented bundle at {generated_bundle}")
        return generated_bundle

    print("Regenerating novel comparison bundle with AWFDRL and MAAFDRL coverage...")
    generated_bundle.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        "plot_novel_comparison_bundle.py",
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(generated_bundle),
        "--eval-episodes",
        "3",
        "--episode-len",
        "30",
        "--n-ues",
        "220",
        "--cache-capacities",
        "10",
        "20",
        "30",
        "--sbs-list",
        "8",
        "12",
        "16",
    ]
    subprocess.run(cmd, check=True)
    return generated_bundle


def plot_related_work_bundle(data_dir: Path, out_root: Path, fmt: str) -> None:
    print("\n[1/8] related_work_compare")
    out_dir = out_root / "related_work_compare"
    summary_csv = data_dir / "related_work_compare" / "summary.csv"
    episode_csv = data_dir / "related_work_compare" / "episode_metrics.csv"
    main_dir = data_dir / "novel_realworld_main"
    if not summary_csv.exists():
        print(f"  ⚠ Missing {summary_csv}")
        return

    scheme_data: dict[str, dict[str, float]] = {}
    for row in load_csv(summary_csv):
        name = map_name(row["scheme"])
        if name is None:
            continue
        scheme_data[name] = {
            "reward_mean": float(row["reward_mean"]),
            "local_hit_mean": float(row["local_hit_mean"]),
            "edge_hit_mean": float(row["paper_hit_mean"]),
            "cloud_fetch_mean": float(row["cloud_fetch_mean"]),
        }

    for name, metrics in parse_summary_txt(main_dir / "summary.txt").items():
        if name in scheme_data:
            continue
        scheme_data[name] = {
            "reward_mean": metrics.get("reward_mean", 0.0),
            "local_hit_mean": metrics.get("local_hit_mean", 0.0),
            "edge_hit_mean": metrics.get("paper_hit_mean", 0.0),
            "cloud_fetch_mean": 1.0 - metrics.get("paper_hit_mean", 0.0),
        }

    scheme_data.pop("Teacher", None)

    for metric, ylabel, stem in [
        ("reward_mean", "Mean Reward", "reward_comparison"),
        ("local_hit_mean", "Local Hit Ratio", "local_hit_comparison"),
        ("edge_hit_mean", "Edge Hit Ratio", "edge_hit_comparison"),
        ("cloud_fetch_mean", "Cloud Fetch Rate", "cloud_fetch_comparison"),
    ]:
        bar_plot(
            {name: values[metric] for name, values in scheme_data.items()},
            ylabel,
            ext_path(out_dir, stem, fmt),
            f"Comparison: {ylabel}",
            include_teacher=False,
        )

    episode_data: dict[str, list[dict[str, float]]] = {}
    if episode_csv.exists():
        for row in load_csv(episode_csv):
            name = map_name(row["scheme"])
            if name is None:
                continue
            episode_data.setdefault(name, []).append(
                {
                    "episode": float(row["episode"]),
                    "reward": float(row["reward"]),
                    "edge_hit_rate": float(row["paper_hit_rate"]),
                }
            )
    for filename, name in [
        ("random_eval.csv", "Random"),
        ("bsg_like_eval.csv", "BSG"),
        ("c_epsilon_greedy_eval.csv", "C-ε-greedy"),
        ("temporal_graph_eval.csv", "TemporalGraph"),
    ]:
        path = main_dir / filename
        if path.exists() and name not in episode_data:
            episode_data[name] = [
                {
                    "episode": row["episode"],
                    "reward": row["reward"],
                    "edge_hit_rate": row["paper_hit_rate"],
                }
                for row in load_float_csv(path)
            ]

    for metric, ylabel, stem in [
        ("reward", "Reward", "reward_vs_episode"),
        ("edge_hit_rate", "Edge Hit Ratio", "edge_hit_vs_episode"),
    ]:
        if not episode_data:
            continue
        filtered = {name: rows for name, rows in episode_data.items() if name != "Teacher"}
        if not filtered:
            continue
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        for name in ordered_names(list(filtered.keys()), include_teacher=False):
            rows = sorted(filtered[name], key=lambda row: row["episode"])
            style = scheme_style(name)
            ax.plot(
                [row["episode"] for row in rows],
                [row[metric] for row in rows],
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
        ax.set_title(f"{ylabel} vs Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(loc="best")
        save_fig(fig, ext_path(out_dir, stem, fmt))


def plot_novel_comparison_bundle(bundle_dir: Path, out_root: Path, fmt: str) -> None:
    print("\n[2/8] novel_comparison_bundle")
    out_dir = out_root / "novel_comparison_bundle"
    capacity_rows = load_csv(bundle_dir / "capacity_sweep.csv")
    sbs_rows = load_csv(bundle_dir / "sbs_sweep.csv")
    cost_rows = load_csv(bundle_dir / "cost_summary.csv")

    capacities = sorted({int(float(row["cache_capacity"])) for row in capacity_rows})
    sbs_counts = sorted({int(float(row["n_sbs"])) for row in sbs_rows})

    for metric, ylabel, stem in [
        ("paper_hit_mean", "Edge Hit Ratio", "edge_hit_vs_cache_capacity"),
        ("local_hit_mean", "Local Hit Ratio", "local_hit_vs_cache_capacity"),
        ("reward_mean", "Mean Reward", "reward_vs_cache_capacity"),
    ]:
        series: dict[str, list[float]] = {}
        for row in capacity_rows:
            name = map_name(row["model"])
            if name is None:
                continue
            series.setdefault(name, []).append(float(row[metric]))
        line_plot(
            capacities,
            series,
            "Cache Capacity",
            ylabel,
            ext_path(out_dir, stem, fmt),
            f"{ylabel} vs Cache Capacity",
            integer_x=True,
        )

    for metric, ylabel, stem in [
        ("paper_hit_mean", "Edge Hit Ratio", "edge_hit_vs_n_sbs"),
        ("local_hit_mean", "Local Hit Ratio", "local_hit_vs_n_sbs"),
        ("reward_mean", "Mean Reward", "reward_vs_n_sbs"),
    ]:
        series: dict[str, list[float]] = {}
        for row in sbs_rows:
            name = map_name(row["model"])
            if name is None:
                continue
            series.setdefault(name, []).append(float(row[metric]))
        line_plot(
            sbs_counts,
            series,
            "Number of SBSs",
            ylabel,
            ext_path(out_dir, stem, fmt),
            f"{ylabel} vs Number of SBSs",
            integer_x=True,
        )

    ordered = ordered_names([map_name(row["model"]) for row in cost_rows if map_name(row["model"]) is not None], include_teacher=False)
    cost_map = {map_name(row["model"]): row for row in cost_rows if map_name(row["model"]) is not None}
    stacked_bar_plot(
        ordered,
        [
            ("Local cost", [float(cost_map[name]["local"]) for name in ordered], "#2a9d8f", "///"),
            ("Neighbor cost", [float(cost_map[name]["neighbor"]) for name in ordered], "#e9c46a", "\\\\\\"),
            ("Cloud cost", [float(cost_map[name]["cloud"]) for name in ordered], "#adb5bd", "xxx"),
            ("Replacement cost", [float(cost_map[name]["replace"]) for name in ordered], "#f28482", "..."),
        ],
        "Mean Cost per Step",
        ext_path(out_dir, "cost_breakdown", fmt),
        "Cost Breakdown",
    )

    trace_files = {
        "Random": "random_trace.csv",
        "BSG": "bsg_trace.csv",
        "C-ε-greedy": "c_epsilon_trace.csv",
        "AWFDRL": "awfdrl_trace.csv",
        "MAAFDRL": "maafdrl_trace.csv",
        "TemporalGraph": "temporal_graph_trace.csv",
    }
    traces = {
        scheme: load_float_csv(bundle_dir / filename)
        for scheme, filename in trace_files.items()
        if (bundle_dir / filename).exists()
    }
    if traces:
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
        for name in ordered_names(list(traces.keys()), include_teacher=False):
            rows = traces[name]
            steps = np.arange(len(rows))
            style = scheme_style(name)
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
        save_fig(fig, ext_path(out_dir, "cache_overlap_diversity", fmt))

    burst_files = {
        "Random": "burst_random.csv",
        "BSG": "burst_bsg.csv",
        "C-ε-greedy": "burst_c_epsilon.csv",
        "AWFDRL": "burst_awfdrl.csv",
        "MAAFDRL": "burst_maafdrl.csv",
        "TemporalGraph": "burst_temporal_graph.csv",
    }
    burst_traces = {
        scheme: load_float_csv(bundle_dir / filename)
        for scheme, filename in burst_files.items()
        if (bundle_dir / filename).exists()
    }
    if burst_traces:
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
        max_step = 0
        for name in ordered_names(list(burst_traces.keys()), include_teacher=False):
            rows = burst_traces[name]
            steps = [int(row["step"]) for row in rows]
            max_step = max(max_step, max(steps))
            style = scheme_style(name)
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
        save_fig(fig, ext_path(out_dir, "burst_adaptation", fmt))


def plot_episode_epoch_bundle(main_dir: Path, out_root: Path, fmt: str) -> None:
    print("\n[3/8] episode_epoch_curves")
    out_dir = out_root / "episode_epoch_curves"
    imit_rows = load_float_csv(main_dir / "policy_imitation.csv")
    eval_rows = load_float_csv(main_dir / "temporal_graph_eval.csv")

    epochs = [int(row["epoch"]) for row in imit_rows]
    episodes = [int(row["episode"]) for row in eval_rows]

    line_plot(
        epochs,
        {"TemporalGraph": [row["local_hit_rate"] for row in imit_rows]},
        "Epoch",
        "Local Hit Ratio",
        ext_path(out_dir, "local_hit_vs_epoch", fmt),
        "Local Hit Ratio vs Epoch",
        integer_x=True,
    )
    line_plot(
        epochs,
        {"TemporalGraph": [row["paper_hit_rate"] for row in imit_rows]},
        "Epoch",
        "Edge Hit Ratio",
        ext_path(out_dir, "edge_hit_vs_epoch", fmt),
        "Edge Hit Ratio vs Epoch",
        integer_x=True,
    )
    line_plot(
        episodes,
        {"TemporalGraph": [row["reward"] for row in eval_rows]},
        "Evaluation Episode",
        "Reward",
        ext_path(out_dir, "reward_vs_episode_only", fmt),
        "Reward vs Evaluation Episode",
        integer_x=True,
    )
    line_plot(
        episodes,
        {"TemporalGraph": [row["paper_hit_rate"] for row in eval_rows]},
        "Evaluation Episode",
        "Edge Hit Ratio",
        ext_path(out_dir, "edge_hit_vs_episode_only", fmt),
        "Edge Hit Ratio vs Evaluation Episode",
        integer_x=True,
    )
    line_plot(
        episodes,
        {"TemporalGraph": [row["cloud_fetch_rate"] for row in eval_rows]},
        "Evaluation Episode",
        "Cloud Fetch Rate",
        ext_path(out_dir, "cloud_fetch_vs_episode_only", fmt),
        "Cloud Fetch Rate vs Evaluation Episode",
        integer_x=True,
    )


def plot_final_no_teacher_bundle(main_dir: Path, out_root: Path, fmt: str) -> None:
    print("\n[4/8] final_no_teacher_bundle")
    out_dir = out_root / "final_no_teacher_bundle"
    rows_by_model = {
        "Random": load_float_csv(main_dir / "random_eval.csv"),
        "BSG": load_float_csv(main_dir / "bsg_like_eval.csv"),
        "C-ε-greedy": load_float_csv(main_dir / "c_epsilon_greedy_eval.csv"),
        "TemporalGraph": load_float_csv(main_dir / "temporal_graph_eval.csv"),
    }

    def metric_mean(name: str, metric: str) -> float:
        return float(np.mean([row[metric] for row in rows_by_model[name]]))

    for metric, ylabel, stem in [
        ("reward", "Mean Episode Reward", "reward_mean_no_teacher"),
        ("paper_hit_rate", "Mean Edge Hit Ratio", "edge_hit_mean_no_teacher"),
        ("cloud_fetch_rate", "Mean Cloud Fetch Rate", "cloud_fetch_mean_no_teacher"),
        ("neighbor_fetch_rate", "Mean Neighbor Fetch Rate", "neighbor_fetch_mean_no_teacher"),
    ]:
        bar_plot(
            {name: metric_mean(name, metric) for name in rows_by_model},
            ylabel,
            ext_path(out_dir, stem, fmt),
            f"{ylabel} Comparison",
            include_teacher=False,
        )

    episodes = [int(row["episode"]) for row in next(iter(rows_by_model.values()))]
    for metric, ylabel, stem in [
        ("reward", "Episode Reward", "reward_vs_episode_no_teacher"),
        ("paper_hit_rate", "Edge Hit Ratio", "edge_hit_vs_episode_no_teacher"),
        ("cloud_fetch_rate", "Cloud Fetch Rate", "cloud_fetch_vs_episode_no_teacher"),
        ("neighbor_fetch_rate", "Neighbor Fetch Rate", "neighbor_fetch_vs_episode_no_teacher"),
    ]:
        line_plot(
            episodes,
            {name: [row[metric] for row in rows] for name, rows in rows_by_model.items()},
            "Evaluation Episode",
            ylabel,
            ext_path(out_dir, stem, fmt),
            f"{ylabel} Across Evaluation Episodes",
            integer_x=True,
        )

    ordered = ordered_names(list(rows_by_model.keys()), include_teacher=False)
    stacked_bar_plot(
        ordered,
        [
            ("Local served", [metric_mean(name, "local_hit_rate") for name in ordered], "#2a9d8f", "///"),
            ("Neighbor served", [metric_mean(name, "neighbor_fetch_rate") for name in ordered], "#e9c46a", "\\\\\\"),
            ("Cloud served", [metric_mean(name, "cloud_fetch_rate") for name in ordered], "#adb5bd", "xxx"),
        ],
        "Request Fraction",
        ext_path(out_dir, "service_source_composition_no_teacher", fmt),
        "Service Source Composition",
    )

    tg_edge = metric_mean("TemporalGraph", "paper_hit_rate")
    gains = {}
    for name in ["Random", "BSG", "C-ε-greedy"]:
        base = metric_mean(name, "paper_hit_rate")
        gains[name] = 100.0 * (tg_edge - base) / max(base, 1e-8)
    bar_plot(gains, "Gain (%)", ext_path(out_dir, "edge_hit_gain_no_teacher", fmt), "TemporalGraph Edge-Hit Gain Over Baselines", include_teacher=False, fmt=".1f")

    episodes_n = len(next(iter(rows_by_model.values())))
    win_counts = {"Reward wins": 0, "Edge-hit wins": 0, "Cloud-min wins": 0, "Neighbor-use wins": 0}
    for idx in range(episodes_n):
        reward_vals = {name: rows[idx]["reward"] for name, rows in rows_by_model.items()}
        edge_vals = {name: rows[idx]["paper_hit_rate"] for name, rows in rows_by_model.items()}
        cloud_vals = {name: rows[idx]["cloud_fetch_rate"] for name, rows in rows_by_model.items()}
        neigh_vals = {name: rows[idx]["neighbor_fetch_rate"] for name, rows in rows_by_model.items()}
        if reward_vals["TemporalGraph"] == max(reward_vals.values()):
            win_counts["Reward wins"] += 1
        if edge_vals["TemporalGraph"] == max(edge_vals.values()):
            win_counts["Edge-hit wins"] += 1
        if cloud_vals["TemporalGraph"] == min(cloud_vals.values()):
            win_counts["Cloud-min wins"] += 1
        if neigh_vals["TemporalGraph"] == max(neigh_vals.values()):
            win_counts["Neighbor-use wins"] += 1
    bar_plot(
        win_counts,
        "Winning Episodes",
        ext_path(out_dir, "temporalgraph_win_count_no_teacher", fmt),
        "TemporalGraph Episode-Level Win Count",
        rotate=14,
        include_teacher=False,
        fmt=".0f",
    )


def plot_temporalgraph_showcase_bundle(main_dir: Path, out_root: Path, fmt: str) -> None:
    print("\n[5/8] temporalgraph_showcase")
    out_dir = out_root / "temporalgraph_showcase"
    summary = parse_summary_txt(main_dir / "summary.txt")
    summary.pop("Teacher", None)
    comparison = {name: values for name, values in summary.items() if name in {"TemporalGraph", "Random", "BSG", "C-ε-greedy"}}
    if not comparison:
        return

    for metric, ylabel, stem in [
        ("reward_mean", "Mean Reward", "ml1m_run_reward_comparison"),
        ("local_hit_mean", "Mean Local Hit Ratio", "ml1m_run_local_hit_comparison"),
        ("paper_hit_mean", "Mean Edge Hit Ratio", "ml1m_run_edge_hit_comparison"),
    ]:
        bar_plot(
            {name: values[metric] for name, values in comparison.items()},
            ylabel,
            ext_path(out_dir, stem, fmt),
            f"MovieLens-1M Run: {ylabel}",
            include_teacher=False,
        )

    tg = comparison["TemporalGraph"]
    gains = {}
    for metric, label in [
        ("reward_mean", "Reward"),
        ("local_hit_mean", "Local Hit"),
        ("paper_hit_mean", "Edge Hit"),
    ]:
        best_baseline = max(values[metric] for name, values in comparison.items() if name != "TemporalGraph")
        gains[label] = 100.0 * (tg[metric] - best_baseline) / max(abs(best_baseline), 1e-8)
    bar_plot(gains, "Relative Gain (%)", ext_path(out_dir, "ml1m_run_relative_gain", fmt), "TemporalGraph Gain Over Best Baseline", rotate=0, include_teacher=False, fmt=".1f")

    imitation_rows = load_float_csv(main_dir / "policy_imitation.csv")
    epochs = [int(row["epoch"]) for row in imitation_rows]
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
    save_fig(fig, ext_path(out_dir, "ml1m_run_imitation_training", fmt))

    eval_rows = load_float_csv(main_dir / "temporal_graph_eval.csv")
    episodes = [int(row["episode"]) for row in eval_rows]
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
    save_fig(fig, ext_path(out_dir, "ml1m_run_eval_episodes", fmt))


def plot_static_vs_dynamic_bundle(data_dir: Path, out_root: Path, fmt: str) -> None:
    print("\n[6/8] static_vs_dynamic_bundle")
    out_dir = out_root / "static_vs_dynamic_bundle"
    csv_path = data_dir / "static_vs_dynamic_bundle" / "static_vs_dynamic_metrics.csv"
    if not csv_path.exists():
        print(f"  ⚠ Missing {csv_path}")
        return
    rows = load_float_csv(csv_path)
    time_s = [row["time_s"] for row in rows]
    fallback_metrics: dict[str, np.ndarray] | None = None

    def load_fallback_metrics() -> dict[str, np.ndarray]:
        nonlocal fallback_metrics
        if fallback_metrics is not None:
            return fallback_metrics
        cfg = SimulationConfig(
            grid_size=300.0,
            n_ues=300,
            n_sbs=8,
            total_time=300.0,
            dt=1.0,
            t_update=5.0,
            max_speed=4.0,
            sbs_max_speed=1.5,
            prediction_horizon_factor=1.0,
            n_hotspots=6,
            hotspot_speed=1.2,
            hotspot_weight=0.75,
            random_seed=42,
            kmeans_iters=30,
        )
        moving_result = run_simulation(cfg)
        fixed_result = freeze_sbs_positions(moving_result)
        fixed = compute_static_dynamic_metrics(fixed_result, cfg, 35.0, 0.2)
        moving = compute_static_dynamic_metrics(moving_result, cfg, 35.0, 0.2)
        fallback_metrics = {
            "fixed_p95_distance": fixed["p95_distance"],
            "moving_p95_distance": moving["p95_distance"],
        }
        return fallback_metrics

    def series(key: str) -> tuple[list[float], list[float]]:
        if key == "cumulative_distance":
            fixed = np.asarray([row["fixed_mean_distance"] for row in rows], dtype=np.float64)
            moving = np.asarray([row["moving_mean_distance"] for row in rows], dtype=np.float64)
            denom = np.arange(1, fixed.shape[0] + 1, dtype=np.float64)
            return (list(np.cumsum(fixed) / denom), list(np.cumsum(moving) / denom))
        if key == "p95_distance":
            metrics = load_fallback_metrics()
            return (metrics["fixed_p95_distance"].tolist(), metrics["moving_p95_distance"].tolist())
        return ([row[f"fixed_{key}"] for row in rows], [row[f"moving_{key}"] for row in rows])

    plot_specs = [
        ("mean_distance", "Distance", "mean_distance_over_time", "Mean UE-to-Serving-SBS Distance", True),
        ("cumulative_distance", "Cumulative Mean Distance", "cumulative_mean_distance", "Cumulative Mean Distance", True),
        ("effective_cost", "Mean Effective Cost", "effective_distance_cost", "Handover-Aware Effective Distance Cost", True),
        ("coverage_fraction", "Coverage Fraction", "coverage_fraction_over_time", "UE Fraction Within Coverage Radius", False),
        ("p95_distance", "P95 Distance", "p95_distance_over_time", "95th Percentile UE-to-SBS Distance", True),
        ("hotspot_tracking_error", "Distance to Nearest SBS", "hotspot_tracking_error", "Hotspot Tracking Error", True),
        ("cluster_capture_rate", "Capture Rate", "cluster_capture_rate", "Cluster Capture Rate", False),
        ("load_cv", "Load CV", "load_imbalance_cv", "SBS Load Imbalance (CV)", True),
        ("handover_rate", "Handover Rate", "handover_rate_over_time", "Handover Rate", True),
    ]
    for key, ylabel, stem, title, lower_is_better in plot_specs:
        fixed_vals, moving_vals = series(key)
        pair_line_plot(time_s, fixed_vals, moving_vals, "Time (s)", ylabel, ext_path(out_dir, stem, fmt), title, lower_is_better)

    fixed_dist, moving_dist = series("mean_distance")
    gain = 100.0 * (np.asarray(fixed_dist) - np.asarray(moving_dist)) / np.maximum(np.asarray(fixed_dist), 1e-8)
    line_plot(time_s, {"TemporalGraph": gain.tolist()}, "Time (s)", "Distance Reduction (%)", ext_path(out_dir, "distance_reduction_gain", fmt), "Distance Reduction from SBS Mobility")

    fixed_hot, moving_hot = series("hotspot_tracking_error")
    hot_gain = 100.0 * (np.asarray(fixed_hot) - np.asarray(moving_hot)) / np.maximum(np.asarray(fixed_hot), 1e-8)
    line_plot(time_s, {"TemporalGraph": hot_gain.tolist()}, "Time (s)", "Tracking Improvement (%)", ext_path(out_dir, "hotspot_tracking_gain", fmt), "Hotspot-Tracking Improvement from SBS Mobility")

    fixed_sorted = np.sort(np.asarray(fixed_dist))
    moving_sorted = np.sort(np.asarray(moving_dist))
    fy = np.arange(1, fixed_sorted.shape[0] + 1) / fixed_sorted.shape[0]
    my = np.arange(1, moving_sorted.shape[0] + 1) / moving_sorted.shape[0]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(fixed_sorted, fy, color=PAIR_STYLE["UE moving, SBS fixed"]["color"], linestyle=PAIR_STYLE["UE moving, SBS fixed"]["linestyle"], linewidth=2.0, label="UE moving, SBS fixed")
    ax.plot(moving_sorted, my, color=PAIR_STYLE["UE moving, SBS moving"]["color"], linestyle=PAIR_STYLE["UE moving, SBS moving"]["linestyle"], linewidth=2.0, label="UE moving, SBS moving")
    ax.set_title("CDF of Mean UE-to-SBS Distance")
    ax.set_xlabel("Mean Distance")
    ax.set_ylabel("CDF")
    ax.legend(loc="best")
    save_fig(fig, ext_path(out_dir, "distance_cdf", fmt))


def plot_clustered_latency_bundle(data_dir: Path, out_root: Path, fmt: str) -> None:
    print("\n[7/8] clustered_latency_study")
    out_dir = out_root / "clustered_latency_study"
    args = SimpleNamespace(
        latency_base_ms=5.0,
        latency_distance_factor_ms=0.01,
        latency_distance_exponent=2.0,
        latency_handover_penalty_ms=0.2,
        latency_hysteresis=0.0,
    )
    base_cfg = SimulationConfig(
        grid_size=300.0,
        n_ues=300,
        n_sbs=8,
        total_time=180.0,
        dt=1.0,
        t_update=5.0,
        max_speed=4.0,
        sbs_max_speed=1.5,
        prediction_horizon_factor=1.0,
        n_hotspots=6,
        hotspot_speed=1.2,
        hotspot_weight=0.75,
        random_seed=42,
        kmeans_iters=30,
    )
    base = evaluate_pair(base_cfg, args)
    pair_line_plot(base["time_s"], base["fixed_latency"], base["moving_latency"], "Time (s)", "Latency (ms)", ext_path(out_dir, "latency_over_time", fmt), "Mean Latency Over Time", True)

    fixed_latency = np.asarray(base["fixed_latency"], dtype=np.float64)
    moving_latency = np.asarray(base["moving_latency"], dtype=np.float64)
    steps = np.arange(1, fixed_latency.shape[0] + 1)
    pair_line_plot(base["time_s"], (np.cumsum(fixed_latency) / steps).tolist(), (np.cumsum(moving_latency) / steps).tolist(), "Time (s)", "Cumulative Mean Latency (ms)", ext_path(out_dir, "cumulative_average_latency", fmt), "Cumulative Average Latency", True)

    fixed_sorted = np.sort(fixed_latency)
    moving_sorted = np.sort(moving_latency)
    fy = np.arange(1, fixed_sorted.shape[0] + 1) / fixed_sorted.shape[0]
    my = np.arange(1, moving_sorted.shape[0] + 1) / moving_sorted.shape[0]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(fixed_sorted, fy, color=PAIR_STYLE["UE moving, SBS fixed"]["color"], linestyle=PAIR_STYLE["UE moving, SBS fixed"]["linestyle"], linewidth=2.0, label="UE moving, SBS fixed")
    ax.plot(moving_sorted, my, color=PAIR_STYLE["UE moving, SBS moving"]["color"], linestyle=PAIR_STYLE["UE moving, SBS moving"]["linestyle"], linewidth=2.0, label="UE moving, SBS moving")
    ax.set_title("Latency CDF")
    ax.set_xlabel("Mean Step Latency (ms)")
    ax.set_ylabel("CDF")
    ax.legend(loc="best")
    save_fig(fig, ext_path(out_dir, "latency_cdf", fmt))

    sweep_specs = [
        ("latency_vs_t_update.csv", "t_update_s", "SBS Update Interval (s)", "latency_vs_t_update", "Mean Latency vs SBS Update Interval"),
        ("latency_vs_sbs_speed.csv", "sbs_max_speed", "Max SBS Speed (grid units/s)", "latency_vs_sbs_speed", "Mean Latency vs SBS Speed"),
        ("latency_vs_hotspot_weight.csv", "hotspot_weight", "Hotspot Attraction Weight", "latency_vs_hotspot_weight", "Mean Latency vs UE Cluster Strength"),
        ("latency_vs_n_sbs.csv", "n_sbs", "Number of SBSs", "latency_vs_n_sbs", "Mean Latency vs Number of SBSs"),
    ]
    for filename, x_key, xlabel, stem, title in sweep_specs:
        rows = load_float_csv(data_dir / "clustered_latency_study" / filename)
        x_vals = [row[x_key] for row in rows]
        pair_line_plot(
            x_vals,
            [row["fixed_mean_latency_ms"] for row in rows],
            [row["moving_mean_latency_ms"] for row in rows],
            xlabel,
            "Mean Latency (ms)",
            ext_path(out_dir, stem, fmt),
            title,
            True,
        )

    n_sbs_rows = load_float_csv(data_dir / "clustered_latency_study" / "latency_vs_n_sbs.csv")
    improvement = {
        str(int(row["n_sbs"])): 100.0 * (row["fixed_mean_latency_ms"] - row["moving_mean_latency_ms"]) / max(row["fixed_mean_latency_ms"], 1e-8)
        for row in n_sbs_rows
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(improvement))
    vals = list(improvement.values())
    ax.bar(x, vals, 0.65, color="#d62828", edgecolor="black", linewidth=0.65, hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels(list(improvement.keys()))
    ax.set_xlabel("Number of SBSs")
    ax.set_ylabel("Latency Reduction (%)")
    ax.set_title("Latency Reduction from SBS Mobility")
    save_fig(fig, ext_path(out_dir, "latency_reduction_vs_n_sbs", fmt))


def plot_consolidated_summary(data_dir: Path, out_root: Path, fmt: str) -> None:
    print("\n[8/8] consolidated_summary")
    scheme_metrics: dict[str, dict[str, float]] = {}
    rw_summary = data_dir / "related_work_compare" / "summary.csv"
    if rw_summary.exists():
        for row in load_csv(rw_summary):
            name = map_name(row["scheme"])
            if name is None:
                continue
            scheme_metrics[name] = {
                "reward": float(row["reward_mean"]),
                "local_hit": float(row["local_hit_mean"]),
                "edge_hit": float(row["paper_hit_mean"]),
                "cloud_fetch": float(row["cloud_fetch_mean"]),
            }
    for name, metrics in parse_summary_txt(data_dir / "novel_realworld_main" / "summary.txt").items():
        if name in scheme_metrics:
            continue
        scheme_metrics[name] = {
            "reward": metrics.get("reward_mean", 0.0),
            "local_hit": metrics.get("local_hit_mean", 0.0),
            "edge_hit": metrics.get("paper_hit_mean", 0.0),
            "cloud_fetch": 1.0 - metrics.get("paper_hit_mean", 0.0),
        }
    scheme_metrics.pop("Teacher", None)
    ordered = ordered_names(list(scheme_metrics.keys()), include_teacher=False)
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
            style = scheme_style(name)
            value = scheme_metrics[name][metric]
            ax.bar(idx, value, 0.62, color=style["color"], edgecolor="black", linewidth=0.55, hatch=style["hatch"])
            ax.annotate(f"{value:.3f}", xy=(idx, value), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=28, ha="right", fontsize=8)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
    save_fig(fig, ext_path(out_root, "consolidated_summary", fmt))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper-ready apr11 plots for the full apr6_v3-style suite.")
    p.add_argument("--data-dir", type=Path, default=Path("outputs/full_suite_20260405_paperhit"))
    p.add_argument("--data-dir-v3", type=Path, default=Path("outputs/plots_apr6_v3"))
    p.add_argument("--run-dir", type=Path, default=None, help="Run directory holding the TemporalGraph checkpoints/eval CSVs.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/plots_apr11"))
    p.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg", "eps"])
    p.add_argument("--python-bin", type=str, default=sys.executable)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir if args.data_dir.exists() else args.data_dir_v3
    if not data_dir.exists():
        print("ERROR: no data directory available.")
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir or (data_dir / "novel_realworld_main")
    if not run_dir.exists():
        print(f"ERROR: run directory {run_dir} does not exist.")
        sys.exit(1)

    print(f"Data directory: {data_dir}")
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Format: {args.format}")
    print("=" * 60)

    bundle_dir = ensure_extended_bundle(data_dir, output_dir, run_dir, args.python_bin)

    plot_related_work_bundle(data_dir, output_dir, args.format)
    plot_novel_comparison_bundle(bundle_dir, output_dir, args.format)
    plot_episode_epoch_bundle(run_dir, output_dir, args.format)
    plot_final_no_teacher_bundle(run_dir, output_dir, args.format)
    plot_temporalgraph_showcase_bundle(run_dir, output_dir, args.format)
    plot_static_vs_dynamic_bundle(data_dir, output_dir, args.format)
    plot_clustered_latency_bundle(data_dir, output_dir, args.format)
    plot_consolidated_summary(data_dir, output_dir, args.format)

    print("=" * 60)
    print(f"All apr11 plots saved under: {output_dir}")


if __name__ == "__main__":
    main()
