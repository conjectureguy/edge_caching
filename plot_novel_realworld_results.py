from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "Random": "Random",
    "BSG-like": "BSG-like",
    "C-epsilon-greedy": "C-epsilon-greedy",
    "LRU": "LRU",
    "LFU": "LFU",
    "Thompson": "Thompson",
    "TemporalGraph": "TemporalGraph",
    "MAAFDRL": "MAAFDRL",
}

COLORS = {
    "Random": "#8d99ae",
    "BSG-like": "#577590",
    "C-epsilon-greedy": "#90be6d",
    "LRU": "#577590",
    "LFU": "#4d908e",
    "Thompson": "#7b2cbf",
    "TemporalGraph": "#d62828",
    "MAAFDRL": "#2a9d8f",
}

RELATED_NAME_MAP = {
    "Our-TemporalGraph": "TemporalGraph",
    "Paper3-MAAFDRL-like": "MAAFDRL",
    "TemporalGraph": "TemporalGraph",
    "MAAFDRL": "MAAFDRL",
    "LRU": "LRU",
    "LFU": "LFU",
    "Thompson": "Thompson",
    "Paper2-AWFDRL-like": None,
    "Paper4-DTS-DDPG-like": None,
}

PLOT_ORDER = [
    "Random",
    "BSG-like",
    "C-epsilon-greedy",
    "LRU",
    "LFU",
    "Thompson",
    "MAAFDRL",
    "TemporalGraph",
]


def _normalize_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _build_model_filter(include_models: list[str], exclude_models: list[str]):
    include = {_normalize_model_name(name) for name in include_models}
    exclude = {_normalize_model_name(name) for name in exclude_models}

    def _allowed(name: str) -> bool:
        normalized = _normalize_model_name(name)
        if include and normalized not in include:
            return False
        if normalized in exclude:
            return False
        return True

    return _allowed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot results for the novel real-world cooperative caching pipeline.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--related-work-dir", type=Path, default=None)
    p.add_argument("--include-models", nargs="*", default=[])
    p.add_argument("--exclude-models", nargs="*", default=[])
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def available_eval_specs(input_dir: Path) -> list[tuple[str, str, str]]:
    specs = [
        ("random_eval.csv", "Random", "#8d99ae"),
        ("bsg_like_eval.csv", "BSG-like", "#577590"),
        ("c_epsilon_greedy_eval.csv", "C-epsilon-greedy", "#90be6d"),
        ("temporal_graph_eval.csv", "TemporalGraph", "#d62828"),
    ]
    return [spec for spec in specs if (input_dir / spec[0]).exists()]


def detect_related_work_dir(input_dir: Path, related_work_dir: Path | None) -> Path | None:
    if related_work_dir is not None:
        return related_work_dir
    candidate = input_dir.parent / "related_work_compare"
    return candidate if candidate.exists() else None


def load_related_summary(related_work_dir: Path | None) -> list[dict[str, float | str]]:
    if related_work_dir is None:
        return []
    path = related_work_dir / "summary.csv"
    if not path.exists():
        return []
    rows: list[dict[str, float | str]] = []
    for row in read_csv(path):
        name = RELATED_NAME_MAP.get(row["scheme"], row["scheme"])
        if name is None:
            continue
        rows.append(
            {
                "label": name,
                "reward": float(row["reward_mean"]),
                "local": float(row["local_hit_mean"]),
                "paper": float(row["paper_hit_mean"]),
                "cloud": float(row["cloud_fetch_mean"]),
            }
        )
    return rows


def load_related_episode_curves(related_work_dir: Path | None) -> list[tuple[list[dict[str, str]], str, str]]:
    if related_work_dir is None:
        return []
    path = related_work_dir / "episode_metrics.csv"
    if not path.exists():
        return []
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(path):
        name = RELATED_NAME_MAP.get(row["scheme"], row["scheme"])
        if name is None:
            continue
        grouped.setdefault(name, []).append(
            {
                "episode": row["episode"],
                "reward": row["reward"],
                "paper_hit_rate": row["paper_hit_rate"],
            }
        )
    return [(rows, name, COLORS[name]) for name, rows in grouped.items() if name != "TemporalGraph"]


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


def plot_eval_bars(input_dir: Path, output_dir: Path, related_work_dir: Path | None, allowed_model) -> None:
    eval_files = [(filename, label) for filename, label, _ in available_eval_specs(input_dir)]
    summary: list[dict[str, float | str]] = []
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
    seen = {str(row["label"]) for row in summary}
    for row in load_related_summary(related_work_dir):
        label = str(row["label"])
        if label in seen or label == "Teacher" or not allowed_model(label):
            continue
        summary.append(row)
        seen.add(label)
    summary = [row for row in summary if allowed_model(str(row["label"]))]

    if not summary:
        return

    summary.sort(key=lambda row: PLOT_ORDER.index(str(row["label"])) if str(row["label"]) in PLOT_ORDER else len(PLOT_ORDER))
    labels = [x["label"] for x in summary]
    x = np.arange(len(labels))
    bar_colors = [COLORS[str(label)] for label in labels]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(x, [s["reward"] for s in summary], color=bar_colors)
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
    plt.savefig(output_dir / "baseline_comparison_no_teacher.png", dpi=180)
    plt.close(fig)


def plot_eval_episode_curves(input_dir: Path, output_dir: Path, related_work_dir: Path | None, allowed_model) -> None:
    eval_files = available_eval_specs(input_dir)
    available = []
    for filename, label, color in eval_files:
        path = input_dir / filename
        if path.exists() and allowed_model(label):
            available.append((read_csv(path), label, color))
    available.extend([entry for entry in load_related_episode_curves(related_work_dir) if allowed_model(entry[1])])
    if not available:
        return
    available.sort(key=lambda item: PLOT_ORDER.index(item[1]) if item[1] in PLOT_ORDER else len(PLOT_ORDER))

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
    plt.savefig(output_dir / "episode_curves_no_teacher.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    related_work_dir = detect_related_work_dir(args.input_dir, args.related_work_dir)
    allowed_model = _build_model_filter(args.include_models, args.exclude_models)
    maybe_plot_temporal_training(args.input_dir, output_dir)
    maybe_plot_policy_imitation(args.input_dir, output_dir)
    plot_eval_bars(args.input_dir, output_dir, related_work_dir, allowed_model)
    plot_eval_episode_curves(args.input_dir, output_dir, related_work_dir, allowed_model)
    print(f"Saved plots under: {output_dir}")


if __name__ == "__main__":
    main()
