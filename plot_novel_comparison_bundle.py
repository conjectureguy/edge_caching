from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from types import MethodType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_related_work_papers import (
    LFUPolicy,
    LRUPolicy,
    MobilityAwareAsyncFDRLPolicy,
    ThompsonPolicy,
    _adapt_candidate_features,
    _adapt_node_features,
)
from movie_edge_sim.data import get_movielens_dataset, load_item_genres_auto, load_ratings_auto
from movie_edge_sim.novel_graph_policy import TemporalGraphCooperativePolicy, logits_to_cache_items
from movie_edge_sim.novel_realworld_env import NovelRealWorldCachingEnv, RealWorldEnvConfig
from movie_edge_sim.temporal_realworld import (
    RealWorldTemporalEncoder,
    build_user_time_histories,
    load_compatible_temporal_state,
)
from train_novel_realworld_cache import eval_bsg, eval_c_epsilon_greedy, eval_random


@dataclass
class BundleConfig:
    run_dir: Path
    output_dir: Path
    data_root: Path
    dataset_name: str
    device: str
    eval_episodes: int
    episode_len: int
    window_size: int
    fp: int
    n_ues: int
    grid_size: float
    cache_capacities: list[int]
    sbs_list: list[int]
    seed: int
    decode_diversity_penalty: float
    c_epsilon: float
    include_models: list[str]
    exclude_models: list[str]


def parse_args() -> BundleConfig:
    p = argparse.ArgumentParser(description="Generate comparison plots for TemporalGraph vs baselines.")
    p.add_argument("--run-dir", type=Path, default=Path("outputs/novel_realworld_ml1m_v3"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/novel_comparison_bundle"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--dataset-name", type=str, default="ml-1m", choices=["ml-100k", "ml-1m"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--eval-episodes", type=int, default=2)
    p.add_argument("--episode-len", type=int, default=60)
    p.add_argument("--window-size", type=int, default=12)
    p.add_argument("--fp", type=int, default=50)
    p.add_argument("--n-ues", type=int, default=220)
    p.add_argument("--grid-size", type=float, default=300.0)
    p.add_argument("--cache-capacities", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--sbs-list", type=int, nargs="+", default=[8, 12, 16])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decode-diversity-penalty", type=float, default=0.35)
    p.add_argument("--c-epsilon", type=float, default=0.18)
    p.add_argument("--include-models", nargs="*", default=[])
    p.add_argument("--exclude-models", nargs="*", default=[])
    args = p.parse_args()
    return BundleConfig(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        device=args.device,
        eval_episodes=args.eval_episodes,
        episode_len=args.episode_len,
        window_size=args.window_size,
        fp=args.fp,
        n_ues=args.n_ues,
        grid_size=args.grid_size,
        cache_capacities=args.cache_capacities,
        sbs_list=args.sbs_list,
        seed=args.seed,
        decode_diversity_penalty=args.decode_diversity_penalty,
        c_epsilon=args.c_epsilon,
        include_models=args.include_models,
        exclude_models=args.exclude_models,
    )


def _normalize_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _allowed_model(name: str, cfg: BundleConfig) -> bool:
    include = {_normalize_model_name(model) for model in cfg.include_models}
    exclude = {_normalize_model_name(model) for model in cfg.exclude_models}
    normalized = _normalize_model_name(name)
    if include and normalized not in include:
        return False
    if normalized in exclude:
        return False
    return True


def load_histories_and_temporal(cfg: BundleConfig):
    dataset_dir = get_movielens_dataset(cfg.data_root, cfg.dataset_name)
    ratings = load_ratings_auto(dataset_dir)
    item_genres, _genre_names = load_item_genres_auto(dataset_dir)
    histories = build_user_time_histories(ratings)
    max_user = max(histories.keys())
    max_item = max(max(hist.items) for hist in histories.values())
    temporal = RealWorldTemporalEncoder(
        num_items=max_item,
        num_users=max_user,
        window_size=cfg.window_size,
        embed_dim=64,
        hidden_dim=128,
        num_heads=4,
    )
    load_compatible_temporal_state(
        temporal,
        torch.load(cfg.run_dir / "realworld_temporal_encoder.pt", map_location=cfg.device, weights_only=True),
        source=str(cfg.run_dir / "realworld_temporal_encoder.pt"),
    )
    temporal.eval()
    return histories, temporal, item_genres


def infer_hidden_dim(run_dir: Path) -> int:
    state = torch.load(run_dir / "temporal_graph_policy.pt", map_location="cpu", weights_only=True)
    return int(state["gat1.proj.weight"].shape[0])


def build_env_and_model(cfg: BundleConfig, histories, temporal_model, item_genres, n_sbs: int, cache_capacity: int):
    env_cfg = RealWorldEnvConfig(
        n_sbs=n_sbs,
        n_ues=cfg.n_ues,
        cache_capacity=cache_capacity,
        fp=cfg.fp,
        window_size=cfg.window_size,
        episode_len=cfg.episode_len,
        grid_size=cfg.grid_size,
        seed=cfg.seed,
    )
    env = NovelRealWorldCachingEnv(env_cfg, temporal_model, histories, item_genres=item_genres)
    policy_state = torch.load(cfg.run_dir / "temporal_graph_policy.pt", map_location=cfg.device, weights_only=True)
    hidden_dim = int(policy_state["gat1.proj.weight"].shape[0])
    expected_node_dim = int(policy_state["gat1.proj.weight"].shape[1])
    expected_candidate_dim = int(policy_state["candidate_encoder.0.weight"].shape[1])
    model = TemporalGraphCooperativePolicy(
        node_feat_dim=expected_node_dim,
        candidate_feat_dim=expected_candidate_dim,
        hidden_dim=hidden_dim,
        fp=cfg.fp,
        use_graph=True,
    ).to(cfg.device)
    model.load_state_dict(policy_state, strict=False)
    model.eval()
    model.expected_node_dim = expected_node_dim
    model.expected_candidate_dim = expected_candidate_dim
    return env, model


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "reward_mean": float(np.mean([r["reward"] for r in rows])),
        "paper_hit_mean": float(np.mean([r["paper_hit_rate"] for r in rows])),
        "local_hit_mean": float(np.mean([r["local_hit_rate"] for r in rows])),
        "neighbor_mean": float(np.mean([r["neighbor_fetch_rate"] for r in rows])),
        "cloud_mean": float(np.mean([r["cloud_fetch_rate"] for r in rows])),
    }


def evaluate_temporal_graph(env: NovelRealWorldCachingEnv, model: TemporalGraphCooperativePolicy, episodes: int, seed: int, device: str, diversity_penalty: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    dev = torch.device(device)
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        reward_sum = 0.0
        local_sum = 0.0
        neighbor_sum = 0.0
        cloud_sum = 0.0
        steps = 0
        while not done:
            node = _adapt_node_features(obs["node_features"], model.expected_node_dim, env.embed_dim)
            cand = _adapt_candidate_features(obs["candidate_features"], model.expected_candidate_dim, env.embed_dim)
            node_t = torch.as_tensor(node, dtype=torch.float32, device=dev)
            cand_t = torch.as_tensor(cand, dtype=torch.float32, device=dev)
            adj_t = torch.as_tensor(obs["adjacency"], dtype=torch.float32, device=dev)
            mask_t = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=dev)
            logits = model(node_t, cand_t, adj_t, mask_t)
            chosen = logits_to_cache_items(logits, env, diversity_penalty=diversity_penalty)
            obs, reward, done, info = env.step_full_cache_items(chosen)
            reward_sum += float(reward)
            local_sum += float(info["local_hit_rate"])
            neighbor_sum += float(info["neighbor_fetch_rate"])
            cloud_sum += float(info["cloud_fetch_rate"])
            steps += 1
        rows.append(
            {
                "episode": ep + 1,
                "reward": reward_sum,
                "local_hit_rate": local_sum / max(1, steps),
                "neighbor_fetch_rate": neighbor_sum / max(1, steps),
                "cloud_fetch_rate": cloud_sum / max(1, steps),
                "paper_hit_rate": (local_sum + neighbor_sum) / max(1, steps),
            }
        )
    return rows


def eval_action_policy(env: NovelRealWorldCachingEnv, action_fn, episodes: int, seed: int) -> tuple[list[dict[str, float]], dict[str, float], list[dict[str, float]]]:
    rows = []
    cost_acc = {"local": 0.0, "neighbor": 0.0, "cloud": 0.0, "replace": 0.0}
    trace: list[dict[str, float]] = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        if hasattr(action_fn, "__self__") and action_fn.__self__ is not None and hasattr(action_fn.__self__, "reset"):
            action_fn.__self__.reset(env)
        done = False
        reward_sum = local_sum = neighbor_sum = cloud_sum = 0.0
        steps = 0
        prev_cache = env.cache_items.copy()
        while not done:
            association = obs["association"].copy()
            action = action_fn(env, obs)
            replaced = float(np.sum(~np.isin(action, prev_cache)))
            prev_cache = action.copy()
            obs, reward, done, info = env.step_full_cache_items(action)
            if hasattr(action_fn, "__self__") and action_fn.__self__ is not None and hasattr(action_fn.__self__, "update"):
                action_fn.__self__.update(env, association)
            reward_sum += float(reward)
            local_sum += float(info["local_hit_rate"])
            neighbor_sum += float(info["neighbor_fetch_rate"])
            cloud_sum += float(info["cloud_fetch_rate"])
            cost_acc["local"] += env.cfg.alpha_local * float(info["local_hit_rate"])
            cost_acc["neighbor"] += env.cfg.beta_neighbor * float(info["neighbor_fetch_rate"])
            cost_acc["cloud"] += env.cfg.chi_cloud * float(info["cloud_fetch_rate"])
            cost_acc["replace"] += env.cfg.delta_replace * (replaced / max(1.0, env.cfg.n_sbs * env.cfg.cache_capacity))
            trace.append(
                {
                    "episode": float(ep + 1),
                    "step": float(steps),
                    "reward": float(reward),
                    "local_hit_rate": float(info["local_hit_rate"]),
                    "neighbor_fetch_rate": float(info["neighbor_fetch_rate"]),
                    "cloud_fetch_rate": float(info["cloud_fetch_rate"]),
                    "paper_hit_rate": float(info["local_hit_rate"] + info["neighbor_fetch_rate"]),
                    "cache_overlap": float(mean_neighbor_overlap(env)),
                    "cache_diversity": float(1.0 - mean_neighbor_overlap(env)),
                }
            )
            steps += 1
        rows.append(
            {
                "episode": ep + 1,
                "reward": reward_sum,
                "local_hit_rate": local_sum / max(1, steps),
                "neighbor_fetch_rate": neighbor_sum / max(1, steps),
                "cloud_fetch_rate": cloud_sum / max(1, steps),
                "paper_hit_rate": (local_sum + neighbor_sum) / max(1, steps),
            }
        )
    denom = float(max(1, episodes * env.cfg.episode_len))
    return rows, {k: v / denom for k, v in cost_acc.items()}, trace


def eval_policy_object(
    env: NovelRealWorldCachingEnv,
    policy,
    episodes: int,
    seed: int,
) -> tuple[list[dict[str, float]], dict[str, float], list[dict[str, float]]]:
    rows = []
    cost_acc = {"local": 0.0, "neighbor": 0.0, "cloud": 0.0, "replace": 0.0}
    trace: list[dict[str, float]] = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        policy.reset(env)
        done = False
        reward_sum = local_sum = neighbor_sum = cloud_sum = 0.0
        steps = 0
        prev_cache = env.cache_items.copy()
        while not done:
            action = policy.select_items(obs, env)
            replaced = float(np.sum(~np.isin(action, prev_cache)))
            prev_cache = action.copy()
            next_obs, reward, done, info = env.step_full_cache_items(action)
            policy.update(env, obs, action, info)
            reward_sum += float(reward)
            local_sum += float(info["local_hit_rate"])
            neighbor_sum += float(info["neighbor_fetch_rate"])
            cloud_sum += float(info["cloud_fetch_rate"])
            cost_acc["local"] += env.cfg.alpha_local * float(info["local_hit_rate"])
            cost_acc["neighbor"] += env.cfg.beta_neighbor * float(info["neighbor_fetch_rate"])
            cost_acc["cloud"] += env.cfg.chi_cloud * float(info["cloud_fetch_rate"])
            cost_acc["replace"] += env.cfg.delta_replace * (replaced / max(1.0, env.cfg.n_sbs * env.cfg.cache_capacity))
            trace.append(
                {
                    "episode": float(ep + 1),
                    "step": float(steps),
                    "reward": float(reward),
                    "local_hit_rate": float(info["local_hit_rate"]),
                    "neighbor_fetch_rate": float(info["neighbor_fetch_rate"]),
                    "cloud_fetch_rate": float(info["cloud_fetch_rate"]),
                    "paper_hit_rate": float(info["local_hit_rate"] + info["neighbor_fetch_rate"]),
                    "cache_overlap": float(mean_neighbor_overlap(env)),
                    "cache_diversity": float(1.0 - mean_neighbor_overlap(env)),
                }
            )
            steps += 1
            obs = next_obs
        rows.append(
            {
                "episode": ep + 1,
                "reward": reward_sum,
                "local_hit_rate": local_sum / max(1, steps),
                "neighbor_fetch_rate": neighbor_sum / max(1, steps),
                "cloud_fetch_rate": cloud_sum / max(1, steps),
                "paper_hit_rate": (local_sum + neighbor_sum) / max(1, steps),
            }
        )
    denom = float(max(1, episodes * env.cfg.episode_len))
    return rows, {k: v / denom for k, v in cost_acc.items()}, trace


def mean_neighbor_overlap(env: NovelRealWorldCachingEnv) -> float:
    overlaps = []
    for b in range(env.cfg.n_sbs):
        neigh = np.where(env.current_adjacency[b] > 0.0)[0]
        neigh = neigh[neigh != b]
        for n in neigh:
            a = set(env.cache_items[b].tolist())
            c = set(env.cache_items[int(n)].tolist())
            union = len(a | c)
            inter = len(a & c)
            if union > 0:
                overlaps.append(inter / union)
    return float(np.mean(overlaps)) if overlaps else 0.0


def random_action(env, obs):
    rng = np.random.default_rng(123)
    universe = np.arange(1, env.num_items + 1, dtype=np.int64)
    out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
    for b in range(env.cfg.n_sbs):
        out[b] = rng.choice(universe, size=env.cfg.cache_capacity, replace=False)
    return out


def bsg_action(env, obs):
    best = np.argsort(env.global_popularity[1:])[-env.cfg.cache_capacity :][::-1] + 1
    return np.tile(best[None, :], (env.cfg.n_sbs, 1)).astype(np.int64)


class CEpsPolicy:
    def __init__(self, epsilon: float = 0.18) -> None:
        self.epsilon = epsilon
        self.counts = None
        self.rng = np.random.default_rng(777)

    def reset(self, env: NovelRealWorldCachingEnv):
        self.counts = np.zeros((env.cfg.n_sbs, env.num_items + 1), dtype=np.float64)

    def action(self, env: NovelRealWorldCachingEnv, obs):
        if self.counts is None:
            self.reset(env)
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        universe = np.arange(1, env.num_items + 1, dtype=np.int64)
        for b in range(env.cfg.n_sbs):
            if self.rng.random() < self.epsilon:
                out[b] = self.rng.choice(universe, size=env.cfg.cache_capacity, replace=False)
            else:
                best = np.argsort(self.counts[b, 1:])[-env.cfg.cache_capacity :][::-1] + 1
                out[b] = best.astype(np.int64)
        return out

    def update(self, env: NovelRealWorldCachingEnv, association: np.ndarray):
        for ue in range(env.cfg.n_ues):
            if not env.last_active_mask[ue]:
                continue
            item = int(env.last_requests[ue])
            if item > 0:
                self.counts[int(association[ue]), item] += 1.0


def temporal_graph_action_fn(model, device: str, diversity_penalty: float):
    dev = torch.device(device)

    def _fn(env: NovelRealWorldCachingEnv, obs):
        node = _adapt_node_features(obs["node_features"], model.expected_node_dim, env.embed_dim)
        cand = _adapt_candidate_features(obs["candidate_features"], model.expected_candidate_dim, env.embed_dim)
        node_t = torch.as_tensor(node, dtype=torch.float32, device=dev)
        cand_t = torch.as_tensor(cand, dtype=torch.float32, device=dev)
        adj_t = torch.as_tensor(obs["adjacency"], dtype=torch.float32, device=dev)
        mask_t = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=dev)
        logits = model(node_t, cand_t, adj_t, mask_t)
        return logits_to_cache_items(logits, env, diversity_penalty=diversity_penalty)

    return _fn


def plot_lines(x, series: dict[str, list[float]], title: str, ylabel: str, out_path: Path) -> None:
    colors = {
        "Random": "#6c757d",
        "BSG-like": "#8d99ae",
        "C-epsilon-greedy": "#457b9d",
        "LRU": "#577590",
        "LFU": "#4d908e",
        "Thompson": "#7b2cbf",
        "MAAFDRL": "#2a9d8f",
        "TemporalGraph": "#d62828",
    }
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for name, vals in series.items():
        ax.plot(x, vals, marker="o", linewidth=2.0, label=name, color=colors[name])
    ax.set_title(title)
    ax.set_xlabel(out_path.stem.split("_vs_")[-1].replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cost_breakdown(costs: dict[str, dict[str, float]], out_path: Path) -> None:
    names = list(costs.keys())
    local = [costs[n]["local"] for n in names]
    neigh = [costs[n]["neighbor"] for n in names]
    cloud = [costs[n]["cloud"] for n in names]
    repl = [costs[n]["replace"] for n in names]
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
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def patch_burst_sampling(env: NovelRealWorldCachingEnv, start_step: int, end_step: int, burst_prob: float = 0.75):
    if not hasattr(env, "_original_sample_requests"):
        env._original_sample_requests = env._sample_requests
    original = env._original_sample_requests

    def _burst_sample_requests(self, association):
        requests, active = original(association)
        if start_step <= self.step_idx < end_step:
            for b in range(self.cfg.n_sbs):
                burst_item = int(self.current_trend_items[b])
                if burst_item <= 0:
                    continue
                ue_idx = np.where(association == b)[0]
                if ue_idx.size == 0:
                    continue
                mask = self.rng.random(ue_idx.size) < burst_prob
                active[ue_idx[mask]] = True
                requests[ue_idx[mask]] = burst_item
        return requests, active

    env._sample_requests = MethodType(_burst_sample_requests, env)


def burst_trace(env: NovelRealWorldCachingEnv, action_fn, episodes: int, seed: int, burst_window: tuple[int, int], ceps_policy: CEpsPolicy | None = None) -> list[dict[str, float]]:
    all_rows = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        if ceps_policy is not None:
            ceps_policy.reset(env)
        patch_burst_sampling(env, burst_window[0], burst_window[1])
        done = False
        step = 0
        while not done:
            association = obs["association"].copy()
            action = action_fn(env, obs)
            burst_items = env.current_trend_items.copy()
            obs, reward, done, info = env.step_full_cache_items(action)
            if ceps_policy is not None:
                ceps_policy.update(env, association)
            burst_local = 0.0
            burst_total = 0.0
            burst_edge = 0.0
            for ue in range(env.cfg.n_ues):
                if not env.last_active_mask[ue]:
                    continue
                b = int(association[ue])
                item = int(env.last_requests[ue])
                if item != int(burst_items[b]) or item <= 0:
                    continue
                burst_total += 1.0
                if item in env.cache_items[b]:
                    burst_local += 1.0
                    burst_edge += 1.0
                else:
                    neigh = np.where(env.current_adjacency[b] > 0.0)[0]
                    neigh = neigh[neigh != b]
                    if any(item in env.cache_items[int(n)] for n in neigh):
                        burst_edge += 1.0
            all_rows.append(
                {
                    "episode": float(ep + 1),
                    "step": float(step),
                    "burst_local_hit": 0.0 if burst_total <= 0 else burst_local / burst_total,
                    "burst_edge_hit": 0.0 if burst_total <= 0 else burst_edge / burst_total,
                }
            )
            step += 1
        if hasattr(env, "_original_sample_requests"):
            env._sample_requests = env._original_sample_requests
    return all_rows


def burst_trace_policy(env: NovelRealWorldCachingEnv, policy, episodes: int, seed: int, burst_window: tuple[int, int]) -> list[dict[str, float]]:
    all_rows = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        policy.reset(env)
        patch_burst_sampling(env, burst_window[0], burst_window[1])
        done = False
        step = 0
        while not done:
            action = policy.select_items(obs, env)
            burst_items = env.current_trend_items.copy()
            next_obs, reward, done, info = env.step_full_cache_items(action)
            policy.update(env, obs, action, info)
            association = obs["association"].copy()
            burst_local = 0.0
            burst_total = 0.0
            burst_edge = 0.0
            for ue in range(env.cfg.n_ues):
                if not env.last_active_mask[ue]:
                    continue
                b = int(association[ue])
                item = int(env.last_requests[ue])
                if item != int(burst_items[b]) or item <= 0:
                    continue
                burst_total += 1.0
                if item in env.cache_items[b]:
                    burst_local += 1.0
                    burst_edge += 1.0
                else:
                    neigh = np.where(env.current_adjacency[b] > 0.0)[0]
                    neigh = neigh[neigh != b]
                    if any(item in env.cache_items[int(n)] for n in neigh):
                        burst_edge += 1.0
            all_rows.append(
                {
                    "episode": float(ep + 1),
                    "step": float(step),
                    "burst_local_hit": 0.0 if burst_total <= 0 else burst_local / burst_total,
                    "burst_edge_hit": 0.0 if burst_total <= 0 else burst_edge / burst_total,
                }
            )
            step += 1
            obs = next_obs
        if hasattr(env, "_original_sample_requests"):
            env._sample_requests = env._original_sample_requests
    return all_rows


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading histories and temporal model...", flush=True)
    histories, temporal_model, item_genres = load_histories_and_temporal(cfg)

    capacity_rows = []
    for cap in cfg.cache_capacities:
        print(f"Capacity sweep | C={cap}", flush=True)
        env, model = build_env_and_model(cfg, histories, temporal_model, item_genres, n_sbs=cfg.sbs_list[0], cache_capacity=cap)
        random_rows = eval_random(env, cfg.eval_episodes, seed=cfg.seed + 100)
        bsg_rows = eval_bsg(env, cfg.eval_episodes, seed=cfg.seed + 200)
        ceps_rows = eval_c_epsilon_greedy(env, cfg.eval_episodes, seed=cfg.seed + 300, epsilon=cfg.c_epsilon)
        tg_rows = evaluate_temporal_graph(env, model, cfg.eval_episodes, seed=cfg.seed + 400, device=cfg.device, diversity_penalty=cfg.decode_diversity_penalty)
        lru_rows, _, _ = eval_policy_object(env, LRUPolicy(), cfg.eval_episodes, seed=cfg.seed + 450)
        lfu_rows, _, _ = eval_policy_object(env, LFUPolicy(), cfg.eval_episodes, seed=cfg.seed + 475)
        thompson_rows, _, _ = eval_policy_object(env, ThompsonPolicy(seed=cfg.seed + 490), cfg.eval_episodes, seed=cfg.seed + 490)
        maaf_rows, _, _ = eval_policy_object(env, MobilityAwareAsyncFDRLPolicy(), cfg.eval_episodes, seed=cfg.seed + 500)
        for name, rows in [
            ("Random", random_rows),
            ("BSG-like", bsg_rows),
            ("C-epsilon-greedy", ceps_rows),
            ("LRU", lru_rows),
            ("LFU", lfu_rows),
            ("Thompson", thompson_rows),
            ("MAAFDRL", maaf_rows),
            ("TemporalGraph", tg_rows),
        ]:
            if not _allowed_model(name, cfg):
                continue
            summary = summarize_rows(rows)
            capacity_rows.append({"cache_capacity": float(cap), "model": name, **summary})

    sbs_rows = []
    for n_sbs in cfg.sbs_list:
        print(f"SBS sweep | n_sbs={n_sbs}", flush=True)
        env, model = build_env_and_model(cfg, histories, temporal_model, item_genres, n_sbs=n_sbs, cache_capacity=cfg.cache_capacities[1 if len(cfg.cache_capacities) > 1 else 0])
        random_rows = eval_random(env, cfg.eval_episodes, seed=cfg.seed + 500)
        bsg_rows = eval_bsg(env, cfg.eval_episodes, seed=cfg.seed + 600)
        ceps_rows = eval_c_epsilon_greedy(env, cfg.eval_episodes, seed=cfg.seed + 700, epsilon=cfg.c_epsilon)
        tg_rows = evaluate_temporal_graph(env, model, cfg.eval_episodes, seed=cfg.seed + 800, device=cfg.device, diversity_penalty=cfg.decode_diversity_penalty)
        lru_rows, _, _ = eval_policy_object(env, LRUPolicy(), cfg.eval_episodes, seed=cfg.seed + 850)
        lfu_rows, _, _ = eval_policy_object(env, LFUPolicy(), cfg.eval_episodes, seed=cfg.seed + 875)
        thompson_rows, _, _ = eval_policy_object(env, ThompsonPolicy(seed=cfg.seed + 890), cfg.eval_episodes, seed=cfg.seed + 890)
        maaf_rows, _, _ = eval_policy_object(env, MobilityAwareAsyncFDRLPolicy(), cfg.eval_episodes, seed=cfg.seed + 900)
        for name, rows in [
            ("Random", random_rows),
            ("BSG-like", bsg_rows),
            ("C-epsilon-greedy", ceps_rows),
            ("LRU", lru_rows),
            ("LFU", lfu_rows),
            ("Thompson", thompson_rows),
            ("MAAFDRL", maaf_rows),
            ("TemporalGraph", tg_rows),
        ]:
            if not _allowed_model(name, cfg):
                continue
            summary = summarize_rows(rows)
            sbs_rows.append({"n_sbs": float(n_sbs), "model": name, **summary})

    print("Collecting trace metrics for cost/overlap...", flush=True)
    env, model = build_env_and_model(cfg, histories, temporal_model, item_genres, n_sbs=cfg.sbs_list[0], cache_capacity=cfg.cache_capacities[1 if len(cfg.cache_capacities) > 1 else 0])
    ceps_policy = CEpsPolicy(epsilon=cfg.c_epsilon)
    _, random_costs, random_trace = eval_action_policy(env, random_action, cfg.eval_episodes, seed=cfg.seed + 900)
    _, bsg_costs, bsg_trace = eval_action_policy(env, bsg_action, cfg.eval_episodes, seed=cfg.seed + 1000)
    _, ceps_costs, ceps_trace = eval_action_policy(env, ceps_policy.action, cfg.eval_episodes, seed=cfg.seed + 1100)
    _, tg_costs, tg_trace = eval_action_policy(env, temporal_graph_action_fn(model, cfg.device, cfg.decode_diversity_penalty), cfg.eval_episodes, seed=cfg.seed + 1200)
    _, lru_costs, lru_trace = eval_policy_object(env, LRUPolicy(), cfg.eval_episodes, seed=cfg.seed + 1250)
    _, lfu_costs, lfu_trace = eval_policy_object(env, LFUPolicy(), cfg.eval_episodes, seed=cfg.seed + 1275)
    _, thompson_costs, thompson_trace = eval_policy_object(env, ThompsonPolicy(seed=cfg.seed + 1285), cfg.eval_episodes, seed=cfg.seed + 1285)
    _, maaf_costs, maaf_trace = eval_policy_object(env, MobilityAwareAsyncFDRLPolicy(), cfg.eval_episodes, seed=cfg.seed + 1295)

    print("Collecting burst-adaptation traces...", flush=True)
    burst_window = (cfg.episode_len // 3, 2 * cfg.episode_len // 3)
    burst_env, burst_model = build_env_and_model(cfg, histories, temporal_model, item_genres, n_sbs=cfg.sbs_list[0], cache_capacity=cfg.cache_capacities[1 if len(cfg.cache_capacities) > 1 else 0])
    burst_random = burst_trace(burst_env, random_action, 1, cfg.seed + 1300, burst_window)
    burst_bsg = burst_trace(burst_env, bsg_action, 1, cfg.seed + 1400, burst_window)
    ceps_burst_policy = CEpsPolicy(epsilon=cfg.c_epsilon)
    burst_ceps = burst_trace(burst_env, ceps_burst_policy.action, 1, cfg.seed + 1500, burst_window, ceps_policy=ceps_burst_policy)
    burst_tg = burst_trace(burst_env, temporal_graph_action_fn(burst_model, cfg.device, cfg.decode_diversity_penalty), 1, cfg.seed + 1600, burst_window)
    burst_lru = burst_trace_policy(burst_env, LRUPolicy(), 1, cfg.seed + 1650, burst_window)
    burst_lfu = burst_trace_policy(burst_env, LFUPolicy(), 1, cfg.seed + 1675, burst_window)
    burst_thompson = burst_trace_policy(burst_env, ThompsonPolicy(seed=cfg.seed + 1685), 1, cfg.seed + 1685, burst_window)
    burst_maaf = burst_trace_policy(burst_env, MobilityAwareAsyncFDRLPolicy(), 1, cfg.seed + 1695, burst_window)

    cost_rows = [
        {"model": "Random", **random_costs},
        {"model": "BSG-like", **bsg_costs},
        {"model": "C-epsilon-greedy", **ceps_costs},
        {"model": "LRU", **lru_costs},
        {"model": "LFU", **lfu_costs},
        {"model": "Thompson", **thompson_costs},
        {"model": "MAAFDRL", **maaf_costs},
        {"model": "TemporalGraph", **tg_costs},
    ]
    cost_rows = [row for row in cost_rows if _allowed_model(str(row["model"]), cfg)]

    print("Saving CSVs and plots...", flush=True)
    # Save CSVs.
    for path, rows in [
        (cfg.output_dir / "capacity_sweep.csv", capacity_rows),
        (cfg.output_dir / "sbs_sweep.csv", sbs_rows),
        (cfg.output_dir / "cost_summary.csv", cost_rows),
        (cfg.output_dir / "random_trace.csv", random_trace),
        (cfg.output_dir / "bsg_trace.csv", bsg_trace),
        (cfg.output_dir / "c_epsilon_trace.csv", ceps_trace),
        (cfg.output_dir / "lru_trace.csv", lru_trace),
        (cfg.output_dir / "lfu_trace.csv", lfu_trace),
        (cfg.output_dir / "thompson_trace.csv", thompson_trace),
        (cfg.output_dir / "maafdrl_trace.csv", maaf_trace),
        (cfg.output_dir / "temporal_graph_trace.csv", tg_trace),
        (cfg.output_dir / "burst_random.csv", burst_random),
        (cfg.output_dir / "burst_bsg.csv", burst_bsg),
        (cfg.output_dir / "burst_c_epsilon.csv", burst_ceps),
        (cfg.output_dir / "burst_lru.csv", burst_lru),
        (cfg.output_dir / "burst_lfu.csv", burst_lfu),
        (cfg.output_dir / "burst_thompson.csv", burst_thompson),
        (cfg.output_dir / "burst_maafdrl.csv", burst_maaf),
        (cfg.output_dir / "burst_temporal_graph.csv", burst_tg),
    ]:
        if rows:
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    # Capacity plots.
    for metric, ylabel, filename in [
        ("paper_hit_mean", "Mean Paper-Hit Rate", "paper_hit_vs_cache_capacity.png"),
        ("reward_mean", "Mean Reward", "reward_vs_cache_capacity.png"),
    ]:
        series = {}
        x = cfg.cache_capacities
        for model_name in [name for name in ["Random", "BSG-like", "C-epsilon-greedy", "LRU", "LFU", "Thompson", "MAAFDRL", "TemporalGraph"] if _allowed_model(name, cfg)]:
            series[model_name] = [next(r[metric] for r in capacity_rows if r["model"] == model_name and int(r["cache_capacity"]) == cap) for cap in x]
        plot_lines(x, series, filename.replace(".png", "").replace("_", " ").title(), ylabel, cfg.output_dir / filename)

    # SBS plots.
    for metric, ylabel, filename in [
        ("paper_hit_mean", "Mean Paper-Hit Rate", "paper_hit_vs_n_sbs.png"),
        ("reward_mean", "Mean Reward", "reward_vs_n_sbs.png"),
    ]:
        series = {}
        x = cfg.sbs_list
        for model_name in [name for name in ["Random", "BSG-like", "C-epsilon-greedy", "LRU", "LFU", "Thompson", "MAAFDRL", "TemporalGraph"] if _allowed_model(name, cfg)]:
            series[model_name] = [next(r[metric] for r in sbs_rows if r["model"] == model_name and int(r["n_sbs"]) == n_sbs) for n_sbs in x]
        plot_lines(x, series, filename.replace(".png", "").replace("_", " ").title(), ylabel, cfg.output_dir / filename)

    # Cost breakdown.
    plot_cost_breakdown(
        {
            "Random": random_costs,
            "BSG-like": bsg_costs,
            "C-epsilon-greedy": ceps_costs,
            "LRU": lru_costs,
            "LFU": lfu_costs,
            "Thompson": thompson_costs,
            "MAAFDRL": maaf_costs,
            "TemporalGraph": tg_costs,
        } if not cfg.include_models and not cfg.exclude_models else {
            name: costs
            for name, costs in {
                "Random": random_costs,
                "BSG-like": bsg_costs,
                "C-epsilon-greedy": ceps_costs,
                "LRU": lru_costs,
                "LFU": lfu_costs,
                "Thompson": thompson_costs,
                "MAAFDRL": maaf_costs,
                "TemporalGraph": tg_costs,
            }.items()
            if _allowed_model(name, cfg)
        },
        cfg.output_dir / "cost_breakdown.png",
    )

    # Burst adaptation.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for label, rows, color in [
        ("Random", burst_random, "#6c757d"),
        ("BSG-like", burst_bsg, "#8d99ae"),
        ("C-epsilon-greedy", burst_ceps, "#457b9d"),
        ("LRU", burst_lru, "#577590"),
        ("LFU", burst_lfu, "#4d908e"),
        ("Thompson", burst_thompson, "#7b2cbf"),
        ("MAAFDRL", burst_maaf, "#2a9d8f"),
        ("TemporalGraph", burst_tg, "#d62828"),
    ]:
        if not _allowed_model(label, cfg):
            continue
        steps = [int(r["step"]) for r in rows]
        axes[0].plot(steps, [r["burst_local_hit"] for r in rows], label=label, color=color, linewidth=2.0)
        axes[1].plot(steps, [r["burst_edge_hit"] for r in rows], label=label, color=color, linewidth=2.0)
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
    fig.savefig(cfg.output_dir / "burst_adaptation.png", dpi=180)
    plt.close(fig)

    # Overlap/diversity.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for label, rows, color in [
        ("Random", random_trace, "#6c757d"),
        ("BSG-like", bsg_trace, "#8d99ae"),
        ("C-epsilon-greedy", ceps_trace, "#457b9d"),
        ("LRU", lru_trace, "#577590"),
        ("LFU", lfu_trace, "#4d908e"),
        ("Thompson", thompson_trace, "#7b2cbf"),
        ("MAAFDRL", maaf_trace, "#2a9d8f"),
        ("TemporalGraph", tg_trace, "#d62828"),
    ]:
        if not _allowed_model(label, cfg):
            continue
        steps = np.arange(len(rows))
        axes[0].plot(steps, [r["cache_overlap"] for r in rows], label=label, color=color, linewidth=2.0)
        axes[1].plot(steps, [r["cache_diversity"] for r in rows], label=label, color=color, linewidth=2.0)
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
    fig.savefig(cfg.output_dir / "cache_overlap_diversity.png", dpi=180)
    plt.close(fig)

    print(f"Saved novel comparison bundle to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
