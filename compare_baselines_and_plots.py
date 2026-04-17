from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from movie_edge_sim.cooperative_env import CooperativeCachingEnv, EnvConfig
from movie_edge_sim.data import download_movielens_100k, load_ratings
from movie_edge_sim.gnn_actor_critic import GNNActorCritic
from movie_edge_sim.temporal_federated import TemporalSpikeEncoder
from movie_edge_sim.temporal_requests import build_user_histories


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run baseline algorithms and generate paper-style comparison plots for cooperative edge caching."
        )
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--temporal-checkpoint", type=Path, required=True)
    p.add_argument("--gnn-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_comparison"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-every-episode", type=int, default=1)

    p.add_argument("--n-sbs", type=int, default=8)
    p.add_argument("--n-ues", type=int, default=220)
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--grid-size", type=float, default=300.0)
    p.add_argument("--episode-len", type=int, default=120)
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--fp", type=int, default=50)
    p.add_argument("--cache-capacities", nargs="+", type=int, default=[10, 20, 30])
    p.add_argument("--sbs-list", nargs="+", type=int, default=[1, 2, 4, 8])

    p.add_argument("--epsilon", type=float, default=0.1, help="epsilon for C-epsilon-greedy baseline.")
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--gnn-hidden-dim", type=int, default=128)
    return p.parse_args()


def setup_logging(level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    return logging.getLogger("baseline_compare")


def _load_temporal_model(
    ckpt: Path,
    num_items: int,
    window_size: int,
    embed_dim: int,
    hidden_dim: int,
    device: str,
) -> TemporalSpikeEncoder:
    model = TemporalSpikeEncoder(
        num_items=num_items,
        window_size=window_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _step_requests(env: CooperativeCachingEnv, association: np.ndarray) -> list[dict[int, int]]:
    demand = [dict() for _ in range(env.cfg.n_sbs)]
    for ue in range(env.cfg.n_ues):
        b = int(association[ue])
        user_id = int(env.user_ids_for_ues[ue])
        ptr = int(env.user_ptrs[ue])
        item = int(env.user_histories[user_id][ptr])
        demand[b][item] = demand[b].get(item, 0) + 1
    return demand


def _valid_action_indices(mask_row: np.ndarray) -> np.ndarray:
    idx = np.where(mask_row > 0)[0]
    if idx.size == 0:
        return np.asarray([0], dtype=np.int64)
    return idx.astype(np.int64)


class BasePolicy:
    name = "base"
    action_mode = "candidate_index"

    def reset(self, env: CooperativeCachingEnv) -> None:
        return

    def select_action(self, obs: dict[str, np.ndarray], env: CooperativeCachingEnv) -> np.ndarray:
        raise NotImplementedError

    def update(
        self,
        obs: dict[str, np.ndarray],
        env: CooperativeCachingEnv,
        action: np.ndarray,
        demand: list[dict[int, int]],
    ) -> None:
        return


class RandomPolicy(BasePolicy):
    name = "Random"
    action_mode = "full_cache_items"

    def __init__(self, num_items: int, seed: int = 42) -> None:
        self.num_items = num_items
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: dict[str, np.ndarray], env: CooperativeCachingEnv) -> np.ndarray:
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        for b in range(env.cfg.n_sbs):
            out[b] = self.rng.choice(
                np.arange(1, self.num_items + 1, dtype=np.int64),
                size=env.cfg.cache_capacity,
                replace=False,
            )
        return out


class CEpsilonGreedyPolicy(BasePolicy):
    name = "C-epsilon-greedy"
    action_mode = "full_cache_items"

    def __init__(self, num_items: int, epsilon: float = 0.3, seed: int = 42) -> None:
        self.num_items = num_items
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.counts: np.ndarray | None = None  # (B, num_items+1)

    def reset(self, env: CooperativeCachingEnv) -> None:
        self.counts = np.zeros((env.cfg.n_sbs, self.num_items + 1), dtype=np.float64)

    def select_action(self, obs: dict[str, np.ndarray], env: CooperativeCachingEnv) -> np.ndarray:
        assert self.counts is not None
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        universe = np.arange(1, self.num_items + 1, dtype=np.int64)
        for b in range(env.cfg.n_sbs):
            if self.rng.random() < self.epsilon:
                out[b] = self.rng.choice(universe, size=env.cfg.cache_capacity, replace=False)
                continue
            scores = self.counts[b, 1:]
            best = np.argsort(scores)[-env.cfg.cache_capacity :][::-1] + 1
            out[b] = best.astype(np.int64)
        return out

    def update(self, obs: dict[str, np.ndarray], env: CooperativeCachingEnv, action: np.ndarray, demand: list[dict[int, int]]) -> None:
        assert self.counts is not None
        for b in range(env.cfg.n_sbs):
            for item, c in demand[b].items():
                if 0 < item <= self.num_items:
                    self.counts[b, item] += float(c)


class ThompsonPolicy(BasePolicy):
    name = "Thompson"
    action_mode = "full_cache_items"

    def __init__(self, num_items: int, seed: int = 42) -> None:
        self.num_items = num_items
        self.rng = np.random.default_rng(seed)
        self.alpha: np.ndarray | None = None
        self.beta: np.ndarray | None = None

    def reset(self, env: CooperativeCachingEnv) -> None:
        self.alpha = np.ones((env.cfg.n_sbs, self.num_items + 1), dtype=np.float64)
        self.beta = np.ones((env.cfg.n_sbs, self.num_items + 1), dtype=np.float64)

    def select_action(self, obs: dict[str, np.ndarray], env: CooperativeCachingEnv) -> np.ndarray:
        assert self.alpha is not None and self.beta is not None
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        for b in range(env.cfg.n_sbs):
            cand_items = np.arange(1, self.num_items + 1, dtype=np.int64)
            a = self.alpha[b, cand_items]
            bb = self.beta[b, cand_items]
            sampled = self.rng.beta(a, bb)
            best = cand_items[np.argsort(sampled)[-env.cfg.cache_capacity :][::-1]]
            out[b] = best.astype(np.int64)
        return out

    def update(self, obs: dict[str, np.ndarray], env: CooperativeCachingEnv, action: np.ndarray, demand: list[dict[int, int]]) -> None:
        assert self.alpha is not None and self.beta is not None
        for b in range(env.cfg.n_sbs):
            for item in action[b]:
                item = int(item)
                if item <= 0:
                    continue
                if demand[b].get(item, 0) > 0:
                    self.alpha[b, item] += 1.0
                else:
                    self.beta[b, item] += 1.0


class BSGPolicy(BasePolicy):
    name = "BSG-like"
    action_mode = "full_cache_items"

    def select_action(self, obs: dict[str, np.ndarray], env: CooperativeCachingEnv) -> np.ndarray:
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        best = np.argsort(env.global_popularity[1:])[-env.cfg.cache_capacity :][::-1] + 1
        for b in range(env.cfg.n_sbs):
            out[b] = best.astype(np.int64)
        return out


class EFNRLPolicy(BasePolicy):
    name = "EFNRL-like"
    action_mode = "full_cache_items"

    def select_action(self, obs: dict[str, np.ndarray], env: CooperativeCachingEnv) -> np.ndarray:
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        for b in range(env.cfg.n_sbs):
            valid = _valid_action_indices(obs["action_mask"][b])
            scores = env.current_candidate_scores[b, valid]
            topk = min(env.cfg.cache_capacity, valid.shape[0])
            order = valid[np.argsort(scores)[-topk :][::-1]]
            out[b] = env.current_candidates[b, order].astype(np.int64)
        return out


class GNNPolicy(BasePolicy):
    name = "GNN-ActorCritic"
    action_mode = "full_cache_items"

    def __init__(
        self,
        ckpt: Path,
        node_feat_dim: int,
        candidate_feat_dim: int,
        hidden_dim: int,
        fp: int,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model = GNNActorCritic(
            node_feat_dim=node_feat_dim,
            candidate_feat_dim=candidate_feat_dim,
            hidden_dim=hidden_dim,
            fp=fp,
        ).to(device)
        self.model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        self.model.eval()

    @torch.no_grad()
    def select_action(self, obs: dict[str, np.ndarray], env: CooperativeCachingEnv) -> np.ndarray:
        node = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=self.device)
        cand = torch.as_tensor(obs["candidate_features"], dtype=torch.float32, device=self.device)
        adj = torch.as_tensor(obs["adjacency"], dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=self.device)
        logits, _ = self.model(node, cand, adj, mask)
        topk = min(env.cfg.cache_capacity, logits.shape[1])
        action_idx = torch.topk(logits, k=topk, dim=-1).indices.detach().cpu().numpy().astype(np.int64)
        return env.candidate_indices_to_items(action_idx, k=topk)


@dataclass
class EvalResult:
    scheme: str
    cache_capacity: int
    n_sbs: int
    reward_mean: float
    cost_mean: float
    local_hit_mean: float
    neighbor_fetch_mean: float
    cloud_fetch_mean: float
    paper_hit_mean: float


def evaluate_policy(
    policy: BasePolicy,
    env_cfg: EnvConfig,
    temporal_model: TemporalSpikeEncoder,
    histories: dict[int, list[int]],
    eval_episodes: int,
    seed: int,
    logger: logging.Logger | None = None,
    log_every_episode: int = 1,
    log_prefix: str = "",
) -> tuple[EvalResult, list[dict[str, float]]]:
    env = CooperativeCachingEnv(env_cfg, temporal_model, histories)
    policy.reset(env)
    if logger is not None:
        logger.info(
            "%sEvaluating %s | n_sbs=%d C=%d episodes=%d",
            log_prefix,
            policy.name,
            env_cfg.n_sbs,
            env_cfg.cache_capacity,
            eval_episodes,
        )

    ep_rows: list[dict[str, float]] = []
    for ep in range(eval_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        reward_sum = 0.0
        local_sum = 0.0
        neighbor_sum = 0.0
        cloud_sum = 0.0
        steps = 0

        while not done:
            demand = _step_requests(env, obs["association"])
            action = policy.select_action(obs, env)
            if policy.action_mode == "item_id":
                next_obs, reward, done, info = env.step_items(action)
            elif policy.action_mode == "full_cache_items":
                next_obs, reward, done, info = env.step_full_cache_items(action)
            else:
                next_obs, reward, done, info = env.step(action)
            policy.update(obs, env, action, demand)

            reward_sum += float(reward)
            local_sum += float(info["local_hit_rate"])
            neighbor_sum += float(info["neighbor_fetch_rate"])
            cloud_sum += float(info["cloud_fetch_rate"])
            steps += 1
            obs = next_obs

        baseline_cost = env_cfg.chi_cloud * env_cfg.n_ues * steps
        actual_cost = baseline_cost - reward_sum
        local = local_sum / max(1, steps)
        neighbor = neighbor_sum / max(1, steps)
        cloud = cloud_sum / max(1, steps)
        ep_rows.append(
            {
                "episode": ep + 1,
                "reward": reward_sum,
                "cost": actual_cost,
                "local_hit_rate": local,
                "neighbor_fetch_rate": neighbor,
                "cloud_fetch_rate": cloud,
                "paper_style_hit_rate": local + neighbor,
            }
        )
        if logger is not None and ((ep + 1) % max(1, log_every_episode) == 0 or (ep + 1) == eval_episodes):
            logger.info(
                "%s%s episode %d/%d | reward=%.4f cost=%.4f local=%.4f neighbor=%.4f cloud=%.4f",
                log_prefix,
                policy.name,
                ep + 1,
                eval_episodes,
                reward_sum,
                actual_cost,
                local,
                neighbor,
                cloud,
            )

    rewards = np.asarray([r["reward"] for r in ep_rows], dtype=np.float64)
    costs = np.asarray([r["cost"] for r in ep_rows], dtype=np.float64)
    local = np.asarray([r["local_hit_rate"] for r in ep_rows], dtype=np.float64)
    neighbor = np.asarray([r["neighbor_fetch_rate"] for r in ep_rows], dtype=np.float64)
    cloud = np.asarray([r["cloud_fetch_rate"] for r in ep_rows], dtype=np.float64)
    paper_hit = np.asarray([r["paper_style_hit_rate"] for r in ep_rows], dtype=np.float64)

    result = EvalResult(
        scheme=policy.name,
        cache_capacity=env_cfg.cache_capacity,
        n_sbs=env_cfg.n_sbs,
        reward_mean=float(rewards.mean()),
        cost_mean=float(costs.mean()),
        local_hit_mean=float(local.mean()),
        neighbor_fetch_mean=float(neighbor.mean()),
        cloud_fetch_mean=float(cloud.mean()),
        paper_hit_mean=float(paper_hit.mean()),
    )
    return result, ep_rows


def save_results(out_dir: Path, results: list[EvalResult], episode_rows: list[dict[str, object]]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_csv = out_dir / "scheme_capacity_summary.csv"
    ep_csv = out_dir / "episode_metrics.csv"

    with agg_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scheme",
                "cache_capacity",
                "n_sbs",
                "reward_mean",
                "cost_mean",
                "local_hit_mean",
                "neighbor_fetch_mean",
                "cloud_fetch_mean",
                "paper_hit_mean",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.scheme,
                    r.cache_capacity,
                    r.n_sbs,
                    f"{r.reward_mean:.8f}",
                    f"{r.cost_mean:.8f}",
                    f"{r.local_hit_mean:.8f}",
                    f"{r.neighbor_fetch_mean:.8f}",
                    f"{r.cloud_fetch_mean:.8f}",
                    f"{r.paper_hit_mean:.8f}",
                ]
            )

    with ep_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group",
                "scheme",
                "cache_capacity",
                "n_sbs",
                "episode",
                "reward",
                "cost",
                "local_hit_rate",
                "neighbor_fetch_rate",
                "cloud_fetch_rate",
                "paper_style_hit_rate",
            ]
        )
        for row in episode_rows:
            writer.writerow(
                [
                    row["group"],
                    row["scheme"],
                    row["cache_capacity"],
                    row["n_sbs"],
                    row["episode"],
                    f"{float(row['reward']):.8f}",
                    f"{float(row['cost']):.8f}",
                    f"{float(row['local_hit_rate']):.8f}",
                    f"{float(row['neighbor_fetch_rate']):.8f}",
                    f"{float(row['cloud_fetch_rate']):.8f}",
                    f"{float(row['paper_style_hit_rate']):.8f}",
                ]
            )
    return agg_csv, ep_csv


def _plot_fig7(results: list[EvalResult], out_dir: Path) -> None:
    # Fig7-like: performance vs cache capacity under different schemes.
    schemes = sorted({r.scheme for r in results if r.n_sbs == max(x.n_sbs for x in results)})
    capacities = sorted({r.cache_capacity for r in results})

    fig, ax = plt.subplots(figsize=(9, 5))
    for s in schemes:
        ys = []
        for c in capacities:
            vals = [r.cost_mean for r in results if r.scheme == s and r.cache_capacity == c and r.n_sbs == max(x.n_sbs for x in results)]
            ys.append(vals[0] if vals else np.nan)
        ax.plot(capacities, ys, marker="o", linewidth=2, label=s)
    ax.set_title("Fig7-like: Cost vs Cache Capacity")
    ax.set_xlabel("Cache Capacity (C)")
    ax.set_ylabel("Cost")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_like_cost_vs_capacity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for s in schemes:
        ys = []
        for c in capacities:
            vals = [r.paper_hit_mean for r in results if r.scheme == s and r.cache_capacity == c and r.n_sbs == max(x.n_sbs for x in results)]
            ys.append(vals[0] if vals else np.nan)
        ax.plot(capacities, ys, marker="o", linewidth=2, label=s)
    ax.set_title("Fig7-like: Cache Hit Ratio vs Cache Capacity")
    ax.set_xlabel("Cache Capacity (C)")
    ax.set_ylabel("Cache Hit Ratio (local+neighbor)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_like_hit_vs_capacity.png", dpi=180)
    plt.close(fig)


def _plot_fig8(episode_rows: list[dict[str, object]], out_dir: Path) -> None:
    # Fig8-like: testing performance vs episode for different cache capacities (GNN scheme).
    rows = [r for r in episode_rows if r["group"] == "fig8"]
    capacities = sorted({int(r["cache_capacity"]) for r in rows})

    def plot_metric(metric: str, title: str, ylab: str, fname: str, ylim: tuple[float, float] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for c in capacities:
            rr = [r for r in rows if int(r["cache_capacity"]) == c]
            rr = sorted(rr, key=lambda x: int(x["episode"]))
            xs = [int(r["episode"]) for r in rr]
            ys = [float(r[metric]) for r in rr]
            ax.plot(xs, ys, linewidth=2, label=f"C={c}")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylab)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=180)
        plt.close(fig)

    plot_metric("cost", "Fig8-like: Cost vs Episode (different C)", "Cost", "fig8_like_cost_vs_episode.png")
    plot_metric("reward", "Fig8-like: Reward vs Episode (different C)", "Reward", "fig8_like_reward_vs_episode.png")
    plot_metric(
        "paper_style_hit_rate",
        "Fig8-like: Cache Hit Ratio vs Episode (different C)",
        "Cache Hit Ratio (local+neighbor)",
        "fig8_like_hit_vs_episode.png",
        ylim=(0.0, 1.0),
    )


def _plot_fig9(results: list[EvalResult], out_dir: Path) -> None:
    # Fig9-like: performance vs cache capacity for different number of SBSs (GNN scheme).
    rows = [r for r in results if r.scheme == "GNN-ActorCritic"]
    sbs_list = sorted({r.n_sbs for r in rows})
    capacities = sorted({r.cache_capacity for r in rows})

    def plot_metric(metric: str, title: str, ylab: str, fname: str, ylim: tuple[float, float] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for b in sbs_list:
            ys = []
            for c in capacities:
                vals = [getattr(r, metric) for r in rows if r.n_sbs == b and r.cache_capacity == c]
                ys.append(vals[0] if vals else np.nan)
            ax.plot(capacities, ys, marker="o", linewidth=2, label=f"SBS={b}")
        ax.set_title(title)
        ax.set_xlabel("Cache Capacity (C)")
        ax.set_ylabel(ylab)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=180)
        plt.close(fig)

    plot_metric("cost_mean", "Fig9-like: Cost vs Cache Capacity (different SBS count)", "Cost", "fig9_like_cost_vs_capacity.png")
    plot_metric(
        "reward_mean",
        "Fig9-like: Reward vs Cache Capacity (different SBS count)",
        "Reward",
        "fig9_like_reward_vs_capacity.png",
    )
    plot_metric(
        "paper_hit_mean",
        "Fig9-like: Cache Hit Ratio vs Cache Capacity (different SBS count)",
        "Cache Hit Ratio (local+neighbor)",
        "fig9_like_hit_vs_capacity.png",
        ylim=(0.0, 1.0),
    )


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    logger.info("Baseline comparison pipeline started.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info("Seeds initialized with seed=%d", args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", args.output_dir)

    logger.info("Stage 1/4: loading dataset and temporal model")
    dataset_dir = download_movielens_100k(args.data_root)
    ratings = load_ratings(dataset_dir)
    histories = build_user_histories(ratings)
    num_items = max(row["item_id"] for row in ratings)
    logger.info("Dataset ready | dir=%s ratings=%d users=%d items=%d", dataset_dir, len(ratings), len(histories), num_items)

    temporal_model = _load_temporal_model(
        ckpt=args.temporal_checkpoint,
        num_items=num_items,
        window_size=args.window_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    logger.info("Temporal checkpoint loaded: %s", args.temporal_checkpoint)

    results: list[EvalResult] = []
    episode_rows: list[dict[str, object]] = []

    # Figure 7-like: compare schemes vs cache capacity (fixed n_sbs).
    logger.info("Stage 2/4: evaluating Fig7-like scheme comparison across cache capacities")
    for c in args.cache_capacities:
        logger.info("Fig7-like | capacity C=%d", c)
        env_cfg = EnvConfig(
            n_sbs=args.n_sbs,
            n_ues=args.n_ues,
            cache_capacity=c,
            fp=args.fp,
            window_size=args.window_size,
            episode_len=args.episode_len,
            grid_size=args.grid_size,
            seed=args.seed,
        )

        probe_env = CooperativeCachingEnv(env_cfg, temporal_model, histories)
        probe_obs = probe_env.reset(seed=args.seed)
        node_feat_dim = int(probe_obs["node_features"].shape[1])
        candidate_feat_dim = int(probe_obs["candidate_features"].shape[2])

        policies: list[BasePolicy] = [
            RandomPolicy(num_items=num_items, seed=args.seed),
            CEpsilonGreedyPolicy(num_items=num_items, epsilon=args.epsilon, seed=args.seed),
            ThompsonPolicy(num_items=num_items, seed=args.seed),
            BSGPolicy(),
            EFNRLPolicy(),
            GNNPolicy(
                ckpt=args.gnn_checkpoint,
                node_feat_dim=node_feat_dim,
                candidate_feat_dim=candidate_feat_dim,
                hidden_dim=args.gnn_hidden_dim,
                fp=args.fp,
                device=args.device,
            ),
        ]

        for policy in policies:
            res, ep = evaluate_policy(
                policy=policy,
                env_cfg=env_cfg,
                temporal_model=temporal_model,
                histories=histories,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
                logger=logger,
                log_every_episode=args.log_every_episode,
                log_prefix=f"[Fig7 C={c}] ",
            )
            results.append(res)
            for row in ep:
                row.update({"group": "fig7", "scheme": policy.name, "cache_capacity": c, "n_sbs": args.n_sbs})
            episode_rows.extend(ep)

    # Figure 8-like: episode-wise curves for GNN scheme under different capacities.
    logger.info("Stage 3/4: evaluating Fig8-like GNN testing curves across cache capacities")
    for c in args.cache_capacities:
        logger.info("Fig8-like | capacity C=%d", c)
        env_cfg = EnvConfig(
            n_sbs=args.n_sbs,
            n_ues=args.n_ues,
            cache_capacity=c,
            fp=args.fp,
            window_size=args.window_size,
            episode_len=args.episode_len,
            grid_size=args.grid_size,
            seed=args.seed,
        )
        probe_env = CooperativeCachingEnv(env_cfg, temporal_model, histories)
        probe_obs = probe_env.reset(seed=args.seed)
        node_feat_dim = int(probe_obs["node_features"].shape[1])
        candidate_feat_dim = int(probe_obs["candidate_features"].shape[2])
        gnn = GNNPolicy(
            ckpt=args.gnn_checkpoint,
            node_feat_dim=node_feat_dim,
            candidate_feat_dim=candidate_feat_dim,
            hidden_dim=args.gnn_hidden_dim,
            fp=args.fp,
            device=args.device,
        )
        _, ep = evaluate_policy(
            policy=gnn,
            env_cfg=env_cfg,
            temporal_model=temporal_model,
            histories=histories,
            eval_episodes=args.eval_episodes,
            seed=args.seed + 1000,
            logger=logger,
            log_every_episode=args.log_every_episode,
            log_prefix=f"[Fig8 C={c}] ",
        )
        for row in ep:
            row.update({"group": "fig8", "scheme": gnn.name, "cache_capacity": c, "n_sbs": args.n_sbs})
        episode_rows.extend(ep)

    # Figure 9-like: GNN scheme for different SBS counts vs capacity.
    logger.info("Stage 4/4: evaluating Fig9-like GNN across SBS counts and capacities")
    for b in args.sbs_list:
        logger.info("Fig9-like | n_sbs=%d", b)
        for c in args.cache_capacities:
            logger.info("Fig9-like | n_sbs=%d capacity C=%d", b, c)
            env_cfg = EnvConfig(
                n_sbs=b,
                n_ues=args.n_ues,
                cache_capacity=c,
                fp=args.fp,
                window_size=args.window_size,
                episode_len=args.episode_len,
                grid_size=args.grid_size,
                seed=args.seed,
            )
            probe_env = CooperativeCachingEnv(env_cfg, temporal_model, histories)
            probe_obs = probe_env.reset(seed=args.seed)
            node_feat_dim = int(probe_obs["node_features"].shape[1])
            candidate_feat_dim = int(probe_obs["candidate_features"].shape[2])
            gnn = GNNPolicy(
                ckpt=args.gnn_checkpoint,
                node_feat_dim=node_feat_dim,
                candidate_feat_dim=candidate_feat_dim,
                hidden_dim=args.gnn_hidden_dim,
                fp=args.fp,
                device=args.device,
            )
            res, ep = evaluate_policy(
                policy=gnn,
                env_cfg=env_cfg,
                temporal_model=temporal_model,
                histories=histories,
                eval_episodes=args.eval_episodes,
                seed=args.seed + 2000 + b * 100 + c,
                logger=logger,
                log_every_episode=args.log_every_episode,
                log_prefix=f"[Fig9 SBS={b} C={c}] ",
            )
            results.append(res)
            for row in ep:
                row.update({"group": "fig9", "scheme": gnn.name, "cache_capacity": c, "n_sbs": b})
            episode_rows.extend(ep)

    agg_csv, ep_csv = save_results(args.output_dir, results, episode_rows)
    _plot_fig7(results, args.output_dir)
    _plot_fig8(episode_rows, args.output_dir)
    _plot_fig9(results, args.output_dir)
    logger.info("Saved aggregate metrics: %s", agg_csv)
    logger.info("Saved episode metrics: %s", ep_csv)
    logger.info("Saved plots under: %s", args.output_dir)
    logger.info("Baseline comparison pipeline finished.")

    print(f"Saved aggregate metrics: {agg_csv}")
    print(f"Saved episode metrics: {ep_csv}")
    print(f"Saved plots under: {args.output_dir}")


if __name__ == "__main__":
    main()
