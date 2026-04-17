from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

_mpl_dir = Path("outputs/.mplconfig")
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from movie_edge_sim.data import get_movielens_dataset, load_ratings_auto
from movie_edge_sim.maddpg_cache import MADDPGCachePolicy, MADDPGConfig
from movie_edge_sim.paper_cefmr_env import PaperCEFMRCooperativeEnv, PaperEnvConfig
from movie_edge_sim.temporal_requests import build_user_histories


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare paper-style CEFMR baselines and generate plots.")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--dataset-name", type=str, default="ml-1m", choices=["ml-100k", "ml-1m"])
    p.add_argument("--cefmr-aae", type=Path, required=True)
    p.add_argument("--cefmr-maddpg", type=Path, required=True)
    p.add_argument("--tfmadrl-aae", type=Path, default=None)
    p.add_argument("--tfmadrl-maddpg", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/paper_cefmr_compare"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-every-episode", type=int, default=1)

    p.add_argument("--n-ues", type=int, default=220)
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--grid-size", type=float, default=300.0)
    p.add_argument("--episode-len", type=int, default=120)
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--fp", type=int, default=50)
    p.add_argument("--cache-capacities", nargs="+", type=int, default=[10, 20, 30])
    p.add_argument("--sbs-list", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--alpha-local", type=float, default=1.0)
    p.add_argument("--beta-neighbor", type=float, default=4.0)
    p.add_argument("--chi-cloud", type=float, default=5.0)
    p.add_argument("--delta-replace", type=float, default=0.1)
    return p.parse_args()


def setup_logging(level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    return logging.getLogger("paper_cefmr_compare")


def load_scores(path: Path) -> dict[int, np.ndarray]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {int(k): np.asarray(v, dtype=np.float32) for k, v in state["user_scores"].items()}


class BasePolicy:
    name = "base"

    def reset(self, env: PaperCEFMRCooperativeEnv) -> None:
        return

    def select_items(self, obs: dict[str, np.ndarray], env: PaperCEFMRCooperativeEnv) -> np.ndarray:
        raise NotImplementedError

    def update(self, env: PaperCEFMRCooperativeEnv, items: np.ndarray, obs: dict[str, np.ndarray]) -> None:
        return


class RandomPolicy(BasePolicy):
    name = "Random"

    def __init__(self, num_items: int, seed: int) -> None:
        self.num_items = num_items
        self.rng = np.random.default_rng(seed)

    def select_items(self, obs: dict[str, np.ndarray], env: PaperCEFMRCooperativeEnv) -> np.ndarray:
        universe = np.arange(1, self.num_items + 1, dtype=np.int64)
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        for b in range(env.cfg.n_sbs):
            out[b] = self.rng.choice(universe, size=env.cfg.cache_capacity, replace=False)
        return out


class CEpsilonGreedyPolicy(BasePolicy):
    name = "C-epsilon-greedy"

    def __init__(self, num_items: int, epsilon: float, seed: int) -> None:
        self.num_items = num_items
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.counts: np.ndarray | None = None

    def reset(self, env: PaperCEFMRCooperativeEnv) -> None:
        self.counts = np.zeros((env.cfg.n_sbs, self.num_items + 1), dtype=np.float64)

    def select_items(self, obs: dict[str, np.ndarray], env: PaperCEFMRCooperativeEnv) -> np.ndarray:
        assert self.counts is not None
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        universe = np.arange(1, self.num_items + 1, dtype=np.int64)
        for b in range(env.cfg.n_sbs):
            if self.rng.random() < self.epsilon:
                out[b] = self.rng.choice(universe, size=env.cfg.cache_capacity, replace=False)
            else:
                best = np.argsort(self.counts[b, 1:])[-env.cfg.cache_capacity :][::-1] + 1
                out[b] = best.astype(np.int64)
        return out

    def update(self, env: PaperCEFMRCooperativeEnv, items: np.ndarray, obs: dict[str, np.ndarray]) -> None:
        assert self.counts is not None
        for ue in range(env.cfg.n_ues):
            b = int(obs["association"][ue])
            user_id = int(env.user_ids_for_ues[ue])
            ptr = int(env.user_ptrs[ue])
            item = int(env.user_histories[user_id][ptr])
            self.counts[b, item] += 1.0


class ThompsonPolicy(BasePolicy):
    name = "Thompson"

    def __init__(self, num_items: int, seed: int) -> None:
        self.num_items = num_items
        self.rng = np.random.default_rng(seed)
        self.alpha: np.ndarray | None = None
        self.beta: np.ndarray | None = None

    def reset(self, env: PaperCEFMRCooperativeEnv) -> None:
        self.alpha = np.ones((env.cfg.n_sbs, self.num_items + 1), dtype=np.float64)
        self.beta = np.ones((env.cfg.n_sbs, self.num_items + 1), dtype=np.float64)

    def select_items(self, obs: dict[str, np.ndarray], env: PaperCEFMRCooperativeEnv) -> np.ndarray:
        assert self.alpha is not None and self.beta is not None
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        candidates = np.arange(1, self.num_items + 1, dtype=np.int64)
        for b in range(env.cfg.n_sbs):
            sample = self.rng.beta(self.alpha[b, candidates], self.beta[b, candidates])
            out[b] = candidates[np.argsort(sample)[-env.cfg.cache_capacity :][::-1]]
        return out

    def update(self, env: PaperCEFMRCooperativeEnv, items: np.ndarray, obs: dict[str, np.ndarray]) -> None:
        assert self.alpha is not None and self.beta is not None
        demand_by_sbs: list[dict[int, int]] = [dict() for _ in range(env.cfg.n_sbs)]
        for ue in range(env.cfg.n_ues):
            b = int(obs["association"][ue])
            user_id = int(env.user_ids_for_ues[ue])
            ptr = int(env.user_ptrs[ue])
            item = int(env.user_histories[user_id][ptr])
            demand_by_sbs[b][item] = demand_by_sbs[b].get(item, 0) + 1
        for b in range(env.cfg.n_sbs):
            for cached in items[b]:
                cached = int(cached)
                if cached <= 0:
                    continue
                if demand_by_sbs[b].get(cached, 0) > 0:
                    self.alpha[b, cached] += 1.0
                else:
                    self.beta[b, cached] += 1.0


class BSGPolicy(BasePolicy):
    name = "BSG"

    def select_items(self, obs: dict[str, np.ndarray], env: PaperCEFMRCooperativeEnv) -> np.ndarray:
        best = np.argsort(env.global_popularity[1:])[-env.cfg.cache_capacity :][::-1] + 1
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        for b in range(env.cfg.n_sbs):
            out[b] = best.astype(np.int64)
        return out


class EFNRLPolicy(BasePolicy):
    name = "EFNRL"

    def select_items(self, obs: dict[str, np.ndarray], env: PaperCEFMRCooperativeEnv) -> np.ndarray:
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        for b in range(env.cfg.n_sbs):
            top = np.argsort(obs["candidate_scores"][b])[-env.cfg.cache_capacity :][::-1]
            out[b] = obs["candidate_items"][b, top].astype(np.int64)
        return out


class MADDPGPolicyWrapper(BasePolicy):
    def __init__(self, name: str, ckpt: Path, n_agents: int, state_dim: int, action_dim: int, device: str) -> None:
        self.name = name
        self.policy = MADDPGCachePolicy(
            n_agents=n_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            cfg=MADDPGConfig(device=device),
        )
        state = torch.load(ckpt, map_location=device, weights_only=False)
        trained_agents = int(state["cfg"]["n_agents"])
        if n_agents > trained_agents:
            raise ValueError(
                f"Checkpoint {ckpt} was trained with {trained_agents} SBS agents, "
                f"but evaluation requested {n_agents}."
            )

        # During comparison we only need deterministic actor inference.
        # For SBS sweeps we allow evaluating a prefix subset of the trained
        # agents, which keeps Fig. 9-style plots usable without forcing a
        # separate checkpoint per smaller SBS count.
        for actor, actor_state in zip(self.policy.actors, state["actors"][:n_agents]):
            actor.load_state_dict(actor_state)

    def select_items(self, obs: dict[str, np.ndarray], env: PaperCEFMRCooperativeEnv) -> np.ndarray:
        scores = self.policy.act(obs["local_states"], explore=False)
        return env.action_scores_to_items(scores)


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
    env_cfg: PaperEnvConfig,
    histories: dict[int, list[int]],
    user_scores: dict[int, np.ndarray],
    eval_episodes: int,
    seed: int,
    logger: logging.Logger | None = None,
    log_every_episode: int = 1,
    log_prefix: str = "",
) -> tuple[EvalResult, list[dict[str, object]]]:
    env = PaperCEFMRCooperativeEnv(env_cfg, histories, user_scores)
    policy.reset(env)
    rows: list[dict[str, object]] = []
    if logger is not None:
        logger.info("%sEvaluating %s | n_sbs=%d C=%d episodes=%d", log_prefix, policy.name, env_cfg.n_sbs, env_cfg.cache_capacity, eval_episodes)

    for ep in range(eval_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        reward_sum = 0.0
        local_sum = 0.0
        neighbor_sum = 0.0
        cloud_sum = 0.0
        steps = 0
        while not done:
            items = policy.select_items(obs, env)
            next_obs, reward, _local_rewards, done, info = env.step_items(items)
            policy.update(env, items, obs)
            reward_sum += float(reward)
            local_sum += float(info["local_hit_rate"])
            neighbor_sum += float(info["neighbor_fetch_rate"])
            cloud_sum += float(info["cloud_fetch_rate"])
            steps += 1
            obs = next_obs

        baseline_cost = env_cfg.chi_cloud * env_cfg.n_ues * steps
        actual_cost = baseline_cost - reward_sum * env_cfg.n_sbs
        row = {
            "episode": ep + 1,
            "reward": reward_sum,
            "cost": actual_cost,
            "local_hit_rate": local_sum / max(1, steps),
            "neighbor_fetch_rate": neighbor_sum / max(1, steps),
            "cloud_fetch_rate": cloud_sum / max(1, steps),
            "paper_style_hit_rate": (local_sum + neighbor_sum) / max(1, steps),
        }
        rows.append(row)
        if logger is not None and ((ep + 1) % max(1, log_every_episode) == 0 or (ep + 1) == eval_episodes):
            logger.info(
                "%s%s episode %d/%d | reward=%.4f cost=%.4f local=%.4f neighbor=%.4f cloud=%.4f",
                log_prefix,
                policy.name,
                ep + 1,
                eval_episodes,
                row["reward"],
                row["cost"],
                row["local_hit_rate"],
                row["neighbor_fetch_rate"],
                row["cloud_fetch_rate"],
            )

    result = EvalResult(
        scheme=policy.name,
        cache_capacity=env_cfg.cache_capacity,
        n_sbs=env_cfg.n_sbs,
        reward_mean=float(np.mean([float(r["reward"]) for r in rows])),
        cost_mean=float(np.mean([float(r["cost"]) for r in rows])),
        local_hit_mean=float(np.mean([float(r["local_hit_rate"]) for r in rows])),
        neighbor_fetch_mean=float(np.mean([float(r["neighbor_fetch_rate"]) for r in rows])),
        cloud_fetch_mean=float(np.mean([float(r["cloud_fetch_rate"]) for r in rows])),
        paper_hit_mean=float(np.mean([float(r["paper_style_hit_rate"]) for r in rows])),
    )
    return result, rows


def save_results(out_dir: Path, results: list[EvalResult], episode_rows: list[dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "scheme_capacity_summary.csv").open("w", newline="") as f:
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
    with (out_dir / "episode_metrics.csv").open("w", newline="") as f:
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


def plot_fig7(results: list[EvalResult], out_dir: Path) -> None:
    capacities = sorted({r.cache_capacity for r in results})
    target_sbs = max(r.n_sbs for r in results)
    schemes = sorted({r.scheme for r in results if r.n_sbs == target_sbs})
    for metric, title, ylabel, fname in [
        ("cost_mean", "Fig7-like: Cost vs Cache Capacity", "Cost", "fig7_like_cost_vs_capacity.png"),
        ("paper_hit_mean", "Fig7-like: Cache Hit Ratio vs Cache Capacity", "Cache Hit Ratio", "fig7_like_hit_vs_capacity.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for scheme in schemes:
            ys = [getattr(next(r for r in results if r.scheme == scheme and r.cache_capacity == c and r.n_sbs == target_sbs), metric) for c in capacities]
            ax.plot(capacities, ys, marker="o", linewidth=2, label=scheme)
        ax.set_title(title)
        ax.set_xlabel("Cache Capacity (C)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=180)
        plt.close(fig)


def plot_fig8(episode_rows: list[dict[str, object]], out_dir: Path) -> None:
    rows = [r for r in episode_rows if r["group"] == "fig8"]
    capacities = sorted({int(r["cache_capacity"]) for r in rows})
    for metric, title, ylabel, fname in [
        ("cost", "Fig8-like: Cost vs Episode", "Cost", "fig8_like_cost_vs_episode.png"),
        ("reward", "Fig8-like: Reward vs Episode", "Reward", "fig8_like_reward_vs_episode.png"),
        ("paper_style_hit_rate", "Fig8-like: Cache Hit Ratio vs Episode", "Cache Hit Ratio", "fig8_like_hit_vs_episode.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for c in capacities:
            rr = sorted([r for r in rows if int(r["cache_capacity"]) == c], key=lambda x: int(x["episode"]))
            ax.plot(
                [int(r["episode"]) for r in rr],
                [float(r[metric]) for r in rr],
                linewidth=2,
                label=f"C={c}",
            )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=180)
        plt.close(fig)


def plot_fig9(results: list[EvalResult], out_dir: Path) -> None:
    capacities = sorted({r.cache_capacity for r in results})
    sbs_list = sorted({r.n_sbs for r in results})
    for metric, title, ylabel, fname in [
        ("cost_mean", "Fig9-like: Cost vs Cache Capacity", "Cost", "fig9_like_cost_vs_capacity.png"),
        ("reward_mean", "Fig9-like: Reward vs Cache Capacity", "Reward", "fig9_like_reward_vs_capacity.png"),
        ("paper_hit_mean", "Fig9-like: Cache Hit Ratio vs Cache Capacity", "Cache Hit Ratio", "fig9_like_hit_vs_capacity.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for n_sbs in sbs_list:
            ys = [getattr(next(r for r in results if r.scheme == "CEFMR" and r.cache_capacity == c and r.n_sbs == n_sbs), metric) for c in capacities]
            ax.plot(capacities, ys, marker="o", linewidth=2, label=f"SBS={n_sbs}")
        ax.set_title(title)
        ax.set_xlabel("Cache Capacity (C)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    dataset_dir = get_movielens_dataset(args.data_root, args.dataset_name)
    histories = build_user_histories(load_ratings_auto(dataset_dir))
    cefmr_scores = load_scores(args.cefmr_aae)
    tfmadrl_scores = load_scores(args.tfmadrl_aae) if args.tfmadrl_aae is not None else None
    num_items = next(iter(cefmr_scores.values())).shape[0]

    results: list[EvalResult] = []
    episode_rows: list[dict[str, object]] = []
    state_dim = args.fp * 4 + 2

    logger.info("Stage 1/3: Fig7-like capacity comparison under different schemes")
    for capacity in args.cache_capacities:
        env_cfg = PaperEnvConfig(
            n_sbs=max(args.sbs_list),
            n_ues=args.n_ues,
            cache_capacity=capacity,
            fp=args.fp,
            episode_len=args.episode_len,
            grid_size=args.grid_size,
            alpha_local=args.alpha_local,
            beta_neighbor=args.beta_neighbor,
            chi_cloud=args.chi_cloud,
            delta_replace=args.delta_replace,
            seed=args.seed,
        )
        policies: list[tuple[BasePolicy, dict[int, np.ndarray]]] = [
            (RandomPolicy(num_items, args.seed), cefmr_scores),
            (CEpsilonGreedyPolicy(num_items, args.epsilon, args.seed), cefmr_scores),
            (ThompsonPolicy(num_items, args.seed), cefmr_scores),
            (BSGPolicy(), cefmr_scores),
            (EFNRLPolicy(), cefmr_scores),
        ]
        if args.tfmadrl_maddpg is not None and tfmadrl_scores is not None:
            policies.append((MADDPGPolicyWrapper("TFMADRL", args.tfmadrl_maddpg, env_cfg.n_sbs, state_dim, args.fp, args.device), tfmadrl_scores))
        policies.append((MADDPGPolicyWrapper("CEFMR", args.cefmr_maddpg, env_cfg.n_sbs, state_dim, args.fp, args.device), cefmr_scores))

        for policy, scores in policies:
            result, rows = evaluate_policy(
                policy,
                env_cfg,
                histories,
                scores,
                args.eval_episodes,
                args.seed,
                logger=logger,
                log_every_episode=args.log_every_episode,
                log_prefix=f"[Fig7 C={capacity}] ",
            )
            results.append(result)
            for row in rows:
                episode_rows.append({"group": "fig7", "scheme": policy.name, "cache_capacity": capacity, "n_sbs": env_cfg.n_sbs, **row})

    logger.info("Stage 2/3: Fig8-like testing stability for CEFMR")
    for capacity in args.cache_capacities:
        env_cfg = PaperEnvConfig(
            n_sbs=max(args.sbs_list),
            n_ues=args.n_ues,
            cache_capacity=capacity,
            fp=args.fp,
            episode_len=args.episode_len,
            grid_size=args.grid_size,
            alpha_local=args.alpha_local,
            beta_neighbor=args.beta_neighbor,
            chi_cloud=args.chi_cloud,
            delta_replace=args.delta_replace,
            seed=args.seed,
        )
        policy = MADDPGPolicyWrapper("CEFMR", args.cefmr_maddpg, env_cfg.n_sbs, state_dim, args.fp, args.device)
        _result, rows = evaluate_policy(
            policy,
            env_cfg,
            histories,
            cefmr_scores,
            args.eval_episodes,
            args.seed + 100,
            logger=logger,
            log_every_episode=args.log_every_episode,
            log_prefix=f"[Fig8 C={capacity}] ",
        )
        for row in rows:
            episode_rows.append({"group": "fig8", "scheme": "CEFMR", "cache_capacity": capacity, "n_sbs": env_cfg.n_sbs, **row})

    logger.info("Stage 3/3: Fig9-like SBS sweep for CEFMR")
    for n_sbs in args.sbs_list:
        for capacity in args.cache_capacities:
            env_cfg = PaperEnvConfig(
                n_sbs=n_sbs,
                n_ues=args.n_ues,
                cache_capacity=capacity,
                fp=args.fp,
                episode_len=args.episode_len,
                grid_size=args.grid_size,
                alpha_local=args.alpha_local,
                beta_neighbor=args.beta_neighbor,
                chi_cloud=args.chi_cloud,
                delta_replace=args.delta_replace,
                seed=args.seed,
            )
            policy = MADDPGPolicyWrapper("CEFMR", args.cefmr_maddpg, n_sbs, state_dim, args.fp, args.device)
            result, rows = evaluate_policy(
                policy,
                env_cfg,
                histories,
                cefmr_scores,
                args.eval_episodes,
                args.seed + 200,
                logger=logger,
                log_every_episode=args.log_every_episode,
                log_prefix=f"[Fig9 SBS={n_sbs} C={capacity}] ",
            )
            results.append(result)
            for row in rows:
                episode_rows.append({"group": "fig9", "scheme": "CEFMR", "cache_capacity": capacity, "n_sbs": n_sbs, **row})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_results(args.output_dir, results, episode_rows)
    plot_fig7(results, args.output_dir)
    plot_fig8(episode_rows, args.output_dir)
    plot_fig9(results, args.output_dir)
    logger.info("Artifacts saved under %s", args.output_dir)


if __name__ == "__main__":
    main()
