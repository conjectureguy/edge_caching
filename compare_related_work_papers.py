from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_mpl_dir = Path("outputs/.mplconfig")
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from movie_edge_sim.data import get_movielens_dataset, load_item_genres_auto, load_ratings_auto
from movie_edge_sim.novel_graph_policy import (
    TemporalGraphCooperativePolicy,
    logits_to_cache_items,
)
from movie_edge_sim.novel_realworld_env import NovelRealWorldCachingEnv, RealWorldEnvConfig
from movie_edge_sim.temporal_realworld import (
    RealWorldTemporalEncoder,
    build_user_time_histories,
    load_compatible_temporal_state,
)


SCHEME_ORDER = ["TemporalGraph", "MAAFDRL", "Thompson", "LFU", "LRU"]
SCHEME_COLORS = {
    "TemporalGraph": "#d62828",
    "MAAFDRL": "#2a9d8f",
    "Thompson": "#7b2cbf",
    "LFU": "#4d908e",
    "LRU": "#577590",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare our TemporalGraph solution against lightweight paper-inspired baselines "
            "for papers 2, 3, and 4."
        )
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--dataset-name", type=str, default="ml-1m", choices=["ml-100k", "ml-1m"])
    p.add_argument("--run-dir", type=Path, default=Path("outputs/novel_realworld_ml1m_final"))
    p.add_argument("--temporal-checkpoint", type=Path, default=None)
    p.add_argument("--policy-checkpoint", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/related_work_compare"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-every-episode", type=int, default=1)

    p.add_argument("--window-size", type=int, default=12)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--policy-hidden-dim", type=int, default=160)

    p.add_argument("--n-sbs", type=int, default=8)
    p.add_argument("--n-ues", type=int, default=220)
    p.add_argument("--cache-capacity", type=int, default=20)
    p.add_argument("--fp", type=int, default=50)
    p.add_argument("--episode-len", type=int, default=100)
    p.add_argument("--grid-size", type=float, default=300.0)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--decode-diversity-penalty", type=float, default=0.35)
    p.add_argument("--teacher-guidance-weight", type=float, default=0.55)
    p.add_argument("--placement-interval", type=int, default=3)
    p.add_argument("--use-recorded-our-eval", action="store_true", default=False)
    p.add_argument(
        "--exclude-schemes",
        nargs="*",
        default=[],
        help="Scheme labels to exclude from saved summaries and plots.",
    )
    return p.parse_args()


def setup_logging(level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    return logging.getLogger("related_work_compare")


@dataclass
class EvalSummary:
    scheme: str
    reward_mean: float
    local_hit_mean: float
    neighbor_fetch_mean: float
    cloud_fetch_mean: float
    paper_hit_mean: float


class BasePolicy:
    name = "base"

    def reset(self, env: NovelRealWorldCachingEnv) -> None:
        return

    def select_items(self, obs: dict[str, np.ndarray], env: NovelRealWorldCachingEnv) -> np.ndarray:
        raise NotImplementedError

    def update(self, env: NovelRealWorldCachingEnv, obs: dict[str, np.ndarray], chosen: np.ndarray, info: dict[str, float]) -> None:
        return


def _adapt_node_features(node_features: np.ndarray, expected_dim: int, embed_dim: int) -> np.ndarray:
    current_dim = int(node_features.shape[1])
    if current_dim == expected_dim:
        return node_features

    current_struct = current_dim - 2 * embed_dim
    expected_struct = expected_dim - 2 * embed_dim
    if current_struct > 0 and expected_struct > 0 and current_dim >= 2 * embed_dim and expected_dim >= 2 * embed_dim:
        structural = node_features[:, : min(current_struct, expected_struct)]
        if expected_struct > current_struct:
            pad = np.zeros((node_features.shape[0], expected_struct - current_struct), dtype=node_features.dtype)
            structural = np.concatenate([structural, pad], axis=1)
        embeddings = node_features[:, current_struct:]
        return np.concatenate([structural, embeddings], axis=1)

    if current_dim > expected_dim:
        return node_features[:, :expected_dim]
    pad = np.zeros((node_features.shape[0], expected_dim - current_dim), dtype=node_features.dtype)
    return np.concatenate([node_features, pad], axis=1)


def _adapt_candidate_features(candidate_features: np.ndarray, expected_dim: int, embed_dim: int) -> np.ndarray:
    current_dim = int(candidate_features.shape[2])
    if current_dim == expected_dim:
        return candidate_features

    current_struct = current_dim - embed_dim
    expected_struct = expected_dim - embed_dim
    if current_struct > 0 and expected_struct > 0 and current_dim >= embed_dim and expected_dim >= embed_dim:
        structural = candidate_features[:, :, : min(current_struct, expected_struct)]
        if expected_struct > current_struct:
            pad = np.zeros(
                (candidate_features.shape[0], candidate_features.shape[1], expected_struct - current_struct),
                dtype=candidate_features.dtype,
            )
            structural = np.concatenate([structural, pad], axis=2)
        embeddings = candidate_features[:, :, current_struct:]
        return np.concatenate([structural, embeddings], axis=2)

    if current_dim > expected_dim:
        return candidate_features[:, :, :expected_dim]
    pad = np.zeros(
        (candidate_features.shape[0], candidate_features.shape[1], expected_dim - current_dim),
        dtype=candidate_features.dtype,
    )
    return np.concatenate([candidate_features, pad], axis=2)


def _slot_scores_to_items(
    env: NovelRealWorldCachingEnv,
    slot_scores: np.ndarray,
    diversity_penalty: float = 0.0,
) -> np.ndarray:
    out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
    planned: list[set[int]] = [set() for _ in range(env.cfg.n_sbs)]
    order = np.argsort(slot_scores.max(axis=1) + 0.5 * env.current_future_load)[::-1]
    for b in order.tolist():
        scores = slot_scores[b].copy()
        neigh = np.where(env.current_adjacency[b] > 0.0)[0]
        neigh = neigh[neigh != b]
        for slot in np.where(env.current_mask[b])[0].tolist():
            item = int(env.current_candidates[b, slot])
            overlap = sum(item in planned[int(n)] for n in neigh)
            scores[slot] -= diversity_penalty * float(overlap)

        ranked = np.argsort(scores)[::-1]
        chosen: list[int] = []
        seen: set[int] = set()
        for slot in ranked.tolist():
            if not env.current_mask[b, int(slot)]:
                continue
            item = int(env.current_candidates[b, int(slot)])
            if item <= 0 or item in seen:
                continue
            chosen.append(item)
            seen.add(item)
            if len(chosen) >= env.cfg.cache_capacity:
                break
        if len(chosen) < env.cfg.cache_capacity:
            for item in (np.argsort(env.global_popularity[1:])[-env.cfg.cache_capacity:][::-1] + 1).tolist():
                item = int(item)
                if item in seen:
                    continue
                chosen.append(item)
                seen.add(item)
                if len(chosen) >= env.cfg.cache_capacity:
                    break
        out[b] = np.asarray(chosen[: env.cfg.cache_capacity], dtype=np.int64)
        planned[b] = set(out[b].tolist())
    return out


class AttentionWeightedFDRLPolicy(BasePolicy):
    name = "AWFDRL"

    def select_items(self, obs: dict[str, np.ndarray], env: NovelRealWorldCachingEnv) -> np.ndarray:
        node = obs["node_features"]
        cand = obs["candidate_features"]
        slot_scores = np.full((env.cfg.n_sbs, env.cfg.fp), -1e9, dtype=np.float64)
        for b in range(env.cfg.n_sbs):
            neigh = np.where(env.current_adjacency[b] > 0.0)[0]
            neigh = neigh[neigh != b]
            attn_neighbor = np.zeros((env.cfg.fp,), dtype=np.float64)
            if neigh.size > 0:
                sims = []
                for n in neigh.tolist():
                    # Attention over heterogeneous neighbors: closer state => higher weight.
                    sims.append(-float(np.linalg.norm(node[b, :9] - node[int(n), :9])))
                weights = np.exp(sims - np.max(sims))
                weights = weights / np.maximum(weights.sum(), 1e-8)
                for weight, n in zip(weights.tolist(), neigh.tolist()):
                    n = int(n)
                    for slot in np.where(env.current_mask[b])[0].tolist():
                        item = int(env.current_candidates[b, slot])
                        n_slots = np.where(env.current_candidates[n] == item)[0]
                        if n_slots.size == 0:
                            continue
                        n_slot = int(n_slots[0])
                        attn_neighbor[slot] += weight * float(env.current_candidate_scores[n, n_slot])

            valid_slots = np.where(env.current_mask[b])[0].tolist()
            if valid_slots:
                attn_neighbor[valid_slots] /= max(
                    1e-8,
                    np.max(np.abs(attn_neighbor[valid_slots])) if np.any(attn_neighbor[valid_slots]) else 1.0,
                )
            slot_scores[b] = (
                0.01 * attn_neighbor
                + 0.01 * cand[b, :, 3]
                + 0.01 * cand[b, :, 1]
                - 0.01 * cand[b, :, 2]
            )
            slot_scores[b, ~obs["action_mask"][b].astype(bool)] = -1e9
        return _slot_scores_to_items(env, slot_scores, diversity_penalty=0.0)


class MobilityAwareAsyncFDRLPolicy(BasePolicy):
    name = "MAAFDRL"

    def __init__(self) -> None:
        self.async_popularity: np.ndarray | None = None

    def reset(self, env: NovelRealWorldCachingEnv) -> None:
        self.async_popularity = np.zeros((env.cfg.n_sbs, env.num_items + 1), dtype=np.float64)

    def select_items(self, obs: dict[str, np.ndarray], env: NovelRealWorldCachingEnv) -> np.ndarray:
        assert self.async_popularity is not None
        cand = obs["candidate_features"]
        slot_scores = np.full((env.cfg.n_sbs, env.cfg.fp), -1e9, dtype=np.float64)
        for b in range(env.cfg.n_sbs):
            async_slot = np.zeros((env.cfg.fp,), dtype=np.float64)
            for slot in np.where(env.current_mask[b])[0].tolist():
                item = int(env.current_candidates[b, slot])
                async_slot[slot] = float(self.async_popularity[b, item])
            valid_slots = np.where(env.current_mask[b])[0].tolist()
            if valid_slots:
                vals = async_slot[valid_slots]
                denom = np.max(vals) if np.any(vals > 0) else 1.0
                async_slot[valid_slots] = vals / max(1e-8, denom)

            mobility_gain = 0.5 * cand[b, :, 5]
            slot_scores[b] = (
                0.15 * async_slot
                + 0.05 * mobility_gain
                + 0.10 * cand[b, :, 3]
                - 0.20 * cand[b, :, 2]
            )
            slot_scores[b, ~obs["action_mask"][b].astype(bool)] = -1e9
        return _slot_scores_to_items(env, slot_scores, diversity_penalty=0.0)

    def update(self, env: NovelRealWorldCachingEnv, obs: dict[str, np.ndarray], chosen: np.ndarray, info: dict[str, float]) -> None:
        assert self.async_popularity is not None
        self.async_popularity *= 0.50
        association = obs["association"]
        for ue in range(env.cfg.n_ues):
            if not env.last_active_mask[ue]:
                continue
            item = int(env.last_requests[ue])
            if item <= 0:
                continue
            b = int(association[ue])
            self.async_popularity[b, item] += 1.0


class LRUPolicy(BasePolicy):
    name = "LRU"

    def __init__(self) -> None:
        self.last_seen: np.ndarray | None = None
        self._tick = 0

    def reset(self, env: NovelRealWorldCachingEnv) -> None:
        self.last_seen = np.zeros((env.cfg.n_sbs, env.num_items + 1), dtype=np.float64)
        self._tick = 0

    def select_items(self, obs: dict[str, np.ndarray], env: NovelRealWorldCachingEnv) -> np.ndarray:
        assert self.last_seen is not None
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        fallback = (np.argsort(env.global_popularity[1:])[-env.cfg.cache_capacity :][::-1] + 1).tolist()
        for b in range(env.cfg.n_sbs):
            scores = self.last_seen[b, 1:]
            ranked = (np.argsort(scores)[::-1] + 1).tolist()
            chosen: list[int] = []
            seen: set[int] = set()
            for item in ranked:
                item = int(item)
                if scores[item - 1] <= 0.0:
                    continue
                if item not in seen:
                    chosen.append(item)
                    seen.add(item)
                if len(chosen) >= env.cfg.cache_capacity:
                    break
            for item in fallback:
                item = int(item)
                if item not in seen:
                    chosen.append(item)
                    seen.add(item)
                if len(chosen) >= env.cfg.cache_capacity:
                    break
            out[b] = np.asarray(chosen[: env.cfg.cache_capacity], dtype=np.int64)
        return out

    def update(self, env: NovelRealWorldCachingEnv, obs: dict[str, np.ndarray], chosen: np.ndarray, info: dict[str, float]) -> None:
        assert self.last_seen is not None
        association = obs["association"]
        for ue in range(env.cfg.n_ues):
            if not env.last_active_mask[ue]:
                continue
            item = int(env.last_requests[ue])
            if item <= 0:
                continue
            self._tick += 1
            self.last_seen[int(association[ue]), item] = float(self._tick)


class LFUPolicy(BasePolicy):
    name = "LFU"

    def __init__(self) -> None:
        self.counts: np.ndarray | None = None

    def reset(self, env: NovelRealWorldCachingEnv) -> None:
        self.counts = np.zeros((env.cfg.n_sbs, env.num_items + 1), dtype=np.float64)

    def select_items(self, obs: dict[str, np.ndarray], env: NovelRealWorldCachingEnv) -> np.ndarray:
        assert self.counts is not None
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        fallback = (np.argsort(env.global_popularity[1:])[-env.cfg.cache_capacity :][::-1] + 1).tolist()
        for b in range(env.cfg.n_sbs):
            scores = self.counts[b, 1:]
            ranked = (np.argsort(scores)[::-1] + 1).tolist()
            chosen: list[int] = []
            seen: set[int] = set()
            for item in ranked:
                item = int(item)
                if scores[item - 1] <= 0.0:
                    continue
                if item not in seen:
                    chosen.append(item)
                    seen.add(item)
                if len(chosen) >= env.cfg.cache_capacity:
                    break
            for item in fallback:
                item = int(item)
                if item not in seen:
                    chosen.append(item)
                    seen.add(item)
                if len(chosen) >= env.cfg.cache_capacity:
                    break
            out[b] = np.asarray(chosen[: env.cfg.cache_capacity], dtype=np.int64)
        return out

    def update(self, env: NovelRealWorldCachingEnv, obs: dict[str, np.ndarray], chosen: np.ndarray, info: dict[str, float]) -> None:
        assert self.counts is not None
        association = obs["association"]
        for ue in range(env.cfg.n_ues):
            if not env.last_active_mask[ue]:
                continue
            item = int(env.last_requests[ue])
            if item <= 0:
                continue
            self.counts[int(association[ue]), item] += 1.0


class ThompsonPolicy(BasePolicy):
    name = "Thompson"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.alpha: np.ndarray | None = None
        self.beta: np.ndarray | None = None

    def reset(self, env: NovelRealWorldCachingEnv) -> None:
        self.alpha = np.ones((env.cfg.n_sbs, env.num_items + 1), dtype=np.float64)
        self.beta = np.ones((env.cfg.n_sbs, env.num_items + 1), dtype=np.float64)

    def select_items(self, obs: dict[str, np.ndarray], env: NovelRealWorldCachingEnv) -> np.ndarray:
        assert self.alpha is not None and self.beta is not None
        out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
        candidates = np.arange(1, env.num_items + 1, dtype=np.int64)
        for b in range(env.cfg.n_sbs):
            sampled = self.rng.beta(self.alpha[b, candidates], self.beta[b, candidates])
            best = candidates[np.argsort(sampled)[-env.cfg.cache_capacity :][::-1]]
            out[b] = best.astype(np.int64)
        return out

    def update(self, env: NovelRealWorldCachingEnv, obs: dict[str, np.ndarray], chosen: np.ndarray, info: dict[str, float]) -> None:
        assert self.alpha is not None and self.beta is not None
        association = obs["association"]
        demand_by_sbs: list[dict[int, int]] = [dict() for _ in range(env.cfg.n_sbs)]
        for ue in range(env.cfg.n_ues):
            if not env.last_active_mask[ue]:
                continue
            item = int(env.last_requests[ue])
            if item <= 0:
                continue
            b = int(association[ue])
            demand_by_sbs[b][item] = demand_by_sbs[b].get(item, 0) + 1
        for b in range(env.cfg.n_sbs):
            for item in chosen[b]:
                item = int(item)
                if item <= 0:
                    continue
                # Treat each cached item as one Bernoulli exposure per step.
                if demand_by_sbs[b].get(item, 0) > 0:
                    self.alpha[b, item] += 1.0
                else:
                    self.beta[b, item] += 1.0


class CooperativeDDPGPolicy(BasePolicy):
    name = "DTS-DDPG"

    def __init__(self, placement_interval: int = 12) -> None:
        self.placement_interval = placement_interval
        self._step = 0
        self._last_action: np.ndarray | None = None

    def reset(self, env: NovelRealWorldCachingEnv) -> None:
        self._step = 0
        self._last_action = None

    def select_items(self, obs: dict[str, np.ndarray], env: NovelRealWorldCachingEnv) -> np.ndarray:
        if self._last_action is not None and (self._step % self.placement_interval) != 0:
            return self._last_action.copy()

        cand = obs["candidate_features"]
        slot_scores = (
            0.20 * cand[:, :, 3]
            + 0.05 * cand[:, :, 1]
            - 0.15 * cand[:, :, 2]
        )
        slot_scores[~obs["action_mask"].astype(bool)] = -1e9
        self._last_action = _slot_scores_to_items(env, slot_scores, diversity_penalty=0.0)
        return self._last_action.copy()

    def update(self, env: NovelRealWorldCachingEnv, obs: dict[str, np.ndarray], chosen: np.ndarray, info: dict[str, float]) -> None:
        self._step += 1


def evaluate_policy(
    policy: BasePolicy,
    env: NovelRealWorldCachingEnv,
    episodes: int,
    seed: int,
    log_every_episode: int = 1,
    logger: logging.Logger | None = None,
    placement_interval: int = 1,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        policy.reset(env)
        done = False
        step_idx = 0
        last_action: np.ndarray | None = None
        reward_sum = 0.0
        local_sum = 0.0
        neighbor_sum = 0.0
        cloud_sum = 0.0
        steps = 0
        while not done:
            if last_action is None or step_idx % max(1, placement_interval) == 0:
                chosen = policy.select_items(obs, env)
            else:
                chosen = last_action.copy()
            last_action = chosen.copy()
            next_obs, reward, done, info = env.step_full_cache_items(chosen)
            policy.update(env, obs, chosen, info)
            reward_sum += float(reward)
            local_sum += float(info["local_hit_rate"])
            neighbor_sum += float(info["neighbor_fetch_rate"])
            cloud_sum += float(info["cloud_fetch_rate"])
            steps += 1
            step_idx += 1
            obs = next_obs
        row = {
            "episode": ep + 1,
            "reward": reward_sum,
            "local_hit_rate": local_sum / max(1, steps),
            "neighbor_fetch_rate": neighbor_sum / max(1, steps),
            "cloud_fetch_rate": cloud_sum / max(1, steps),
            "paper_hit_rate": (local_sum + neighbor_sum) / max(1, steps),
        }
        rows.append(row)
        if logger is not None and ((ep + 1) % max(1, log_every_episode) == 0 or (ep + 1) == episodes):
            logger.info(
                "%s episode %d/%d | reward=%.4f local=%.4f neighbor=%.4f cloud=%.4f paper_hit=%.4f",
                policy.name,
                ep + 1,
                episodes,
                row["reward"],
                row["local_hit_rate"],
                row["neighbor_fetch_rate"],
                row["cloud_fetch_rate"],
                row["paper_hit_rate"],
            )
    return rows


@torch.no_grad()
def evaluate_our_policy(
    env: NovelRealWorldCachingEnv,
    model: TemporalGraphCooperativePolicy,
    episodes: int,
    seed: int,
    device: str,
    expected_node_dim: int,
    expected_candidate_dim: int,
    decode_diversity_penalty: float,
    teacher_guidance_weight: float,
    placement_interval: int,
) -> list[dict[str, float]]:
    model = model.to(device)
    model.eval()
    rows: list[dict[str, float]] = []
    dev = torch.device(device)
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        step_idx = 0
        last_action: np.ndarray | None = None
        reward_sum = 0.0
        local_sum = 0.0
        neighbor_sum = 0.0
        cloud_sum = 0.0
        steps = 0
        while not done:
            node = _adapt_node_features(obs["node_features"], expected_node_dim, env.embed_dim)
            cand = _adapt_candidate_features(obs["candidate_features"], expected_candidate_dim, env.embed_dim)
            node_t = torch.as_tensor(node, dtype=torch.float32, device=dev)
            cand_t = torch.as_tensor(cand, dtype=torch.float32, device=dev)
            adj_t = torch.as_tensor(obs["adjacency"], dtype=torch.float32, device=dev)
            mask_t = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=dev)
            logits = model(node_t, cand_t, adj_t, mask_t)
            teacher_scores = env.cooperative_teacher_scores()
            chosen = logits_to_cache_items(
                logits,
                env,
                diversity_penalty=decode_diversity_penalty,
                teacher_scores=teacher_scores,
                teacher_guidance_weight=teacher_guidance_weight,
            )
            if last_action is None or step_idx % max(1, placement_interval) == 0:
                action_to_apply = chosen
            else:
                action_to_apply = last_action
            last_action = action_to_apply.copy()
            obs, reward, done, info = env.step_full_cache_items(action_to_apply)
            reward_sum += float(reward)
            local_sum += float(info["local_hit_rate"])
            neighbor_sum += float(info["neighbor_fetch_rate"])
            cloud_sum += float(info["cloud_fetch_rate"])
            steps += 1
            step_idx += 1
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


def summarize(name: str, rows: list[dict[str, float]]) -> EvalSummary:
    return EvalSummary(
        scheme=name,
        reward_mean=float(np.mean([r["reward"] for r in rows])),
        local_hit_mean=float(np.mean([r["local_hit_rate"] for r in rows])),
        neighbor_fetch_mean=float(np.mean([r["neighbor_fetch_rate"] for r in rows])),
        cloud_fetch_mean=float(np.mean([r["cloud_fetch_rate"] for r in rows])),
        paper_hit_mean=float(np.mean([r["paper_hit_rate"] for r in rows])),
    )


def save_outputs(
    out_dir: Path,
    results: list[EvalSummary],
    episode_rows: list[dict[str, float | str]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scheme",
                "reward_mean",
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
                    f"{r.reward_mean:.8f}",
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
                "scheme",
                "episode",
                "reward",
                "local_hit_rate",
                "neighbor_fetch_rate",
                "cloud_fetch_rate",
                "paper_hit_rate",
            ]
        )
        for row in episode_rows:
            writer.writerow(
                [
                    row["scheme"],
                    row["episode"],
                    f"{float(row['reward']):.8f}",
                    f"{float(row['local_hit_rate']):.8f}",
                    f"{float(row['neighbor_fetch_rate']):.8f}",
                    f"{float(row['cloud_fetch_rate']):.8f}",
                    f"{float(row['paper_hit_rate']):.8f}",
                ]
            )


def _bar_plot(results: list[EvalSummary], metric: str, title: str, ylabel: str, out_path: Path) -> None:
    ordered_results = sorted(results, key=lambda r: SCHEME_ORDER.index(r.scheme) if r.scheme in SCHEME_ORDER else len(SCHEME_ORDER))
    schemes = [r.scheme for r in ordered_results]
    values = [getattr(r, metric) for r in results]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(schemes, [getattr(r, metric) for r in ordered_results], color=[SCHEME_COLORS.get(name, "#6c757d") for name in schemes])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, [getattr(r, metric) for r in ordered_results]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _episode_plot(rows: list[dict[str, float | str]], metric: str, title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    schemes = sorted(
        {str(r["scheme"]) for r in rows},
        key=lambda name: SCHEME_ORDER.index(name) if name in SCHEME_ORDER else len(SCHEME_ORDER),
    )
    for scheme in schemes:
        rr = sorted([r for r in rows if str(r["scheme"]) == scheme], key=lambda x: int(x["episode"]))
        ax.plot(
            [int(r["episode"]) for r in rr],
            [float(r[metric]) for r in rr],
            marker="o",
            linewidth=2,
            color=SCHEME_COLORS.get(scheme, "#6c757d"),
            label=scheme,
        )
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_outputs(out_dir: Path, results: list[EvalSummary], episode_rows: list[dict[str, float | str]]) -> None:
    _bar_plot(results, "reward_mean", "Related Work Comparison: Reward", "Reward", out_dir / "reward_comparison.png")
    _bar_plot(results, "local_hit_mean", "Related Work Comparison: Local Hit Rate", "Local Hit Rate", out_dir / "local_hit_comparison.png")
    _bar_plot(results, "paper_hit_mean", "Related Work Comparison: Paper Hit Rate", "Paper Hit Rate", out_dir / "paper_hit_comparison.png")
    _bar_plot(results, "cloud_fetch_mean", "Related Work Comparison: Cloud Fetch Rate", "Cloud Fetch Rate", out_dir / "cloud_fetch_comparison.png")
    _episode_plot(episode_rows, "reward", "Reward by Episode", "Reward", out_dir / "reward_vs_episode.png")
    _episode_plot(episode_rows, "paper_hit_rate", "Paper Hit Rate by Episode", "Paper Hit Rate", out_dir / "paper_hit_vs_episode.png")


def load_recorded_eval(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "episode": int(row["episode"]),
                    "reward": float(row["reward"]),
                    "local_hit_rate": float(row["local_hit_rate"]),
                    "neighbor_fetch_rate": float(row["neighbor_fetch_rate"]),
                    "cloud_fetch_rate": float(row["cloud_fetch_rate"]),
                    "paper_hit_rate": float(row["paper_hit_rate"]),
                }
            )
    return rows


def filter_outputs(
    results: list[EvalSummary],
    episode_rows: list[dict[str, float | str]],
    excluded: set[str],
) -> tuple[list[EvalSummary], list[dict[str, float | str]]]:
    if not excluded:
        return results, episode_rows
    filtered_results = [r for r in results if r.scheme not in excluded]
    filtered_episode_rows = [row for row in episode_rows if str(row["scheme"]) not in excluded]
    return filtered_results, filtered_episode_rows


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    temporal_ckpt = args.temporal_checkpoint or (args.run_dir / "realworld_temporal_encoder.pt")
    policy_ckpt = args.policy_checkpoint or (args.run_dir / "temporal_graph_policy.pt")

    logger.info("Stage 1/4: loading dataset %s", args.dataset_name)
    dataset_dir = get_movielens_dataset(args.data_root, args.dataset_name)
    ratings = load_ratings_auto(dataset_dir)
    item_genres, _genre_names = load_item_genres_auto(dataset_dir)
    histories = build_user_time_histories(ratings)

    logger.info("Stage 2/4: loading trained temporal encoder from %s", temporal_ckpt)
    max_user_id = max(histories.keys())
    max_item_id = max(max(hist.items) for hist in histories.values())
    temporal_model = RealWorldTemporalEncoder(
        num_items=max_item_id,
        num_users=max_user_id,
        window_size=args.window_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
    ).to(args.device)
    load_compatible_temporal_state(
        temporal_model,
        torch.load(temporal_ckpt, map_location=args.device, weights_only=True),
        logger=logger,
        source=str(temporal_ckpt),
    )
    temporal_model.eval()

    logger.info("Stage 3/4: building evaluation environment")
    env_cfg = RealWorldEnvConfig(
        n_sbs=args.n_sbs,
        n_ues=args.n_ues,
        cache_capacity=args.cache_capacity,
        fp=args.fp,
        window_size=args.window_size,
        episode_len=args.episode_len,
        grid_size=args.grid_size,
        seed=args.seed,
    )
    env = NovelRealWorldCachingEnv(env_cfg, temporal_model, histories, item_genres=item_genres)
    obs = env.reset(seed=args.seed)

    node_dim = int(obs["node_features"].shape[1])
    cand_dim = int(obs["candidate_features"].shape[2])
    logger.info("Environment ready | node_dim=%d candidate_dim=%d", node_dim, cand_dim)

    logger.info("Stage 4/4: evaluating our method and paper-inspired baselines")
    policy_state = torch.load(policy_ckpt, map_location=args.device, weights_only=True)
    expected_node_dim = int(policy_state["gat1.proj.weight"].shape[1])
    expected_candidate_dim = int(policy_state["candidate_encoder.0.weight"].shape[1])
    expected_hidden_dim = int(policy_state["gat1.proj.weight"].shape[0])
    temporal_graph = TemporalGraphCooperativePolicy(
        node_feat_dim=expected_node_dim,
        candidate_feat_dim=expected_candidate_dim,
        hidden_dim=expected_hidden_dim,
        fp=args.fp,
        use_graph=True,
    ).to(args.device)
    temporal_graph.load_state_dict(policy_state, strict=False)
    temporal_graph.eval()

    recorded_eval_path = args.run_dir / "temporal_graph_eval.csv"
    if args.use_recorded_our_eval and recorded_eval_path.exists():
        logger.info("Using recorded TemporalGraph evaluation from %s", recorded_eval_path)
        our_rows = load_recorded_eval(recorded_eval_path)
    else:
        logger.info("Evaluating TemporalGraph in the same fresh harness as the paper-inspired baselines")
        our_rows = evaluate_our_policy(
            env,
            temporal_graph,
            episodes=args.eval_episodes,
            seed=args.seed + 3000,
            device=args.device,
            expected_node_dim=expected_node_dim,
            expected_candidate_dim=expected_candidate_dim,
            decode_diversity_penalty=args.decode_diversity_penalty,
            teacher_guidance_weight=args.teacher_guidance_weight,
            placement_interval=args.placement_interval,
        )
    logger.info("Evaluating MAAFDRL baseline")
    maafdrl_rows = evaluate_policy(
        MobilityAwareAsyncFDRLPolicy(),
        env,
        episodes=args.eval_episodes,
        seed=args.seed + 4000,
        logger=logger,
        log_every_episode=args.log_every_episode,
        placement_interval=args.placement_interval,
    )
    logger.info("Evaluating Thompson baseline")
    thompson_rows = evaluate_policy(
        ThompsonPolicy(seed=args.seed + 5000),
        env,
        episodes=args.eval_episodes,
        seed=args.seed + 5000,
        logger=logger,
        log_every_episode=args.log_every_episode,
        placement_interval=args.placement_interval,
    )
    logger.info("Evaluating LFU baseline")
    lfu_rows = evaluate_policy(
        LFUPolicy(),
        env,
        episodes=args.eval_episodes,
        seed=args.seed + 6000,
        logger=logger,
        log_every_episode=args.log_every_episode,
        placement_interval=args.placement_interval,
    )
    logger.info("Evaluating LRU baseline")
    lru_rows = evaluate_policy(
        LRUPolicy(),
        env,
        episodes=args.eval_episodes,
        seed=args.seed + 7000,
        logger=logger,
        log_every_episode=args.log_every_episode,
        placement_interval=args.placement_interval,
    )

    results = [
        summarize("TemporalGraph", our_rows),
        summarize("MAAFDRL", maafdrl_rows),
        summarize("Thompson", thompson_rows),
        summarize("LFU", lfu_rows),
        summarize("LRU", lru_rows),
    ]

    episode_rows: list[dict[str, float | str]] = []
    for scheme, rows in [
        ("TemporalGraph", our_rows),
        ("MAAFDRL", maafdrl_rows),
        ("Thompson", thompson_rows),
        ("LFU", lfu_rows),
        ("LRU", lru_rows),
    ]:
        for row in rows:
            episode_rows.append({"scheme": scheme, **row})

    excluded = set(args.exclude_schemes)
    if excluded:
        logger.info("Excluding schemes from saved outputs: %s", ", ".join(sorted(excluded)))
    results, episode_rows = filter_outputs(results, episode_rows, excluded)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_outputs(args.output_dir, results, episode_rows)
    plot_outputs(args.output_dir, results, episode_rows)

    summary_lines = [
        f"Run dir: {args.run_dir}",
        f"Dataset: {args.dataset_name}",
        f"Evaluation episodes: {args.eval_episodes}",
    ]
    for r in results:
        summary_lines.append(
            f"{r.scheme}: reward_mean={r.reward_mean:.4f} local_hit_mean={r.local_hit_mean:.4f} "
            f"paper_hit_mean={r.paper_hit_mean:.4f} cloud_fetch_mean={r.cloud_fetch_mean:.4f}"
        )
    (args.output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    for line in summary_lines[3:]:
        logger.info(line)


if __name__ == "__main__":
    main()
