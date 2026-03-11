from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from movie_edge_sim.simulation import SimulationConfig, run_simulation
from movie_edge_sim.temporal_federated import TemporalSpikeEncoder


@dataclass
class EnvConfig:
    n_sbs: int = 8
    n_ues: int = 240
    cache_capacity: int = 20
    fp: int = 40
    window_size: int = 10
    episode_len: int = 200
    grid_size: float = 300.0
    ue_max_speed: float = 1.2
    sbs_max_speed: float = 0.4
    sbs_update_interval: float = 10.0
    neighbor_radius: float = 130.0
    replacements_per_step: int = 3
    candidate_recent_weight: float = 0.6
    cache_hit_decay: float = 0.98
    alpha_local: float = 1.0
    beta_neighbor: float = 4.0
    chi_cloud: float = 5.0
    delta_replace: float = 0.1
    seed: int = 42


class CooperativeCachingEnv:
    """
    Environment for cooperative SBS caching:
    - Candidate set per SBS: greedy top-Fp by temporal popularity scores.
    - Action per SBS: select one candidate item to enforce into cache.
    - Reward: cost saved against cloud-only baseline.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        temporal_model: TemporalSpikeEncoder,
        user_histories: dict[int, list[int]],
    ) -> None:
        self.cfg = cfg
        self.temporal_model = temporal_model.eval()
        self.user_histories = user_histories
        self.rng = np.random.default_rng(cfg.seed)

        self._device = next(self.temporal_model.parameters()).device
        self._embed_weight = self.temporal_model.embed.weight.detach().to(self._device)
        self.embed_dim = int(self._embed_weight.shape[1])
        self.num_items = int(self.temporal_model.num_items)
        if cfg.fp <= 0 or cfg.cache_capacity <= 0:
            raise ValueError("fp and cache_capacity must be > 0")
        if cfg.fp > self.num_items:
            raise ValueError(f"fp ({cfg.fp}) cannot exceed number of items ({self.num_items})")
        if cfg.cache_capacity > cfg.fp:
            raise ValueError("cache_capacity should be <= fp for meaningful candidate-based caching")
        if cfg.replacements_per_step <= 0:
            raise ValueError("replacements_per_step must be > 0")
        if not (0.0 <= cfg.candidate_recent_weight <= 1.0):
            raise ValueError("candidate_recent_weight must be in [0, 1]")
        if not (0.0 < cfg.cache_hit_decay <= 1.0):
            raise ValueError("cache_hit_decay must be in (0, 1]")

        self._eligible_users = [u for u, seq in self.user_histories.items() if len(seq) > cfg.window_size + 1]
        if len(self._eligible_users) < cfg.n_ues:
            raise ValueError(
                f"Not enough users with history > window_size. Need {cfg.n_ues}, found {len(self._eligible_users)}."
            )

        self.global_popularity = self._build_global_popularity()
        self._build_mobility()

        self.step_idx = 0
        self.user_ids_for_ues = np.zeros((cfg.n_ues,), dtype=np.int64)
        self.user_ptrs = np.zeros((cfg.n_ues,), dtype=np.int64)
        self.cache_items = np.zeros((cfg.n_sbs, cfg.cache_capacity), dtype=np.int64)
        self.cache_hits = np.zeros((cfg.n_sbs, cfg.cache_capacity), dtype=np.float64)
        self.last_hit_rate = np.zeros((cfg.n_sbs,), dtype=np.float64)

        self.current_candidates = np.zeros((cfg.n_sbs, cfg.fp), dtype=np.int64)
        self.current_candidate_scores = np.zeros((cfg.n_sbs, cfg.fp), dtype=np.float64)
        self.current_mask = np.zeros((cfg.n_sbs, cfg.fp), dtype=bool)
        self.current_adjacency = np.eye(cfg.n_sbs, dtype=np.float32)

    def _build_global_popularity(self) -> np.ndarray:
        popularity = np.zeros((self.num_items + 1,), dtype=np.float64)
        for seq in self.user_histories.values():
            for item in seq:
                if 0 < item <= self.num_items:
                    popularity[item] += 1.0
        popularity /= max(1.0, popularity.sum())
        return popularity

    def _build_mobility(self) -> None:
        sim_cfg = SimulationConfig(
            grid_size=self.cfg.grid_size,
            n_ues=self.cfg.n_ues,
            n_sbs=self.cfg.n_sbs,
            total_time=float(self.cfg.episode_len),
            dt=1.0,
            t_update=self.cfg.sbs_update_interval,
            max_speed=self.cfg.ue_max_speed,
            sbs_max_speed=self.cfg.sbs_max_speed,
            random_seed=self.cfg.seed,
            kmeans_iters=30,
        )
        sim_result = run_simulation(sim_cfg)
        self.ue_positions = sim_result.ue_positions_over_time  # (T+1, U, 2)
        self.sbs_positions_updates = sim_result.sbs_positions_over_time  # (n_updates, B, 2)
        self.sbs_update_times = sim_result.update_times.astype(np.int64)
        self.update_stride = int(round(self.cfg.sbs_update_interval))

    def _sbs_positions_at(self, step: int) -> np.ndarray:
        update_idx = min(step // max(1, self.update_stride), self.sbs_positions_updates.shape[0] - 1)
        return self.sbs_positions_updates[update_idx]

    def _associate_ues_to_sbs(self, ue_positions: np.ndarray, sbs_positions: np.ndarray) -> np.ndarray:
        d2 = ((ue_positions[:, None, :] - sbs_positions[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1).astype(np.int64)

    def _adjacency_from_sbs(self, sbs_positions: np.ndarray) -> np.ndarray:
        d = np.sqrt(((sbs_positions[:, None, :] - sbs_positions[None, :, :]) ** 2).sum(axis=2))
        adj = (d <= self.cfg.neighbor_radius).astype(np.float32)
        np.fill_diagonal(adj, 1.0)
        return adj

    def _make_context(self, user_id: int, ptr: int) -> np.ndarray:
        seq = self.user_histories[user_id]
        k = self.cfg.window_size
        return np.asarray(seq[ptr - k : ptr], dtype=np.int64)

    @torch.no_grad()
    def _predict_top_candidates(self, contexts: np.ndarray, recent_items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if contexts.shape[0] == 0:
            top_items = np.argsort(self.global_popularity[1:])[-self.cfg.fp :][::-1] + 1
            top_scores = self.global_popularity[top_items]
            return top_items.astype(np.int64), top_scores.astype(np.float64)

        x = torch.as_tensor(contexts, dtype=torch.long, device=self._device)
        probs = self.temporal_model.predict_scores(x).detach().cpu().numpy()
        temporal_scores = probs.sum(axis=0)  # (num_items+1,)

        # Blend sequential-model score with very recent request spikes from associated UEs.
        recent_counts = np.bincount(recent_items, minlength=self.num_items + 1).astype(np.float64)
        if recent_counts.sum() > 0:
            recent_scores = recent_counts / recent_counts.sum()
        else:
            recent_scores = recent_counts
        w = self.cfg.candidate_recent_weight
        agg = (1.0 - w) * temporal_scores + w * recent_scores
        agg[0] = -np.inf
        top_items = np.argsort(agg)[-self.cfg.fp :][::-1]
        top_scores = agg[top_items]
        return top_items.astype(np.int64), top_scores.astype(np.float64)

    def _refresh_candidates(self, association: np.ndarray) -> None:
        B = self.cfg.n_sbs
        Fp = self.cfg.fp
        self.current_candidates.fill(0)
        self.current_candidate_scores.fill(0.0)
        self.current_mask.fill(False)

        for b in range(B):
            ue_idx = np.where(association == b)[0]
            contexts = []
            recent_items = []
            for u in ue_idx:
                user_id = int(self.user_ids_for_ues[u])
                ptr = int(self.user_ptrs[u])
                contexts.append(self._make_context(user_id, ptr))
                recent_items.append(int(self.user_histories[user_id][ptr]))
            if contexts:
                ctx = np.asarray(contexts, dtype=np.int64)
                rec = np.asarray(recent_items, dtype=np.int64)
            else:
                ctx = np.zeros((0, self.cfg.window_size), dtype=np.int64)
                rec = np.zeros((0,), dtype=np.int64)

            items, scores = self._predict_top_candidates(ctx, rec)
            valid = min(Fp, items.shape[0])
            self.current_candidates[b, :valid] = items[:valid]
            self.current_candidate_scores[b, :valid] = scores[:valid]
            self.current_mask[b, :valid] = self.current_candidates[b, :valid] > 0

    def _initialize_caches(self) -> None:
        C = self.cfg.cache_capacity
        fallback = np.argsort(self.global_popularity[1:])[-self.cfg.fp :][::-1] + 1
        for b in range(self.cfg.n_sbs):
            candidates = self.current_candidates[b]
            mask = self.current_mask[b]
            valid_items = candidates[mask]
            if valid_items.size >= C:
                self.cache_items[b] = valid_items[:C]
            elif valid_items.size > 0:
                self.cache_items[b, : valid_items.size] = valid_items
                j = valid_items.size
                for item in fallback:
                    if item <= 0 or item in self.cache_items[b, :j]:
                        continue
                    self.cache_items[b, j] = item
                    j += 1
                    if j >= C:
                        break
            else:
                self.cache_items[b] = fallback[:C]
        self.cache_hits.fill(0.0)

    def _build_node_features(self, association: np.ndarray, sbs_positions: np.ndarray) -> np.ndarray:
        B = self.cfg.n_sbs
        features = np.zeros((B, 2 + 2 + 2 * self.embed_dim), dtype=np.float32)

        cache_tensor = torch.as_tensor(self.cache_items, dtype=torch.long, device=self._device)
        cache_emb = self._embed_weight[cache_tensor].mean(dim=1).detach().cpu().numpy()  # (B, E)

        for b in range(B):
            pos = sbs_positions[b] / self.cfg.grid_size
            ue_count = float(np.sum(association == b)) / max(1, self.cfg.n_ues)
            hit = float(self.last_hit_rate[b])

            cand = self.current_candidates[b]
            mask = self.current_mask[b]
            if np.any(mask):
                cand_items = torch.as_tensor(cand[mask], dtype=torch.long, device=self._device)
                cand_scores = torch.as_tensor(self.current_candidate_scores[b, mask], dtype=torch.float32, device=self._device)
                cand_scores = cand_scores / (cand_scores.sum() + 1e-8)
                cand_emb = (self._embed_weight[cand_items] * cand_scores[:, None]).sum(dim=0).detach().cpu().numpy()
            else:
                cand_emb = np.zeros((self.embed_dim,), dtype=np.float32)

            features[b, 0:2] = pos
            features[b, 2] = ue_count
            features[b, 3] = hit
            features[b, 4 : 4 + self.embed_dim] = cache_emb[b]
            features[b, 4 + self.embed_dim :] = cand_emb
        return features

    def _build_observation(self) -> dict[str, np.ndarray]:
        ue_pos = self.ue_positions[self.step_idx]
        sbs_pos = self._sbs_positions_at(self.step_idx)
        association = self._associate_ues_to_sbs(ue_pos, sbs_pos)
        self._refresh_candidates(association)
        self.current_adjacency = self._adjacency_from_sbs(sbs_pos)
        node_features = self._build_node_features(association, sbs_pos)
        return {
            "node_features": node_features,
            "adjacency": self.current_adjacency.copy(),
            "action_mask": self.current_mask.astype(np.float32).copy(),
            "association": association.astype(np.int64),
        }

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_idx = 0
        sampled_users = self.rng.choice(np.asarray(self._eligible_users, dtype=np.int64), size=self.cfg.n_ues, replace=False)
        self.user_ids_for_ues = sampled_users.astype(np.int64)
        self.user_ptrs = np.zeros((self.cfg.n_ues,), dtype=np.int64)
        for i, user_id in enumerate(self.user_ids_for_ues):
            seq_len = len(self.user_histories[int(user_id)])
            start = int(self.rng.integers(self.cfg.window_size, max(self.cfg.window_size + 1, seq_len - 1)))
            self.user_ptrs[i] = start

        self.last_hit_rate.fill(0.0)
        self._build_observation()
        self._initialize_caches()
        return self._build_observation()

    def _apply_actions(self, action_idx: np.ndarray) -> np.ndarray:
        self.cache_hits *= self.cfg.cache_hit_decay
        replaced_counts = np.zeros((self.cfg.n_sbs,), dtype=np.float64)
        for b in range(self.cfg.n_sbs):
            idx = int(action_idx[b])
            idx = np.clip(idx, 0, self.cfg.fp - 1)
            candidate_order = [idx]
            ranked = np.argsort(self.current_candidate_scores[b])[::-1]
            for j in ranked:
                j = int(j)
                if j == idx:
                    continue
                candidate_order.append(j)

            applied = 0
            for cand_idx in candidate_order:
                if applied >= self.cfg.replacements_per_step:
                    break
                if not self.current_mask[b, cand_idx]:
                    continue
                chosen = int(self.current_candidates[b, cand_idx])
                if chosen <= 0:
                    continue
                if chosen in self.cache_items[b]:
                    continue
                evict_slot = int(np.argmin(self.cache_hits[b]))
                self.cache_items[b, evict_slot] = chosen
                self.cache_hits[b, evict_slot] = 0.0
                replaced_counts[b] += 1.0
                applied += 1
        return replaced_counts

    def _compute_reward(self, association: np.ndarray, replaced: np.ndarray) -> tuple[float, dict[str, float]]:
        sbs_pos = self._sbs_positions_at(self.step_idx)
        adjacency = self._adjacency_from_sbs(sbs_pos)

        local_hits = np.zeros((self.cfg.n_sbs,), dtype=np.float64)
        neighbor_hits = np.zeros((self.cfg.n_sbs,), dtype=np.float64)
        cloud_hits = np.zeros((self.cfg.n_sbs,), dtype=np.float64)
        totals = np.zeros((self.cfg.n_sbs,), dtype=np.float64)

        for ue in range(self.cfg.n_ues):
            b = int(association[ue])
            user_id = int(self.user_ids_for_ues[ue])
            ptr = int(self.user_ptrs[ue])
            seq = self.user_histories[user_id]
            requested = int(seq[ptr])
            totals[b] += 1.0

            cache = self.cache_items[b]
            if requested in cache:
                local_hits[b] += 1.0
                hit_slot = int(np.where(cache == requested)[0][0])
                self.cache_hits[b, hit_slot] += 1.0
            else:
                neigh = np.where(adjacency[b] > 0.0)[0]
                neigh = neigh[neigh != b]
                found_neighbor = False
                for n in neigh:
                    if requested in self.cache_items[int(n)]:
                        found_neighbor = True
                        break
                if found_neighbor:
                    neighbor_hits[b] += 1.0
                else:
                    cloud_hits[b] += 1.0

            next_ptr = ptr + 1
            if next_ptr >= len(seq):
                next_ptr = self.cfg.window_size
            self.user_ptrs[ue] = next_ptr

        baseline = self.cfg.chi_cloud * totals
        actual = (
            self.cfg.alpha_local * local_hits
            + self.cfg.beta_neighbor * neighbor_hits
            + self.cfg.chi_cloud * cloud_hits
            + self.cfg.delta_replace * replaced
        )
        reward_per_sbs = baseline - actual
        reward = float(np.sum(reward_per_sbs))

        denom = np.maximum(1.0, totals)
        self.last_hit_rate = local_hits / denom
        info = {
            "reward": reward,
            "local_hit_rate": float(np.sum(local_hits) / max(1.0, np.sum(totals))),
            "neighbor_fetch_rate": float(np.sum(neighbor_hits) / max(1.0, np.sum(totals))),
            "cloud_fetch_rate": float(np.sum(cloud_hits) / max(1.0, np.sum(totals))),
            "replace_count": float(np.sum(replaced)),
        }
        return reward, info

    def step(self, action_idx: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, float]]:
        action_idx = np.asarray(action_idx, dtype=np.int64)
        if action_idx.shape != (self.cfg.n_sbs,):
            raise ValueError(f"Expected action shape ({self.cfg.n_sbs},), got {action_idx.shape}")

        obs_before = self._build_observation()
        association = obs_before["association"]
        replaced = self._apply_actions(action_idx)
        reward, info = self._compute_reward(association, replaced)

        self.step_idx += 1
        done = self.step_idx >= self.cfg.episode_len
        if done:
            next_obs = obs_before
        else:
            next_obs = self._build_observation()
        return next_obs, reward, done, info
