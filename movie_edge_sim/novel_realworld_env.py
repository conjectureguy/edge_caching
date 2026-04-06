from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from movie_edge_sim.simulation import SimulationConfig, run_simulation
from movie_edge_sim.temporal_realworld import RealWorldTemporalEncoder, UserTimeHistory


@dataclass
class RealWorldEnvConfig:
    n_sbs: int = 8
    n_ues: int = 220
    cache_capacity: int = 20
    fp: int = 50
    window_size: int = 10
    episode_len: int = 120
    grid_size: float = 300.0
    ue_max_speed: float = 1.2
    sbs_max_speed: float = 0.35
    sbs_update_interval: float = 10.0
    neighbor_radius: float = 130.0
    alpha_local: float = 1.0
    beta_neighbor: float = 4.0
    chi_cloud: float = 5.0
    delta_replace: float = 0.1
    temporal_weight: float = 0.55
    recency_weight: float = 0.20
    trend_weight: float = 0.15
    popularity_weight: float = 0.10
    active_base_scale: float = 0.22
    active_hour_scale: float = 1.2
    trend_refresh_steps: int = 8
    trend_decay: float = 0.85
    teacher_diversity_penalty: float = 0.45
    teacher_locality_bonus: float = 1.1
    teacher_use_ensemble: bool = True
    teacher_base_weight: float = 0.45
    teacher_attention_weight: float = 0.20
    teacher_mobility_weight: float = 0.20
    teacher_ddpg_weight: float = 0.15
    use_temporal_features: bool = True
    use_mobility_features: bool = True
    use_trend_features: bool = True
    use_semantic_features: bool = True
    freshness_tau_hours: float = 6.0
    semantic_score_weight: float = 0.10
    semantic_future_weight: float = 0.05
    freshness_score_weight: float = 0.08
    seed: int = 42


class NovelRealWorldCachingEnv:
    def __init__(
        self,
        cfg: RealWorldEnvConfig,
        temporal_model: RealWorldTemporalEncoder,
        histories: dict[int, UserTimeHistory],
        item_genres: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.temporal_model = temporal_model.eval()
        self.histories = histories
        self.rng = np.random.default_rng(cfg.seed)
        self._device = next(self.temporal_model.parameters()).device
        self._embed_weight = self.temporal_model.embed.weight.detach().to(self._device)
        self.embed_dim = int(self._embed_weight.shape[1])
        self.num_items = int(self.temporal_model.num_items)
        if item_genres is None or item_genres.size == 0:
            self.item_genres = np.zeros((self.num_items + 1, 1), dtype=np.float32)
        else:
            trimmed = item_genres[: self.num_items + 1]
            if trimmed.shape[0] < self.num_items + 1:
                pad = np.zeros((self.num_items + 1 - trimmed.shape[0], trimmed.shape[1]), dtype=trimmed.dtype)
                trimmed = np.concatenate([trimmed, pad], axis=0)
            row_sums = np.maximum(trimmed.sum(axis=1, keepdims=True), 1e-8)
            self.item_genres = (trimmed / row_sums).astype(np.float32)
        self.num_genres = int(self.item_genres.shape[1])

        eligible = [u for u, h in histories.items() if len(h.items) > cfg.window_size + 5]
        if len(eligible) < cfg.n_ues:
            raise ValueError(f"Need at least {cfg.n_ues} eligible users, found {len(eligible)}")
        self._eligible_users = np.asarray(eligible, dtype=np.int64)

        self.global_popularity = self._build_global_popularity()
        self.hour_profile = self._build_hour_profile()
        self.user_mean_log_delta = self._build_user_delta_profile()
        self.user_base_activity = self._build_user_activity_profile()
        self._build_mobility()

        self.step_idx = 0
        self.user_ids_for_ues = np.zeros((cfg.n_ues,), dtype=np.int64)
        self.cache_items = np.zeros((cfg.n_sbs, cfg.cache_capacity), dtype=np.int64)
        self.cache_hits = np.zeros((cfg.n_sbs, cfg.cache_capacity), dtype=np.float64)
        self.last_hit_rate = np.zeros((cfg.n_sbs,), dtype=np.float64)
        self.current_adjacency = np.eye(cfg.n_sbs, dtype=np.float32)
        self.current_candidates = np.zeros((cfg.n_sbs, cfg.fp), dtype=np.int64)
        self.current_candidate_scores = np.zeros((cfg.n_sbs, cfg.fp), dtype=np.float64)
        self.current_mask = np.zeros((cfg.n_sbs, cfg.fp), dtype=bool)
        self.current_neighbor_support = np.zeros((cfg.n_sbs, cfg.fp), dtype=np.float64)
        self.current_neighbor_shortage = np.zeros((cfg.n_sbs, cfg.fp), dtype=np.float64)
        self.current_trend_items = np.zeros((cfg.n_sbs,), dtype=np.int64)
        self.current_trend_strength = np.zeros((cfg.n_sbs,), dtype=np.float64)
        self.current_future_load = np.zeros((cfg.n_sbs,), dtype=np.float64)
        self.current_sbs_velocity = np.zeros((cfg.n_sbs, 2), dtype=np.float64)
        self.current_semantic_affinity = np.zeros((cfg.n_sbs, cfg.fp), dtype=np.float64)
        self.current_freshness_relevance = np.zeros((cfg.n_sbs, cfg.fp), dtype=np.float64)

        self.ue_context_items = np.zeros((cfg.n_ues, cfg.window_size), dtype=np.int64)
        self.ue_context_deltas = np.zeros((cfg.n_ues, cfg.window_size), dtype=np.float32)
        self.ue_context_hours = np.zeros((cfg.n_ues, cfg.window_size), dtype=np.int64)
        self.ue_last_hour = np.zeros((cfg.n_ues,), dtype=np.int64)
        self.last_requests = np.zeros((cfg.n_ues,), dtype=np.int64)
        self.last_active_mask = np.zeros((cfg.n_ues,), dtype=bool)
        self._cached_obs: dict[str, np.ndarray] | None = None
        self._cached_temporal_probs: np.ndarray | None = None

    def _select_candidate_pool(
        self,
        score: np.ndarray,
        temporal_scores: np.ndarray,
        recent_scores: np.ndarray,
        future_scores: np.ndarray,
        cache_items: np.ndarray,
        trend_item: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        fp = self.cfg.fp
        score = score.copy()
        score[0] = 0.0
        temporal_scores = temporal_scores.copy()
        temporal_scores[0] = 0.0
        recent_scores = recent_scores.copy()
        recent_scores[0] = 0.0
        future_scores = future_scores.copy()
        future_scores[0] = 0.0

        top_score = np.argsort(score[1:])[-fp:][::-1] + 1
        top_temporal = np.argsort(temporal_scores[1:])[-max(self.cfg.cache_capacity, fp // 3) :][::-1] + 1
        top_recent = np.argsort(recent_scores[1:])[-max(self.cfg.cache_capacity, fp // 4) :][::-1] + 1
        top_future = np.argsort(future_scores[1:])[-max(4, fp // 5) :][::-1] + 1
        top_pop = np.argsort(self.global_popularity[1:])[-max(4, self.cfg.cache_capacity // 2) :][::-1] + 1

        candidate_pool: set[int] = set(int(item) for item in top_score.tolist())
        candidate_pool.update(int(item) for item in top_temporal.tolist())
        candidate_pool.update(int(item) for item in top_recent.tolist())
        candidate_pool.update(int(item) for item in top_future.tolist())
        candidate_pool.update(int(item) for item in top_pop.tolist())
        candidate_pool.update(int(item) for item in cache_items.tolist() if int(item) > 0)
        if trend_item > 0:
            candidate_pool.add(int(trend_item))

        pool = np.asarray(sorted(candidate_pool), dtype=np.int64)
        if pool.size == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)

        score_scale = max(float(np.max(score[pool])), 1e-8)
        rescue_bonus = np.zeros((pool.shape[0],), dtype=np.float64)
        rescue_bonus += 0.10 * score_scale * np.isin(pool, top_temporal)
        rescue_bonus += 0.09 * score_scale * np.isin(pool, top_recent)
        rescue_bonus += 0.07 * score_scale * np.isin(pool, top_future)
        rescue_bonus += 0.05 * score_scale * np.isin(pool, cache_items)
        if trend_item > 0:
            rescue_bonus += 0.04 * score_scale * (pool == int(trend_item))

        pooled_scores = score[pool] + rescue_bonus
        order = np.argsort(pooled_scores)[::-1]
        chosen = pool[order[:fp]]
        chosen_scores = pooled_scores[order[:fp]]
        return chosen, chosen_scores

    def _build_global_popularity(self) -> np.ndarray:
        popularity = np.zeros((self.num_items + 1,), dtype=np.float64)
        for hist in self.histories.values():
            for item in hist.items:
                if 0 < item <= self.num_items:
                    popularity[item] += 1.0
        popularity /= max(popularity.sum(), 1.0)
        return popularity

    def _build_hour_profile(self) -> np.ndarray:
        hist = np.ones((24,), dtype=np.float64)
        for user_hist in self.histories.values():
            for ts in user_hist.timestamps:
                hist[int((ts // 3600) % 24)] += 1.0
        hist /= hist.mean()
        return hist

    def _build_user_delta_profile(self) -> dict[int, float]:
        profiles: dict[int, float] = {}
        for user_id, hist in self.histories.items():
            if len(hist.timestamps) < 2:
                profiles[user_id] = 0.0
                continue
            deltas = np.diff(np.asarray(hist.timestamps, dtype=np.float64)) / 3600.0
            profiles[user_id] = float(np.mean(np.log1p(np.maximum(deltas, 0.0))))
        return profiles

    def _build_user_activity_profile(self) -> dict[int, float]:
        lengths = np.asarray([len(hist.items) for hist in self.histories.values()], dtype=np.float64)
        lo = float(np.min(lengths))
        hi = float(np.max(lengths))
        profiles: dict[int, float] = {}
        for user_id, hist in self.histories.items():
            norm = (len(hist.items) - lo) / max(1e-6, hi - lo)
            profiles[user_id] = 0.35 + 0.65 * float(norm)
        return profiles

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
        sim = run_simulation(sim_cfg)
        self.ue_positions = sim.ue_positions_over_time
        self.sbs_positions = sim.sbs_positions_over_time
        self.update_stride = int(round(self.cfg.sbs_update_interval))

    def _sbs_positions_at(self, step: int) -> np.ndarray:
        idx = min(step // max(1, self.update_stride), self.sbs_positions.shape[0] - 1)
        return self.sbs_positions[idx]

    def _associate_ues(self, ue_positions: np.ndarray, sbs_positions: np.ndarray) -> np.ndarray:
        d2 = ((ue_positions[:, None, :] - sbs_positions[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1).astype(np.int64)

    def _adjacency(self, sbs_positions: np.ndarray) -> np.ndarray:
        d = np.sqrt(((sbs_positions[:, None, :] - sbs_positions[None, :, :]) ** 2).sum(axis=2))
        adj = (d <= self.cfg.neighbor_radius).astype(np.float32)
        np.fill_diagonal(adj, 1.0)
        return adj

    def _future_state(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.cfg.use_mobility_features:
            current_sbs = self._sbs_positions_at(self.step_idx)
            current_ue = self.ue_positions[self.step_idx]
            current_assoc = self._associate_ues(current_ue, current_sbs)
            self.current_future_load = np.asarray(
                [float(np.mean(current_assoc == b)) for b in range(self.cfg.n_sbs)],
                dtype=np.float64,
            )
            self.current_sbs_velocity.fill(0.0)
            return current_sbs, current_assoc
        next_step = min(self.step_idx + 1, self.ue_positions.shape[0] - 1)
        next_ue = self.ue_positions[next_step]
        next_sbs = self._sbs_positions_at(next_step)
        future_assoc = self._associate_ues(next_ue, next_sbs)
        self.current_future_load = np.asarray(
            [float(np.mean(future_assoc == b)) for b in range(self.cfg.n_sbs)],
            dtype=np.float64,
        )
        current_sbs = self._sbs_positions_at(self.step_idx)
        self.current_sbs_velocity = (next_sbs - current_sbs) / max(1.0, float(next_step - self.step_idx))
        return next_sbs, future_assoc

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_idx = 0
        self.user_ids_for_ues = self.rng.choice(self._eligible_users, size=self.cfg.n_ues, replace=False)
        self.cache_hits.fill(0.0)
        self.last_hit_rate.fill(0.0)
        self.current_trend_items.fill(0)
        self.current_trend_strength.fill(0.0)
        self.last_requests.fill(0)
        self.last_active_mask.fill(False)
        self._cached_temporal_probs = None

        for ue, user_id in enumerate(self.user_ids_for_ues.tolist()):
            hist = self.histories[int(user_id)]
            start = int(self.rng.integers(self.cfg.window_size, len(hist.items) - 1))
            items = hist.items[start - self.cfg.window_size : start]
            ts = hist.timestamps[start - self.cfg.window_size : start]
            deltas = [0.0]
            for i in range(1, len(ts)):
                deltas.append(np.log1p(max(0.0, (ts[i] - ts[i - 1]) / 3600.0)))
            hours = [int((stamp // 3600) % 24) for stamp in ts]
            self.ue_context_items[ue] = np.asarray(items, dtype=np.int64)
            self.ue_context_deltas[ue] = np.asarray(deltas, dtype=np.float32)
            self.ue_context_hours[ue] = np.asarray(hours, dtype=np.int64)
            self.ue_last_hour[ue] = int(hours[-1])

        obs = self._build_observation()
        self.cache_items[:] = self.cooperative_teacher_action()
        self._cached_obs = self._build_observation()
        return self._cached_obs

    @torch.no_grad()
    def _temporal_probs_for_ues(self, ue_indices: np.ndarray) -> np.ndarray:
        if ue_indices.size == 0 or not self.cfg.use_temporal_features:
            return np.zeros((0, self.num_items + 1), dtype=np.float64)
        if self._cached_temporal_probs is None:
            items = torch.as_tensor(self.ue_context_items, dtype=torch.long, device=self._device)
            deltas = torch.as_tensor(self.ue_context_deltas, dtype=torch.float32, device=self._device)
            hours = torch.as_tensor(self.ue_context_hours, dtype=torch.long, device=self._device)
            users = torch.as_tensor(self.user_ids_for_ues, dtype=torch.long, device=self._device)
            probs = self.temporal_model.predict_scores(items, deltas, hours, users)
            self._cached_temporal_probs = probs.detach().cpu().numpy().astype(np.float64)
        return self._cached_temporal_probs[ue_indices]

    def _freshness_weights(self, ue_indices: np.ndarray) -> np.ndarray:
        if ue_indices.size == 0:
            return np.zeros((0,), dtype=np.float64)
        if not self.cfg.use_temporal_features:
            return np.ones((ue_indices.size,), dtype=np.float64)
        last_deltas = np.expm1(self.ue_context_deltas[ue_indices, -1].astype(np.float64))
        weights = np.exp(-last_deltas / max(1e-6, self.cfg.freshness_tau_hours))
        return np.clip(weights, 0.05, 1.0)

    def _semantic_profile(self, ue_indices: np.ndarray) -> np.ndarray:
        if ue_indices.size == 0 or not self.cfg.use_semantic_features:
            return np.zeros((self.num_genres,), dtype=np.float64)
        profile = np.zeros((self.num_genres,), dtype=np.float64)
        freshness = self._freshness_weights(ue_indices)
        recency_kernel = np.linspace(0.55, 1.0, self.cfg.window_size, dtype=np.float64)
        for idx, ue in enumerate(ue_indices.tolist()):
            items = self.ue_context_items[int(ue)]
            for pos, item in enumerate(items.tolist()):
                if item <= 0 or item >= self.item_genres.shape[0]:
                    continue
                profile += freshness[idx] * recency_kernel[pos] * self.item_genres[item]
        total = float(profile.sum())
        if total <= 1e-8:
            return profile
        return profile / total

    def _refresh_trends(self, association: np.ndarray) -> None:
        if not self.cfg.use_trend_features:
            self.current_trend_items.fill(0)
            self.current_trend_strength.fill(0.0)
            return
        self.current_trend_strength *= self.cfg.trend_decay
        if self.step_idx % max(1, self.cfg.trend_refresh_steps) != 0:
            return
        for b in range(self.cfg.n_sbs):
            ue_idx = np.where(association == b)[0]
            probs = self._temporal_probs_for_ues(ue_idx)
            if probs.shape[0] == 0:
                cand = np.argsort(self.global_popularity[1:])[-10:] + 1
                weights = self.global_popularity[cand]
            else:
                agg = probs.sum(axis=0)
                agg[0] = 0.0
                cand = np.argsort(agg[1:])[-10:] + 1
                weights = agg[cand]
            weights = weights / max(weights.sum(), 1e-8)
            self.current_trend_items[b] = int(self.rng.choice(cand, p=weights))
            self.current_trend_strength[b] = 1.0

    def _sample_requests(self, association: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hour = int(self.step_idx % 24)
        requests = np.zeros((self.cfg.n_ues,), dtype=np.int64)
        active = np.zeros((self.cfg.n_ues,), dtype=bool)
        probs = self._temporal_probs_for_ues(np.arange(self.cfg.n_ues, dtype=np.int64))
        for ue in range(self.cfg.n_ues):
            user_id = int(self.user_ids_for_ues[ue])
            act_p = self.cfg.active_base_scale * self.user_base_activity[user_id] * (
                self.cfg.active_hour_scale * self.hour_profile[hour] / max(np.max(self.hour_profile), 1.0)
            )
            act_p = float(np.clip(act_p, 0.03, 0.92))
            if self.rng.random() > act_p:
                continue
            active[ue] = True
            if self.cfg.use_temporal_features:
                base = probs[ue].copy()
                base[0] = 0.0
            else:
                base = np.zeros((self.num_items + 1,), dtype=np.float64)
            recency = np.bincount(self.ue_context_items[ue], minlength=self.num_items + 1).astype(np.float64)
            recency /= max(recency.sum(), 1.0)
            trend = np.zeros((self.num_items + 1,), dtype=np.float64)
            trend_item = int(self.current_trend_items[int(association[ue])])
            if self.cfg.use_trend_features and trend_item > 0:
                trend[trend_item] = float(self.current_trend_strength[int(association[ue])])
            mix = (
                self.cfg.temporal_weight * base
                + self.cfg.recency_weight * recency
                + self.cfg.trend_weight * trend
                + self.cfg.popularity_weight * self.global_popularity
            )
            mix[0] = 0.0
            mix /= max(mix.sum(), 1e-8)
            requests[ue] = int(self.rng.choice(np.arange(self.num_items + 1), p=mix))
        return requests, active

    def _advance_user_context(self, ue: int, item_id: int) -> None:
        user_id = int(self.user_ids_for_ues[ue])
        mean_log_delta = self.user_mean_log_delta.get(user_id, 0.0)
        sampled_delta = float(np.exp(self.rng.normal(mean_log_delta, 0.35)) - 1.0)
        sampled_delta = float(np.clip(sampled_delta, 0.0, 24.0))
        next_hour = int((self.ue_last_hour[ue] + max(1, int(round(sampled_delta)))) % 24)

        self.ue_context_items[ue, :-1] = self.ue_context_items[ue, 1:]
        self.ue_context_items[ue, -1] = item_id
        self.ue_context_deltas[ue, :-1] = self.ue_context_deltas[ue, 1:]
        self.ue_context_deltas[ue, -1] = np.log1p(sampled_delta)
        self.ue_context_hours[ue, :-1] = self.ue_context_hours[ue, 1:]
        self.ue_context_hours[ue, -1] = next_hour
        self.ue_last_hour[ue] = next_hour

    def _refresh_candidates(self, association: np.ndarray, future_association: np.ndarray) -> None:
        self.current_candidates.fill(0)
        self.current_candidate_scores.fill(0.0)
        self.current_mask.fill(False)
        self.current_semantic_affinity.fill(0.0)
        self.current_freshness_relevance.fill(0.0)
        for b in range(self.cfg.n_sbs):
            ue_idx = np.where(association == b)[0]
            future_ue_idx = np.where(future_association == b)[0]
            probs = self._temporal_probs_for_ues(ue_idx)
            future_probs = self._temporal_probs_for_ues(future_ue_idx)
            current_freshness = self._freshness_weights(ue_idx)
            future_freshness = self._freshness_weights(future_ue_idx)
            current_profile = self._semantic_profile(ue_idx)
            future_profile = self._semantic_profile(future_ue_idx)
            active_recent = np.zeros((self.num_items + 1,), dtype=np.float64)
            agg = np.zeros((self.num_items + 1,), dtype=np.float64)
            future_agg = np.zeros((self.num_items + 1,), dtype=np.float64)
            trend_item = int(self.current_trend_items[b])
            if probs.shape[0] == 0 and not self.cfg.use_temporal_features:
                active_recent = np.bincount(self.ue_context_items[ue_idx, -1], minlength=self.num_items + 1).astype(np.float64)
                active_recent /= max(active_recent.sum(), 1.0)
                trend = np.zeros((self.num_items + 1,), dtype=np.float64)
                if self.cfg.use_trend_features and trend_item > 0:
                    trend[trend_item] = float(self.current_trend_strength[b])
                semantic = np.zeros((self.num_items + 1,), dtype=np.float64)
                if self.cfg.use_semantic_features and np.any(current_profile):
                    semantic = self.item_genres @ current_profile
                score = 0.62 * active_recent + 0.18 * self.global_popularity + 0.10 * trend + self.cfg.semantic_score_weight * semantic
            elif probs.shape[0] == 0:
                score = self.global_popularity.copy()
            else:
                if current_freshness.size > 0:
                    agg = (probs * current_freshness[:, None]).sum(axis=0)
                else:
                    agg = probs.sum(axis=0)
                agg[0] = 0.0
                recent_items = self.ue_context_items[ue_idx, -3:].reshape(-1) if ue_idx.size > 0 else np.zeros((0,), dtype=np.int64)
                if recent_items.size > 0:
                    recent_weights = np.repeat(np.maximum(current_freshness, 1e-6), 3) * np.tile(np.asarray([0.6, 0.8, 1.0], dtype=np.float64), len(ue_idx))
                    active_recent = np.bincount(recent_items, weights=recent_weights, minlength=self.num_items + 1).astype(np.float64)
                else:
                    active_recent = np.zeros((self.num_items + 1,), dtype=np.float64)
                active_recent /= max(active_recent.sum(), 1.0)
                if self.cfg.use_mobility_features and future_probs.shape[0] > 0:
                    if future_freshness.size > 0:
                        future_agg = (future_probs * future_freshness[:, None]).sum(axis=0)
                    else:
                        future_agg = future_probs.sum(axis=0)
                    future_agg[0] = 0.0
                trend = np.zeros((self.num_items + 1,), dtype=np.float64)
                if self.cfg.use_trend_features and trend_item > 0:
                    trend[trend_item] = float(self.current_trend_strength[b])
                semantic = np.zeros((self.num_items + 1,), dtype=np.float64)
                if self.cfg.use_semantic_features and np.any(current_profile):
                    semantic += self.item_genres @ current_profile
                if self.cfg.use_semantic_features and np.any(future_profile):
                    semantic += self.cfg.semantic_future_weight * (self.item_genres @ future_profile)
                freshness_recent = active_recent.copy()
                current_strength = 0.50 + 0.10 * float(np.mean(current_freshness)) if current_freshness.size > 0 else 0.45
                future_strength = (0.10 + 0.10 * float(np.mean(future_freshness))) if future_freshness.size > 0 else 0.0
                score = (
                    current_strength * agg
                    + 0.18 * active_recent
                    + (future_strength * future_agg if self.cfg.use_mobility_features else 0.0)
                    + 0.10 * self.global_popularity
                    + (0.05 * trend if self.cfg.use_trend_features else 0.0)
                    + (self.cfg.semantic_score_weight * semantic if self.cfg.use_semantic_features else 0.0)
                    + self.cfg.freshness_score_weight * freshness_recent
                )
            top, scores = self._select_candidate_pool(
                score=score,
                temporal_scores=agg,
                recent_scores=active_recent,
                future_scores=future_agg,
                cache_items=self.cache_items[b],
                trend_item=trend_item,
            )
            valid = min(self.cfg.fp, top.shape[0])
            self.current_candidates[b, :valid] = top[:valid]
            self.current_candidate_scores[b, :valid] = scores[:valid]
            self.current_mask[b, :valid] = True
            if valid > 0:
                items = top[:valid]
                if self.cfg.use_semantic_features and np.any(current_profile):
                    self.current_semantic_affinity[b, :valid] = (self.item_genres[items] @ current_profile).astype(np.float64)
                fresh_vals = active_recent[items]
                if np.any(fresh_vals):
                    fresh_vals = fresh_vals / max(float(np.max(fresh_vals)), 1e-8)
                self.current_freshness_relevance[b, :valid] = fresh_vals

    def _build_node_features(self, association: np.ndarray, sbs_positions: np.ndarray) -> np.ndarray:
        feat_dim = 9 + 2 * self.embed_dim
        features = np.zeros((self.cfg.n_sbs, feat_dim), dtype=np.float32)
        cache_emb = self._embed_weight[torch.as_tensor(self.cache_items, dtype=torch.long, device=self._device)].mean(dim=1)
        cache_emb = cache_emb.detach().cpu().numpy()
        for b in range(self.cfg.n_sbs):
            pos = sbs_positions[b] / self.cfg.grid_size
            ue_frac = float(np.sum(association == b) / max(1, self.cfg.n_ues))
            hit = float(self.last_hit_rate[b])
            trend_strength = float(self.current_trend_strength[b]) if self.cfg.use_trend_features else 0.0
            hour = float((self.step_idx % 24) / 24.0)
            sbs_vel = self.current_sbs_velocity[b] / max(1.0, self.cfg.grid_size) if self.cfg.use_mobility_features else 0.0
            future_load = float(self.current_future_load[b]) if self.cfg.use_mobility_features else 0.0
            cand_mask = self.current_mask[b]
            if np.any(cand_mask):
                cand_items = torch.as_tensor(self.current_candidates[b, cand_mask], dtype=torch.long, device=self._device)
                weights = torch.as_tensor(self.current_candidate_scores[b, cand_mask], dtype=torch.float32, device=self._device)
                weights = weights / (weights.sum() + 1e-8)
                cand_emb = (self._embed_weight[cand_items] * weights[:, None]).sum(dim=0).detach().cpu().numpy()
            else:
                cand_emb = np.zeros((self.embed_dim,), dtype=np.float32)
            features[b, 0:2] = pos
            features[b, 2] = ue_frac
            features[b, 3] = hit
            features[b, 4] = trend_strength
            features[b, 5] = hour
            features[b, 6:8] = sbs_vel
            features[b, 8] = future_load
            features[b, 9 : 9 + self.embed_dim] = cache_emb[b]
            features[b, 9 + self.embed_dim :] = cand_emb
        return features

    def _build_candidate_features(self) -> np.ndarray:
        feat_dim = self.embed_dim + 10
        features = np.zeros((self.cfg.n_sbs, self.cfg.fp, feat_dim), dtype=np.float32)
        self.current_neighbor_support.fill(0.0)
        self.current_neighbor_shortage.fill(0.0)
        for b in range(self.cfg.n_sbs):
            neigh = np.where(self.current_adjacency[b] > 0.0)[0]
            neigh = neigh[neigh != b]
            neigh_count = max(1, len(neigh))
            valid_slots = np.where(self.current_mask[b])[0]
            if valid_slots.size == 0:
                continue
            item_ids = self.current_candidates[b, valid_slots]
            scores = self.current_candidate_scores[b, valid_slots]
            score_norm = scores / max(np.sum(np.maximum(scores, 0.0)), 1e-8)
            item_emb = self._embed_weight[torch.as_tensor(item_ids, dtype=torch.long, device=self._device)].detach().cpu().numpy()
            for idx, slot in enumerate(valid_slots):
                item = int(item_ids[idx])
                neighbor_support = 0.0
                for n in neigh:
                    n_slots = np.where(self.current_candidates[int(n)] == item)[0]
                    if n_slots.size == 0:
                        continue
                    n_slot = int(n_slots[0])
                    neighbor_support += float(self.current_candidate_scores[int(n), n_slot])
                neighbor_support /= float(neigh_count)
                neighbor_overlap = float(sum(item in self.cache_items[int(n)] for n in neigh) / neigh_count)
                shortage = neighbor_support * max(0.0, 1.0 - neighbor_overlap)
                self.current_neighbor_support[b, slot] = neighbor_support
                self.current_neighbor_shortage[b, slot] = shortage
                features[b, slot, 0] = float(score_norm[idx])
                features[b, slot, 1] = float(item in self.cache_items[b])
                features[b, slot, 2] = neighbor_overlap
                features[b, slot, 3] = float(self.global_popularity[item])
                features[b, slot, 4] = (
                    float(item == int(self.current_trend_items[b])) * float(self.current_trend_strength[b])
                    if self.cfg.use_trend_features
                    else 0.0
                )
                features[b, slot, 5] = float(self.current_future_load[b]) if self.cfg.use_mobility_features else 0.0
                features[b, slot, 6] = float(neighbor_support)
                features[b, slot, 7] = float(shortage)
                features[b, slot, 8] = float(self.current_semantic_affinity[b, slot])
                features[b, slot, 9] = float(self.current_freshness_relevance[b, slot])
                features[b, slot, 10:] = item_emb[idx]
        return features

    def _build_observation(self) -> dict[str, np.ndarray]:
        ue_pos = self.ue_positions[self.step_idx]
        sbs_pos = self._sbs_positions_at(self.step_idx)
        association = self._associate_ues(ue_pos, sbs_pos)
        _, future_association = self._future_state()
        self.current_adjacency = self._adjacency(sbs_pos)
        self._refresh_trends(association)
        self._refresh_candidates(association, future_association)
        return {
            "node_features": self._build_node_features(association, sbs_pos),
            "candidate_features": self._build_candidate_features(),
            "adjacency": self.current_adjacency.copy(),
            "action_mask": self.current_mask.astype(np.float32).copy(),
            "association": association.copy(),
        }

    def get_observation(self) -> dict[str, np.ndarray]:
        if self._cached_obs is None:
            self._cached_obs = self._build_observation()
        return self._cached_obs

    def _teacher_scores_and_actions(self) -> tuple[np.ndarray, np.ndarray]:
        scores_all = np.full((self.cfg.n_sbs, self.cfg.fp), -1e9, dtype=np.float32)
        actions = np.zeros((self.cfg.n_sbs, self.cfg.cache_capacity), dtype=np.int64)
        planned: list[set[int]] = [set() for _ in range(self.cfg.n_sbs)]
        order = np.argsort(self.current_candidate_scores.sum(axis=1))[::-1]
        cand_all = self._build_candidate_features() if self.cfg.teacher_use_ensemble else None
        for b in order.tolist():
            neigh = np.where(self.current_adjacency[b] > 0.0)[0]
            neigh = neigh[neigh != b]
            for slot in np.where(self.current_mask[b])[0].tolist():
                item = int(self.current_candidates[b, slot])
                local_score = float(self.current_candidate_scores[b, slot])
                overlap = sum(item in planned[int(n)] for n in neigh)
                in_cache = float(item in self.cache_items[b])
                trend_bonus = float(item == int(self.current_trend_items[b])) * float(self.current_trend_strength[b])
                score = (
                    self.cfg.teacher_locality_bonus * local_score
                    + 0.15 * float(self.global_popularity[item])
                    + 0.20 * trend_bonus
                    - self.cfg.teacher_diversity_penalty * float(overlap)
                    + 0.20 * in_cache
                    + 0.16 * float(self.current_semantic_affinity[b, slot])
                    + 0.12 * float(self.current_freshness_relevance[b, slot])
                )
                scores_all[b, slot] = float(score)

            row_scores = scores_all[b].copy()
            if self.cfg.teacher_use_ensemble:
                cand = cand_all[b]
                attn_scores = np.full((self.cfg.fp,), -1e9, dtype=np.float32)
                mobility_scores = np.full((self.cfg.fp,), -1e9, dtype=np.float32)
                ddpg_scores = np.full((self.cfg.fp,), -1e9, dtype=np.float32)
                for slot in np.where(self.current_mask[b])[0].tolist():
                    attn_scores[slot] = float(
                        0.40 * cand[slot, 0]
                        + 0.15 * cand[slot, 3]
                        + 0.12 * cand[slot, 6]
                        + 0.08 * cand[slot, 7]
                        + 0.10 * cand[slot, 8]
                        + 0.10 * cand[slot, 1]
                        - 0.18 * cand[slot, 2]
                    )
                    mobility_scores[slot] = float(
                        0.32 * cand[slot, 0]
                        + 0.18 * cand[slot, 5]
                        + 0.10 * cand[slot, 4]
                        + 0.12 * cand[slot, 7]
                        + 0.10 * cand[slot, 9]
                        + 0.08 * cand[slot, 1]
                        - 0.16 * cand[slot, 2]
                    )
                    ddpg_scores[slot] = float(
                        0.45 * cand[slot, 0]
                        + 0.14 * cand[slot, 3]
                        + 0.10 * cand[slot, 7]
                        + 0.08 * cand[slot, 8]
                        + 0.06 * cand[slot, 1]
                        - 0.15 * cand[slot, 2]
                    )
                row_scores = (
                    self.cfg.teacher_base_weight * scores_all[b]
                    + self.cfg.teacher_attention_weight * attn_scores
                    + self.cfg.teacher_mobility_weight * mobility_scores
                    + self.cfg.teacher_ddpg_weight * ddpg_scores
                )
                scores_all[b] = row_scores

            ranked = np.argsort(row_scores)[::-1]
            chosen = []
            seen: set[int] = set()
            for slot in ranked.tolist():
                item = int(self.current_candidates[b, slot])
                if item <= 0 or item in seen or not self.current_mask[b, slot]:
                    continue
                chosen.append(item)
                seen.add(item)
                if len(chosen) >= self.cfg.cache_capacity:
                    break
            if len(chosen) < self.cfg.cache_capacity:
                fallback = np.argsort(self.global_popularity[1:])[-self.cfg.cache_capacity :][::-1] + 1
                for item in fallback.tolist():
                    if item in seen:
                        continue
                    chosen.append(int(item))
                    seen.add(int(item))
                    if len(chosen) >= self.cfg.cache_capacity:
                        break
            actions[b] = np.asarray(chosen[: self.cfg.cache_capacity], dtype=np.int64)
            planned[b] = set(actions[b].tolist())
        return scores_all, actions

    def cooperative_teacher_action(self) -> np.ndarray:
        _, actions = self._teacher_scores_and_actions()
        return actions

    def cooperative_teacher_scores(self) -> np.ndarray:
        scores, _ = self._teacher_scores_and_actions()
        return scores

    def candidate_items_to_slot_mask(self, chosen_items: np.ndarray) -> np.ndarray:
        target = np.zeros((self.cfg.n_sbs, self.cfg.fp), dtype=np.float32)
        for b in range(self.cfg.n_sbs):
            item_to_slot = {int(item): int(slot) for slot, item in enumerate(self.current_candidates[b]) if item > 0}
            for item in chosen_items[b]:
                slot = item_to_slot.get(int(item))
                if slot is not None:
                    target[b, slot] = 1.0
        return target

    def _set_cache(self, cache_items: np.ndarray) -> np.ndarray:
        replaced = np.zeros((self.cfg.n_sbs,), dtype=np.float64)
        cache_items = np.asarray(cache_items, dtype=np.int64)
        for b in range(self.cfg.n_sbs):
            new_items = []
            seen: set[int] = set()
            for item in cache_items[b]:
                item = int(item)
                if item <= 0 or item in seen:
                    continue
                new_items.append(item)
                seen.add(item)
                if len(new_items) >= self.cfg.cache_capacity:
                    break
            if len(new_items) < self.cfg.cache_capacity:
                for item in (np.argsort(self.global_popularity[1:])[-self.cfg.cache_capacity :][::-1] + 1).tolist():
                    if item in seen:
                        continue
                    new_items.append(int(item))
                    seen.add(int(item))
                    if len(new_items) >= self.cfg.cache_capacity:
                        break
            new_arr = np.asarray(new_items[: self.cfg.cache_capacity], dtype=np.int64)
            replaced[b] = float(np.sum(~np.isin(new_arr, self.cache_items[b])))
            self.cache_items[b] = new_arr
            self.cache_hits[b] *= 0.95
        return replaced

    def step_full_cache_items(self, cache_items: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, float]]:
        obs_before = self.get_observation()
        association = obs_before["association"]
        replaced = self._set_cache(cache_items)
        requests, active = self._sample_requests(association)

        local_hits = np.zeros((self.cfg.n_sbs,), dtype=np.float64)
        neighbor_hits = np.zeros((self.cfg.n_sbs,), dtype=np.float64)
        cloud_hits = np.zeros((self.cfg.n_sbs,), dtype=np.float64)
        totals = np.zeros((self.cfg.n_sbs,), dtype=np.float64)

        for ue in range(self.cfg.n_ues):
            if not active[ue]:
                continue
            b = int(association[ue])
            item = int(requests[ue])
            if item <= 0:
                continue
            totals[b] += 1.0
            if item in self.cache_items[b]:
                local_hits[b] += 1.0
                slot = int(np.where(self.cache_items[b] == item)[0][0])
                self.cache_hits[b, slot] += 1.0
            else:
                neigh = np.where(self.current_adjacency[b] > 0.0)[0]
                neigh = neigh[neigh != b]
                hit_neighbor = False
                for n in neigh:
                    if item in self.cache_items[int(n)]:
                        hit_neighbor = True
                        break
                if hit_neighbor:
                    neighbor_hits[b] += 1.0
                else:
                    cloud_hits[b] += 1.0
            self._advance_user_context(ue, item)

        baseline = self.cfg.chi_cloud * totals
        actual = (
            self.cfg.alpha_local * local_hits
            + self.cfg.beta_neighbor * neighbor_hits
            + self.cfg.chi_cloud * cloud_hits
            + self.cfg.delta_replace * replaced
        )
        reward_per_sbs = baseline - actual
        reward = float(np.sum(reward_per_sbs))
        denom = np.maximum(totals, 1.0)
        self.last_hit_rate = local_hits / denom
        self.last_requests = requests
        self.last_active_mask = active
        self._cached_temporal_probs = None

        self.step_idx += 1
        done = self.step_idx >= self.cfg.episode_len
        next_obs = obs_before if done else self._build_observation()
        self._cached_obs = next_obs
        info = {
            "reward": reward,
            "reward_per_sbs": reward_per_sbs.astype(np.float32),
            "local_hit_rate": float(np.sum(local_hits) / max(np.sum(totals), 1.0)),
            "neighbor_fetch_rate": float(np.sum(neighbor_hits) / max(np.sum(totals), 1.0)),
            "cloud_fetch_rate": float(np.sum(cloud_hits) / max(np.sum(totals), 1.0)),
            "active_request_count": float(np.sum(totals)),
        }
        return next_obs, reward, done, info
