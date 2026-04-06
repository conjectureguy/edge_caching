from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from movie_edge_sim.simulation import SimulationConfig, run_simulation


@dataclass
class PaperEnvConfig:
    n_sbs: int = 8
    n_ues: int = 220
    cache_capacity: int = 20
    fp: int = 50
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
    seed: int = 42


class PaperCEFMRCooperativeEnv:
    def __init__(
        self,
        cfg: PaperEnvConfig,
        user_histories: dict[int, list[int]],
        user_scores: dict[int, np.ndarray],
    ) -> None:
        self.cfg = cfg
        self.user_histories = user_histories
        self.user_scores = user_scores
        self.rng = np.random.default_rng(cfg.seed)

        self.num_items = int(next(iter(user_scores.values())).shape[0])
        self.global_popularity = self._build_global_popularity()
        self._eligible_users = [u for u, seq in user_histories.items() if len(seq) > 2]
        if len(self._eligible_users) < cfg.n_ues:
            raise ValueError(f"Need at least {cfg.n_ues} users with enough history, found {len(self._eligible_users)}")

        self._build_mobility()

        self.step_idx = 0
        self.user_ids_for_ues = np.zeros((cfg.n_ues,), dtype=np.int64)
        self.user_ptrs = np.zeros((cfg.n_ues,), dtype=np.int64)
        self.cache_items = np.zeros((cfg.n_sbs, cfg.cache_capacity), dtype=np.int64)
        self.last_hit_rate = np.zeros((cfg.n_sbs,), dtype=np.float64)

        self.current_candidates = np.zeros((cfg.n_sbs, cfg.fp), dtype=np.int64)
        self.current_candidate_scores = np.zeros((cfg.n_sbs, cfg.fp), dtype=np.float64)
        self.current_adjacency = np.eye(cfg.n_sbs, dtype=np.float32)
        self.current_association = np.zeros((cfg.n_ues,), dtype=np.int64)
        self.current_state = np.zeros((cfg.n_sbs, self.state_dim), dtype=np.float32)

    @property
    def state_dim(self) -> int:
        # Candidate score, cached flag, neighbor overlap, global popularity, plus load and last hit.
        return self.cfg.fp * 4 + 2

    def _build_global_popularity(self) -> np.ndarray:
        pop = np.zeros((self.num_items + 1,), dtype=np.float64)
        for seq in self.user_histories.values():
            for item in seq:
                item = int(item)
                if 1 <= item <= self.num_items:
                    pop[item] += 1.0
        pop /= max(1.0, pop.sum())
        return pop

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

    def _associate(self, ue_positions: np.ndarray, sbs_positions: np.ndarray) -> np.ndarray:
        d2 = ((ue_positions[:, None, :] - sbs_positions[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1).astype(np.int64)

    def _adjacency(self, sbs_positions: np.ndarray) -> np.ndarray:
        d = np.sqrt(((sbs_positions[:, None, :] - sbs_positions[None, :, :]) ** 2).sum(axis=2))
        adj = (d <= self.cfg.neighbor_radius).astype(np.float32)
        np.fill_diagonal(adj, 1.0)
        return adj

    def _aggregate_predicted_popular_contents(self) -> None:
        self.current_candidates.fill(0)
        self.current_candidate_scores.fill(0.0)
        for b in range(self.cfg.n_sbs):
            ue_idx = np.where(self.current_association == b)[0]
            agg = np.zeros((self.num_items + 1,), dtype=np.float64)
            for ue in ue_idx:
                user_id = int(self.user_ids_for_ues[ue])
                scores = self.user_scores[user_id]
                top = np.argsort(scores)[-self.cfg.fp :][::-1] + 1
                agg[top] += scores[top - 1]
            if np.sum(agg) <= 0:
                fallback = np.argsort(self.global_popularity[1:])[-self.cfg.fp :][::-1] + 1
                self.current_candidates[b] = fallback.astype(np.int64)
                self.current_candidate_scores[b] = self.global_popularity[fallback].astype(np.float64)
                continue
            agg[0] = -np.inf
            top_items = np.argsort(agg)[-self.cfg.fp :][::-1]
            self.current_candidates[b] = top_items.astype(np.int64)
            self.current_candidate_scores[b] = agg[top_items].astype(np.float64)

    def _build_local_states(self) -> np.ndarray:
        state = np.zeros((self.cfg.n_sbs, self.state_dim), dtype=np.float32)
        for b in range(self.cfg.n_sbs):
            load = float(np.mean(self.current_association == b))
            slot = 0
            neigh = np.where(self.current_adjacency[b] > 0.0)[0]
            neigh = neigh[neigh != b]
            neigh_count = max(1, neigh.shape[0])
            denom = float(np.sum(np.maximum(self.current_candidate_scores[b], 0.0))) + 1e-8
            for cand_idx, item in enumerate(self.current_candidates[b]):
                item = int(item)
                score = float(max(self.current_candidate_scores[b, cand_idx], 0.0) / denom)
                in_cache = float(item in self.cache_items[b])
                neighbor_overlap = 0.0
                if neigh.shape[0] > 0:
                    neighbor_overlap = float(sum(item in self.cache_items[int(n)] for n in neigh) / neigh_count)
                global_pop = float(self.global_popularity[item]) if item > 0 else 0.0
                state[b, slot : slot + 4] = (score, in_cache, neighbor_overlap, global_pop)
                slot += 4
            state[b, -2] = load
            state[b, -1] = float(self.last_hit_rate[b])
        self.current_state = state
        return state

    def _build_obs(self) -> dict[str, np.ndarray]:
        ue_pos = self.ue_positions[self.step_idx]
        sbs_pos = self._sbs_positions_at(self.step_idx)
        self.current_association = self._associate(ue_pos, sbs_pos)
        self.current_adjacency = self._adjacency(sbs_pos)
        self._aggregate_predicted_popular_contents()
        local_states = self._build_local_states()
        return {
            "local_states": local_states.copy(),
            "global_state": local_states.reshape(-1).copy(),
            "candidate_items": self.current_candidates.copy(),
            "candidate_scores": self.current_candidate_scores.copy(),
            "adjacency": self.current_adjacency.copy(),
            "association": self.current_association.copy(),
        }

    def _initial_cache(self) -> None:
        fallback = np.argsort(self.global_popularity[1:])[-self.cfg.fp :][::-1] + 1
        for b in range(self.cfg.n_sbs):
            items = self.current_candidates[b]
            chosen = []
            for item in items:
                item = int(item)
                if item > 0 and item not in chosen:
                    chosen.append(item)
                if len(chosen) >= self.cfg.cache_capacity:
                    break
            for item in fallback:
                item = int(item)
                if item not in chosen:
                    chosen.append(item)
                if len(chosen) >= self.cfg.cache_capacity:
                    break
            self.cache_items[b] = np.asarray(chosen[: self.cfg.cache_capacity], dtype=np.int64)

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_idx = 0
        self.last_hit_rate.fill(0.0)
        self.user_ids_for_ues = self.rng.choice(
            np.asarray(self._eligible_users, dtype=np.int64),
            size=self.cfg.n_ues,
            replace=False,
        ).astype(np.int64)
        for idx, user_id in enumerate(self.user_ids_for_ues):
            seq_len = len(self.user_histories[int(user_id)])
            self.user_ptrs[idx] = int(self.rng.integers(0, max(1, seq_len - 1)))
        obs = self._build_obs()
        self._initial_cache()
        return self._build_obs()

    def action_scores_to_items(self, action_scores: np.ndarray) -> np.ndarray:
        action_scores = np.asarray(action_scores, dtype=np.float32)
        out = np.zeros((self.cfg.n_sbs, self.cfg.cache_capacity), dtype=np.int64)
        for b in range(self.cfg.n_sbs):
            scores = action_scores[b]
            top = np.argsort(scores)[-self.cfg.cache_capacity :][::-1]
            out[b] = self.current_candidates[b, top].astype(np.int64)
        return out

    def items_to_binary_action(self, items: np.ndarray) -> np.ndarray:
        binary = np.zeros((self.cfg.n_sbs, self.cfg.fp), dtype=np.float32)
        for b in range(self.cfg.n_sbs):
            wanted = {int(x) for x in items[b].tolist() if int(x) > 0}
            for idx, item in enumerate(self.current_candidates[b].tolist()):
                if int(item) in wanted:
                    binary[b, idx] = 1.0
        return binary

    def _set_cache_items(self, items: np.ndarray) -> np.ndarray:
        items = np.asarray(items, dtype=np.int64)
        replaced = np.zeros((self.cfg.n_sbs,), dtype=np.float64)
        for b in range(self.cfg.n_sbs):
            old = self.cache_items[b].copy()
            new = []
            seen: set[int] = set()
            for item in items[b]:
                item = int(item)
                if item <= 0 or item in seen:
                    continue
                new.append(item)
                seen.add(item)
                if len(new) >= self.cfg.cache_capacity:
                    break
            if len(new) < self.cfg.cache_capacity:
                fallback = np.argsort(self.global_popularity[1:])[-self.cfg.fp :][::-1] + 1
                for item in fallback:
                    item = int(item)
                    if item not in seen:
                        new.append(item)
                        seen.add(item)
                    if len(new) >= self.cfg.cache_capacity:
                        break
            new_arr = np.asarray(new[: self.cfg.cache_capacity], dtype=np.int64)
            replaced[b] = float(np.sum(~np.isin(new_arr, old)))
            self.cache_items[b] = new_arr
        return replaced

    def _compute_reward(self, replaced: np.ndarray) -> tuple[float, np.ndarray, dict[str, float]]:
        local_hits = np.zeros((self.cfg.n_sbs,), dtype=np.float64)
        neighbor_hits = np.zeros((self.cfg.n_sbs,), dtype=np.float64)
        cloud_hits = np.zeros((self.cfg.n_sbs,), dtype=np.float64)
        totals = np.zeros((self.cfg.n_sbs,), dtype=np.float64)

        for ue in range(self.cfg.n_ues):
            b = int(self.current_association[ue])
            user_id = int(self.user_ids_for_ues[ue])
            ptr = int(self.user_ptrs[ue])
            seq = self.user_histories[user_id]
            item = int(seq[ptr])
            totals[b] += 1.0

            if item in self.cache_items[b]:
                local_hits[b] += 1.0
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

            self.user_ptrs[ue] = int((ptr + 1) % len(seq))

        local_rewards = (self.cfg.chi_cloud - self.cfg.alpha_local) * local_hits + (
            self.cfg.chi_cloud - self.cfg.beta_neighbor
        ) * neighbor_hits - self.cfg.delta_replace * replaced
        global_reward = float(np.mean(local_rewards))
        denom = max(1.0, np.sum(totals))
        self.last_hit_rate = local_hits / np.maximum(1.0, totals)
        info = {
            "reward": global_reward,
            "local_hit_rate": float(np.sum(local_hits) / denom),
            "neighbor_fetch_rate": float(np.sum(neighbor_hits) / denom),
            "cloud_fetch_rate": float(np.sum(cloud_hits) / denom),
            "paper_hit_rate": float((np.sum(local_hits) + np.sum(neighbor_hits)) / denom),
            "replace_count": float(np.sum(replaced)),
        }
        return global_reward, local_rewards.astype(np.float32), info

    def step_items(self, items: np.ndarray) -> tuple[dict[str, np.ndarray], float, np.ndarray, bool, dict[str, float]]:
        replaced = self._set_cache_items(items)
        reward, local_rewards, info = self._compute_reward(replaced)
        self.step_idx += 1
        done = self.step_idx >= self.cfg.episode_len
        next_obs = self._build_obs() if not done else self._build_obs()
        return next_obs, reward, local_rewards, done, info
