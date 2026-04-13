from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import copy


class GraphAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Linear(out_dim, 1, bias=False)
        self.attn_dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        src = self.attn_src(h)
        dst = self.attn_dst(h)
        e = self.leaky_relu(src + dst.transpose(0, 1))
        e = e.masked_fill(adj <= 0, -1e9)
        edge_bias = torch.where(adj > 0, torch.log(torch.clamp(adj, min=1e-6)), torch.full_like(adj, -1e9))
        alpha = torch.softmax(e + edge_bias, dim=-1)
        return alpha @ h


class TemporalGraphCooperativePolicy(nn.Module):
    def __init__(self, node_feat_dim: int, candidate_feat_dim: int, hidden_dim: int, fp: int, use_graph: bool = True) -> None:
        super().__init__()
        self.use_graph = use_graph
        self.gat1 = GraphAttention(node_feat_dim, hidden_dim)
        self.gat2 = GraphAttention(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.candidate_norm = nn.LayerNorm(candidate_feat_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.structural_dim = min(candidate_feat_dim, 10)
        init_weights = torch.tensor([20.0, 1.5, -1.5, 1.2, 4.0, 1.5, 1.2, 2.5, 1.6, 1.2], dtype=torch.float32)
        self.heuristic_weights = nn.Parameter(init_weights[: self.structural_dim].clone())
        self.heuristic_scale = nn.Parameter(torch.tensor(0.85, dtype=torch.float32))
        self.heuristic_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.fp = fp

    def forward(
        self,
        node_features: torch.Tensor,
        candidate_features: torch.Tensor,
        adjacency: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_graph:
            h = torch.relu(self.gat1(node_features, adjacency))
            h = self.norm(torch.relu(self.gat2(h, adjacency)) + h)
        else:
            h = self.node_encoder(node_features)
        normalized_candidates = self.candidate_norm(candidate_features)
        cand = self.candidate_encoder(normalized_candidates)
        q = self.query(h).unsqueeze(1).expand(-1, cand.shape[1], -1)
        fused = torch.cat([q * cand, cand], dim=-1)
        logits = self.score_head(fused).squeeze(-1)
        structural = candidate_features[..., : self.structural_dim]
        heuristic_bias = torch.einsum("bsf,f->bs", structural, self.heuristic_weights)
        logits = logits + torch.tanh(self.heuristic_scale) * heuristic_bias + self.heuristic_bias
        no_valid = action_mask.sum(dim=-1) <= 0
        if torch.any(no_valid):
            action_mask = action_mask.clone()
            action_mask[no_valid, 0] = 1.0
        return logits.masked_fill(action_mask <= 0, -1e9)


@dataclass
class ImitationConfig:
    epochs: int = 12
    episodes_per_epoch: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-5
    device: str = "cpu"
    teacher_forcing_prob: float = 0.8
    teacher_forcing_final_prob: float = 0.2
    teacher_score_loss_weight: float = 0.35
    teacher_rank_loss_weight: float = 0.20
    label_smoothing: float = 0.05
    decode_diversity_penalty: float = 0.25
    teacher_guidance_weight: float = 0.55
    placement_interval: int = 3
    checkpoint_reward_weight: float = 1.0
    checkpoint_local_hit_weight: float = 800.0
    checkpoint_paper_hit_weight: float = 1800.0
    checkpoint_eval_episodes: int = 4


@dataclass
class ImitationHistory:
    losses: list[float]
    rewards: list[float]
    local_hit_rates: list[float]
    paper_hit_rates: list[float]


@dataclass
class ReinforceConfig:
    epochs: int = 6
    episodes_per_epoch: int = 4
    lr: float = 1e-4
    gamma: float = 0.99
    entropy_weight: float = 1e-3
    device: str = "cpu"
    decode_diversity_penalty: float = 0.25
    teacher_guidance_weight: float = 0.55
    placement_interval: int = 3
    edge_hit_bonus_weight: float = 1400.0
    checkpoint_reward_weight: float = 1.0
    checkpoint_local_hit_weight: float = 900.0
    checkpoint_paper_hit_weight: float = 2200.0
    checkpoint_eval_episodes: int = 4


def _selection_score(
    reward_mean: float,
    local_hit_mean: float,
    paper_hit_mean: float,
    reward_weight: float,
    local_weight: float,
    paper_weight: float,
) -> float:
    return (
        reward_weight * (reward_mean / 100.0)
        + local_weight * local_hit_mean
        + paper_weight * paper_hit_mean
    )


@dataclass
class ReinforceHistory:
    losses: list[float]
    rewards: list[float]
    local_hit_rates: list[float]
    paper_hit_rates: list[float]


def _obs_to_tensors(obs: dict[str, np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
    cand = torch.as_tensor(obs["candidate_features"], dtype=torch.float32, device=device)
    adj = torch.as_tensor(obs["adjacency"], dtype=torch.float32, device=device)
    mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=device)
    return node, cand, adj, mask


def _normalized_teacher_scores(
    teacher_scores: np.ndarray | None,
    b: int,
    valid_slots: list[int],
    fp: int,
) -> np.ndarray:
    out = np.zeros((fp,), dtype=np.float64)
    if teacher_scores is None or not valid_slots:
        return out
    vals = teacher_scores[b, valid_slots].astype(np.float64)
    vals = vals - float(np.mean(vals))
    scale = max(float(np.std(vals)), 1e-6)
    out[np.asarray(valid_slots, dtype=np.int64)] = vals / scale
    return out


def _normalized_local_scores(env, b: int, valid_slots: list[int]) -> np.ndarray:
    out = np.zeros((env.cfg.fp,), dtype=np.float64)
    if not valid_slots:
        return out
    vals = env.current_candidate_scores[b, valid_slots].astype(np.float64)
    vals = vals - float(np.mean(vals))
    scale = max(float(np.std(vals)), 1e-6)
    out[np.asarray(valid_slots, dtype=np.int64)] = vals / scale
    return out


def _teacher_rank_loss(logits: torch.Tensor, teacher_mask: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for b in range(logits.shape[0]):
        valid = action_mask[b] > 0
        pos = logits[b][teacher_mask[b] > 0.5]
        neg = logits[b][valid & (teacher_mask[b] <= 0.5)]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        margin = 0.25
        losses.append(torch.relu(margin - pos.unsqueeze(1) + neg.unsqueeze(0)).mean())
    if not losses:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return torch.stack(losses).mean()


def _should_refresh_cache(step_idx: int, placement_interval: int) -> bool:
    return placement_interval <= 1 or step_idx % placement_interval == 0


def _ranked_slots_to_items(env, b: int, ranked_slots: np.ndarray) -> list[int]:
    chosen: list[int] = []
    seen: set[int] = set()
    for slot in ranked_slots.tolist():
        item = int(env.current_candidates[b, int(slot)])
        if item <= 0 or item in seen:
            continue
        if not env.current_mask[b, int(slot)]:
            continue
        chosen.append(item)
        seen.add(item)
        if len(chosen) >= env.cfg.cache_capacity:
            break
    if len(chosen) < env.cfg.cache_capacity:
        for item in (np.argsort(env.global_popularity[1:])[-env.cfg.cache_capacity :][::-1] + 1).tolist():
            if item in seen:
                continue
            chosen.append(int(item))
            seen.add(int(item))
            if len(chosen) >= env.cfg.cache_capacity:
                break
    return chosen[: env.cfg.cache_capacity]


def _action_proxy_value(env, b: int, items: list[int], planned: list[set[int]]) -> float:
    neigh = np.where(env.current_adjacency[b] > 0.0)[0]
    neigh = neigh[neigh != b]
    item_to_slot = {int(env.current_candidates[b, slot]): int(slot) for slot in np.where(env.current_mask[b])[0].tolist()}
    total = 0.0
    for item in items:
        slot = item_to_slot.get(int(item))
        if slot is None:
            continue
        overlap = sum(int(item) in planned[int(n)] for n in neigh)
        total += 1.65 * float(env.current_candidate_scores[b, slot])
        total += 0.30 * float(env.global_popularity[int(item)])
        total += 0.22 * float(int(item) in env.cache_items[b])
        total += 0.28 * float(env.current_neighbor_shortage[b, slot])
        total += 0.18 * float(env.current_semantic_affinity[b, slot])
        total += 0.14 * float(env.current_freshness_relevance[b, slot])
        total -= 0.06 * float(overlap)
    return total


def logits_to_cache_items(
    logits: torch.Tensor,
    env,
    diversity_penalty: float = 0.0,
    teacher_scores: np.ndarray | None = None,
    teacher_guidance_weight: float = 0.0,
) -> np.ndarray:
    slot_scores = logits.detach().cpu().numpy().astype(np.float64)
    out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
    planned: list[set[int]] = [set() for _ in range(env.cfg.n_sbs)]
    order = np.argsort(slot_scores.max(axis=1) + 0.5 * env.current_future_load)[::-1]
    for b in order.tolist():
        chosen = []
        seen: set[int] = set()
        adjusted = slot_scores[b].copy()
        valid_slots = np.where(env.current_mask[b])[0].tolist()
        local_norm = np.zeros((env.cfg.fp,), dtype=np.float64)
        if valid_slots:
            raw = np.maximum(env.current_candidate_scores[b, valid_slots], 0.0)
            denom = max(float(raw.sum()), 1e-8)
            local_norm[np.asarray(valid_slots, dtype=np.int64)] = raw / denom
        local_prior = _normalized_local_scores(env, b, valid_slots)
        teacher_norm = _normalized_teacher_scores(teacher_scores, b, valid_slots, env.cfg.fp)
        adjusted += 0.70 * local_prior
        if teacher_guidance_weight > 0.0:
            adjusted += teacher_guidance_weight * teacher_norm
        if diversity_penalty > 0.0:
            neigh = np.where(env.current_adjacency[b] > 0.0)[0]
            neigh = neigh[neigh != b]
            for slot in valid_slots:
                item = int(env.current_candidates[b, int(slot)])
                overlap = sum(item in planned[int(n)] for n in neigh)
                teacher_keep = max(0.0, float(teacher_norm[slot]))
                local_keep = 0.60 + 0.90 * float(local_norm[slot]) + 0.18 * float(item in env.cache_items[b]) + 0.08 * teacher_keep
                adjusted[slot] -= diversity_penalty * float(overlap) * max(0.10, 0.85 - local_keep)
                adjusted[slot] += 0.14 * float(item in env.cache_items[b])
                adjusted[slot] += 1.15 * float(env.current_neighbor_shortage[b, int(slot)])
                adjusted[slot] += 0.42 * float(local_norm[slot])
                adjusted[slot] += 0.04 * float(env.global_popularity[item])
                adjusted[slot] += 0.08 * float(item == int(env.current_trend_items[b])) * float(env.current_trend_strength[b])
                adjusted[slot] += 0.22 * float(env.current_neighbor_support[b, int(slot)])
                adjusted[slot] += 0.07 * float(env.current_future_load[b]) * float(local_norm[slot])
                adjusted[slot] += 0.18 * float(env.current_semantic_affinity[b, int(slot)])
                adjusted[slot] += 0.12 * float(env.current_freshness_relevance[b, int(slot)])
                adjusted[slot] += 0.10 * teacher_keep
        ranked = np.argsort(adjusted)[::-1]
        chosen = _ranked_slots_to_items(env, b, ranked)

        local_fallback_scores = np.full((env.cfg.fp,), -1e9, dtype=np.float64)
        if valid_slots:
            local_fallback_scores[np.asarray(valid_slots, dtype=np.int64)] = (
                1.20 * local_prior[np.asarray(valid_slots, dtype=np.int64)]
                + 0.30 * teacher_norm[np.asarray(valid_slots, dtype=np.int64)]
                + 0.28 * env.current_candidate_scores[b, np.asarray(valid_slots, dtype=np.int64)]
                + 0.18 * env.current_neighbor_shortage[b, np.asarray(valid_slots, dtype=np.int64)]
                + 0.10 * env.current_semantic_affinity[b, np.asarray(valid_slots, dtype=np.int64)]
                + 0.08 * env.current_freshness_relevance[b, np.asarray(valid_slots, dtype=np.int64)]
                + 0.06 * env.current_neighbor_support[b, np.asarray(valid_slots, dtype=np.int64)]
            )
        local_fallback = _ranked_slots_to_items(env, b, np.argsort(local_fallback_scores)[::-1])

        ddpg_fallback_scores = np.full((env.cfg.fp,), -1e9, dtype=np.float64)
        for slot in valid_slots:
            item = int(env.current_candidates[b, int(slot)])
            neigh = np.where(env.current_adjacency[b] > 0.0)[0]
            neigh = neigh[neigh != b]
            neighbor_overlap = float(sum(item in env.cache_items[int(n)] for n in neigh) / max(1, len(neigh)))
            ddpg_fallback_scores[int(slot)] = (
                0.20 * float(env.global_popularity[item])
                + 0.05 * float(item in env.cache_items[b])
                - 0.15 * neighbor_overlap
            )
        ddpg_fallback = _ranked_slots_to_items(env, b, np.argsort(ddpg_fallback_scores)[::-1])

        candidates = [chosen, local_fallback, ddpg_fallback]
        if teacher_scores is not None:
            teacher_fallback = _ranked_slots_to_items(env, b, np.argsort(teacher_norm)[::-1])
            candidates.append(teacher_fallback)

        best_items = max(candidates, key=lambda items: _action_proxy_value(env, b, items, planned))
        out[b] = np.asarray(best_items, dtype=np.int64)
        planned[b] = set(out[b].tolist())
    return out


def sample_cache_items(
    logits: torch.Tensor,
    env,
    diversity_penalty: float = 0.0,
    teacher_scores: np.ndarray | None = None,
    teacher_guidance_weight: float = 0.0,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    device = logits.device
    out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
    total_log_prob = torch.zeros((), dtype=torch.float32, device=device)
    total_entropy = torch.zeros((), dtype=torch.float32, device=device)
    planned: list[set[int]] = [set() for _ in range(env.cfg.n_sbs)]
    order = np.argsort(logits.detach().cpu().numpy().max(axis=1) + 0.5 * env.current_future_load)[::-1]

    for b in order.tolist():
        scores = logits[b].clone()
        valid_slots = np.where(env.current_mask[b])[0].tolist()
        neigh = np.where(env.current_adjacency[b] > 0.0)[0]
        neigh = neigh[neigh != b]
        local_norm = np.zeros((env.cfg.fp,), dtype=np.float64)
        if valid_slots:
            raw = np.maximum(env.current_candidate_scores[b, valid_slots], 0.0)
            denom = max(float(raw.sum()), 1e-8)
            local_norm[np.asarray(valid_slots, dtype=np.int64)] = raw / denom
        local_prior = _normalized_local_scores(env, b, valid_slots)
        teacher_norm = _normalized_teacher_scores(teacher_scores, b, valid_slots, env.cfg.fp)
        for slot in valid_slots:
            item = int(env.current_candidates[b, slot])
            overlap = sum(item in planned[int(n)] for n in neigh)
            teacher_keep = max(0.0, float(teacher_norm[slot]))
            local_keep = 0.60 + 0.90 * float(local_norm[slot]) + 0.18 * float(item in env.cache_items[b]) + 0.08 * teacher_keep
            scores[slot] = scores[slot] + 0.70 * float(local_prior[slot])
            scores[slot] = scores[slot] + teacher_guidance_weight * float(teacher_norm[slot])
            scores[slot] = scores[slot] - diversity_penalty * float(overlap) * max(0.10, 0.85 - local_keep)
            scores[slot] = scores[slot] + 0.14 * float(item in env.cache_items[b])
            scores[slot] = scores[slot] + 1.15 * float(env.current_neighbor_shortage[b, slot])
            scores[slot] = scores[slot] + 0.42 * float(local_norm[slot])
            scores[slot] = scores[slot] + 0.04 * float(env.global_popularity[item])
            scores[slot] = scores[slot] + 0.08 * float(item == int(env.current_trend_items[b])) * float(env.current_trend_strength[b])
            scores[slot] = scores[slot] + 0.22 * float(env.current_neighbor_support[b, slot])
            scores[slot] = scores[slot] + 0.07 * float(env.current_future_load[b]) * float(local_norm[slot])
            scores[slot] = scores[slot] + 0.18 * float(env.current_semantic_affinity[b, slot])
            scores[slot] = scores[slot] + 0.12 * float(env.current_freshness_relevance[b, slot])
            scores[slot] = scores[slot] + 0.10 * teacher_keep

        used_slots: set[int] = set()
        chosen: list[int] = []
        seen_items: set[int] = set()
        for _ in range(env.cfg.cache_capacity):
            masked_scores = scores.clone()
            if used_slots:
                masked_scores[list(used_slots)] = -1e9
            invalid_slots = np.where(~env.current_mask[b])[0]
            if invalid_slots.size > 0:
                masked_scores[torch.as_tensor(invalid_slots, dtype=torch.long, device=device)] = -1e9
            probs = torch.softmax(masked_scores, dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
                break
            dist = torch.distributions.Categorical(probs=probs)
            slot_tensor = dist.sample()
            slot = int(slot_tensor.item())
            used_slots.add(slot)
            total_log_prob = total_log_prob + dist.log_prob(slot_tensor)
            total_entropy = total_entropy + dist.entropy()
            item = int(env.current_candidates[b, slot])
            if item <= 0 or item in seen_items:
                continue
            chosen.append(item)
            seen_items.add(item)
            if len(chosen) >= env.cfg.cache_capacity:
                break

        if len(chosen) < env.cfg.cache_capacity:
            for item in (np.argsort(env.global_popularity[1:])[-env.cfg.cache_capacity :][::-1] + 1).tolist():
                if item in seen_items:
                    continue
                chosen.append(int(item))
                seen_items.add(int(item))
                if len(chosen) >= env.cfg.cache_capacity:
                    break
        out[b] = np.asarray(chosen[: env.cfg.cache_capacity], dtype=np.int64)
        planned[b] = set(out[b].tolist())
    return out, total_log_prob, total_entropy


def train_graph_cache_policy_imitation(
    env,
    model: TemporalGraphCooperativePolicy,
    cfg: ImitationConfig,
    seed: int = 42,
    logger=None,
    log_every_epoch: int = 1,
    log_every_episode: int = 1,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(cfg.device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=device))
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")

    losses: list[float] = []
    rewards: list[float] = []
    local_hits: list[float] = []
    paper_hits: list[float] = []

    for epoch in range(cfg.epochs):
        frac = 0.0 if cfg.epochs <= 1 else epoch / (cfg.epochs - 1)
        teacher_forcing_prob = (
            (1.0 - frac) * cfg.teacher_forcing_prob + frac * cfg.teacher_forcing_final_prob
        )
        epoch_losses = []
        epoch_rewards = []
        epoch_local = []
        epoch_paper = []
        for episode in range(cfg.episodes_per_epoch):
            obs = env.reset(seed=seed + epoch * 100 + episode)
            done = False
            step_idx = 0
            last_action: np.ndarray | None = None
            while not done:
                teacher_items = env.cooperative_teacher_action()
                teacher_scores = env.cooperative_teacher_scores()
                teacher_mask = env.candidate_items_to_slot_mask(teacher_items)
                node, cand, adj, mask = _obs_to_tensors(obs, device)
                logits = model(node, cand, adj, mask)
                target = torch.as_tensor(teacher_mask, dtype=torch.float32, device=device)
                target = target * (1.0 - cfg.label_smoothing) + cfg.label_smoothing * 0.5
                cls_loss = loss_fn(logits, target)

                teacher_scores_t = torch.as_tensor(teacher_scores, dtype=torch.float32, device=device)
                teacher_scores_t = teacher_scores_t.masked_fill(mask <= 0, -1e9)
                teacher_probs = torch.softmax(teacher_scores_t, dim=-1)
                student_log_probs = torch.log_softmax(logits, dim=-1)
                score_loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
                rank_loss = _teacher_rank_loss(logits, torch.as_tensor(teacher_mask, dtype=torch.float32, device=device), mask)
                loss = cls_loss + cfg.teacher_score_loss_weight * score_loss + cfg.teacher_rank_loss_weight * rank_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_losses.append(float(loss.item()))

                use_teacher = np.random.random() < teacher_forcing_prob
                desired_items = teacher_items if use_teacher else logits_to_cache_items(
                    logits,
                    env,
                    diversity_penalty=cfg.decode_diversity_penalty,
                    teacher_scores=teacher_scores,
                    teacher_guidance_weight=cfg.teacher_guidance_weight,
                )
                if last_action is None or _should_refresh_cache(step_idx, cfg.placement_interval):
                    chosen_items = desired_items
                else:
                    chosen_items = last_action
                last_action = chosen_items.copy()
                obs, reward, done, info = env.step_full_cache_items(chosen_items)
                epoch_rewards.append(float(reward))
                epoch_local.append(float(info["local_hit_rate"]))
                epoch_paper.append(float(info["local_hit_rate"] + info["neighbor_fetch_rate"]))
                step_idx += 1

            if logger is not None and (
                (episode + 1) % max(1, log_every_episode) == 0 or (episode + 1) == cfg.episodes_per_epoch
            ):
                logger.info(
                    "Imitation epoch %d/%d episode %d/%d complete | teacher_forcing=%.3f reward=%.4f local_hit=%.4f paper_hit=%.4f",
                    epoch + 1,
                    cfg.epochs,
                    episode + 1,
                    cfg.episodes_per_epoch,
                    teacher_forcing_prob,
                    float(np.mean(epoch_rewards)) if epoch_rewards else float("nan"),
                    float(np.mean(epoch_local)) if epoch_local else float("nan"),
                    float(np.mean(epoch_paper)) if epoch_paper else float("nan"),
                )

        epoch_loss_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        epoch_reward_mean = float(np.mean(epoch_rewards)) if epoch_rewards else float("nan")
        epoch_local_mean = float(np.mean(epoch_local)) if epoch_local else float("nan")
        epoch_paper_mean = float(np.mean(epoch_paper)) if epoch_paper else float("nan")

        losses.append(epoch_loss_mean)
        rewards.append(epoch_reward_mean)
        local_hits.append(epoch_local_mean)
        paper_hits.append(epoch_paper_mean)

        if cfg.checkpoint_eval_episodes > 0:
            eval_rows = evaluate_graph_cache_policy(
                env,
                model,
                episodes=cfg.checkpoint_eval_episodes,
                seed=seed + 200000 + epoch * 17,
                device=cfg.device,
                decode_diversity_penalty=cfg.decode_diversity_penalty,
                teacher_guidance_weight=cfg.teacher_guidance_weight,
                placement_interval=cfg.placement_interval,
            )
            sel_reward_mean = float(np.mean([row["reward"] for row in eval_rows]))
            sel_local_mean = float(np.mean([row["local_hit_rate"] for row in eval_rows]))
            sel_paper_mean = float(np.mean([row["paper_hit_rate"] for row in eval_rows]))
        else:
            sel_reward_mean = epoch_reward_mean
            sel_local_mean = epoch_local_mean
            sel_paper_mean = epoch_paper_mean

        epoch_score = _selection_score(
            sel_reward_mean,
            sel_local_mean,
            sel_paper_mean,
            cfg.checkpoint_reward_weight,
            cfg.checkpoint_local_hit_weight,
            cfg.checkpoint_paper_hit_weight,
        )
        if epoch_score > best_score:
            best_score = epoch_score
            best_state = copy.deepcopy(model.state_dict())

        if logger is not None and (
            (epoch + 1) % max(1, log_every_epoch) == 0 or (epoch + 1) == cfg.epochs
        ):
            logger.info(
                "Imitation epoch %d/%d summary | loss=%.6f reward=%.4f local_hit=%.4f paper_hit=%.4f sel_reward=%.4f sel_local=%.4f sel_paper=%.4f best_score=%.4f",
                epoch + 1,
                cfg.epochs,
                epoch_loss_mean,
                epoch_reward_mean,
                epoch_local_mean,
                epoch_paper_mean,
                sel_reward_mean,
                sel_local_mean,
                sel_paper_mean,
                best_score,
            )

    model.load_state_dict(best_state)

    return ImitationHistory(
        losses=losses,
        rewards=rewards,
        local_hit_rates=local_hits,
        paper_hit_rates=paper_hits,
    )


def fine_tune_graph_cache_policy_reinforce(
    env,
    model: TemporalGraphCooperativePolicy,
    cfg: ReinforceConfig,
    seed: int = 42,
    logger=None,
    log_every_epoch: int = 1,
    log_every_episode: int = 1,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(cfg.device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_state = copy.deepcopy(model.state_dict())
    initial_eval = evaluate_graph_cache_policy(
        env,
        model,
        episodes=1,
        seed=seed + 9000,
        device=cfg.device,
        decode_diversity_penalty=cfg.decode_diversity_penalty,
        teacher_guidance_weight=cfg.teacher_guidance_weight,
        placement_interval=cfg.placement_interval,
    )[0]
    best_score = _selection_score(
        float(initial_eval["reward"]),
        float(initial_eval["local_hit_rate"]),
        float(initial_eval["paper_hit_rate"]),
        cfg.checkpoint_reward_weight,
        cfg.checkpoint_local_hit_weight,
        cfg.checkpoint_paper_hit_weight,
    )
    running_baseline = 0.0

    losses: list[float] = []
    rewards: list[float] = []
    local_hits: list[float] = []
    paper_hits: list[float] = []

    for epoch in range(cfg.epochs):
        epoch_losses = []
        epoch_rewards = []
        epoch_local = []
        epoch_paper = []
        for episode in range(cfg.episodes_per_epoch):
            obs = env.reset(seed=seed + 10000 + epoch * 100 + episode)
            done = False
            step_idx = 0
            last_action: np.ndarray | None = None
            log_probs: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            rewards_per_step: list[float] = []
            raw_rewards_per_step: list[float] = []
            local_per_step: list[float] = []
            paper_per_step: list[float] = []

            while not done:
                node, cand, adj, mask = _obs_to_tensors(obs, device)
                logits = model(node, cand, adj, mask)
                teacher_scores = env.cooperative_teacher_scores()
                chosen, log_prob, entropy = sample_cache_items(
                    logits,
                    env,
                    diversity_penalty=cfg.decode_diversity_penalty,
                    teacher_scores=teacher_scores,
                    teacher_guidance_weight=cfg.teacher_guidance_weight,
                )
                if last_action is None or _should_refresh_cache(step_idx, cfg.placement_interval):
                    chosen_items = chosen
                else:
                    chosen_items = last_action
                last_action = chosen_items.copy()
                obs, reward, done, info = env.step_full_cache_items(chosen_items)
                paper_hit = float(info["local_hit_rate"] + info["neighbor_fetch_rate"])
                shaped_reward = float(reward) + cfg.edge_hit_bonus_weight * paper_hit
                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards_per_step.append(shaped_reward)
                raw_rewards_per_step.append(float(reward))
                local_per_step.append(float(info["local_hit_rate"]))
                paper_per_step.append(paper_hit)
                step_idx += 1

            returns = []
            ret = 0.0
            for reward in reversed(rewards_per_step):
                ret = reward + cfg.gamma * ret
                returns.append(ret)
            returns.reverse()
            returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
            episode_mean_return = float(returns_t.mean().item()) if returns else 0.0
            running_baseline = 0.9 * running_baseline + 0.1 * episode_mean_return
            adv = returns_t - running_baseline
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
            log_prob_t = torch.stack(log_probs)
            entropy_t = torch.stack(entropies)
            loss = -(adv.detach() * log_prob_t).mean() - cfg.entropy_weight * entropy_t.mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_losses.append(float(loss.item()))
            epoch_rewards.append(float(np.sum(raw_rewards_per_step)))
            epoch_local.append(float(np.mean(local_per_step)) if local_per_step else 0.0)
            epoch_paper.append(float(np.mean(paper_per_step)) if paper_per_step else 0.0)

            if logger is not None and (
                (episode + 1) % max(1, log_every_episode) == 0 or (episode + 1) == cfg.episodes_per_epoch
            ):
                logger.info(
                    "Reinforce epoch %d/%d episode %d/%d complete | loss=%.6f reward=%.4f local_hit=%.4f paper_hit=%.4f",
                    epoch + 1,
                    cfg.epochs,
                    episode + 1,
                    cfg.episodes_per_epoch,
                    float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                    float(np.mean(epoch_rewards)) if epoch_rewards else float("nan"),
                    float(np.mean(epoch_local)) if epoch_local else float("nan"),
                    float(np.mean(epoch_paper)) if epoch_paper else float("nan"),
                )

        epoch_loss_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        epoch_reward_mean = float(np.mean(epoch_rewards)) if epoch_rewards else float("nan")
        epoch_local_mean = float(np.mean(epoch_local)) if epoch_local else float("nan")
        epoch_paper_mean = float(np.mean(epoch_paper)) if epoch_paper else float("nan")
        losses.append(epoch_loss_mean)
        rewards.append(epoch_reward_mean)
        local_hits.append(epoch_local_mean)
        paper_hits.append(epoch_paper_mean)

        if cfg.checkpoint_eval_episodes > 0:
            eval_rows = evaluate_graph_cache_policy(
                env,
                model,
                episodes=cfg.checkpoint_eval_episodes,
                seed=seed + 300000 + epoch * 19,
                device=cfg.device,
                decode_diversity_penalty=cfg.decode_diversity_penalty,
                teacher_guidance_weight=cfg.teacher_guidance_weight,
                placement_interval=cfg.placement_interval,
            )
            sel_reward_mean = float(np.mean([row["reward"] for row in eval_rows]))
            sel_local_mean = float(np.mean([row["local_hit_rate"] for row in eval_rows]))
            sel_paper_mean = float(np.mean([row["paper_hit_rate"] for row in eval_rows]))
        else:
            sel_reward_mean = epoch_reward_mean
            sel_local_mean = epoch_local_mean
            sel_paper_mean = epoch_paper_mean

        epoch_score = _selection_score(
            sel_reward_mean,
            sel_local_mean,
            sel_paper_mean,
            cfg.checkpoint_reward_weight,
            cfg.checkpoint_local_hit_weight,
            cfg.checkpoint_paper_hit_weight,
        )
        if epoch_score > best_score:
            best_score = epoch_score
            best_state = copy.deepcopy(model.state_dict())

        if logger is not None and (
            (epoch + 1) % max(1, log_every_epoch) == 0 or (epoch + 1) == cfg.epochs
        ):
            logger.info(
                "Reinforce epoch %d/%d summary | loss=%.6f reward=%.4f local_hit=%.4f paper_hit=%.4f sel_reward=%.4f sel_local=%.4f sel_paper=%.4f best_score=%.4f",
                epoch + 1,
                cfg.epochs,
                epoch_loss_mean,
                epoch_reward_mean,
                epoch_local_mean,
                epoch_paper_mean,
                sel_reward_mean,
                sel_local_mean,
                sel_paper_mean,
                best_score,
            )

    model.load_state_dict(best_state)
    return ReinforceHistory(
        losses=losses,
        rewards=rewards,
        local_hit_rates=local_hits,
        paper_hit_rates=paper_hits,
    )


@torch.no_grad()
def evaluate_graph_cache_policy(
    env,
    model: TemporalGraphCooperativePolicy,
    episodes: int,
    seed: int = 42,
    device: str = "cpu",
    decode_diversity_penalty: float = 0.35,
    teacher_guidance_weight: float = 0.55,
    placement_interval: int = 3,
):
    model = model.to(device)
    model.eval()
    rows = []
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
            node, cand, adj, mask = _obs_to_tensors(obs, torch.device(device))
            logits = model(node, cand, adj, mask)
            teacher_scores = env.cooperative_teacher_scores()
            chosen = logits_to_cache_items(
                logits,
                env,
                diversity_penalty=decode_diversity_penalty,
                teacher_scores=teacher_scores,
                teacher_guidance_weight=teacher_guidance_weight,
            )
            if last_action is None or _should_refresh_cache(step_idx, placement_interval):
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
