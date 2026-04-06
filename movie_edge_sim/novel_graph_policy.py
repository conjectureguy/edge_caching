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
        alpha = torch.softmax(e, dim=-1)
        return alpha @ h


class TemporalGraphCooperativePolicy(nn.Module):
    def __init__(self, node_feat_dim: int, candidate_feat_dim: int, hidden_dim: int, fp: int, use_graph: bool = True) -> None:
        super().__init__()
        self.use_graph = use_graph
        self.gat1 = GraphAttention(node_feat_dim, hidden_dim)
        self.gat2 = GraphAttention(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
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
        cand = self.candidate_encoder(candidate_features)
        q = self.query(h).unsqueeze(1).expand(-1, cand.shape[1], -1)
        fused = torch.cat([q * cand, cand], dim=-1)
        logits = self.score_head(fused).squeeze(-1)
        bias = 20.0 * candidate_features[..., 0] + 4.0 * candidate_features[..., 4] + 1.5 * candidate_features[..., 5]
        bias = bias + 1.5 * candidate_features[..., 1] - 1.5 * candidate_features[..., 2]
        bias = bias + 1.2 * candidate_features[..., 6] + 2.5 * candidate_features[..., 7]
        if candidate_features.shape[-1] >= 10:
            bias = bias + 1.6 * candidate_features[..., 8] + 1.2 * candidate_features[..., 9]
        logits = logits + bias
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
    label_smoothing: float = 0.05
    decode_diversity_penalty: float = 0.25
    checkpoint_reward_weight: float = 1.0
    checkpoint_local_hit_weight: float = 800.0
    checkpoint_paper_hit_weight: float = 1800.0
    checkpoint_eval_episodes: int = 2


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
    edge_hit_bonus_weight: float = 1400.0
    checkpoint_reward_weight: float = 1.0
    checkpoint_local_hit_weight: float = 900.0
    checkpoint_paper_hit_weight: float = 2200.0
    checkpoint_eval_episodes: int = 2


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


def logits_to_cache_items(logits: torch.Tensor, env, diversity_penalty: float = 0.0) -> np.ndarray:
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
        if diversity_penalty > 0.0:
            neigh = np.where(env.current_adjacency[b] > 0.0)[0]
            neigh = neigh[neigh != b]
            for slot in valid_slots:
                item = int(env.current_candidates[b, int(slot)])
                overlap = sum(item in planned[int(n)] for n in neigh)
                local_keep = 0.55 + 0.75 * float(local_norm[slot]) + 0.15 * float(item in env.cache_items[b])
                adjusted[slot] -= diversity_penalty * float(overlap) * max(0.20, 1.0 - local_keep)
                adjusted[slot] += 0.06 * float(item in env.cache_items[b])
                adjusted[slot] += 1.15 * float(env.current_neighbor_shortage[b, int(slot)])
                adjusted[slot] += 0.14 * float(local_norm[slot])
                adjusted[slot] += 0.04 * float(env.global_popularity[item])
                adjusted[slot] += 0.08 * float(item == int(env.current_trend_items[b])) * float(env.current_trend_strength[b])
                adjusted[slot] += 0.22 * float(env.current_neighbor_support[b, int(slot)])
                adjusted[slot] += 0.07 * float(env.current_future_load[b]) * float(local_norm[slot])
                adjusted[slot] += 0.18 * float(env.current_semantic_affinity[b, int(slot)])
                adjusted[slot] += 0.12 * float(env.current_freshness_relevance[b, int(slot)])
        ranked = np.argsort(adjusted)[::-1]
        for slot in ranked.tolist():
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
        out[b] = np.asarray(chosen[: env.cfg.cache_capacity], dtype=np.int64)
        planned[b] = set(out[b].tolist())
    return out


def sample_cache_items(
    logits: torch.Tensor,
    env,
    diversity_penalty: float = 0.0,
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
        for slot in valid_slots:
            item = int(env.current_candidates[b, slot])
            overlap = sum(item in planned[int(n)] for n in neigh)
            local_keep = 0.55 + 0.75 * float(local_norm[slot]) + 0.15 * float(item in env.cache_items[b])
            scores[slot] = scores[slot] - diversity_penalty * float(overlap) * max(0.20, 1.0 - local_keep)
            scores[slot] = scores[slot] + 0.06 * float(item in env.cache_items[b])
            scores[slot] = scores[slot] + 1.15 * float(env.current_neighbor_shortage[b, slot])
            scores[slot] = scores[slot] + 0.14 * float(local_norm[slot])
            scores[slot] = scores[slot] + 0.04 * float(env.global_popularity[item])
            scores[slot] = scores[slot] + 0.08 * float(item == int(env.current_trend_items[b])) * float(env.current_trend_strength[b])
            scores[slot] = scores[slot] + 0.22 * float(env.current_neighbor_support[b, slot])
            scores[slot] = scores[slot] + 0.07 * float(env.current_future_load[b]) * float(local_norm[slot])
            scores[slot] = scores[slot] + 0.18 * float(env.current_semantic_affinity[b, slot])
            scores[slot] = scores[slot] + 0.12 * float(env.current_freshness_relevance[b, slot])

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
                loss = cls_loss + cfg.teacher_score_loss_weight * score_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_losses.append(float(loss.item()))

                use_teacher = np.random.random() < teacher_forcing_prob
                chosen_items = teacher_items if use_teacher else logits_to_cache_items(
                    logits,
                    env,
                    diversity_penalty=cfg.decode_diversity_penalty,
                )
                obs, reward, done, info = env.step_full_cache_items(chosen_items)
                epoch_rewards.append(float(reward))
                epoch_local.append(float(info["local_hit_rate"]))
                epoch_paper.append(float(info["local_hit_rate"] + info["neighbor_fetch_rate"]))

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
            log_probs: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            rewards_per_step: list[float] = []
            raw_rewards_per_step: list[float] = []
            local_per_step: list[float] = []
            paper_per_step: list[float] = []

            while not done:
                node, cand, adj, mask = _obs_to_tensors(obs, device)
                logits = model(node, cand, adj, mask)
                chosen, log_prob, entropy = sample_cache_items(
                    logits,
                    env,
                    diversity_penalty=cfg.decode_diversity_penalty,
                )
                obs, reward, done, info = env.step_full_cache_items(chosen)
                paper_hit = float(info["local_hit_rate"] + info["neighbor_fetch_rate"])
                shaped_reward = float(reward) + cfg.edge_hit_bonus_weight * paper_hit
                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards_per_step.append(shaped_reward)
                raw_rewards_per_step.append(float(reward))
                local_per_step.append(float(info["local_hit_rate"]))
                paper_per_step.append(paper_hit)

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
):
    model = model.to(device)
    model.eval()
    rows = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        reward_sum = 0.0
        local_sum = 0.0
        neighbor_sum = 0.0
        cloud_sum = 0.0
        steps = 0
        while not done:
            node, cand, adj, mask = _obs_to_tensors(obs, torch.device(device))
            logits = model(node, cand, adj, mask)
            chosen = logits_to_cache_items(logits, env, diversity_penalty=decode_diversity_penalty)
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
