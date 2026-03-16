from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, F), adj: (B, B)
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        agg = (adj / deg) @ x
        return self.linear(agg)


class GNNActorCritic(nn.Module):
    def __init__(self, node_feat_dim: int, candidate_feat_dim: int, hidden_dim: int, fp: int) -> None:
        super().__init__()
        self.g1 = GraphConv(node_feat_dim, hidden_dim)
        self.g2 = GraphConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.candidate_mlp = nn.Sequential(
            nn.Linear(hidden_dim + candidate_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.fp = fp
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.candidate_mlp[0].weight)
        nn.init.zeros_(self.candidate_mlp[0].bias)
        nn.init.zeros_(self.candidate_mlp[2].weight)
        nn.init.zeros_(self.candidate_mlp[2].bias)
        nn.init.zeros_(self.critic[0].weight)
        nn.init.zeros_(self.critic[0].bias)
        nn.init.zeros_(self.critic[2].weight)
        nn.init.zeros_(self.critic[2].bias)

    def encode_nodes(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.g1(node_features, adjacency))
        h = torch.relu(self.g2(h, adjacency))
        return self.norm(h)

    def forward(
        self,
        node_features: torch.Tensor,
        candidate_features: torch.Tensor,
        adjacency: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # node_features: (B,F), candidate_features: (B,Fp,C), adjacency: (B,B), action_mask: (B,Fp)
        h = self.encode_nodes(node_features, adjacency)
        repeated_h = h.unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        slot_features = torch.cat([repeated_h, candidate_features], dim=-1)
        learned_logits = self.candidate_mlp(slot_features).squeeze(-1)
        score_bonus = 30.0 * candidate_features[..., 0]
        local_cache_penalty = 3.0 * candidate_features[..., 1]
        neighbor_cache_penalty = 2.5 * candidate_features[..., 2]
        popularity_bonus = 2.0 * candidate_features[..., 3]
        logits = learned_logits + score_bonus + popularity_bonus - local_cache_penalty - neighbor_cache_penalty
        # Guarantee at least one valid action per SBS node.
        no_valid = action_mask.sum(dim=-1) <= 0
        if torch.any(no_valid):
            action_mask = action_mask.clone()
            action_mask[no_valid, 0] = 1.0
        masked_logits = logits.masked_fill(action_mask <= 0, -1e9)
        value = self.critic(h).squeeze(-1)
        return masked_logits, value


@dataclass
class PPOConfig:
    episodes: int = 60
    horizon: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    update_epochs: int = 4
    max_grad_norm: float = 1.0
    device: str = "cpu"


@dataclass
class TrainingHistory:
    episode_rewards: list[float]
    episode_hit_rates: list[float]
    episode_neighbor_rates: list[float]
    episode_cloud_rates: list[float]
    losses: list[float]


def _to_tensors(
    obs: dict[str, np.ndarray], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_f = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
    cand_f = torch.as_tensor(obs["candidate_features"], dtype=torch.float32, device=device)
    adj = torch.as_tensor(obs["adjacency"], dtype=torch.float32, device=device)
    mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=device)
    return node_f, cand_f, adj, mask


def _sample_actions_without_replacement(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, fp = logits.shape
    selected_actions = []
    logps = []
    entropies = []
    available = action_mask.clone()

    for _ in range(k):
        valid_any = available.sum(dim=-1, keepdim=True) > 0
        safe_available = available.clone()
        safe_available[~valid_any.expand_as(safe_available)] = 0.0
        safe_available[~valid_any.squeeze(-1), 0] = 1.0

        masked_logits = logits.masked_fill(safe_available <= 0, -1e9)
        dist = Categorical(logits=masked_logits)
        act = dist.sample()
        selected_actions.append(act)
        logps.append(dist.log_prob(act))
        entropies.append(dist.entropy())
        available.scatter_(1, act.unsqueeze(1), 0.0)

    action_tensor = torch.stack(selected_actions, dim=1)
    logp = torch.stack(logps, dim=1).sum(dim=1)
    entropy = torch.stack(entropies, dim=1).mean(dim=1)
    return action_tensor, logp, entropy


def _logprob_actions_without_replacement(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    available = action_mask.clone()
    logps = []
    entropies = []
    for i in range(actions.shape[1]):
        valid_any = available.sum(dim=-1, keepdim=True) > 0
        safe_available = available.clone()
        safe_available[~valid_any.expand_as(safe_available)] = 0.0
        safe_available[~valid_any.squeeze(-1), 0] = 1.0

        masked_logits = logits.masked_fill(safe_available <= 0, -1e9)
        dist = Categorical(logits=masked_logits)
        act = actions[:, i]
        logps.append(dist.log_prob(act))
        entropies.append(dist.entropy())
        available.scatter_(1, act.unsqueeze(1), 0.0)
    return torch.stack(logps, dim=1).sum(dim=1), torch.stack(entropies, dim=1).mean(dim=1)


def _gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    last_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_value = values[t + 1] if t < T - 1 else last_value
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def train_gnn_ppo(
    env,
    model: GNNActorCritic,
    cfg: PPOConfig,
    seed: int = 42,
    logger: logging.Logger | None = None,
    log_every: int = 1,
) -> TrainingHistory:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    episode_rewards: list[float] = []
    episode_hit_rates: list[float] = []
    episode_neighbor_rates: list[float] = []
    episode_cloud_rates: list[float] = []
    losses: list[float] = []
    action_items_per_sbs = int(getattr(env.cfg, "cache_capacity", 1))

    if logger is not None:
        logger.info(
            "GNN PPO training started: episodes=%d horizon=%d update_epochs=%d",
            cfg.episodes,
            cfg.horizon,
            cfg.update_epochs,
        )

    for ep in range(cfg.episodes):
        obs = env.reset(seed=seed + ep)
        done = False

        ep_reward = 0.0
        ep_hit = 0.0
        ep_neighbor = 0.0
        ep_cloud = 0.0
        ep_steps = 0

        while not done:
            traj_obs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            traj_actions: list[torch.Tensor] = []
            traj_logp: list[torch.Tensor] = []
            traj_rewards: list[torch.Tensor] = []
            traj_values: list[torch.Tensor] = []
            traj_dones: list[torch.Tensor] = []
            traj_entropy: list[torch.Tensor] = []

            for _ in range(cfg.horizon):
                node_f, cand_f, adj, mask = _to_tensors(obs, device)
                logits, value = model(node_f, cand_f, adj, mask)

                action, action_logp, action_entropy = _sample_actions_without_replacement(
                    logits, mask, action_items_per_sbs
                )
                action_items = env.candidate_indices_to_items(action.detach().cpu().numpy(), k=action_items_per_sbs)
                next_obs, reward, done, info = env.step_full_cache_items(action_items)
                reward_per_sbs = torch.as_tensor(info["reward_per_sbs"], dtype=torch.float32, device=device)

                traj_obs.append((node_f, cand_f, adj, mask))
                traj_actions.append(action)
                traj_logp.append(action_logp.detach())
                traj_values.append(value)
                traj_rewards.append(reward_per_sbs)
                traj_dones.append(torch.full_like(reward_per_sbs, float(done)))
                traj_entropy.append(action_entropy.detach())

                ep_reward += float(reward)
                ep_hit += float(info.get("local_hit_rate", 0.0))
                ep_neighbor += float(info.get("neighbor_fetch_rate", 0.0))
                ep_cloud += float(info.get("cloud_fetch_rate", 0.0))
                ep_steps += 1
                obs = next_obs
                if done:
                    break

            rewards_t = torch.stack(traj_rewards).to(device)
            dones_t = torch.stack(traj_dones).to(device)
            values_t = torch.stack(traj_values).to(device)
            old_logp_t = torch.stack(traj_logp).to(device)
            if done:
                bootstrap_value = torch.zeros((env.cfg.n_sbs,), dtype=torch.float32, device=device)
            else:
                with torch.no_grad():
                    n_f, n_cand_f, n_adj, n_mask = _to_tensors(obs, device)
                    _, bootstrap_value = model(n_f, n_cand_f, n_adj, n_mask)

            advantages, returns = _gae(
                rewards_t,
                values_t.detach(),
                dones_t,
                cfg.gamma,
                cfg.gae_lambda,
                bootstrap_value.detach(),
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(cfg.update_epochs):
                new_logps = []
                new_values = []
                entropies = []
                for (node_f, cand_f, adj, mask), action in zip(traj_obs, traj_actions):
                    logits, value = model(node_f, cand_f, adj, mask)
                    action_logp, action_entropy = _logprob_actions_without_replacement(logits, mask, action)
                    new_logps.append(action_logp)
                    new_values.append(value)
                    entropies.append(action_entropy)

                new_logp_t = torch.stack(new_logps)
                new_values_t = torch.stack(new_values)
                entropy_t = torch.stack(entropies).mean()

                ratio = torch.exp(new_logp_t - old_logp_t)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.mean((new_values_t - returns) ** 2)
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_t

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                losses.append(float(loss.item()))

        episode_rewards.append(ep_reward)
        episode_hit_rates.append(ep_hit / max(1, ep_steps))
        episode_neighbor_rates.append(ep_neighbor / max(1, ep_steps))
        episode_cloud_rates.append(ep_cloud / max(1, ep_steps))
        if logger is not None and ((ep + 1) % max(1, log_every) == 0 or (ep + 1) == cfg.episodes):
            logger.info(
                "RL episode %d/%d complete | reward=%.6f local_hit_rate=%.6f neighbor_rate=%.6f cloud_rate=%.6f steps=%d",
                ep + 1,
                cfg.episodes,
                episode_rewards[-1],
                episode_hit_rates[-1],
                episode_neighbor_rates[-1],
                episode_cloud_rates[-1],
                ep_steps,
            )

    return TrainingHistory(
        episode_rewards=episode_rewards,
        episode_hit_rates=episode_hit_rates,
        episode_neighbor_rates=episode_neighbor_rates,
        episode_cloud_rates=episode_cloud_rates,
        losses=losses,
    )
