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
    def __init__(self, node_feat_dim: int, hidden_dim: int, fp: int) -> None:
        super().__init__()
        self.g1 = GraphConv(node_feat_dim, hidden_dim)
        self.g2 = GraphConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.actor = nn.Linear(hidden_dim, fp)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.g1(node_features, adjacency))
        h = torch.relu(self.g2(h, adjacency))
        return self.norm(h)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # node_features: (B,F), adjacency: (B,B), action_mask: (B,Fp)
        h = self.encode_nodes(node_features, adjacency)
        logits = self.actor(h)
        # Guarantee at least one valid action per SBS node.
        no_valid = action_mask.sum(dim=-1) <= 0
        if torch.any(no_valid):
            action_mask = action_mask.clone()
            action_mask[no_valid, 0] = 1.0
        masked_logits = logits.masked_fill(action_mask <= 0, -1e9)
        global_h = h.mean(dim=0)
        value = self.critic(global_h).squeeze(-1)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_f = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
    adj = torch.as_tensor(obs["adjacency"], dtype=torch.float32, device=device)
    mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=device)
    return node_f, adj, mask


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
    last_gae = 0.0
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
            traj_obs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            traj_actions: list[torch.Tensor] = []
            traj_logp: list[torch.Tensor] = []
            traj_rewards: list[float] = []
            traj_values: list[torch.Tensor] = []
            traj_dones: list[float] = []
            traj_entropy: list[torch.Tensor] = []

            for _ in range(cfg.horizon):
                node_f, adj, mask = _to_tensors(obs, device)
                logits, value = model(node_f, adj, mask)

                dist = Categorical(logits=logits)
                action = dist.sample()  # (B,)
                logp = dist.log_prob(action).sum()
                entropy = dist.entropy().mean()

                next_obs, reward, done, info = env.step(action.detach().cpu().numpy())

                traj_obs.append((node_f, adj, mask))
                traj_actions.append(action)
                traj_logp.append(logp.detach())
                traj_values.append(value)
                traj_rewards.append(float(reward))
                traj_dones.append(float(done))
                traj_entropy.append(entropy.detach())

                ep_reward += float(reward)
                ep_hit += float(info.get("local_hit_rate", 0.0))
                ep_neighbor += float(info.get("neighbor_fetch_rate", 0.0))
                ep_cloud += float(info.get("cloud_fetch_rate", 0.0))
                ep_steps += 1
                obs = next_obs
                if done:
                    break

            rewards_t = torch.as_tensor(traj_rewards, dtype=torch.float32, device=device)
            dones_t = torch.as_tensor(traj_dones, dtype=torch.float32, device=device)
            values_t = torch.stack(traj_values).to(device).view(-1)
            old_logp_t = torch.stack(traj_logp).to(device).view(-1)
            if done:
                bootstrap_value = torch.zeros((), dtype=torch.float32, device=device)
            else:
                with torch.no_grad():
                    n_f, n_adj, n_mask = _to_tensors(obs, device)
                    _, bootstrap_value = model(n_f, n_adj, n_mask)
                    bootstrap_value = bootstrap_value.view(())

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
                for (node_f, adj, mask), action in zip(traj_obs, traj_actions):
                    logits, value = model(node_f, adj, mask)
                    dist = Categorical(logits=logits)
                    new_logps.append(dist.log_prob(action).sum())
                    new_values.append(value.view(()))
                    entropies.append(dist.entropy().mean())

                new_logp_t = torch.stack(new_logps).view(-1)
                new_values_t = torch.stack(new_values).view(-1)
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
