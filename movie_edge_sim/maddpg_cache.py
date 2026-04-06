from __future__ import annotations

import copy
import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(state))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


@dataclass
class MADDPGConfig:
    episodes: int = 120
    batch_size: int = 64
    replay_size: int = 10000
    gamma: float = 0.99
    tau: float = 0.01
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    hidden_dim: int = 128
    warmup_steps: int = 100
    update_every: int = 1
    noise_std: float = 0.10
    device: str = "cpu"


@dataclass
class MADDPGHistory:
    episode_rewards: list[float]
    episode_local_hit_rates: list[float]
    episode_neighbor_rates: list[float]
    episode_cloud_rates: list[float]
    global_critic_losses: list[float]
    local_critic_losses: list[float]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[dict[str, np.ndarray | float | bool]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: dict[str, np.ndarray | float | bool]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[int(i)] for i in idx]
        keys = batch[0].keys()
        out: dict[str, np.ndarray] = {}
        for key in keys:
            vals = [item[key] for item in batch]
            out[key] = np.asarray(vals)
        return out


class MADDPGCachePolicy:
    def __init__(self, n_agents: int, state_dim: int, action_dim: int, cfg: MADDPGConfig) -> None:
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actors = [Actor(state_dim, cfg.hidden_dim, action_dim).to(self.device) for _ in range(n_agents)]
        self.target_actors = [copy.deepcopy(actor).to(self.device) for actor in self.actors]
        self.local_critics = [Critic(state_dim, action_dim, cfg.hidden_dim).to(self.device) for _ in range(n_agents)]
        self.target_local_critics = [copy.deepcopy(critic).to(self.device) for critic in self.local_critics]

        joint_state_dim = n_agents * state_dim
        joint_action_dim = n_agents * action_dim
        self.global_critics = [Critic(joint_state_dim, joint_action_dim, cfg.hidden_dim).to(self.device) for _ in range(2)]
        self.target_global_critics = [copy.deepcopy(critic).to(self.device) for critic in self.global_critics]

        self.actor_opts = [torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr) for actor in self.actors]
        self.local_critic_opts = [torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr) for critic in self.local_critics]
        self.global_critic_opts = [torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr) for critic in self.global_critics]

    @torch.no_grad()
    def act(self, local_states: np.ndarray, explore: bool = True) -> np.ndarray:
        state = torch.as_tensor(local_states, dtype=torch.float32, device=self.device)
        outs = []
        for agent_id, actor in enumerate(self.actors):
            action = actor(state[agent_id : agent_id + 1]).squeeze(0)
            if explore:
                action = action + self.cfg.noise_std * torch.randn_like(action)
            outs.append(action.clamp(0.0, 1.0).detach().cpu().numpy())
        return np.stack(outs, axis=0).astype(np.float32)

    def state_dict(self) -> dict[str, object]:
        return {
            "actors": [actor.state_dict() for actor in self.actors],
            "local_critics": [critic.state_dict() for critic in self.local_critics],
            "global_critics": [critic.state_dict() for critic in self.global_critics],
            "cfg": {
                "n_agents": self.n_agents,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        for actor, actor_state in zip(self.actors, state["actors"]):
            actor.load_state_dict(actor_state)
        for target, src in zip(self.target_actors, self.actors):
            target.load_state_dict(src.state_dict())
        for critic, critic_state in zip(self.local_critics, state["local_critics"]):
            critic.load_state_dict(critic_state)
        for target, src in zip(self.target_local_critics, self.local_critics):
            target.load_state_dict(src.state_dict())
        for critic, critic_state in zip(self.global_critics, state["global_critics"]):
            critic.load_state_dict(critic_state)
        for target, src in zip(self.target_global_critics, self.global_critics):
            target.load_state_dict(src.state_dict())

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.cfg.tau * source_param.data + (1.0 - self.cfg.tau) * target_param.data)

    def update(self, batch: dict[str, np.ndarray]) -> tuple[float, float]:
        device = self.device
        local_states = torch.as_tensor(batch["local_states"], dtype=torch.float32, device=device)
        next_local_states = torch.as_tensor(batch["next_local_states"], dtype=torch.float32, device=device)
        global_states = torch.as_tensor(batch["global_states"], dtype=torch.float32, device=device)
        next_global_states = torch.as_tensor(batch["next_global_states"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
        local_rewards = torch.as_tensor(batch["local_rewards"], dtype=torch.float32, device=device)
        global_rewards = torch.as_tensor(batch["global_rewards"], dtype=torch.float32, device=device).unsqueeze(-1)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device).unsqueeze(-1)

        batch_size = local_states.shape[0]
        next_joint_actions = []
        for agent_id in range(self.n_agents):
            next_joint_actions.append(self.target_actors[agent_id](next_local_states[:, agent_id, :]))
        next_joint_actions_t = torch.cat(next_joint_actions, dim=-1)
        actions_flat = actions.reshape(batch_size, -1)

        with torch.no_grad():
            y_global = global_rewards + self.cfg.gamma * (
                torch.minimum(
                    self.target_global_critics[0](next_global_states, next_joint_actions_t),
                    self.target_global_critics[1](next_global_states, next_joint_actions_t),
                )
            ) * (1.0 - dones)

        global_losses = []
        for critic, opt in zip(self.global_critics, self.global_critic_opts):
            pred = critic(global_states, actions_flat)
            loss = torch.mean((pred - y_global) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_losses.append(float(loss.item()))

        local_losses = []

        for agent_id in range(self.n_agents):
            with torch.no_grad():
                next_local_action = self.target_actors[agent_id](next_local_states[:, agent_id, :])
                y_local = local_rewards[:, agent_id : agent_id + 1] + self.cfg.gamma * self.target_local_critics[
                    agent_id
                ](next_local_states[:, agent_id, :], next_local_action) * (1.0 - dones)

            local_pred = self.local_critics[agent_id](local_states[:, agent_id, :], actions[:, agent_id, :])
            local_loss = torch.mean((local_pred - y_local) ** 2)
            self.local_critic_opts[agent_id].zero_grad()
            local_loss.backward()
            self.local_critic_opts[agent_id].step()
            local_losses.append(float(local_loss.item()))

            self.actor_opts[agent_id].zero_grad()
            # Rebuild the joint action for each agent update so only the
            # current actor keeps a live computation graph. Other agents'
            # actions are detached, which matches decentralized actor updates
            # and avoids backpropagating through a freed shared graph.
            joint_actions_for_actor = []
            local_action = None
            for other_id in range(self.n_agents):
                action = self.actors[other_id](local_states[:, other_id, :])
                if other_id == agent_id:
                    local_action = action
                    joint_actions_for_actor.append(action)
                else:
                    joint_actions_for_actor.append(action.detach())
            assert local_action is not None
            joint_for_actor = torch.cat(joint_actions_for_actor, dim=-1)
            actor_loss = -(
                self.global_critics[0](global_states, joint_for_actor).mean()
                + self.local_critics[agent_id](local_states[:, agent_id, :], local_action).mean()
            )
            actor_loss.backward()
            self.actor_opts[agent_id].step()

        for target, src in zip(self.target_actors, self.actors):
            self._soft_update(target, src)
        for target, src in zip(self.target_local_critics, self.local_critics):
            self._soft_update(target, src)
        for target, src in zip(self.target_global_critics, self.global_critics):
            self._soft_update(target, src)

        return float(np.mean(global_losses)), float(np.mean(local_losses))


def train_maddpg_cache_policy(
    env,
    policy: MADDPGCachePolicy,
    cfg: MADDPGConfig,
    seed: int = 42,
    logger: logging.Logger | None = None,
    log_every: int = 1,
) -> MADDPGHistory:
    np.random.seed(seed)
    torch.manual_seed(seed)
    replay = ReplayBuffer(cfg.replay_size)

    episode_rewards: list[float] = []
    episode_local_hit_rates: list[float] = []
    episode_neighbor_rates: list[float] = []
    episode_cloud_rates: list[float] = []
    global_critic_losses: list[float] = []
    local_critic_losses: list[float] = []

    total_steps = 0
    if logger is not None:
        logger.info(
            "MADDPG training started | episodes=%d batch_size=%d warmup_steps=%d",
            cfg.episodes,
            cfg.batch_size,
            cfg.warmup_steps,
        )

    for episode in range(cfg.episodes):
        obs = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0
        ep_local = 0.0
        ep_neighbor = 0.0
        ep_cloud = 0.0
        ep_steps = 0

        while not done:
            action_scores = policy.act(obs["local_states"], explore=True)
            action_items = env.action_scores_to_items(action_scores)
            binary_action = env.items_to_binary_action(action_items)
            next_obs, reward, local_rewards, done, info = env.step_items(action_items)

            replay.push(
                {
                    "local_states": obs["local_states"].astype(np.float32),
                    "next_local_states": next_obs["local_states"].astype(np.float32),
                    "global_states": obs["global_state"].astype(np.float32),
                    "next_global_states": next_obs["global_state"].astype(np.float32),
                    "actions": binary_action.astype(np.float32),
                    "local_rewards": local_rewards.astype(np.float32),
                    "global_rewards": np.float32(reward),
                    "dones": np.float32(done),
                }
            )

            obs = next_obs
            ep_reward += float(reward)
            ep_local += float(info["local_hit_rate"])
            ep_neighbor += float(info["neighbor_fetch_rate"])
            ep_cloud += float(info["cloud_fetch_rate"])
            ep_steps += 1
            total_steps += 1

            if len(replay) >= max(cfg.batch_size, cfg.warmup_steps) and total_steps % max(1, cfg.update_every) == 0:
                batch = replay.sample(cfg.batch_size)
                g_loss, l_loss = policy.update(batch)
                global_critic_losses.append(g_loss)
                local_critic_losses.append(l_loss)

        episode_rewards.append(ep_reward)
        episode_local_hit_rates.append(ep_local / max(1, ep_steps))
        episode_neighbor_rates.append(ep_neighbor / max(1, ep_steps))
        episode_cloud_rates.append(ep_cloud / max(1, ep_steps))
        if logger is not None and ((episode + 1) % max(1, log_every) == 0 or (episode + 1) == cfg.episodes):
            logger.info(
                "MADDPG episode %d/%d complete | reward=%.6f local_hit=%.6f neighbor=%.6f cloud=%.6f",
                episode + 1,
                cfg.episodes,
                episode_rewards[-1],
                episode_local_hit_rates[-1],
                episode_neighbor_rates[-1],
                episode_cloud_rates[-1],
            )

    return MADDPGHistory(
        episode_rewards=episode_rewards,
        episode_local_hit_rates=episode_local_hit_rates,
        episode_neighbor_rates=episode_neighbor_rates,
        episode_cloud_rates=episode_cloud_rates,
        global_critic_losses=global_critic_losses,
        local_critic_losses=local_critic_losses,
    )
