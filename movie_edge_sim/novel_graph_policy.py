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
    def __init__(self, node_feat_dim: int, candidate_feat_dim: int, hidden_dim: int, fp: int) -> None:
        super().__init__()
        self.gat1 = GraphAttention(node_feat_dim, hidden_dim)
        self.gat2 = GraphAttention(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
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
        h = torch.relu(self.gat1(node_features, adjacency))
        h = self.norm(torch.relu(self.gat2(h, adjacency)) + h)
        cand = self.candidate_encoder(candidate_features)
        q = self.query(h).unsqueeze(1).expand(-1, cand.shape[1], -1)
        fused = torch.cat([q * cand, cand], dim=-1)
        logits = self.score_head(fused).squeeze(-1)
        bias = 20.0 * candidate_features[..., 0] + 4.0 * candidate_features[..., 4] + 1.5 * candidate_features[..., 5]
        bias = bias + 1.5 * candidate_features[..., 1] - 1.5 * candidate_features[..., 2]
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


@dataclass
class ImitationHistory:
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


def logits_to_cache_items(logits: torch.Tensor, env) -> np.ndarray:
    topk = min(env.cfg.cache_capacity, logits.shape[1])
    slots = torch.topk(logits, k=topk, dim=-1).indices.detach().cpu().numpy().astype(np.int64)
    out = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
    for b in range(env.cfg.n_sbs):
        chosen = []
        seen: set[int] = set()
        for slot in slots[b]:
            item = int(env.current_candidates[b, int(slot)])
            if item <= 0 or item in seen:
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
    return out


def train_graph_cache_policy_imitation(env, model: TemporalGraphCooperativePolicy, cfg: ImitationConfig, seed: int = 42):
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
                chosen_items = teacher_items if use_teacher else logits_to_cache_items(logits, env)
                obs, reward, done, info = env.step_full_cache_items(chosen_items)
                epoch_rewards.append(float(reward))
                epoch_local.append(float(info["local_hit_rate"]))
                epoch_paper.append(float(info["local_hit_rate"] + info["neighbor_fetch_rate"]))

        epoch_loss_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        epoch_reward_mean = float(np.mean(epoch_rewards)) if epoch_rewards else float("nan")
        epoch_local_mean = float(np.mean(epoch_local)) if epoch_local else float("nan")
        epoch_paper_mean = float(np.mean(epoch_paper)) if epoch_paper else float("nan")

        losses.append(epoch_loss_mean)
        rewards.append(epoch_reward_mean)
        local_hits.append(epoch_local_mean)
        paper_hits.append(epoch_paper_mean)

        epoch_score = epoch_reward_mean + 500.0 * epoch_local_mean + 250.0 * epoch_paper_mean
        if epoch_score > best_score:
            best_score = epoch_score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    return ImitationHistory(
        losses=losses,
        rewards=rewards,
        local_hit_rates=local_hits,
        paper_hit_rates=paper_hits,
    )


@torch.no_grad()
def evaluate_graph_cache_policy(env, model: TemporalGraphCooperativePolicy, episodes: int, seed: int = 42, device: str = "cpu"):
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
            chosen = logits_to_cache_items(logits, env)
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
