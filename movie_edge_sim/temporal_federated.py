from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from movie_edge_sim.temporal_requests import TemporalDataset


class TemporalWindowTorchDataset(Dataset):
    def __init__(self, temporal: TemporalDataset) -> None:
        self.contexts = torch.as_tensor(temporal.contexts, dtype=torch.long)
        self.targets = torch.as_tensor(temporal.targets, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.contexts.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.contexts[idx], self.targets[idx]


class TemporalSpikeEncoder(nn.Module):
    """
    Temporal encoder replacing AAE:
    - captures local temporal spikes with Conv1d
    - captures dependencies with GRU
    - predicts next requested item
    """

    def __init__(
        self,
        num_items: int,
        window_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.window_size = window_size
        self.embed = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.temporal_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.out = nn.Linear(embed_dim, num_items + 1)

    def encode(self, context: torch.Tensor) -> torch.Tensor:
        x = self.embed(context)  # (B, K, E)
        x_conv = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        x = torch.relu(x + x_conv)
        x, _ = self.gru(x)
        h = x[:, -1, :]
        z = self.dropout(torch.relu(self.proj(h)))
        return z

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        z = self.encode(context)
        return self.out(z)

    @torch.no_grad()
    def predict_scores(self, context: torch.Tensor) -> torch.Tensor:
        logits = self.forward(context)
        probs = torch.softmax(logits, dim=-1)
        return probs


@dataclass
class FederatedConfig:
    rounds: int = 15
    clients_per_round: int = 60
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    elastic_tau: float = 2.0
    seed: int = 42
    device: str = "cpu"


@dataclass
class FederatedTrainResult:
    model: TemporalSpikeEncoder
    round_losses: list[float]
    val_losses: list[float]


def _train_one_local_model(
    model: TemporalSpikeEncoder,
    dataset: TemporalWindowTorchDataset,
    indices: np.ndarray,
    cfg: FederatedConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    model = copy.deepcopy(model).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    subset = Subset(dataset, [int(i) for i in indices.tolist()])
    loader = DataLoader(subset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    running = 0.0
    count = 0
    for _ in range(cfg.local_epochs):
        for context, target in loader:
            context = context.to(device)
            target = target.to(device)
            logits = model(context)
            loss = loss_fn(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            count += 1

    local_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    avg_loss = running / max(1, count)
    return local_state, avg_loss


def _state_l2_distance(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in a:
        diff = (a[key].float() - b[key].float()).pow(2).mean().item()
        total += diff
    return float(np.sqrt(total))


def _aggregate_states(
    global_state: dict[str, torch.Tensor],
    local_states: list[dict[str, torch.Tensor]],
    local_sizes: list[int],
    distances: list[float],
    tau: float,
) -> dict[str, torch.Tensor]:
    if not local_states:
        return global_state

    sizes = np.asarray(local_sizes, dtype=np.float64)
    dists = np.asarray(distances, dtype=np.float64)
    elastic = np.exp(-dists / max(1e-8, tau))
    weights = sizes * elastic
    weights = weights / max(1e-12, weights.sum())

    agg: dict[str, torch.Tensor] = {}
    for key in global_state:
        stacked = torch.stack([ls[key].float() for ls in local_states], dim=0)
        w = torch.as_tensor(weights, dtype=stacked.dtype).view(-1, *([1] * (stacked.ndim - 1)))
        agg[key] = (stacked * w).sum(dim=0).to(dtype=global_state[key].dtype)
    return agg


@torch.no_grad()
def evaluate_next_item_loss(
    model: TemporalSpikeEncoder,
    dataset: TemporalWindowTorchDataset,
    indices: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> float:
    model.eval()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    subset = Subset(dataset, [int(i) for i in indices.tolist()])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False)

    total_loss = 0.0
    total_count = 0
    for context, target in loader:
        context = context.to(device)
        target = target.to(device)
        logits = model(context)
        loss = loss_fn(logits, target)
        bs = int(context.shape[0])
        total_loss += float(loss.item()) * bs
        total_count += bs
    return total_loss / max(1, total_count)


def train_temporal_encoder_federated(
    temporal: TemporalDataset,
    train_user_indices: dict[int, np.ndarray],
    val_indices: np.ndarray,
    cfg: FederatedConfig,
    embed_dim: int = 64,
    hidden_dim: int = 96,
    dropout: float = 0.1,
    logger: logging.Logger | None = None,
    log_every: int = 1,
) -> FederatedTrainResult:
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)

    torch_dataset = TemporalWindowTorchDataset(temporal)
    model = TemporalSpikeEncoder(
        num_items=temporal.num_items,
        window_size=temporal.window_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    round_losses: list[float] = []
    val_losses: list[float] = []

    user_ids = np.asarray(sorted(train_user_indices.keys()), dtype=np.int64)
    if logger is not None:
        logger.info(
            "Temporal federated training started: rounds=%d, clients_per_round=%d, users=%d",
            cfg.rounds,
            cfg.clients_per_round,
            len(user_ids),
        )

    for r in range(cfg.rounds):
        global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        n_clients = min(cfg.clients_per_round, len(user_ids))
        chosen = rng.choice(user_ids, size=n_clients, replace=False)

        local_states: list[dict[str, torch.Tensor]] = []
        local_sizes: list[int] = []
        distances: list[float] = []
        local_losses: list[float] = []

        for user_id in chosen:
            idx = train_user_indices[int(user_id)]
            if idx.size == 0:
                continue
            local_state, local_loss = _train_one_local_model(model, torch_dataset, idx, cfg, device)
            dist = _state_l2_distance(local_state, global_state)
            local_states.append(local_state)
            local_sizes.append(int(idx.size))
            distances.append(dist)
            local_losses.append(local_loss)

        aggregated = _aggregate_states(global_state, local_states, local_sizes, distances, cfg.elastic_tau)
        model.load_state_dict(aggregated)

        round_loss = float(np.mean(local_losses)) if local_losses else 0.0
        round_losses.append(round_loss)
        val_loss = evaluate_next_item_loss(model, torch_dataset, val_indices, device=cfg.device)
        val_losses.append(val_loss)
        if logger is not None and ((r + 1) % max(1, log_every) == 0 or (r + 1) == cfg.rounds):
            logger.info(
                "Temporal round %d/%d complete | train_loss=%.6f val_loss=%.6f active_clients=%d",
                r + 1,
                cfg.rounds,
                round_loss,
                val_loss,
                len(local_losses),
            )

    return FederatedTrainResult(model=model, round_losses=round_losses, val_losses=val_losses)
