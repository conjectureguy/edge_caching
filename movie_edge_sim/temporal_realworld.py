from __future__ import annotations

import copy
import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class RealWorldTemporalDataset:
    context_items: np.ndarray
    context_deltas: np.ndarray
    context_hours: np.ndarray
    target_items: np.ndarray
    user_ids: np.ndarray
    num_users: int
    num_items: int
    window_size: int


@dataclass
class UserTimeHistory:
    items: list[int]
    timestamps: list[int]


def build_user_time_histories(ratings: list[dict[str, int]]) -> dict[int, UserTimeHistory]:
    by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in ratings:
        by_user[row["user_id"]].append((row["timestamp"], row["item_id"]))

    histories: dict[int, UserTimeHistory] = {}
    for user_id, rows in by_user.items():
        rows.sort(key=lambda x: x[0])
        histories[user_id] = UserTimeHistory(
            items=[item for _, item in rows],
            timestamps=[ts for ts, _ in rows],
        )
    return histories


def build_realworld_temporal_dataset(
    histories: dict[int, UserTimeHistory],
    window_size: int,
    min_history: int | None = None,
) -> RealWorldTemporalDataset:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    minimum = min_history if min_history is not None else window_size + 1
    context_items: list[list[int]] = []
    context_deltas: list[list[float]] = []
    context_hours: list[list[int]] = []
    target_items: list[int] = []
    user_ids: list[int] = []

    max_item_id = 0
    max_user_id = 0
    for user_id, hist in histories.items():
        if len(hist.items) < minimum:
            continue
        max_user_id = max(max_user_id, user_id)
        max_item_id = max(max_item_id, max(hist.items))
        ts = hist.timestamps
        items = hist.items
        for t in range(window_size, len(items)):
            prev_ts = ts[t - window_size : t]
            prev_items = items[t - window_size : t]
            deltas = [0.0]
            for i in range(1, len(prev_ts)):
                delta_hours = max(0.0, float(prev_ts[i] - prev_ts[i - 1]) / 3600.0)
                deltas.append(np.log1p(delta_hours))
            hours = [int((stamp // 3600) % 24) for stamp in prev_ts]
            context_items.append(prev_items)
            context_deltas.append(deltas)
            context_hours.append(hours)
            target_items.append(items[t])
            user_ids.append(user_id)

    if not context_items:
        raise ValueError("No timestamp-aware temporal samples generated.")

    return RealWorldTemporalDataset(
        context_items=np.asarray(context_items, dtype=np.int64),
        context_deltas=np.asarray(context_deltas, dtype=np.float32),
        context_hours=np.asarray(context_hours, dtype=np.int64),
        target_items=np.asarray(target_items, dtype=np.int64),
        user_ids=np.asarray(user_ids, dtype=np.int64),
        num_users=max_user_id,
        num_items=max_item_id,
        window_size=window_size,
    )


def chronological_train_val_split(
    dataset: RealWorldTemporalDataset,
    val_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    per_user: dict[int, list[int]] = defaultdict(list)
    for idx, user_id in enumerate(dataset.user_ids.tolist()):
        per_user[int(user_id)].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for idxs in per_user.values():
        n_val = max(1, int(len(idxs) * val_ratio))
        split = len(idxs) - n_val
        train_idx.extend(idxs[:split])
        val_idx.extend(idxs[split:])
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def grouped_indices_by_user(dataset: RealWorldTemporalDataset, indices: np.ndarray | None = None) -> dict[int, np.ndarray]:
    if indices is None:
        indices = np.arange(dataset.context_items.shape[0], dtype=np.int64)
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx in indices:
        grouped[int(dataset.user_ids[int(idx)])].append(int(idx))
    return {uid: np.asarray(idxs, dtype=np.int64) for uid, idxs in grouped.items()}


class RealWorldTemporalTorchDataset(Dataset):
    def __init__(self, temporal: RealWorldTemporalDataset) -> None:
        self.items = torch.as_tensor(temporal.context_items, dtype=torch.long)
        self.deltas = torch.as_tensor(temporal.context_deltas, dtype=torch.float32)
        self.hours = torch.as_tensor(temporal.context_hours, dtype=torch.long)
        self.targets = torch.as_tensor(temporal.target_items, dtype=torch.long)
        self.user_ids = torch.as_tensor(temporal.user_ids, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.items.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.items[idx], self.deltas[idx], self.hours[idx], self.user_ids[idx], self.targets[idx]


class Time2Vec(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        if out_dim < 2:
            raise ValueError("out_dim must be >= 2")
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, out_dim - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = self.linear(x)
        periodic = torch.sin(self.periodic(x))
        return torch.cat([linear, periodic], dim=-1)


class RealWorldTemporalEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_users: int,
        window_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.num_users = num_users
        self.window_size = window_size
        self.embed = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.user_embed = nn.Embedding(num_users + 1, embed_dim)
        self.hour_embed = nn.Embedding(24, embed_dim)
        self.time2vec = Time2Vec(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size, embed_dim))
        self.log_taus = nn.Parameter(torch.log(torch.as_tensor([0.5, 2.0, 8.0, 24.0], dtype=torch.float32)))
        self.temporal_kernel_proj = nn.Sequential(
            nn.Linear(2 + 2 * self.log_taus.numel(), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.input_norm = nn.LayerNorm(embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.local_pattern_mixer = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
        )
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.proj = nn.Sequential(
            nn.Linear(3 * hidden_dim + embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_bias = nn.Parameter(torch.zeros(num_items + 1))
        self.contrast_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _ages_from_deltas(self, context_deltas: torch.Tensor) -> torch.Tensor:
        delta_hours = torch.expm1(context_deltas.clamp_min(0.0))
        if context_deltas.shape[1] <= 1:
            return torch.zeros_like(context_deltas)
        suffix = torch.flip(torch.cumsum(torch.flip(delta_hours[:, 1:], dims=[1]), dim=1), dims=[1])
        zeros = torch.zeros((context_deltas.shape[0], 1), dtype=context_deltas.dtype, device=context_deltas.device)
        return torch.cat([suffix, zeros], dim=1)

    def _temporal_kernel_embedding(self, context_deltas: torch.Tensor) -> torch.Tensor:
        ages = self._ages_from_deltas(context_deltas)
        taus = torch.exp(self.log_taus).view(1, 1, -1)
        delta_hours = torch.expm1(context_deltas.clamp_min(0.0))
        gap_kernel = torch.exp(-delta_hours.unsqueeze(-1) / taus)
        age_kernel = torch.exp(-ages.unsqueeze(-1) / taus)
        raw = torch.cat(
            [
                context_deltas.unsqueeze(-1),
                torch.log1p(ages).unsqueeze(-1),
                gap_kernel,
                age_kernel,
            ],
            dim=-1,
        )
        return self.temporal_kernel_proj(raw)

    def _last_valid_hidden(self, seq: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid_counts = (~padding_mask).sum(dim=1).clamp(min=1)
        last_idx = valid_counts - 1
        batch_idx = torch.arange(seq.shape[0], device=seq.device)
        return seq[batch_idx, last_idx]

    def _sequence_forward(
        self,
        context_items: torch.Tensor,
        context_deltas: torch.Tensor,
        context_hours: torch.Tensor,
        user_ids: torch.Tensor,
        mask_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = context_items <= 0
        item_emb = self.embed(context_items)
        if mask_positions is not None:
            item_emb = torch.where(mask_positions.unsqueeze(-1), self.mask_token.view(1, 1, -1), item_emb)
        hour_emb = self.hour_embed(context_hours.clamp(min=0, max=23))
        delta_emb = self.time2vec(context_deltas.unsqueeze(-1))
        kernel_emb = self._temporal_kernel_embedding(context_deltas)
        user_emb = self.user_embed(user_ids).unsqueeze(1)
        x = item_emb + hour_emb + delta_emb + kernel_emb + user_emb + self.pos_embed[:, : context_items.shape[1], :]
        x = self.input_norm(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        local = self.local_pattern_mixer(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.dropout(local)
        x, _ = self.gru(x)

        attn_logits = self.attn(x).squeeze(-1).masked_fill(padding_mask, -1e9)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.where(padding_mask, torch.zeros_like(attn), attn)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = (x * attn.unsqueeze(-1)).sum(dim=1)
        last_hidden = self._last_valid_hidden(x, padding_mask)
        denom = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
        mean_hidden = (x * (~padding_mask).unsqueeze(-1)).sum(dim=1) / denom
        fused = torch.cat([pooled, last_hidden, mean_hidden, self.user_embed(user_ids)], dim=-1)
        seq_embed = self.proj(fused)
        return x, self.output_norm(seq_embed)

    def _item_logits(self, reps: torch.Tensor) -> torch.Tensor:
        return F.linear(reps, self.embed.weight, self.output_bias)

    def encode(
        self,
        context_items: torch.Tensor,
        context_deltas: torch.Tensor,
        context_hours: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        _seq, pooled = self._sequence_forward(context_items, context_deltas, context_hours, user_ids)
        return pooled

    def project_contrastive(
        self,
        context_items: torch.Tensor,
        context_deltas: torch.Tensor,
        context_hours: torch.Tensor,
        user_ids: torch.Tensor,
        mask_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.encode_with_masks(context_items, context_deltas, context_hours, user_ids, mask_positions)
        return F.normalize(self.contrast_proj(z), dim=-1)

    def encode_with_masks(
        self,
        context_items: torch.Tensor,
        context_deltas: torch.Tensor,
        context_hours: torch.Tensor,
        user_ids: torch.Tensor,
        mask_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _seq, pooled = self._sequence_forward(context_items, context_deltas, context_hours, user_ids, mask_positions)
        return pooled

    def forward(
        self,
        context_items: torch.Tensor,
        context_deltas: torch.Tensor,
        context_hours: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(context_items, context_deltas, context_hours, user_ids)
        return self._item_logits(z)

    def masked_item_logits(
        self,
        context_items: torch.Tensor,
        context_deltas: torch.Tensor,
        context_hours: torch.Tensor,
        user_ids: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        seq, _pooled = self._sequence_forward(context_items, context_deltas, context_hours, user_ids, mask_positions)
        token_embed = self.token_proj(seq)
        return self._item_logits(token_embed)

    @torch.no_grad()
    def predict_scores(
        self,
        context_items: torch.Tensor,
        context_deltas: torch.Tensor,
        context_hours: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(context_items, context_deltas, context_hours, user_ids)
        return torch.softmax(logits, dim=-1)


@dataclass
class FederatedConfig:
    rounds: int = 15
    clients_per_round: int = 60
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    elastic_tau: float = 2.0
    mask_prob: float = 0.20
    mlm_weight: float = 0.35
    contrastive_weight: float = 0.10
    contrastive_temperature: float = 0.20
    seed: int = 42
    device: str = "cpu"


@dataclass
class FederatedTrainResult:
    model: RealWorldTemporalEncoder
    round_losses: list[float]
    val_losses: list[float]


def load_compatible_temporal_state(
    model: RealWorldTemporalEncoder,
    state: dict[str, torch.Tensor],
    logger: logging.Logger | None = None,
    source: str | None = None,
) -> None:
    current = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state.items():
        if key not in current or current[key].shape != value.shape:
            skipped.append(key)
            continue
        matched[key] = value
    current.update(matched)
    model.load_state_dict(current)
    if logger is not None:
        origin = source or "temporal checkpoint"
        logger.info(
            "Loaded compatible %s | matched=%d skipped=%d",
            origin,
            len(matched),
            len(skipped),
        )
        if skipped:
            logger.info("Skipped temporal keys: %s", ", ".join(skipped[:8]) + (" ..." if len(skipped) > 8 else ""))


def _sample_mask_positions(items: torch.Tensor, mask_prob: float) -> torch.Tensor:
    if mask_prob <= 0.0:
        return torch.zeros_like(items, dtype=torch.bool)
    valid = items > 0
    mask = (torch.rand_like(items, dtype=torch.float32) < mask_prob) & valid
    missing = (~mask.any(dim=1)) & valid.any(dim=1)
    if missing.any():
        valid_counts = valid.sum(dim=1)
        last_valid = (valid_counts - 1).clamp(min=0)
        mask[missing, last_valid[missing]] = True
    return mask


def _masked_item_loss(
    model: RealWorldTemporalEncoder,
    items: torch.Tensor,
    deltas: torch.Tensor,
    hours: torch.Tensor,
    users: torch.Tensor,
    mask_prob: float,
) -> torch.Tensor:
    mask_positions = _sample_mask_positions(items, mask_prob)
    if not mask_positions.any():
        return items.new_zeros((), dtype=torch.float32)
    logits = model.masked_item_logits(items, deltas, hours, users, mask_positions)
    return F.cross_entropy(logits[mask_positions], items[mask_positions])


def _contrastive_loss(
    model: RealWorldTemporalEncoder,
    items: torch.Tensor,
    deltas: torch.Tensor,
    hours: torch.Tensor,
    users: torch.Tensor,
    mask_prob: float,
    temperature: float,
) -> torch.Tensor:
    if items.shape[0] < 2:
        return items.new_zeros((), dtype=torch.float32)
    view1 = _sample_mask_positions(items, max(0.05, mask_prob * 0.75))
    view2 = _sample_mask_positions(items, max(0.05, mask_prob * 0.75))
    z1 = model.project_contrastive(items, deltas, hours, users, view1)
    z2 = model.project_contrastive(items, deltas, hours, users, view2)
    reps = torch.cat([z1, z2], dim=0)
    logits = torch.matmul(reps, reps.t()) / max(temperature, 1e-6)
    logits.fill_diagonal_(-1e9)
    batch = z1.shape[0]
    targets = torch.cat(
        [
            torch.arange(batch, 2 * batch, device=items.device),
            torch.arange(0, batch, device=items.device),
        ],
        dim=0,
    )
    return F.cross_entropy(logits, targets)


def _train_one_local_model(
    model: RealWorldTemporalEncoder,
    dataset: RealWorldTemporalTorchDataset,
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

    total_loss = 0.0
    total_steps = 0
    for _ in range(cfg.local_epochs):
        for items, deltas, hours, users, targets in loader:
            items = items.to(device)
            deltas = deltas.to(device)
            hours = hours.to(device)
            users = users.to(device)
            targets = targets.to(device)
            logits = model(items, deltas, hours, users)
            next_loss = loss_fn(logits, targets)
            mlm_loss = _masked_item_loss(model, items, deltas, hours, users, cfg.mask_prob)
            cont_loss = _contrastive_loss(
                model,
                items,
                deltas,
                hours,
                users,
                cfg.mask_prob,
                cfg.contrastive_temperature,
            )
            loss = next_loss + cfg.mlm_weight * mlm_loss + cfg.contrastive_weight * cont_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            total_steps += 1

    local_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return local_state, total_loss / max(1, total_steps)


def _state_l2_distance(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in a:
        total += (a[key].float() - b[key].float()).pow(2).mean().item()
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
    weights = weights / max(weights.sum(), 1e-12)

    out: dict[str, torch.Tensor] = {}
    for key in global_state:
        stacked = torch.stack([state[key].float() for state in local_states], dim=0)
        w = torch.as_tensor(weights, dtype=stacked.dtype).view(-1, *([1] * (stacked.ndim - 1)))
        out[key] = (stacked * w).sum(dim=0).to(dtype=global_state[key].dtype)
    return out


@torch.no_grad()
def evaluate_next_item_loss(
    model: RealWorldTemporalEncoder,
    dataset: RealWorldTemporalTorchDataset,
    indices: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> float:
    model.eval().to(device)
    subset = Subset(dataset, [int(i) for i in indices.tolist()])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0
    for items, deltas, hours, users, targets in loader:
        items = items.to(device)
        deltas = deltas.to(device)
        hours = hours.to(device)
        users = users.to(device)
        targets = targets.to(device)
        logits = model(items, deltas, hours, users)
        loss = loss_fn(logits, targets)
        bs = int(items.shape[0])
        total_loss += float(loss.item()) * bs
        total_count += bs
    return total_loss / max(1, total_count)


def train_realworld_temporal_encoder_federated(
    temporal: RealWorldTemporalDataset,
    train_user_indices: dict[int, np.ndarray],
    val_indices: np.ndarray,
    cfg: FederatedConfig,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    num_heads: int = 4,
    dropout: float = 0.1,
    logger: logging.Logger | None = None,
    log_every: int = 1,
) -> FederatedTrainResult:
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    dataset = RealWorldTemporalTorchDataset(temporal)
    model = RealWorldTemporalEncoder(
        num_items=temporal.num_items,
        num_users=temporal.num_users,
        window_size=temporal.window_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    round_losses: list[float] = []
    val_losses: list[float] = []
    user_ids = np.asarray(sorted(train_user_indices.keys()), dtype=np.int64)
    if logger is not None:
        logger.info(
            "Real-world temporal federated training started: rounds=%d clients_per_round=%d users=%d",
            cfg.rounds,
            cfg.clients_per_round,
            len(user_ids),
        )

    for round_idx in range(cfg.rounds):
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
            local_state, local_loss = _train_one_local_model(model, dataset, idx, cfg, device)
            local_states.append(local_state)
            local_sizes.append(int(idx.size))
            distances.append(_state_l2_distance(local_state, global_state))
            local_losses.append(local_loss)

        agg_state = _aggregate_states(global_state, local_states, local_sizes, distances, cfg.elastic_tau)
        model.load_state_dict(agg_state)
        train_loss = float(np.mean(local_losses)) if local_losses else float("nan")
        val_loss = evaluate_next_item_loss(model, dataset, val_indices, device=cfg.device)
        round_losses.append(train_loss)
        val_losses.append(val_loss)

        if logger is not None and ((round_idx + 1) % max(1, log_every) == 0 or (round_idx + 1) == cfg.rounds):
            logger.info(
                "Real-world temporal round %d/%d | train_loss=%.6f val_loss=%.6f",
                round_idx + 1,
                cfg.rounds,
                train_loss,
                val_loss,
            )

    model.eval()
    return FederatedTrainResult(model=model, round_losses=round_losses, val_losses=val_losses)
