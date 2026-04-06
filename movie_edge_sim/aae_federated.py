from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def build_user_item_matrix(
    histories: dict[int, list[int]],
    num_items: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    user_ids = np.asarray(sorted(histories.keys()), dtype=np.int64)
    if num_items is None:
        max_item = 0
        for seq in histories.values():
            if seq:
                max_item = max(max_item, int(max(seq)))
        num_items = max_item

    matrix = np.zeros((user_ids.shape[0], int(num_items)), dtype=np.float32)
    for row, user_id in enumerate(user_ids):
        seq = histories[int(user_id)]
        for item in seq:
            item_idx = int(item) - 1
            if 0 <= item_idx < num_items:
                matrix[row, item_idx] += 1.0
    row_sums = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    matrix = matrix / row_sums
    return user_ids, matrix


class AdversarialAutoEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.encoder = nn.Sequential(
            nn.Linear(self.num_items, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_items),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(self.encode(x)))


@dataclass
class AAEFederatedConfig:
    rounds: int = 12
    clients_per_round: int = 80
    local_epochs: int = 1
    lr: float = 1e-3
    elastic_tau: float = 2.0
    hidden_dim: int = 256
    latent_dim: int = 64
    seed: int = 42
    device: str = "cpu"
    use_elastic: bool = True


@dataclass
class AAEFederatedResult:
    global_model: AdversarialAutoEncoder
    global_state: dict[str, torch.Tensor]
    user_states: dict[int, dict[str, torch.Tensor]]
    reconstructed_scores: dict[int, np.ndarray]
    round_recon_losses: list[float]
    round_adv_losses: list[float]


def _state_l2_distance(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in a:
        total += float((a[key].float() - b[key].float()).pow(2).mean().item())
    return float(np.sqrt(total))


def _blend_states(
    global_state: dict[str, torch.Tensor],
    previous_local: dict[str, torch.Tensor] | None,
    alpha: float,
) -> dict[str, torch.Tensor]:
    if previous_local is None:
        return {k: v.clone() for k, v in global_state.items()}
    blended: dict[str, torch.Tensor] = {}
    for key in global_state:
        blended[key] = alpha * global_state[key] + (1.0 - alpha) * previous_local[key]
    return blended


def _train_local_aae(
    global_model: AdversarialAutoEncoder,
    x: torch.Tensor,
    cfg: AAEFederatedConfig,
    global_state: dict[str, torch.Tensor],
    prev_local_state: dict[str, torch.Tensor] | None,
    prev_distance: float | None,
) -> tuple[dict[str, torch.Tensor], float, float]:
    device = torch.device(cfg.device)
    model = copy.deepcopy(global_model).to(device)
    if cfg.use_elastic and prev_local_state is not None and prev_distance is not None:
        alpha = float(np.clip(prev_distance, 0.0, 1.0))
        init_state = _blend_states(global_state, prev_local_state, alpha)
        model.load_state_dict(init_state)
    else:
        model.load_state_dict(global_state)
    model.train()

    opt_recon = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=cfg.lr,
    )
    opt_disc = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.lr)
    opt_gen = torch.optim.Adam(model.encoder.parameters(), lr=cfg.lr)
    bce_logits = nn.BCEWithLogitsLoss()

    x = x.to(device)
    recon_losses: list[float] = []
    adv_losses: list[float] = []
    for _ in range(cfg.local_epochs):
        # Reconstruction stage.
        z = model.encode(x)
        recon_logits = model.decode_logits(z)
        recon_loss = bce_logits(recon_logits, x)
        opt_recon.zero_grad()
        recon_loss.backward()
        opt_recon.step()
        recon_losses.append(float(recon_loss.item()))

        # Discriminator stage.
        z_fake = model.encode(x).detach()
        z_real = torch.randn_like(z_fake)
        disc_real = model.discriminator(z_real)
        disc_fake = model.discriminator(z_fake)
        disc_loss = bce_logits(disc_real, torch.ones_like(disc_real)) + bce_logits(
            disc_fake, torch.zeros_like(disc_fake)
        )
        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        # Encoder adversarial stage.
        z_fake = model.encode(x)
        gen_logits = model.discriminator(z_fake)
        gen_loss = bce_logits(gen_logits, torch.ones_like(gen_logits))
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        adv_losses.append(float((disc_loss + gen_loss).item()))

    local_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return (
        local_state,
        float(np.mean(recon_losses)) if recon_losses else 0.0,
        float(np.mean(adv_losses)) if adv_losses else 0.0,
    )


@torch.no_grad()
def reconstruct_user_scores(
    model: AdversarialAutoEncoder,
    user_matrix: np.ndarray,
    user_ids: np.ndarray,
    device: str = "cpu",
) -> dict[int, np.ndarray]:
    model = copy.deepcopy(model).to(device)
    model.eval()
    x = torch.as_tensor(user_matrix, dtype=torch.float32, device=device)
    scores = model.reconstruct(x).detach().cpu().numpy()
    return {int(uid): scores[idx].astype(np.float32) for idx, uid in enumerate(user_ids.tolist())}


def train_aae_federated(
    user_ids: np.ndarray,
    user_matrix: np.ndarray,
    cfg: AAEFederatedConfig,
    logger: logging.Logger | None = None,
    log_every: int = 1,
) -> AAEFederatedResult:
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    model = AdversarialAutoEncoder(
        num_items=user_matrix.shape[1],
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
    ).to(device)

    user_states: dict[int, dict[str, torch.Tensor]] = {}
    previous_distances: dict[int, float] = {}
    round_recon_losses: list[float] = []
    round_adv_losses: list[float] = []

    if logger is not None:
        logger.info(
            "AAE elastic federated training started | rounds=%d clients_per_round=%d users=%d elastic=%s",
            cfg.rounds,
            cfg.clients_per_round,
            user_ids.shape[0],
            cfg.use_elastic,
        )

    for round_idx in range(cfg.rounds):
        global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        chosen = rng.choice(user_ids, size=min(cfg.clients_per_round, user_ids.shape[0]), replace=False)

        local_states: list[dict[str, torch.Tensor]] = []
        local_sizes: list[float] = []
        distances: list[float] = []
        recon_losses: list[float] = []
        adv_losses: list[float] = []

        for user_id in chosen.tolist():
            row = int(np.where(user_ids == user_id)[0][0])
            x = torch.as_tensor(user_matrix[row : row + 1], dtype=torch.float32)
            local_state, recon_loss, adv_loss = _train_local_aae(
                model,
                x,
                cfg,
                global_state,
                user_states.get(int(user_id)),
                previous_distances.get(int(user_id)),
            )
            dist = _state_l2_distance(local_state, global_state)
            local_states.append(local_state)
            local_sizes.append(float(max(1.0, user_matrix[row].sum())))
            distances.append(dist)
            recon_losses.append(recon_loss)
            adv_losses.append(adv_loss)
            user_states[int(user_id)] = local_state
            previous_distances[int(user_id)] = dist

        if local_states:
            sizes = np.asarray(local_sizes, dtype=np.float64)
            if cfg.use_elastic:
                dist_arr = np.asarray(distances, dtype=np.float64)
                elastic = np.exp(-dist_arr / max(1e-8, cfg.elastic_tau))
                weights = sizes * elastic
            else:
                weights = sizes
            weights = weights / max(1e-12, weights.sum())
            agg: dict[str, torch.Tensor] = {}
            for key in global_state:
                stacked = torch.stack([state[key].float() for state in local_states], dim=0)
                w = torch.as_tensor(weights, dtype=stacked.dtype).view(-1, *([1] * (stacked.ndim - 1)))
                agg[key] = (stacked * w).sum(dim=0).to(dtype=global_state[key].dtype)
            model.load_state_dict(agg)

        round_recon_losses.append(float(np.mean(recon_losses)) if recon_losses else 0.0)
        round_adv_losses.append(float(np.mean(adv_losses)) if adv_losses else 0.0)
        if logger is not None and ((round_idx + 1) % max(1, log_every) == 0 or (round_idx + 1) == cfg.rounds):
            logger.info(
                "AAE round %d/%d complete | recon_loss=%.6f adv_loss=%.6f clients=%d",
                round_idx + 1,
                cfg.rounds,
                round_recon_losses[-1],
                round_adv_losses[-1],
                len(local_states),
            )

    for user_id in user_ids.tolist():
        if int(user_id) not in user_states:
            user_states[int(user_id)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    reconstructed_scores: dict[int, np.ndarray] = {}
    for row, user_id in enumerate(user_ids.tolist()):
        local_model = AdversarialAutoEncoder(
            num_items=user_matrix.shape[1],
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
        ).to(device)
        local_model.load_state_dict(user_states[int(user_id)])
        local_model.eval()
        x = torch.as_tensor(user_matrix[row : row + 1], dtype=torch.float32, device=device)
        scores = local_model.reconstruct(x).detach().cpu().numpy()[0]
        reconstructed_scores[int(user_id)] = scores.astype(np.float32)

    return AAEFederatedResult(
        global_model=model.eval(),
        global_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        user_states=user_states,
        reconstructed_scores=reconstructed_scores,
        round_recon_losses=round_recon_losses,
        round_adv_losses=round_adv_losses,
    )
