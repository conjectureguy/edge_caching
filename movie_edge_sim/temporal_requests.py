from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class TemporalDataset:
    """Temporal windows used to train a sequential preference model."""

    contexts: np.ndarray  # (N, k)
    targets: np.ndarray  # (N,)
    user_ids: np.ndarray  # (N,)
    num_users: int
    num_items: int
    window_size: int


def build_user_histories(ratings: list[dict[str, int]]) -> dict[int, list[int]]:
    """Build chronological item-request history per user from MovieLens ratings."""
    by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in ratings:
        by_user[row["user_id"]].append((row["timestamp"], row["item_id"]))

    histories: dict[int, list[int]] = {}
    for user_id, rows in by_user.items():
        rows.sort(key=lambda x: x[0])
        histories[user_id] = [item_id for _, item_id in rows]
    return histories


def build_temporal_dataset(
    histories: dict[int, list[int]], window_size: int, min_history: int | None = None
) -> TemporalDataset:
    """Create (k-history -> next item) samples over all users."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    minimum = min_history if min_history is not None else window_size + 1

    contexts = []
    targets = []
    users = []

    max_item_id = 0
    max_user_id = 0
    for user_id, seq in histories.items():
        if len(seq) < minimum:
            continue
        max_user_id = max(max_user_id, user_id)
        for item in seq:
            max_item_id = max(max_item_id, item)
        for t in range(window_size, len(seq)):
            ctx = seq[t - window_size : t]
            target = seq[t]
            contexts.append(ctx)
            targets.append(target)
            users.append(user_id)

    if not contexts:
        raise ValueError(
            "No temporal samples were generated. "
            "Lower window_size or min_history, or provide a larger request history."
        )

    return TemporalDataset(
        contexts=np.asarray(contexts, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
        user_ids=np.asarray(users, dtype=np.int64),
        num_users=max_user_id,
        num_items=max_item_id,
        window_size=window_size,
    )


def train_val_split(dataset: TemporalDataset, val_ratio: float = 0.1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return train and validation index arrays."""
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    n = dataset.contexts.shape[0]
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return train_idx, val_idx


def grouped_indices_by_user(dataset: TemporalDataset, indices: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """Build per-user index lists for federated local training."""
    if indices is None:
        indices = np.arange(dataset.contexts.shape[0], dtype=np.int64)
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx in indices:
        grouped[int(dataset.user_ids[idx])].append(int(idx))
    return {uid: np.asarray(idxs, dtype=np.int64) for uid, idxs in grouped.items()}

