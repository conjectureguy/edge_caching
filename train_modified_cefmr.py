from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch

from movie_edge_sim.cooperative_env import CooperativeCachingEnv, EnvConfig
from movie_edge_sim.data import download_movielens_100k, load_ratings
from movie_edge_sim.gnn_actor_critic import GNNActorCritic, PPOConfig, train_gnn_ppo
from movie_edge_sim.temporal_federated import FederatedConfig, TemporalSpikeEncoder, train_temporal_encoder_federated
from movie_edge_sim.temporal_requests import (
    build_temporal_dataset,
    build_user_histories,
    grouped_indices_by_user,
    train_val_split,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Modified CEFMR implementation: temporal-window encoder (AAE replacement) + "
            "GNN actor-critic cooperative caching."
        )
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/modified_cefmr"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-every-round", type=int, default=1)
    p.add_argument("--log-every-episode", type=int, default=1)
    p.add_argument(
        "--temporal-checkpoint",
        type=Path,
        default=None,
        help="Optional temporal checkpoint path. If provided, skip temporal federated training.",
    )

    # Temporal encoder
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--federated-rounds", type=int, default=20)
    p.add_argument("--clients-per-round", type=int, default=80)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--temporal-batch-size", type=int, default=64)
    p.add_argument("--temporal-lr", type=float, default=1e-3)
    p.add_argument("--elastic-tau", type=float, default=2.0)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=96)

    # Environment
    p.add_argument("--n-sbs", type=int, default=8)
    p.add_argument("--n-ues", type=int, default=220)
    p.add_argument("--cache-capacity", type=int, default=20)
    p.add_argument("--fp", type=int, default=40)
    p.add_argument("--episode-len", type=int, default=150)
    p.add_argument("--grid-size", type=float, default=300.0)
    p.add_argument("--ue-max-speed", type=float, default=1.2)
    p.add_argument("--sbs-max-speed", type=float, default=0.35)
    p.add_argument("--sbs-update-interval", type=float, default=10.0)
    p.add_argument("--neighbor-radius", type=float, default=130.0)
    p.add_argument("--replacements-per-step", type=int, default=3)
    p.add_argument("--candidate-recent-weight", type=float, default=0.6)
    p.add_argument("--cache-hit-decay", type=float, default=0.98)
    p.add_argument("--alpha-local", type=float, default=1.0)
    p.add_argument("--beta-neighbor", type=float, default=4.0)
    p.add_argument("--chi-cloud", type=float, default=5.0)
    p.add_argument("--delta-replace", type=float, default=0.1)

    # GNN PPO
    p.add_argument("--ppo-episodes", type=int, default=80)
    p.add_argument("--ppo-horizon", type=int, default=128)
    p.add_argument("--ppo-lr", type=float, default=3e-4)
    p.add_argument("--ppo-update-epochs", type=int, default=4)
    p.add_argument("--ppo-hidden-dim", type=int, default=128)
    return p.parse_args()


def setup_logging(level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("modified_cefmr")


def save_training_curves(
    out_dir: Path,
    temporal_round_losses: list[float],
    temporal_val_losses: list[float],
    episode_rewards: list[float],
    episode_hit_rates: list[float],
    episode_neighbor_rates: list[float],
    episode_cloud_rates: list[float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    temporal_csv = out_dir / "temporal_training.csv"
    rl_csv = out_dir / "rl_training.csv"

    with temporal_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "train_loss", "val_loss"])
        for i, (tr, vl) in enumerate(zip(temporal_round_losses, temporal_val_losses), start=1):
            writer.writerow([i, f"{tr:.8f}", f"{vl:.8f}"])

    with rl_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "local_hit_rate", "neighbor_fetch_rate", "cloud_fetch_rate"])
        for i, (rew, hit, nbr, cld) in enumerate(
            zip(episode_rewards, episode_hit_rates, episode_neighbor_rates, episode_cloud_rates), start=1
        ):
            writer.writerow([i, f"{rew:.8f}", f"{hit:.8f}", f"{nbr:.8f}", f"{cld:.8f}"])


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    logger.info("Pipeline started.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info("Seeds initialized with seed=%d", args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        device = "cpu"
    logger.info("Using device=%s", device)

    logger.info("Stage 1/5: loading MovieLens-100k dataset")
    dataset_dir = download_movielens_100k(args.data_root)
    ratings = load_ratings(dataset_dir)
    histories = build_user_histories(ratings)
    logger.info("Dataset ready | ratings=%d users=%d", len(ratings), len(histories))

    logger.info("Stage 2/5: building temporal window dataset")
    temporal = build_temporal_dataset(histories, window_size=args.window_size)
    train_idx, val_idx = train_val_split(temporal, val_ratio=args.val_ratio, seed=args.seed)
    train_users = grouped_indices_by_user(temporal, train_idx)
    logger.info(
        "Temporal dataset ready | samples=%d window_size=%d train_samples=%d val_samples=%d users_with_train_data=%d",
        temporal.contexts.shape[0],
        args.window_size,
        train_idx.shape[0],
        val_idx.shape[0],
        len(train_users),
    )

    temporal_round_losses: list[float] = []
    temporal_val_losses: list[float] = []
    if args.temporal_checkpoint is None:
        logger.info("Stage 3/5: training temporal encoder with elastic federated learning")
        fed_cfg = FederatedConfig(
            rounds=args.federated_rounds,
            clients_per_round=args.clients_per_round,
            local_epochs=args.local_epochs,
            batch_size=args.temporal_batch_size,
            lr=args.temporal_lr,
            elastic_tau=args.elastic_tau,
            seed=args.seed,
            device=device,
        )
        fed_result = train_temporal_encoder_federated(
            temporal=temporal,
            train_user_indices=train_users,
            val_indices=val_idx,
            cfg=fed_cfg,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            logger=logger,
            log_every=args.log_every_round,
        )
        temporal_model = fed_result.model.eval()
        temporal_round_losses = fed_result.round_losses
        temporal_val_losses = fed_result.val_losses
        logger.info(
            "Temporal training complete | final_train_loss=%.6f final_val_loss=%.6f",
            fed_result.round_losses[-1] if fed_result.round_losses else float("nan"),
            fed_result.val_losses[-1] if fed_result.val_losses else float("nan"),
        )
    else:
        logger.info("Stage 3/5: loading temporal encoder checkpoint: %s", args.temporal_checkpoint)
        temporal_model = TemporalSpikeEncoder(
            num_items=temporal.num_items,
            window_size=temporal.window_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        state = torch.load(args.temporal_checkpoint, map_location=device, weights_only=True)
        temporal_model.load_state_dict(state)
        temporal_model.eval()
        logger.info("Temporal checkpoint loaded successfully; skipping temporal training.")

    logger.info("Stage 4/5: initializing cooperative environment and GNN actor-critic")
    env_cfg = EnvConfig(
        n_sbs=args.n_sbs,
        n_ues=args.n_ues,
        cache_capacity=args.cache_capacity,
        fp=args.fp,
        window_size=args.window_size,
        episode_len=args.episode_len,
        grid_size=args.grid_size,
        ue_max_speed=args.ue_max_speed,
        sbs_max_speed=args.sbs_max_speed,
        sbs_update_interval=args.sbs_update_interval,
        neighbor_radius=args.neighbor_radius,
        replacements_per_step=args.replacements_per_step,
        candidate_recent_weight=args.candidate_recent_weight,
        cache_hit_decay=args.cache_hit_decay,
        alpha_local=args.alpha_local,
        beta_neighbor=args.beta_neighbor,
        chi_cloud=args.chi_cloud,
        delta_replace=args.delta_replace,
        seed=args.seed,
    )
    env = CooperativeCachingEnv(env_cfg, temporal_model, histories)
    init_obs = env.reset(seed=args.seed)
    node_feat_dim = int(init_obs["node_features"].shape[1])
    logger.info(
        "Environment ready | n_sbs=%d n_ues=%d fp=%d cache_capacity=%d node_feat_dim=%d",
        args.n_sbs,
        args.n_ues,
        args.fp,
        args.cache_capacity,
        node_feat_dim,
    )

    actor_critic = GNNActorCritic(
        node_feat_dim=node_feat_dim,
        hidden_dim=args.ppo_hidden_dim,
        fp=args.fp,
    )
    ppo_cfg = PPOConfig(
        episodes=args.ppo_episodes,
        horizon=args.ppo_horizon,
        lr=args.ppo_lr,
        update_epochs=args.ppo_update_epochs,
        device=device,
    )
    logger.info("Stage 5/5: training GNN PPO policy")
    hist = train_gnn_ppo(
        env,
        actor_critic,
        ppo_cfg,
        seed=args.seed,
        logger=logger,
        log_every=args.log_every_episode,
    )
    logger.info(
        "RL training complete | final_reward=%.6f final_local_hit_rate=%.6f",
        hist.episode_rewards[-1] if hist.episode_rewards else float("nan"),
        hist.episode_hit_rates[-1] if hist.episode_hit_rates else float("nan"),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporal_ckpt = args.output_dir / "temporal_encoder.pt"
    gnn_ckpt = args.output_dir / "gnn_actor_critic.pt"
    torch.save(temporal_model.state_dict(), temporal_ckpt)
    torch.save(actor_critic.state_dict(), gnn_ckpt)

    save_training_curves(
        args.output_dir,
        temporal_round_losses,
        temporal_val_losses,
        hist.episode_rewards,
        hist.episode_hit_rates,
        hist.episode_neighbor_rates,
        hist.episode_cloud_rates,
    )
    logger.info("Artifacts saved under %s", args.output_dir)

    print(f"MovieLens dir: {dataset_dir}")
    print(f"Ratings loaded: {len(ratings)}")
    print(f"Temporal samples: {temporal.contexts.shape[0]}")
    if temporal_val_losses:
        print(f"Temporal val loss (last): {temporal_val_losses[-1]:.6f}")
    else:
        print("Temporal val loss (last): N/A (loaded from checkpoint)")
    print(f"RL reward (last episode): {hist.episode_rewards[-1]:.6f}")
    print(f"RL local hit-rate (last episode): {hist.episode_hit_rates[-1]:.6f}")
    print(f"Saved temporal checkpoint: {temporal_ckpt}")
    print(f"Saved GNN checkpoint: {gnn_ckpt}")
    print(f"Saved metrics CSVs under: {args.output_dir}")


if __name__ == "__main__":
    main()
