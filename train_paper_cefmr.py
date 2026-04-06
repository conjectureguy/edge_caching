from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch

from movie_edge_sim.aae_federated import (
    AAEFederatedConfig,
    build_user_item_matrix,
    train_aae_federated,
)
from movie_edge_sim.data import get_movielens_dataset, load_ratings_auto
from movie_edge_sim.maddpg_cache import MADDPGCachePolicy, MADDPGConfig, train_maddpg_cache_policy
from movie_edge_sim.paper_cefmr_env import PaperCEFMRCooperativeEnv, PaperEnvConfig
from movie_edge_sim.temporal_requests import build_user_histories


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-style CEFMR implementation: elastic FL AAE + MADDPG.")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--dataset-name", type=str, default="ml-1m", choices=["ml-100k", "ml-1m"])
    p.add_argument("--output-dir", type=Path, default=Path("outputs/paper_cefmr"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-every-round", type=int, default=1)
    p.add_argument("--log-every-episode", type=int, default=1)

    p.add_argument("--federated-rounds", type=int, default=12)
    p.add_argument("--clients-per-round", type=int, default=80)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--aae-lr", type=float, default=1e-3)
    p.add_argument("--elastic-tau", type=float, default=2.0)
    p.add_argument("--aae-hidden-dim", type=int, default=256)
    p.add_argument("--aae-latent-dim", type=int, default=64)
    p.add_argument("--train-tfmadrl", action="store_true", help="Also train the non-elastic FL + MADDPG baseline.")

    p.add_argument("--n-sbs", type=int, default=8)
    p.add_argument("--n-ues", type=int, default=220)
    p.add_argument("--cache-capacity", type=int, default=20)
    p.add_argument("--fp", type=int, default=50)
    p.add_argument("--episode-len", type=int, default=120)
    p.add_argument("--grid-size", type=float, default=300.0)
    p.add_argument("--ue-max-speed", type=float, default=1.2)
    p.add_argument("--sbs-max-speed", type=float, default=0.35)
    p.add_argument("--sbs-update-interval", type=float, default=10.0)
    p.add_argument("--neighbor-radius", type=float, default=130.0)
    p.add_argument("--alpha-local", type=float, default=1.0)
    p.add_argument("--beta-neighbor", type=float, default=4.0)
    p.add_argument("--chi-cloud", type=float, default=5.0)
    p.add_argument("--delta-replace", type=float, default=0.1)

    p.add_argument("--maddpg-episodes", type=int, default=120)
    p.add_argument("--maddpg-batch-size", type=int, default=64)
    p.add_argument("--maddpg-replay-size", type=int, default=10000)
    p.add_argument("--maddpg-gamma", type=float, default=0.99)
    p.add_argument("--maddpg-tau", type=float, default=0.01)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--maddpg-hidden-dim", type=int, default=128)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--noise-std", type=float, default=0.1)
    return p.parse_args()


def setup_logging(level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    return logging.getLogger("paper_cefmr")


def save_aae_csv(out_dir: Path, name: str, recon_losses: list[float], adv_losses: list[float]) -> None:
    with (out_dir / f"{name}_aae_training.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "reconstruction_loss", "adversarial_loss"])
        for idx, (recon, adv) in enumerate(zip(recon_losses, adv_losses), start=1):
            writer.writerow([idx, f"{recon:.8f}", f"{adv:.8f}"])


def save_maddpg_csv(out_dir: Path, name: str, hist) -> None:
    with (out_dir / f"{name}_maddpg_training.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "local_hit_rate", "neighbor_fetch_rate", "cloud_fetch_rate"])
        for idx, (reward, local, neigh, cloud) in enumerate(
            zip(
                hist.episode_rewards,
                hist.episode_local_hit_rates,
                hist.episode_neighbor_rates,
                hist.episode_cloud_rates,
            ),
            start=1,
        ):
            writer.writerow([idx, f"{reward:.8f}", f"{local:.8f}", f"{neigh:.8f}", f"{cloud:.8f}"])


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Stage 1/4: loading dataset %s", args.dataset_name)
    dataset_dir = get_movielens_dataset(args.data_root, args.dataset_name)
    ratings = load_ratings_auto(dataset_dir)
    histories = build_user_histories(ratings)
    user_ids, user_matrix = build_user_item_matrix(histories)
    logger.info(
        "Dataset ready | ratings=%d users=%d items=%d",
        len(ratings),
        user_ids.shape[0],
        user_matrix.shape[1],
    )

    env_cfg = PaperEnvConfig(
        n_sbs=args.n_sbs,
        n_ues=args.n_ues,
        cache_capacity=args.cache_capacity,
        fp=args.fp,
        episode_len=args.episode_len,
        grid_size=args.grid_size,
        ue_max_speed=args.ue_max_speed,
        sbs_max_speed=args.sbs_max_speed,
        sbs_update_interval=args.sbs_update_interval,
        neighbor_radius=args.neighbor_radius,
        alpha_local=args.alpha_local,
        beta_neighbor=args.beta_neighbor,
        chi_cloud=args.chi_cloud,
        delta_replace=args.delta_replace,
        seed=args.seed,
    )

    def train_variant(name: str, use_elastic: bool) -> None:
        logger.info("Stage 2/4: training %s AAE with federated learning", name)
        aae_cfg = AAEFederatedConfig(
            rounds=args.federated_rounds,
            clients_per_round=args.clients_per_round,
            local_epochs=args.local_epochs,
            lr=args.aae_lr,
            elastic_tau=args.elastic_tau,
            hidden_dim=args.aae_hidden_dim,
            latent_dim=args.aae_latent_dim,
            seed=args.seed,
            device=args.device,
            use_elastic=use_elastic,
        )
        aae_result = train_aae_federated(
            user_ids=user_ids,
            user_matrix=user_matrix,
            cfg=aae_cfg,
            logger=logger,
            log_every=args.log_every_round,
        )
        save_aae_csv(args.output_dir, name, aae_result.round_recon_losses, aae_result.round_adv_losses)
        torch.save(
            {
                "global_state": aae_result.global_state,
                "user_scores": aae_result.reconstructed_scores,
                "num_items": user_matrix.shape[1],
                "user_ids": user_ids,
                "use_elastic": use_elastic,
                "aae_hidden_dim": args.aae_hidden_dim,
                "aae_latent_dim": args.aae_latent_dim,
            },
            args.output_dir / f"{name}_aae.pt",
        )

        logger.info("Stage 3/4: training %s MADDPG cooperative caching policy", name)
        env = PaperCEFMRCooperativeEnv(env_cfg, histories, aae_result.reconstructed_scores)
        obs = env.reset(seed=args.seed)
        policy = MADDPGCachePolicy(
            n_agents=args.n_sbs,
            state_dim=obs["local_states"].shape[1],
            action_dim=args.fp,
            cfg=MADDPGConfig(
                episodes=args.maddpg_episodes,
                batch_size=args.maddpg_batch_size,
                replay_size=args.maddpg_replay_size,
                gamma=args.maddpg_gamma,
                tau=args.maddpg_tau,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                hidden_dim=args.maddpg_hidden_dim,
                warmup_steps=args.warmup_steps,
                noise_std=args.noise_std,
                device=args.device,
            ),
        )
        hist = train_maddpg_cache_policy(
            env,
            policy,
            policy.cfg,
            seed=args.seed,
            logger=logger,
            log_every=args.log_every_episode,
        )
        save_maddpg_csv(args.output_dir, name, hist)
        torch.save(policy.state_dict(), args.output_dir / f"{name}_maddpg.pt")
        logger.info(
            "%s complete | final_reward=%.6f final_local_hit=%.6f",
            name,
            hist.episode_rewards[-1] if hist.episode_rewards else float("nan"),
            hist.episode_local_hit_rates[-1] if hist.episode_local_hit_rates else float("nan"),
        )

    train_variant("cefmr", use_elastic=True)
    if args.train_tfmadrl:
        train_variant("tfmadrl", use_elastic=False)

    logger.info("Stage 4/4: training finished. Artifacts saved under %s", args.output_dir)


if __name__ == "__main__":
    main()
