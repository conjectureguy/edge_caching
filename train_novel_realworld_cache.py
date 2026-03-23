from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch

from movie_edge_sim.data import get_movielens_dataset, load_ratings_auto
from movie_edge_sim.novel_graph_policy import (
    ImitationConfig,
    ReinforceConfig,
    TemporalGraphCooperativePolicy,
    evaluate_graph_cache_policy,
    fine_tune_graph_cache_policy_reinforce,
    train_graph_cache_policy_imitation,
)
from movie_edge_sim.novel_realworld_env import NovelRealWorldCachingEnv, RealWorldEnvConfig
from movie_edge_sim.temporal_realworld import (
    FederatedConfig,
    RealWorldTemporalEncoder,
    build_realworld_temporal_dataset,
    build_user_time_histories,
    chronological_train_val_split,
    grouped_indices_by_user,
    train_realworld_temporal_encoder_federated,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Novel real-world cooperative edge caching: timestamp-aware temporal encoder + graph cooperative cache planner."
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--dataset-name", type=str, default="ml-1m", choices=["ml-100k", "ml-1m"])
    p.add_argument("--output-dir", type=Path, default=Path("outputs/novel_realworld_cache"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--temporal-checkpoint", type=Path, default=None)

    p.add_argument("--window-size", type=int, default=12)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--fed-rounds", type=int, default=12)
    p.add_argument("--clients-per-round", type=int, default=80)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--temporal-batch-size", type=int, default=64)
    p.add_argument("--temporal-lr", type=float, default=1e-3)
    p.add_argument("--elastic-tau", type=float, default=2.0)

    p.add_argument("--n-sbs", type=int, default=8)
    p.add_argument("--n-ues", type=int, default=220)
    p.add_argument("--cache-capacity", type=int, default=20)
    p.add_argument("--fp", type=int, default=50)
    p.add_argument("--episode-len", type=int, default=100)
    p.add_argument("--grid-size", type=float, default=300.0)

    p.add_argument("--policy-hidden-dim", type=int, default=128)
    p.add_argument("--imitation-epochs", type=int, default=10)
    p.add_argument("--episodes-per-epoch", type=int, default=6)
    p.add_argument("--policy-lr", type=float, default=2e-4)
    p.add_argument("--teacher-forcing-prob", type=float, default=0.8)
    p.add_argument("--teacher-forcing-final-prob", type=float, default=0.2)
    p.add_argument("--teacher-score-loss-weight", type=float, default=0.35)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--decode-diversity-penalty", type=float, default=0.35)
    p.add_argument("--disable-graph", action="store_true")
    p.add_argument("--disable-temporal", action="store_true")
    p.add_argument("--disable-mobility", action="store_true")
    p.add_argument("--disable-trend", action="store_true")
    p.add_argument("--log-every-imitation-epoch", type=int, default=1)
    p.add_argument("--log-every-imitation-episode", type=int, default=1)
    p.add_argument("--reinforce-epochs", type=int, default=6)
    p.add_argument("--reinforce-episodes-per-epoch", type=int, default=4)
    p.add_argument("--reinforce-lr", type=float, default=1e-4)
    p.add_argument("--reinforce-gamma", type=float, default=0.99)
    p.add_argument("--reinforce-entropy-weight", type=float, default=1e-3)
    p.add_argument("--log-every-reinforce-epoch", type=int, default=1)
    p.add_argument("--log-every-reinforce-episode", type=int, default=1)

    p.add_argument("--eval-episodes", type=int, default=5)
    return p.parse_args()


def setup_logging(level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    return logging.getLogger("novel_realworld_cache")


def save_history(out_dir: Path, temporal_round_losses: list[float], temporal_val_losses: list[float], imitation_hist, reinforce_hist=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "temporal_training.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "train_loss", "val_loss"])
        for i, (tr, vl) in enumerate(zip(temporal_round_losses, temporal_val_losses), start=1):
            writer.writerow([i, f"{tr:.8f}", f"{vl:.8f}"])
    with (out_dir / "policy_imitation.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "reward", "local_hit_rate", "paper_hit_rate"])
        for i, (loss, reward, local_hit, paper_hit) in enumerate(
            zip(imitation_hist.losses, imitation_hist.rewards, imitation_hist.local_hit_rates, imitation_hist.paper_hit_rates),
            start=1,
            ):
            writer.writerow([i, f"{loss:.8f}", f"{reward:.8f}", f"{local_hit:.8f}", f"{paper_hit:.8f}"])
    if reinforce_hist is not None:
        with (out_dir / "policy_reinforce.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "reward", "local_hit_rate", "paper_hit_rate"])
            for i, (loss, reward, local_hit, paper_hit) in enumerate(
                zip(reinforce_hist.losses, reinforce_hist.rewards, reinforce_hist.local_hit_rates, reinforce_hist.paper_hit_rates),
                start=1,
            ):
                writer.writerow([i, f"{loss:.8f}", f"{reward:.8f}", f"{local_hit:.8f}", f"{paper_hit:.8f}"])


def save_eval(out_dir: Path, name: str, rows: list[dict[str, float]]) -> None:
    with (out_dir / f"{name}_eval.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "local_hit_rate", "neighbor_fetch_rate", "cloud_fetch_rate", "paper_hit_rate"])
        for row in rows:
            writer.writerow(
                [
                    row["episode"],
                    f"{row['reward']:.8f}",
                    f"{row['local_hit_rate']:.8f}",
                    f"{row['neighbor_fetch_rate']:.8f}",
                    f"{row['cloud_fetch_rate']:.8f}",
                    f"{row['paper_hit_rate']:.8f}",
                ]
            )


def eval_random(env: NovelRealWorldCachingEnv, episodes: int, seed: int) -> list[dict[str, float]]:
    rows = []
    rng = np.random.default_rng(seed)
    universe = np.arange(1, env.num_items + 1, dtype=np.int64)
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done = False
        reward_sum = 0.0
        local_sum = 0.0
        neighbor_sum = 0.0
        cloud_sum = 0.0
        steps = 0
        while not done:
            action = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
            for b in range(env.cfg.n_sbs):
                action[b] = rng.choice(universe, size=env.cfg.cache_capacity, replace=False)
            _, reward, done, info = env.step_full_cache_items(action)
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


def eval_bsg(env: NovelRealWorldCachingEnv, episodes: int, seed: int) -> list[dict[str, float]]:
    rows = []
    best = np.argsort(env.global_popularity[1:])[-env.cfg.cache_capacity :][::-1] + 1
    action = np.tile(best[None, :], (env.cfg.n_sbs, 1)).astype(np.int64)
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done = False
        reward_sum = 0.0
        local_sum = 0.0
        neighbor_sum = 0.0
        cloud_sum = 0.0
        steps = 0
        while not done:
            _, reward, done, info = env.step_full_cache_items(action)
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


def eval_c_epsilon_greedy(env: NovelRealWorldCachingEnv, episodes: int, seed: int, epsilon: float = 0.1) -> list[dict[str, float]]:
    rows = []
    rng = np.random.default_rng(seed)
    universe = np.arange(1, env.num_items + 1, dtype=np.int64)
    counts = np.zeros((env.cfg.n_sbs, env.num_items + 1), dtype=np.float64)
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done = False
        reward_sum = 0.0
        local_sum = 0.0
        neighbor_sum = 0.0
        cloud_sum = 0.0
        steps = 0
        while not done:
            action = np.zeros((env.cfg.n_sbs, env.cfg.cache_capacity), dtype=np.int64)
            for b in range(env.cfg.n_sbs):
                if rng.random() < epsilon:
                    action[b] = rng.choice(universe, size=env.cfg.cache_capacity, replace=False)
                else:
                    best = np.argsort(counts[b, 1:])[-env.cfg.cache_capacity :][::-1] + 1
                    action[b] = best.astype(np.int64)
            obs_before = env.get_observation()
            association = obs_before["association"]
            _, reward, done, info = env.step_full_cache_items(action)
            for ue in range(env.cfg.n_ues):
                if not env.last_active_mask[ue]:
                    continue
                item = int(env.last_requests[ue])
                if item <= 0:
                    continue
                counts[int(association[ue]), item] += 1.0
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


def eval_teacher(env: NovelRealWorldCachingEnv, episodes: int, seed: int) -> list[dict[str, float]]:
    rows = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done = False
        reward_sum = 0.0
        local_sum = 0.0
        neighbor_sum = 0.0
        cloud_sum = 0.0
        steps = 0
        while not done:
            action = env.cooperative_teacher_action()
            _, reward, done, info = env.step_full_cache_items(action)
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


def summarize(name: str, rows: list[dict[str, float]]) -> str:
    reward = np.mean([r["reward"] for r in rows])
    local = np.mean([r["local_hit_rate"] for r in rows])
    paper = np.mean([r["paper_hit_rate"] for r in rows])
    return f"{name}: reward_mean={reward:.4f} local_hit_mean={local:.4f} paper_hit_mean={paper:.4f}"


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("Stage 1/5: loading %s", args.dataset_name)
    dataset_dir = get_movielens_dataset(args.data_root, args.dataset_name)
    ratings = load_ratings_auto(dataset_dir)
    histories = build_user_time_histories(ratings)
    logger.info("Dataset ready | dataset=%s ratings=%d users=%d", args.dataset_name, len(ratings), len(histories))

    temporal = None
    train_idx = None
    val_idx = None
    train_users = None
    if args.temporal_checkpoint is None:
        logger.info("Stage 2/5: building real-world timestamp-aware temporal dataset")
        temporal = build_realworld_temporal_dataset(histories, window_size=args.window_size)
        train_idx, val_idx = chronological_train_val_split(temporal, val_ratio=args.val_ratio)
        train_users = grouped_indices_by_user(temporal, train_idx)
        logger.info(
            "Temporal dataset ready | samples=%d train=%d val=%d",
            temporal.context_items.shape[0],
            train_idx.shape[0],
            val_idx.shape[0],
        )
    else:
        logger.info("Stage 2/5: skipping temporal dataset build because checkpoint was provided")

    temporal_round_losses: list[float] = []
    temporal_val_losses: list[float] = []
    if args.temporal_checkpoint is None:
        logger.info("Stage 3/5: elastic federated training for real-world temporal encoder")
        assert temporal is not None and train_users is not None and val_idx is not None
        fed_cfg = FederatedConfig(
            rounds=args.fed_rounds,
            clients_per_round=args.clients_per_round,
            local_epochs=args.local_epochs,
            batch_size=args.temporal_batch_size,
            lr=args.temporal_lr,
            elastic_tau=args.elastic_tau,
            seed=args.seed,
            device=args.device,
        )
        result = train_realworld_temporal_encoder_federated(
            temporal=temporal,
            train_user_indices=train_users,
            val_indices=val_idx,
            cfg=fed_cfg,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            logger=logger,
        )
        temporal_model = result.model.eval()
        temporal_round_losses = result.round_losses
        temporal_val_losses = result.val_losses
    else:
        logger.info("Stage 3/5: loading temporal checkpoint %s", args.temporal_checkpoint)
        max_user_id = max(histories.keys())
        max_item_id = max(max(hist.items) for hist in histories.values())
        temporal_model = RealWorldTemporalEncoder(
            num_items=max_item_id,
            num_users=max_user_id,
            window_size=args.window_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
        ).to(args.device)
        state = torch.load(args.temporal_checkpoint, map_location=args.device, weights_only=True)
        temporal_model.load_state_dict(state)
        temporal_model.eval()

    logger.info("Stage 4/5: building novel real-world caching environment")
    env_cfg = RealWorldEnvConfig(
        n_sbs=args.n_sbs,
        n_ues=args.n_ues,
        cache_capacity=args.cache_capacity,
        fp=args.fp,
        window_size=args.window_size,
        episode_len=args.episode_len,
        grid_size=args.grid_size,
        use_temporal_features=not args.disable_temporal,
        use_mobility_features=not args.disable_mobility,
        use_trend_features=not args.disable_trend,
        seed=args.seed,
    )
    env = NovelRealWorldCachingEnv(env_cfg, temporal_model, histories)
    obs = env.reset(seed=args.seed)
    node_dim = int(obs["node_features"].shape[1])
    cand_dim = int(obs["candidate_features"].shape[2])
    logger.info("Environment ready | node_dim=%d candidate_dim=%d", node_dim, cand_dim)

    logger.info("Stage 5/5: training temporal-graph cooperative cache policy")
    policy = TemporalGraphCooperativePolicy(
        node_feat_dim=node_dim,
        candidate_feat_dim=cand_dim,
        hidden_dim=args.policy_hidden_dim,
        fp=args.fp,
        use_graph=not args.disable_graph,
    )
    imit_cfg = ImitationConfig(
        epochs=args.imitation_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        lr=args.policy_lr,
        device=args.device,
        teacher_forcing_prob=args.teacher_forcing_prob,
        teacher_forcing_final_prob=args.teacher_forcing_final_prob,
        teacher_score_loss_weight=args.teacher_score_loss_weight,
        label_smoothing=args.label_smoothing,
        decode_diversity_penalty=args.decode_diversity_penalty,
    )
    imitation_hist = train_graph_cache_policy_imitation(
        env,
        policy,
        imit_cfg,
        seed=args.seed,
        logger=logger,
        log_every_epoch=args.log_every_imitation_epoch,
        log_every_episode=args.log_every_imitation_episode,
    )
    reinforce_hist = None
    if args.reinforce_epochs > 0:
        logger.info("Stage 5b/5: reward fine-tuning temporal-graph policy")
        reinforce_cfg = ReinforceConfig(
            epochs=args.reinforce_epochs,
            episodes_per_epoch=args.reinforce_episodes_per_epoch,
            lr=args.reinforce_lr,
            gamma=args.reinforce_gamma,
            entropy_weight=args.reinforce_entropy_weight,
            device=args.device,
            decode_diversity_penalty=args.decode_diversity_penalty,
        )
        reinforce_hist = fine_tune_graph_cache_policy_reinforce(
            env,
            policy,
            reinforce_cfg,
            seed=args.seed,
            logger=logger,
            log_every_epoch=args.log_every_reinforce_epoch,
            log_every_episode=args.log_every_reinforce_episode,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(temporal_model.state_dict(), args.output_dir / "realworld_temporal_encoder.pt")
    torch.save(policy.state_dict(), args.output_dir / "temporal_graph_policy.pt")
    save_history(args.output_dir, temporal_round_losses, temporal_val_losses, imitation_hist, reinforce_hist)

    logger.info("Evaluating random baseline")
    random_rows = eval_random(env, args.eval_episodes, seed=args.seed + 1000)
    logger.info("Evaluating BSG-like baseline")
    bsg_rows = eval_bsg(env, args.eval_episodes, seed=args.seed + 1500)
    logger.info("Evaluating C-epsilon-greedy baseline")
    c_eps_rows = eval_c_epsilon_greedy(env, args.eval_episodes, seed=args.seed + 1750)
    logger.info("Evaluating cooperative teacher")
    teacher_rows = eval_teacher(env, args.eval_episodes, seed=args.seed + 2000)
    logger.info("Evaluating learned graph policy")
    learned_rows = evaluate_graph_cache_policy(
        env,
        policy,
        args.eval_episodes,
        seed=args.seed + 3000,
        device=args.device,
        decode_diversity_penalty=args.decode_diversity_penalty,
    )
    save_eval(args.output_dir, "random", random_rows)
    save_eval(args.output_dir, "bsg_like", bsg_rows)
    save_eval(args.output_dir, "c_epsilon_greedy", c_eps_rows)
    save_eval(args.output_dir, "teacher", teacher_rows)
    save_eval(args.output_dir, "temporal_graph", learned_rows)

    summary_lines = [
        f"Ablations: graph={'off' if args.disable_graph else 'on'} temporal={'off' if args.disable_temporal else 'on'} mobility={'off' if args.disable_mobility else 'on'} trend={'off' if args.disable_trend else 'on'} teacher_score_loss_weight={args.teacher_score_loss_weight:.3f} decode_diversity_penalty={args.decode_diversity_penalty:.3f}",
        f"Reinforce: epochs={args.reinforce_epochs} episodes_per_epoch={args.reinforce_episodes_per_epoch} lr={args.reinforce_lr:.6f} gamma={args.reinforce_gamma:.3f} entropy_weight={args.reinforce_entropy_weight:.6f}",
        summarize("Random", random_rows),
        summarize("BSG-like", bsg_rows),
        summarize("C-epsilon-greedy", c_eps_rows),
        summarize("Teacher", teacher_rows),
        summarize("TemporalGraph", learned_rows),
    ]
    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    for line in summary_lines:
        logger.info(line)
        print(line)
    print(f"Dataset: {args.dataset_name}")
    print(f"Saved outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
