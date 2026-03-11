# Edge Caching Mobility Simulation (from scratch)

This project does three things:
1. Downloads and extracts the MovieLens 100K dataset (`ml-100k`).
2. Creates User Equipments (UEs) scattered over a `300 x 300` grid (default).
3. Moves UEs slowly with random walk and updates SBS locations every `t_update` seconds using K-means clustering.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 run_simulation.py
```

## Useful flags

```bash
python3 run_simulation.py \
  --n-ues 500 \
  --n-sbs 10 \
  --total-time 600 \
  --dt 1 \
  --t-update 10 \
  --max-speed 1.0 \
  --sbs-max-speed 0.4
```

## Plot trajectories

```bash
python3 run_simulation.py --plot
```

If you only want mobility + plots and do not want to download MovieLens in that run:

```bash
python3 run_simulation.py --skip-dataset-download --plot
```

## Outputs

- `outputs/ue_trajectories.csv`: UE positions at each time-step.
- `outputs/sbs_positions.csv`: SBS positions at each update instant.
- `outputs/trajectories.png`: UE/SBS trajectory visualization (when `--plot` is used).

## Modified CEFMR (Temporal Encoder + GNN Actor-Critic)

This repository now also includes a modified implementation of the paper:
`Cooperative Edge Caching Based on Elastic Federated and MADRL`.

Implemented changes in this codebase:
1. Replaces the AAE popularity module with a temporal encoder that consumes a window of `k` previous requests per user.
2. Keeps chronological request order (timestamp sequence) by constructing `(k previous -> next item)` training samples per user.
3. Keeps greedy top-`Fp` candidate extraction per SBS, then learns cooperative caching.
4. Replaces vanilla actor-critic with a GNN actor-critic that uses SBS adjacency for message passing.

Timestamp usage:
- Yes, MovieLens `u.data` timestamps are used to sort each user's requests in chronological order before building temporal windows.

Main script:

```bash
python3 train_modified_cefmr.py
```

Fast RL-only tuning (reuse pretrained temporal encoder):

```bash
python3 train_modified_cefmr.py \
  --temporal-checkpoint outputs/modified_cefmr/temporal_encoder.pt
```

Useful options:

```bash
python3 train_modified_cefmr.py \
  --window-size 12 \
  --fp 40 \
  --cache-capacity 20 \
  --n-sbs 8 \
  --n-ues 220 \
  --ppo-episodes 60 \
  --federated-rounds 10
```

Local-hit tuning options:

```bash
python3 train_modified_cefmr.py \
  --replacements-per-step 3 \
  --candidate-recent-weight 0.6 \
  --cache-hit-decay 0.98 \
  --beta-neighbor 4.0 \
  --federated-rounds 20 \
  --ppo-episodes 80
```

Logging options:

```bash
python3 train_modified_cefmr.py \
  --log-level INFO \
  --log-every-round 1 \
  --log-every-episode 1
```

Generated outputs (default directory: `outputs/modified_cefmr`):
- `temporal_encoder.pt`
- `gnn_actor_critic.pt`
- `temporal_training.csv`
- `rl_training.csv` (`reward`, `local_hit_rate`, `neighbor_fetch_rate`, `cloud_fetch_rate` per episode)

Plot results:

```bash
python3 plot_modified_cefmr_results.py \
  --run-dirs outputs/modified_cefmr outputs/modified_cefmr_capacity30 \
  --out-dir outputs/plots_modified_cefmr_compare
```

Baseline comparison (paper-style figures):

```bash
python3 compare_baselines_and_plots.py \
  --temporal-checkpoint outputs/modified_cefmr/temporal_encoder.pt \
  --gnn-checkpoint outputs/modified_cefmr_capacity30/gnn_actor_critic.pt \
  --output-dir outputs/baseline_comparison_quick \
  --cache-capacities 10 20 30 \
  --eval-episodes 2 \
  --episode-len 50 \
  --sbs-list 1 4 8
```

Implemented baseline schemes:
- `Random`
- `C-epsilon-greedy`
- `Thompson`
- `BSG-like`
- `EFNRL-like`
- `GNN-ActorCritic` (our main policy)

Request flow (with central server):
1. UE sends a content request to its local SBS.
2. If content is cached locally, it is served locally (best case).
3. Else local SBS checks adjacent SBSs (cooperative fetch).
4. If not found in neighbors, local SBS fetches from central server (cloud/CS).
5. Response returns back to UE through the local SBS.
