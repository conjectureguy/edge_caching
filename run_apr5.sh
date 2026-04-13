#!/usr/bin/env bash
set -euo pipefail

cd /home/rahul/Desktop/edge_caching_new

PYTHON_BIN="/home/rahul/miniforge3/bin/python"

ROOT="outputs/plots_apr6_v4"
RUN_DIR="$ROOT/novel_realworld_main"
EPISODE_EPOCH_DIR="$ROOT/episode_epoch_curves"
LATENCY_CLUSTERED_DIR="$ROOT/clustered_latency_study"
STATIC_DYNAMIC_DIR="$ROOT/static_vs_dynamic_bundle"
SHOWCASE_DIR="$ROOT/temporalgraph_showcase"
FINAL_NO_TEACHER_DIR="$ROOT/final_no_teacher_bundle"
COMPARISON_BUNDLE_DIR="$ROOT/novel_comparison_bundle"
RELATED_WORK_DIR="$ROOT/related_work_compare"

mkdir -p "$ROOT"

"$PYTHON_BIN" train_novel_realworld_cache.py \
  --dataset-name ml-1m \
  --output-dir "$RUN_DIR" \
  --device cpu \
  --window-size 12 \
  --fed-rounds 15 \
  --clients-per-round 100 \
  --local-epochs 1 \
  --temporal-batch-size 128 \
  --temporal-lr 0.0008 \
  --elastic-tau 2.0 \
  --n-sbs 8 \
  --n-ues 220 \
  --cache-capacity 20 \
  --fp 50 \
  --episode-len 120 \
  --grid-size 300 \
  --policy-hidden-dim 160 \
  --imitation-epochs 25 \
  --episodes-per-epoch 10 \
  --policy-lr 0.0002 \
  --teacher-forcing-prob 0.9 \
  --teacher-forcing-final-prob 0.05 \
  --teacher-score-loss-weight 0.6 \
  --label-smoothing 0.03 \
  --decode-diversity-penalty 0.25 \
  --teacher-base-weight 0.40 \
  --teacher-attention-weight 0.25 \
  --teacher-mobility-weight 0.20 \
  --teacher-ddpg-weight 0.15 \
  --freshness-tau-hours 6.0 \
  --semantic-score-weight 0.10 \
  --semantic-future-weight 0.05 \
  --freshness-score-weight 0.08 \
  --reinforce-epochs 15 \
  --reinforce-episodes-per-epoch 8 \
  --reinforce-lr 0.00005 \
  --reinforce-gamma 0.99 \
  --reinforce-entropy-weight 0.0005 \
  --eval-episodes 10 \
  --seed 42 \
  --log-level INFO \
  --log-every-imitation-epoch 1 \
  --log-every-imitation-episode 2 \
  --log-every-reinforce-epoch 1 \
  --log-every-reinforce-episode 2 \
  2>&1 | tee "$ROOT/train.log"

"$PYTHON_BIN" plot_novel_realworld_results.py \
  --input-dir "$RUN_DIR"

"$PYTHON_BIN" plot_episode_epoch_curves.py \
  --input-dir "$RUN_DIR" \
  --output-dir "$EPISODE_EPOCH_DIR"

"$PYTHON_BIN" plot_clustered_latency_study.py \
  --output-dir "$LATENCY_CLUSTERED_DIR"

"$PYTHON_BIN" plot_static_vs_dynamic_bundle.py \
  --output-dir "$STATIC_DYNAMIC_DIR"

"$PYTHON_BIN" plot_temporalgraph_showcase.py \
  --primary-run "$RUN_DIR" \
  --output-dir "$SHOWCASE_DIR" \
  --exclude-teacher \
  --skip-secondary

"$PYTHON_BIN" plot_final_no_teacher_bundle.py \
  --input-dir "$RUN_DIR" \
  --output-dir "$FINAL_NO_TEACHER_DIR"

"$PYTHON_BIN" plot_novel_comparison_bundle.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$COMPARISON_BUNDLE_DIR" \
  --eval-episodes 3 \
  --episode-len 30 \
  --n-ues 220 \
  --cache-capacities 10 20 30 \
  --sbs-list 8 12 16

"$PYTHON_BIN" compare_related_work_papers.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$RELATED_WORK_DIR" \
  --dataset-name ml-1m \
  --device cpu \
  --eval-episodes 3 \
  --episode-len 120 \
  --n-sbs 8 \
  --n-ues 220 \
  --cache-capacity 20 \
  --fp 50 \
  --grid-size 300 \
  --policy-hidden-dim 160 \
  --log-level INFO \
  --log-every-episode 1 \
  --exclude-schemes MAAFDRL \
  2>&1 | tee "$ROOT/related_work.log"
