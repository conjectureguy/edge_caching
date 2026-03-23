#!/usr/bin/env bash
set -euo pipefail

cd /home/rahul/Desktop/edge_caching_new

PYTHON_BIN="/home/rahul/miniforge3/bin/python"

"$PYTHON_BIN" train_novel_realworld_cache.py \
  --dataset-name ml-1m \
  --output-dir outputs/novel_realworld_paper_today \
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
  --teacher-score-loss-weight 0.5 \
  --label-smoothing 0.03 \
  --decode-diversity-penalty 0.35 \
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
  --log-every-reinforce-episode 2

"$PYTHON_BIN" plot_novel_realworld_results.py \
  --input-dir outputs/novel_realworld_paper_today

"$PYTHON_BIN" plot_episode_epoch_curves.py \
  --input-dir outputs/novel_realworld_paper_today \
  --output-dir outputs/novel_realworld_paper_today/episode_epoch_curves

"$PYTHON_BIN" plot_clustered_latency_study.py \
  --output-dir outputs/clustered_latency_study_paper_today

"$PYTHON_BIN" plot_static_vs_dynamic_bundle.py \
  --output-dir outputs/static_vs_dynamic_bundle_paper_today

"$PYTHON_BIN" plot_temporalgraph_showcase.py \
  --primary-run outputs/novel_realworld_paper_today \
  --output-dir outputs/temporalgraph_showcase_paper_today \
  --exclude-teacher \
  --skip-secondary

"$PYTHON_BIN" plot_final_no_teacher_bundle.py \
  --input-dir outputs/novel_realworld_paper_today \
  --output-dir outputs/final_no_teacher_bundle_paper_today

"$PYTHON_BIN" plot_novel_comparison_bundle.py \
  --run-dir outputs/novel_realworld_paper_today \
  --output-dir outputs/novel_comparison_bundle_paper_today \
  --eval-episodes 1 \
  --episode-len 12 \
  --n-ues 120 \
  --cache-capacities 10 20 \
  --sbs-list 8 16
