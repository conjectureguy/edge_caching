#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/kaggle/working/edge_caching"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(which python3)}"
# PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
DEVICE="${DEVICE:-cuda}"
DATASET_NAME="${DATASET_NAME:-ml-1m}"
ROOT_BASE="${ROOT_BASE:-outputs/research_benchmark_runs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
FAIL_IF_OURS_NOT_BEST="${FAIL_IF_OURS_NOT_BEST:-0}"

ROOT="${ROOT_BASE}/temporalgraph_${RUN_TAG}"
RUN_DIR="${ROOT}/novel_realworld_main"
EPISODE_EPOCH_DIR="${ROOT}/episode_epoch_curves"
LATENCY_CLUSTERED_DIR="${ROOT}/clustered_latency_study"
STATIC_DYNAMIC_DIR="${ROOT}/static_vs_dynamic_bundle"
SHOWCASE_DIR="${ROOT}/temporalgraph_showcase"
FINAL_NO_TEACHER_DIR="${ROOT}/final_no_teacher_bundle"
COMPARISON_BUNDLE_DIR="${ROOT}/novel_comparison_bundle"
RELATED_WORK_DIR="${ROOT}/related_work_compare"
PAPER_READY_DIR="${ROOT}/paper_ready_bundle"
MANIFEST_PATH="${ROOT}/manifest.txt"
REPORT_PATH="${ROOT}/benchmark_report.txt"
LATEST_LINK="${ROOT_BASE}/latest"

mkdir -p "${ROOT_BASE}" "${ROOT}"
ln -sfn "$(basename "${ROOT}")" "${LATEST_LINK}"

echo "============================================================"
echo "TemporalGraph full benchmark run"
echo "Repo root   : ${REPO_ROOT}"
echo "Python      : ${PYTHON_BIN}"
echo "Device      : ${DEVICE}"
echo "Dataset     : ${DATASET_NAME}"
echo "Output root : ${ROOT}"
echo "============================================================"

echo "[0/9] Validating entrypoints"
"${PYTHON_BIN}" -m py_compile \
  train_novel_realworld_cache.py \
  compare_related_work_papers.py \
  plot_novel_realworld_results.py \
  plot_episode_epoch_curves.py \
  plot_clustered_latency_study.py \
  plot_static_vs_dynamic_bundle.py \
  plot_temporalgraph_showcase.py \
  plot_final_no_teacher_bundle.py \
  plot_novel_comparison_bundle.py \
  generate_plots_apr11.py

echo "[1/9] Training Temporal Encoder + Elastic FL + GNN policy"
"${PYTHON_BIN}" train_novel_realworld_cache.py \
  --dataset-name "${DATASET_NAME}" \
  --output-dir "${RUN_DIR}" \
  --device "${DEVICE}" \
  --window-size 12 \
  --embed-dim 64 \
  --hidden-dim 128 \
  --num-heads 4 \
  --fed-rounds 18 \
  --clients-per-round 120 \
  --local-epochs 1 \
  --temporal-batch-size 128 \
  --temporal-lr 0.0007 \
  --temporal-weight-decay 0.00001 \
  --elastic-tau 2.0 \
  --temporal-mask-prob 0.15 \
  --temporal-mlm-weight 0.20 \
  --temporal-contrastive-weight 0.05 \
  --temporal-contrastive-temperature 0.25 \
  --n-sbs 8 \
  --n-ues 220 \
  --cache-capacity 20 \
  --fp 50 \
  --episode-len 120 \
  --grid-size 300 \
  --policy-hidden-dim 160 \
  --imitation-epochs 30 \
  --episodes-per-epoch 10 \
  --policy-lr 0.0002 \
  --teacher-forcing-prob 0.90 \
  --teacher-forcing-final-prob 0.05 \
  --teacher-score-loss-weight 0.60 \
  --teacher-rank-loss-weight 0.25 \
  --label-smoothing 0.03 \
  --decode-diversity-penalty 0.18 \
  --teacher-guidance-weight 0.70 \
  --placement-interval 3 \
  --teacher-base-weight 0.42 \
  --teacher-attention-weight 0.24 \
  --teacher-mobility-weight 0.20 \
  --teacher-ddpg-weight 0.14 \
  --freshness-tau-hours 6.0 \
  --semantic-score-weight 0.10 \
  --semantic-future-weight 0.05 \
  --freshness-score-weight 0.08 \
  --checkpoint-eval-episodes 6 \
  --reinforce-epochs 0 \
  --eval-episodes 10 \
  --seed 42 \
  --log-level INFO \
  --log-every-imitation-epoch 1 \
  --log-every-imitation-episode 2 \
  2>&1 | tee "${ROOT}/train.log"

echo "[2/9] Running related-work benchmark"
"${PYTHON_BIN}" compare_related_work_papers.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${RELATED_WORK_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --device "${DEVICE}" \
  --eval-episodes 10 \
  --episode-len 120 \
  --n-sbs 8 \
  --n-ues 220 \
  --cache-capacity 20 \
  --fp 50 \
  --grid-size 300 \
  --policy-hidden-dim 160 \
  --decode-diversity-penalty 0.18 \
  --teacher-guidance-weight 0.70 \
  --placement-interval 3 \
  --log-level INFO \
  --log-every-episode 1 \
  2>&1 | tee "${ROOT}/related_work.log"

echo "[3/9] Regenerating main-run summary plots for the final model comparisons"
"${PYTHON_BIN}" plot_novel_realworld_results.py \
  --input-dir "${RUN_DIR}" \
  --output-dir "${RUN_DIR}" \
  --related-work-dir "${RELATED_WORK_DIR}"

echo "[4/9] Episode/epoch curves"
"${PYTHON_BIN}" plot_episode_epoch_curves.py \
  --input-dir "${RUN_DIR}" \
  --output-dir "${EPISODE_EPOCH_DIR}"

echo "[5/9] Static-dynamic and latency bundles"
"${PYTHON_BIN}" plot_clustered_latency_study.py \
  --output-dir "${LATENCY_CLUSTERED_DIR}"

"${PYTHON_BIN}" plot_static_vs_dynamic_bundle.py \
  --output-dir "${STATIC_DYNAMIC_DIR}"

echo "[6/9] Showcase and no-teacher bundles"
"${PYTHON_BIN}" plot_temporalgraph_showcase.py \
  --primary-run "${RUN_DIR}" \
  --output-dir "${SHOWCASE_DIR}" \
  --related-work-dir "${RELATED_WORK_DIR}" \
  --exclude-teacher \
  --skip-secondary

"${PYTHON_BIN}" plot_final_no_teacher_bundle.py \
  --input-dir "${RUN_DIR}" \
  --output-dir "${FINAL_NO_TEACHER_DIR}" \
  --related-work-dir "${RELATED_WORK_DIR}"

echo "[7/9] Extended comparison bundle"
"${PYTHON_BIN}" plot_novel_comparison_bundle.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${COMPARISON_BUNDLE_DIR}" \
  --device "${DEVICE}" \
  --eval-episodes 10 \
  --episode-len 60 \
  --n-ues 220 \
  --cache-capacities 10 20 30 \
  --sbs-list 8 12 16

echo "[8/9] Paper-ready PDF bundle"
"${PYTHON_BIN}" generate_plots_apr11.py \
  --data-dir "${ROOT}" \
  --data-dir-v3 "${ROOT}" \
  --run-dir "${RUN_DIR}" \
  --output-dir "${PAPER_READY_DIR}" \
  --format pdf \
  --python-bin "${PYTHON_BIN}"

echo "[9/9] Writing benchmark report and manifest"
"${PYTHON_BIN}" - "${RELATED_WORK_DIR}/summary.csv" "${REPORT_PATH}" "${FAIL_IF_OURS_NOT_BEST}" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
report_path = Path(sys.argv[2])
fail_if_not_best = sys.argv[3] == "1"

with summary_csv.open() as f:
    rows = list(csv.DictReader(f))

ours = None
for row in rows:
    if row["scheme"] in {"Our-TemporalGraph", "TemporalGraph"}:
        ours = row
        break

if ours is None:
    raise SystemExit("TemporalGraph row missing from related-work summary.")

reward_best = max(rows, key=lambda r: float(r["reward_mean"]))
local_best = max(rows, key=lambda r: float(r["local_hit_mean"]))
paper_best = max(rows, key=lambda r: float(r["paper_hit_mean"]))
cloud_best = min(rows, key=lambda r: float(r["cloud_fetch_mean"]))

lines = [
    f"Related-work summary file: {summary_csv}",
    f"Our reward_mean={float(ours['reward_mean']):.6f} | best={reward_best['scheme']}",
    f"Our local_hit_mean={float(ours['local_hit_mean']):.6f} | best={local_best['scheme']}",
    f"Our paper_hit_mean={float(ours['paper_hit_mean']):.6f} | best={paper_best['scheme']}",
    f"Our cloud_fetch_mean={float(ours['cloud_fetch_mean']):.6f} | best={cloud_best['scheme']}",
]

all_best = (
    reward_best["scheme"] in {"Our-TemporalGraph", "TemporalGraph"}
    and local_best["scheme"] in {"Our-TemporalGraph", "TemporalGraph"}
    and paper_best["scheme"] in {"Our-TemporalGraph", "TemporalGraph"}
    and cloud_best["scheme"] in {"Our-TemporalGraph", "TemporalGraph"}
)
lines.append(f"Ours best on all four summary metrics: {'yes' if all_best else 'no'}")

report_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

if fail_if_not_best and not all_best:
    raise SystemExit(1)
PY

find "${ROOT}" -maxdepth 2 -type f | sort > "${MANIFEST_PATH}"

echo "============================================================"
echo "Run complete"
echo "Root bundle      : ${ROOT}"
echo "Main run         : ${RUN_DIR}"
echo "Related work     : ${RELATED_WORK_DIR}"
echo "PNG bundle       : ${ROOT}"
echo "PDF bundle       : ${PAPER_READY_DIR}"
echo "Manifest         : ${MANIFEST_PATH}"
echo "Benchmark report : ${REPORT_PATH}"
echo "Latest symlink   : ${LATEST_LINK}"
echo "============================================================"
