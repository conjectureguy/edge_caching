#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/rahul/Desktop/edge_caching"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/rahul/miniforge3/bin/python}"
DEVICE="${DEVICE:-cpu}"
DATASET_NAME="${DATASET_NAME:-ml-100k}"
ROOT_BASE="${ROOT_BASE:-outputs/fast_100k_runs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
FAIL_IF_OURS_NOT_BEST="${FAIL_IF_OURS_NOT_BEST:-0}"

ROOT="${ROOT_BASE}/temporalgraph_${RUN_TAG}"
RUN_DIR="${ROOT}/novel_realworld_main"
RELATED_WORK_DIR="${ROOT}/related_work_compare"
REPORT_PATH="${ROOT}/benchmark_report.txt"
LATEST_LINK="${ROOT_BASE}/latest"

mkdir -p "${ROOT_BASE}" "${ROOT}"
ln -sfn "$(basename "${ROOT}")" "${LATEST_LINK}"

echo "============================================================"
echo "TemporalGraph FAST 100k Prototype run"
echo "Repo root   : ${REPO_ROOT}"
echo "Python      : ${PYTHON_BIN}"
echo "Device      : ${DEVICE}"
echo "Dataset     : ${DATASET_NAME}"
echo "Output root : ${ROOT}"
echo "============================================================"

echo "[1/4] Training Temporal Encoder + Elastic FL + GNN policy"
"${PYTHON_BIN}" train_novel_realworld_cache.py \
  --dataset-name "${DATASET_NAME}" \
  --output-dir "${RUN_DIR}" \
  --device "${DEVICE}" \
  --window-size 8 \
  --embed-dim 32 \
  --hidden-dim 64 \
  --num-heads 2 \
  --fed-rounds 3 \
  --clients-per-round 40 \
  --local-epochs 1 \
  --temporal-batch-size 32 \
  --n-sbs 8 \
  --n-ues 60 \
  --cache-capacity 10 \
  --fp 30 \
  --episode-len 40 \
  --grid-size 200 \
  --policy-hidden-dim 64 \
  --imitation-epochs 8 \
  --episodes-per-epoch 4 \
  --policy-lr 0.0004 \
  --teacher-forcing-prob 0.90 \
  --teacher-forcing-final-prob 0.05 \
  --decode-diversity-penalty 0.35 \
  --teacher-guidance-weight 1.50 \
  --checkpoint-eval-episodes 3 \
  --reinforce-epochs 4 \
  --eval-episodes 3 \
  --seed 42 \
  --log-level INFO \
  2>&1 | tee "${ROOT}/train.log"

echo "[2/4] Running related-work benchmark"
"${PYTHON_BIN}" compare_related_work_papers.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${RELATED_WORK_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --device "${DEVICE}" \
  --eval-episodes 3 \
  --episode-len 40 \
  --n-sbs 8 \
  --n-ues 60 \
  --cache-capacity 10 \
  --fp 30 \
  --grid-size 200 \
  --decode-diversity-penalty 0.35 \
  --teacher-guidance-weight 1.50 \
  --log-level INFO \
  2>&1 | tee "${ROOT}/related_work.log"

echo "[3/4] Regenerating main-run summary plots"
"${PYTHON_BIN}" plot_novel_realworld_results.py \
  --input-dir "${RUN_DIR}" \
  --output-dir "${RUN_DIR}" \
  --related-work-dir "${RELATED_WORK_DIR}"

echo "[4/4] Writing benchmark report"
"${PYTHON_BIN}" - "${RELATED_WORK_DIR}/summary.csv" "${REPORT_PATH}" "${FAIL_IF_OURS_NOT_BEST}" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
report_path = Path(sys.argv[2])
fail_if_not_best = sys.argv[3] == "1"

with summary_csv.open() as f:
    rows = list(csv.DictReader(f))

ours = next((r for r in rows if r["scheme"] in {"Our-TemporalGraph", "TemporalGraph"}), None)
if ours is None:
    raise SystemExit("TemporalGraph row missing from related-work summary.")

reward_best = max(rows, key=lambda r: float(r["reward_mean"]))
local_best = max(rows, key=lambda r: float(r["local_hit_mean"]))
paper_best = max(rows, key=lambda r: float(r["paper_hit_mean"]))

lines = [
    f"Related-work summary file: {summary_csv}",
    f"Our reward_mean={float(ours['reward_mean']):.6f} | best={reward_best['scheme']}",
    f"Our local_hit_mean={float(ours['local_hit_mean']):.6f} | best={local_best['scheme']}",
    f"Our paper_hit_mean={float(ours['paper_hit_mean']):.6f} | best={paper_best['scheme']}",
]

report_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "============================================================"
echo "Fast Prototype Run Complete: ${ROOT}"
echo "============================================================"
