#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/rahul/Desktop/edge_caching"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/rahul/miniforge3/bin/python}"
DEVICE="${DEVICE:-cpu}"
DATASET_NAME="${DATASET_NAME:-ml-1m}"
RUN_ROOT="${RUN_ROOT:-outputs/research_benchmark_runs/temporalgraph_20260417_041129}"
FAIL_IF_OURS_NOT_BEST="${FAIL_IF_OURS_NOT_BEST:-0}"
C_EPSILON="${C_EPSILON:-0.18}"
PLOT_INCLUDE_MODELS="${PLOT_INCLUDE_MODELS:-TemporalGraph,MAAFDRL,Thompson,LFU,LRU,C-epsilon-greedy,BSG-like,Random}"
PLOT_EXCLUDE_MODELS="${PLOT_EXCLUDE_MODELS:-}"

COMPARISON_EVAL_EPISODES="${COMPARISON_EVAL_EPISODES:-16}"
COMPARISON_EPISODE_LEN="${COMPARISON_EPISODE_LEN:-80}"

ROOT="${RUN_ROOT}"
ROOT_BASE="$(dirname "${ROOT}")"
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

mkdir -p \
  "${ROOT_BASE}" \
  "${ROOT}" \
  "${RUN_DIR}" \
  "${EPISODE_EPOCH_DIR}" \
  "${LATENCY_CLUSTERED_DIR}" \
  "${STATIC_DYNAMIC_DIR}" \
  "${SHOWCASE_DIR}" \
  "${FINAL_NO_TEACHER_DIR}" \
  "${COMPARISON_BUNDLE_DIR}" \
  "${RELATED_WORK_DIR}" \
  "${PAPER_READY_DIR}"
ln -sfn "$(basename "${ROOT}")" "${LATEST_LINK}"

csv_to_arg_array() {
  local flag="$1"
  local csv="$2"
  local -n out_ref="$3"
  out_ref=()
  if [[ -z "${csv}" ]]; then
    return
  fi
  csv="${csv//,/ }"
  # shellcheck disable=SC2206
  local values=( ${csv} )
  if [[ ${#values[@]} -gt 0 ]]; then
    out_ref=("${flag}" "${values[@]}")
  fi
}

PLOT_INCLUDE_ARGS=()
PLOT_EXCLUDE_ARGS=()
csv_to_arg_array "--include-models" "${PLOT_INCLUDE_MODELS}" PLOT_INCLUDE_ARGS
csv_to_arg_array "--exclude-models" "${PLOT_EXCLUDE_MODELS}" PLOT_EXCLUDE_ARGS

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file missing: ${path}" >&2
    exit 1
  fi
}

require_file "${RUN_DIR}/realworld_temporal_encoder.pt"
require_file "${RUN_DIR}/temporal_graph_policy.pt"
require_file "${RUN_DIR}/temporal_graph_eval.csv"
require_file "${RUN_DIR}/teacher_eval.csv"
require_file "${RELATED_WORK_DIR}/summary.csv"
require_file "${RELATED_WORK_DIR}/episode_metrics.csv"

echo "============================================================"
echo "TemporalGraph late-stage benchmark resume"
echo "Repo root   : ${REPO_ROOT}"
echo "Python      : ${PYTHON_BIN}"
echo "Device      : ${DEVICE}"
echo "Dataset     : ${DATASET_NAME}"
echo "Run root    : ${ROOT}"
echo "Main run dir: ${RUN_DIR}"
echo "C-epsilon   : ${C_EPSILON}"
echo "Plot include: ${PLOT_INCLUDE_MODELS}"
echo "Plot exclude: ${PLOT_EXCLUDE_MODELS:-<none>}"
echo "============================================================"

echo "[0/7] Validating late-stage entrypoints"
"${PYTHON_BIN}" -m py_compile \
  plot_novel_realworld_results.py \
  plot_episode_epoch_curves.py \
  plot_clustered_latency_study.py \
  plot_static_vs_dynamic_bundle.py \
  plot_temporalgraph_showcase.py \
  plot_final_no_teacher_bundle.py \
  plot_novel_comparison_bundle.py \
  generate_plots_apr11.py

echo "[1/7] Regenerating main-run summary plots for the final model comparisons"
"${PYTHON_BIN}" plot_novel_realworld_results.py \
  --input-dir "${RUN_DIR}" \
  --output-dir "${RUN_DIR}" \
  --related-work-dir "${RELATED_WORK_DIR}" \
  "${PLOT_INCLUDE_ARGS[@]}" \
  "${PLOT_EXCLUDE_ARGS[@]}"

echo "[2/7] Episode/epoch curves"
"${PYTHON_BIN}" plot_episode_epoch_curves.py \
  --input-dir "${RUN_DIR}" \
  --output-dir "${EPISODE_EPOCH_DIR}"

echo "[3/7] Static-dynamic and latency bundles"
"${PYTHON_BIN}" plot_clustered_latency_study.py \
  --output-dir "${LATENCY_CLUSTERED_DIR}"

"${PYTHON_BIN}" plot_static_vs_dynamic_bundle.py \
  --output-dir "${STATIC_DYNAMIC_DIR}"

echo "[4/7] Showcase and no-teacher bundles"
"${PYTHON_BIN}" plot_temporalgraph_showcase.py \
  --primary-run "${RUN_DIR}" \
  --output-dir "${SHOWCASE_DIR}" \
  --related-work-dir "${RELATED_WORK_DIR}" \
  --exclude-teacher \
  --skip-secondary \
  "${PLOT_INCLUDE_ARGS[@]}" \
  "${PLOT_EXCLUDE_ARGS[@]}"

"${PYTHON_BIN}" plot_final_no_teacher_bundle.py \
  --input-dir "${RUN_DIR}" \
  --output-dir "${FINAL_NO_TEACHER_DIR}" \
  --related-work-dir "${RELATED_WORK_DIR}" \
  "${PLOT_INCLUDE_ARGS[@]}" \
  "${PLOT_EXCLUDE_ARGS[@]}"

echo "[5/7] Extended comparison bundle"
"${PYTHON_BIN}" plot_novel_comparison_bundle.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${COMPARISON_BUNDLE_DIR}" \
  --device "${DEVICE}" \
  --eval-episodes "${COMPARISON_EVAL_EPISODES}" \
  --episode-len "${COMPARISON_EPISODE_LEN}" \
  --n-ues 220 \
  --cache-capacities 10 20 30 \
  --sbs-list 8 12 16 \
  --c-epsilon "${C_EPSILON}" \
  "${PLOT_INCLUDE_ARGS[@]}" \
  "${PLOT_EXCLUDE_ARGS[@]}"

echo "[6/7] Paper-ready PDF bundle"
"${PYTHON_BIN}" generate_plots_apr11.py \
  --data-dir "${ROOT}" \
  --data-dir-v3 "${ROOT}" \
  --run-dir "${RUN_DIR}" \
  --output-dir "${PAPER_READY_DIR}" \
  --format pdf \
  --python-bin "${PYTHON_BIN}" \
  "${PLOT_INCLUDE_ARGS[@]}" \
  "${PLOT_EXCLUDE_ARGS[@]}"

echo "[7/7] Writing benchmark report and manifest"
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
echo "Resume complete"
echo "Root bundle      : ${ROOT}"
echo "Main run         : ${RUN_DIR}"
echo "Related work     : ${RELATED_WORK_DIR}"
echo "PNG bundle       : ${ROOT}"
echo "PDF bundle       : ${PAPER_READY_DIR}"
echo "Manifest         : ${MANIFEST_PATH}"
echo "Benchmark report : ${REPORT_PATH}"
echo "Latest symlink   : ${LATEST_LINK}"
echo "============================================================"
