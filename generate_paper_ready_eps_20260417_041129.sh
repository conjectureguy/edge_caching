#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/rahul/Desktop/edge_caching"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/rahul/miniforge3/bin/python}"
RUN_ROOT="${RUN_ROOT:-/home/rahul/Desktop/edge_caching/plots-for-paper/temporalgraph_20260417_041129}"
SOURCE_PDF_DIR="${SOURCE_PDF_DIR:-${RUN_ROOT}/paper_ready_bundle}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/paper_ready_bundle_eps}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/novel_realworld_main}"
PLOT_INCLUDE_MODELS="${PLOT_INCLUDE_MODELS:-}"
PLOT_EXCLUDE_MODELS="${PLOT_EXCLUDE_MODELS:-}"

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

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Required path missing: ${path}" >&2
    exit 1
  fi
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Required command not found: ${cmd}" >&2
    exit 1
  fi
}

require_path "${RUN_ROOT}"
require_path "${SOURCE_PDF_DIR}"
require_path "${RUN_DIR}"
require_path "${RUN_ROOT}/related_work_compare"
require_path "${RUN_ROOT}/novel_comparison_bundle"
require_command "${PYTHON_BIN}"
require_command pdftops

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output directory: ${OUTPUT_DIR}" >&2
  exit 1
fi

PLOT_INCLUDE_ARGS=()
PLOT_EXCLUDE_ARGS=()
csv_to_arg_array "--include-models" "${PLOT_INCLUDE_MODELS}" PLOT_INCLUDE_ARGS
csv_to_arg_array "--exclude-models" "${PLOT_EXCLUDE_MODELS}" PLOT_EXCLUDE_ARGS

TMP_OUTPUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/paper_ready_eps_XXXXXX")"
cleanup() {
  rm -rf "${TMP_OUTPUT_DIR}"
}
trap cleanup EXIT

echo "============================================================"
echo "Generate EPS files for paper-ready bundle"
echo "Repo root    : ${REPO_ROOT}"
echo "Python       : ${PYTHON_BIN}"
echo "Run root     : ${RUN_ROOT}"
echo "Run dir      : ${RUN_DIR}"
echo "Source PDFs  : ${SOURCE_PDF_DIR}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Include list : ${PLOT_INCLUDE_MODELS:-<all models>}"
echo "Exclude list : ${PLOT_EXCLUDE_MODELS:-<none>}"
echo "============================================================"

echo "[1/3] Rendering native EPS plots via generate_plots_apr11.py"
"${PYTHON_BIN}" generate_plots_apr11.py \
  --data-dir "${RUN_ROOT}" \
  --data-dir-v3 "${RUN_ROOT}" \
  --run-dir "${RUN_DIR}" \
  --output-dir "${TMP_OUTPUT_DIR}" \
  --format eps \
  --python-bin "${PYTHON_BIN}" \
  "${PLOT_INCLUDE_ARGS[@]}" \
  "${PLOT_EXCLUDE_ARGS[@]}"

echo "[2/3] Converting any remaining PDF-only plots to EPS in the temp workspace"
converted=0
skipped=0
while IFS= read -r -d '' pdf_path; do
  rel_path="${pdf_path#${SOURCE_PDF_DIR}/}"
  eps_path="${TMP_OUTPUT_DIR}/${rel_path%.pdf}.eps"
  if [[ -f "${eps_path}" ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  mkdir -p "$(dirname "${eps_path}")"
  pdftops -eps "${pdf_path}" "${eps_path}"
  converted=$((converted + 1))
done < <(find "${SOURCE_PDF_DIR}" -type f -name '*.pdf' -print0)

echo "[3/3] Copying only EPS files into the new output directory"
mkdir -p "${OUTPUT_DIR}"
export SOURCE_PDF_DIR TMP_OUTPUT_DIR OUTPUT_DIR
copied="$("${PYTHON_BIN}" - <<'PY'
import os
import shutil
from pathlib import Path

source_pdf_dir = Path(os.environ["SOURCE_PDF_DIR"])
tmp_output_dir = Path(os.environ["TMP_OUTPUT_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])

count = 0
for eps_path in sorted(tmp_output_dir.rglob("*.eps")):
    rel = eps_path.relative_to(tmp_output_dir)
    dest = output_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(eps_path, dest)
    count += 1

print(count)
PY
)"

echo "============================================================"
echo "EPS generation complete"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Copied EPS files  : ${copied}"
echo "New PDF->EPS conv : ${converted}"
echo "Already had EPS   : ${skipped}"
echo "============================================================"
