#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Generate research-paper-quality plots → outputs/plots_apr11
#  The output mirrors the old apr6_v3 bundle structure with
#  paper-ready PDFs and an auto-regenerated comparison bundle
#  that includes AWFDRL and MAAFDRL across sweeps/traces.
#
#  Usage:
#    chmod +x run_plots_apr11.sh
#    ./run_plots_apr11.sh
#
#  Prerequisites: matplotlib, numpy (already in project env)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

cd /home/rahul/Desktop/edge_caching

PYTHON_BIN="${PYTHON_BIN:-/home/rahul/miniforge3/bin/python}"
OUTPUT_DIR="outputs/plots_apr11"

# Primary data source (full 10-episode eval runs with paperhit)
DATA_DIR="outputs/full_suite_20260405_paperhit"

# Fallback data source
DATA_DIR_V3="outputs/plots_apr6_v3"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating research-paper-quality plots"
echo "  Output: ${OUTPUT_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "${OUTPUT_DIR}"

"$PYTHON_BIN" generate_plots_apr11.py \
  --data-dir "$DATA_DIR" \
  --data-dir-v3 "$DATA_DIR_V3" \
  --run-dir "$DATA_DIR/novel_realworld_main" \
  --output-dir "$OUTPUT_DIR" \
  --format pdf

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ All plots generated in: ${OUTPUT_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
find "${OUTPUT_DIR}" -type f | sort
