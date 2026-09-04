#!/usr/bin/env bash
# End-to-end pipeline test on a single output tile before the basin run.
#
# Default tile n40x00_w075x25 (Philadelphia) exercises every branch: dense
# S1M land, CUDEM tidal river, active CENAP eHydro surveys, NHD shoreline,
# urban hydro-flattening. Try n41x75_w075x00 (Hancock reach) for EAARL-B.
#
# Usage: bash scripts/run_test_slice.sh [tile]
set -euo pipefail
cd "$(dirname "$0")/.."

TILE="${1:-n40x00_w075x25}"
echo "=== test slice: ${TILE} ==="

python scripts/01_get_boundary.py
python scripts/02_audit_cudem_grid.py
python scripts/03_discover_tiles.py
python scripts/04_download.py --tile "${TILE}"
python scripts/05_build_source_vrts.py
python scripts/06_process_ehydro.py
python scripts/07_build_water_mask.py
python scripts/08_composite_tile.py --tile "${TILE}"
python scripts/09_make_cogs_vrt.py --tile "${TILE}"
python scripts/10_qa_checks.py --tile "${TILE}"

echo "=== test slice complete ==="
echo "inspect: data/final/tiles/ and data/final/quicklooks/"
