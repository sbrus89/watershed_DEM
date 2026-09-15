#!/usr/bin/env bash
# Stages 08-10 for the current clip_bbox, composing tiles in parallel.
#
# Stages 01-07 must already be done (source VRTs, ehydro grids, water mask).
# Stage 08 is idempotent per tile, so re-running after an interruption only
# composes what is missing -- but note that means an EXISTING tile is never
# rebuilt: delete a tile if its inputs changed.
#
# Usage: bash scripts/run_stages_08_10.sh [n_parallel]   (default 12)
set -euo pipefail
cd "$(dirname "$0")/.."

NPROC="${1:-12}"
TILES=$(python -c "
import json
import dem_pipeline as dp
fc = json.load(open(dp.output_tiles_geojson()))
print('\n'.join(f['properties']['tile'] for f in fc['features']))")
N=$(echo "$TILES" | wc -l)

echo "=== stage 08: composing ${N} tiles, ${NPROC} at a time ==="
echo "$TILES" | xargs -P "$NPROC" -I{} -n1 \
    python scripts/08_composite_tile.py --tile {}

echo "=== stage 09: COGs + mosaic VRT ==="
python scripts/09_make_cogs_vrt.py

echo "=== stage 10: QA ==="
python scripts/10_qa_checks.py

echo "=== done: data/final/tiles/, quicklooks in data/final/quicklooks/ ==="
