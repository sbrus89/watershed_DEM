#!/usr/bin/env python
"""Stage 08: compose output tiles (land/water stacks, mask, feather).

    python scripts/08_composite_tile.py --tile n40x00_w075x25
    python scripts/08_composite_tile.py --all
    ls config/tiles | parallel -j6 python scripts/08_composite_tile.py --tile {}

Tiles come from config/tiles_0204.geojson (stage 02). Idempotent per tile.
"""

import argparse
import json
import logging

import dem_pipeline as dp
from dem_pipeline.composite import compose_tile

log = logging.getLogger("08_composite_tile")


def all_tiles() -> list[str]:
    with open(dp.path("config", "tiles_0204.geojson")) as f:
        fc = json.load(f)
    return [f_["properties"]["tile"] for f_ in fc["features"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tile", help="single tile name, e.g. n40x00_w075x25")
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()
    dp.setup_logging()

    names = [args.tile] if args.tile else all_tiles()
    failures = []
    for name in names:
        try:
            compose_tile(name)
        except Exception as e:
            log.error("%s: %s", name, e)
            failures.append(name)
    if failures:
        raise SystemExit(f"{len(failures)} tiles failed: {failures}")


if __name__ == "__main__":
    main()
