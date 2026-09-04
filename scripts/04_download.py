#!/usr/bin/env python
"""Stage 04: execute manifests/downloads.csv (resumable, concurrent).

Rerun freely: complete files are skipped, partial files resume via Range.

    python scripts/04_download.py                        # everything
    python scripts/04_download.py --sources s1m          # one source
    python scripts/04_download.py --tile n40x00_w075x25  # test-slice subset
"""

import argparse

import dem_pipeline as dp
from dem_pipeline.download import fetch_manifest
from dem_pipeline.grid import nominal_extent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="*", default=None,
                    help="restrict to these source names")
    ap.add_argument("--tile", default=None,
                    help="only files intersecting this output tile")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    dp.setup_logging()

    bounds = nominal_extent(args.tile) if args.tile else None
    fetch_manifest(dp.path("manifests", "downloads.csv"),
                   sources=args.sources, workers=args.workers, bounds=bounds)


if __name__ == "__main__":
    main()
