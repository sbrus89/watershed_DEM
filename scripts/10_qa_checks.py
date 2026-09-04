#!/usr/bin/env python
"""Stage 10: QA suite -> data/final/qa/qa_report.json + quicklooks.

Checks: grid conformity (bit-exact vs CUDEM), cross-tile seams, land-overlap
statistics against CUDEM, per-tile source-mix coverage, hillshade quicklooks.
Exits nonzero if any hard check fails.
"""

import argparse
import csv
import json
import logging
import re

import dem_pipeline as dp
from dem_pipeline import qa

log = logging.getLogger("10_qa_checks")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", default=None, help="restrict to one tile name")
    ap.add_argument("--no-quicklooks", action="store_true")
    args = ap.parse_args()
    dp.setup_logging()
    cfg = dp.load_config()

    out_dir = dp.path(cfg["output"]["tile_dir"])
    tile_paths = {}
    for p in dp.product_tifs(out_dir, f"{cfg['project']['name']}_*.tif"):
        if p.stem.endswith("_source"):
            continue
        m = re.search(r"n\d{2}x\d{2}_w\d{3}x\d{2}", p.stem)
        if m:
            tile_paths[m.group(0)] = p
    if args.tile:
        tile_paths = {k: v for k, v in tile_paths.items() if k == args.tile}
    if not tile_paths:
        raise SystemExit("no final tiles found -- run stage 09 first")
    log.info("QA over %d tiles", len(tile_paths))

    checks = [
        qa.check_grid_conformity(tile_paths),
        qa.check_seams(tile_paths),
        qa.check_cudem_land_overlap(tile_paths),
        qa.coverage_report(tile_paths),
    ]

    qa_dir = dp.path(cfg["qa"]["dir"])
    qa_dir.mkdir(parents=True, exist_ok=True)
    with open(qa_dir / "qa_report.json", "w") as f:
        json.dump(checks, f, indent=2)

    cov = next(c for c in checks if c["check"] == "coverage")
    sources = sorted({k for row in cov["tiles"].values() for k in row})
    with open(qa_dir / "coverage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile"] + sources)
        for tile, row in sorted(cov["tiles"].items()):
            w.writerow([tile] + [row.get(s, 0) for s in sources])

    if not args.no_quicklooks:
        for name, p in tile_paths.items():
            qa.hillshade_quicklook(p)
            qa.provenance_quicklook(p)
        log.info("quicklooks in %s", dp.path(cfg["qa"]["quicklook_dir"]))

    hard = [c for c in checks if "pass" in c and not c["pass"]]
    for c in checks:
        if "pass" in c:
            log.info("%-22s %s", c["check"], "PASS" if c["pass"] else "FAIL")
    if hard:
        raise SystemExit(f"QA FAILED: {[c['check'] for c in hard]} "
                         f"-- see {qa_dir / 'qa_report.json'}")
    log.info("QA passed; report at %s", qa_dir / "qa_report.json")


if __name__ == "__main__":
    main()
