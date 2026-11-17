#!/usr/bin/env python
"""Merge finished per-basin products into one region-wide tile set.

Steps (each idempotent; `all` runs them in order):
  prepare   merged tile_grid.json, basin territories, region polygon, tile list
  tiles     merge every tile (or --tile T) in parallel (--nproc)
  vrt       mosaic VRT + overviews
  qa        grid / seams / region-nodata / merge-summary report

Reads only the basins' data/final/tiles; never writes into a basin checkout.
See dem_pipeline/merge.py and docs/ARCHITECTURE.md §11.

  python scripts/merge_region.py --config config/merge_region02.yaml all --nproc 2
"""

import argparse
import logging
import multiprocessing as mp
import sys

import dem_pipeline as dp
from dem_pipeline import merge

log = logging.getLogger("merge_region")

_STATE = {}


def _init(cfg_path):
    dp.setup_logging()
    mcfg = merge.load_merge_config(cfg_path)
    basins, spec, terr, digest, ids, tiles = merge.load_prepared(mcfg)
    _STATE.update(mcfg=mcfg, spec=spec, terr=terr, digest=digest, ids=ids,
                  tiles=tiles)


def _one(tile):
    s = _STATE
    try:
        st = merge.merge_tile(tile, s["tiles"][tile], s["mcfg"], s["spec"],
                              s["terr"], s["digest"], s["ids"])
        return tile, st["mode"], None
    except Exception as e:  # report and keep going; exit status reflects it
        logging.getLogger("merge_region").exception("%s failed", tile)
        return tile, None, repr(e)


def run_tiles(cfg_path, only, nproc):
    _init(cfg_path)
    tiles = list(_STATE["tiles"])
    if only:
        unknown = set(only) - set(tiles)
        if unknown:
            raise SystemExit(f"no basin produced: {sorted(unknown)}")
        tiles = [t for t in tiles if t in only]
    # multi-basin tiles first: they are the slow, memory-heavy ones
    tiles.sort(key=lambda t: -len(_STATE["tiles"][t]))
    failed = []
    if nproc <= 1:
        results = map(_one, tiles)
    else:
        pool = mp.get_context("fork").Pool(nproc, initializer=_init,
                                           initargs=(cfg_path,),
                                           maxtasksperchild=4)
        results = pool.imap_unordered(_one, tiles)
    for i, (tile, mode, err) in enumerate(results, 1):
        if err:
            failed.append(tile)
        log.info("[%d/%d] %s %s", i, len(tiles), tile, mode or f"FAILED {err}")
    if nproc > 1:
        pool.close()
        pool.join()
    if failed:
        raise SystemExit(f"{len(failed)} tile(s) failed: {failed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(dp.path("config", "merge_region02.yaml")))
    ap.add_argument("step", choices=["prepare", "tiles", "vrt", "qa", "all"])
    ap.add_argument("--tile", action="append", help="restrict 'tiles' (repeatable)")
    ap.add_argument("--nproc", type=int, default=2,
                    help="parallel tiles; a 3-basin tile peaks ~5 GB, and the "
                         "login-node per-user cap is 30 GB")
    args = ap.parse_args()
    dp.setup_logging()
    mcfg = merge.load_merge_config(args.config)

    if args.step in ("prepare", "all"):
        merge.prepare(mcfg)
    if args.step in ("tiles", "all"):
        run_tiles(args.config, args.tile, args.nproc)
    if args.step in ("vrt", "all"):
        merge.build_vrt(mcfg)
    if args.step in ("qa", "all"):
        checks = merge.run_qa(mcfg)
        hard = [c["check"] for c in checks if "pass" in c and not c["pass"]]
        for c in checks:
            if "pass" in c:
                log.info("%-18s %s", c["check"], "PASS" if c["pass"] else "FAIL")
        if hard:
            sys.exit(f"QA FAILED: {hard} -- see "
                     f"{merge.output_root(mcfg) / 'qa' / 'qa_report.json'}")


if __name__ == "__main__":
    main()
