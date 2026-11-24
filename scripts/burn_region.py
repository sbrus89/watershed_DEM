#!/usr/bin/env python
"""Burn mapped stream channels into the merged region product (optional).

Writes a NEW tile set (config project.name, output.root); the merged product
is only read. Steps (each idempotent; `all` runs them in order):
  prepare   check provenance ids; per-HUC-4 ML attribute table + TIGER crossings
  network   per-HUC-4 channel network pass -> per-tile burn specs (single process)
  sources   per-reach depth-source table for the viewer (<root>/qa/depth_source/)
  tiles     rasterize + apply every tile (or --tile T) in parallel (--nproc)
  vrt       mosaic VRT + overviews
  qa        grid / seams / burn summary (--connectivity: ponding on the network
            before vs after for qa.connectivity_tiles; slow, full-tile)

See dem_pipeline/burn.py and docs/ARCHITECTURE.md §12.

  python scripts/burn_region.py --config config/burn_region02.yaml all --nproc 4
"""

import argparse
import logging
import multiprocessing as mp
import sys

import dem_pipeline as dp
from dem_pipeline import burn

log = logging.getLogger("burn_region")

_STATE = {}


def _init(cfg_path):
    dp.setup_logging()
    b = burn.load_burn_config(cfg_path)
    spec, tiles = burn.load_input(b)
    _STATE.update(b=b, spec=spec, tiles=tiles, L=burn.Lattice(spec, tiles))


def _one(tile):
    s = _STATE
    try:
        st = burn.burn_tile(s["b"], tile, s["spec"], s["L"])
        return tile, st["mode"], st.get("n_lowered", 0), None
    except Exception as e:  # report and keep going; exit status reflects it
        logging.getLogger("burn_region").exception("%s failed", tile)
        return tile, None, 0, repr(e)


def run_tiles(cfg_path, only, nproc):
    _init(cfg_path)
    tiles = list(_STATE["tiles"])
    if only:
        unknown = set(only) - set(tiles)
        if unknown:
            raise SystemExit(f"not in the merged product: {sorted(unknown)}")
        tiles = [t for t in tiles if t in only]
    failed = []
    if nproc <= 1:
        results = map(_one, tiles)
    else:
        pool = mp.get_context("fork").Pool(nproc, initializer=_init,
                                           initargs=(cfg_path,), maxtasksperchild=8)
        results = pool.imap_unordered(_one, tiles)
    for i, (tile, mode, n, err) in enumerate(results, 1):
        if err:
            failed.append(tile)
        log.info("[%d/%d] %s %s", i, len(tiles), tile,
                 f"FAILED {err}" if err else f"{mode} ({n} px lowered)")
    if nproc > 1:
        # Every result is in hand, so nothing is pending: terminate rather
        # than close()+join().  join() hung 8 h after the last tile in job
        # 61684422 (a maxtasksperchild replacement worker forked while a lock
        # was held never reads its shutdown sentinel).
        pool.terminate()
        pool.join()
    if failed:
        raise SystemExit(f"{len(failed)} tile(s) failed: {failed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(dp.path("config", "burn_region02.yaml")))
    ap.add_argument("step", choices=["prepare", "network", "sources", "tiles", "vrt", "qa", "all"])
    ap.add_argument("--tile", action="append", help="restrict 'tiles' (repeatable)")
    ap.add_argument("--huc4", help="restrict 'network' / 'sources' to one HUC-4")
    ap.add_argument("--force", action="store_true", help="rerun 'network' / 'sources' even if up to date")
    ap.add_argument("--nproc", type=int, default=2,
                    help="parallel tiles (~2 GB each; login-node cap is 30 GB)")
    ap.add_argument("--connectivity", action="store_true",
                    help="qa: add the depression-fill check on qa.connectivity_tiles")
    args = ap.parse_args()
    dp.setup_logging()
    b = burn.load_burn_config(args.config)

    if args.step in ("prepare", "all"):
        burn.prepare(b)
    if args.step in ("network", "all"):
        burn.run_network(b, args.huc4, args.force)
    if args.step in ("sources", "all"):
        burn.depth_sources(b, args.huc4, args.force)
    if args.step in ("tiles", "all"):
        run_tiles(args.config, args.tile, args.nproc)
    if args.step in ("vrt", "all"):
        burn.build_vrt(b)
    if args.step in ("qa", "all"):
        checks = burn.run_qa(b, args.connectivity)
        hard = [c["check"] for c in checks if "pass" in c and not c["pass"]]
        for c in checks:
            if "pass" in c:
                log.info("%-18s %s", c["check"], "PASS" if c["pass"] else "FAIL")
        if hard:
            sys.exit(f"QA FAILED: {hard} -- see "
                     f"{burn.out_root(b) / 'qa' / 'qa_report.json'}")


if __name__ == "__main__":
    main()
