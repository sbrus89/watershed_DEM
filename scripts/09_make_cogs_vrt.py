#!/usr/bin/env python
"""Stage 09: convert composed tiles to COGs with final names + mosaic VRT.

data/intermediate/tiles/<name>.tif -> data/final/tiles/dewatershed19_<name>_<ver>.tif
plus data/final/dewatershed19_<ver>.vrt with external average overviews.
"""

import argparse
import logging
import shutil
import subprocess

import dem_pipeline as dp
from dem_pipeline.grid import GridSpec, output_filename

log = logging.getLogger("09_make_cogs_vrt")


def make_cog(src, in_dir, out_dir, cfg: dict, spec: GridSpec):
    """Build one final COG (+ provenance copy) from an intermediate tile.

    Idempotent, but only trusts an existing output whose grid still matches
    the current GridSpec -- a COG left over from a run under an older config
    (different collar/shift constants) must be rebuilt, not skipped just
    because the file happens to exist.
    """
    ocfg = cfg["output"]["cog"]
    name = src.stem
    dst = out_dir / output_filename(name, cfg)
    if dst.exists():
        if spec.tile_matches(name, dst):
            return dst
        log.warning("%s: existing COG's grid doesn't match the current "
                    "GridSpec -- rebuilding", name)
    tmp = dst.with_suffix(".tmp.tif")
    subprocess.run(
        ["gdal_translate", "-of", "COG",
         "-a_srs", f"EPSG:{cfg['target_grid']['epsg']}",
         "-co", f"COMPRESS={ocfg['compress']}",
         "-co", f"PREDICTOR={ocfg['predictor']}",
         "-co", f"BLOCKSIZE={ocfg['blocksize']}",
         "-co", f"RESAMPLING={ocfg['overview_resampling']}",
         "-co", "OVERVIEWS=IGNORE_EXISTING",
         str(src), str(tmp)],
        check=True,
    )
    # provenance raster ships alongside, deflate but not COG. Copy before
    # the elevation tile is renamed into place: dst.exists() is the resume
    # check, so it must not become true until both files are complete.
    src_prov = in_dir / f"{name}_source.tif"
    if src_prov.exists():
        prov_dst = out_dir / (dst.stem + "_source.tif")
        prov_tmp = prov_dst.with_suffix(".tmp.tif")
        shutil.copy2(src_prov, prov_tmp)
        prov_tmp.rename(prov_dst)
    tmp.rename(dst)
    log.info("wrote %s", dst.name)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", default=None, help="restrict to one tile name")
    args = ap.parse_args()
    dp.setup_logging()
    cfg = dp.load_config()

    in_dir = dp.path("data", "intermediate", "tiles")
    out_dir = dp.path(cfg["output"]["tile_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = GridSpec.load()

    tifs = [p for p in dp.product_tifs(in_dir)
            if not p.stem.endswith("_source")]
    if args.tile:
        tifs = [p for p in tifs if p.stem == args.tile]

    for src in tifs:
        make_cog(src, in_dir, out_dir, cfg, spec)

    vrt = dp.path(cfg["output"]["vrt"])
    finals = [str(p) for p in dp.product_tifs(
        out_dir, f"{cfg['project']['name']}_*.tif")
        if not p.stem.endswith("_source")]
    if finals:
        listfile = vrt.with_suffix(".inputs.txt")
        listfile.write_text("\n".join(finals) + "\n")
        subprocess.run(["gdalbuildvrt", "-overwrite", "-input_file_list",
                        str(listfile), str(vrt)], check=True)
        subprocess.run(["gdaladdo", "-r", "average", "-ro", str(vrt),
                        "16", "64", "256"], check=True)
        log.info("mosaic VRT: %s (%d tiles)", vrt, len(finals))


if __name__ == "__main__":
    main()
