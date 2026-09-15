#!/usr/bin/env python
"""Stage 01: fetch WBD, extract the HUC-4 0204 boundary, write buffered domain.

Outputs data/intermediate/huc4_0204.gpkg with layers:
    huc4           dissolved basin polygon (EPSG:4269)
    huc4_buffered  +2 km buffer (buffered in UTM 18N, stored in 4269)
"""

import argparse
import logging
import zipfile

import dem_pipeline as dp
from dem_pipeline.download import fetch

log = logging.getLogger("01_get_boundary")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    dp.setup_logging()
    cfg = dp.load_config()

    import geopandas as gpd

    wbd_zip = dp.path("data", "raw", "wbd", "WBD_02_HU2_GPKG.zip")
    fetch(cfg["wbd"]["url"], wbd_zip)

    wbd_dir = wbd_zip.parent
    gpkgs = list(wbd_dir.rglob("*.gpkg"))
    if not gpkgs:
        with zipfile.ZipFile(wbd_zip) as z:
            z.extractall(wbd_dir)
        gpkgs = list(wbd_dir.rglob("*.gpkg"))
    gpkg = gpkgs[0]
    log.info("reading WBDHU4 from %s", gpkg)

    hu4 = gpd.read_file(gpkg, layer="WBDHU4")
    basin = hu4[hu4["huc4"] == cfg["project"]["huc4"]]
    if basin.empty:
        raise SystemExit(f"huc4 {cfg['project']['huc4']} not found in WBDHU4")
    basin = basin.dissolve().to_crs(epsg=4269)[["geometry"]]

    # Optional development clip. Every downstream stage derives its scope from
    # this boundary, so restricting it here restricts the whole pipeline.
    clip = cfg["project"].get("clip_bbox")
    if clip:
        from shapely.geometry import box

        basin = gpd.clip(basin, box(*clip))
        if basin.empty or basin.geometry.is_empty.all():
            raise SystemExit(f"clip_bbox {clip} does not intersect the basin")
        log.warning("clip_bbox ACTIVE: domain restricted to %s -- this is a "
                    "sub-region, NOT the full HUC-4 %s", clip,
                    cfg["project"]["huc4"])

    buffered = (
        basin.to_crs(epsg=26918)
        .buffer(cfg["project"]["boundary_buffer_m"])
        .to_crs(epsg=4269)
        .to_frame("geometry")
    )

    out = dp.huc4_boundary_gpkg(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    basin.to_file(out, layer="huc4", driver="GPKG")
    buffered.to_file(out, layer="huc4_buffered", driver="GPKG")
    log.info("wrote %s; basin bounds %s", out, basin.total_bounds.round(3))


if __name__ == "__main__":
    main()
