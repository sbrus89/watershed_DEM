#!/usr/bin/env python
"""Stage 02: download basin CUDEM 1/9" tiles and derive the target grid spec.

The CUDEM tiles are the compatibility contract for the whole product: this
stage inspects every downloaded ncei19 tile, verifies they share one grid
convention (pixel size 1/32400 deg, common collar/registration, CRS, nodata),
and writes the empirical spec to config/tile_grid.json. It then emits
config/tiles_0204.geojson: the full output tile list (0.25 deg lattice
intersecting the buffered basin, unioned with CUDEM coverage).

Fails loudly if CUDEM tiles disagree with each other; in that case the
per-tile geotransforms recorded in tile_grid.json remain authoritative where
CUDEM exists and the modal convention is used elsewhere.
"""

import argparse
import json
import logging
import re
from collections import Counter

import dem_pipeline as dp
from dem_pipeline.download import fetch, write_manifest
from dem_pipeline.grid import (GridSpec, PIXEL_DEG, TILE_PX, lattice_tiles,
                               nominal_extent, parse_tile_name)

log = logging.getLogger("02_audit_cudem_grid")

# Real CUDEM tiles do NOT store the pixel size exactly: each vintage wrote it
# at whatever precision it used, and the observed spread across the Delaware
# basin's 49 tiles is up to 3.7e-12 deg/px (2019v1 rounds to 8 significant
# figures; one 2018v2 tile carries its own value). Gate on how far that drifts
# across a whole tile rather than on a raw epsilon: 1e-7 deg is ~1 cm, roughly
# 3x the worst observed (~3 mm), and still rejects a genuinely different pixel
# size -- 1/3 arc-second is off by 2e-5 deg/px, five orders of magnitude out.
MAX_TILE_DRIFT_DEG = 1e-7


def select_cudem_urls(cfg, basin_geom) -> list[str]:
    """Basin-intersecting tile URLs parsed from the published url list."""
    import requests
    from shapely.geometry import box

    r = requests.get(cfg["sources"]["cudem19"]["urllist"], timeout=120)
    r.raise_for_status()
    urls = []
    for line in r.text.splitlines():
        line = line.strip()
        if not line.endswith(".tif") or "ncei19_" not in line:
            continue
        try:
            north, west = parse_tile_name(line)
        except ValueError:
            continue
        if basin_geom.intersects(box(west, north - 0.25, west + 0.25, north)):
            urls.append(line)
    return sorted(set(urls))


def audit_tiles(tile_paths) -> GridSpec:
    import rasterio

    per_tile = {}
    conventions = Counter()
    pixel_sizes = Counter()
    for p in tile_paths:
        name_match = re.search(r"n\d{2}x\d{2}_w\d{3}x\d{2}", p.name)
        name = name_match.group(0)
        with rasterio.open(p) as src:
            gt = src.transform.to_gdal()
            epsg = src.crs.to_epsg() if src.crs else None
            nodata = src.nodata
            per_tile[name] = {"gt": list(gt), "size": [src.width, src.height],
                              "epsg": epsg, "nodata": nodata}
        px = gt[1]
        drift = abs(px - PIXEL_DEG) * TILE_PX
        if drift > MAX_TILE_DRIFT_DEG:
            raise SystemExit(
                f"{p.name}: pixel size {px!r} drifts {drift:.3e} deg across a "
                f"tile (limit {MAX_TILE_DRIFT_DEG:g}) -- not the 1/9\" lattice")
        pixel_sizes[f"{px:.17g}"] += 1
        west, south, east, north = nominal_extent(name)
        collar_px = round((west - gt[0]) / px)
        x_shift = (gt[0] - (west - collar_px * px)) / px
        y_shift = (gt[3] - (north + collar_px * px)) / px
        conventions[(collar_px, round(x_shift, 6), round(y_shift, 6),
                     src.width, epsg, nodata)] += 1

    epsgs = Counter(v["epsg"] for v in per_tile.values())
    nodatas = Counter(v["nodata"] for v in per_tile.values())
    if len(epsgs) > 1 or len(nodatas) > 1:
        log.warning(
            "CUDEM tiles are NOT internally consistent: epsg=%s nodata=%s. "
            "There is no single CUDEM convention to inherit, so the product "
            "uses the configured target_grid values; per-tile geotransforms "
            "stay verbatim, so bit-exactness per tile is unaffected.",
            dict(epsgs), dict(nodatas))
    if len(pixel_sizes) > 1:
        log.info("pixel sizes present (all within the drift limit): %s",
                 dict(pixel_sizes))
    if len(conventions) > 1:
        log.warning("CUDEM tiles carry %d distinct grid conventions: %s "
                    "-- using the modal one for non-CUDEM tiles; CUDEM tiles "
                    "keep their exact geotransforms",
                    len(conventions), dict(conventions))
    collar, xs, ys, width, epsg, nodata = conventions.most_common(1)[0][0]
    if width != TILE_PX + 2 * collar:
        log.warning("modal width %d inconsistent with collar %d", width, collar)

    return GridSpec(
        epsg=epsg or 4269,
        nodata=float(nodata) if nodata is not None else -999999.0,
        pixel_deg=PIXEL_DEG,
        collar_px=collar,
        x_shift_px=xs,
        y_shift_px=ys,
        cudem_geotransforms={k: v["gt"] for k, v in per_tile.items()},
        cudem_sizes={k: v["size"] for k, v in per_tile.items()},
    )


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    dp.setup_logging()
    cfg = dp.load_config()

    import geopandas as gpd
    from shapely.geometry import box, mapping

    boundary = gpd.read_file(
        dp.path("data", "intermediate", "huc4_0204.gpkg"), layer="huc4_buffered"
    )
    basin_geom = boundary.geometry.union_all()

    urls = select_cudem_urls(cfg, basin_geom)
    log.info("selected %d CUDEM tiles intersecting the buffered basin", len(urls))
    raw_dir = dp.path(cfg["sources"]["cudem19"]["dir"])
    tile_paths = [fetch(u, raw_dir / u.rsplit("/", 1)[1]) for u in urls]
    write_manifest(
        [{"url": u, "dest": str(raw_dir / u.rsplit("/", 1)[1]),
          "source": "cudem19"} for u in urls],
        dp.path("manifests", "cudem19.csv"),
    )

    spec = audit_tiles(tile_paths)
    spec.dump()

    # The product contract lives in config; the audit only reports divergence.
    # Inheriting these would make the output CRS/nodata depend on which tiles a
    # given run audited (a clip changes the modal vintage).
    tg = cfg["target_grid"]
    if spec.epsg != tg["epsg"]:
        log.warning("modal CUDEM EPSG %s != configured target_grid.epsg %s -- "
                    "output tiles use %s; change the config if that is wrong",
                    spec.epsg, tg["epsg"], tg["epsg"])
    if spec.nodata != tg["nodata"]:
        log.warning("modal CUDEM nodata %s != configured target_grid.nodata %s "
                    "-- output tiles use %s", spec.nodata, tg["nodata"],
                    tg["nodata"])
    log.info("wrote config/tile_grid.json (epsg=%s nodata=%s collar=%d "
             "shift=(%g,%g), %d CUDEM tiles recorded)",
             spec.epsg, spec.nodata, spec.collar_px,
             spec.x_shift_px, spec.y_shift_px, len(spec.cudem_geotransforms))

    # Output tile list: lattice over buffered basin, plus all CUDEM tiles.
    names = [n for n in lattice_tiles(tuple(boundary.total_bounds))
             if basin_geom.intersects(box(*nominal_extent(n)))]
    names = sorted(set(names) | set(spec.cudem_geotransforms))
    features = []
    for n in names:
        w, s, e, no = nominal_extent(n)
        features.append({
            "type": "Feature",
            "properties": {"tile": n, "has_cudem": n in spec.cudem_geotransforms},
            "geometry": mapping(box(w, s, e, no)),
        })
    out = dp.path("config", "tiles_0204.geojson")
    with open(out, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    log.info("wrote %s: %d output tiles (%d with CUDEM overlap)",
             out, len(names), len(spec.cudem_geotransforms))


if __name__ == "__main__":
    main()
