#!/usr/bin/env python
"""Overview figures for the current output tile set.

  1. <name>_tiles_on_imagery.png -- the output tile lattice over Esri World
     Imagery, with the basin boundary and each tile's source mix.
  2. <name>_mosaic_topo.png      -- every final tile, downsampled, on one
     cmocean `topo` ramp centred at NAVD88 zero so land/water read correctly.

Imagery tiles are fetched straight from the Esri XYZ endpoint and cached under
data/final/quicklooks/.basemap_cache/ -- reruns are offline. Nothing here
feeds the product; it is inspection only.

Usage: python scripts/plot_overview.py [--zoom N] [--max-px N]
"""

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np

import dem_pipeline as dp

log = logging.getLogger("plot_overview")

ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}")
TILE_PX = 256


def _deg2num(lon, lat, z):
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def _num2deg(x, y, z):
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def fetch_basemap(bounds, zoom, cache_dir):
    """Stitched Esri World Imagery for a lon/lat bbox.

    Returns (image RGB array, extent in lon/lat). Tiles are cached on disk so
    a rerun needs no network; a tile that will not download is left black
    rather than aborting the figure.
    """
    import requests
    from PIL import Image

    w, s, e, n = bounds
    x0f, y0f = _deg2num(w, n, zoom)
    x1f, y1f = _deg2num(e, s, zoom)
    x0, x1 = int(math.floor(x0f)), int(math.ceil(x1f))
    y0, y1 = int(math.floor(y0f)), int(math.ceil(y1f))
    nx, ny = x1 - x0, y1 - y0
    log.info("basemap: z=%d, %d x %d tiles (%d total)", zoom, nx, ny, nx * ny)

    cache_dir.mkdir(parents=True, exist_ok=True)
    mosaic = Image.new("RGB", (nx * TILE_PX, ny * TILE_PX))
    session = requests.Session()
    session.headers["User-Agent"] = "DE_watershed_DEM/overview-plot"
    n_fetched = 0
    for xi in range(x0, x1):
        for yi in range(y0, y1):
            cached = cache_dir / f"{zoom}_{xi}_{yi}.jpg"
            if not cached.exists():
                try:
                    r = session.get(ESRI.format(z=zoom, x=xi, y=yi), timeout=60)
                    r.raise_for_status()
                    cached.write_bytes(r.content)
                    n_fetched += 1
                except Exception as exc:            # noqa: BLE001
                    log.warning("tile %d/%d/%d: %s", zoom, xi, yi, exc)
                    continue
            try:
                mosaic.paste(Image.open(cached).convert("RGB"),
                             ((xi - x0) * TILE_PX, (yi - y0) * TILE_PX))
            except Exception as exc:                # noqa: BLE001
                log.warning("tile %d/%d/%d unreadable: %s", zoom, xi, yi, exc)
    log.info("basemap: %d newly fetched, rest cached", n_fetched)

    wl, nl = _num2deg(x0, y0, zoom)
    el, sl = _num2deg(x1, y1, zoom)
    return np.asarray(mosaic), (wl, el, sl, nl)


def plot_tiles_on_imagery(cfg, out_png, zoom):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    with open(dp.output_tiles_geojson(cfg)) as f:
        fc = json.load(f)
    tiles = gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4269")

    # per-tile source mix, if stage 10 has run
    cov = {}
    cov_csv = dp.path(cfg["qa"]["dir"]) / "coverage.csv"
    if cov_csv.exists():
        import csv
        with open(cov_csv) as f:
            for row in csv.DictReader(f):
                cov[row["tile"]] = {k: float(v) for k, v in row.items()
                                    if k != "tile"}

    w, s, e, n = tiles.total_bounds
    pad = 0.06
    bounds = (w - pad, s - pad, e + pad, n + pad)
    img, extent = fetch_basemap(bounds, zoom,
                                dp.path(cfg["qa"]["quicklook_dir"],
                                        ".basemap_cache"))

    fig, ax = plt.subplots(figsize=(11, 15))
    ax.imshow(img, extent=[extent[0], extent[1], extent[2], extent[3]],
              origin="upper", interpolation="bilinear")

    basin = dp.huc4_boundary_gpkg(cfg)
    if basin.exists():
        gpd.read_file(basin, layer="huc4").boundary.plot(
            ax=ax, color="#ffe066", lw=2.0, zorder=3,
            label=f"HUC-4 {cfg['project']['huc4']} boundary (clipped)")

    tiles.boundary.plot(ax=ax, color="#ff3b30", lw=1.3, zorder=4)
    for _, row in tiles.iterrows():
        name = row["tile"]
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        c = cov.get(name)
        label = name.replace("_", "\n")
        if c:
            mix = " ".join(f"{k[:4]} {c[k]:.0f}" for k in
                           ("s1m", "cudem19", "ehydro", "coned")
                           if c.get(k, 0) >= 1)
            nd = c.get("nodata", 0)
            label += f"\n{mix}" + (f"\nnodata {nd:.0f}%" if nd >= 1 else "")
        ax.text(cx, cy, label, ha="center", va="center", fontsize=6.2,
                color="white", zorder=5,
                path_effects=_stroke(), linespacing=1.25)

    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"{cfg['project']['name']} {cfg['project']['version']}: "
                 f"{len(tiles)} output tiles\n"
                 "Esri World Imagery basemap; labels show source mix (%)",
                 fontsize=11)
    handles = [mpatches.Patch(edgecolor="#ff3b30", facecolor="none",
                              label="output tile (0.25 deg)")]
    if basin.exists():
        handles.append(mpatches.Patch(edgecolor="#ffe066", facecolor="none",
                                      label="basin boundary (clipped)"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_png)


def _stroke():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=1.8, foreground="black")]


def plot_mosaic(cfg, out_png, max_px):
    """All final tiles on one cmocean `topo` ramp, centred at NAVD88 zero."""
    import cmocean
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import TwoSlopeNorm
    from rasterio.enums import Resampling

    vrt = dp.path(cfg["output"]["vrt"])
    if not vrt.exists():
        raise SystemExit(f"{vrt} not found -- run stage 09 first")

    with rasterio.open(vrt) as src:
        scale = max(1, math.ceil(max(src.width, src.height) / max_px))
        out_h, out_w = src.height // scale, src.width // scale
        log.info("mosaic: %d x %d -> %d x %d (1/%d)",
                 src.width, src.height, out_w, out_h, scale)
        z = src.read(1, out_shape=(1, out_h, out_w),
                     resampling=Resampling.average, masked=True)
        b = src.bounds
    z = np.ma.masked_invalid(z)
    if z.count() == 0:
        raise SystemExit("mosaic is entirely nodata")

    # symmetric about zero so `topo`'s land/water split lands on the shoreline;
    # percentile clipping keeps a few deep-dredge or tower outliers from
    # flattening the whole ramp
    lo = float(np.percentile(z.compressed(), 0.5))
    hi = float(np.percentile(z.compressed(), 99.5))
    lo = min(lo, -1.0); hi = max(hi, 1.0)
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)

    # nodata must NOT render white: the top of `topo` is near-white too, so
    # a clipped corner would read as high ground.
    cmap = cmocean.cm.topo.copy()
    cmap.set_bad("#8c8c8c")

    fig, ax = plt.subplots(figsize=(11, 15))
    ax.set_facecolor("#8c8c8c")
    im = ax.imshow(z, extent=[b.left, b.right, b.bottom, b.top],
                   origin="upper", cmap=cmap, norm=norm,
                   interpolation="nearest")

    handles = [mpatches.Patch(facecolor="#8c8c8c", edgecolor="k",
                              label="no data (outside source coverage)")]
    basin = dp.huc4_boundary_gpkg(cfg)
    if basin.exists():
        gpd.read_file(basin, layer="huc4").boundary.plot(
            ax=ax, color="#ffe066", lw=1.4, zorder=3)
        handles.append(mpatches.Patch(
            edgecolor="#ffe066", facecolor="none",
            label=f"HUC-4 {cfg['project']['huc4']} boundary (clipped)"))

    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"{cfg['project']['name']} {cfg['project']['version']}: "
                 f"merged topobathy, 1/{scale} of full resolution\n"
                 "NAVD88 m, cmocean topo centred at 0", fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02,
                      extend="both", label="elevation (m, NAVD88)")
    cb.ax.axhline(norm(0.0), color="k", lw=0.8)
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s  [%.1f, %.1f] m", out_png, lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zoom", type=int, default=12,
                    help="Esri imagery zoom level (default 12)")
    ap.add_argument("--max-px", type=int, default=2600,
                    help="longest side of the downsampled mosaic")
    ap.add_argument("--skip-imagery", action="store_true")
    args = ap.parse_args()
    dp.setup_logging()
    cfg = dp.load_config()

    qdir = dp.path(cfg["qa"]["quicklook_dir"])
    qdir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg['project']['name']}_{cfg['project']['version']}"

    if not args.skip_imagery:
        plot_tiles_on_imagery(cfg, qdir / f"{stem}_tiles_on_imagery.png",
                              args.zoom)
    plot_mosaic(cfg, qdir / f"{stem}_mosaic_topo.png", args.max_px)


if __name__ == "__main__":
    main()
