#!/usr/bin/env python
"""Interactive QA viewer for the output tile set.

Serves a Leaflet map of the output tile lattice over Esri World Imagery
(replacing the fixed-view <name>_tiles_on_imagery.png with a real pannable
basemap), with a depth (elevation) layer that can be toggled on top of the
imagery -- rendered as real georeferenced map tiles read on demand from
data/final/dewatershed19_2026v1.vrt, not the matplotlib mosaic_topo.png
(whose baked-in title/colorbar/axes made it misregister badly as a plain
image overlay). Clicking a tile renders that tile's hillshade, provenance,
and depth images fresh from data/final/tiles/<name>.tif and
<name>_source.tif -- not from the pipeline's precomputed quicklook PNGs, so
resolution and framing aren't tied to whatever scripts/10_qa_checks.py
happened to write.

Read-only and inspection-only: nothing here feeds the product, and it never
writes into data/ or config/ (rendered images are cached under
data/final/quicklooks/.viewer_cache/, next to the pipeline's own quicklook
cache directory).

Usage: python scripts/viewer.py [--port 8765] [--host 127.0.0.1]
       python scripts/viewer.py --merge-config config/merge_region02.yaml
       python scripts/viewer.py --burn-config config/burn_region02.yaml

--burn-config views the channel-burned product (scripts/burn_region.py,
e.g. config/burn_region02.yaml) the same way, and adds a burn-depth layer:
original (merged) minus burned elevation, from the per-tile
<name>_burndepth.tif sidecars (which equal that difference exactly, 0
where nothing was lowered) -- and, if scripts/burn_gauge_compare.py has
written <root>/qa/gauge_compare.geojson, a USGS gauge layer (burn model
depth vs observed mean depth at mean flow) -- and, if `burn_region.py
sources` has written <root>/qa/depth_source/<huc4>.gpkg, a channel layer
coloured by where each reach's burn depth came from (tier 1 Zarrabi /
curve fallback / tier 1b / tier-2 regional curve; click a channel for its
D, width, drainage area, curve).

--merge-config views a region merge (scripts/merge_region.py) instead of
this checkout's basin product: tiles, mosaic VRT and QA report come from the
merge's output root, the boundary layer shows every member HUC-4, and the
render cache lives under <root>/quicklooks/.viewer_cache/.

On a remote host (e.g. a NERSC login node), the server only binds to
127.0.0.1 by default -- tunnel to it from your laptop with:
    ssh -L 8765:localhost:8765 <host>
then open http://localhost:8765/ locally.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import dem_pipeline as dp
from dem_pipeline.grid import parse_tile_name, tile_name

log = logging.getLogger("viewer")

STATIC_DIR = Path(__file__).parent / "viewer_static"
MIME = {".html": "text/html; charset=utf-8", ".js": "application/javascript",
        ".css": "text/css", ".png": "image/png", ".json": "application/json"}

# Same palette as dem_pipeline.qa.provenance_quicklook, so a tile's rendered
# source image reads consistently with every _src.png the pipeline already
# produced.
SOURCE_COLORS_BY_NAME = {
    "nodata": (255, 255, 255), "s1m": (198, 148, 74),
    "dem1m_projects": (232, 196, 128), "dem13": (245, 226, 180),
    "ehydro": (26, 92, 158), "eaarlb": (70, 160, 190),
    "cudem19": (140, 200, 214), "coned": (190, 225, 232),
    "nos_grids": (96, 72, 180),
    "filled": (220, 40, 40),
    "burn": (255, 0, 200),
}

# Burn-depth layer: log colour scale (most cuts are tens of cm, culvert cuts
# reach tens of m); below BURN_MIN_M is transparent.
BURN_LO_M, BURN_HI_M, BURN_MIN_M = 0.05, 10.0, 0.01


# Set by --merge-config: a config-shaped dict whose paths are absolute
# (dp.path() of an absolute path returns it unchanged) plus the merge's
# tile-list / boundary files. None = this checkout's basin product.
_MERGE = None


def _merge_view(cfg_path: str) -> dict:
    from dem_pipeline import merge

    mcfg = merge.load_merge_config(cfg_path)
    root = merge.output_root(mcfg)
    p = mcfg["project"]
    paths = merge.prepared_paths(mcfg)
    return {
        "cfg": {
            "project": {"name": p["name"], "version": p["version"]},
            "output": {"tile_dir": str(merge.tiles_dir(mcfg)),
                       "vrt": str(root / f"{p['name']}_{p['version']}.vrt")},
            "qa": {"dir": str(root / "qa"),
                   "quicklook_dir": str(root / "quicklooks")},
            "compositing": {"source_ids": merge.merged_source_ids(
                merge.load_basins(mcfg))},
        },
        "tiles_geojson": paths["tiles"],
        "boundary": (paths["region"], "basins"),
    }


def _burn_view(cfg_path: str) -> dict:
    """The burned product's view: the merge's tile list/boundary, the burn's
    output root, and the merged source ids plus the burn id."""
    from dem_pipeline import burn

    b = burn.load_burn_config(cfg_path)
    root = burn.out_root(b)
    p = b["project"]
    view = _merge_view(b["_merge"]["_path"])
    ids = {**view["cfg"]["compositing"]["source_ids"],
           "burn": int(b["burn"]["burn_id"])}
    view["cfg"] = {
        "project": {"name": p["name"], "version": p["version"]},
        "output": {"tile_dir": str(root / "tiles"),
                   "vrt": str(root / f"{p['name']}_{p['version']}.vrt")},
        "qa": {"dir": str(root / "qa"), "quicklook_dir": str(root / "quicklooks")},
        "compositing": {"source_ids": ids},
    }
    view["burn"] = True
    view["depth_source"] = [burn.depth_source_path(b, str(bas["huc4"]))
                            for bas in b["basins"]]
    view["curves"] = [burn.curve_label(c) + f" ({c['c']}·DA^{c['e']})"
                      for c in b["curves"]]
    view["factors"] = {"t1": b["burn"]["t1_zarrabi_factor"], "t2": b["burn"]["t2_factor"]}
    return view


def _is_burn() -> bool:
    return bool(_MERGE and _MERGE.get("burn"))


def _cfg():
    return _MERGE["cfg"] if _MERGE else dp.load_config()


def _tiles_geojson_path() -> Path:
    return _MERGE["tiles_geojson"] if _MERGE else dp.output_tiles_geojson(_cfg())


def _boundary_source() -> tuple[Path, str]:
    return _MERGE["boundary"] if _MERGE else (dp.huc4_boundary_gpkg(), "huc4")


def _cache_dir() -> Path:
    d = dp.path(_cfg()["qa"]["quicklook_dir"], ".viewer_cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _burndepth_path(canon: str) -> Path:
    cfg = _cfg()
    stem = f"{cfg['project']['name']}_{canon}_{cfg['project']['version']}"
    return dp.path(cfg["output"]["tile_dir"]) / f"{stem}_burndepth.tif"


def _tile_paths(name: str) -> tuple[str, Path, Path]:
    """(canonical tile name, elevation tif, provenance tif) for a tile name.

    Re-normalizes through parse/tile_name rather than trusting the input
    string directly, so a malformed or path-traversal-flavored name can
    never reach a filesystem join.
    """
    north, west = parse_tile_name(name)  # raises ValueError if unparseable
    canon = tile_name(north, west)
    cfg = _cfg()
    stem = f"{cfg['project']['name']}_{canon}_{cfg['project']['version']}"
    tile_dir = dp.path(cfg["output"]["tile_dir"])
    return canon, tile_dir / f"{stem}.tif", tile_dir / f"{stem}_source.tif"


def _cached_png(cache_key: str, render) -> bytes:
    p = _cache_dir() / f"{cache_key}.png"
    if p.exists():
        return p.read_bytes()
    data = render()
    tmp = p.with_suffix(".tmp.png")
    tmp.write_bytes(data)
    tmp.rename(p)
    return data


def _decimated_read(path: Path, max_px: int) -> tuple[np.ndarray, float | None, float]:
    """Band 1 decimated to ~2x max_px, leaning on the tile's own overviews.

    Full 8100x8100 native resolution buys no visible detail once the result
    is only ever shown at max_px -- and reading via out_shape here lets GDAL
    pull from a lower COG overview level instead of inflating the full
    DEFLATE-compressed native band on every request.

    Returns (array as f4, nodata value, decimated pixel size in degrees).
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        compute_px = min(max(src.width, src.height), max(2 * max_px, 1600))
        factor = max(src.width, src.height) / compute_px
        out_shape = (max(1, round(src.height / factor)), max(1, round(src.width / factor)))
        arr = src.read(1, out_shape=out_shape, resampling=Resampling.average).astype("f4")
        nodata = src.nodata
        px_deg = abs(src.transform.a) * (src.width / out_shape[1])
    return arr, nodata, px_deg


def render_hillshade(tile_path: Path, max_px: int) -> bytes:
    """Hillshade computed at (decimated) resolution, shaded before resizing.

    Shading first and downsampling after (rather than the reverse) keeps
    ridge/channel detail that a pre-decimated gradient would blur away --
    the same order gdaldem-then-shrink uses in
    dem_pipeline.qa.hillshade_quicklook, just done in-process.
    """
    from PIL import Image

    z, nodata, px_deg = _decimated_read(tile_path, max_px)
    if nodata is not None:
        z[z == nodata] = np.nan
    cellsize_m = px_deg * 111320.0  # geographic-CRS scale, matches qa.py's -s 111120
    gy, gx = np.gradient(z, cellsize_m)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = math.radians(315.0), math.radians(45.0)
    shaded = (np.sin(alt) * np.sin(slope)
              + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))

    alpha = np.where(np.isnan(z), 0, 255).astype("u1")
    gray = np.nan_to_num(np.clip(shaded, 0, 1) * 255, nan=0).astype("u1")
    img = Image.fromarray(np.dstack([gray, gray, gray, alpha]), "RGBA")

    scale = max(1, round(max(img.size) / max_px))
    if scale > 1:
        img = img.resize((max(1, img.width // scale), max(1, img.height // scale)),
                         Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def render_source(source_path: Path, max_px: int) -> bytes:
    """Provenance raster colorized with the pipeline's source palette."""
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling

    ids = _cfg()["compositing"]["source_ids"]
    colors = {ids[name]: rgb for name, rgb in SOURCE_COLORS_BY_NAME.items()
              if name in ids}

    with rasterio.open(source_path) as src:
        scale = max(1, math.ceil(max(src.height, src.width) / max_px))
        arr = src.read(1, out_shape=(src.height // scale, src.width // scale),
                       resampling=Resampling.nearest)

    rgb = np.zeros(arr.shape + (3,), dtype="u1")
    for value, colour in colors.items():
        rgb[arr == value] = colour
    img = Image.fromarray(np.dstack([rgb, np.full(arr.shape, 255, "u1")]), "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@lru_cache(maxsize=1)
def compute_depth_norm() -> tuple[float, float]:
    """Basin-wide (lo, hi) elevation bounds for the depth color scale.

    Computed once from a downsampled read of the basin VRT (same method as
    the old plot_mosaic()'s percentile clipping) and cached for the life of
    the server: every depth tile and every per-tile depth image must share
    one color scale, or adjacent map tiles and different tiles' panels would
    each stretch their own contrast and show visible seams / incomparable
    colors.
    """
    import rasterio
    from rasterio.enums import Resampling

    vrt = dp.path(_cfg()["output"]["vrt"])
    with rasterio.open(vrt) as src:
        scale = max(1, math.ceil(max(src.width, src.height) / 1500))
        out_h, out_w = src.height // scale, src.width // scale
        z = src.read(1, out_shape=(1, out_h, out_w), resampling=Resampling.average,
                     masked=True)
    z = np.ma.masked_invalid(z)
    lo = float(np.percentile(z.compressed(), 0.5)) if z.count() else -1.0
    hi = float(np.percentile(z.compressed(), 99.5)) if z.count() else 1.0
    return min(lo, -1.0), max(hi, 1.0)


def _colorize_depth(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """RGBA uint8 array: cmocean `topo` centered at NAVD88 zero, basin-wide scale."""
    import cmocean
    from matplotlib.colors import TwoSlopeNorm

    lo, hi = compute_depth_norm()
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    rgba = cmocean.cm.topo(norm(np.where(valid, z, 0.0)))
    rgb = (rgba[..., :3] * 255).astype("u1")
    alpha = np.where(valid, 255, 0).astype("u1")
    return np.dstack([rgb, alpha])


def render_depth(tile_path: Path, max_px: int) -> bytes:
    """One tile's elevation, colorized -- the panel equivalent of plot_mosaic."""
    from PIL import Image

    z, nodata, _ = _decimated_read(tile_path, max_px)
    valid = np.isfinite(z) if nodata is None else (z != nodata)
    img = Image.fromarray(_colorize_depth(z, valid), "RGBA")
    scale = max(1, round(max(img.size) / max_px))
    if scale > 1:
        img = img.resize((max(1, img.width // scale), max(1, img.height // scale)),
                         Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _maxpool(a: np.ndarray, f: int) -> np.ndarray:
    """f x f block maximum (NaN-aware): a few-pixel-wide channel stays visible
    when shrunk, where averaging with the surrounding zeros would fade it."""
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return np.nanmax(a[:h, :w].reshape(h // f, f, w // f, f), axis=(1, 3))


def _colorize_burn(d: np.ndarray) -> np.ndarray:
    import matplotlib
    from matplotlib.colors import LogNorm

    show = np.isfinite(d) & (d >= BURN_MIN_M)
    norm = LogNorm(vmin=BURN_LO_M, vmax=BURN_HI_M, clip=True)
    rgba = matplotlib.colormaps["plasma"](norm(np.where(show, d, BURN_LO_M)))
    rgb = (rgba[..., :3] * 255).astype("u1")
    return np.dstack([rgb, np.where(show, 255, 0).astype("u1")])


def render_burndepth(path: Path, max_px: int) -> bytes:
    """One tile's burn depth (original - burned), block-max shrunk."""
    import rasterio
    from PIL import Image

    with rasterio.open(path) as src:
        d = src.read(1).astype("f4")
        if src.nodata is not None:
            d[d == src.nodata] = np.nan
    f = max(1, math.ceil(max(d.shape) / max_px))
    if f > 1:
        d = _maxpool(d, f)
    img = Image.fromarray(_colorize_burn(d), "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _burndepth_mosaic_vrt() -> Path | None:
    """Mosaic of the per-tile burn-depth sidecars (cache dir, rebuilt when
    the sidecar set changes) -- same approach as _source_mosaic_vrt. Tiles
    with nothing lowered have no sidecar and read as nodata (= 0 m)."""
    import subprocess

    cfg = _cfg()
    tifs = dp.product_tifs(dp.path(cfg["output"]["tile_dir"]),
                           f"{cfg['project']['name']}_*_burndepth.tif")
    if not tifs:
        return None
    sig = f"{len(tifs)}_{max(p.stat().st_mtime_ns for p in tifs)}"
    vrt = _cache_dir() / f"burndepth_mosaic_{sig}.vrt"
    if not vrt.exists():
        listfile = vrt.with_suffix(".inputs.txt")
        listfile.write_text("\n".join(str(p) for p in tifs) + "\n")
        subprocess.run(["gdalbuildvrt", "-q", "-overwrite", "-input_file_list",
                        str(listfile), str(vrt)], check=True)
    return vrt


def render_burndepth_tile(z: int, x: int, y: int, tile_px: int = 256) -> bytes:
    """XYZ tile of burn depth: read 4x oversampled (from the sidecars' own
    overviews) and block-max shrunk, so channels survive zooming out. At low
    zoom the overviews themselves are averages, so depths read shallower."""
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    vrt = _burndepth_mosaic_vrt()
    if vrt is None:
        return _transparent_png(tile_px)
    west, south, east, north = _slippy_tile_bounds(z, x, y)
    over = 4
    with rasterio.open(vrt) as src:
        b = src.bounds
        if east <= b.left or west >= b.right or north <= b.bottom or south >= b.top:
            return _transparent_png(tile_px)
        nodata = src.nodata
        window = from_bounds(west, south, east, north, transform=src.transform)
        arr = src.read(1, window=window, out_shape=(tile_px * over, tile_px * over),
                       resampling=Resampling.average, boundless=True,
                       fill_value=nodata if nodata is not None else np.nan).astype("f4")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    if not np.any(arr >= BURN_MIN_M):
        return _transparent_png(tile_px)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks
        d = _maxpool(arr, over)
    img = Image.fromarray(_colorize_burn(d), "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def burndepth_legend() -> dict:
    import matplotlib

    cm = matplotlib.colormaps["plasma"]
    stops = [[round(p * 100, 1), "#%02x%02x%02x" % tuple(int(c * 255) for c in cm(p)[:3])]
             for p in np.linspace(0, 1, 11)]
    return {"lo": BURN_LO_M, "hi": BURN_HI_M, "stops": stops}


# ---- channel depth-source layer (burned product only) --------------------
#
# Vector layer from <root>/qa/depth_source/<huc4>.gpkg (scripts/burn_region.py
# sources): every network reach coloured by where its burn depth D came
# from. Drawn server-side into XYZ PNGs (~400k reaches region-wide: too many
# for GeoJSON in the browser), small streams dropped at low zoom; clicks are
# answered by /api/reach_at.

DS_COLORS = {
    "t1_zarrabi": (0, 229, 255), "t1b_zarrabi": (47, 123, 255),
    "t1_curve": (255, 45, 85),
    "not_routed": (255, 255, 255), "lake": (130, 130, 140), "tidal": (90, 90, 100),
}
# tier-2 regional curves, by position in the burn config's curves list
DS_CURVE_COLORS = [(255, 214, 10), (255, 149, 0), (163, 230, 53), (52, 211, 153),
                   (233, 213, 255), (255, 120, 200), (180, 140, 90)]
DS_LABELS = {      # {t1} / {t2}: the burn config's depth factors
    "t1_zarrabi": "tier 1 river: {t1} × Zarrabi mean-flow depth",
    "t1b_zarrabi": "tier 1b (no river polygon): {t1} × Zarrabi",
    "t1_curve": "tier 1 river: no Zarrabi match → {t2} × regional curve",
    "not_routed": "not burned: no path routed",
    "lake": "not burned: lake / reservoir",
    "tidal": "not burned: tidal (bathymetry)",
}
_DS = {"lock": __import__("threading").Lock(), "data": None, "sig": None}


def _ds_label(key: str) -> str:
    f = _MERGE["factors"]
    if key.startswith("t2_curve:"):
        return f"tier 2: {f['t2']} × curve {_MERGE['curves'][int(key.split(':')[1])]}"
    return DS_LABELS.get(key, key).format(**f)


def _ds_files() -> list[Path]:
    return [p for p in (_MERGE or {}).get("depth_source", []) if p.exists()]


def _ds_key(src: str, curve_idx: int) -> str:
    return f"t2_curve:{curve_idx}" if src == "t2_curve" else src


def _ds_color(key: str) -> tuple:
    if key.startswith("t2_curve:"):
        return DS_CURVE_COLORS[int(key.split(":")[1]) % len(DS_CURVE_COLORS)]
    return DS_COLORS.get(key, (255, 0, 255))


def _ds_data():
    """All basins' reaches (loaded once; reloaded if a gpkg changes)."""
    files = _ds_files()
    sig = tuple((str(p), p.stat().st_mtime_ns) for p in files)
    with _DS["lock"]:
        if _DS["data"] is None or _DS["sig"] != sig:
            import geopandas as gpd
            import pandas as pd
            import shapely
            if not files:
                return None
            g = pd.concat([gpd.read_file(p) for p in files], ignore_index=True)
            g = gpd.GeoDataFrame(g, geometry="geometry", crs=4269)
            g["key"] = [_ds_key(a, int(b)) for a, b in zip(g.src, g.curve_idx)]
            burned = ~g.src.isin(["lake", "tidal", "not_routed"])
            # draw order: not-burned first, then small to large
            g = g.assign(_o=burned.astype(int) * 1e7 + g.DA).sort_values("_o")
            g = g.reset_index(drop=True)
            _DS["data"] = {"g": g, "tree": shapely.STRtree(g.geometry.values),
                           "burned": (~g.src.isin(["lake", "tidal", "not_routed"])).to_numpy(),
                           "da": g.DA.to_numpy()}
            _DS["sig"] = sig
            log.info("depth-source layer: %d reaches from %d files", len(g), len(files))
        return _DS["data"]


def _ds_da_min(z: int) -> float:
    return {8: 300.0, 9: 100.0, 10: 30.0, 11: 10.0, 12: 3.0}.get(
        z, 1000.0 if z < 8 else 0.0)


def render_depthsrc_tile(z: int, x: int, y: int, tile_px: int = 256) -> bytes:
    import shapely
    from PIL import Image, ImageDraw

    d = _ds_data()
    west, south, east, north = _slippy_tile_bounds(z, x, y)
    pad = (east - west) * 0.05
    idx = d["tree"].query(shapely.box(west - pad, south - pad, east + pad, north + pad))
    idx = np.sort(idx[d["da"][idx] >= _ds_da_min(z)])     # sorted = draw order
    if not len(idx):
        return _transparent_png(tile_px)
    ss = 2                                     # supersample, then shrink
    n = 2.0 ** z * 256 * ss
    g = d["g"]
    coords, part = shapely.get_coordinates(g.geometry.values[idx], return_index=True)
    px = (coords[:, 0] + 180.0) / 360.0 * n - x * 256 * ss
    lat = np.radians(np.clip(coords[:, 1], -85, 85))
    py = (1 - np.log(np.tan(lat) + 1 / np.cos(lat)) / math.pi) / 2 * n - y * 256 * ss
    img = Image.new("RGBA", (tile_px * ss, tile_px * ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    base = 1.0 if z <= 10 else 1.5 if z <= 12 else 2.5 if z <= 14 else 3.5
    cuts = np.flatnonzero(np.diff(part)) + 1
    for j, (a, b) in enumerate(zip(np.r_[0, cuts], np.r_[cuts, len(part)])):
        k = idx[j]
        if b - a < 2:
            continue
        w = base * (1.0 if d["burned"][k] else 0.6) * (1.6 if d["da"][k] >= 100 else 1.0)
        draw.line(list(zip(px[a:b], py[a:b])), fill=_ds_color(g.key.iat[k]) + (255,),
                  width=max(1, int(round(w * ss))), joint="curve")
    img = img.resize((tile_px, tile_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def depthsrc_legend() -> dict:
    d = _ds_data()
    counts = d["g"].key.value_counts().to_dict()
    curves = _MERGE.get("curves", [])
    order = ["t1_zarrabi", "t1b_zarrabi", "t1_curve"] + \
        [f"t2_curve:{i}" for i in range(len(curves))] + ["not_routed", "lake", "tidal"]
    out = []
    for k in order:
        out.append({"key": k, "label": _ds_label(k), "color": "#%02x%02x%02x" % _ds_color(k),
                    "count": int(counts.get(k, 0))})
    return {"entries": out}


def reach_at(lat: float, lon: float, z: int) -> dict | None:
    """Properties of the visible reach nearest a click (within ~8 px)."""
    import shapely

    d = _ds_data()
    tol = 8 * 360.0 / (256 * 2.0 ** z)               # degrees of longitude
    cl = math.cos(math.radians(lat))
    pt = shapely.Point(lon, lat)
    idx = d["tree"].query(shapely.box(lon - tol, lat - tol * cl, lon + tol, lat + tol * cl))
    idx = idx[d["da"][idx] >= _ds_da_min(z)]
    if not len(idx):
        return None
    geoms = d["g"].geometry.values[idx]
    # distance in (scaled-lon, lat) degrees: shrink longitudes by cos(lat)
    sg = shapely.transform(geoms, lambda c: c * [cl, 1.0])
    dist = shapely.distance(sg, shapely.Point(lon * cl, lat))
    j = int(np.argmin(dist))
    if dist[j] > tol * cl:
        return None
    r = d["g"].iloc[idx[j]]
    out = {}
    for k in ("huc4", "NHDPlusID", "name", "src", "key", "tier", "D", "W", "DA",
              "TotDASqKm", "da_from_div", "zarrabi_mf_depth", "curve", "physio",
              "conduit", "max_cut"):
        v = r[k]
        if isinstance(v, (np.floating, float)):
            v = None if not np.isfinite(v) else round(float(v), 3)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        out[k] = v
    out["NHDPlusID"] = str(out["NHDPlusID"])      # > 2^53: keep exact in JS
    out["label"] = _ds_label(r.key)
    return out


@lru_cache(maxsize=4)
def _transparent_png(size: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (size, size), (0, 0, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


def _slippy_tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(west, south, east, north) in degrees for a standard XYZ/Web-Mercator tile."""
    n = 2.0 ** z

    def num2deg(xt, yt):
        lon = xt / n * 360.0 - 180.0
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yt / n))))
        return lon, lat

    west, north = num2deg(x, y)
    east, south = num2deg(x + 1, y + 1)
    return west, south, east, north


def _source_tifs() -> list[Path]:
    cfg = _cfg()
    tile_dir = dp.path(cfg["output"]["tile_dir"])
    return dp.product_tifs(tile_dir, f"{cfg['project']['name']}_*_source.tif")


# Real source ids run 0-99 (see config/sources.yaml's source_ids), so 255 is
# safe to use as the mosaic VRT's own NoData: it marks pixels not covered by
# any per-tile provenance raster (a gap in the lattice) without colliding
# with id 0, which is a legitimate "nothing composited here" category that
# can appear *inside* a real tile and must still render (as the per-tile
# panel already does) rather than read as transparent.
_MOSAIC_NODATA = 255


def _source_mosaic_vrt() -> Path | None:
    """Basin-wide provenance mosaic, lazily built from per-tile *_source.tif.

    The pipeline only ever produces per-tile provenance rasters -- there's no
    stage 09 equivalent for provenance the way there is for elevation
    (data/final/<name>.vrt) -- so the viewer builds one itself, the same
    "mosaic VRT + windowed reads" approach stage 09 uses, cached under its
    own cache dir (never data/final/tiles/) since this is an inspection
    artifact, not a pipeline product. Rebuilt whenever the tile count or
    newest mtime changes, so a run that adds/rewrites tiles is picked up on
    the next request without restarting the server.
    """
    import subprocess

    tifs = _source_tifs()
    if not tifs:
        return None
    sig = f"{len(tifs)}_{max(p.stat().st_mtime_ns for p in tifs)}"
    vrt = _cache_dir() / f"source_mosaic_{sig}.vrt"
    if not vrt.exists():
        listfile = vrt.with_suffix(".inputs.txt")
        listfile.write_text("\n".join(str(p) for p in tifs) + "\n")
        subprocess.run(["gdalbuildvrt", "-q", "-overwrite", "-vrtnodata",
                        str(_MOSAIC_NODATA), "-input_file_list", str(listfile),
                        str(vrt)], check=True)
    return vrt


def render_source_tile(z: int, x: int, y: int, tile_px: int = 256) -> bytes:
    """One 256x256 XYZ map tile of basin source provenance.

    The provenance analogue of render_depth_tile: windowed straight from the
    lazily-built mosaic VRT rather than a basin-scale intermediate, colorized
    with the same palette as the per-tile source.png so the basin overlay and
    a selected tile's provenance panel read consistently. Resampling is
    nearest, not average -- these are categorical ids, and averaging them
    would blend e.g. s1m (1) and cudem19 (13) into a meaningless value.
    """
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    vrt = _source_mosaic_vrt()
    if vrt is None:
        return _transparent_png(tile_px)

    ids = _cfg()["compositing"]["source_ids"]
    colors = {ids[name]: rgb for name, rgb in SOURCE_COLORS_BY_NAME.items()
              if name in ids}

    west, south, east, north = _slippy_tile_bounds(z, x, y)
    with rasterio.open(vrt) as src:
        b = src.bounds
        if east <= b.left or west >= b.right or north <= b.bottom or south >= b.top:
            return _transparent_png(tile_px)
        nodata = src.nodata
        window = from_bounds(west, south, east, north, transform=src.transform)
        arr = src.read(1, window=window, out_shape=(tile_px, tile_px),
                       resampling=Resampling.nearest, boundless=True,
                       fill_value=nodata)
    valid = arr != nodata
    if not np.any(valid):
        return _transparent_png(tile_px)
    rgb = np.zeros(arr.shape + (3,), dtype="u1")
    for value, colour in colors.items():
        rgb[arr == value] = colour
    alpha = np.where(valid, 255, 0).astype("u1")
    img = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def render_depth_tile(z: int, x: int, y: int, tile_px: int = 256) -> bytes:
    """One 256x256 XYZ map tile of basin elevation, windowed straight from the VRT.

    This replaces overlaying the matplotlib mosaic_topo.png as a single image:
    that PNG's title/colorbar/axes padding meant its pixel bounds didn't match
    its geographic bounds, so it never registered cleanly over the imagery.
    Reading a real per-tile window off the VRT (same "warp windows in memory,
    no basin-scale intermediate" approach the pipeline itself uses) gives a
    correctly georeferenced layer at whatever zoom the map is at.
    """
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    vrt = dp.path(_cfg()["output"]["vrt"])
    west, south, east, north = _slippy_tile_bounds(z, x, y)
    with rasterio.open(vrt) as src:
        b = src.bounds
        if east <= b.left or west >= b.right or north <= b.bottom or south >= b.top:
            return _transparent_png(tile_px)
        nodata = src.nodata
        window = from_bounds(west, south, east, north, transform=src.transform)
        arr = src.read(1, window=window, out_shape=(tile_px, tile_px),
                       resampling=Resampling.average, boundless=True,
                       fill_value=nodata if nodata is not None else np.nan).astype("f4")
    valid = np.isfinite(arr) if nodata is None else (arr != nodata)
    if not np.any(valid):
        return _transparent_png(tile_px)
    img = Image.fromarray(_colorize_depth(arr, valid), "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def depth_legend() -> dict:
    """Color-scale stops for the depth layer's legend, sampled from cmocean topo.

    TwoSlopeNorm always maps vcenter to 0.5 regardless of how asymmetric
    (lo, hi) are, so sampling the colormap uniformly over [0, 1] and labeling
    the 0/50/100% stops with lo/0/hi is exactly the norm's own mapping --
    no need to invert it.
    """
    import cmocean

    lo, hi = compute_depth_norm()
    stops = []
    for i in range(11):
        p = i / 10
        r, g, b, _ = cmocean.cm.topo(p)
        stops.append([round(p * 100, 1), "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))])
    return {"lo": lo, "hi": hi, "stops": stops}


def build_tiles_geojson() -> dict:
    """The configured HUC-4's output tile list, enriched with coverage/QA flags."""
    cfg = _cfg()
    with open(_tiles_geojson_path()) as f:
        fc = json.load(f)

    coverage, land_overlap = {}, {}
    seam_flags, size_mismatches = {}, {}
    qa_path = dp.path(cfg["qa"]["dir"]) / "qa_report.json"
    if qa_path.exists():
        with open(qa_path) as f:
            checks = json.load(f)
        if isinstance(checks, dict):        # merge report: {"checks": [...]}
            checks = checks.get("checks", [])
        by_name = {c.get("check"): c for c in checks}
        coverage = by_name.get("coverage", {}).get("tiles", {})
        land_overlap = by_name.get("cudem_land_overlap", {}).get("tiles", {})
        seams = by_name.get("seams", {})
        for edge_dict, bucket in ((seams.get("flagged", {}), seam_flags),
                                  (seams.get("size_mismatches", {}), size_mismatches)):
            for edge, val in edge_dict.items():
                for t in edge.split("|"):
                    bucket.setdefault(t, []).append({"edge": edge, "value": val})

    tile_dir = dp.path(cfg["output"]["tile_dir"])
    stem_fmt = f"{cfg['project']['name']}_{{}}_{cfg['project']['version']}.tif"
    for feat in fc["features"]:
        name = feat["properties"]["tile"]
        feat["properties"]["coverage"] = coverage.get(name, {})
        feat["properties"]["land_overlap"] = land_overlap.get(name)
        feat["properties"]["seam_flags"] = seam_flags.get(name, [])
        feat["properties"]["size_mismatches"] = size_mismatches.get(name, [])
        feat["properties"]["has_final_tile"] = (tile_dir / stem_fmt.format(name)).exists()
    return fc


def source_legend() -> dict:
    ids = _cfg()["compositing"]["source_ids"]
    return {name: {"id": ids[name], "color": "#%02x%02x%02x" % rgb}
            for name, rgb in SOURCE_COLORS_BY_NAME.items() if name in ids}


@lru_cache(maxsize=1)
def basin_boundary_geojson() -> dict:
    """The configured HUC-4's basin boundary -- stage 01's dissolved WBD polygon.

    Served straight from its native NAD83 (EPSG:4269) coordinates, same as
    the output tile list: nothing in this app reprojects to WGS84, and the
    NAD83/WGS84 difference is sub-meter here, invisible at map scale. Cached
    for the server's lifetime -- this file is a stage 01 input, never
    touched again once the pipeline has run.
    """
    import geopandas as gpd

    gpkg, layer = _boundary_source()
    gdf = gpd.read_file(gpkg, layer=layer)
    return json.loads(gdf.to_json())


class Handler(BaseHTTPRequestHandler):
    server_version = "DEViewer/1.0"

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status=200):
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _send_static(self, rel: str):
        p = (STATIC_DIR / rel).resolve()
        if STATIC_DIR.resolve() not in p.parents and p != STATIC_DIR.resolve():
            self._send_json({"error": "not found"}, 404)
            return
        if not p.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        self._send(200, p.read_bytes(), MIME.get(p.suffix, "application/octet-stream"))

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path == "/" or path == "/index.html":
                self._send_static("index.html")
            elif path in ("/app.js", "/style.css"):
                self._send_static(path.lstrip("/"))
            elif path == "/api/meta":
                cfg = _cfg()
                self._send_json({"name": cfg["project"]["name"],
                                 "version": cfg["project"]["version"],
                                 "legend": source_legend(),
                                 "burn": _is_burn()})
            elif path == "/api/tiles.geojson":
                self._send_json(build_tiles_geojson())
            elif path == "/api/basin_boundary.geojson":
                gpkg, _ = _boundary_source()
                if not gpkg.exists():
                    self._send_json({"error": "boundary not found"}, 404)
                else:
                    self._send_json(basin_boundary_geojson())
            elif path == "/api/depth_legend":
                cfg = _cfg()
                if not dp.path(cfg["output"]["vrt"]).exists():
                    self._send_json({"error": "vrt not found"}, 404)
                else:
                    self._send_json(depth_legend())
            elif path == "/api/source_legend":
                if _source_mosaic_vrt() is None:
                    self._send_json({"error": "no provenance rasters found"}, 404)
                else:
                    self._send_json(source_legend())
            elif path == "/api/burndepth_legend":
                if not _is_burn() or _burndepth_mosaic_vrt() is None:
                    self._send_json({"error": "no burn-depth rasters"}, 404)
                else:
                    self._send_json(burndepth_legend())
            elif path == "/api/gauges.geojson":
                gp = dp.path(_cfg()["qa"]["dir"]) / "gauge_compare.geojson"
                if not (_is_burn() and gp.exists()):
                    self._send_json({"error": "run scripts/burn_gauge_compare.py first"}, 404)
                else:
                    self._send(200, gp.read_bytes(), "application/json")
            elif path == "/api/depthsrc_legend":
                if not (_is_burn() and _ds_files()):
                    self._send_json({"error": "run scripts/burn_region.py sources first"}, 404)
                else:
                    self._send_json(depthsrc_legend())
            elif path == "/api/reach_at":
                if not (_is_burn() and _ds_files()):
                    self._send_json({"error": "no depth-source layer"}, 404)
                else:
                    hit = reach_at(float(qs["lat"][0]), float(qs["lon"][0]),
                                   int(qs["z"][0]))
                    self._send_json(hit or {})
            elif path.startswith("/api/depthsrc_tile/"):
                self._handle_depthsrc_tile(path)
            elif path.startswith("/api/burndepth_tile/"):
                self._handle_burndepth_tile(path)
            elif path.startswith("/api/depth_tile/"):
                self._handle_depth_tile(path)
            elif path.startswith("/api/source_tile/"):
                self._handle_source_tile(path)
            elif path.startswith("/api/tile/"):
                self._handle_tile(path, qs)
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            log.exception("request failed: %s", path)
            self._send_json({"error": str(exc)}, 500)

    def _handle_tile(self, path: str, qs: dict):
        # /api/tile/<name>/hillshade.png or /api/tile/<name>/source.png
        parts = path.split("/")
        if len(parts) != 5:
            self._send_json({"error": "bad path"}, 404)
            return
        _, _, _, raw_name, kind = parts
        max_px = max(200, min(4000, int(qs.get("max", ["1400"])[0])))
        canon, tile_path, source_path = _tile_paths(raw_name)  # validates raw_name

        if kind == "hillshade.png":
            if not tile_path.exists():
                self._send_json({"error": f"no final tile for {canon}"}, 404)
                return
            key = f"{canon}_hs_{max_px}_{tile_path.stat().st_mtime_ns}"
            png = _cached_png(hashlib.sha1(key.encode()).hexdigest(),
                              lambda: render_hillshade(tile_path, max_px))
            self._send(200, png, "image/png")
        elif kind == "source.png":
            if not source_path.exists():
                self._send_json({"error": f"no provenance raster for {canon}"}, 404)
                return
            key = f"{canon}_src_{max_px}_{source_path.stat().st_mtime_ns}"
            png = _cached_png(hashlib.sha1(key.encode()).hexdigest(),
                              lambda: render_source(source_path, max_px))
            self._send(200, png, "image/png")
        elif kind == "depth.png":
            if not tile_path.exists():
                self._send_json({"error": f"no final tile for {canon}"}, 404)
                return
            key = f"{canon}_depth_{max_px}_{tile_path.stat().st_mtime_ns}"
            png = _cached_png(hashlib.sha1(key.encode()).hexdigest(),
                              lambda: render_depth(tile_path, max_px))
            self._send(200, png, "image/png")
        elif kind == "burndepth.png":
            bd = _burndepth_path(canon)
            if not (_is_burn() and bd.exists()):
                self._send_json({"error": f"nothing burned on {canon}"}, 404)
                return
            key = f"{canon}_burn_{max_px}_{bd.stat().st_mtime_ns}"
            png = _cached_png(hashlib.sha1(key.encode()).hexdigest(),
                              lambda: render_burndepth(bd, max_px))
            self._send(200, png, "image/png")
        else:
            self._send_json({"error": "unknown resource"}, 404)

    def _handle_burndepth_tile(self, path: str):
        # /api/burndepth_tile/<z>/<x>/<y>.png
        parts = path.split("/")
        if len(parts) != 6 or not parts[5].endswith(".png"):
            self._send_json({"error": "bad path"}, 404)
            return
        try:
            z, x, y = int(parts[3]), int(parts[4]), int(parts[5][:-4])
        except ValueError:
            self._send_json({"error": "bad tile coordinates"}, 400)
            return
        vrt = _burndepth_mosaic_vrt() if _is_burn() else None
        if vrt is None:
            self._send_json({"error": "no burn-depth rasters"}, 404)
            return
        key = f"burn_{z}_{x}_{y}_{vrt.stat().st_mtime_ns}"
        png = _cached_png(hashlib.sha1(key.encode()).hexdigest(),
                          lambda: render_burndepth_tile(z, x, y))
        self._send(200, png, "image/png")

    def _handle_depthsrc_tile(self, path: str):
        # /api/depthsrc_tile/<z>/<x>/<y>.png
        parts = path.split("/")
        if len(parts) != 6 or not parts[5].endswith(".png"):
            self._send_json({"error": "bad path"}, 404)
            return
        try:
            z, x, y = int(parts[3]), int(parts[4]), int(parts[5][:-4])
        except ValueError:
            self._send_json({"error": "bad tile coordinates"}, 400)
            return
        files = _ds_files() if _is_burn() else []
        if not files:
            self._send_json({"error": "no depth-source layer"}, 404)
            return
        sig = "_".join(str(p.stat().st_mtime_ns) for p in files)
        key = f"dsrc_{z}_{x}_{y}_{sig}"
        png = _cached_png(hashlib.sha1(key.encode()).hexdigest(),
                          lambda: render_depthsrc_tile(z, x, y))
        self._send(200, png, "image/png")

    def _handle_depth_tile(self, path: str):
        # /api/depth_tile/<z>/<x>/<y>.png
        parts = path.split("/")
        if len(parts) != 6 or not parts[5].endswith(".png"):
            self._send_json({"error": "bad path"}, 404)
            return
        try:
            z, x, y = int(parts[3]), int(parts[4]), int(parts[5][:-4])
        except ValueError:
            self._send_json({"error": "bad tile coordinates"}, 400)
            return
        vrt = dp.path(_cfg()["output"]["vrt"])
        if not vrt.exists():
            self._send_json({"error": "vrt not found"}, 404)
            return
        key = f"depth_{z}_{x}_{y}_{vrt.stat().st_mtime_ns}"
        png = _cached_png(hashlib.sha1(key.encode()).hexdigest(),
                          lambda: render_depth_tile(z, x, y))
        self._send(200, png, "image/png")

    def _handle_source_tile(self, path: str):
        # /api/source_tile/<z>/<x>/<y>.png
        parts = path.split("/")
        if len(parts) != 6 or not parts[5].endswith(".png"):
            self._send_json({"error": "bad path"}, 404)
            return
        try:
            z, x, y = int(parts[3]), int(parts[4]), int(parts[5][:-4])
        except ValueError:
            self._send_json({"error": "bad tile coordinates"}, 400)
            return
        vrt = _source_mosaic_vrt()
        if vrt is None:
            self._send_json({"error": "no provenance rasters found"}, 404)
            return
        key = f"source_{z}_{x}_{y}_{vrt.stat().st_mtime_ns}"
        png = _cached_png(hashlib.sha1(key.encode()).hexdigest(),
                          lambda: render_source_tile(z, x, y))
        self._send(200, png, "image/png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1; use 0.0.0.0 to allow "
                         "LAN access -- only on a trusted network)")
    ap.add_argument("--merge-config", default=None,
                    help="view a region merge (e.g. config/merge_region02.yaml) "
                         "instead of this checkout's basin product")
    ap.add_argument("--burn-config", default=None,
                    help="view a channel-burned product (e.g. "
                         "config/burn_region02.yaml), with a burn-depth layer")
    args = ap.parse_args()
    dp.setup_logging()
    global _MERGE
    if args.burn_config and args.merge_config:
        ap.error("--burn-config and --merge-config are exclusive")
    if args.merge_config:
        _MERGE = _merge_view(args.merge_config)
    elif args.burn_config:
        _MERGE = _burn_view(args.burn_config)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"DE watershed DEM viewer running at http://{args.host}:{args.port}/")
    if args.host in ("127.0.0.1", "localhost"):
        print("On a remote host, tunnel from your laptop with:\n"
              f"  ssh -L {args.port}:localhost:{args.port} <this host>\n"
              f"then open http://localhost:{args.port}/ locally.")
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
