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
    "filled": (220, 40, 40),
}


def _cfg():
    return dp.load_config()


def _cache_dir() -> Path:
    d = dp.path(_cfg()["qa"]["quicklook_dir"], ".viewer_cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    """config/tiles_0204.geojson features, enriched with coverage/QA flags."""
    cfg = _cfg()
    with open(dp.path("config", "tiles_0204.geojson")) as f:
        fc = json.load(f)

    coverage, land_overlap = {}, {}
    seam_flags, size_mismatches = {}, {}
    qa_path = dp.path(cfg["qa"]["dir"]) / "qa_report.json"
    if qa_path.exists():
        with open(qa_path) as f:
            checks = json.load(f)
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
                                 "legend": source_legend()})
            elif path == "/api/tiles.geojson":
                self._send_json(build_tiles_geojson())
            elif path == "/api/depth_legend":
                cfg = _cfg()
                if not dp.path(cfg["output"]["vrt"]).exists():
                    self._send_json({"error": "vrt not found"}, 404)
                else:
                    self._send_json(depth_legend())
            elif path.startswith("/api/depth_tile/"):
                self._handle_depth_tile(path)
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
        else:
            self._send_json({"error": "unknown resource"}, 404)

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1; use 0.0.0.0 to allow "
                         "LAN access -- only on a trusted network)")
    args = ap.parse_args()
    dp.setup_logging()

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
