"""QA checks for the final tile set.

Covers: grid conformity (bit-exact vs CUDEM where applicable), cross-tile
seam continuity, land-overlap statistics against CUDEM, per-tile source mix /
coverage, and hillshade quicklooks. Each check returns a dict that
scripts/10_qa_checks.py aggregates into data/final/qa/qa_report.json.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

from . import load_config, path as repo_path
from .grid import GridSpec, parse_tile_name, tile_name, TILE_DEG

log = logging.getLogger(__name__)


def check_grid_conformity(tile_paths: dict[str, Path]) -> dict:
    """Every tile's geotransform/size must match the GridSpec exactly."""
    import rasterio

    spec = GridSpec.load()
    failures = {}
    for name, p in tile_paths.items():
        gt_expected, w, h = spec.tile_geotransform(name)
        with rasterio.open(p) as src:
            gt = src.transform.to_gdal()
            actual = [list(gt), src.width, src.height]
            ok = (src.width == w and src.height == h
                  and all(a == b for a, b in zip(gt, gt_expected)))
        if not ok:
            failures[name] = {"expected": [list(gt_expected), w, h],
                              "actual": actual}
    return {"check": "grid_conformity", "pass": not failures,
            "n_tiles": len(tile_paths), "failures": failures}


def check_seams(tile_paths: dict[str, Path]) -> dict:
    """Difference abutting edge rows/cols of neighboring tiles.

    With zero collar the shared boundary has no duplicated samples, so we
    compare the facing edge rows/cols and report the max step; with a collar,
    overlapping pixels must agree exactly.
    """
    import rasterio

    cfg = load_config()["qa"]
    spec = GridSpec.load()
    results = {}
    for name, p in tile_paths.items():
        north, west = parse_tile_name(name)
        east_neighbor = tile_name(north, west + TILE_DEG)
        south_neighbor = tile_name(north - TILE_DEG, west)
        for other, axis in ((east_neighbor, "x"), (south_neighbor, "y")):
            if other not in tile_paths:
                continue
            with rasterio.open(p) as a, rasterio.open(tile_paths[other]) as b:
                if spec.collar_px > 0:
                    n = 2 * spec.collar_px
                    if axis == "x":
                        ea = a.read(1)[:, -n:]
                        eb = b.read(1)[:, :n]
                    else:
                        ea = a.read(1)[-n:, :]
                        eb = b.read(1)[:n, :]
                    metric = float(np.nanmax(np.abs(
                        _nanify(ea, a.nodata) - _nanify(eb, b.nodata))))
                else:
                    if axis == "x":
                        ea = a.read(1)[:, -1]
                        eb = b.read(1)[:, 0]
                    else:
                        ea = a.read(1)[-1, :]
                        eb = b.read(1)[0, :]
                    metric = float(np.nanmax(np.abs(
                        _nanify(ea, a.nodata) - _nanify(eb, b.nodata))))
            results[f"{name}|{other}"] = metric
    flagged = {k: v for k, v in results.items()
               if np.isfinite(v) and v > cfg["seam_gradient_flag_m"]}
    return {"check": "seams", "pass": not flagged,
            "n_edges": len(results), "flagged": flagged}


def _nanify(arr, nodata):
    out = arr.astype("f8")
    if nodata is not None:
        out[out == nodata] = np.nan
    return out


def check_cudem_land_overlap(tile_paths: dict[str, Path]) -> dict:
    """Histogram of (ours - CUDEM) over land in overlapping tiles.

    Expect |median| under qa.land_overlap_median_max_m in stable terrain;
    a larger bias indicates a datum/geoid handling error.
    """
    import rasterio

    cfg = load_config()
    spec = GridSpec.load()
    cudem_dir = repo_path(cfg["sources"]["cudem19"]["dir"])
    stats = {}
    for name, p in tile_paths.items():
        matches = sorted(cudem_dir.glob(f"ncei19_{name}_*.tif"))
        if not matches:
            continue
        with rasterio.open(p) as a, rasterio.open(matches[0]) as b:
            if a.shape != b.shape:
                stats[name] = {"error": "shape mismatch with CUDEM"}
                continue
            za = _nanify(a.read(1), a.nodata)
            zb = _nanify(b.read(1), b.nodata)
        # restrict to land (above +1 m in both) to exclude bathy updates
        land = (za > 1.0) & (zb > 1.0)
        d = (za - zb)[land]
        d = d[np.isfinite(d)]
        if d.size:
            stats[name] = {
                "n": int(d.size),
                "median": float(np.median(d)),
                "p05": float(np.percentile(d, 5)),
                "p95": float(np.percentile(d, 95)),
            }
    worst = max((abs(s["median"]) for s in stats.values() if "median" in s),
                default=0.0)
    return {"check": "cudem_land_overlap",
            "pass": worst <= cfg["qa"]["land_overlap_median_max_m"],
            "worst_abs_median_m": worst, "tiles": stats}


def coverage_report(tile_paths: dict[str, Path]) -> dict:
    """Per-tile % of pixels by provenance source id."""
    import rasterio

    id_map = load_config()["compositing"]["source_ids"]
    inv = {v: k for k, v in id_map.items()}
    rows = {}
    for name, p in tile_paths.items():
        src_p = Path(str(p).replace(".tif", "_source.tif"))
        if not src_p.exists():
            src_p = p.parent.parent.parent / "intermediate" / "tiles" / f"{name}_source.tif"
        if not src_p.exists():
            rows[name] = {"error": "no source raster"}
            continue
        with rasterio.open(src_p) as src:
            ids = src.read(1)
        vals, counts = np.unique(ids, return_counts=True)
        rows[name] = {inv.get(int(v), str(v)): round(100.0 * c / ids.size, 2)
                      for v, c in zip(vals, counts)}
    return {"check": "coverage", "tiles": rows}


def provenance_quicklook(tile_path: Path, out_dir: Path | None = None) -> Path:
    """Colour-mapped PNG of the uint8 provenance raster.

    The hillshade shows whether the surface looks right; this shows WHY -- where
    each source won, how wide the feather bands are, and where nodata had to be
    filled. Reading a source mix as percentages hides all of that geometry.
    """
    import numpy as np
    import rasterio

    cfg = load_config()
    ids = cfg["compositing"]["source_ids"]
    out_dir = out_dir or repo_path(cfg["qa"]["quicklook_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    src_path = tile_path.parent / (tile_path.stem + "_source.tif")
    if not src_path.exists():
        raise FileNotFoundError(f"no provenance raster beside {tile_path.name}")
    png = out_dir / (tile_path.stem + "_src.png")

    # distinguishable hues; land sources warm, water sources cool
    colours = {
        ids["nodata"]: (255, 255, 255), ids["s1m"]: (198, 148, 74),
        ids["dem1m_projects"]: (232, 196, 128), ids["dem13"]: (245, 226, 180),
        ids["ehydro"]: (26, 92, 158), ids["eaarlb"]: (70, 160, 190),
        ids["cudem19"]: (140, 200, 214), ids["coned"]: (190, 225, 232),
        ids["filled"]: (220, 40, 40),
    }
    with rasterio.open(src_path) as src:
        arr = src.read(1, out_shape=(src.height // 4, src.width // 4))
    rgb = np.zeros(arr.shape + (3,), dtype="u1")
    for value, colour in colours.items():
        rgb[arr == value] = colour

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.imsave(png, rgb)
    return png


def hillshade_quicklook(tile_path: Path, out_dir: Path | None = None) -> Path:
    """gdaldem hillshade -> PNG quicklook for visual inspection."""
    cfg = load_config()["qa"]
    out_dir = out_dir or repo_path(cfg["quicklook_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shade = out_dir / (tile_path.stem + "_hs.tif")
    png = out_dir / (tile_path.stem + "_hs.png")
    if png.exists():
        return png
    subprocess.run(
        ["gdaldem", "hillshade", "-compute_edges", "-z", "1.0",
         "-s", "111120",  # meters per degree: geographic CRS scale factor
         str(tile_path), str(shade)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["gdal_translate", "-of", "PNG", "-outsize", "25%", "25%",
         str(shade), str(png)],
        check=True, capture_output=True,
    )
    shade.unlink(missing_ok=True)
    Path(str(png) + ".aux.xml").unlink(missing_ok=True)
    return png
