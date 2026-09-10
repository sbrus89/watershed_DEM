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

    A flagged edge whose worst-disagreement pixel traces to `cudem19`
    provenance on *both* sides is an inherent CUDEM-vintage registration
    mismatch: the grid contract requires verbatim passthrough of each tile's
    audited CUDEM geotransform (`resampling: near`), and two real adjacent
    CUDEM tiles from different survey vintages aren't necessarily mutually
    pixel-aligned -- see ARCHITECTURE.md §9. That case is reported separately
    as `inherent_misalignment` rather than failing the check, the same way
    `size_mismatches` is recorded rather than papered over. This rule is
    deliberately narrow: it only fires on a same-source (`cudem19`/`cudem19`)
    pair. A mixed-source edge is never auto-accepted this way, even if it
    looks superficially similar -- one `cudem19`/`dem13` edge turned out to
    be this same inherent mismatch on manual review, while another
    `cudem19`/`dem13` edge turned out to be a real, still-undiagnosed bug
    (see ARCHITECTURE.md §9). Automating past that ambiguity would risk
    silently swallowing the real one.
    """
    import rasterio

    full_cfg = load_config()
    cfg = full_cfg["qa"]
    cudem_id = full_cfg.get("compositing", {}).get("source_ids", {}).get("cudem19")
    spec = GridSpec.load()
    results = {}
    flagged = {}
    inherent_misalignment = {}
    size_mismatches = {}
    for name, p in tile_paths.items():
        north, west = parse_tile_name(name)
        east_neighbor = tile_name(north, west + TILE_DEG)
        south_neighbor = tile_name(north - TILE_DEG, west)
        for other, axis in ((east_neighbor, "x"), (south_neighbor, "y")):
            if other not in tile_paths:
                continue
            key = f"{name}|{other}"
            with rasterio.open(p) as a, rasterio.open(tile_paths[other]) as b:
                # A tile with an empirically-audited CUDEM size can differ by
                # a pixel or two from a neighbor sized off the nominal
                # collar_px lattice (no CUDEM coverage there) -- this is a
                # real grid-contract gap between the two sizing paths, not
                # something to silently paper over, so it's recorded
                # separately rather than just clipped away.
                shared = a.height if axis == "x" else a.width
                other_shared = b.height if axis == "x" else b.width
                if shared != other_shared:
                    size_mismatches[key] = [shared, other_shared]
                    shared = min(shared, other_shared)
                if spec.collar_px > 0:
                    n = 2 * spec.collar_px
                    if axis == "x":
                        ea = a.read(1)[:shared, -n:]
                        eb = b.read(1)[:shared, :n]
                        a_px = lambda idx, w=a.width: (idx[0], w - n + idx[1])
                        b_px = lambda idx: (idx[0], idx[1])
                    else:
                        ea = a.read(1)[-n:, :shared]
                        eb = b.read(1)[:n, :shared]
                        a_px = lambda idx, h=a.height: (h - n + idx[0], idx[1])
                        b_px = lambda idx: (idx[0], idx[1])
                else:
                    if axis == "x":
                        ea = a.read(1)[:shared, -1]
                        eb = b.read(1)[:shared, 0]
                        a_px = lambda idx, w=a.width: (idx[0], w - 1)
                        b_px = lambda idx: (idx[0], 0)
                    else:
                        ea = a.read(1)[-1, :shared]
                        eb = b.read(1)[0, :shared]
                        a_px = lambda idx, h=a.height: (h - 1, idx[0])
                        b_px = lambda idx: (0, idx[0])
                diff = np.abs(_nanify(ea, a.nodata) - _nanify(eb, b.nodata))
            metric = (float(np.nanmax(diff)) if np.any(np.isfinite(diff))
                      else float("nan"))
            results[key] = metric
            if np.isfinite(metric) and metric > cfg["seam_gradient_flag_m"]:
                idx = np.unravel_index(np.nanargmax(diff), diff.shape)
                ida = _provenance_at(p, *a_px(idx))
                idb = _provenance_at(tile_paths[other], *b_px(idx))
                if cudem_id is not None and ida == cudem_id and idb == cudem_id:
                    inherent_misalignment[key] = metric
                else:
                    flagged[key] = metric
    return {"check": "seams", "pass": not flagged and not size_mismatches,
            "n_edges": len(results), "flagged": flagged,
            "inherent_misalignment": inherent_misalignment,
            "size_mismatches": size_mismatches}


def _provenance_at(main_path: Path, row: int, col: int) -> int | None:
    """Provenance source id at one pixel, or None if unavailable.

    Best-effort: a missing companion `_source.tif` (e.g. in tests, or an
    older product predating provenance tracking) or an out-of-range pixel
    just means "can't classify," not an error.
    """
    import rasterio

    src_p = main_path.parent / (main_path.stem + "_source.tif")
    if not src_p.exists():
        return None
    with rasterio.open(src_p) as src:
        if not (0 <= row < src.height and 0 <= col < src.width):
            return None
        return int(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])


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

    # Underlay the hillshade at reduced opacity so terrain relief still reads
    # through the flat provenance colors. hillshade_quicklook is idempotent
    # (skips if its PNG already exists), so this is cheap even when stage 10
    # already generated it. Resized to arr's exact shape rather than trusted
    # to match: gdal_translate's "-outsize 25% 25%" and this raster's own
    # `// 4` downsample round independently and can differ by a pixel.
    from PIL import Image

    hs_png = hillshade_quicklook(tile_path, out_dir)
    hillshade = np.asarray(Image.open(hs_png).convert("L").resize(
        (arr.shape[1], arr.shape[0]), Image.BILINEAR))

    overlay_alpha = 0.6
    blended = (hillshade[..., None] * (1 - overlay_alpha)
              + rgb * overlay_alpha).astype("u1")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    names = {v: k for k, v in ids.items()}
    present = [v for v in colours if np.any(arr == v)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(blended)
    ax.axis("off")
    ax.set_title(tile_path.stem, fontsize=9)
    handles = [mpatches.Patch(facecolor=np.array(colours[v]) / 255,
                              edgecolor="black", linewidth=0.4,
                              label=names.get(v, str(v)))
               for v in present]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              fontsize=7, framealpha=0.9, title="provenance")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
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
