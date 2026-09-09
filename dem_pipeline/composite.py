"""Per-tile masked two-stack compositor.

For each output tile we build a LAND composite (S1M -> project 1-m -> 1/3")
and a WATER composite (eHydro -> EAARL-B -> CUDEM -> CoNED), each first-valid
by priority, then blend them through a water mask with an EDT-based feather
so no artificial step enters the subgrid lookup tables. Sources are read
through per-source VRTs and warped window-by-window in memory; no basin-scale
mosaic is ever materialized.

Outputs per tile (float32 elevation + uint8 provenance):
    data/intermediate/tiles/<name>.tif
    data/intermediate/tiles/<name>_source.tif
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import load_config, path as repo_path
from .grid import GridSpec, output_filename

log = logging.getLogger(__name__)

LAND_SOURCES = ["s1m", "dem1m_projects", "dem13"]
WATER_SOURCES = ["ehydro", "eaarlb", "cudem19", "coned"]

_RESAMPLING = {"near": "nearest", "average": "average", "bilinear": "bilinear"}

# Plausibility window for LAND elevations (m NAVD88), basin-wide: below sea
# level tidal marsh/subsidence up past the actual highest terrain in the
# basin -- confirmed against raw S1M pixels under Slide Mountain (the
# Catskills' and this basin's high point, 1274 m) reading a smooth ~1275 m
# summit, with margin above that. An earlier 1000 m ceiling was clipping
# that real high-Catskills terrain as "implausible" and replacing over a
# million real elevation pixels per affected tile with interpolated filler
# -- a correctness bug, not just a seam artifact (2026-09-08). Unlike
# eHydro's window (scripts/06), this never aborts a run -- some real-world
# S1M distribution tiles carry a minority of corrupt edge pixels that are
# literal fractional blends of the nodata sentinel with real data (e.g. a
# raw pixel of -529999.5 in a tile whose nodata is -999999.0 is
# -999999.0 * 0.53 -- not a unit/datum bug, a bad tile). Masking just those
# per-pixel outliers lets first_valid() fall through to the next-priority
# land source instead of losing (or corrupting) the whole tile over a
# handful of bad edge pixels.
LAND_PLAUSIBLE_MIN, LAND_PLAUSIBLE_MAX = -15.0, 1500.0

# Some land products (confirmed for both `s1m` and `dem13`, 2026-09-09) fill
# known-open-water areas with a flat 0 m plate instead of proper nodata,
# rather than leaving a hole. Combined with NHDPlus HR (the water mask's only
# source, scripts/07_build_water_mask.py) having zero coverage over the open
# Atlantic, that plate silently read as valid land elevation basin-wide: 3
# whole output tiles on the continental shelf south of Cape May came out
# ~100% flat 0.0 "land" with real CUDEM bathymetry (down to -23 m) sitting
# unused in the water stack at the same pixels (230M px / 2.7% of the basin
# in total). Real lidar/DEM elevations essentially never land on a bit-exact
# 0.0, so it's a safe tell -- see the water_mask construction in
# compose_tile. See ARCHITECTURE.md §9.
LAND_VOID_FILL_VALUE = 0.0


def source_vrt(name: str) -> Path | None:
    """Path to a source's VRT, or None if the source has no data yet."""
    p = repo_path("data", "intermediate", "vrt", f"{name}.vrt")
    return p if p.exists() else None


def warp_to_tile(src_path: Path, gt, width: int, height: int, dst_crs,
                 resampling: str, dst_nodata: float) -> np.ndarray:
    """Warp any raster/VRT onto the exact tile grid; nodata -> NaN."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.vrt import WarpedVRT

    transform = Affine.from_gdal(*gt)
    with rasterio.open(src_path) as src:
        with WarpedVRT(
            src,
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            resampling=Resampling[_RESAMPLING[resampling]],
            src_nodata=src.nodata,
            nodata=dst_nodata,
        ) as vrt:
            arr = vrt.read(1).astype("f4")
    arr[arr == dst_nodata] = np.nan
    return arr


def first_valid(arrays: dict[str, np.ndarray], id_map: dict[str, int]):
    """Stack arrays by insertion order; first finite value wins per pixel."""
    shape = next(iter(arrays.values())).shape
    out = np.full(shape, np.nan, dtype="f4")
    ids = np.zeros(shape, dtype="u1")
    for name, arr in arrays.items():
        take = np.isnan(out) & np.isfinite(arr)
        out[take] = arr[take]
        ids[take] = id_map[name]
    return out, ids


def _rasterize(geoms, gt, shape) -> np.ndarray:
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    if not geoms:
        return np.zeros(shape, dtype=bool)
    return rasterize(
        [(g, 1) for g in geoms],
        out_shape=shape,
        transform=Affine.from_gdal(*gt),
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)


def _read_mask_geoms(gpkg: Path, layer: str, bounds) -> list:
    if not gpkg.exists():
        return []
    import geopandas as gpd
    import pyogrio

    # An absent layer is expected (e.g. no eHydro footprints yet); any other
    # read failure must propagate rather than silently composing as land.
    if layer not in {name for name, *_ in pyogrio.list_layers(gpkg)}:
        return []
    gdf = gpd.read_file(gpkg, layer=layer, bbox=bounds)
    return list(gdf.geometry.values)


def feather_alpha(water_mask: np.ndarray, feather_px: int) -> np.ndarray:
    """0 on land, 1 in water, linear ramp across the mask boundary."""
    from scipy.ndimage import distance_transform_edt

    # Degenerate masks have no boundary: EDT of an all-True mask measures
    # distance to the array corner, which would dip alpha below 1 there and
    # average land values into open-water tiles.
    if feather_px <= 0 or water_mask.all() or not water_mask.any():
        return water_mask.astype("f4")
    d_out = distance_transform_edt(~water_mask)   # distance into land
    d_in = distance_transform_edt(water_mask)     # distance into water
    # signed distance from boundary, positive into water
    signed = np.where(water_mask, d_in, -d_out)
    alpha = np.clip((signed + feather_px) / (2.0 * feather_px), 0.0, 1.0)
    return alpha.astype("f4")


def feather_overlay(base: np.ndarray, base_ids: np.ndarray,
                    overlay: np.ndarray, overlay_id: int,
                    feather_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Blend a higher-priority overlay onto a base, ramping in from the
    overlay's edge so survey boundaries don't step against older data.

    Weight w rises 0 -> 1 over feather_px pixels *inside* the overlay's valid
    footprint; where the base is missing the overlay applies fully.
    """
    from scipy.ndimage import distance_transform_edt

    valid = np.isfinite(overlay)
    if not valid.any():
        return base, base_ids
    if feather_px > 0 and not valid.all():
        w = np.clip(distance_transform_edt(valid) / feather_px, 0.0, 1.0)
    else:
        w = valid.astype("f8")
    w = np.where(valid, w, 0.0)
    w = np.where(valid & np.isnan(base), 1.0, w)

    out = np.where(w >= 1.0, overlay,
                   np.where(w <= 0.0, base,
                            w * overlay + (1.0 - w) * base)).astype("f4")
    out[np.isnan(overlay) & np.isnan(base)] = np.nan
    ids = np.where(w >= 0.5, np.uint8(overlay_id), base_ids).astype("u1")
    ids[np.isnan(out)] = 0
    return out, ids


def feathered_first_valid(arrays: dict[str, np.ndarray],
                          id_map: dict[str, int], feather_px: int):
    """Priority stack like first_valid, but each higher-priority source is
    feathered in from its footprint edge instead of hard-switched."""
    names = list(arrays)
    base, ids = first_valid({names[-1]: arrays[names[-1]]}, id_map)
    for name in reversed(names[:-1]):
        base, ids = feather_overlay(base, ids, arrays[name],
                                    id_map[name], feather_px)
    return base, ids


def compose_tile(name: str, out_dir: Path | None = None) -> Path:
    """Build one output tile. Idempotent: skips if the product exists."""
    import rasterio
    from rasterio.fill import fillnodata
    from rasterio.transform import Affine

    cfg = load_config()
    spec = GridSpec.load()
    comp = cfg["compositing"]
    id_map = comp["source_ids"]
    nodata = cfg["target_grid"]["nodata"]
    # The output CRS and nodata are PRODUCT choices from config, not inherited
    # from the audited CUDEM tiles. Real CUDEM is not internally consistent
    # (EPSG 5498 and 4269, nodata -99999/-9999 across vintages), so inheriting
    # would make the product contract depend on which tiles a given run
    # happened to audit. Stage 02 cross-checks these and warns on divergence;
    # per-tile bit-exactness comes from the verbatim geotransforms, not here.
    dst_crs = f"EPSG:{cfg['target_grid']['epsg']}"

    out_dir = out_dir or repo_path("data", "intermediate", "tiles")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"{name}.tif"
    src_tif = out_dir / f"{name}_source.tif"
    if out_tif.exists() and src_tif.exists():
        if spec.tile_matches(name, out_tif):
            log.info("%s: already composed", name)
            return out_tif
        log.warning("%s: existing product's grid doesn't match the current "
                    "GridSpec (stale from a run under an older config) -- "
                    "recomposing", name)

    gt, width, height = spec.tile_geotransform(name)

    # feather_alpha's EDT and fillnodata's IDW both only "see" the array
    # they're handed. Computed at exactly the output tile's own bounds, a
    # feather/fill zone that straddles a tile boundary sees a different
    # neighborhood on each side (this tile's edge vs. nothing) and the two
    # neighboring tiles disagree at the seam -- not from any real elevation
    # difference, purely from each tile's independent computation not seeing
    # past its own edge. Padding the WORKING window by enough pixels to
    # cover every EDT/fill radius used below, then cropping back to the
    # exact output grid before writing, gives both tiles the same true
    # geographic neighborhood in their shared boundary zone so they converge
    # on the same answer. The written geotransform/size are unaffected.
    pad_px = max(comp["feather_px"], comp["survey_feather_px"],
                comp["fill_max_px"])
    pad_gt = (gt[0] - pad_px * gt[1], gt[1], 0.0,
              gt[3] - pad_px * gt[5], 0.0, gt[5])
    pad_width, pad_height = width + 2 * pad_px, height + 2 * pad_px
    pad_shape = (pad_height, pad_width)
    pad_bounds = (pad_gt[0], pad_gt[3] + pad_height * pad_gt[5],
                 pad_gt[0] + pad_width * pad_gt[1], pad_gt[3])

    def warp_stack(source_names: list[str]) -> dict[str, np.ndarray]:
        arrays = {}
        for s in source_names:
            scfg = cfg["sources"][s]
            if not scfg.get("enabled", True):
                continue
            vrt = source_vrt(s)
            if vrt is None:
                continue
            arr = warp_to_tile(vrt, pad_gt, pad_width, pad_height, dst_crs,
                               scfg["resampling"], nodata)
            if s in LAND_SOURCES:
                bad = np.isfinite(arr) & (
                    (arr < LAND_PLAUSIBLE_MIN) | (arr > LAND_PLAUSIBLE_MAX))
                if bad.any():
                    log.warning(
                        "%s: %s: masking %d implausible land px (range "
                        "[%.1f, %.1f] outside [%g, %g])",
                        name, s, int(bad.sum()),
                        float(np.min(arr[bad])), float(np.max(arr[bad])),
                        LAND_PLAUSIBLE_MIN, LAND_PLAUSIBLE_MAX)
                    arr[bad] = np.nan
            if np.isfinite(arr).any():
                arrays[s] = arr
        return arrays

    land_arrays = warp_stack(LAND_SOURCES)
    water_arrays = warp_stack(WATER_SOURCES)
    if not land_arrays and not water_arrays:
        raise RuntimeError(f"{name}: no source data intersects this tile")

    land, land_ids = (first_valid(land_arrays, id_map) if land_arrays
                      else (np.full(pad_shape, np.nan, "f4"), np.zeros(pad_shape, "u1")))
    # Water sources are feathered into each other at survey footprint edges
    # (eHydro/EAARL-B channels must not step against the older CUDEM grid).
    water, water_ids = (
        feathered_first_valid(water_arrays, id_map, comp["survey_feather_px"])
        if water_arrays
        else (np.full(pad_shape, np.nan, "f4"), np.zeros(pad_shape, "u1")))

    # --- water mask: NHD polygons U survey footprints (minus EAARL voids)
    #     U (missing or untrustworthy land data where a water source has
    #     real data) ---
    mask_gpkg = repo_path("data", "intermediate", "water_mask", "water_mask.gpkg")
    water_mask = _rasterize(_read_mask_geoms(mask_gpkg, "water", pad_bounds), pad_gt, pad_shape)
    for layer in ("ehydro_footprint", "eaarlb_footprint"):
        water_mask |= _rasterize(_read_mask_geoms(mask_gpkg, layer, pad_bounds), pad_gt, pad_shape)
    void = _rasterize(_read_mask_geoms(mask_gpkg, "eaarlb_voids", pad_bounds), pad_gt, pad_shape)
    water_mask &= ~void
    # NHDPlus HR has no coverage over the open Atlantic. Two more signals
    # close that gap: no land source has any elevation here at all, or the
    # land composite is the LAND_VOID_FILL_VALUE plate (see its definition
    # above). Both are harmless to OR in unconditionally -- the trailing
    # `& isfinite(water)` below already drops them back out wherever no
    # water source actually has data either.
    water_mask |= np.isnan(land) | (land == LAND_VOID_FILL_VALUE)
    # Water sources only count where they actually have data.
    water_mask &= np.isfinite(water)

    # --- feathered blend, hard switch where one side is missing ---
    alpha = feather_alpha(water_mask, comp["feather_px"])
    alpha[np.isnan(water)] = 0.0
    alpha[np.isnan(land)] = 1.0
    both_missing = np.isnan(land) & np.isnan(water)

    z = np.where(
        alpha >= 1.0, water,
        np.where(alpha <= 0.0, land,
                 alpha * water + (1.0 - alpha) * land),
    ).astype("f4")
    z[both_missing] = np.nan
    source_id = np.where(alpha >= 0.5, water_ids, land_ids).astype("u1")
    source_id[both_missing] = id_map["nodata"]

    # --- residual nodata fill (rare; flagged in provenance) ---
    holes = np.isnan(z)
    if holes.any() and np.isfinite(z).any():
        filled = fillnodata(
            np.where(holes, nodata, z),
            mask=~holes,
            max_search_distance=comp["fill_max_px"],
        )
        newly = holes & (filled != nodata)
        z[newly] = filled[newly]
        source_id[newly] = id_map["filled"]

    # crop the padded working window back to the exact output grid
    z = z[pad_px:pad_px + height, pad_px:pad_px + width]
    source_id = source_id[pad_px:pad_px + height, pad_px:pad_px + width]

    # --- write ---
    transform = Affine.from_gdal(*gt)
    profile = dict(
        driver="GTiff", dtype="float32", count=1, width=width, height=height,
        transform=transform, crs=dst_crs, nodata=nodata,
        compress="deflate", predictor=3, tiled=True,
        blockxsize=512, blockysize=512,
    )
    tmp = out_tif.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(np.where(np.isnan(z), nodata, z), 1)
        dst.update_tags(
            AREA_OR_POINT="Area",
            vertical_datum="NAVD88 (m)",
            product=output_filename(name, cfg),
        )
    tmp.rename(out_tif)

    profile.update(dtype="uint8", nodata=None, predictor=1)
    tmp = src_tif.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(source_id, 1)
        dst.update_tags(**{str(v): k for k, v in id_map.items()})
    tmp.rename(src_tif)

    pct = {k: 100.0 * np.mean(source_id == v)
           for k, v in id_map.items() if np.any(source_id == v)}
    log.info("%s: composed; source mix %s", name,
             {k: f"{v:.1f}%" for k, v in pct.items()})
    return out_tif
