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
        log.info("%s: already composed", name)
        return out_tif

    gt, width, height = spec.tile_geotransform(name)
    bounds = spec.tile_bounds(name)
    shape = (height, width)

    def warp_stack(source_names: list[str]) -> dict[str, np.ndarray]:
        arrays = {}
        for s in source_names:
            scfg = cfg["sources"][s]
            if not scfg.get("enabled", True):
                continue
            vrt = source_vrt(s)
            if vrt is None:
                continue
            arr = warp_to_tile(vrt, gt, width, height, dst_crs,
                               scfg["resampling"], nodata)
            if np.isfinite(arr).any():
                arrays[s] = arr
        return arrays

    land_arrays = warp_stack(LAND_SOURCES)
    water_arrays = warp_stack(WATER_SOURCES)
    if not land_arrays and not water_arrays:
        raise RuntimeError(f"{name}: no source data intersects this tile")

    land, land_ids = (first_valid(land_arrays, id_map) if land_arrays
                      else (np.full(shape, np.nan, "f4"), np.zeros(shape, "u1")))
    # Water sources are feathered into each other at survey footprint edges
    # (eHydro/EAARL-B channels must not step against the older CUDEM grid).
    water, water_ids = (
        feathered_first_valid(water_arrays, id_map, comp["survey_feather_px"])
        if water_arrays
        else (np.full(shape, np.nan, "f4"), np.zeros(shape, "u1")))

    # --- water mask: NHD polygons U survey footprints (minus EAARL voids)
    #     U (S1M nodata where CUDEM has data) ---
    mask_gpkg = repo_path("data", "intermediate", "water_mask", "water_mask.gpkg")
    water_mask = _rasterize(_read_mask_geoms(mask_gpkg, "water", bounds), gt, shape)
    for layer in ("ehydro_footprint", "eaarlb_footprint"):
        water_mask |= _rasterize(_read_mask_geoms(mask_gpkg, layer, bounds), gt, shape)
    void = _rasterize(_read_mask_geoms(mask_gpkg, "eaarlb_voids", bounds), gt, shape)
    water_mask &= ~void
    if "s1m" in land_arrays and "cudem19" in water_arrays:
        water_mask |= np.isnan(land_arrays["s1m"]) & np.isfinite(
            water_arrays["cudem19"]
        )
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
