"""Merge finished per-basin products into one region-wide tile set.

Each HUC-4 basin is run as its own product in its own checkout (see
docs/ARCHITECTURE.md §11). Neighbouring basins produce the SAME 0.25° tile
wherever their +2 km buffered domains overlap, and those versions differ:

* beyond its own data edge a basin only has 50-px ``fillnodata``
  extrapolation (provenance ``filled``) -- tens of metres wrong;
* in shared estuaries each basin composed water from its own eHydro set,
  water mask and feathering -- metres apart;
* some ocean pixels lie outside every HUC-4 polygon.

The merge never recomposes from sources. Per pixel it blends the finished
basin versions:

1. **Territory**: each basin owns its WBD HUC-4 polygon; pixels outside all
   polygons belong to the basin whose polygon is nearest (a Voronoi
   extension computed once in vector space, so adjacent tiles agree).
2. **Feather**: each basin's weight is ``composite.feather_alpha`` of its
   territory mask (0.5 on the boundary, 0/1 beyond ``feather_px``), so no
   step is introduced along basin boundaries.
3. **Real data first**: a version's weight is zeroed where it is nodata or
   carries a non-real provenance (``filled``, ``nodata`` ids). If that
   leaves no weight, the pixel takes the real version of the basin with the
   largest territory weight (tier 2), else any finite version (tier 3).

Single-basin tiles pass through unchanged (byte copy when their grid
already equals the merged spec, otherwise a lossless geotransform restamp of
at most ``restamp_tol_px``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import TMP_TIF_SUFFIX, product_tifs
from .grid import GridSpec, _NAME_RE

log = logging.getLogger(__name__)

# Bump whenever blend semantics change: it is part of every tile's
# fingerprint, so a rerun rebuilds tiles merged under the old rules instead
# of trusting them because the files exist.
MERGE_VERSION = 1

NODATA = -999999.0


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Basin:
    key: str
    huc4: str
    root: Path
    name: str           # product name, e.g. dewatershed19
    version: str
    tile_dir: Path

    @property
    def code(self) -> int:
        """uint8 basin-of-origin value: the HUC-4's last two digits (0204 -> 4)."""
        return int(self.huc4) % 100

    def tile_path(self, tile: str) -> Path:
        return self.tile_dir / f"{self.name}_{tile}_{self.version}.tif"

    def source_path(self, tile: str) -> Path:
        return self.tile_dir / f"{self.name}_{tile}_{self.version}_source.tif"


def load_merge_config(path: str | Path) -> dict:
    with open(path) as f:
        mcfg = yaml.safe_load(f)
    mcfg["_path"] = str(Path(path).resolve())
    return mcfg


def output_root(mcfg: dict) -> Path:
    return Path(mcfg["output"]["root"])


def load_basins(mcfg: dict) -> list[Basin]:
    """Basins from the merge config, each validated against its own checkout's
    config/sources.yaml (the product name/version/tile dir come from there, so
    the merge reads exactly what that basin's stage 09 wrote)."""
    basins = []
    for b in mcfg["basins"]:
        root = Path(b["root"])
        with open(root / "config" / "sources.yaml") as f:
            cfg = yaml.safe_load(f)
        huc4 = str(cfg["project"]["huc4"])
        if str(b["huc4"]) != huc4:
            raise ValueError(f"basin {b['key']}: merge config says huc4 "
                             f"{b['huc4']} but {root} is configured for {huc4}")
        basins.append(Basin(
            key=b["key"], huc4=huc4, root=root,
            name=cfg["project"]["name"], version=cfg["project"]["version"],
            tile_dir=root / cfg["output"]["tile_dir"]))
    codes = [b.code for b in basins]
    if len(set(codes)) != len(codes) or not all(0 < c < 256 for c in codes):
        raise ValueError(f"basin codes (huc4 % 100) must be unique and in "
                         f"1..255: {dict(zip([b.key for b in basins], codes))}")
    return basins


def merged_source_ids(basins: list[Basin]) -> dict[str, int]:
    """Union of the basins' provenance id tables; any disagreement is fatal
    (a provenance raster is only meaningful under one table)."""
    ids: dict[str, int] = {}
    for b in basins:
        with open(b.root / "config" / "sources.yaml") as f:
            table = yaml.safe_load(f)["compositing"]["source_ids"]
        for k, v in table.items():
            if k in ids and ids[k] != v:
                raise ValueError(f"source id {k!r}: {ids[k]} vs {v} in {b.key}")
            ids[k] = v
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"two sources share an id: {ids}")
    return ids


def discover_tiles(basins: list[Basin]) -> dict[str, list[Basin]]:
    """tile name -> basins that produced it (config order)."""
    out: dict[str, list[Basin]] = {}
    for b in basins:
        for p in product_tifs(b.tile_dir, f"{b.name}_*_{b.version}.tif"):
            m = _NAME_RE.search(p.name)
            if m and p.name == f"{b.name}_{m.group(0)}_{b.version}.tif":
                out.setdefault(m.group(0), []).append(b)
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

def build_grid_spec(specs: dict[str, GridSpec], convention_from: str) -> GridSpec:
    """Combined spec: union of every basin's audited CUDEM geotransforms
    (any conflict is fatal -- the grid contract cannot hold two answers for
    one tile), with the modal collar/shift convention of ``convention_from``
    for tiles that overlap no CUDEM tile."""
    gts: dict[str, list] = {}
    sizes: dict[str, list] = {}
    for key, s in specs.items():
        for name, gt in s.cudem_geotransforms.items():
            gt, size = list(gt), list(s.cudem_sizes[name])
            if name in gts and (gts[name] != gt or sizes[name] != size):
                raise ValueError(f"CUDEM grid conflict for {name}: "
                                 f"{gts[name]} {sizes[name]} vs {gt} {size} "
                                 f"(from {key})")
            gts[name], sizes[name] = gt, size
    base = specs[convention_from]
    return GridSpec(epsg=base.epsg, nodata=base.nodata,
                    pixel_deg=base.pixel_deg, collar_px=base.collar_px,
                    x_shift_px=base.x_shift_px, y_shift_px=base.y_shift_px,
                    cudem_geotransforms=dict(sorted(gts.items())),
                    cudem_sizes=dict(sorted(sizes.items())))


def check_restamp(gt_in, size_in, gt_out, size_out, tol_px: float) -> None:
    """Raise unless a raster on (gt_in, size_in) can be relabelled to
    (gt_out, size_out) without resampling: same size, no rotation, and every
    pixel centre moves by at most ``tol_px`` pixels."""
    if tuple(size_in) != tuple(size_out):
        raise ValueError(f"size {size_in} != target {size_out}")
    if gt_in[2] or gt_in[4] or gt_out[2] or gt_out[4]:
        raise ValueError("rotated geotransform")
    px = abs(gt_out[1])
    w, h = size_out
    for o, step, n in ((0, 1, w), (3, 5, h)):
        worst = max(abs(gt_in[o] - gt_out[o]),
                    abs((gt_in[o] + n * gt_in[step]) - (gt_out[o] + n * gt_out[step])))
        if worst > tol_px * px:
            raise ValueError(f"grid offset {worst / px:.4g} px exceeds the "
                             f"{tol_px} px restamp tolerance: {gt_in} -> {gt_out}")


def padded_geotransform(gt, pad: int):
    return (gt[0] - pad * gt[1], gt[1], 0.0, gt[3] - pad * gt[5], 0.0, gt[5])


# --------------------------------------------------------------------------
# territories
# --------------------------------------------------------------------------

def build_territories(polys: dict[int, object], *, simplify_m: float = 200.0,
                      spacing_m: float = 1000.0, extend_m: float = 500e3,
                      crs: str = "EPSG:4269") -> dict[int, tuple]:
    """Partition the plane among the basin polygons.

    Returns {code: (polygon, exterior)}. Inside the union of ``polys`` a
    basin owns exactly its polygon (returned untouched, in ``crs``).
    Outside, every point goes to the polygon whose (simplified, densified)
    boundary is nearest -- a Voronoi diagram of boundary points, dissolved
    by basin, minus the union; computed in EPSG:5070 metres and reprojected.
    The two parts are kept apart rather than unioned: the reprojected
    exterior can differ from the polygon edge at the 1e-9 deg level, and a
    union of the two then trips GEOS topology errors. rasterize_territories
    burns exteriors first and polygons on top, so the polygons win.
    """
    import geopandas as gpd
    import shapely

    gs = gpd.GeoSeries(list(polys.values()), index=list(polys), crs=crs).to_crs(5070)
    union = shapely.union_all(gs.values)
    pts, labels = [], []
    for code, g in gs.items():
        ring = shapely.segmentize(g.boundary.simplify(simplify_m), spacing_m)
        c = shapely.get_coordinates(ring)
        pts.append(c)
        labels.append(np.full(len(c), code))
    pts, labels = np.vstack(pts), np.concatenate(labels)
    _, keep = np.unique(np.round(pts, 1), axis=0, return_index=True)
    pts, labels = pts[keep], labels[keep]
    env = shapely.box(*union.bounds).buffer(extend_m)
    cells = shapely.get_parts(shapely.voronoi_polygons(
        shapely.multipoints(pts), extend_to=env, ordered=True))
    if len(cells) != len(pts):
        raise RuntimeError("voronoi cell count != point count")
    ext = {}
    for code in polys:
        cell = shapely.union_all(cells[labels == code])
        ext[code] = shapely.difference(shapely.intersection(cell, env), union)
    ext = gpd.GeoSeries(list(ext.values()), index=list(ext), crs=5070).to_crs(crs)
    return {code: (shapely.make_valid(polys[code]), shapely.make_valid(ext[code]))
            for code in polys}


def rasterize_territories(terr: dict[int, tuple], gt, shape) -> np.ndarray:
    """uint8 basin-code raster on (gt, shape) from build_territories output.
    Exteriors are burned first and polygons last (later features win), so
    inside the region ownership is exactly the HUC-4 polygons. Pixels no
    geometry claims (sub-pixel slivers) take the nearest label."""
    import shapely
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from scipy.ndimage import distance_transform_edt

    h, w = shape
    x0, y0 = gt[0], gt[3]
    rect = (min(x0, x0 + w * gt[1]), min(y0, y0 + h * gt[5]),
            max(x0, x0 + w * gt[1]), max(y0, y0 + h * gt[5]))
    feats = []
    for part in (1, 0):                         # exteriors, then polygons
        for code, geoms in terr.items():
            g = shapely.clip_by_rect(geoms[part], *rect)   # tolerant of
            if not g.is_empty:                             # invalid input
                feats.append((g, int(code)))
    if not feats:
        raise ValueError("no territory intersects the raster")
    lab = rasterize(feats, out_shape=shape, transform=Affine.from_gdal(*gt),
                    fill=0, dtype="uint8")
    if (lab == 0).any():
        _, (ri, ci) = distance_transform_edt(lab == 0, return_indices=True)
        lab = lab[ri, ci]
    return lab


def territory_alphas(lab: np.ndarray, codes: list[int], feather_px: int,
                     pad: int) -> list[np.ndarray]:
    """Per-basin feather weight on the padded label raster, cropped by
    ``pad`` px on every side (the pad lets the ramp see a boundary just past
    the tile edge, so overlapping collars of adjacent tiles agree)."""
    from .composite import feather_alpha

    crop = (slice(pad, lab.shape[0] - pad), slice(pad, lab.shape[1] - pad))
    return [feather_alpha(lab == c, feather_px)[crop] for c in codes]


# --------------------------------------------------------------------------
# blend
# --------------------------------------------------------------------------

def blend(zs: list[np.ndarray], provs: list[np.ndarray],
          alphas: list[np.ndarray], nonreal_ids) -> dict:
    """Blend basin versions of one tile. See the module docstring.

    zs: float32 elevations with NaN for nodata; provs: uint8 provenance;
    alphas: territory weights. Index order is the tier-2/3 tie-break
    priority. Returns z (NaN = nodata), prov (uint8), winner (int8 index
    into the inputs, -1 = nodata) and per-tier pixel counts + the
    inter-basin disagreement inside the blend band.
    """
    shape = zs[0].shape
    nonreal = np.asarray(list(nonreal_ids), dtype="u1")
    num = np.zeros(shape, "f8")
    wsum = np.zeros(shape, "f8")
    best_w = np.full(shape, -1.0, "f4")
    win1 = np.full(shape, -1, "i1")
    rank2 = np.full(shape, -np.inf, "f4")
    pick2 = np.full(shape, -1, "i1")
    rank3 = np.full(shape, -np.inf, "f4")
    pick3 = np.full(shape, -1, "i1")
    zmax = np.full(shape, -np.inf, "f4")
    zmin = np.full(shape, np.inf, "f4")
    n = len(zs)
    for i, (z, p, a) in enumerate(zip(zs, provs, alphas)):
        valid = np.isfinite(z)
        real = valid & ~np.isin(p, nonreal)
        w = np.where(real, a, 0.0).astype("f4")
        # f8 products: where one basin carries all the weight the result is
        # then exactly its value (w*z/w would round in float32)
        num += w.astype("f8") * np.where(real, z, 0.0).astype("f8")
        wsum += w
        upd = w > best_w
        best_w[upd], win1[upd] = w[upd], i
        contributes = w > 0
        zmax[contributes] = np.maximum(zmax[contributes], z[contributes])
        zmin[contributes] = np.minimum(zmin[contributes], z[contributes])
        r = a + 1e-6 * (n - i)        # tie-break: config order
        m = real & (r > rank2)
        rank2[m], pick2[m] = r[m], i
        m = valid & (r > rank3)
        rank3[m], pick3[m] = r[m], i
    tier1 = wsum > 0
    tier2 = ~tier1 & (pick2 >= 0)
    tier3 = ~tier1 & ~tier2 & (pick3 >= 0)
    winner = np.where(tier1, win1, np.where(tier2, pick2,
                                            np.where(tier3, pick3, -1))).astype("i1")
    out = np.full(shape, np.nan, "f4")
    out[tier1] = (num[tier1] / wsum[tier1]).astype("f4")
    prov = np.zeros(shape, "u1")
    for i, (z, p) in enumerate(zip(zs, provs)):
        sel = winner == i
        prov[sel] = p[sel]
        fb = sel & ~tier1
        out[fb] = z[fb]
    band = tier1 & (best_w < wsum)          # >1 basin carries weight
    spread = (zmax - zmin)[band]
    return {
        "z": out, "prov": prov, "winner": winner,
        "counts": {"blended": int(band.sum()),
                   "single": int((tier1 & ~band).sum()),
                   "fallback_real": int(tier2.sum()),
                   "fallback_filled": int(tier3.sum()),
                   "nodata": int((winner < 0).sum())},
        "band_spread_m": ({"p50": float(np.percentile(spread, 50)),
                           "p99": float(np.percentile(spread, 99)),
                           "max": float(spread.max())} if spread.size else None),
    }


# --------------------------------------------------------------------------
# per-tile driver
# --------------------------------------------------------------------------

def output_name(mcfg: dict, tile: str, suffix: str = "") -> str:
    p = mcfg["project"]
    return f"{p['name']}_{tile}_{p['version']}{suffix}.tif"


def tiles_dir(mcfg: dict) -> Path:
    return output_root(mcfg) / "tiles"


def stats_path(mcfg: dict, tile: str) -> Path:
    return output_root(mcfg) / "manifests" / "merge_stats" / f"{tile}.json"


def _file_sig(p: Path) -> list:
    st = p.stat()
    return [str(p), st.st_size, st.st_mtime_ns]


def tile_fingerprint(mcfg: dict, tile: str, contributors: list[Basin],
                     spec: GridSpec, territory_digest: str,
                     nonreal_ids) -> str:
    gt, w, h = spec.tile_geotransform(tile)
    doc = {
        "merge_version": MERGE_VERSION,
        "feather_px": mcfg["merge"]["feather_px"],
        "restamp_tol_px": mcfg["merge"]["restamp_tol_px"],
        "nonreal_ids": sorted(int(i) for i in nonreal_ids),
        "grid": [list(gt), w, h],
        "territories": territory_digest if len(contributors) > 1 else None,
        "inputs": [[b.key, _file_sig(b.tile_path(tile)),
                    _file_sig(b.source_path(tile))] for b in contributors],
    }
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def _atomic_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.rename(path)


def _write_u1(path: Path, arr: np.ndarray, gt, epsg: int) -> None:
    import rasterio
    from rasterio.transform import Affine

    tmp = path.with_name(path.name[:-4] + TMP_TIF_SUFFIX)
    with rasterio.open(tmp, "w", driver="GTiff", dtype="uint8", count=1,
                       width=arr.shape[1], height=arr.shape[0], nodata=0,
                       crs=f"EPSG:{epsg}", transform=Affine.from_gdal(*gt),
                       compress="deflate", tiled=True) as dst:
        dst.write(arr, 1)
    tmp.rename(path)


def _write_cog(path: Path, z: np.ndarray, gt, epsg: int, cog: dict) -> None:
    import rasterio
    from rasterio.transform import Affine

    raw = path.with_name(path.name[:-4] + ".raw" + TMP_TIF_SUFFIX)
    tmp = path.with_name(path.name[:-4] + TMP_TIF_SUFFIX)
    arr = np.where(np.isfinite(z), z, NODATA).astype("f4")
    with rasterio.open(raw, "w", driver="GTiff", dtype="float32", count=1,
                       width=arr.shape[1], height=arr.shape[0], nodata=NODATA,
                       crs=f"EPSG:{epsg}", transform=Affine.from_gdal(*gt),
                       tiled=True, compress="deflate", BIGTIFF="IF_SAFER") as dst:
        dst.write(arr, 1)
    try:
        subprocess.run(
            ["gdal_translate", "-q", "-of", "COG", "-a_srs", f"EPSG:{epsg}",
             "-co", f"COMPRESS={cog['compress']}",
             "-co", f"PREDICTOR={cog['predictor']}",
             "-co", f"BLOCKSIZE={cog['blocksize']}",
             "-co", f"RESAMPLING={cog['overview_resampling']}",
             "-co", "OVERVIEWS=IGNORE_EXISTING", str(raw), str(tmp)],
            check=True)
    finally:
        raw.unlink(missing_ok=True)
    tmp.rename(path)


def _read(path: Path, dtype: str):
    import rasterio

    with rasterio.open(path) as src:
        a = src.read(1)
        gt, size, nd = src.transform.to_gdal(), (src.width, src.height), src.nodata
    if dtype == "f4":
        a = a.astype("f4")
        if nd is not None:
            a[a == nd] = np.nan
    return a, gt, size


def merge_tile(tile: str, contributors: list[Basin], mcfg: dict,
               spec: GridSpec, terr: dict[int, object] | None,
               territory_digest: str, source_ids: dict[str, int]) -> dict:
    """Write the merged elevation COG, provenance and basin-of-origin rasters
    for one tile, then its stats JSON (the completion marker). Skips when
    the stats fingerprint still matches the inputs and the outputs exist on
    the current grid."""
    nonreal = [source_ids["nodata"], source_ids["filled"]]
    out_dir = tiles_dir(mcfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    elev = out_dir / output_name(mcfg, tile)
    prov_p = out_dir / output_name(mcfg, tile, "_source")
    basin_p = out_dir / output_name(mcfg, tile, "_basin")
    sp = stats_path(mcfg, tile)
    fp = tile_fingerprint(mcfg, tile, contributors, spec, territory_digest,
                          nonreal)
    if sp.exists() and all(p.exists() for p in (elev, prov_p, basin_p)):
        old = json.loads(sp.read_text())
        if old.get("fingerprint") == fp and spec.tile_matches(tile, elev):
            return old
    for p in (elev, prov_p, basin_p, sp):
        p.unlink(missing_ok=True)

    gt, w, h = spec.tile_geotransform(tile)
    gt = tuple(gt)
    tol = float(mcfg["merge"]["restamp_tol_px"])
    epsg = int(mcfg["output"]["epsg"])
    zs, provs, gts = [], [], []
    for b in contributors:
        z, gt_in, size = _read(b.tile_path(tile), "f4")
        p, gt_p, size_p = _read(b.source_path(tile), "u1")
        if (gt_p, size_p) != (gt_in, size):
            raise ValueError(f"{tile}/{b.key}: provenance grid != elevation grid")
        check_restamp(gt_in, size, gt, (w, h), tol)
        zs.append(z)
        provs.append(p)
        gts.append(tuple(gt_in))

    codes = [b.code for b in contributors]
    stats = {"tile": tile, "fingerprint": fp,
             "contributors": [b.key for b in contributors]}
    if len(contributors) == 1:
        b, z = contributors[0], zs[0]
        if gts[0] == gt:
            mode = "copy"
            for src, dst in ((b.tile_path(tile), elev),
                             (b.source_path(tile), prov_p)):
                tmp = dst.with_name(dst.name[:-4] + TMP_TIF_SUFFIX)
                shutil.copy2(src, tmp)
                tmp.rename(dst)
        else:
            mode = "restamp"
            _write_cog(elev, z, gt, epsg, mcfg["output"]["cog"])
            back, _, _ = _read(elev, "f4")
            if not np.array_equal(back, z, equal_nan=True):
                raise RuntimeError(f"{tile}: restamped COG differs from input")
            _write_u1(prov_p, provs[0], gt, epsg)
        _write_u1(basin_p, np.where(np.isfinite(z), b.code, 0).astype("u1"),
                  gt, epsg)
        stats.update(mode=mode, counts={
            "single": int(np.isfinite(z).sum()),
            "nodata": int((~np.isfinite(z)).sum())})
    else:
        f = int(mcfg["merge"]["feather_px"])
        pad = f + 2
        lab = rasterize_territories(terr, padded_geotransform(gt, pad),
                                    (h + 2 * pad, w + 2 * pad))
        alphas = territory_alphas(lab, codes, f, pad)
        res = blend(zs, provs, alphas, nonreal)
        basin = np.zeros((h, w), "u1")
        for i, c in enumerate(codes):
            basin[res["winner"] == i] = c
        _write_cog(elev, res["z"], gt, epsg, mcfg["output"]["cog"])
        _write_u1(prov_p, res["prov"], gt, epsg)
        _write_u1(basin_p, basin, gt, epsg)
        stats.update(mode="blend", counts=res["counts"],
                     band_spread_m=res["band_spread_m"],
                     basin_px={b.key: int((res["winner"] == i).sum())
                               for i, b in enumerate(contributors)})
    if not spec.tile_matches(tile, elev):
        raise RuntimeError(f"{tile}: written tile does not match merged spec")
    _atomic_json(sp, stats)
    return stats


# --------------------------------------------------------------------------
# prepare / load prepared state
# --------------------------------------------------------------------------

def prepared_paths(mcfg: dict) -> dict[str, Path]:
    cfgd = output_root(mcfg) / "config"
    return {"grid": cfgd / "tile_grid.json",
            "territories": cfgd / "territories.gpkg",
            "region": cfgd / "region_polygon.gpkg",
            "tiles": cfgd / f"tiles_{mcfg['project']['region_id']}.geojson",
            "state": cfgd / "merge_prepared.json"}


def load_basin_polygons(mcfg: dict, basins: list[Basin]):
    """GeoDataFrame (EPSG:4269) of the basins' UNBUFFERED WBD HUC-4 polygons."""
    import geopandas as gpd

    w = mcfg["wbd"]
    gdf = gpd.read_file(w["gpkg"], layer=w.get("layer", "WBDHU4")).to_crs(4269)
    field = w.get("field", "huc4")
    want = {b.huc4: b for b in basins}
    gdf = gdf[gdf[field].astype(str).isin(want)].copy()
    missing = set(want) - set(gdf[field].astype(str))
    if missing:
        raise ValueError(f"HUC-4s not in {w['gpkg']}: {sorted(missing)}")
    gdf = gdf.dissolve(by=field).reset_index()
    gdf["code"] = [want[str(h)].code for h in gdf[field]]
    gdf["basin"] = [want[str(h)].key for h in gdf[field]]
    return gdf[["basin", "code", field, "geometry"]].rename(columns={field: "huc4"})


def _same_territories(a: Path, b: Path) -> bool:
    import geopandas as gpd
    import shapely

    da = gpd.read_file(a, layer="territories")
    db = gpd.read_file(b, layer="territories")
    if list(zip(da["code"], da["kind"])) != list(zip(db["code"], db["kind"])):
        return False
    return all(shapely.to_wkb(x) == shapely.to_wkb(y)
               for x, y in zip(da.geometry, db.geometry))


def file_digest(path: Path) -> str:
    """sha256 of a file's bytes (territories.gpkg: WKB does not survive a
    GPKG round trip byte-for-byte, the file itself is stable once written)."""
    hsh = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hsh.update(chunk)
    return hsh.hexdigest()


def prepare(mcfg: dict) -> dict:
    """Write the merged grid spec, territories, region polygon and tile list
    under <output root>/config/. Cheap; rerun whenever basins change."""
    import geopandas as gpd

    basins = load_basins(mcfg)
    merged_source_ids(basins)
    paths = prepared_paths(mcfg)
    paths["grid"].parent.mkdir(parents=True, exist_ok=True)

    specs = {b.key: GridSpec.load(b.root / "config" / "tile_grid.json")
             for b in basins}
    conv = mcfg["merge"]["grid_convention_from"]
    spec = build_grid_spec(specs, conv)
    for key, s in specs.items():
        if (s.collar_px, s.x_shift_px, s.y_shift_px, s.pixel_deg) != (
                spec.collar_px, spec.x_shift_px, spec.y_shift_px, spec.pixel_deg):
            log.info("basin %s uses a different non-CUDEM convention "
                     "(collar %s, shift %s/%s); its non-CUDEM tiles will be "
                     "restamped to %s's", key, s.collar_px, s.x_shift_px,
                     s.y_shift_px, conv)
    spec.dump(paths["grid"])

    polys = load_basin_polygons(mcfg, basins)
    tmp = paths["region"].with_suffix(".tmp.gpkg")
    polys.to_file(tmp, layer="basins", driver="GPKG")
    gpd.GeoDataFrame({"region": [mcfg["project"]["region_id"]]},
                     geometry=[polys.union_all()], crs=4269).to_file(
        tmp, layer="region", driver="GPKG")
    tmp.rename(paths["region"])

    terr = build_territories(dict(zip(polys["code"], polys.geometry)))
    tmp = paths["territories"].with_suffix(".tmp.gpkg")
    rows = [(c, kind, terr[c][i]) for c in terr
            for i, kind in enumerate(("polygon", "exterior"))]
    gpd.GeoDataFrame({"code": [r[0] for r in rows], "kind": [r[1] for r in rows]},
                     geometry=[r[2] for r in rows], crs=4269).to_file(
        tmp, layer="territories", driver="GPKG")
    # The gpkg's bytes change on every write and its digest is part of every
    # blended tile's fingerprint, so only replace it when the geometry did:
    # otherwise each `all` rerun after an interruption would redo every blend.
    if paths["territories"].exists() and _same_territories(tmp, paths["territories"]):
        tmp.unlink()
    else:
        tmp.rename(paths["territories"])

    tiles = discover_tiles(basins)
    import shapely
    feats = []
    for t, bs in tiles.items():
        gt, w, h = spec.tile_geotransform(t)
        feats.append({"tile": t, "basins": ",".join(b.key for b in bs),
                      "n_basins": len(bs),
                      "geometry": shapely.box(gt[0], gt[3] + h * gt[5],
                                              gt[0] + w * gt[1], gt[3])})
    gdf = gpd.GeoDataFrame(feats, crs=4269)
    tmp = paths["tiles"].with_suffix(".tmp.geojson")
    gdf.to_file(tmp, driver="GeoJSON")
    tmp.rename(paths["tiles"])

    state = {"territory_digest": file_digest(paths["territories"]),
             "n_tiles": len(tiles),
             "by_n_basins": {str(k): sum(1 for v in tiles.values() if len(v) == k)
                             for k in sorted({len(v) for v in tiles.values()})}}
    _atomic_json(paths["state"], state)
    log.info("prepared %d tiles %s", len(tiles), state["by_n_basins"])
    return state


def load_prepared(mcfg: dict):
    """(basins, spec, territories, digest, source_ids, tiles) from prepare()."""
    import geopandas as gpd

    paths = prepared_paths(mcfg)
    if not paths["state"].exists():
        raise SystemExit("run the 'prepare' step first")
    basins = load_basins(mcfg)
    spec = GridSpec.load(paths["grid"])
    tdf = gpd.read_file(paths["territories"], layer="territories")
    parts: dict[int, dict] = {}
    for c, kind, g in zip(tdf["code"].astype(int), tdf["kind"], tdf.geometry):
        parts.setdefault(c, {})[kind] = g
    terr = {c: (d["polygon"], d["exterior"]) for c, d in parts.items()}
    digest = json.loads(paths["state"].read_text())["territory_digest"]
    if file_digest(paths["territories"]) != digest:
        raise SystemExit("territories.gpkg does not match merge_prepared.json "
                         "-- rerun 'prepare'")
    return basins, spec, terr, digest, merged_source_ids(basins), discover_tiles(basins)


def build_vrt(mcfg: dict) -> Path:
    out = output_root(mcfg) / f"{mcfg['project']['name']}_{mcfg['project']['version']}.vrt"
    p = mcfg["project"]
    finals = [str(x) for x in product_tifs(tiles_dir(mcfg),
                                           f"{p['name']}_*_{p['version']}.tif")
              if not x.stem.endswith(("_source", "_basin"))]
    if not finals:
        raise SystemExit("no merged tiles yet")
    listfile = out.with_suffix(".inputs.txt")
    listfile.write_text("\n".join(finals) + "\n")
    subprocess.run(["gdalbuildvrt", "-q", "-overwrite", "-input_file_list",
                    str(listfile), str(out)], check=True)
    subprocess.run(["gdaladdo", "-q", "-r", "average", "-ro", str(out),
                    "16", "64", "256"], check=True,
                   env={**os.environ, "GDAL_NUM_THREADS": "4"})
    log.info("mosaic VRT %s (%d tiles)", out, len(finals))
    return out


# --------------------------------------------------------------------------
# QA
# --------------------------------------------------------------------------

def apply_seam_acceptance(seams: dict, qcfg: dict) -> dict:
    """Move documented exceptions out of a qa.check_seams result and set its
    pass flag. An accepted edge (qa.seams_accepted, {"a|b": max_m}) only
    passes while its step stays <= the recorded value; inherent size
    mismatches (e.g. an 8111-px CUDEM tile beside 8112-px neighbours) are
    accepted by exact pair and sizes (qa.size_mismatches_accepted)."""
    accepted = {k: float(v) for k, v in (qcfg.get("seams_accepted") or {}).items()}
    seams["accepted"] = {k: v for k, v in seams["flagged"].items()
                         if k in accepted and v <= accepted[k] + 1e-6}
    seams["flagged"] = {k: v for k, v in seams["flagged"].items()
                        if k not in seams["accepted"]}
    sm_ok = {k: list(v) for k, v in (qcfg.get("size_mismatches_accepted") or {}).items()}
    seams["accepted_size_mismatches"] = {
        k: v for k, v in seams["size_mismatches"].items() if sm_ok.get(k) == list(v)}
    seams["size_mismatches"] = {k: v for k, v in seams["size_mismatches"].items()
                                if k not in seams["accepted_size_mismatches"]}
    seams["pass"] = not seams["flagged"] and not seams["size_mismatches"]
    return seams


def run_qa(mcfg: dict) -> list[dict]:
    """Grid conformity, cross-tile seams, nodata inside the region polygon,
    and a summary of the per-tile merge stats. Returns the check dicts;
    the caller fails on any hard check whose ``pass`` is False.

    Accepted, documented exceptions come from the merge config's ``qa``
    block: ``basin_nodata_accepted_px`` (known gaps, see
    check_basin_nodata), ``seams_accepted`` ({"a|b": max_m}) -- an accepted
    edge only passes while its step stays <= the recorded value -- and
    ``size_mismatches_accepted`` ({"a|b": [n_a, n_b]}, exact match).
    """
    import geopandas as gpd

    from . import qa

    basins, spec, _, _, ids, tiles = load_prepared(mcfg)
    p = mcfg["project"]
    tile_paths = {}
    missing = []
    for t in tiles:
        e = tiles_dir(mcfg) / output_name(mcfg, t)
        if e.exists() and stats_path(mcfg, t).exists():
            tile_paths[t] = e
        else:
            missing.append(t)
    qcfg = dict(mcfg["qa"])
    full = {"qa": qcfg, "compositing": {"source_ids": ids}}
    checks = [{"check": "completeness", "pass": not missing,
               "n_expected": len(tiles), "missing": missing}]
    checks.append(qa.check_grid_conformity(tile_paths, spec=spec))

    seams = qa.check_seams(tile_paths, spec=spec, cfg=full)
    checks.append(apply_seam_acceptance(seams, qcfg))

    region = gpd.read_file(prepared_paths(mcfg)["region"], layer="region")
    checks.append(qa.check_basin_nodata(tile_paths, poly=region, cfg=full))

    modes, tiers, spread = {}, {}, {}
    flag_m = float(qcfg.get("band_spread_flag_m", 5.0))
    for t in tile_paths:
        s = json.loads(stats_path(mcfg, t).read_text())
        modes[s["mode"]] = modes.get(s["mode"], 0) + 1
        for k, v in s["counts"].items():
            tiers[k] = tiers.get(k, 0) + v
        bs = s.get("band_spread_m")
        if bs and bs["p99"] > flag_m:
            spread[t] = {"p99": round(bs["p99"], 3), "max": round(bs["max"], 3),
                         "basins": s["contributors"]}
    checks.append({"check": "merge_summary", "modes": modes,
                   "pixel_tiers": tiers,
                   "band_spread_review": dict(sorted(
                       spread.items(), key=lambda kv: -kv[1]["p99"])),
                   "band_spread_flag_m": flag_m,
                   "product": f"{p['name']}_<tile>_{p['version']}"})
    qa_dir = output_root(mcfg) / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(qa_dir / "qa_report.json", {"checks": checks})
    return checks
