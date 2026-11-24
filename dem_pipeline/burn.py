"""Channel burn: an optional post-processing step on the finished MERGED product.

The merged region tiles (dem_pipeline.merge) carry the lidar *water surface*
in every stream that has no bathymetry. The subgrid model samples the DEM at
native resolution and uses edge statistics, so a channel that is too shallow,
or that is dammed by a road embankment, blocks flow it should carry. This
module lowers mapped NHDPlus HR channels below that surface and writes the
result as a NEW tile set (its own product name, its own directory). The
merged product is only ever read. Design and validation: docs/ARCHITECTURE.md
§12; the method is the scratch prototype v3
(channel_geom_explore/proto/proto.py) moved onto the merged product.

Two phases, because the channel profile is a property of the NETWORK while
the product is tiled:

A. ``run_network`` (per HUC-4, vector + windowed reads, no basin raster):
   * tier 1 (DA >= t1_da inside an NHD river polygon): a downstream-monotone
     water surface sampled from lidar pixels along each reach, with mean
     depth D = t1_zarrabi_factor x Zarrabi mean-flow depth;
   * tier 2 (smaller reaches, and tier 1b = big reaches with no polygon): the
     HR line is snapped to the valley bottom by a least-cost path, given a
     downstream-monotone (cut-only) bed and thalweg = bed - 1.5 D; cuts deeper
     than cut_cap need a mapped road/rail crossing (culvert) and must not
     touch an NID dam keep-out buffer (a dam crest carries a road too); a
     short NHD conduit (connector / pipeline <= conduit_max_m: a mapped
     culvert) is itself a crossing; 1D depressions deeper than pit_deep
     (quarries) are never channel.
   The result is split into per-tile burn specs (spec/<huc4>/<tile>.npz) in
   global lattice coordinates, so two tiles sharing a collar rasterize the
   same path identically.

B. ``burn_tile`` (per tile, parallel, fingerprinted): rasterize the specs
   (tier 1 parabolic area-preserving section, tier 2 parabola + 4-connected
   spine; a tier-1 pixel whose own surface is more than t1_surface_tol_m
   above its assigned water surface -- a reservoir pool next to the reach
   below its dam -- is dropped by ``guard_tier1``; likewise a tier-2 pixel
   outside the path's 4-connected core more than t2_lateral_tol_m above the
   local bed), then ``apply_burn``: a pixel is lowered only if its provenance is
   burnable (lidar / residual fill -- never bathymetry), the merged
   basin-of-origin raster says it belongs to that network's HUC-4, it has
   data, and the target is BELOW it. Lowered pixels get provenance burn_id.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import subprocess
import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import TMP_TIF_SUFFIX, product_tifs
from . import path as repo_path
from . import merge
from .grid import GridSpec

log = logging.getLogger(__name__)

# Part of every network and tile fingerprint: bump when burn semantics change
# so a rerun rebuilds instead of trusting existing files.
BURN_VERSION = 4
# The network pass has its own version and ignores parameters only the
# per-tile apply step uses, so changing what may be lowered re-burns tiles
# without redoing the (~30 min per basin) network.
NETWORK_VERSION = 4
APPLY_ONLY_KEYS = ("burn_id", "burnable_ids", "burnable_outside_water_ids",
                   "t1_surface_tol_m", "t2_lateral_tol_m")

NODATA = merge.NODATA
M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQ = 111320.0


# --------------------------------------------------------------------------
# configuration / paths
# --------------------------------------------------------------------------

def load_burn_config(path: str | Path) -> dict:
    with open(path) as f:
        b = yaml.safe_load(f)
    b["_path"] = str(Path(path).resolve())
    mp = Path(b["merge_config"])
    if not mp.is_absolute():
        mp = repo_path(str(mp))
    b["_merge"] = merge.load_merge_config(mp)
    return b


def out_root(b: dict) -> Path:
    return Path(b["output"]["root"])


def in_tile(b: dict, tile: str, suffix: str = "") -> Path:
    """A tile of the merged (input) product."""
    m = b["_merge"]
    return merge.tiles_dir(m) / merge.output_name(m, tile, suffix)


def out_tile(b: dict, tile: str, suffix: str = "") -> Path:
    p = b["project"]
    return out_root(b) / "tiles" / f"{p['name']}_{tile}_{p['version']}{suffix}.tif"


def prep_dir(b: dict, huc4: str) -> Path:
    return out_root(b) / "prep" / str(huc4)


def spec_path(b: dict, huc4: str, tile: str) -> Path:
    return out_root(b) / "spec" / str(huc4) / f"{tile}.npz"


def network_manifest(b: dict, huc4: str) -> Path:
    return out_root(b) / "manifests" / f"network_{huc4}.json"


def stats_path(b: dict, tile: str) -> Path:
    return out_root(b) / "manifests" / "burn_stats" / f"{tile}.json"


def load_input(b: dict) -> tuple[GridSpec, list[str]]:
    """The merged product's grid spec and tile list (written by its prepare)."""
    paths = merge.prepared_paths(b["_merge"])
    if not paths["grid"].exists() or not paths["tiles"].exists():
        raise SystemExit(f"merged product not prepared: {paths['grid'].parent}")
    spec = GridSpec.load(paths["grid"])
    with open(paths["tiles"]) as f:
        tiles = sorted(ft["properties"]["tile"] for ft in json.load(f)["features"])
    return spec, tiles


@dataclass(frozen=True)
class Params:
    burn_id: int
    burnable_ids: tuple
    burnable_outside_water_ids: tuple
    wse_ids: tuple
    da_min_km2: float
    skip_tidal: bool
    t1_da_km2: float
    t1_zarrabi_factor: float
    t1_corridor_m: float
    t1_sample_m: float
    t2_factor: float
    t2_min_depth_m: float
    t2_narrow_da_km2: float
    node_snap_m: float
    snap_corridor_m: float
    window_margin_m: float
    cut_cap_m: float
    crossing_buffer_m: float
    pit_deep_m: float
    max_nan_frac: float
    dam_buffer_m: float
    t1_surface_tol_m: float
    t2_lateral_tol_m: float
    conduit_ftypes: tuple = (334, 428)
    conduit_max_m: float = 300.0

    @classmethod
    def from_config(cls, b: dict) -> "Params":
        d = dict(b["burn"])
        d["burnable_ids"] = tuple(int(i) for i in d["burnable_ids"])
        d["wse_ids"] = tuple(int(i) for i in d["wse_ids"])
        if "conduit_ftypes" in d:
            d["conduit_ftypes"] = tuple(int(i) for i in d["conduit_ftypes"])
        d["burnable_outside_water_ids"] = tuple(
            int(i) for i in d.get("burnable_outside_water_ids") or ())
        return cls(**d)


def pixel_m(res_deg: float, lat: float) -> tuple[float, float]:
    """(dx, dy) in metres of a res_deg pixel at latitude lat."""
    return (res_deg * M_PER_DEG_LON_EQ * math.cos(math.radians(lat)),
            res_deg * M_PER_DEG_LAT)


def _file_sig(p: Path) -> list:
    p = Path(p)
    if p.is_dir():          # a FileGDB is a directory
        files = sorted(q for q in p.rglob("*") if q.is_file())
        return [str(p), sum(q.stat().st_size for q in files),
                max((q.stat().st_mtime_ns for q in files), default=0)]
    st = p.stat()
    return [str(p), st.st_size, st.st_mtime_ns]


def _digest(doc) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def _atomic_json(path: Path, doc: dict) -> None:
    merge._atomic_json(path, doc)


# --------------------------------------------------------------------------
# the global lattice and a resampling-free mosaic reader
# --------------------------------------------------------------------------

class Lattice:
    """One integer pixel lattice over every tile of a product.

    All merged tiles share pixel size and differ in origin by whole pixels
    (to within the merge's restamp tolerance); a burn spec stored as global
    (row, col) therefore maps to exactly one pixel in every tile containing
    it -- including both tiles of a shared collar -- with no resampling.
    """

    def __init__(self, spec: GridSpec, tiles: list[str], tol_px: float = 0.01):
        gts = {t: spec.tile_geotransform(t) for t in tiles}
        self.res = float(spec.pixel_deg)
        for t, (gt, w, h) in gts.items():
            # CUDEM vintages carry slightly different pixel sizes; what matters
            # is the drift across the tile, in pixels
            drift = max(abs(gt[1] - self.res), abs(gt[5] + self.res)) / self.res * max(w, h)
            if drift > tol_px:
                raise ValueError(f"{t}: pixel size {gt[1]}/{gt[5]} drifts {drift:.4f} px "
                                 f"from {self.res} across the tile (tol {tol_px})")
        self.x0 = min(gt[0] for gt, _, _ in gts.values())
        self.y0 = max(gt[3] for gt, _, _ in gts.values())
        self.off: dict[str, tuple[int, int, int, int]] = {}
        # Tiles registered off the product lattice (e.g. the half-pixel-shifted
        # CUDEM n37x00_w076x00 in region 02) cannot share global pixel
        # coordinates; they are never read by the network pass and are copied
        # through unburned (burn_tile refuses one that a spec reaches).
        self.off_lattice: dict[str, tuple[float, float]] = {}
        for t, (gt, w, h) in sorted(gts.items()):
            c = (gt[0] - self.x0) / self.res
            r = (self.y0 - gt[3]) / self.res
            ci, ri = round(c), round(r)
            if abs(c - ci) > tol_px or abs(r - ri) > tol_px:
                self.off_lattice[t] = (r - ri, c - ci)
                log.warning("%s: origin %+.4f/%+.4f px off the product lattice -- "
                            "excluded from the burn (copied unchanged)", t, r - ri, c - ci)
                continue
            self.off[t] = (ri, ci, h, w)

    def transform(self, I0: int = 0, J0: int = 0):
        from rasterio.transform import Affine
        return Affine(self.res, 0.0, self.x0 + J0 * self.res,
                      0.0, -self.res, self.y0 - I0 * self.res)

    def rowcol(self, x, y):
        I = np.floor((self.y0 - np.asarray(y, "f8")) / self.res).astype("i8")
        J = np.floor((np.asarray(x, "f8") - self.x0) / self.res).astype("i8")
        return I, J

    def xy(self, I, J):
        return (self.x0 + (np.asarray(J, "f8") + 0.5) * self.res,
                self.y0 - (np.asarray(I, "f8") + 0.5) * self.res)

    def lat(self, I) -> float:
        return float(self.y0 - (float(np.mean(I)) + 0.5) * self.res)

    def window(self, tile: str) -> tuple[int, int, int, int]:
        r, c, h, w = self.off[tile]
        return r, r + h, c, c + w

    def tiles_overlapping(self, I0, I1, J0, J1) -> list[str]:
        return [t for t, (r, c, h, w) in self.off.items()
                if r < I1 and r + h > I0 and c < J1 and c + w > J0]


class Mosaic:
    """Windowed reads across tiles on a Lattice (first tile in name order
    wins inside a shared collar; outside every tile: NaN / 0)."""

    def __init__(self, lattice: Lattice, path_fn, max_open: int = 32):
        self.L = lattice
        self.path_fn = path_fn          # (tile, kind) -> Path; kind z|prov|basin
        self.max_open = max_open
        self._ds: OrderedDict = OrderedDict()

    def _open(self, tile, kind):
        import rasterio
        k = (tile, kind)
        if k in self._ds:
            self._ds.move_to_end(k)
            return self._ds[k]
        ds = rasterio.open(self.path_fn(tile, kind))
        self._ds[k] = ds
        if len(self._ds) > self.max_open:
            self._ds.popitem(last=False)[1].close()
        return ds

    def close(self):
        for ds in self._ds.values():
            ds.close()
        self._ds.clear()

    def read(self, kind: str, I0: int, I1: int, J0: int, J1: int) -> np.ndarray:
        from rasterio.windows import Window
        is_z = kind == "z"
        out = np.full((I1 - I0, J1 - J0), np.nan if is_z else 0,
                      "f4" if is_z else "u1")
        done = np.zeros(out.shape, bool)
        for t in self.L.tiles_overlapping(I0, I1, J0, J1):
            r, c, h, w = self.L.off[t]
            a0, a1 = max(I0, r), min(I1, r + h)
            b0, b1 = max(J0, c), min(J1, c + w)
            ds = self._open(t, kind)
            arr = ds.read(1, window=Window(b0 - c, a0 - r, b1 - b0, a1 - a0))
            if is_z:
                arr = arr.astype("f4")
                if ds.nodata is not None:
                    arr[arr == ds.nodata] = np.nan
            sub = out[a0 - I0:a1 - I0, b0 - J0:b1 - J0]
            dn = done[a0 - I0:a1 - I0, b0 - J0:b1 - J0]
            sub[~dn] = arr[~dn]
            dn[:] = True
        return out


# --------------------------------------------------------------------------
# small numeric pieces (unit-tested)
# --------------------------------------------------------------------------

def pava_dec(y: np.ndarray) -> np.ndarray:
    """Least-squares non-increasing fit (pool-adjacent-violators)."""
    y = np.asarray(y, "f8")
    if len(y) < 2:
        return y.copy()
    vals, wts, cnt = [], [], []
    for v in y:
        vals.append(v); wts.append(1.0); cnt.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            w = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            n = cnt[-2] + cnt[-1]
            vals[-2:] = [v2]; wts[-2:] = [w]; cnt[-2:] = [n]
    return np.repeat(vals, cnt)


def path_bed(zp: np.ndarray, cap_in: float, pit_deep: float):
    """Downstream-monotone cut-only bed along a snapped path.

    Returns (bed, quarry_mask, cap_ignored). zp is smoothed by a 5-pt
    median; points inside a 1D depression deeper than pit_deep (quarries,
    real pits -- not channel) are excluded from the bed, and an upstream cap
    more than pit_deep below this path's own fill level is ignored (it came
    from a pit upstream)."""
    from scipy import ndimage as ndi
    zs = ndi.median_filter(zp, size=5, mode="nearest") if len(zp) >= 5 else zp
    fill1d = np.maximum.accumulate(zs[::-1])[::-1]
    fd = fill1d - zs
    lab, nq = ndi.label(fd > 0.5)
    deep = [k for k in range(1, nq + 1) if fd[lab == k].max() > pit_deep]
    quarry = np.isin(lab, deep)
    ignored = False
    if cap_in < fill1d[0] - pit_deep:
        cap_in, ignored = np.inf, True
    # The monotone minimum restarts after every quarry gap: carrying the
    # pre-pit level across it put that level on whatever lies beyond the
    # gap -- on a reach that climbs a ridge (an HR routing error that is
    # one long 'pit') that was a hilltop, cut 70-147 m (region 02,
    # 2026-11-20). The upstream cap applies only to the run starting at 0.
    bed = np.where(quarry, np.inf, zs).astype("f8")
    lab, nrun = ndi.label(~quarry)
    for k in range(1, nrun + 1):
        idx = np.flatnonzero(lab == k)
        seg = bed[idx] if idx[0] else np.minimum(bed[idx], cap_in)
        bed[idx] = np.minimum.accumulate(seg)
    bed = np.where(np.isfinite(bed), bed, zp)
    return bed, quarry, ignored


def cap_uncrossed_cuts(zp, bed, at_crossing, cut_cap: float, at_dam=None):
    """Limit cut segments deeper than cut_cap that touch no mapped crossing,
    or that touch a dam keep-out (a dam crest usually carries a road, and a
    real dam must stay a dam).

    ``at_crossing(idx) -> bool`` / ``at_dam(idx) -> bool`` say whether any
    path point in idx lies at a road/rail crossing / inside an NID dam
    buffer. Returns (bed, capped_mask, n_capped, n_at_crossing, n_at_dam);
    n_capped includes the n_at_dam segments."""
    from scipy import ndimage as ndi
    cut0 = zp - bed
    lab, nseg = ndi.label(cut0 > 0.5)
    capped = np.zeros(len(bed), bool)
    n_capped = n_x = n_dam = 0
    for k in range(1, nseg + 1):
        idx = np.where(lab == k)[0]
        if cut0[idx].max() <= cut_cap:
            continue
        if at_dam is not None and at_dam(idx):
            capped[idx] = True
            n_capped += 1
            n_dam += 1
            continue
        if at_crossing(idx):
            n_x += 1
            continue
        capped[idx] = True
        n_capped += 1
    return (np.where(capped, np.maximum(bed, zp - cut_cap), bed), capped,
            n_capped, n_x, n_dam)


def thalweg(bed, D, quarry, capped, cap_in: float, pit_deep: float):
    """bed - 1.5 D made monotone across the network (a junction where D
    shrinks downstream must not step up); capped segments keep their
    retained obstruction; quarry points burn nothing (inf) and do not
    propagate: the running level is reset by a quarry gap, so the first
    point beyond it starts from its own bed (see ``path_bed``). Returns
    (thal, cap_ignored)."""
    raw = bed - 1.5 * D
    thal = np.empty_like(raw)
    cur, ignored = cap_in, False
    if cur < raw[0] - pit_deep:
        cur, ignored = np.inf, True
    for k in range(len(raw)):
        if quarry[k]:
            thal[k] = np.inf
            cur = np.inf
            continue
        cur = raw[k] if capped[k] else min(cur, raw[k])
        thal[k] = cur
    return thal, ignored


def guard_tier1(t1, wse, z, tol: float):
    """Drop tier-1 targets on pixels whose own surface is more than tol above
    the water surface they were assigned. The tier-1 surface comes from the
    NEAREST centerline sample, so a reservoir pool in the corridor polygon
    next to the reach below its dam would otherwise be trenched down to the
    downstream river (Cannonsville: 349 -> 307 m). Returns (t1, n_dropped),
    n_dropped counting only pixels the target would actually have lowered."""
    bad = np.isfinite(t1) & ~(z <= wse + tol)
    n = int((bad & (t1 < z)).sum())
    return np.where(bad, np.inf, t1).astype(t1.dtype), n


def apply_burn(z, prov, basin, targets: dict[int, np.ndarray],
               burnable_ids, burn_id: int, outside_water_ids=(),
               outside_water: dict[int, np.ndarray] | None = None):
    """Lower z to the burn target where allowed. targets maps basin code
    (huc4 % 100) -> target array (inf = no burn). A pixel is lowered only
    when its basin code matches the target's, it has data, the target is
    strictly below it, and its provenance is burnable -- either always
    (burnable_ids: lidar, residual fill) or only OUTSIDE that basin's NHD
    water polygons (outside_water_ids, with outside_water[code] the mask):
    survey rasters such as EAARL-B carry merged land topo off-channel, which
    is topo, not bathymetry. Bathymetry inside water, other basins' pixels
    and nodata are untouched and nothing is raised."""
    z_new = z.copy()
    prov_new = prov.copy()
    ok = np.isin(prov, np.asarray(burnable_ids, "u1")) & np.isfinite(z)
    topo = np.isin(prov, np.asarray(outside_water_ids, "u1")) & np.isfinite(z)
    for code, tgt in targets.items():
        allowed = ok
        if topo.any() and outside_water is not None and code in outside_water:
            allowed = ok | (topo & outside_water[code])
        m = allowed & (basin == code) & np.isfinite(tgt) & (tgt < z_new)
        z_new[m] = tgt[m]
        prov_new[m] = burn_id
    depth = np.where(np.isfinite(z), z - z_new, np.nan).astype("f4")
    return z_new, prov_new, depth


# --------------------------------------------------------------------------
# prepare: per-HUC-4 ML attribute table and road/rail crossings
# --------------------------------------------------------------------------

def _huc4_polygons(b: dict):
    import geopandas as gpd
    w = b["_merge"]["wbd"]
    g = gpd.read_file(w["gpkg"], layer=w["layer"]).to_crs(4269)
    return {str(h): geom for h, geom in zip(g[w["field"]].astype(str), g.geometry)}


def prepare(b: dict) -> None:
    """Check provenance ids against the merged product, then build each
    basin's ML attribute table (prep/<huc4>/ml.gpkg: Zarrabi mean-flow
    depth/width + STREAM-geo RF width and tidal flag, by NHDPlusV2.1 COMID)
    , road/rail crossing lines (prep/<huc4>/roads.gpkg, TIGER) and NID dam
    points (prep/<huc4>/dams.gpkg). All are built once; delete a file to
    rebuild it."""
    P = Params.from_config(b)
    ids = merge.merged_source_ids(merge.load_basins(b["_merge"]))
    vals = set(ids.values())
    if P.burn_id in vals:
        raise ValueError(f"burn_id {P.burn_id} is already a source id: {ids}")
    for k in ("burnable_ids", "wse_ids", "burnable_outside_water_ids"):
        bad = set(getattr(P, k)) - vals
        if bad:
            raise ValueError(f"{k} {sorted(bad)} are not source ids: {ids}")
    for k in ("ehydro", "eaarlb", "cudem19", "coned", "nos_grids"):
        if k in ids and ids[k] in P.burnable_ids:
            raise ValueError(f"bathymetry source {k} must never be burnable")
    polys = _huc4_polygons(b)
    for bas in b["basins"]:
        huc = str(bas["huc4"])
        d = prep_dir(b, huc)
        d.mkdir(parents=True, exist_ok=True)
        poly = polys[huc]
        if not (d / "ml.gpkg").exists():
            _write_gpkg(build_ml_table(b, poly), d / "ml.gpkg")
            log.info("%s: ML attribute table written", huc)
        if not (d / "roads.gpkg").exists():
            _write_gpkg(build_roads(b, poly), d / "roads.gpkg")
            log.info("%s: road/rail crossings written", huc)
        if not (d / "dams.gpkg").exists():
            from .download import fetch
            csv = fetch(b["data"]["nid_csv_url"], out_root(b) / "prep" / "nid" / "nid_nation.csv")
            dams = build_dams(csv, poly)
            _write_gpkg(dams, d / "dams.gpkg")
            log.info("%s: %d NID dams written", huc, len(dams))


def _write_gpkg(gdf, path: Path) -> None:
    tmp = path.with_suffix(".tmp.gpkg")
    tmp.unlink(missing_ok=True)
    gdf.to_file(tmp, driver="GPKG")
    tmp.rename(path)


def build_ml_table(b: dict, poly):
    import geopandas as gpd
    import pandas as pd
    z = gpd.read_file(b["data"]["zarrabi_shp"], bbox=tuple(poly.bounds),
                      columns=["COMID", "TotDASqKM", "mf_depth", "mf_width"])
    z = z.to_crs(4269)
    z = z[z.intersects(poly)].copy()
    z["geometry"] = z.geometry.force_2d()
    sg = pd.read_csv(b["data"]["streamgeo_csv"],
                     usecols=["COMID", "RF_Width_m", "Tidal_Flag"])
    sg = sg.drop_duplicates("COMID")
    return z.merge(sg, on="COMID", how="left")


def build_roads(b: dict, poly):
    import geopandas as gpd
    import pandas as pd
    from .download import fetch
    yr = b["data"]["tiger_year"]
    tdir = out_root(b) / "prep" / "tiger"
    base = f"https://www2.census.gov/geo/tiger"
    cz = fetch(f"{base}/GENZ{yr}/shp/cb_{yr}_us_county_500k.zip",
               tdir / f"cb_{yr}_us_county_500k.zip")
    pad = poly.buffer(0.02)
    cty = gpd.read_file(f"zip://{cz}", bbox=tuple(pad.bounds)).to_crs(4269)
    cty = cty[cty.intersects(pad)]
    parts = []
    bbox = tuple(pad.bounds)
    for geoid in sorted(cty["GEOID"]):
        f = fetch(f"{base}/TIGER{yr}/ROADS/tl_{yr}_{geoid}_roads.zip",
                  tdir / f"tl_{yr}_{geoid}_roads.zip")
        g = gpd.read_file(f"zip://{f}", bbox=bbox, columns=["MTFCC"])
        if len(g):
            parts.append(g.to_crs(4269).assign(kind="road"))
    f = fetch(f"{base}/TIGER{yr}/RAILS/tl_{yr}_us_rails.zip",
              tdir / f"tl_{yr}_us_rails.zip")
    g = gpd.read_file(f"zip://{f}", bbox=bbox, columns=["MTFCC"])
    if len(g):
        parts.append(g.to_crs(4269).assign(kind="rail"))
    log.info("roads from %d counties", len(cty))
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4269)


def build_dams(csv_path, poly, pad_deg: float = 0.05):
    """NID dam points (with crest length) within pad_deg of the basin. The
    national NID CSV starts with a 'Data Last Updated' line before its
    header."""
    import geopandas as gpd
    import pandas as pd
    with open(csv_path) as f:
        first = f.readline()
    skip = 1 if first.startswith("Data Last Updated") else 0
    cols = {"NID ID": "nid_id", "Dam Name": "name", "Latitude": "lat",
            "Longitude": "lon", "Dam Length (Ft)": "length_ft",
            "NID Height (Ft)": "height_ft"}
    n = pd.read_csv(csv_path, skiprows=skip, usecols=list(cols), low_memory=False)
    n = n.rename(columns=cols).dropna(subset=["lat", "lon"])
    x0, y0, x1, y1 = poly.buffer(pad_deg).bounds
    n = n[n.lon.between(x0, x1) & n.lat.between(y0, y1)]
    g = gpd.GeoDataFrame(n, geometry=gpd.points_from_xy(n.lon, n.lat), crs=4269)
    g = g[g.intersects(poly.buffer(pad_deg))].drop(columns=["lat", "lon"])
    for c in ("length_ft", "height_ft"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g.reset_index(drop=True)


def dam_buffers(dams, buffer_m: float, metric_epsg: int):
    """Keep-out polygon per dam: buffer_m + half the crest length around the
    NID point (NID points sit anywhere along a crest up to ~1 km long)."""
    import geopandas as gpd
    if dams is None or not len(dams):
        return gpd.GeoDataFrame(geometry=[], crs=4269)
    half = 0.5 * 0.3048 * dams["length_ft"].fillna(0.0).to_numpy(dtype="f8")
    m = dams.to_crs(metric_epsg)
    return gpd.GeoDataFrame(geometry=m.buffer(buffer_m + half), crs=metric_epsg).to_crs(4269)


# --------------------------------------------------------------------------
# phase A: network
# --------------------------------------------------------------------------

class Net:
    """Everything the network pass needs (one HUC-4)."""

    def __init__(self, b: dict, bas: dict, lattice: Lattice, mosaic: Mosaic):
        import pyproj
        self.b, self.bas, self.L, self.M = b, bas, lattice, mosaic
        self.P = Params.from_config(b)
        self.metric = int(b["data"]["metric_epsg"])
        self.to_geo = pyproj.Transformer.from_crs(self.metric, 4269, always_xy=True)
        self.roads = None
        self.dams = None            # dam keep-out polygons (dam_buffers)
        self.report: dict = {}

    def px(self, lat):
        return pixel_m(self.L.res, lat)

    def sample(self, gm, step):
        """Points every step metres along metric geometry gm -> (d, I, J)."""
        import shapely
        d = np.arange(0.0, gm.length + 1e-6, step)
        pts = shapely.line_interpolate_point(gm, d)
        lon, lat = self.to_geo.transform(shapely.get_x(pts), shapely.get_y(pts))
        I, J = self.L.rowcol(lon, lat)
        return d, I, J

    def dam(self, I0, I1, J0, J1):
        """Window mask of px inside a dam keep-out polygon."""
        from rasterio.features import rasterize
        shape = (I1 - I0, J1 - J0)
        if self.dams is None or not len(self.dams):
            return np.zeros(shape, bool)
        x0, y1 = self.L.x0 + J0 * self.L.res, self.L.y0 - I0 * self.L.res
        x1, y0 = self.L.x0 + J1 * self.L.res, self.L.y0 - I1 * self.L.res
        idx = self.dams.sindex.query(_box(x0, y0, x1, y1))
        if not len(idx):
            return np.zeros(shape, bool)
        return rasterize([(g, 1) for g in self.dams.geometry.iloc[idx]],
                         out_shape=shape, transform=self.L.transform(I0, J0),
                         all_touched=True).astype(bool)

    def crossing(self, I0, I1, J0, J1, lat):
        """Window mask of px within crossing_buffer_m of a mapped road/rail."""
        from rasterio.features import rasterize
        from scipy import ndimage as ndi
        shape = (I1 - I0, J1 - J0)
        if self.roads is None or not len(self.roads):
            return np.zeros(shape, bool)
        x0, y1 = self.L.x0 + J0 * self.L.res, self.L.y0 - I0 * self.L.res
        x1, y0 = self.L.x0 + J1 * self.L.res, self.L.y0 - I1 * self.L.res
        idx = self.roads.sindex.query(_box(x0, y0, x1, y1))
        if not len(idx):
            return np.zeros(shape, bool)
        m = rasterize([(g, 1) for g in self.roads.geometry.iloc[idx]],
                      out_shape=shape, transform=self.L.transform(I0, J0),
                      all_touched=True).astype(bool)
        dx, dy = self.px(lat)
        return ndi.distance_transform_edt(~m, sampling=(dy, dx)) <= self.P.crossing_buffer_m


def _box(x0, y0, x1, y1):
    import shapely
    return shapely.box(x0, y0, x1, y1)


def network_line_mask(fl, P) -> np.ndarray:
    """In-network HR lines the burn routes: streams (460), artificial paths
    (558) and short conduits -- connectors / pipelines (P.conduit_ftypes) no
    longer than P.conduit_max_m, i.e. mapped culverts. Dropping those broke
    the network at road crossings: the burn stopped at the culvert inlet and
    the embankment dammed the reach above (Plymouth Creek, PA, 6 m ponds;
    ~11,000 such gaps in region 02, 94 % of them <= 300 m). Longer conduits
    are buried streams / aqueducts and get no surface trench."""
    ft = fl.FType.to_numpy()
    conduit = np.isin(ft, P.conduit_ftypes) & (fl.LengthKM.to_numpy() * 1e3 <= P.conduit_max_m)
    return (np.isin(ft, (460, 558)) | conduit) & (fl.InNetwork.to_numpy() == 1)


def load_lines(net: Net):
    """In-network HR flowlines (network_line_mask) with VAA, DA >= da_min,
    lake flag and ``conduit`` flag."""
    import geopandas as gpd
    import pandas as pd
    gdb = net.bas["hr_gdb"]
    fl = gpd.read_file(gdb, layer="NHDFlowline",
                       columns=["NHDPlusID", "FType", "WBArea_Permanent_Identifier",
                                "GNIS_Name", "InNetwork", "LengthKM"])
    fl = fl[network_line_mask(fl, net.P)].copy()
    fl["conduit"] = fl.FType.isin(net.P.conduit_ftypes)
    vaa = gpd.read_file(gdb, layer="NHDPlusFlowlineVAA", read_geometry=False,
                        columns=["NHDPlusID", "FromNode", "ToNode", "HydroSeq",
                                 "TotDASqKm", "DivDASqKm", "StreamOrde", "Divergence"])
    fl = fl.merge(pd.DataFrame(vaa), on="NHDPlusID")
    # superset: the effective DA (attach_ml) is <= TotDASqKm, and the
    # da_min cut is applied to it in run_network
    fl = fl[fl.TotDASqKm >= net.P.da_min_km2].copy()
    fl["geometry"] = fl.geometry.force_2d().line_merge(directed=True)
    fl = fl[fl.geometry.geom_type == "LineString"].to_crs(4269)
    wb = gpd.read_file(gdb, layer="NHDWaterbody", read_geometry=False,
                       columns=["Permanent_Identifier", "FType"])
    lakes = set(wb.loc[wb.FType.isin([390, 436]), "Permanent_Identifier"])
    fl["in_lake"] = fl.WBArea_Permanent_Identifier.isin(lakes)
    return fl.reset_index(drop=True)


def attach_ml(fl, ml, metric_epsg: int):
    """Copy V2.1 ML attributes onto HR flowlines and set the effective
    drainage area ``DA`` used everywhere downstream.

    A V2.1 reach within 200 m matches if EITHER HR drainage area --
    TotDASqKm or the divergence-routed DivDASqKm -- is within [0.5, 2] of
    its DA; the closest wins, and the HR field that matched becomes DA.
    Unmatched lines keep TotDASqKm. Why both: HR's TotDASqKm routes whole
    basins through canals (the Delaware's 16,000 km2 through the D&R Canal
    into the Millstone/Raritan: no V2.1 match, so no Zarrabi depth and the
    rivers went unburned), where DivDASqKm is right; but on most lines
    where the two disagree (braids, marsh creeks) V2.1 sides with
    TotDASqKm (71-89 % region 02), so DivDASqKm is not used blindly."""
    import geopandas as gpd
    cols = ["COMID", "mf_depth", "mf_width", "RF_Width_m", "Tidal_Flag"]
    a = fl[["NHDPlusID", "TotDASqKm", "DivDASqKm", "geometry"]].to_crs(metric_epsg)
    v = ml[["TotDASqKM", "geometry"] + cols].to_crs(metric_epsg)
    v["geometry"] = v.buffer(200)
    j = gpd.sjoin(a, v, predicate="intersects")
    ref = j.TotDASqKM.clip(lower=1e-3)
    lt = np.abs(np.log(j.TotDASqKm.clip(lower=1e-3) / ref))
    ld = np.abs(np.log(j.DivDASqKm.clip(lower=1e-3) / ref))
    j["lr"] = np.minimum(lt, ld)
    j["DA"] = np.where(ld < lt, j.DivDASqKm, j.TotDASqKm)
    j = j[j.lr < np.log(2)].sort_values("lr").drop_duplicates("NHDPlusID")
    out = fl.merge(j[["NHDPlusID", "DA"] + cols], on="NHDPlusID", how="left")
    out["DA"] = out.DA.fillna(out.TotDASqKm)
    return out


def curve_index(province, section, curves: list[dict]) -> int:
    """Index of the first curve matching PROVINCE / SECTION."""
    p, s = str(province).upper(), str(section).upper()
    for i, c in enumerate(curves):
        if "province" in c and c["province"].upper() not in p:
            continue
        if "section" in c and c["section"].upper() not in s:
            continue
        return i
    raise ValueError("curves: no default entry (one with neither province nor section)")


def curve_for(province, section, curves: list[dict]) -> tuple[float, float]:
    c = curves[curve_index(province, section, curves)]
    return float(c["c"]), float(c["e"])


def curve_label(c: dict) -> str:
    """Short name of a curves entry, e.g. 'PLATEAU / CATSKILL'."""
    return " / ".join(c[k] for k in ("province", "section") if k in c) or "default"


def attach_curve(fl, physio_shp, curves):
    import geopandas as gpd
    ph = gpd.read_file(physio_shp)
    if ph.crs is None:
        ph = ph.set_crs(4269)
    mid = gpd.GeoDataFrame(geometry=fl.geometry.interpolate(0.5, normalized=True),
                           crs=fl.crs, index=fl.index).to_crs(ph.crs)
    j = gpd.sjoin(mid, ph[["PROVINCE", "SECTION", "geometry"]], how="left",
                  predicate="within")
    j = j[~j.index.duplicated()].reindex(fl.index)
    ci = [curve_index(p, s, curves) for p, s in zip(j.PROVINCE, j.SECTION)]
    fl = fl.copy()
    fl["curve_idx"] = ci
    fl["bf_c"] = [float(curves[i]["c"]) for i in ci]
    fl["bf_e"] = [float(curves[i]["e"]) for i in ci]
    fl["physio"] = j.SECTION.values
    return fl


def _corridor_poly(polys, lines, corridor_m, metric_epsg):
    """River polygons touching ``lines``, clipped to corridor_m of them."""
    import shapely
    if not len(lines):
        return shapely.Polygon()
    u = lines.geometry.union_all()
    hit = polys[polys.intersects(u)]
    if not len(hit):
        return shapely.Polygon()
    buf = lines.to_crs(metric_epsg).buffer(corridor_m).to_crs(4269).union_all()
    return shapely.make_valid(hit.geometry.union_all().intersection(buf))


def tier2_depth(da, bf_c, bf_e, P) -> np.ndarray:
    """Tier-2 mean depth: t2_factor x regional bankfull curve, floored."""
    return np.maximum(P.t2_factor * np.asarray(bf_c, "f8") * np.asarray(da, "f8")
                      ** np.asarray(bf_e, "f8"), P.t2_min_depth_m)


def _tier1_depth(t1p, P) -> tuple[np.ndarray, np.ndarray]:
    """Tier-1 mean depth D per reach and the mask of fallback reaches.

    D = t1_zarrabi_factor x median Zarrabi depth over tier-1 reaches with DA
    within +-25 % (smoothed along the network). Where no such reach has a
    Zarrabi depth, D falls back to the tier-2 regional curve -- never NaN: a
    NaN D silently burned nothing (19 % of 0203's tier-1 samples, the
    Raritan/Millstone, before the DA fix)."""
    da = t1p.DA.to_numpy(dtype="f8")
    mf = t1p.mf_depth.to_numpy(dtype="f8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sm = np.array([np.nanmedian(mf[(da / x >= 0.8) & (da / x <= 1.25)])
                       for x in da]) if len(da) else np.zeros(0)
    D = P.t1_zarrabi_factor * sm
    fb = ~np.isfinite(D)
    if fb.any():
        D[fb] = tier2_depth(da[fb], t1p.bf_c.to_numpy()[fb], t1p.bf_e.to_numpy()[fb], P)
    return D, fb


def tier1_depth(t1p, P) -> tuple[np.ndarray, int]:
    """Tier-1 mean depth D per reach and the number of fallback reaches
    (see _tier1_depth)."""
    D, fb = _tier1_depth(t1p, P)
    return D, int(fb.sum())


def select_tier1(net: Net, fl, polys):
    """Tier-1 reaches: (candidates t1, reaches inside a river polygon t1p --
    with mean depth ``D`` and ``D_fallback`` -- and the corridor polygon).
    Vector-only (no rasters): shared by tier1_network and depth_sources.
    Returns None when the basin has no tier-1 candidate."""
    import shapely
    P = net.P
    t1 = fl[(fl.DA >= P.t1_da_km2) & ~fl.in_lake]
    if t1.empty:
        return None
    gm = t1.to_crs(net.metric).geometry
    c0 = _corridor_poly(polys, t1, P.t1_corridor_m, net.metric)
    shapely.prepare(c0)
    inp = []
    for g in gm:
        d = np.arange(0.0, g.length + 1e-6, 50.0)
        pts = shapely.line_interpolate_point(g, d)
        lon, lat = net.to_geo.transform(shapely.get_x(pts), shapely.get_y(pts))
        inp.append(shapely.contains_xy(c0, lon, lat).mean() > 0.5)
    t1p = t1[np.asarray(inp, bool)].copy()
    # Pass 2: only polygons of reaches actually in a polygon, so a polygon
    # piece around a non-tier-1 reach never borrows a far tier-1 surface.
    cpoly = _corridor_poly(polys, t1p, P.t1_corridor_m, net.metric)
    shapely.prepare(cpoly)
    t1p["D"], t1p["D_fallback"] = _tier1_depth(t1p, P)
    return t1, t1p, cpoly


def tier1_network(net: Net, fl, polys):
    """Tier-1 water-surface samples. Returns dict or None."""
    P = net.P
    sel = select_tier1(net, fl, polys)
    if sel is None:
        return None
    t1, t1p, cpoly = sel
    net.report["tier1"] = {"reaches": int(len(t1)), "in_polygon": int(len(t1p)),
                           "depth_fallback_reaches": int(t1p.D_fallback.sum())}
    if t1p.empty:
        return {"I": np.zeros(0, "i8"), "J": np.zeros(0, "i8"),
                "wse": np.zeros(0, "f4"), "D": np.zeros(0, "f4"),
                "cpoly": cpoly, "ids": set()}
    wse_ids = np.asarray(P.wse_ids, "u1")
    cap: dict = {}
    out = {"I": [], "J": [], "wse": [], "D": []}
    gm_p = t1p.to_crs(net.metric).geometry
    order = np.argsort(-t1p.HydroSeq.to_numpy(), kind="stable")
    for k in order:
        r = t1p.iloc[k]
        d, I, J = net.sample(gm_p.iloc[k], P.t1_sample_m)
        if len(d) < 2:
            continue
        raw = np.full(len(d), np.nan)
        for s0 in range(0, len(d), 200):       # bounded windows on long reaches
            sl = slice(s0, s0 + 200)
            Ii, Jj = I[sl], J[sl]
            R0, C0 = Ii.min() - 2, Jj.min() - 2
            z = net.M.read("z", R0, Ii.max() + 3, C0, Jj.max() + 3)
            pv = net.M.read("prov", R0, Ii.max() + 3, C0, Jj.max() + 3)
            z = np.where(np.isin(pv, wse_ids), z, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                raw[sl] = [np.nanmedian(z[i - R0 - 2:i - R0 + 3, j - C0 - 2:j - C0 + 3])
                           for i, j in zip(Ii, Jj)]
        good = np.isfinite(raw)
        if good.sum() < 2:
            continue
        wse = np.interp(d, d[good], pava_dec(raw[good]))
        wse = np.minimum(wse, cap.get(r.FromNode, np.inf))
        cap[r.ToNode] = min(cap.get(r.ToNode, np.inf), wse[-1])
        out["I"].append(I); out["J"].append(J); out["wse"].append(wse)
        out["D"].append(np.full(len(d), r.D))
    res = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}
    res["I"], res["J"] = res["I"].astype("i8"), res["J"].astype("i8")
    res["wse"], res["D"] = res["wse"].astype("f4"), res["D"].astype("f4")
    res["cpoly"], res["ids"] = cpoly, set(t1p.NHDPlusID)
    net.report["tier1"]["samples"] = int(len(res["I"]))
    return res


def tier2_section(r, P, pxm: float) -> tuple[float, float, str]:
    """(mean depth D, width Wd, tier) of a routed non-tier-1 reach: tier 1b
    (DA >= t1_da_km2 with a Zarrabi match) = t1_zarrabi_factor x the
    reach's own Zarrabi depth/width; else tier 2 = t2_factor x regional
    curve depth, STREAM-geo RF width (halved below t2_narrow_da_km2,
    floored at 2 px)."""
    if (r.DA >= P.t1_da_km2 and np.isfinite(r.mf_depth)
            and np.isfinite(r.mf_width)):
        return P.t1_zarrabi_factor * r.mf_depth, r.mf_width, "1b"
    D = float(tier2_depth(r.DA, r.bf_c, r.bf_e, P))
    wrf = r.RF_Width_m if np.isfinite(r.RF_Width_m) else 0.0
    if r.DA < P.t2_narrow_da_km2:
        wrf *= 0.5
    return D, max(2 * pxm, wrf), "2"


def tier2_network(net: Net, fl, t1):
    """Snapped, monotone tier-2 paths. Returns (paths, cuts DataFrame)."""
    import pandas as pd
    import shapely
    from scipy import ndimage as ndi
    from skimage.graph import route_through_array
    P = net.P
    cpoly = t1["cpoly"] if t1 else shapely.Polygon()
    shapely.prepare(cpoly)
    t1_ids = t1["ids"] if t1 else set()
    t2 = fl[~fl.NHDPlusID.isin(t1_ids) & ~fl.in_lake].reset_index(drop=True)
    gm = t2.to_crs(net.metric).geometry
    lat_r = shapely.get_y(shapely.centroid(t2.geometry.values)) if len(t2) else np.zeros(0)
    # node snapping: lowest px within node_snap_m (nodes inside the tier-1
    # polygon stay put: they meet the tier-1 surface)
    node_xy = {}
    for r, g in zip(t2.itertuples(), t2.geometry):
        cs = g.coords
        node_xy.setdefault(r.FromNode, cs[0])
        node_xy.setdefault(r.ToNode, cs[-1])
    node_px, in_poly = {}, set()
    for n, (x, y) in node_xy.items():
        I, J = (int(v) for v in net.L.rowcol(x, y))
        if shapely.contains_xy(cpoly, x, y):
            node_px[n] = (I, J)
            in_poly.add(n)
            continue
        dx, dy = net.px(y)
        rad = int(round(P.node_snap_m / (0.5 * (dx + dy))))
        win = net.M.read("z", I - rad, I + rad + 1, J - rad, J + rad + 1)
        if np.isnan(win).all():
            continue
        k = int(np.nanargmin(win))
        node_px[n] = (I - rad + k // win.shape[1], J - rad + k % win.shape[1])
    cap, capT = {}, {}
    paths, cuts = [], []
    nq_px = nq_cap = skipped = 0
    order = np.argsort(-t2.HydroSeq.to_numpy(), kind="stable")
    for n_done, k in enumerate(order, 1):
        if n_done % 5000 == 0:
            log.info("tier2 %d/%d reaches, %d paths", n_done, len(order), len(paths))
        r = t2.iloc[k]
        if r.FromNode not in node_px or r.ToNode not in node_px:
            skipped += 1
            continue
        (i0, j0), (i1, j1) = node_px[r.FromNode], node_px[r.ToNode]
        dx, dy = net.px(lat_r[k])
        pxm = 0.5 * (dx + dy)
        _, lr, lc = net.sample(gm.iloc[k], pxm)
        m = int(round(P.window_margin_m / pxm))
        R0, R1 = min(lr.min(), i0, i1) - m, max(lr.max(), i0, i1) + m + 1
        C0, C1 = min(lc.min(), j0, j1) - m, max(lc.max(), j0, j1) + m + 1
        z = net.M.read("z", R0, R1, C0, C1)
        if np.isnan(z).mean() > P.max_nan_frac:
            skipped += 1
            continue
        zf = np.where(np.isnan(z), np.nanmax(z), z)
        hand = zf - ndi.minimum_filter(zf, size=15)
        line = np.zeros(z.shape, bool)
        line[lr - R0, lc - C0] = True
        far = ndi.distance_transform_edt(~line, sampling=(dy, dx)) > P.snap_corridor_m
        cost = hand + 0.05
        cost[far] = 1e4
        try:
            path, _ = route_through_array(cost, (i0 - R0, j0 - C0), (i1 - R0, j1 - C0),
                                          fully_connected=True, geometric=True)
        except Exception:
            skipped += 1
            continue
        path = np.asarray(path)
        pr, pc = path[:, 0] + R0, path[:, 1] + C0
        zp = z[path[:, 0], path[:, 1]].astype("f8")
        if np.isnan(zp).any():
            skipped += 1
            continue
        bed, quarry, ign = path_bed(zp, cap.get(r.FromNode, np.inf), P.pit_deep_m)
        nq_cap += ign
        nq_px += int(quarry.sum())
        if r.ToNode in in_poly:     # meet the tier-1 water surface (last point only)
            bed[-1] = min(bed[-1], z[i1 - R0, j1 - C0])
        xing = {}

        def at_crossing(idx):
            if r.conduit:           # a mapped culvert is its own crossing
                return True
            if "m" not in xing:
                xing["m"] = net.crossing(R0, R1, C0, C1, lat_r[k])
            return bool(xing["m"][path[idx, 0], path[idx, 1]].any())

        def at_dam(idx):
            if "d" not in xing:
                xing["d"] = net.dam(R0, R1, C0, C1)
            return bool(xing["d"][path[idx, 0], path[idx, 1]].any())

        cut0max = float((zp - bed).max())
        bed, capped, n_capped, n_x, n_dam = cap_uncrossed_cuts(
            zp, bed, at_crossing, P.cut_cap_m, at_dam)
        cap[r.ToNode] = min(cap.get(r.ToNode, np.inf), bed[-1])
        D, Wd, tier = tier2_section(r, P, pxm)
        thal, ign = thalweg(bed, D, quarry, capped, capT.get(r.FromNode, np.inf), P.pit_deep_m)
        nq_cap += ign
        capT[r.ToNode] = min(capT.get(r.ToNode, np.inf), thal[-1])
        cut = zp - bed
        paths.append(dict(id=int(r.NHDPlusID), I=pr.astype("i8"), J=pc.astype("i8"),
                          thal=thal.astype("f4"), D=float(D), Wd=float(Wd),
                          lat=float(lat_r[k])))
        cuts.append(dict(NHDPlusID=int(r.NHDPlusID), tier=tier, da=float(r.DA),
                         name=r.GNIS_Name, max_cut=float(cut.max()),
                         max_cut_uncapped=cut0max, n_seg_capped=n_capped,
                         n_seg_at_crossing=n_x, n_seg_at_dam=n_dam, length_m=float(len(pr) * pxm),
                         conduit=bool(r.conduit),
                         lon_max=float(net.L.xy(pr[cut.argmax()], pc[cut.argmax()])[0]),
                         lat_max=float(net.L.xy(pr[cut.argmax()], pc[cut.argmax()])[1])))
    cuts = pd.DataFrame(cuts)
    net.report["tier2"] = dict(
        reaches_total=int(len(t2)), reaches_routed=len(paths), reaches_skipped=skipped,
        network_km=float(cuts.length_m.sum() / 1e3) if len(cuts) else 0.0,
        max_cut_quantiles=(cuts.max_cut.quantile([.5, .9, .99]).round(2).tolist()
                           if len(cuts) else []),
        seg_capped=int(cuts.n_seg_capped.sum()) if len(cuts) else 0,
        seg_deep_cut_at_crossing=int(cuts.n_seg_at_crossing.sum()) if len(cuts) else 0,
        seg_capped_at_dam=int(cuts.n_seg_at_dam.sum()) if len(cuts) else 0,
        conduits_routed=int(cuts.conduit.sum()) if len(cuts) else 0,
        quarry_px_skipped=nq_px, quarry_caps_ignored=int(nq_cap))
    return paths, cuts


# ----- per-tile spec files

def _spec_digest(d: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(d):
        a = np.ascontiguousarray(d[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def path_margin_px(Wd: float, lat: float, res: float) -> int:
    dx, dy = pixel_m(res, lat)
    return int(np.ceil(Wd / 2 / min(dx, dy))) + 2


def split_specs(L: Lattice, t1: dict | None, paths: list[dict],
                corridor_m: float) -> dict[str, dict]:
    """Per-tile spec dicts (only tiles something touches). Refuses a NaN
    depth/width or water surface: rasterizing one lowers nothing, silently."""
    import shapely
    if t1 is not None and not (np.isfinite(t1["D"]).all() and np.isfinite(t1["wse"]).all()):
        raise ValueError(f"tier-1 spec has {int((~np.isfinite(t1['D'])).sum())} non-finite "
                         f"D and {int((~np.isfinite(t1['wse'])).sum())} non-finite wse samples")
    bad = [p["id"] for p in paths if not (np.isfinite(p["D"]) and np.isfinite(p["Wd"]))]
    if bad:
        raise ValueError(f"{len(bad)} tier-2 paths with non-finite D/Wd, e.g. {bad[:5]}")
    dmin = pixel_m(L.res, 60.0)[0]          # conservative (small) px size
    pad_poly = int(np.ceil((corridor_m + 100.0) / dmin))
    pad_pts = int(np.ceil((2 * corridor_m + 100.0) / dmin))
    lat_ref = L.lat(t1["I"]) if t1 is not None and len(t1["I"]) else 0.0
    pb = [(p["I"].min() - path_margin_px(p["Wd"], p["lat"], L.res),
           p["I"].max() + path_margin_px(p["Wd"], p["lat"], L.res) + 1,
           p["J"].min() - path_margin_px(p["Wd"], p["lat"], L.res),
           p["J"].max() + path_margin_px(p["Wd"], p["lat"], L.res) + 1) for p in paths]
    pb = np.asarray(pb, "i8").reshape(-1, 4)
    out = {}
    for t in L.off:
        r0, r1, c0, c1 = L.window(t)
        d: dict = {}
        if t1 is not None and len(t1["I"]):
            m = ((t1["I"] >= r0 - pad_pts) & (t1["I"] < r1 + pad_pts)
                 & (t1["J"] >= c0 - pad_pts) & (t1["J"] < c1 + pad_pts))
            x0, y1 = L.xy(r0 - pad_poly, c0 - pad_poly)
            x1, y0 = L.xy(r1 + pad_poly, c1 + pad_poly)
            clip = t1["cpoly"].intersection(shapely.box(x0, y0, x1, y1))
            if m.any() and not clip.is_empty:
                d.update(t1_I=t1["I"][m], t1_J=t1["J"][m], t1_wse=t1["wse"][m],
                         t1_D=t1["D"][m], t1_lat_ref=np.float64(lat_ref),
                         t1_cpoly=np.frombuffer(shapely.to_wkb(clip), "u1"))
        if len(pb):
            sel = np.where((pb[:, 0] < r1) & (pb[:, 1] > r0)
                           & (pb[:, 2] < c1) & (pb[:, 3] > c0))[0]
            if len(sel):
                ps = [paths[i] for i in sel]
                d.update(
                    t2_off=np.cumsum([0] + [len(p["I"]) for p in ps]).astype("i8"),
                    t2_I=np.concatenate([p["I"] for p in ps]),
                    t2_J=np.concatenate([p["J"] for p in ps]),
                    t2_thal=np.concatenate([p["thal"] for p in ps]),
                    t2_D=np.array([p["D"] for p in ps], "f8"),
                    t2_Wd=np.array([p["Wd"] for p in ps], "f8"),
                    t2_lat=np.array([p["lat"] for p in ps], "f8"),
                    t2_id=np.array([p["id"] for p in ps], "i8"))
        if d:
            out[t] = d
    return out


def network_fingerprint(b: dict, bas: dict, L: Lattice, tiles: list[str]) -> str:
    huc = str(bas["huc4"])
    d = prep_dir(b, huc)
    doc = {"network_version": NETWORK_VERSION,
           "burn": {k: v for k, v in b["burn"].items() if k not in APPLY_ONLY_KEYS},
           "curves": b["curves"],
           "lattice": [L.res, L.x0, L.y0],
           "inputs": [_file_sig(Path(bas["hr_gdb"])), _file_sig(Path(bas["water_mask"])),
                      _file_sig(d / "ml.gpkg"), _file_sig(d / "roads.gpkg"),
                      _file_sig(d / "dams.gpkg"),
                      _file_sig(Path(b["data"]["physio_shp"]))],
           "tiles": [[t, _file_sig(in_tile(b, t)), _file_sig(in_tile(b, t, "_source"))]
                     for t in tiles]}
    return _digest(doc)


def _huc_tiles(b, bas, spec, tiles, polys=None):
    polys = polys or _huc4_polygons(b)
    x0, y0, x1, y1 = polys[str(bas["huc4"])].bounds
    sel = []
    for t in tiles:
        w, s, e, n = spec.tile_bounds(t)
        if w < x1 and e > x0 and s < y1 and n > y0:
            sel.append(t)
    return sel


def network_lines(net: Net):
    """The network pass's vector inputs (no rasters): (burnable lines with
    effective DA, curve and ML attrs; the tidal lines skip_tidal removed;
    the basin's NHD water polygons)."""
    import geopandas as gpd
    b, bas = net.b, net.bas
    d = prep_dir(b, str(bas["huc4"]))
    fl = load_lines(net)
    fl = attach_ml(fl, gpd.read_file(d / "ml.gpkg"), net.metric)
    n_da = int((fl.DA < fl.TotDASqKm).sum())
    fl = fl[fl.DA >= net.P.da_min_km2].reset_index(drop=True)
    net.report["da_from_divda"] = n_da
    fl = attach_curve(fl, b["data"]["physio_shp"], b["curves"])
    tidal = fl[fl.Tidal_Flag == 1] if net.P.skip_tidal else fl.iloc[:0]
    fl = fl.drop(tidal.index).reset_index(drop=True)
    wp = gpd.read_file(bas["water_mask"], layer="water").to_crs(4269)
    wp["geometry"] = wp.geometry.force_2d()
    return fl, tidal.reset_index(drop=True), wp


def run_network(b: dict, huc4: str | None = None, force: bool = False) -> None:
    import geopandas as gpd
    spec, tiles = load_input(b)
    L = Lattice(spec, tiles)
    polys = _huc4_polygons(b)
    for bas in b["basins"]:
        huc = str(bas["huc4"])
        if huc4 and huc != str(huc4):
            continue
        d = prep_dir(b, huc)
        if not all((d / f).exists() for f in ("ml.gpkg", "roads.gpkg", "dams.gpkg")):
            raise SystemExit(f"{huc}: run the 'prepare' step first")
        fp = network_fingerprint(b, bas, L, _huc_tiles(b, bas, spec, tiles, polys))
        mp = network_manifest(b, huc)
        if not force and mp.exists() and json.loads(mp.read_text()).get("fingerprint") == fp:
            log.info("%s: network up to date", huc)
            continue
        M = Mosaic(L, lambda t, k: in_tile(b, t, {"z": "", "prov": "_source",
                                                  "basin": "_basin"}[k]))
        net = Net(b, bas, L, M)
        net.roads = gpd.read_file(d / "roads.gpkg")
        net.dams = dam_buffers(gpd.read_file(d / "dams.gpkg"), net.P.dam_buffer_m, net.metric)
        fl, tidal, wp = network_lines(net)
        log.info("%s: %d HR flowlines (%d with ML attrs, %d tidal %s)", huc, len(fl),
                 int(fl.COMID.notna().sum()), len(tidal),
                 "skipped" if net.P.skip_tidal else "kept")
        t1 = tier1_network(net, fl, wp)
        log.info("%s: tier1 %s", huc, net.report.get("tier1"))
        paths, cuts = tier2_network(net, fl, t1)
        log.info("%s: tier2 %s", huc, net.report.get("tier2"))
        M.close()
        specs = split_specs(L, t1, paths, net.P.t1_corridor_m)
        digests = {}
        for t, sd in specs.items():
            p = spec_path(b, huc, t)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.stem + ".tmp.npz")
            np.savez(tmp, **sd)
            tmp.rename(p)
            digests[t] = _spec_digest(sd)
        for p in sorted((out_root(b) / "spec" / huc).glob("*.npz")):
            if p.stem not in specs:             # a tile this network no longer touches
                p.unlink()
        cuts.to_csv(d / "tier2_reaches.csv", index=False)
        if t1 is not None:
            gpd.GeoDataFrame(geometry=[t1["cpoly"]], crs=4269).to_file(
                d / "tier1_polygon.gpkg", driver="GPKG")
        _atomic_json(mp, {"fingerprint": fp, "huc4": huc, "tiles": digests,
                          "report": net.report})
        log.info("%s: network done, specs for %d tiles", huc, len(specs))


# ----- per-reach depth sources (QA / viewer; never feeds the burn)

# What a reach's burn depth D came from (``src`` in depth_sources output).
DEPTH_SOURCES = {
    "t1_zarrabi": "tier 1 (river polygon): t1_zarrabi_factor x Zarrabi "
                  "mean-flow depth, median over reaches of similar DA",
    "t1_curve": "tier 1 (river polygon): no Zarrabi depth within DA +-25 % "
                "-> t2_factor x regional curve (fallback)",
    "t1b_zarrabi": "tier 1b (DA >= t1_da_km2, outside a river polygon): "
                   "t1_zarrabi_factor x the reach's own Zarrabi depth",
    "t2_curve": "tier 2: t2_factor x regional bankfull curve (curve_idx)",
    "not_routed": "not burned: no least-cost path (void window / unsnapped node)",
    "lake": "not burned: artificial path through a lake / reservoir",
    "tidal": "not burned: tidal reach (bathymetry's job)",
}


def depth_source_path(b: dict, huc4: str) -> Path:
    return out_root(b) / "qa" / "depth_source" / f"{huc4}.gpkg"


def depth_sources(b: dict, huc4: str | None = None, force: bool = False) -> None:
    """Per-reach table of where each burned channel's depth came from
    (DEPTH_SOURCES), one GPKG per HUC-4 under <root>/qa/depth_source/.

    Recomputed from the network's own vector steps (network_lines,
    select_tier1, tier2_section -- the same functions run_network uses, no
    rasters), so it can only be built against a CURRENT network: refuses
    if the manifest fingerprint is stale. Whether a non-tier-1 reach was
    routed comes from the network's tier2_reaches.csv, and the routed
    reaches' tier there must agree with the recomputed one (hard error:
    the layer would misreport the burn)."""
    import geopandas as gpd
    import pandas as pd
    import shapely
    spec, tiles = load_input(b)
    L = Lattice(spec, tiles)
    polys = _huc4_polygons(b)
    curves = b["curves"]
    for bas in b["basins"]:
        huc = str(bas["huc4"])
        if huc4 and huc != str(huc4):
            continue
        fp = network_fingerprint(b, bas, L, _huc_tiles(b, bas, spec, tiles, polys))
        mp = network_manifest(b, huc)
        if not mp.exists() or json.loads(mp.read_text()).get("fingerprint") != fp:
            raise SystemExit(f"{huc}: network not current -- run the 'network' step first")
        out = depth_source_path(b, huc)
        side = out.with_suffix(".json")
        if (not force and out.exists() and side.exists()
                and json.loads(side.read_text()).get("fingerprint") == fp):
            log.info("%s: depth sources up to date", huc)
            continue
        net = Net(b, bas, L, None)
        fl, tidal, wp = network_lines(net)
        P = net.P
        sel = select_tier1(net, fl, wp)
        t1p = sel[1] if sel is not None else fl.iloc[:0].assign(D=[], D_fallback=[])
        cuts = pd.read_csv(prep_dir(b, huc) / "tier2_reaches.csv",
                           dtype={"tier": str})
        routed = cuts.set_index("NHDPlusID")
        lat = shapely.get_y(shapely.centroid(fl.geometry.values))
        src, D, W, tier = [], [], [], []
        D1 = dict(zip(t1p.NHDPlusID, zip(t1p.D, t1p.D_fallback)))
        for k, r in enumerate(fl.itertuples()):
            if r.NHDPlusID in D1:
                d, fb = D1[r.NHDPlusID]
                src.append("t1_curve" if fb else "t1_zarrabi")
                D.append(float(d)); W.append(np.nan); tier.append("1")
                continue
            if r.in_lake:
                src.append("lake"); D.append(np.nan); W.append(np.nan); tier.append("")
                continue
            dx, dy = pixel_m(L.res, lat[k])
            d, w, t = tier2_section(r, P, 0.5 * (dx + dy))
            if r.NHDPlusID not in routed.index:
                src.append("not_routed"); D.append(float(d)); W.append(float(w))
                tier.append(t)
                continue
            if str(routed.at[r.NHDPlusID, "tier"]) != t:
                raise RuntimeError(f"{huc}: reach {r.NHDPlusID} recomputed tier {t} != "
                                   f"network tier {routed.at[r.NHDPlusID, 'tier']}")
            src.append("t1b_zarrabi" if t == "1b" else "t2_curve")
            D.append(float(d)); W.append(float(w)); tier.append(t)
        fl = fl.assign(src=src, D=D, W=W, tier=tier)
        tidal = tidal.assign(src="tidal", D=np.nan, W=np.nan, tier="")
        g = pd.concat([fl, tidal], ignore_index=True)
        mc = routed.max_cut.reindex(g.NHDPlusID).to_numpy()
        g = gpd.GeoDataFrame({
            "huc4": huc, "NHDPlusID": g.NHDPlusID.astype("int64"),
            "name": g.GNIS_Name.fillna(""), "src": g.src, "tier": g.tier,
            "D": g.D.astype("f4"), "W": g.W.astype("f4"),
            "DA": g.DA.astype("f4"), "TotDASqKm": g.TotDASqKm.astype("f4"),
            "da_from_div": (g.DA < g.TotDASqKm).to_numpy(),
            "zarrabi_mf_depth": g.mf_depth.astype("f4"),
            "curve_idx": g.curve_idx.astype("int16"),
            "curve": [curve_label(curves[i]) for i in g.curve_idx],
            "physio": g.physio.fillna(""), "conduit": g.conduit.astype(bool),
            "max_cut": mc.astype("f4")}, geometry=g.geometry.values, crs=fl.crs)
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_gpkg(g, out)
        counts = {k: int(v) for k, v in g.src.value_counts().items()}
        _atomic_json(side, {"fingerprint": fp, "huc4": huc, "counts": counts,
                            "curves": [dict(c, label=curve_label(c)) for c in curves]})
        log.info("%s: depth sources %s -> %s", huc, counts, out)


def load_specs(b: dict, tile: str) -> dict[str, dict]:
    """{huc4: spec dict} for one tile, from each network's manifest (never a
    bare glob of spec/: a stale file from an interrupted run is ignored, and
    a file that no longer matches its manifest digest is fatal)."""
    out = {}
    for bas in b["basins"]:
        huc = str(bas["huc4"])
        mp = network_manifest(b, huc)
        if not mp.exists():
            raise SystemExit(f"{huc}: run the 'network' step first")
        man = json.loads(mp.read_text())
        if tile not in man["tiles"]:
            continue
        with np.load(spec_path(b, huc, tile)) as z:
            d = {k: z[k] for k in z.files}
        if _spec_digest(d) != man["tiles"][tile]:
            raise RuntimeError(f"{spec_path(b, huc, tile)} does not match "
                               f"{mp.name} -- rerun 'network --force'")
        out[huc] = d
    return out


# --------------------------------------------------------------------------
# phase B: rasterize + apply per tile
# --------------------------------------------------------------------------

def rasterize_tier1(L: Lattice, tile: str, d: dict, return_wse: bool = False):
    """Parabolic area-preserving section: bed = wse - D * 1.5 (1 - s^2),
    s = d_centerline / (d_centerline + d_bank) (0 on the centerline, 1 at
    the bank); mean over a symmetric section = D. With return_wse, also the
    per-pixel assigned water surface (inf outside) for ``guard_tier1``."""
    import shapely
    from rasterio.features import rasterize
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    r0, r1, c0, c1 = L.window(tile)
    h, w = r1 - r0, c1 - c0
    out = np.full((h, w), np.inf, "f4")
    wse = np.full((h, w), np.inf, "f4")
    if "t1_I" not in d or not len(d["t1_I"]):
        return (out, wse) if return_wse else out
    dx, dy = pixel_m(L.res, float(d["t1_lat_ref"]))
    poly = shapely.from_wkb(d["t1_cpoly"].tobytes())
    pad = int(np.ceil(700.0 / min(dx, dy)))
    W0, H0 = c0 - pad, r0 - pad
    C = rasterize([(poly, 1)], out_shape=(h + 2 * pad, w + 2 * pad),
                  transform=L.transform(H0, W0)).astype(bool)
    inner = C[pad:pad + h, pad:pad + w]
    if not inner.any():
        return (out, wse) if return_wse else out
    d_bank = ndi.distance_transform_edt(C, sampling=(dy, dx))[pad:pad + h, pad:pad + w]
    ii, jj = np.nonzero(inner)
    kd = cKDTree(np.c_[d["t1_I"] * dy, d["t1_J"] * dx])
    d_cl, nn = kd.query(np.c_[(ii + r0) * dy, (jj + c0) * dx])
    db = d_bank[ii, jj]
    s = d_cl / (d_cl + db + 1e-6)
    shape = 1.5 * (1 - s ** 2)
    out[ii, jj] = d["t1_wse"][nn] - d["t1_D"][nn] * shape
    out[~np.isfinite(out)] = np.inf
    if return_wse:
        wse[ii, jj] = d["t1_wse"][nn]
        return out, wse
    return out


def path_core(pr, pc, R0, C0, shape):
    """The minimal 4-connected channel of a path: its pixels plus, at every
    diagonal step, the connector (row of the first point, column of the
    second) -- deterministic from global coordinates, so tiles sharing a
    collar agree. Never dropped by the lateral guard: dropping a diagonal
    connector re-dams the channel in 4-connected (edge) statistics."""
    core = np.zeros(shape, bool)
    core[pr - R0, pc - C0] = True
    if len(pr) > 1:
        diag = (np.abs(np.diff(pr)) == 1) & (np.abs(np.diff(pc)) == 1)
        core[pr[:-1][diag] - R0, pc[1:][diag] - C0] = True
    return core


def rasterize_tier2(L: Lattice, tile: str, d: dict, z: np.ndarray | None = None,
                    tol: float | None = None):
    """Per path: parabola of mean depth D over width Wd around the snapped
    path (thalweg at the centre), plus a >=2 px 4-connected spine at
    thalweg level. Each path is rasterized in its own window from global
    coordinates, so neighbouring tiles agree in a shared collar.

    Lateral guard (with the tile's merged z and tol): outside the path's
    ``path_core``, a pixel whose own surface is more than tol above the
    local bed (thalweg + 1.5 D) is not burned -- otherwise the section digs
    into whatever steep ground lies beside the path (Downsville dam face,
    410 -> 364 m, beside a path correctly capped at the dam). Returns
    (target, n_dropped) when z is given, else target."""
    from scipy import ndimage as ndi
    r0, r1, c0, c1 = L.window(tile)
    out = np.full((r1 - r0, c1 - c0), np.inf, "f4")
    n_drop = np.zeros(out.shape, bool) if z is not None else None
    if "t2_off" not in d:
        return (out, 0) if z is not None else out
    cross = ndi.generate_binary_structure(2, 1)
    off = d["t2_off"]
    for k in range(len(off) - 1):
        pr, pc = d["t2_I"][off[k]:off[k + 1]], d["t2_J"][off[k]:off[k + 1]]
        thal = d["t2_thal"][off[k]:off[k + 1]]
        D, Wd, lat = float(d["t2_D"][k]), float(d["t2_Wd"][k]), float(d["t2_lat"][k])
        dx, dy = pixel_m(L.res, lat)
        hw = Wd / 2
        m = path_margin_px(Wd, lat, L.res)
        R0, R1, C0, C1 = pr.min() - m, pr.max() + m + 1, pc.min() - m, pc.max() + m + 1
        a0, a1, b0, b1 = max(R0, r0), min(R1, r1), max(C0, c0), min(C1, c1)
        if a0 >= a1 or b0 >= b1:
            continue
        Pm = np.zeros((R1 - R0, C1 - C0), bool)
        Pm[pr - R0, pc - C0] = True
        bedr = np.full(Pm.shape, np.nan, "f4")
        bedr[pr - R0, pc - C0] = thal
        dist, (ir, ic) = ndi.distance_transform_edt(~Pm, sampling=(dy, dx),
                                                    return_indices=True)
        bz = bedr[ir, ic]
        prof = np.where(dist <= hw, bz + 1.5 * D * (dist / hw) ** 2, np.inf)
        sp = ndi.binary_dilation(Pm, cross)
        prof = np.where(sp, np.fmin(prof, bz), prof)
        prof[~np.isfinite(prof)] = np.inf
        sub = prof[a0 - R0:a1 - R0, b0 - C0:b1 - C0]
        if z is not None:
            core = path_core(pr, pc, R0, C0, Pm.shape)[a0 - R0:a1 - R0, b0 - C0:b1 - C0]
            zs = z[a0 - r0:a1 - r0, b0 - c0:b1 - c0]
            ref = (bz + 1.5 * D)[a0 - R0:a1 - R0, b0 - C0:b1 - C0]
            bad = np.isfinite(sub) & ~core & ~(zs <= ref + tol)
            n_drop[a0 - r0:a1 - r0, b0 - c0:b1 - c0] |= bad & (sub < zs)
            sub = np.where(bad, np.inf, sub)
        o = out[a0 - r0:a1 - r0, b0 - c0:b1 - c0]
        np.fmin(o, sub, out=o)
    if z is not None:
        return out, int((n_drop & ~(out < z)).sum())
    return out


def water_mask_tile(water_gpkg, L: Lattice, tile: str) -> np.ndarray:
    """The basin's stage-07 NHD water polygons (layer 'water') on one tile."""
    import geopandas as gpd
    from rasterio.features import rasterize
    r0, r1, c0, c1 = L.window(tile)
    x0, y1 = L.x0 + c0 * L.res, L.y0 - r0 * L.res
    x1, y0 = L.x0 + c1 * L.res, L.y0 - r1 * L.res
    wp = gpd.read_file(water_gpkg, layer="water", bbox=(x0, y0, x1, y1)).to_crs(4269)
    if not len(wp):
        return np.zeros((r1 - r0, c1 - c0), bool)
    return rasterize([(g, 1) for g in wp.geometry.force_2d()], out_shape=(r1 - r0, c1 - c0),
                     transform=L.transform(r0, c0)).astype(bool)


def tile_fingerprint(b: dict, tile: str, spec: GridSpec, specs: dict) -> str:
    gt, w, h = spec.tile_geotransform(tile)
    doc = {"burn_version": BURN_VERSION, "burn": b["burn"],
           "grid": [list(gt), w, h],
           "inputs": [_file_sig(in_tile(b, tile, s)) for s in ("", "_source", "_basin")],
           "water": ([_file_sig(Path(bas["water_mask"])) for bas in b["basins"]
                      if str(bas["huc4"]) in specs]
                     if b["burn"].get("burnable_outside_water_ids") else None),
           "specs": {h4: _spec_digest(d) for h4, d in sorted(specs.items())}}
    return _digest(doc)


def burn_tile(b: dict, tile: str, spec: GridSpec, L: Lattice) -> dict:
    """Write the burned elevation COG, provenance, basin and (if anything was
    lowered) burn-depth rasters for one tile, then its stats JSON (the
    completion marker). Skips when the fingerprint still matches and the
    outputs exist on the current grid."""
    P = Params.from_config(b)
    specs = load_specs(b, tile)
    fp = tile_fingerprint(b, tile, spec, specs)
    elev, prov_p = out_tile(b, tile), out_tile(b, tile, "_source")
    basin_p, depth_p = out_tile(b, tile, "_basin"), out_tile(b, tile, "_burndepth")
    sp = stats_path(b, tile)
    if sp.exists() and all(p.exists() for p in (elev, prov_p, basin_p)):
        old = json.loads(sp.read_text())
        if (old.get("fingerprint") == fp and spec.tile_matches(tile, elev)
                and (old["mode"] != "burn" or depth_p.exists())):
            return old
    for p in (elev, prov_p, basin_p, depth_p, sp):
        p.unlink(missing_ok=True)
    elev.parent.mkdir(parents=True, exist_ok=True)

    if tile in L.off_lattice:
        if specs:
            raise RuntimeError(f"{tile}: off the product lattice but a burn spec reaches it")
        for src, dst in ((in_tile(b, tile), elev), (in_tile(b, tile, "_source"), prov_p),
                         (in_tile(b, tile, "_basin"), basin_p)):
            tmp = dst.with_name(dst.name[:-4] + TMP_TIF_SUFFIX)
            shutil.copy2(src, tmp)
            tmp.rename(dst)
        stats = {"tile": tile, "fingerprint": fp, "mode": "copy_off_lattice", "n_lowered": 0}
        _atomic_json(sp, stats)
        return stats

    z, gt, size = merge._read(in_tile(b, tile), "f4")
    prov, gt_p, size_p = merge._read(in_tile(b, tile, "_source"), "u1")
    basin, gt_b, size_b = merge._read(in_tile(b, tile, "_basin"), "u1")
    gt_s, w, h = spec.tile_geotransform(tile)
    if not (tuple(gt) == tuple(gt_p) == tuple(gt_b) == tuple(gt_s)
            and size == size_p == size_b == (w, h)):
        raise ValueError(f"{tile}: merged inputs are not on the merged grid spec")
    stats = {"tile": tile, "fingerprint": fp}

    targets, t1_any, t2_any = {}, None, None
    n_guard = n_guard2 = 0
    for huc, d in specs.items():
        t1, w1 = rasterize_tier1(L, tile, d, return_wse=True)
        t1, ng = guard_tier1(t1, w1, z, P.t1_surface_tol_m)
        n_guard += ng
        t2, ng2 = rasterize_tier2(L, tile, d, z, P.t2_lateral_tol_m)
        n_guard2 += ng2
        targets[int(huc) % 100] = np.fmin(t1, t2)
        t1_any = t1 if t1_any is None else np.fmin(t1_any, t1)
        t2_any = t2 if t2_any is None else np.fmin(t2_any, t2)
    outside = {}
    if targets and P.burnable_outside_water_ids and np.isin(
            prov, np.asarray(P.burnable_outside_water_ids, "u1")).any():
        for bas in b["basins"]:
            if str(bas["huc4"]) in specs:
                outside[int(bas["huc4"]) % 100] = ~water_mask_tile(
                    bas["water_mask"], L, tile)
    if targets:
        z_new, prov_new, depth = apply_burn(z, prov, basin, targets,
                                            P.burnable_ids, P.burn_id,
                                            P.burnable_outside_water_ids, outside)
    lowered = (prov_new == P.burn_id) if targets else np.zeros(z.shape, bool)
    if not lowered.any():
        for src, dst in ((in_tile(b, tile), elev), (in_tile(b, tile, "_source"), prov_p),
                         (in_tile(b, tile, "_basin"), basin_p)):
            tmp = dst.with_name(dst.name[:-4] + TMP_TIF_SUFFIX)
            shutil.copy2(src, tmp)
            tmp.rename(dst)
        stats.update(mode="copy", n_lowered=0, n_tier1_surface_guarded=n_guard,
                     n_tier2_lateral_guarded=n_guard2)
    else:
        # Invariants, checked on the arrays actually written.
        changed = ~((z_new == z) | (np.isnan(z_new) & np.isnan(z)))
        if np.any(z_new[changed] > z[changed]):
            raise RuntimeError(f"{tile}: burn raised a pixel")
        always = np.isin(prov, np.asarray(P.burnable_ids, "u1"))
        topo_ok = np.zeros(z.shape, bool)
        for code, ow in outside.items():
            topo_ok |= (np.isin(prov, np.asarray(P.burnable_outside_water_ids, "u1"))
                        & ow & (basin == code))
        if np.any(~(always | topo_ok)[changed]):
            raise RuntimeError(f"{tile}: burn changed a non-burnable pixel")
        epsg = int(b["_merge"]["output"]["epsg"])
        cog = b["_merge"]["output"]["cog"]
        merge._write_cog(elev, z_new, gt_s, epsg, cog)
        merge._write_u1(prov_p, prov_new, gt_s, epsg)
        tmp = basin_p.with_name(basin_p.name[:-4] + TMP_TIF_SUFFIX)
        shutil.copy2(in_tile(b, tile, "_basin"), tmp)
        tmp.rename(basin_p)
        merge._write_cog(depth_p, depth, gt_s, epsg, cog)
        dx, dy = pixel_m(L.res, L.lat(np.array([L.off[tile][0] + h / 2])))
        dep = depth[lowered]
        tier1_px = lowered & np.isfinite(t1_any) & (t1_any <= t2_any)
        stats.update(mode="burn", n_lowered=int(lowered.sum()),
                     n_lowered_tier1=int(tier1_px.sum()),
                     n_lowered_tier2=int((lowered & ~tier1_px).sum()),
                     n_lowered_filled=int((lowered & (prov == 99)).sum()),
                     n_lowered_survey_topo=int((lowered & ~always).sum()),
                     n_tier1_surface_guarded=n_guard,
                     n_tier2_lateral_guarded=n_guard2,
                     volume_m3=float(dep.sum() * dx * dy),
                     depth_m={"mean": float(dep.mean()), "p50": float(np.median(dep)),
                              "p99": float(np.quantile(dep, 0.99)), "max": float(dep.max())})
    if not spec.tile_matches(tile, elev):
        raise RuntimeError(f"{tile}: written tile does not match the grid spec")
    _atomic_json(sp, stats)
    return stats


# --------------------------------------------------------------------------
# VRT / QA
# --------------------------------------------------------------------------

def build_vrt(b: dict) -> Path:
    p = b["project"]
    out = out_root(b) / f"{p['name']}_{p['version']}.vrt"
    finals = [str(x) for x in product_tifs(out_root(b) / "tiles",
                                           f"{p['name']}_*_{p['version']}.tif")
              if not x.stem.endswith(("_source", "_basin", "_burndepth"))]
    if not finals:
        raise SystemExit("no burned tiles yet")
    listfile = out.with_suffix(".inputs.txt")
    listfile.write_text("\n".join(finals) + "\n")
    subprocess.run(["gdalbuildvrt", "-q", "-overwrite", "-input_file_list",
                    str(listfile), str(out)], check=True)
    subprocess.run(["gdaladdo", "-q", "-r", "average", "-ro", str(out),
                    "16", "64", "256"], check=True,
                   env={**os.environ, "GDAL_NUM_THREADS": "4"})
    log.info("mosaic VRT %s (%d tiles)", out, len(finals))
    return out


def fill_depth(z: np.ndarray) -> np.ndarray:
    """Depression depth by 4-connected morphological reconstruction (the
    water a sealed surface would pond)."""
    from scipy import ndimage as ndi
    from skimage.morphology import reconstruction
    a = np.where(np.isnan(z), np.nanmin(z) - 1, z).astype("f8")
    seed = a.copy()
    seed[1:-1, 1:-1] = a.max()
    return (reconstruction(seed, a, method="erosion",
                           footprint=ndi.generate_binary_structure(2, 1)) - a).astype("f4")


def connectivity(b: dict, tile: str, L: Lattice) -> dict:
    """Ponding on the snapped tier-2 network before vs after the burn."""
    from scipy import ndimage as ndi
    r0, r1, c0, c1 = L.window(tile)
    net = np.zeros((r1 - r0, c1 - c0), bool)
    for d in load_specs(b, tile).values():
        if "t2_I" in d:
            m = (d["t2_I"] >= r0) & (d["t2_I"] < r1) & (d["t2_J"] >= c0) & (d["t2_J"] < c1)
            net[d["t2_I"][m] - r0, d["t2_J"][m] - c0] = True
    res = {"net_px": int(net.sum())}
    if not net.any():
        return res
    cross = ndi.generate_binary_structure(2, 1)
    for nm, p in (("merged", in_tile(b, tile)), ("burned", out_tile(b, tile))):
        f = fill_depth(merge._read(p, "f4")[0])
        fn = f[net]
        lab = ndi.label(f > 0.1, cross)[0]
        res[nm] = {"frac_gt_0p1": float((fn > 0.1).mean()),
                   "frac_gt_0p5": float((fn > 0.5).mean()),
                   "n_pits_on_net": int(len(np.unique(lab[net & (f > 0.1)])))}
    return res


def run_qa(b: dict, with_connectivity: bool = False) -> list[dict]:
    from . import qa
    spec, tiles = load_input(b)
    ids = merge.merged_source_ids(merge.load_basins(b["_merge"]))
    ids = {**ids, "burn": int(b["burn"]["burn_id"])}
    tile_paths, missing = {}, []
    for t in tiles:
        if out_tile(b, t).exists() and stats_path(b, t).exists():
            tile_paths[t] = out_tile(b, t)
        else:
            missing.append(t)
    checks = [{"check": "completeness", "pass": not missing,
               "n_expected": len(tiles), "missing": missing}]
    checks.append(qa.check_grid_conformity(tile_paths, spec=spec))
    qcfg = dict(b["_merge"]["qa"])
    seams = qa.check_seams(tile_paths, spec=spec,
                           cfg={"qa": qcfg, "compositing": {"source_ids": ids}})
    checks.append(merge.apply_seam_acceptance(seams, qcfg))
    tot = {"tiles_burned": 0, "tiles_copied": 0, "n_lowered": 0, "n_lowered_tier1": 0,
           "n_lowered_tier2": 0, "n_lowered_filled": 0, "n_lowered_survey_topo": 0,
           "n_tier1_surface_guarded": 0, "n_tier2_lateral_guarded": 0, "volume_m3": 0.0}
    deepest = []
    for t in tile_paths:
        s = json.loads(stats_path(b, t).read_text())
        tot["tiles_burned" if s["mode"] == "burn" else "tiles_copied"] += 1
        if s["mode"] == "copy_off_lattice":
            tot.setdefault("off_lattice_tiles", []).append(t)
        for k in ("n_lowered", "n_lowered_tier1", "n_lowered_tier2",
                  "n_lowered_filled", "n_lowered_survey_topo",
                  "n_tier1_surface_guarded", "n_tier2_lateral_guarded", "volume_m3"):
            tot[k] += s.get(k, 0)
        if s["mode"] == "burn":
            deepest.append((s["depth_m"]["max"], t))
    networks = {}
    for bas in b["basins"]:
        mp = network_manifest(b, str(bas["huc4"]))
        if mp.exists():
            networks[str(bas["huc4"])] = json.loads(mp.read_text())["report"]
    checks.append({"check": "burn_summary", **tot,
                   "deepest_tiles": [{"tile": t, "max_m": round(v, 2)}
                                     for v, t in sorted(deepest, reverse=True)[:10]],
                   "networks": networks})
    if with_connectivity:
        L = Lattice(spec, tiles)
        checks.append({"check": "connectivity",
                       "tiles": {t: connectivity(b, t, L)
                                 for t in b["qa"]["connectivity_tiles"] if t in tile_paths}})
    qa_dir = out_root(b) / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(qa_dir / "qa_report.json", {"checks": checks})
    return checks
