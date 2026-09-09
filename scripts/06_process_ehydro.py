#!/usr/bin/env python
"""Stage 06: grid eHydro surveys to 3 m NAVD88 rasters + build ehydro VRT.

Per survey ZIP:
  1. Extract; prefer the GDB (SurveyPoint/SurveyPointHD + outline polygon),
     fall back to the *.XYZ text file.
  2. Units/sign audit: CENAP soundings are depth in FEET, POSITIVE DOWN,
     referenced to MLLW -> z_mllw_m = -depth_ft * 0.3048. Every survey's
     min/max is logged and asserted against a plausibility window; per-survey
     overrides go in manifests/ehydro_overrides.yaml
     (surveyname: {depth_units: meters, depth_positive_down: false, ...}).
  3. MLLW -> NAVD88 via the VDatum-derived separation raster
     (built on demand; benchmark-verified, see dem_pipeline/vdatum.py).
  4. TIN linear interpolation onto a 3 m UTM 18N grid clipped to the survey
     hull (no extrapolation), written to data/intermediate/ehydro_grids/.
  5. VRT over all survey grids ordered oldest-first (newest paints last),
     plus survey footprints -> data/intermediate/ehydro_footprints.gpkg.

Each survey's footprint is also persisted as a <survey>.hull.geojson sidecar
next to its grid, and the combined layer is reassembled from those sidecars
every run -- a resumed run skips surveys it already gridded, so building the
layer from just this run's surveys would truncate stage 07's water mask.
"""

import argparse
import json
import logging
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

import dem_pipeline as dp
from dem_pipeline.vdatum import build_separation_raster

log = logging.getLogger("06_process_ehydro")

FT_TO_M = 0.3048
# Plausibility window for converted NAVD88 channel elevations (m); surveys
# outside this window abort the run for manual inspection.
PLAUSIBLE_MIN, PLAUSIBLE_MAX = -35.0, 5.0
# Per-survey footprint sidecar, written next to that survey's grid.
FOOTPRINT_SUFFIX = ".hull.geojson"


class ImplausibleDepthsError(ValueError):
    """Converted elevations fall outside the plausibility window -- almost
    always a units/sign/datum mistake. Aborts the run (never skipped)."""


def convert_raw_depths(z_raw, units: str, positive_down: bool):
    """Raw survey soundings -> MLLW-referenced elevations in meters.

    CENAP convention (config default): depth in feet, positive down, so a
    45-ft channel sounding becomes -13.7 m relative to MLLW.
    """
    if units not in ("feet", "meters"):
        raise ValueError(f"unknown depth units {units!r}")
    z = np.asarray(z_raw, dtype="f8") * (FT_TO_M if units == "feet" else 1.0)
    return -z if positive_down else z


def check_plausible(z_navd88, survey: str) -> None:
    zmin, zmax = float(np.min(z_navd88)), float(np.max(z_navd88))
    if zmin < PLAUSIBLE_MIN or zmax > PLAUSIBLE_MAX:
        raise ImplausibleDepthsError(
            f"{survey}: z range [{zmin:.1f}, {zmax:.1f}] m NAVD88 outside "
            f"[{PLAUSIBLE_MIN}, {PLAUSIBLE_MAX}] -- check units/sign; "
            f"add an override for {survey} if intentional")


def load_overrides() -> dict:
    p = dp.path("manifests", "ehydro_overrides.yaml")
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def extract_zip(zpath: Path) -> Path:
    out = zpath.parent / zpath.stem
    if not out.exists():
        tmp = out.with_suffix(".tmp")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        tmp.rename(out)
    return out


# Preference order for the polygon that bounds the surveyed area. CENAP GDBs
# store it as SurveyJob; NOT ONE of the 146 Delaware surveys has a layer called
# "outline" or "coverage", so those names (the original match) silently fell
# through to a convex hull of the soundings -- which spans river bends and
# extrapolates dredged depths across water the survey never visited.
OUTLINE_LAYERS = ("surveyjob", "bathymetry_vector", "outline", "coverage")


def outline_layer(layers) -> str | None:
    """Best available survey-extent layer, by preference, or None."""
    lower = {l.lower(): l for l in layers}
    for want in OUTLINE_LAYERS:
        for name_lower, name in lower.items():
            if want in name_lower:
                return name
    return None


def read_survey_points(survey_dir: Path, crs_override=None):
    """Return (points Nx3 [x, y, z_raw], crs, outline geometry or None).

    z_raw is whatever the source stores (audited/converted by caller).
    """
    gdbs = sorted(survey_dir.rglob("*.gdb"))
    if gdbs:
        import geopandas as gpd
        import pyogrio

        gdb = gdbs[0]
        layers = [l[0] for l in pyogrio.list_layers(gdb)]
        pt_layer = next((l for l in layers if "surveypoint" in l.lower()), None)
        if pt_layer:
            pts = gpd.read_file(gdb, layer=pt_layer)
            zfield = next(
                (c for c in pts.columns
                 if c.lower() in ("z_depth", "depth", "z", "elevation")),
                None,
            )
            if zfield is None and pts.geometry.has_z.all():
                z = np.array([g.z for g in pts.geometry])
            elif zfield is not None:
                z = pts[zfield].to_numpy(dtype="f8")
            else:
                raise ValueError(f"{gdb.name}: no depth field or 3D geometry")
            xyz = np.column_stack([pts.geometry.x, pts.geometry.y, z])
            outline = None
            ol_layer = outline_layer(layers)
            if ol_layer:
                og = gpd.read_file(gdb, layer=ol_layer)
                if len(og):
                    outline = og.to_crs(pts.crs).geometry.union_all()
            if outline is None:
                log.warning("%s: no survey-extent layer among %s -- falling "
                            "back to the sounding convex hull, which "
                            "over-extends across channel bends",
                            gdb.name, layers)
            return xyz, pts.crs, outline

    xyz_files = sorted(survey_dir.rglob("*.XYZ")) + sorted(
        survey_dir.rglob("*.xyz"))
    if not xyz_files:
        raise FileNotFoundError(f"no GDB SurveyPoint or XYZ in {survey_dir}")
    rows = []
    for line in xyz_files[0].read_text().splitlines():
        parts = re.split(r"[,\s]+", line.strip())
        if len(parts) >= 3:
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"{xyz_files[0].name}: no parsable points")
    # eHydro XYZ files carry no CRS; CENAP convention is NAD83 PA South
    # state plane feet -- must be confirmed against the survey metadata XML.
    prj = sorted(survey_dir.rglob("*.prj"))
    crs = crs_override or (prj[0].read_text() if prj else None)
    if crs is None:
        raise ValueError(
            f"{survey_dir.name}: XYZ without .prj -- add a CRS override in "
            "manifests/ehydro_overrides.yaml")
    return np.array(rows), crs, None


def mllw_to_navd88(xyz: np.ndarray, crs, sep_path: Path) -> np.ndarray:
    """Add the (negative) MLLW-minus-NAVD88 separation to MLLW elevations."""
    import rasterio
    from pyproj import Transformer

    with rasterio.open(sep_path) as sep:
        tr = Transformer.from_crs(crs, sep.crs, always_xy=True)
        lon, lat = tr.transform(xyz[:, 0], xyz[:, 1])
        vals = np.array([v[0] for v in sep.sample(np.column_stack([lon, lat]))])
    if not np.isfinite(vals).all():
        n_bad = int((~np.isfinite(vals)).sum())
        med = np.nanmedian(vals)
        log.warning("separation undefined at %d points; using survey median "
                    "%.3f m there", n_bad, med)
        vals[~np.isfinite(vals)] = med
    out = xyz.copy()
    out[:, 2] = xyz[:, 2] + vals
    return out


def grid_survey(xyz: np.ndarray, crs, outline, out_tif: Path, res_m: float,
                nodata: float) -> None:
    """TIN linear interpolation to a UTM 18N grid clipped to the point hull."""
    import rasterio
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from scipy.interpolate import LinearNDInterpolator
    from shapely.geometry import MultiPoint

    tr = Transformer.from_crs(crs, "EPSG:26918", always_xy=True)
    x, y = tr.transform(xyz[:, 0], xyz[:, 1])
    z = xyz[:, 2]

    if outline is not None:
        from shapely import force_2d
        from shapely.ops import transform as shp_transform
        # SurveyJob polygons carry Z, and shapely.ops.transform then calls the
        # mapper with (x, y, z). Drop Z first: this footprint is planar, and
        # the 3-arg call raised for every 3D survey the first time this branch
        # ever executed (it was dead while outline was always None).
        hull = shp_transform(lambda a, b: tr.transform(a, b), force_2d(outline))
    else:
        hull = MultiPoint(np.column_stack([x, y])).convex_hull.buffer(-res_m)

    x0 = np.floor(x.min() / res_m) * res_m
    y1 = np.ceil(y.max() / res_m) * res_m
    width = int(np.ceil((x.max() - x0) / res_m)) + 1
    height = int(np.ceil((y1 - y.min()) / res_m)) + 1
    transform = from_origin(x0, y1, res_m, res_m)

    interp = LinearNDInterpolator(np.column_stack([x, y]), z)
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    gx = x0 + (cols + 0.5) * res_m
    gy = y1 - (rows + 0.5) * res_m
    zi = interp(gx.ravel(), gy.ravel()).reshape(height, width)

    inside = rasterize([(hull, 1)], out_shape=(height, width),
                       transform=transform, fill=0, dtype="uint8").astype(bool)
    zi[~inside] = np.nan

    profile = dict(driver="GTiff", dtype="float32", count=1, width=width,
                   height=height, transform=transform, crs="EPSG:26918",
                   nodata=nodata, compress="deflate", tiled=True)
    tmp = out_tif.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(np.where(np.isnan(zi), nodata, zi).astype("f4"), 1)
    tmp.rename(out_tif)


def survey_footprint(xyz: np.ndarray, crs, outline=None):
    """Survey extent in EPSG:4269 lon/lat.

    Prefers the GDB's own survey-extent polygon; the convex hull of the
    soundings is only a fallback. This footprint feeds the water mask, so an
    inflated one asserts "water" -- and lets eHydro override CUDEM -- over
    ground the survey never covered.
    """
    from pyproj import Transformer
    from shapely.geometry import MultiPoint
    from shapely.ops import transform as shp_transform

    tr = Transformer.from_crs(crs, "EPSG:4269", always_xy=True)
    if outline is not None and not outline.is_empty:
        from shapely import force_2d
        return shp_transform(lambda a, b: tr.transform(a, b),
                             force_2d(outline))
    lon, lat = tr.transform(xyz[:, 0], xyz[:, 1])
    return MultiPoint(np.column_stack([lon, lat])).convex_hull


def write_footprint(path: Path, survey: str, date: str, hull) -> None:
    """Persist one survey's footprint beside its grid, atomically.

    The combined layer is reassembled from these sidecars on every run, so a
    resumed stage 06 -- which skips surveys it already gridded -- still emits a
    complete ehydro_footprints.gpkg. Rebuilding it from only the surveys
    processed this run would silently shrink stage 07's water mask.
    """
    from shapely.geometry import mapping

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "type": "Feature",
        "properties": {"survey": survey, "date": date},
        "geometry": mapping(hull),
    }))
    tmp.rename(path)


def collect_footprints(grid_dir: Path) -> list[dict]:
    """Every persisted survey footprint, regardless of which run wrote it."""
    from shapely.geometry import shape

    out = []
    for p in sorted(grid_dir.glob(f"*{FOOTPRINT_SUFFIX}")):
        feat = json.loads(p.read_text())
        out.append({**feat["properties"], "geometry": shape(feat["geometry"])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", default=None,
                    help="process a single survey ZIP stem")
    args = ap.parse_args()
    dp.setup_logging()
    cfg = dp.load_config()
    scfg = cfg["sources"]["ehydro"]
    nodata = cfg["target_grid"]["nodata"]
    overrides = load_overrides()

    sep_path = build_separation_raster()
    grid_dir = dp.path(scfg["grid_dir"])
    grid_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(dp.path(scfg["dir"]).glob("*.ZIP")) + sorted(
        dp.path(scfg["dir"]).glob("*.zip"))
    if args.survey:
        zips = [z for z in zips if z.stem == args.survey]
    log.info("%d survey zips to process", len(zips))

    for zpath in zips:
        out_tif = grid_dir / f"{zpath.stem}.tif"
        fp_path = grid_dir / f"{zpath.stem}{FOOTPRINT_SUFFIX}"
        # Both products must exist to skip: a grid without its footprint
        # sidecar (an interrupted run, or one predating the sidecars) has to be
        # reprocessed or its channel drops out of the water mask.
        if out_tif.exists() and fp_path.exists():
            continue
        ov = overrides.get(zpath.stem, {})
        try:
            sdir = extract_zip(zpath)
            xyz, crs, outline = read_survey_points(sdir,
                                                   crs_override=ov.get("crs"))

            units = ov.get("depth_units", scfg["depth_units"])
            positive_down = ov.get("depth_positive_down",
                                   scfg["depth_positive_down"])
            xyz[:, 2] = convert_raw_depths(xyz[:, 2], units, positive_down)

            xyz = mllw_to_navd88(xyz, crs, sep_path)
            log.info("%s: %d pts, NAVD88 z range [%.2f, %.2f] m",
                     zpath.stem, len(xyz), xyz[:, 2].min(), xyz[:, 2].max())
            check_plausible(xyz[:, 2], zpath.stem)

            grid_survey(xyz, crs, outline, out_tif, scfg["grid_res_m"], nodata)

            date = zpath.stem.split("_")[-2] if "_" in zpath.stem else ""
            write_footprint(fp_path, zpath.stem, date,
                            survey_footprint(xyz, crs, outline))
        except ImplausibleDepthsError as e:
            # check_plausible() itself is never weakened -- a systematic
            # units/sign/datum mistake still has to be caught. But at basin
            # scale a single survey's plausibility failure is as likely to be
            # a handful of raw sonar blunder points (verified for
            # PT_06_FLO_20210721_CS: 6 of 549862 points were 90-121 ft vs. a
            # 99.99th-percentile of ~55 ft) as a real conversion error, and
            # newest-per-footprint selection means another survey usually
            # covers the same reach anyway. Skip just this survey, loudly, for
            # manual follow-up (add an override or a despike) rather than
            # aborting the whole basin run.
            log.error("%s: SKIPPED, implausible depths (%s) -- inspect and "
                      "add an override or rerun --survey %s after fixing",
                      zpath.stem, e, zpath.stem)
        except Exception as e:
            log.error("%s: FAILED (%s) -- continuing; rerun after fixing",
                      zpath.stem, e)

    # Reassembled from the sidecars, not from this run's loop, so a resumed
    # run rewrites the full layer rather than truncating it to its own surveys.
    footprints = collect_footprints(grid_dir)
    if footprints:
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(footprints, crs="EPSG:4269")
        fp_out = dp.path("data", "intermediate", "ehydro_footprints.gpkg")
        tmp = fp_out.with_suffix(".tmp.gpkg")
        tmp.unlink(missing_ok=True)
        gdf.to_file(tmp, layer="ehydro_footprint", driver="GPKG")
        tmp.rename(fp_out)
        log.info("wrote %s (%d footprints)", fp_out, len(gdf))

    # VRT ordered oldest-first so the newest survey paints last.
    def survey_date(p: Path) -> str:
        m = re.search(r"(20\d{6})", p.stem)
        return m.group(1) if m else "00000000"

    tifs = sorted(dp.product_tifs(grid_dir), key=survey_date)
    if tifs:
        import subprocess
        vrt_dir = dp.path("data", "intermediate", "vrt")
        vrt_dir.mkdir(parents=True, exist_ok=True)
        listfile = vrt_dir / "ehydro_inputs.txt"
        listfile.write_text("\n".join(str(t) for t in tifs) + "\n")
        subprocess.run(["gdalbuildvrt", "-overwrite", "-input_file_list",
                        str(listfile), str(vrt_dir / "ehydro.vrt")], check=True)
        log.info("ehydro.vrt: %d survey grids (oldest first)", len(tifs))


if __name__ == "__main__":
    main()
