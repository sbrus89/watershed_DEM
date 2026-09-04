"""MLLW -> NAVD88 vertical transformation for eHydro soundings.

Builds a separation raster from NOAA VDatum regional grids (pure GDAL, no
Java). The raster stores the elevation of the MLLW surface relative to
NAVD88 (``mllw_minus_navd88``, typically -0.5 to -0.9 m in the Delaware
estuary), so that::

    z_navd88 = z_mllw + mllw_minus_navd88(x, y)

VDatum ships two gtx grids per tidal region: ``tss.gtx`` (NAVD88 <-> LMSL sea
surface topography) and ``mllw.gtx`` (LMSL <-> MLLW). Their sign conventions
are easy to get backwards, so instead of hard-coding them we evaluate all
four sign combinations at NOAA CO-OPS benchmark stations with published
datums and keep the combination that reproduces them; the build hard-fails
if no combination fits within ``vdatum.max_benchmark_residual_m``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import requests

from . import load_config, path as repo_path

log = logging.getLogger(__name__)


def station_mllw_in_navd88(station: str) -> tuple[float, float, float]:
    """(lon, lat, mllw_minus_navd88 [m]) from the CO-OPS metadata API.

    Two requests: datums.json carries the datum values but NOT the station
    coordinates (its top-level "LAT" is Lowest Astronomical Tide, not
    latitude); lon/lat come from the station metadata endpoint.
    """
    cfg = load_config()["vdatum"]
    r = requests.get(cfg["datums_api"].format(station=station),
                     params={"units": "metric"}, timeout=60)
    r.raise_for_status()
    datums = {d["name"]: d["value"] for d in r.json()["datums"]}
    if "NAVD88" not in datums or "MLLW" not in datums:
        raise ValueError(f"station {station} lacks NAVD88/MLLW datums")

    r = requests.get(cfg["station_api"].format(station=station), timeout=60)
    r.raise_for_status()
    meta = r.json()["stations"][0]
    # Values are elevations of each datum surface above the station datum,
    # so MLLW - NAVD88 is the elevation of MLLW in the NAVD88 frame.
    return meta["lng"], meta["lat"], datums["MLLW"] - datums["NAVD88"]


SAMPLE_FIELDS = ["lon", "lat", "sep", "uncertainty"]

# VDatum signals "no tidal model here" with a -999999.0 sentinel in t_z rather
# than an error, and float() takes it happily: 327 of 476 apparently-successful
# responses over this estuary were sentinels. Interpolating them produced
# separations of ~-850000 m. Anything beyond a few metres is not a real
# MLLW-NAVD88 separation anywhere in CONUS.
PLAUSIBLE_SEP_M = 10.0


def _service_params(cfg: dict, lon: float, lat: float) -> dict:
    """Query for 'what is the NAVD88 elevation of the MLLW surface here?'.

    s_z=0 in MLLW converted to NAVD88 IS the separation, by definition.
    """
    return {
        "s_x": f"{lon:.6f}", "s_y": f"{lat:.6f}", "s_z": "0.0",
        "region": cfg["region"],
        "s_h_frame": cfg["source_h_frame"], "s_coor": "geo", "s_h_unit": "m",
        "s_v_frame": "MLLW", "s_v_unit": "m", "s_v_elevation": "height",
        "s_v_geoid": cfg["geoid"],
        "t_h_frame": cfg["target_h_frame"], "t_coor": "geo", "t_h_unit": "m",
        "t_v_frame": "NAVD88", "t_v_unit": "m", "t_v_elevation": "height",
        "t_v_geoid": cfg["geoid"],
    }


def sample_separation(lon: float, lat: float, cfg: dict):
    """(separation_m, uncertainty_m) at one point, or None.

    None means VDatum has no tidal model there -- dry land, or outside the
    region -- which is expected over much of a regular sampling grid and must
    not be confused with a transport failure.
    """
    try:
        r = requests.get(cfg["service_url"],
                         params=_service_params(cfg, lon, lat), timeout=90)
        payload = r.json()
    except (requests.RequestException, ValueError):
        return None
    try:
        sep = float(payload["t_z"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (abs(sep) <= PLAUSIBLE_SEP_M):     # also rejects NaN
        return None
    try:
        unc = float(payload.get("uncertainty"))
    except (TypeError, ValueError):
        unc = float("nan")
    return sep, unc


def _read_sample_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (round(float(row["lon"]), 6), round(float(row["lat"]), 6))
            # A blank sep is a cached MISS (land / outside coverage). Keeping
            # those is the point: without them every run re-queries hundreds
            # of dry points that will never return a value.
            out[key] = (float(row["sep"]) if row["sep"] else None,
                        float(row["uncertainty"]) if row["uncertainty"] else
                        float("nan"))
    return out


def _write_sample_cache(path: Path, samples: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
        w.writeheader()
        for (lon, lat), (sep, unc) in sorted(samples.items()):
            w.writerow({"lon": lon, "lat": lat,
                        "sep": "" if sep is None else f"{sep:.4f}",
                        "uncertainty": "" if sep is None else f"{unc:.4f}"})
    tmp.rename(path)


def collect_samples(cfg: dict, cache_path: Path | None = None) -> dict:
    """Sample the VDatum service over the configured grid, with caching.

    Returns {(lon, lat): (sep|None, uncertainty)}. Cached points are never
    re-queried, so this costs one pass and the pipeline stays reproducible
    (and offline-capable) afterwards.
    """
    from concurrent.futures import ThreadPoolExecutor

    cache_path = Path(cache_path or repo_path(cfg["sample_cache"]))
    samples = _read_sample_cache(cache_path)

    w, s_, e, n = cfg["sample_bounds"]
    step = cfg["sample_spacing_deg"]
    grid = [(round(w + i * step, 6), round(s_ + j * step, 6))
            for i in range(int(round((e - w) / step)) + 1)
            for j in range(int(round((n - s_) / step)) + 1)]
    todo = [pt for pt in grid if pt not in samples]
    if todo:
        log.info("sampling VDatum at %d points (%d already cached)",
                 len(todo), len(samples))
        with ThreadPoolExecutor(max_workers=cfg.get("service_workers", 4)) as ex:
            results = list(ex.map(
                lambda pt: sample_separation(pt[0], pt[1], cfg), todo))
        for pt, res in zip(todo, results):
            samples[pt] = res if res is not None else (None, float("nan"))
        _write_sample_cache(cache_path, samples)

    hits = {k: v for k, v in samples.items() if v[0] is not None}
    log.info("VDatum samples: %d of %d points have a tidal model",
             len(hits), len(samples))
    return samples


def build_separation_raster(out_path: Path | None = None) -> Path:
    """MLLW-minus-NAVD88 separation raster, interpolated from VDatum samples.

    Raster value = elevation of the MLLW surface in the NAVD88 frame (negative
    in this estuary): z_navd88 = z_mllw + value.

    Composing VDatum's raw tss/mllw .gtx grids by hand does NOT reproduce
    VDatum's own transformation. Measured at 11 CO-OPS gauges on 2026-08-27,
    `mllw` alone is off by up to 0.18 m and `tss + mllw` by up to 0.70 m, both
    with a systematic along-estuary trend; the service reproduces the gauges to
    0.026 m median / 0.082 m max. The service is therefore the authority, and
    the sign-combination search that used to live here is gone with the
    arithmetic that needed it.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from scipy.interpolate import griddata

    cfg = load_config()["vdatum"]
    out_path = Path(out_path or repo_path(cfg["sep_raster"]))
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(cfg)
    usable = [((lon, lat), sep) for (lon, lat), (sep, _) in samples.items()
              if sep is not None and abs(sep) <= PLAUSIBLE_SEP_M]
    pts = np.array([[lon, lat] for (lon, lat), _ in usable], dtype="f8")
    vals = np.array([sep for _, sep in usable], dtype="f8")
    if len(pts) < 4:
        raise RuntimeError(
            f"only {len(pts)} usable VDatum samples over {cfg['sample_bounds']}"
            " -- cannot interpolate a separation surface; check the service "
            "parameters (this region needs s_h_frame=IGS14 for tidal datums)")

    w, s_, e, n = cfg["sample_bounds"]
    res = cfg["sep_res_deg"]
    width = int(round((e - w) / res))
    height = int(round((n - s_) / res))
    transform = from_origin(w, n, res, res)
    # cell centres
    gx, gy = np.meshgrid(w + (np.arange(width) + 0.5) * res,
                         n - (np.arange(height) + 0.5) * res)

    sep = griddata(pts, vals, (gx, gy), method="linear")
    # Linear interpolation is undefined outside the sample convex hull; fill
    # those with nearest so a survey just past the last wet sample still
    # converts, rather than silently falling back to a survey-wide median.
    holes = ~np.isfinite(sep)
    if holes.any():
        sep[holes] = griddata(pts, vals, (gx[holes], gy[holes]),
                              method="nearest")

    # --- benchmark gate: the separation surface must reproduce published
    #     CO-OPS datums, or a units/sign/model error reaches the bathymetry ---
    tol = cfg["max_benchmark_residual_m"]
    residuals = {}
    for station in cfg["benchmark_stations"]:
        lon, lat, truth = station_mllw_in_navd88(station)
        col = int((lon - w) / res)
        row = int((n - lat) / res)
        if 0 <= row < height and 0 <= col < width and np.isfinite(sep[row, col]):
            residuals[station] = abs(float(sep[row, col]) - truth)
    if not residuals:
        raise RuntimeError(
            "no benchmark station fell inside the separation raster -- check "
            f"sample_bounds {cfg['sample_bounds']} against the station list")
    worst_station = max(residuals, key=residuals.get)
    worst = residuals[worst_station]
    log.info("benchmark residuals (m): %s",
             {k: round(v, 3) for k, v in sorted(residuals.items())})
    if worst > tol:
        raise RuntimeError(
            f"separation raster disagrees with CO-OPS station {worst_station} "
            f"by {worst:.3f} m (limit {tol} m); do not weaken this gate -- it "
            "is what stops a datum error reaching the channel bathymetry")
    log.info("separation verified: worst residual %.3f m at %s (limit %.2f)",
             worst, worst_station, tol)

    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "width": width, "height": height, "transform": transform,
        "crs": "EPSG:4269", "nodata": np.nan, "compress": "deflate",
    }
    tmp = out_path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(sep.astype("f4"), 1)
        dst.update_tags(
            description="elevation of MLLW surface relative to NAVD88 (m); "
                        "z_navd88 = z_mllw + this",
            source=f"VDatum service, region={cfg['region']}, "
                   f"geoid={cfg['geoid']}, s_h_frame={cfg['source_h_frame']}",
            n_samples=str(len(pts)),
            worst_benchmark_residual_m=f"{worst:.4f}",
            worst_benchmark_station=worst_station,
        )
    tmp.rename(out_path)
    return out_path
