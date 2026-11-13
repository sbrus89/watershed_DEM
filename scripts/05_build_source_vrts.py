#!/usr/bin/env python
"""Stage 05: build one VRT per source under data/intermediate/vrt/.

eHydro is excluded here -- its VRT is built by stage 06 after gridding, with
inputs ordered oldest-first so the newest survey paints last.
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import dem_pipeline as dp

log = logging.getLogger("05_build_source_vrts")

SOURCES = ["s1m", "dem1m_projects", "dem13", "cudem19", "eaarlb", "coned",
           "nos_grids"]

# Whole-raster sanity window for sources that arrive ALREADY in NAVD88 metres
# (no conversion of ours to catch a units/sign slip, unlike eHydro's
# ImplausibleDepthsError). Checked per input file from its own statistics.
NAVD88_CHECKED_SOURCES = {"nos_grids"}


class ImplausibleRasterError(SystemExit):
    """A NAVD88 water raster's values are inconsistent with bathymetry in
    metres, positive up -- e.g. depths stored positive-down, or feet."""


def check_navd88_water_raster(path, lo: float, hi: float) -> None:
    """Fail loudly unless `path` looks like NAVD88 metres, positive up.

    Uses the smallest overview for speed (the 0.5 m Hudson surveys are
    multi-GB): the range must sit inside [lo, hi], and the median must be
    below 0 -- a positive-down survey has a positive median, which a range
    check alone would accept for shallow water.
    """
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ImplausibleRasterError(f"{path}: no CRS -- cannot place it")
        ovr = src.overviews(1)
        shape = (src.height // ovr[-1], src.width // ovr[-1]) if ovr else None
        a = src.read(1, out_shape=shape, masked=True).astype("f8")
    v = a.compressed()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ImplausibleRasterError(f"{path}: no valid pixels")
    zmin, zmax, zmed = float(v.min()), float(v.max()), float(np.median(v))
    if zmin < lo or zmax > hi or not zmed < 0.0:
        raise ImplausibleRasterError(
            f"{path}: values [{zmin:.2f}, {zmax:.2f}] median {zmed:.2f} m are "
            f"not NAVD88 bathymetry in metres, positive up (expected within "
            f"[{lo}, {hi}] with a negative median) -- check units/sign")
    log.info("%s: NAVD88 check ok, [%.2f, %.2f] median %.2f m",
             Path(path).name, zmin, zmax, zmed)


def extract_archives(raw_dir) -> int:
    """Extract any downloaded ZIP that holds rasters, so the glob below sees
    them.

    EAARL-B ships one GeoTIFF per zip -- nothing else in the pipeline unpacks
    them, so stage 05 saw an empty source dir against a 24-row manifest and
    aborted with "stage 04 did not complete". Extraction is atomic (tmp dir
    then rename) and skipped when the target already exists, so a killed run
    cannot leave a half-unpacked directory that looks finished.
    """
    import zipfile

    n = 0
    for z in sorted(raw_dir.glob("*.zip")):
        out = z.with_suffix("")
        if out.exists():
            continue
        with zipfile.ZipFile(z) as zf:
            if not any(m.lower().endswith(".tif") for m in zf.namelist()):
                continue          # index shapefiles, void polygons, metadata
            tmp = z.with_suffix(".tmp_extract")
            if tmp.exists():
                shutil.rmtree(tmp)
            zf.extractall(tmp)
        tmp.rename(out)
        n += 1
    if n:
        log.info("extracted %d raster archive(s) in %s", n, raw_dir)
    return n


def missing_downloads(bounds=None) -> dict:
    """Manifest rows per source whose file is absent from disk.

    With `bounds` (w, s, e, n), only rows intersecting them count -- the same
    filter stage 04 applies under --tile. Without it, a single-tile test
    slice false-failed whenever a gap-fill source (dem1m_projects) had
    basin-wide rows outside the tile: stage 04 correctly fetched none of
    them, and this check read that as an interrupted download.

    This is how stage 05 tells 'this source legitimately contributes no
    raster' apart from 'stage 04 did not finish'. Counting manifest rows
    against *.tif cannot make that distinction: EAARL-B's manifest rows are
    zips plus an xml, a shapefile index and the void polygons, so a source
    with every byte downloaded still shows zero .tif files until stage 05
    unpacks the archives.
    """
    from collections import Counter

    from dem_pipeline.download import read_manifest, rows_in_bounds

    manifest = dp.path("manifests", "downloads.csv")
    if not manifest.exists():
        return {}
    rows = read_manifest(manifest)
    if bounds:
        rows = rows_in_bounds(rows, bounds)
    return Counter(r["source"] for r in rows if not Path(r["dest"]).exists())


def horizontal_epsg(path) -> int | None:
    """Horizontal (2D) EPSG of a raster.

    A compound CRS shares its horizontal frame with the plain 2D code --
    EPSG:5498 is NAD83 + NAVD88 height, horizontally EPSG:4269 -- and a VRT
    only places its sources horizontally, so those can safely share one VRT.
    """
    import pyproj
    import rasterio

    with rasterio.open(path) as src:
        crs = src.crs
    return pyproj.CRS.from_user_input(crs).to_2d().to_epsg() if crs else None


def build_vrt(name: str, tifs, vrt_dir) -> None:
    vrt = vrt_dir / f"{name}.vrt"
    listfile = vrt_dir / f"{name}_inputs.txt"
    listfile.write_text("\n".join(str(t) for t in tifs) + "\n")

    # Real CUDEM mixes EPSG:5498 and EPSG:4269 across vintages. gdalbuildvrt
    # refuses heterogeneous projections by SILENTLY SKIPPING the odd ones out
    # and still exiting 0 -- 28 of the 49 Delaware tiles vanished that way,
    # taking all bathymetry with them. The two codes are horizontally
    # identical, so allow the difference, but only after proving that.
    horiz = {horizontal_epsg(t) for t in tifs}
    if len(horiz) > 1:
        raise SystemExit(
            f"{name}: inputs span horizontal CRSs {sorted(map(str, horiz))} -- "
            "a VRT cannot mix these; reproject before building the VRT")
    subprocess.run(
        ["gdalbuildvrt", "-overwrite", "-allow_projection_difference",
         "-input_file_list", str(listfile), str(vrt)],
        check=True,
    )

    # gdalbuildvrt exits 0 even when it drops sources, so never trust the
    # exit code alone: count what actually landed in the VRT.
    n_src = vrt.read_text().count("<SourceFilename")
    if n_src != len(tifs):
        raise SystemExit(
            f"{name}: VRT contains {n_src} of {len(tifs)} inputs -- "
            "gdalbuildvrt dropped sources (CRS, band count or dtype mismatch)")
    log.info("%s: %d inputs -> %s", name, n_src, vrt)


def main() -> None:
    from dem_pipeline.grid import nominal_extent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", default=None,
                    help="only require downloads intersecting this tile "
                         "(match the --tile given to stage 04)")
    args = ap.parse_args()
    dp.setup_logging()
    cfg = dp.load_config()

    vrt_dir = dp.path("data", "intermediate", "vrt")
    vrt_dir.mkdir(parents=True, exist_ok=True)
    missing = missing_downloads(
        nominal_extent(args.tile) if args.tile else None)

    for name in SOURCES:
        scfg = cfg["sources"].get(name)
        if scfg is None or not scfg.get("enabled", True):
            continue
        raw = dp.path(scfg["dir"])
        if raw.exists():
            extract_archives(raw)
        tifs = sorted(raw.rglob("*.tif"))
        if not tifs:
            # An empty source directory is normal (EAARL-B outside the upper
            # basin, CoNED disabled). It is NOT normal when stage 04 was told
            # to fetch files for it -- that means the download stage did not
            # finish, and continuing would compose tiles with a silently
            # missing source.
            if missing.get(name):
                raise SystemExit(
                    f"{name}: {missing[name]} manifest file(s) are absent from "
                    f"{raw} -- stage 04 did not complete for this source "
                    "(check for .part files); rerun stage 04 before this one")
            log.info("%s: downloads complete but no rasters, skipping", name)
            continue
        if name in NAVD88_CHECKED_SOURCES:
            lo, hi = scfg["plausible_range_m"]
            for t in tifs:
                check_navd88_water_raster(t, lo, hi)
        build_vrt(name, tifs, vrt_dir)


if __name__ == "__main__":
    main()
