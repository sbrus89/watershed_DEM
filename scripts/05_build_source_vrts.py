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

SOURCES = ["s1m", "dem1m_projects", "dem13", "cudem19", "eaarlb", "coned"]


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


def missing_downloads() -> dict:
    """Manifest rows per source whose file is absent from disk.

    This is how stage 05 tells 'this source legitimately contributes no
    raster' apart from 'stage 04 did not finish'. Counting manifest rows
    against *.tif cannot make that distinction: EAARL-B's manifest rows are
    zips plus an xml, a shapefile index and the void polygons, so a source
    with every byte downloaded still shows zero .tif files until stage 05
    unpacks the archives.
    """
    from collections import Counter

    from dem_pipeline.download import read_manifest

    manifest = dp.path("manifests", "downloads.csv")
    if not manifest.exists():
        return {}
    return Counter(r["source"] for r in read_manifest(manifest)
                   if not Path(r["dest"]).exists())


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
    argparse.ArgumentParser(description=__doc__).parse_args()
    dp.setup_logging()
    cfg = dp.load_config()

    vrt_dir = dp.path("data", "intermediate", "vrt")
    vrt_dir.mkdir(parents=True, exist_ok=True)
    missing = missing_downloads()

    for name in SOURCES:
        scfg = cfg["sources"][name]
        if not scfg.get("enabled", True):
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
        build_vrt(name, tifs, vrt_dir)


if __name__ == "__main__":
    main()
