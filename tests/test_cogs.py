"""Tests for stage 09's per-tile COG build (scripts/09_make_cogs_vrt.py).

The bug this guards: skip-if-exists alone can't tell a COG built under an
older GridSpec (different collar/shift constants) from a fresh, correct one
-- that let stale COGs from an earlier config silently fail grid_conformity
in production. make_cog() must check the existing file's actual geotransform
before trusting it, exactly like stage 08's compose_tile().
"""

import importlib.util
import subprocess
from pathlib import Path

from tests.util import importorskip, make_geotiff

_SPEC = importlib.util.spec_from_file_location(
    "make_cogs_vrt",
    Path(__file__).resolve().parents[1] / "scripts" / "09_make_cogs_vrt.py",
)
mkcogs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mkcogs)

from dem_pipeline.grid import GridSpec  # noqa: E402

TILE = "n40x25_w075x25"
CFG = {
    "project": {"name": "test19", "version": "2026v1"},
    "target_grid": {"epsg": 4269},
    "output": {"cog": {"compress": "deflate", "predictor": 3,
                       "blocksize": 512, "overview_resampling": "average"}},
}


def _stub_gdal_translate(monkeypatch, calls):
    """Fake gdal_translate: copy the source's own geotransform/size through,
    as the real tool would for a straight format conversion."""
    import rasterio

    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[0] == "gdal_translate":
            src, dst = Path(cmd[-2]), Path(cmd[-1])
            with rasterio.open(src) as s:
                make_geotiff(dst, s.read(1), s.transform, s.crs, s.nodata)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(mkcogs.subprocess, "run", run)


def test_make_cog_rebuilds_when_grid_spec_changed(tmp_path, monkeypatch):
    rasterio = importorskip("rasterio")
    from rasterio.transform import Affine

    spec = GridSpec(epsg=4269, nodata=-999999.0, collar_px=1,
                    x_shift_px=0.0, y_shift_px=0.0)
    gt, w, h = spec.tile_geotransform(TILE)

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    src = make_geotiff(in_dir / f"{TILE}.tif", [[1.0] * w] * h,
                       Affine.from_gdal(*gt), "EPSG:4269", nodata=-999999.0)

    # plant a stale COG under an old shifted origin
    stale_gt = list(gt)
    stale_gt[0] += 5 * spec.pixel_deg
    dst = out_dir / f"test19_{TILE}_2026v1.tif"
    make_geotiff(dst, [[1.0] * w] * h, Affine.from_gdal(*stale_gt),
                "EPSG:4269", nodata=-999999.0)

    calls = []
    _stub_gdal_translate(monkeypatch, calls)

    out = mkcogs.make_cog(src, in_dir, out_dir, CFG, spec)

    assert any(c[0] == "gdal_translate" for c in calls), (
        "stale COG must be rebuilt, not skipped")
    with rasterio.open(out) as result:
        assert result.transform.to_gdal() == gt
        assert (result.width, result.height) == (w, h)


def test_make_cog_skips_when_already_current(tmp_path, monkeypatch):
    rasterio = importorskip("rasterio")
    from rasterio.transform import Affine

    spec = GridSpec(epsg=4269, nodata=-999999.0, collar_px=1,
                    x_shift_px=0.0, y_shift_px=0.0)
    gt, w, h = spec.tile_geotransform(TILE)

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    src = make_geotiff(in_dir / f"{TILE}.tif", [[1.0] * w] * h,
                       Affine.from_gdal(*gt), "EPSG:4269", nodata=-999999.0)
    dst = out_dir / f"test19_{TILE}_2026v1.tif"
    make_geotiff(dst, [[1.0] * w] * h, Affine.from_gdal(*gt),
                "EPSG:4269", nodata=-999999.0)

    calls = []
    _stub_gdal_translate(monkeypatch, calls)

    mkcogs.make_cog(src, in_dir, out_dir, CFG, spec)

    assert calls == [], "a COG already matching the current GridSpec must not be rebuilt"
    del rasterio
