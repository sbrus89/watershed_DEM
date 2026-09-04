"""Tests for stage 02's audit_tiles -- the function that derives the target
grid spec from real CUDEM tiles. The entire product's drop-in compatibility
(bit-exact geotransforms) hinges on this derivation being right.

Loaded via importlib (numbered script). Synthetic tiles use the true 1/9"
pixel size but small pixel counts, which audit_tiles permits (it warns on
width-vs-collar inconsistency but only hard-fails on pixel size).
"""

import importlib.util
from pathlib import Path

import numpy as np

from tests.util import importorskip, make_geotiff, raises

_SPEC = importlib.util.spec_from_file_location(
    "audit_cudem_grid",
    Path(__file__).resolve().parents[1] / "scripts" / "02_audit_cudem_grid.py",
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

PX = 1.0 / 32400  # exact 1/9 arc-second
NODATA = -999999.0


def _cudem_tile(tmp_path, name, west, north, collar_px=0, size=60,
                pixel=PX, nodata=NODATA):
    importorskip("rasterio")
    from rasterio.transform import Affine

    gt = (west - collar_px * pixel, pixel, 0.0,
          north + collar_px * pixel, 0.0, -pixel)
    return make_geotiff(
        tmp_path / f"ncei19_{name}_2014v1.tif",
        np.zeros((size, size), dtype="f4"),
        Affine.from_gdal(*gt), "EPSG:4269", nodata=nodata)


def test_audit_derives_modal_convention_and_records_geotransforms(tmp_path):
    paths = [
        _cudem_tile(tmp_path, "n40x00_w075x25", -75.25, 40.00),
        _cudem_tile(tmp_path, "n40x00_w075x00", -75.00, 40.00),
        # one odd tile with a 6-px collar: the modal (zero-collar) convention
        # must win for non-CUDEM tiles, while this tile keeps its exact gt
        _cudem_tile(tmp_path, "n40x25_w075x25", -75.25, 40.25, collar_px=6),
    ]
    spec = audit.audit_tiles(paths)

    assert spec.epsg == 4269
    assert spec.nodata == NODATA
    assert spec.collar_px == 0 and spec.x_shift_px == 0.0
    assert len(spec.cudem_geotransforms) == 3

    # CUDEM tiles reproduce their own geotransforms verbatim (bit-exact)
    gt, w, h = spec.tile_geotransform("n40x25_w075x25")
    assert gt[0] == -75.25 - 6 * PX and gt[3] == 40.25 + 6 * PX
    assert (w, h) == (60, 60)
    # non-CUDEM tiles get the modal convention on the nominal lattice
    gt, w, h = spec.tile_geotransform("n39x75_w075x25")
    assert gt[0] == -75.25 and gt[3] == 39.75 and w == 8100


def test_audit_rejects_wrong_pixel_size(tmp_path):
    bad = _cudem_tile(tmp_path, "n40x00_w075x25", -75.25, 40.00,
                      pixel=1.0 / 10800)  # 1/3", not 1/9"
    with raises(SystemExit):
        audit.audit_tiles([bad])


def test_audit_detects_half_cell_registration(tmp_path):
    """A vintage with pixel centers (not edges) on the quarter-degree line
    shows up as a half-pixel shift, which must be captured, not silently
    snapped to the nominal lattice."""
    importorskip("rasterio")
    from rasterio.transform import Affine

    gt = (-75.25 - 0.5 * PX, PX, 0.0, 40.00 + 0.5 * PX, 0.0, -PX)
    p = make_geotiff(tmp_path / "ncei19_n40x00_w075x25_2014v1.tif",
                     np.zeros((60, 60), dtype="f4"),
                     Affine.from_gdal(*gt), "EPSG:4269", nodata=NODATA)
    spec = audit.audit_tiles([p])
    assert spec.x_shift_px == -0.5 and spec.y_shift_px == 0.5
    # applying the spec to a new tile reproduces the same registration
    gt2, _, _ = spec.tile_geotransform("n39x75_w075x00")
    assert abs(gt2[0] - (-75.0 - 0.5 * PX)) < 1e-12
    assert abs(gt2[3] - (39.75 + 0.5 * PX)) < 1e-12


def test_gate_accepts_real_vintage_roundings_but_rejects_wrong_lattice(tmp_path):
    """Real CUDEM tiles store the pixel size at each vintage's own precision.

    Measured across the 49 Delaware-basin tiles, the spread reaches 3.7e-12
    deg/px (2019v1 rounds to 8 significant figures) -- ~3 mm of drift across a
    full 8100-px tile, i.e. the same lattice printed differently. Rejecting
    those aborts stage 02 on real data; accepting a genuinely different pixel
    size would silently put every output tile off-grid. The gate has to
    separate the two.
    """
    importorskip("rasterio")

    # the two vintages that fail a raw 1e-12 epsilon but are the same grid
    for i, px in enumerate((3.0864194e-05, 3.0864201183432992e-05)):
        _cudem_tile(tmp_path, f"n40x0{i}_w075x25", -75.25, 40.0 + i, pixel=px)
    spec = audit.audit_tiles(sorted(tmp_path.glob("*.tif")))
    assert len(spec.cudem_geotransforms) == 2
    # each tile keeps its own geotransform verbatim -- that is what preserves
    # bit-exactness against CUDEM, not the tolerance
    assert spec.cudem_geotransforms["n40x00_w075x25"][1] == 3.0864194e-05

    # 1/3 arc-second is a different lattice and must still abort
    bad = tmp_path / "wrong"
    bad.mkdir()
    _cudem_tile(bad, "n41x00_w075x25", -75.25, 41.0, pixel=1.0 / 10800)
    with raises(SystemExit):
        audit.audit_tiles(sorted(bad.glob("*.tif")))
