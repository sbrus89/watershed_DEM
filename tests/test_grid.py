"""Unit tests for the CUDEM tile lattice math (dem_pipeline.grid)."""

import json

from dem_pipeline.grid import (CELLS_PER_DEG, PIXEL_DEG, TILE_PX, GridSpec,
                               lattice_tiles, nominal_extent, output_filename,
                               parse_tile_name, tile_name)
from tests.util import raises


def test_tile_name_roundtrip():
    for north, west in [(40.50, -75.25), (38.75, -76.00), (42.25, -74.75)]:
        name = tile_name(north, west)
        assert parse_tile_name(name) == (north, west)


def test_parse_from_cudem_filename():
    assert parse_tile_name("ncei19_n40x25_w075x00_2014v1.tif") == (40.25, -75.0)
    assert parse_tile_name("dewatershed19_n39x75_w075x50_2026v1.tif") == (
        39.75, -75.5)


def test_parse_rejects_garbage():
    with raises(ValueError):
        parse_tile_name("not_a_tile.tif")


def test_nominal_extent():
    w, s, e, n = nominal_extent("n40x00_w075x25")
    assert (w, s, e, n) == (-75.25, 39.75, -75.0, 40.0)
    assert abs((e - w) - 0.25) < 1e-15
    assert abs((n - s) - 0.25) < 1e-15


def test_nominal_geotransform_exact():
    spec = GridSpec()
    gt, width, height = spec.tile_geotransform("n40x00_w075x25")
    assert (width, height) == (TILE_PX, TILE_PX)
    assert gt[0] == -75.25 and gt[3] == 40.0
    # pixel edges land exactly on the quarter-degree lines
    assert abs(gt[1] * TILE_PX - 0.25) < 1e-12
    assert abs(gt[5] * TILE_PX + 0.25) < 1e-12
    assert abs(gt[1] - 1.0 / CELLS_PER_DEG) < 1e-18


def test_collar_and_shift():
    spec = GridSpec(collar_px=6, x_shift_px=0.5, y_shift_px=-0.5)
    gt, width, height = spec.tile_geotransform("n40x00_w075x25")
    assert width == height == TILE_PX + 12
    assert abs(gt[0] - (-75.25 - 6 * PIXEL_DEG + 0.5 * PIXEL_DEG)) < 1e-15
    assert abs(gt[3] - (40.0 + 6 * PIXEL_DEG - 0.5 * PIXEL_DEG)) < 1e-15


def test_cudem_geotransform_verbatim():
    gt_cudem = [-75.2500463, PIXEL_DEG, 0.0, 40.0000463, 0.0, -PIXEL_DEG]
    spec = GridSpec(
        cudem_geotransforms={"n40x00_w075x25": gt_cudem},
        cudem_sizes={"n40x00_w075x25": [8103, 8103]},
    )
    gt, w, h = spec.tile_geotransform("n40x00_w075x25")
    assert list(gt) == gt_cudem and (w, h) == (8103, 8103)
    # other tiles still use the nominal lattice
    gt2, w2, _ = spec.tile_geotransform("n40x00_w075x00")
    assert gt2[0] == -75.0 and w2 == TILE_PX


def test_gridspec_dump_load_roundtrip(tmp_path):
    spec = GridSpec(epsg=4269, nodata=-999999.0, collar_px=3,
                    cudem_geotransforms={"n40x00_w075x25": [1, 2, 3, 4, 5, 6]},
                    cudem_sizes={"n40x00_w075x25": [10, 10]})
    p = tmp_path / "tile_grid.json"
    spec.dump(p)
    loaded = GridSpec.load(p)
    assert loaded == spec
    # file is valid json with expected keys
    d = json.loads(p.read_text())
    assert d["collar_px"] == 3


def test_lattice_tiles_cover_bbox():
    tiles = lattice_tiles((-75.3, 39.8, -75.1, 40.1))
    # bbox straddles 2 columns x 2 rows of the quarter-degree lattice
    assert set(tiles) == {"n40x00_w075x25", "n40x00_w075x50",
                          "n40x25_w075x25", "n40x25_w075x50"}
    for t in tiles:
        w, s, e, n = nominal_extent(t)
        assert w < -75.1 and e > -75.3 and s < 40.1 and n > 39.8


def test_output_filename_uses_project_config():
    cfg = {"project": {"name": "dewatershed19", "version": "2026v1"}}
    assert output_filename("n40x00_w075x25", cfg) == \
        "dewatershed19_n40x00_w075x25_2026v1.tif"
