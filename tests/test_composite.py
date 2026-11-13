"""Tests for the per-tile compositor (dem_pipeline.composite).

compose_tile is exercised end-to-end on a miniature 90x90 tile grid with
synthetic sources: a "s1m" land raster with a nodata hole over the "river"
and a "cudem19" raster carrying bathymetry there, so the dynamic
S1M-void & CUDEM-valid mask term drives the land/water blend.
"""

import numpy as np

from dem_pipeline.composite import feather_alpha, first_valid, warp_to_tile
from tests.util import importorskip, make_geotiff

NODATA = -999999.0

# 90x90 px quarter-degree test tile => 1/360 deg pixels; stored as a
# "CUDEM-audited" geotransform so GridSpec uses it verbatim.
TILE = "n40x00_w075x25"
PX = 0.25 / 90
GT = [-75.25, PX, 0.0, 40.0, 0.0, -PX]


def test_first_valid_priority_and_ids():
    a = np.full((4, 4), np.nan, dtype="f4")
    a[0, 0] = 1.0
    b = np.full((4, 4), 2.0, dtype="f4")
    b[3, 3] = np.nan
    out, ids = first_valid({"hi": a, "lo": b}, {"hi": 1, "lo": 2})
    assert out[0, 0] == 1.0 and ids[0, 0] == 1          # hi wins where valid
    assert out[1, 1] == 2.0 and ids[1, 1] == 2          # lo fills the rest
    assert np.isnan(out[3, 3]) and ids[3, 3] == 0       # nobody valid


def test_feather_alpha_ramp():
    importorskip("scipy")
    mask = np.zeros((20, 40), dtype=bool)
    mask[:, 20:] = True  # right half is water
    alpha = feather_alpha(mask, feather_px=5)
    assert np.all(alpha[:, :14] == 0.0)   # deep land
    assert np.all(alpha[:, 26:] == 1.0)   # deep water
    ramp = alpha[10, 14:26]
    assert np.all(np.diff(ramp) >= 0)     # monotonic across the boundary
    assert 0.0 < alpha[10, 20] < 1.0


def test_feather_alpha_zero_width_is_hard_mask():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    alpha = feather_alpha(mask, feather_px=0)
    assert np.array_equal(alpha, mask.astype("f4"))


def test_feather_alpha_degenerate_masks_are_constant():
    """All-water/all-land masks have no boundary; the EDT of an all-True
    mask measures distance to the array corner, which (unguarded) would
    average land values into open-water tiles at that corner."""
    importorskip("scipy")
    all_water = np.ones((40, 40), dtype=bool)
    assert np.all(feather_alpha(all_water, feather_px=15) == 1.0)
    all_land = np.zeros((40, 40), dtype=bool)
    assert np.all(feather_alpha(all_land, feather_px=15) == 0.0)


def test_feather_overlay_ramps_from_footprint_edge():
    importorskip("scipy")
    from dem_pipeline.composite import feather_overlay

    base = np.full((10, 30), -10.0, dtype="f4")       # old CUDEM channel
    base_ids = np.full((10, 30), 13, dtype="u1")
    overlay = np.full((10, 30), np.nan, dtype="f4")   # new eHydro survey
    overlay[:, 10:20] = -14.0                          # dredged deeper
    out, ids = feather_overlay(base, base_ids, overlay, 11, feather_px=4)

    assert np.allclose(out[:, :10], -10.0)             # outside footprint
    assert np.allclose(out[:, 14:16], -14.0)           # survey interior
    edge = out[5, 10:14]
    assert np.all(np.diff(edge) < 0)                   # monotonic ramp down
    assert -14.0 < edge[0] < -10.0                     # no step at the edge
    assert np.all(ids[:, :10] == 13) and np.all(ids[5, 14:16] == 11)


def test_feather_overlay_full_where_base_missing():
    importorskip("scipy")
    from dem_pipeline.composite import feather_overlay

    base = np.full((8, 8), np.nan, dtype="f4")
    base_ids = np.zeros((8, 8), dtype="u1")
    overlay = np.full((8, 8), -3.0, dtype="f4")
    overlay[0, 0] = np.nan
    out, ids = feather_overlay(base, base_ids, overlay, 12, feather_px=4)
    # no base to blend against: overlay applies at full strength immediately
    assert np.allclose(out[1:, 1:], -3.0)
    assert np.isnan(out[0, 0]) and ids[0, 0] == 0


def test_feathered_first_valid_priority_order():
    importorskip("scipy")
    from dem_pipeline.composite import feathered_first_valid

    shape = (6, 40)
    newest = np.full(shape, np.nan, dtype="f4")
    newest[:, 15:25] = -12.0
    middle = np.full(shape, np.nan, dtype="f4")
    middle[:, 10:30] = -9.0
    oldest = np.full(shape, -6.0, dtype="f4")
    out, ids = feathered_first_valid(
        {"ehydro": newest, "eaarlb": middle, "cudem19": oldest},
        {"ehydro": 11, "eaarlb": 12, "cudem19": 13}, feather_px=2)

    assert np.allclose(out[:, :8], -6.0) and np.all(ids[:, :8] == 13)
    assert np.allclose(out[:, 19:21], -12.0) and np.all(ids[:, 19:21] == 11)
    assert np.allclose(out[:, 27:28], -9.0) and np.all(ids[:, 27:28] == 12)
    assert np.all(np.isfinite(out))          # priorities fill all gaps


def test_warp_to_tile_identity_and_nodata(tmp_path):
    rasterio = importorskip("rasterio")
    from rasterio.transform import Affine

    data = np.arange(90 * 90, dtype="f4").reshape(90, 90)
    data[0, 0] = NODATA
    src = make_geotiff(tmp_path / "src.tif", data,
                       Affine.from_gdal(*GT), "EPSG:4269", nodata=NODATA)
    out = warp_to_tile(src, GT, 90, 90, "EPSG:4269", "near", NODATA)
    assert np.isnan(out[0, 0])
    assert np.allclose(out[1:, 1:], data[1:, 1:])
    del rasterio


def _test_cfg(tmp_path):
    return {
        "project": {"name": "test19", "version": "2026v1"},
        "target_grid": {"nodata": NODATA, "epsg": 4269},
        "compositing": {
            "feather_px": 3,
            "survey_feather_px": 2,
            "fill_max_px": 10,
            "source_ids": {"nodata": 0, "s1m": 1, "dem1m_projects": 2,
                           "dem13": 3, "ehydro": 11, "eaarlb": 12,
                           "cudem19": 13, "coned": 14, "filled": 99},
        },
        "sources": {
            "s1m": {"resampling": "near"},
            "dem1m_projects": {"resampling": "near"},
            "dem13": {"resampling": "near"},
            "ehydro": {"resampling": "near"},
            "eaarlb": {"resampling": "near"},
            "cudem19": {"resampling": "near"},
            "coned": {"resampling": "near", "enabled": False},
        },
    }


def test_compose_tile_end_to_end(tmp_path, monkeypatch):
    importorskip("scipy")
    rasterio = importorskip("rasterio")
    from rasterio.transform import Affine

    import dem_pipeline.composite as composite
    import dem_pipeline.grid as grid_mod
    from dem_pipeline.grid import GridSpec

    # sandboxed repo layout: tile_grid.json + per-source "VRTs" (GeoTIFFs
    # named .vrt -- GDAL identifies drivers by content, not extension)
    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "intermediate" / "vrt").mkdir(parents=True)
    GridSpec(
        epsg=4269, nodata=NODATA, collar_px=0,
        cudem_geotransforms={TILE: GT}, cudem_sizes={TILE: [90, 90]},
    ).dump(tmp_path / "config" / "tile_grid.json")

    def fake_repo_path(*parts):
        return tmp_path.joinpath(*parts)

    monkeypatch.setattr(grid_mod, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "load_config", lambda: _test_cfg(tmp_path))

    transform = Affine.from_gdal(*GT)
    # land: 10 m everywhere except a nodata "river" strip (cols 40-49)
    land = np.full((90, 90), 10.0, dtype="f4")
    land[:, 40:50] = NODATA
    make_geotiff(tmp_path / "data/intermediate/vrt/s1m.vrt", land,
                 transform, "EPSG:4269", nodata=NODATA)
    # cudem: -5 m bathymetry in the river strip (plus land values elsewhere,
    # which must NOT win over s1m on land)
    cudem = np.full((90, 90), 9.0, dtype="f4")
    cudem[:, 40:50] = -5.0
    make_geotiff(tmp_path / "data/intermediate/vrt/cudem19.vrt", cudem,
                 transform, "EPSG:4269", nodata=NODATA)

    out_tif = composite.compose_tile(TILE, out_dir=tmp_path / "out")

    with rasterio.open(out_tif) as src:
        assert src.transform.to_gdal() == tuple(GT)
        assert src.nodata == NODATA
        z = src.read(1)
    with rasterio.open(tmp_path / "out" / f"{TILE}_source.tif") as src:
        ids = src.read(1)

    assert np.all(z != NODATA)                 # no holes in the product
    assert np.allclose(z[:, :35], 10.0)        # deep land: s1m wins
    assert np.allclose(z[:, 44:46], -5.0)      # river core: cudem bathymetry
    assert np.all(ids[:, :35] == 1)            # provenance: s1m
    assert np.all(ids[:, 44:46] == 13)         # provenance: cudem19
    # feather band exists: values strictly between bathy and land nearby
    band = z[:, 38:40]
    assert np.all(band <= 10.0) and np.any(band < 10.0)

    # idempotency: second call is a no-op returning the same product
    again = composite.compose_tile(TILE, out_dir=tmp_path / "out")
    assert again == out_tif


def test_compose_tile_masks_implausible_land_values(tmp_path, monkeypatch):
    """Some real S1M distribution tiles carry a minority of corrupt edge
    pixels that are literal fractional blends of the nodata sentinel with
    real data (e.g. -529999.5 in a tile whose nodata is -999999.0 is
    -999999.0 * 0.53) -- not flagged as nodata, so first_valid would
    otherwise happily pick s1m's garbage over a good lower-priority source.
    """
    importorskip("scipy")
    rasterio = importorskip("rasterio")
    from rasterio.transform import Affine

    import dem_pipeline.composite as composite
    import dem_pipeline.grid as grid_mod
    from dem_pipeline.grid import GridSpec

    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "intermediate" / "vrt").mkdir(parents=True)
    GridSpec(
        epsg=4269, nodata=NODATA, collar_px=0,
        cudem_geotransforms={TILE: GT}, cudem_sizes={TILE: [90, 90]},
    ).dump(tmp_path / "config" / "tile_grid.json")

    def fake_repo_path(*parts):
        return tmp_path.joinpath(*parts)

    monkeypatch.setattr(grid_mod, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "load_config", lambda: _test_cfg(tmp_path))

    transform = Affine.from_gdal(*GT)
    # s1m: normal 10 m land, but a corrupted sentinel-blend value at (5, 5)
    s1m = np.full((90, 90), 10.0, dtype="f4")
    s1m[5, 5] = -529999.5
    make_geotiff(tmp_path / "data/intermediate/vrt/s1m.vrt", s1m,
                transform, "EPSG:4269", nodata=NODATA)
    # lower-priority land source: good data everywhere, including (5, 5)
    dem1m = np.full((90, 90), 8.0, dtype="f4")
    make_geotiff(tmp_path / "data/intermediate/vrt/dem1m_projects.vrt", dem1m,
                transform, "EPSG:4269", nodata=NODATA)

    out_tif = composite.compose_tile(TILE, out_dir=tmp_path / "out")
    with rasterio.open(out_tif) as src:
        z = src.read(1)
    with rasterio.open(tmp_path / "out" / f"{TILE}_source.tif") as src:
        ids = src.read(1)

    assert z[5, 5] == 8.0                # fell through to dem1m_projects
    assert ids[5, 5] == 2                # provenance: dem1m_projects
    assert np.all(z[z != 8.0] == 10.0)   # everywhere else, s1m's 10 m stands


def test_compose_tile_land_void_fill_yields_to_bathymetry(tmp_path, monkeypatch):
    """Some land products (confirmed for both s1m and dem13, 2026-09-09) fill
    known-open-water areas with a flat 0 m plate instead of proper nodata.
    Past NHDPlus HR's coverage (no `water` mask layer at all here, standing
    in for the open Atlantic) that plate must not be trusted as land just
    because it isn't literally nodata -- real bathymetry sitting right there
    in the water stack must win. This is also the case the old
    `"s1m" in land_arrays` guard missed entirely: a tile with zero s1m
    coverage (only dem13) never even evaluated the correction."""
    importorskip("scipy")
    rasterio = importorskip("rasterio")
    from rasterio.transform import Affine

    import dem_pipeline.composite as composite
    import dem_pipeline.grid as grid_mod
    from dem_pipeline.grid import GridSpec

    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "intermediate" / "vrt").mkdir(parents=True)
    GridSpec(
        epsg=4269, nodata=NODATA, collar_px=0,
        cudem_geotransforms={TILE: GT}, cudem_sizes={TILE: [90, 90]},
    ).dump(tmp_path / "config" / "tile_grid.json")

    def fake_repo_path(*parts):
        return tmp_path.joinpath(*parts)

    monkeypatch.setattr(grid_mod, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "load_config", lambda: _test_cfg(tmp_path))

    # compose_tile pads its working window by fill_max_px (10, per _test_cfg)
    # beyond the tile's own edges; source rasters must extend past the tile
    # edges by at least that much or the padding zone reads as nodata purely
    # because the *test fixture* stops there (see
    # test_compose_tile_vector_water_mask_and_voids).
    pad = 10
    lon0, px = GT[0], GT[1]
    pad_transform = Affine.from_gdal(lon0 - pad * px, px, 0.0,
                                     40.0 + pad * px, 0.0, -px)
    n = 90 + 2 * pad
    # dem13 only (no s1m anywhere in this tile) -- a flat 0 m void-fill
    # plate, exactly like a fully-offshore tile.
    make_geotiff(tmp_path / "data/intermediate/vrt/dem13.vrt",
                np.full((n, n), 0.0, dtype="f4"), pad_transform,
                "EPSG:4269", nodata=NODATA)
    # real bathymetry everywhere; no water_mask.gpkg is written at all, so
    # the NHD-derived "water" layer has zero coverage here (standing in for
    # the open Atlantic, past NHDPlus HR's inland/estuarine extent).
    make_geotiff(tmp_path / "data/intermediate/vrt/cudem19.vrt",
                np.full((n, n), -10.0, dtype="f4"), pad_transform,
                "EPSG:4269", nodata=NODATA)

    out_tif = composite.compose_tile(TILE, out_dir=tmp_path / "out")
    with rasterio.open(out_tif) as src:
        z = src.read(1)
    with rasterio.open(tmp_path / "out" / f"{TILE}_source.tif") as src:
        ids = src.read(1)

    assert np.allclose(z, -10.0)   # bathymetry wins, not the 0 m void plate
    assert np.all(ids == 13)       # provenance: cudem19, not dem13


def test_compose_tile_real_zero_elevation_land_not_overridden(tmp_path, monkeypatch):
    """A real, non-void-fill land elevation that happens to be non-zero must
    keep winning over co-located bathymetry -- the void-fill correction is
    keyed on the exact 0.0 sentinel, not "any land value where water data
    also exists." (Real lidar/DEM values essentially never land on a
    bit-exact 0.0 by chance, so this is a safe, narrow signature.)"""
    importorskip("scipy")
    rasterio = importorskip("rasterio")
    from rasterio.transform import Affine

    import dem_pipeline.composite as composite
    import dem_pipeline.grid as grid_mod
    from dem_pipeline.grid import GridSpec

    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "intermediate" / "vrt").mkdir(parents=True)
    GridSpec(
        epsg=4269, nodata=NODATA, collar_px=0,
        cudem_geotransforms={TILE: GT}, cudem_sizes={TILE: [90, 90]},
    ).dump(tmp_path / "config" / "tile_grid.json")

    def fake_repo_path(*parts):
        return tmp_path.joinpath(*parts)

    monkeypatch.setattr(grid_mod, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "load_config", lambda: _test_cfg(tmp_path))

    pad = 10
    lon0, px = GT[0], GT[1]
    pad_transform = Affine.from_gdal(lon0 - pad * px, px, 0.0,
                                     40.0 + pad * px, 0.0, -px)
    n = 90 + 2 * pad
    # real, low-lying coastal marsh: a genuine non-zero elevation
    make_geotiff(tmp_path / "data/intermediate/vrt/dem13.vrt",
                np.full((n, n), 0.37, dtype="f4"), pad_transform,
                "EPSG:4269", nodata=NODATA)
    make_geotiff(tmp_path / "data/intermediate/vrt/cudem19.vrt",
                np.full((n, n), -10.0, dtype="f4"), pad_transform,
                "EPSG:4269", nodata=NODATA)

    out_tif = composite.compose_tile(TILE, out_dir=tmp_path / "out")
    with rasterio.open(out_tif) as src:
        z = src.read(1)
    with rasterio.open(tmp_path / "out" / f"{TILE}_source.tif") as src:
        ids = src.read(1)

    assert np.allclose(z, 0.37)    # real land elevation stands
    assert np.all(ids == 3)        # provenance: dem13


def test_compose_tile_recomposes_when_grid_spec_changed(tmp_path, monkeypatch):
    """A product left over from a run under an older GridSpec (e.g. before
    stage 02's audit was rerun with different nominal shift constants) must
    be regenerated, not trusted just because the file exists -- this is the
    exact bug that let stale tiles silently fail grid_conformity in
    production: skip-if-exists alone can't tell a stale file from a fresh
    one."""
    importorskip("scipy")
    rasterio = importorskip("rasterio")
    from rasterio.transform import Affine

    import dem_pipeline.composite as composite
    import dem_pipeline.grid as grid_mod
    from dem_pipeline.grid import GridSpec

    # a tile NOT in cudem_geotransforms, so its geotransform comes from the
    # nominal-lattice formula and moves when collar/shift constants change
    tile = "n40x25_w075x25"
    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "intermediate" / "vrt").mkdir(parents=True)
    spec = GridSpec(epsg=4269, nodata=NODATA, collar_px=2,
                    x_shift_px=0.0, y_shift_px=0.0)
    spec.dump(tmp_path / "config" / "tile_grid.json")
    gt, width, height = spec.tile_geotransform(tile)

    def fake_repo_path(*parts):
        return tmp_path.joinpath(*parts)

    monkeypatch.setattr(grid_mod, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "load_config", lambda: _test_cfg(tmp_path))

    land = np.full((height, width), 10.0, dtype="f4")
    make_geotiff(tmp_path / "data/intermediate/vrt/s1m.vrt", land,
                Affine.from_gdal(*gt), "EPSG:4269", nodata=NODATA)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # plant a "stale" product+source pair with a shifted origin, as if
    # composed under a GridSpec with different shift constants
    stale_gt = list(gt)
    stale_gt[0] += 5 * spec.pixel_deg
    make_geotiff(out_dir / f"{tile}.tif", land,
                Affine.from_gdal(*stale_gt), "EPSG:4269", nodata=NODATA)
    make_geotiff(out_dir / f"{tile}_source.tif",
                np.ones((height, width), dtype="u1"),
                Affine.from_gdal(*stale_gt), "EPSG:4269")

    out_tif = composite.compose_tile(tile, out_dir=out_dir)
    with rasterio.open(out_tif) as src:
        assert src.transform.to_gdal() == gt
        assert (src.width, src.height) == (width, height)


def test_compose_tile_feathers_consistently_across_tile_boundary(tmp_path, monkeypatch):
    """feather_alpha's EDT must see past a tile's own edge: a water mask
    that starts immediately south of tile N's bottom row (i.e. entirely
    within the neighboring tile S) still has to feather tile N's last row,
    because feather_px worth of real water sits just beyond it. Without the
    padded working window (dem_pipeline.composite.compose_tile), tile N's
    computation is blind to that water and wrongly renders pure land there.
    """
    importorskip("scipy")
    rasterio = importorskip("rasterio")
    gpd = importorskip("geopandas")
    from rasterio.transform import Affine
    from shapely.geometry import box

    import dem_pipeline.composite as composite
    import dem_pipeline.grid as grid_mod
    from dem_pipeline.composite import feather_alpha
    from dem_pipeline.grid import GridSpec

    tile_n, tile_s = "n40x00_w075x25", "n39x75_w075x25"
    gt_n = GT                                      # north edge 40.00
    gt_s = [GT[0], PX, 0.0, 39.75, 0.0, -PX]       # north edge 39.75 (= tile_n's south edge)

    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "intermediate" / "vrt").mkdir(parents=True)
    (tmp_path / "data" / "intermediate" / "water_mask").mkdir(parents=True)
    GridSpec(epsg=4269, nodata=NODATA, collar_px=0,
            cudem_geotransforms={tile_n: gt_n, tile_s: gt_s},
            cudem_sizes={tile_n: [90, 90], tile_s: [90, 90]}).dump(
        tmp_path / "config" / "tile_grid.json")

    monkeypatch.setattr(grid_mod, "repo_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(composite, "repo_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(composite, "load_config", lambda: _test_cfg(tmp_path))

    pad = 10  # matches _test_cfg's fill_max_px, the largest of the 3 radii
    lon0 = GT[0]
    combined_transform = Affine.from_gdal(lon0 - pad * PX, PX, 0.0,
                                          40.0 + pad * PX, 0.0, -PX)
    combined_h = 180 + 2 * pad   # both 90-row tiles, stacked, plus padding
    combined_w = 90 + 2 * pad
    make_geotiff(tmp_path / "data/intermediate/vrt/s1m.vrt",
                np.full((combined_h, combined_w), 10.0, dtype="f4"),
                combined_transform, "EPSG:4269", nodata=NODATA)
    make_geotiff(tmp_path / "data/intermediate/vrt/cudem19.vrt",
                np.full((combined_h, combined_w), -5.0, dtype="f4"),
                combined_transform, "EPSG:4269", nodata=NODATA)

    # water covers all of tile_s (and beyond, south), starting exactly at
    # the tile_n/tile_s boundary -- none of it is visible within tile_n's
    # own 90x90 window, only within its padded margin.
    water_poly = box(lon0 - pad * PX, 39.75 - (90 + pad) * PX,
                     lon0 + (90 + pad) * PX, 39.75)
    gpkg = tmp_path / "data/intermediate/water_mask/water_mask.gpkg"
    gpd.GeoDataFrame(geometry=[water_poly], crs="EPSG:4269").to_file(
        gpkg, layer="water", driver="GPKG")

    out_n = composite.compose_tile(tile_n, out_dir=tmp_path / "out")
    out_s = composite.compose_tile(tile_s, out_dir=tmp_path / "out")
    with rasterio.open(out_n) as src:
        z_n = src.read(1)
    with rasterio.open(out_s) as src:
        z_s = src.read(1)

    # ground truth: feather_alpha computed with full, unbounded context
    ref_mask = np.zeros((combined_h, combined_w), dtype=bool)
    ref_mask[pad + 90:, :] = True   # water starts at the tile_n/tile_s row
    feather_px = _test_cfg(tmp_path)["compositing"]["feather_px"]
    ref_alpha = feather_alpha(ref_mask, feather_px)
    expected_last_row_of_n = (ref_alpha[pad + 89, pad:pad + 90] * -5.0
                              + (1 - ref_alpha[pad + 89, pad:pad + 90]) * 10.0)

    assert not np.allclose(z_n[-1, :], 10.0), (
        "tile_n's last row must be feathered by water it can only see "
        "through the padded working window, not rendered as pure land")
    assert np.allclose(z_n[-1, :], expected_last_row_of_n, atol=1e-3)
    assert np.allclose(z_s[10, :], -5.0)  # well past the feather ramp: pure water


def test_compose_tile_vector_water_mask_and_voids(tmp_path, monkeypatch):
    """The gpkg water-mask branch: an NHD-style polygon forces the water
    stack where BOTH stacks are valid, and a void polygon punches back out.
    (The dynamic S1M-nodata term is inactive here: land is valid everywhere.)
    """
    importorskip("scipy")
    rasterio = importorskip("rasterio")
    gpd = importorskip("geopandas")
    from rasterio.transform import Affine
    from shapely.geometry import box

    import dem_pipeline.composite as composite
    import dem_pipeline.grid as grid_mod
    from dem_pipeline.grid import GridSpec

    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "intermediate" / "vrt").mkdir(parents=True)
    (tmp_path / "data" / "intermediate" / "water_mask").mkdir(parents=True)
    GridSpec(epsg=4269, nodata=NODATA, collar_px=0,
             cudem_geotransforms={TILE: GT},
             cudem_sizes={TILE: [90, 90]}).dump(
        tmp_path / "config" / "tile_grid.json")

    monkeypatch.setattr(grid_mod, "repo_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(composite, "repo_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(composite, "load_config", lambda: _test_cfg(tmp_path))

    # compose_tile pads its working window by up to fill_max_px (10, per
    # _test_cfg) beyond the tile's own edges so feathering/fill has real
    # cross-boundary context (see test_compose_tile_seams_match_across_tiles
    # for why). Source rasters and the water/void polygons must extend past
    # the tile edges by at least that much, or the padding zone reads as
    # nodata/non-water purely because the *test fixture* stops there -- a
    # boundary that would never occur against a real, basin-wide source VRT.
    pad = 10
    lon0, px = GT[0], GT[1]
    pad_transform = Affine.from_gdal(lon0 - pad * px, px, 0.0,
                                     40.0 + pad * px, 0.0, -px)
    n = 90 + 2 * pad
    make_geotiff(tmp_path / "data/intermediate/vrt/s1m.vrt",
                 np.full((n, n), 10.0, dtype="f4"), pad_transform,
                 "EPSG:4269", nodata=NODATA)
    make_geotiff(tmp_path / "data/intermediate/vrt/cudem19.vrt",
                 np.full((n, n), -5.0, dtype="f4"), pad_transform,
                 "EPSG:4269", nodata=NODATA)

    # water polygon over pixel cols 60-74 (extended past N/S tile edges),
    # void punching out cols 70-74
    water_poly = box(lon0 + 60 * px, 39.75 - pad * px, lon0 + 75 * px, 40.0 + pad * px)
    void_poly = box(lon0 + 70 * px, 39.75 - pad * px, lon0 + 75 * px, 40.0 + pad * px)
    gpkg = tmp_path / "data/intermediate/water_mask/water_mask.gpkg"
    gpd.GeoDataFrame(geometry=[water_poly], crs="EPSG:4269").to_file(
        gpkg, layer="water", driver="GPKG")
    gpd.GeoDataFrame(geometry=[void_poly], crs="EPSG:4269").to_file(
        gpkg, layer="eaarlb_voids", driver="GPKG")

    out_tif = composite.compose_tile(TILE, out_dir=tmp_path / "out")
    with rasterio.open(out_tif) as src:
        z = src.read(1)
    with rasterio.open(tmp_path / "out" / f"{TILE}_source.tif") as src:
        ids = src.read(1)

    assert np.allclose(z[:, :55], 10.0)          # land far from the polygon
    assert np.allclose(z[:, 64:66], -5.0)        # inside the water polygon
    assert np.all(ids[:, 64:66] == 13)
    assert np.allclose(z[:, 74], 10.0)           # void interior: land again
    assert np.all(ids[:, 74] == 1)


def test_water_stack_order_ehydro_then_nos_grids_then_cudem(tmp_path, monkeypatch):
    """nos_grids (2015-18 NOAA multibeam of the Hudson) must outrank the 2014
    CUDEM compilation but yield to newer eHydro channel surveys. Getting the
    order wrong silently keeps the stale grid wherever they overlap. (Every
    other compose test's config omits nos_grids entirely, which also pins
    that an unconfigured source is skipped rather than raising KeyError.)"""
    importorskip("scipy")
    rasterio = importorskip("rasterio")
    from rasterio.transform import Affine

    import dem_pipeline.composite as composite
    import dem_pipeline.grid as grid_mod
    from dem_pipeline.grid import GridSpec

    (tmp_path / "config").mkdir()
    vdir = tmp_path / "data" / "intermediate" / "vrt"
    vdir.mkdir(parents=True)
    GridSpec(epsg=4269, nodata=NODATA, collar_px=0,
             cudem_geotransforms={TILE: GT}, cudem_sizes={TILE: [90, 90]},
             ).dump(tmp_path / "config" / "tile_grid.json")

    def fake_repo_path(*parts):
        return tmp_path.joinpath(*parts)

    cfg = _test_cfg(tmp_path)
    cfg["sources"]["nos_grids"] = {"resampling": "near"}
    cfg["compositing"]["source_ids"]["nos_grids"] = 15
    monkeypatch.setattr(grid_mod, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "load_config", lambda: cfg)

    # rasters extend fill_max_px (10) past the tile so the compositor's
    # padded window does not see a fixture edge as a footprint edge
    pad, n = 10, 110
    t = Affine.from_gdal(GT[0] - pad * PX, PX, 0.0, 40.0 + pad * PX, 0.0, -PX)
    c = lambda a, b: slice(a + pad, b + pad)      # tile cols -> raster cols
    land = np.full((n, n), 10.0, "f4")
    land[:, c(20, 70)] = NODATA                   # wide river
    cudem = np.full((n, n), NODATA, "f4")
    cudem[:, c(20, 70)] = -5.0                    # whole river
    nos = np.full((n, n), NODATA, "f4")
    nos[:, c(32, 58)] = -8.0                      # most of the river
    eh = np.full((n, n), NODATA, "f4")
    eh[:, c(42, 48)] = -12.0                      # the channel only
    for nm, a in (("s1m", land), ("cudem19", cudem), ("nos_grids", nos),
                  ("ehydro", eh)):
        make_geotiff(vdir / f"{nm}.vrt", a, t, "EPSG:4269", nodata=NODATA)

    composite.compose_tile(TILE, out_dir=tmp_path / "out")
    with rasterio.open(tmp_path / "out" / f"{TILE}.tif") as src:
        z = src.read(1)
    with rasterio.open(tmp_path / "out" / f"{TILE}_source.tif") as src:
        ids = src.read(1)

    assert np.allclose(z[:, 44:46], -12.0) and np.all(ids[:, 44:46] == 11)
    assert np.allclose(z[:, 36:39], -8.0) and np.all(ids[:, 36:39] == 15)
    assert np.allclose(z[:, 25:29], -5.0) and np.all(ids[:, 25:29] == 13)


def test_compose_tile_no_sources_raises(tmp_path, monkeypatch):
    from unittest import SkipTest
    try:
        import scipy  # noqa: F401
    except ImportError:
        raise SkipTest("requires scipy")

    import dem_pipeline.composite as composite
    import dem_pipeline.grid as grid_mod
    from dem_pipeline.grid import GridSpec
    from tests.util import raises

    (tmp_path / "config").mkdir()
    GridSpec(cudem_geotransforms={TILE: GT},
             cudem_sizes={TILE: [90, 90]}).dump(
        tmp_path / "config" / "tile_grid.json")

    monkeypatch.setattr(grid_mod, "repo_path",
                        lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(composite, "repo_path",
                        lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(composite, "load_config", lambda: _test_cfg(tmp_path))

    with raises(RuntimeError):
        composite.compose_tile(TILE, out_dir=tmp_path / "out")


def test_output_crs_comes_from_config_not_the_audited_spec(tmp_path, monkeypatch):
    """The product CRS must not depend on which CUDEM tiles a run audited.

    Real CUDEM is not internally consistent -- 2014/2015/2018 tiles are
    EPSG:5498 and 2019 tiles are EPSG:4269 -- so `GridSpec.epsg` is whatever
    vintage happened to dominate the audited set, and a clip_bbox change flips
    it. Reading the output CRS from the spec would silently retag the product
    between runs; it has to come from config/target_grid.
    """
    importorskip("scipy")
    rasterio = importorskip("rasterio")
    from rasterio.transform import Affine

    import dem_pipeline.composite as composite
    import dem_pipeline.grid as grid_mod
    from dem_pipeline.grid import GridSpec

    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "intermediate" / "vrt").mkdir(parents=True)
    # spec says 5498 (a 2014v1-dominated audit); config says 4269
    GridSpec(
        epsg=5498, nodata=NODATA, collar_px=0,
        cudem_geotransforms={TILE: GT}, cudem_sizes={TILE: [90, 90]},
    ).dump(tmp_path / "config" / "tile_grid.json")

    def fake_repo_path(*parts):
        return tmp_path.joinpath(*parts)

    monkeypatch.setattr(grid_mod, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "repo_path", fake_repo_path)
    monkeypatch.setattr(composite, "load_config", lambda: _test_cfg(tmp_path))

    make_geotiff(tmp_path / "data/intermediate/vrt/s1m.vrt",
                 np.full((90, 90), 10.0, dtype="f4"),
                 Affine.from_gdal(*GT), "EPSG:4269", nodata=NODATA)

    out_tif = composite.compose_tile(TILE, out_dir=tmp_path / "out")
    with rasterio.open(out_tif) as src:
        assert src.crs.to_epsg() == 4269, (
            f"output tagged EPSG:{src.crs.to_epsg()} -- inherited from the "
            "audited spec (5498) instead of config")
        # the geotransform still comes from the spec, verbatim
        assert src.transform.to_gdal() == tuple(GT)
