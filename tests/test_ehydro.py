"""Tests for the eHydro survey processing stage (scripts/06_process_ehydro.py).

The numbered script is loaded via importlib. Covers: the production
units/sign conversion (convert_raw_depths) including override combinations,
the plausibility gate (must catch missed unit conversions and sign flips),
XYZ parsing with and without CRS overrides, the MLLW->NAVD88 application,
and TIN gridding pinned to absolute world coordinates so mirrored/flipped
rasters cannot pass.
"""

import importlib.util
from pathlib import Path

import numpy as np

from tests.util import importorskip, make_geotiff, raises

_SPEC = importlib.util.spec_from_file_location(
    "process_ehydro",
    Path(__file__).resolve().parents[1] / "scripts" / "06_process_ehydro.py",
)
eh = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eh)


def test_convert_raw_depths_feet_positive_down():
    # CENAP default: 45-ft channel sounding -> -13.716 m relative to MLLW
    z = eh.convert_raw_depths([45.0, 10.0], "feet", True)
    assert np.allclose(z, [-13.716, -3.048])


def test_convert_raw_depths_override_combinations():
    raw = [7.5, -1.0]
    assert np.allclose(eh.convert_raw_depths(raw, "meters", True), [-7.5, 1.0])
    assert np.allclose(eh.convert_raw_depths(raw, "meters", False), [7.5, -1.0])
    assert np.allclose(eh.convert_raw_depths(raw, "feet", False),
                       [7.5 * 0.3048, -0.3048])


def test_convert_raw_depths_rejects_unknown_units():
    with raises(ValueError):
        eh.convert_raw_depths([1.0], "fathoms", True)


def test_plausibility_gate_catches_missed_conversion():
    # a correctly converted 20-ft-deep survey passes ...
    ok = eh.convert_raw_depths([20.0, 5.0], "feet", True) + (-0.7)
    eh.check_plausible(ok, "good_survey")
    # ... but the same survey with the feet conversion dropped (a silent
    # 3.28x depth error) or the sign flipped must abort loudly
    with raises(eh.ImplausibleDepthsError):
        eh.check_plausible(np.array([-45.0, -20.0]), "unconverted_feet")
    with raises(eh.ImplausibleDepthsError):
        eh.check_plausible(np.array([13.7, 6.1]), "sign_flipped")


def test_read_survey_points_xyz(tmp_path):
    sdir = tmp_path / "survey"
    sdir.mkdir()
    (sdir / "PS_TEST.XYZ").write_text(
        "500000.0, 4400000.0, 12.5\n"
        "500010.0  4400010.0  13.0\n"      # whitespace-delimited also valid
        "junk line\n"
        "500020.0, 4400020.0, 11.0, extra\n"
    )
    (sdir / "PS_TEST.prj").write_text('EPSG:26918')
    xyz, crs, outline = eh.read_survey_points(sdir)
    assert xyz.shape == (3, 3)
    assert xyz[1, 2] == 13.0
    assert crs == "EPSG:26918"
    assert outline is None


def test_read_survey_points_crs_override_reaches_xyz_without_prj(tmp_path):
    sdir = tmp_path / "survey"
    sdir.mkdir()
    (sdir / "PS_TEST.XYZ").write_text("1, 2, 3\n")
    # without an override this survey is unusable ...
    with raises(ValueError):
        eh.read_survey_points(sdir)
    # ... but the manifests/ehydro_overrides.yaml crs path must work
    xyz, crs, _ = eh.read_survey_points(sdir, crs_override="EPSG:26918")
    assert crs == "EPSG:26918" and xyz.shape == (1, 3)


def test_read_survey_points_empty_dir_fails(tmp_path):
    sdir = tmp_path / "survey"
    sdir.mkdir()
    with raises(FileNotFoundError):
        eh.read_survey_points(sdir)


def test_mllw_to_navd88_applies_separation(tmp_path):
    importorskip("rasterio")
    from rasterio.transform import from_origin

    # constant separation field of -0.7 m over the survey area
    sep = make_geotiff(tmp_path / "sep.tif", np.full((20, 20), -0.7),
                       from_origin(-76.0, 40.5, 0.1, 0.1), "EPSG:4269")
    # points in UTM 18N around (-75.5E, 39.5N)
    xyz = np.array([[456000.0, 4372000.0, 0.0],
                    [456100.0, 4372100.0, -10.0]])
    out = eh.mllw_to_navd88(xyz, "EPSG:26918", sep)
    assert np.allclose(out[:, 2], [-0.7, -10.7])
    assert np.allclose(out[:, :2], xyz[:, :2])  # horizontal untouched


def test_grid_survey_pins_absolute_coordinates(tmp_path):
    """TIN gridding on an x+y-sloped plane, checked at known world points.

    A y-invariant plane with only a range assertion would let a mirrored or
    shifted raster pass; independent slopes + point sampling pin both axis
    orientations and the half-pixel registration.
    """
    importorskip("scipy")
    rasterio = importorskip("rasterio")

    x0, y0 = 500000.0, 4400000.0
    rng = np.random.default_rng(42)
    x = x0 + rng.uniform(0, 300, 600)
    y = y0 + rng.uniform(0, 300, 600)
    z = -8.0 + 0.01 * (x - x0) + 0.02 * (y - y0)
    xyz = np.column_stack([x, y, z])

    out = tmp_path / "survey.tif"
    eh.grid_survey(xyz, "EPSG:26918", None, out, res_m=3.0, nodata=-999999.0)

    with rasterio.open(out) as src:
        assert src.crs.to_epsg() == 26918
        assert src.res == (3.0, 3.0)
        arr = src.read(1, masked=True)
        # sample well inside the hull at three asymmetric world locations;
        # any axis mirror/swap/offset changes these values by >> tolerance
        probes = [(x0 + 60.0, y0 + 200.0), (x0 + 200.0, y0 + 60.0),
                  (x0 + 150.0, y0 + 150.0)]
        sampled = [v[0] for v in src.sample(probes)]
    expected = [-8.0 + 0.01 * 60 + 0.02 * 200,    # -3.4
                -8.0 + 0.01 * 200 + 0.02 * 60,    # -4.8
                -8.0 + 0.01 * 150 + 0.02 * 150]   # -3.5
    # linear TIN reproduces a plane exactly; tolerance covers the half-pixel
    # offset between probe point and cell center (<= 2.1 m * max slope)
    assert np.allclose(sampled, expected, atol=0.06), (sampled, expected)
    # hull clipping: padded raster corners stay nodata
    assert arr.mask[0, 0] and arr.mask[-1, -1]


def test_survey_footprint_is_a_lonlat_hull():
    """Footprints feed the water mask in EPSG:4269; an untransformed or
    lon/lat-swapped hull would place the channel thousands of km away and
    silently drop that survey's bathymetry from every tile."""
    importorskip("pyproj")
    importorskip("shapely")

    # a 1 km square of soundings in UTM 18N over the Philadelphia reach
    xyz = np.array([[485000.0, 4422000.0, -10.0],
                    [486000.0, 4422000.0, -11.0],
                    [486000.0, 4423000.0, -12.0],
                    [485000.0, 4423000.0, -9.0]])
    w, s, e, n = eh.survey_footprint(xyz, "EPSG:26918").bounds
    assert -75.20 < w < e < -75.14, (w, e)
    assert 39.93 < s < n < 39.97, (s, n)


def test_footprints_survive_a_resumed_run(tmp_path):
    """Stage 06 skips surveys it already gridded, so the combined footprint
    layer must be reassembled from the per-survey sidecars on disk. Building it
    from only the surveys processed this run truncates the layer, and stage 07
    then hands the compositor a water mask missing those channels."""
    importorskip("shapely")
    from shapely.geometry import Polygon

    grid_dir = tmp_path / "ehydro_grids"
    grid_dir.mkdir()
    first = Polygon([(-75.2, 39.9), (-75.1, 39.9), (-75.1, 40.0)])
    second = Polygon([(-75.4, 39.7), (-75.3, 39.7), (-75.3, 39.8)])

    # an earlier run gridded survey A, then was interrupted ...
    eh.write_footprint(grid_dir / f"PS_A_20240101_CS{eh.FOOTPRINT_SUFFIX}",
                       "PS_A_20240101_CS", "20240101", first)
    # ... and the resumed run processes only survey B
    eh.write_footprint(grid_dir / f"PS_B_20250601_CS{eh.FOOTPRINT_SUFFIX}",
                       "PS_B_20250601_CS", "20250601", second)

    got = eh.collect_footprints(grid_dir)
    assert [f["survey"] for f in got] == ["PS_A_20240101_CS",
                                          "PS_B_20250601_CS"]
    assert [f["date"] for f in got] == ["20240101", "20250601"]
    assert got[0]["geometry"].equals(first)
    assert got[1]["geometry"].equals(second)
    # the sidecar write is atomic; no scratch file is left to be collected
    assert not list(grid_dir.glob("*.tmp"))


def test_outline_layer_prefers_the_real_survey_extent():
    """CENAP GDBs name the surveyed extent SurveyJob.

    The original match looked for "outline"/"coverage" -- names that appear in
    none of the 146 Delaware survey GDBs -- so every survey silently fell back
    to a convex hull of its soundings, spanning river bends and extrapolating
    dredged depths across water the survey never visited (7.3% of the test
    tile, vs 2.6% once the real extent is used).
    """
    real = ['Bathymetry_Vector', 'SurveyJob', 'ElevationContour',
            'Channel_Availability', 'ShoalPolygon', 'SurveyPoint', 'version']
    assert eh.outline_layer(real) == 'SurveyJob'
    # preference order, not merely "any match"
    assert eh.outline_layer(['Bathymetry_Vector', 'SurveyPoint']) == 'Bathymetry_Vector'
    assert eh.outline_layer(['Survey_Outline', 'SurveyPoint']) == 'Survey_Outline'
    # no extent layer -> None, so the caller warns and falls back explicitly
    assert eh.outline_layer(['SurveyPoint', 'version']) is None


def test_survey_footprint_uses_the_outline_and_drops_z(tmp_path):
    """SurveyJob polygons carry Z; shapely.ops.transform then calls the mapper
    with (x, y, z). That raised for every 3D survey the first time this branch
    executed -- 44 of 146 failed -- because the branch was dead while outline
    was always None."""
    importorskip("pyproj")
    importorskip("shapely")
    from shapely.geometry import Polygon

    xyz = np.array([[485000.0, 4422000.0, -10.0], [486000.0, 4423000.0, -12.0]])
    # a 3D outline covering a *different* area than the points' hull
    outline = Polygon([(484000, 4421000, 1.0), (484500, 4421000, 1.0),
                       (484500, 4421500, 1.0), (484000, 4421500, 1.0)])
    assert outline.has_z

    got = eh.survey_footprint(xyz, "EPSG:26918", outline)
    assert not got.has_z, "footprint must be planar"
    # it followed the outline, not the sounding hull
    w, s, e, n = got.bounds
    assert -75.20 < w < e < -75.18, (w, e)
    assert 39.93 < s < n < 39.95, (s, n)
    # and it is south-west of the soundings' own hull, so the two cannot be
    # confused by a coincidental bounds check
    assert n < eh.survey_footprint(xyz, "EPSG:26918", None).bounds[1]

    # without an outline the sounding hull is still the fallback
    fallback = eh.survey_footprint(xyz, "EPSG:26918", None)
    assert fallback.bounds != got.bounds
