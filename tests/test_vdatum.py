"""Tests for the MLLW->NAVD88 separation builder (dem_pipeline.vdatum).

The separation surface is sampled from the VDatum transformation service, not
composed from raw tss/mllw .gtx grids -- measured at 11 CO-OPS gauges, hand
composition was off by up to 0.18 m (`mllw` alone) and 0.70 m (`tss + mllw`)
with a systematic along-estuary trend, while the service reproduces the gauges
to 0.026 m median.

The synthetic field below VARIES IN BOTH AXES and the benchmark is pinned to
the station's absolute coordinates, so a rowcol/axis/lon-normalisation bug
cannot pass: it would sample the wrong place and miss the benchmark.
"""

import numpy as np

from tests.util import importorskip, raises

STATION_LON, STATION_LAT = -75.30, 39.40
BOUNDS = [-75.60, 39.20, -75.00, 39.80]


def _sep_at(lon, lat):
    """Synthetic separation, sloping in both axes (negative, as in the real
    estuary). Linear so that grid interpolation reproduces it exactly."""
    return -0.70 + 0.30 * (lat - 39.50) + 0.20 * (lon + 75.30)


TRUTH = _sep_at(STATION_LON, STATION_LAT)


def _patch(monkeypatch, tmp_path, benchmark_value, tol=0.10):
    importorskip("rasterio")
    importorskip("scipy")
    import dem_pipeline.vdatum as vd

    monkeypatch.setattr(vd, "repo_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(vd, "sample_separation",
                        lambda lon, lat, cfg: (_sep_at(lon, lat), 0.09))
    monkeypatch.setattr(vd, "station_mllw_in_navd88",
                        lambda s: (STATION_LON, STATION_LAT, benchmark_value))
    monkeypatch.setattr(vd, "load_config", lambda: {"vdatum": {
        "benchmark_stations": ["A"],
        "max_benchmark_residual_m": tol,
        "sample_bounds": BOUNDS,
        "sample_spacing_deg": 0.05,
        "sep_res_deg": 0.02,
        "sample_cache": "manifests/samples.csv",
        "service_workers": 2,
        "region": "chesapeak_delaware",
        "geoid": "geoid18",
        "source_h_frame": "IGS14",
    }})
    return vd


def test_separation_is_reproduced_at_absolute_coordinates(tmp_path, monkeypatch):
    import rasterio

    vd = _patch(monkeypatch, tmp_path, TRUTH)
    out = vd.build_separation_raster(tmp_path / "sep.tif")
    with rasterio.open(out) as src:
        arr = src.read(1)
        tags = src.tags()
        row, col = src.index(STATION_LON, STATION_LAT)

    assert float(tags["worst_benchmark_residual_m"]) < 0.01
    # the value at the station's own cell, not merely somewhere in range
    assert abs(arr[row, col] - TRUTH) < 0.01, (arr[row, col], TRUTH)
    # sign convention: z_navd88 = z_mllw + sep, and sep is negative here
    assert TRUTH < 0 and abs((0.0 + arr[row, col]) - TRUTH) < 0.01
    assert "VDatum service" in tags["source"]


def test_benchmark_disagreement_hard_fails(tmp_path, monkeypatch):
    """The gate must abort rather than emit a plausible-looking raster: a bad
    separation puts a systematic bias straight into the channel bathymetry."""
    vd = _patch(monkeypatch, tmp_path, TRUTH + 0.5)
    with raises(RuntimeError):
        vd.build_separation_raster(tmp_path / "sep.tif")


def test_wrong_sampling_location_cannot_pass(tmp_path, monkeypatch):
    """A benchmark consistent with a point 0.5 deg away must not fit -- this is
    what catches sampling the wrong cell (the 0-360 longitude bug sampled
    ~180,000 columns off and reported 'no benchmark fell inside')."""
    offset = _sep_at(STATION_LON + 0.5, STATION_LAT + 0.5)
    assert abs(offset - TRUTH) > 0.10
    vd = _patch(monkeypatch, tmp_path, offset)
    with raises(RuntimeError):
        vd.build_separation_raster(tmp_path / "sep.tif")


def test_existing_raster_short_circuits(tmp_path, monkeypatch):
    vd = _patch(monkeypatch, tmp_path, TRUTH)
    out = tmp_path / "sep.tif"
    out.write_bytes(b"placeholder")
    assert vd.build_separation_raster(out) == out
    assert out.read_bytes() == b"placeholder"


def test_novalue_sentinel_is_not_treated_as_data(monkeypatch):
    """VDatum answers 'no tidal model here' with t_z=-999999.0 and HTTP 200,
    not an error. float() accepts it, and 327 of 476 responses over this
    estuary were sentinels -- interpolating them gave separations near
    -850000 m, which the benchmark gate then reported as a datum failure."""
    import dem_pipeline.vdatum as vd

    cfg = {"service_url": "http://x", "region": "r", "source_h_frame": "IGS14",
           "target_h_frame": "NAD83_2011", "geoid": "geoid18"}

    class _R:
        def __init__(self, z):
            self._z = z

        def json(self):
            return {"t_z": self._z, "uncertainty": "0.09"}

    monkeypatch.setattr(vd.requests, "get",
                        lambda *a, **k: _R("-999999.0"))
    assert vd.sample_separation(-75.3, 39.4, cfg) is None

    monkeypatch.setattr(vd.requests, "get", lambda *a, **k: _R("-0.873"))
    got = vd.sample_separation(-75.3, 39.4, cfg)
    assert got is not None and abs(got[0] + 0.873) < 1e-9


def test_cache_is_reused_and_remembers_misses(tmp_path, monkeypatch):
    """Misses must be cached too. Land points never return a value, so
    re-querying them costs hundreds of requests on every rerun."""
    import dem_pipeline.vdatum as vd

    cfg = {"sample_bounds": [-75.10, 39.20, -75.00, 39.30],
           "sample_spacing_deg": 0.05, "service_workers": 2,
           "sample_cache": "unused"}
    calls = []

    def fake(lon, lat, _cfg):
        calls.append((lon, lat))
        return (_sep_at(lon, lat), 0.09) if lat > 39.24 else None

    monkeypatch.setattr(vd, "sample_separation", fake)
    cache = tmp_path / "samples.csv"

    first = vd.collect_samples(cfg, cache)
    assert len(calls) == 9                      # 3x3 grid, all queried
    misses = [k for k, v in first.items() if v[0] is None]
    assert misses, "land points should be recorded as misses"

    calls.clear()
    second = vd.collect_samples(cfg, cache)
    assert calls == [], "cached points must not be re-queried"
    # values survive the CSV round trip (stored to 0.1 mm, ample for a
    # separation surface); misses stay misses
    assert set(second) == set(first)
    for k, (sep, _) in first.items():
        got = second[k][0]
        assert (sep is None) == (got is None)
        if sep is not None:
            assert abs(got - sep) < 1e-4


def test_station_mllw_in_navd88_parses_coops_payloads(monkeypatch):
    """The datums endpoint has NO station coordinates (its top-level 'LAT'
    is Lowest Astronomical Tide!); lon/lat must come from the station
    metadata endpoint."""
    import dem_pipeline.vdatum as vd

    class FakeResponse:
        def __init__(self, payload):
            self._p = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/datums.json"):
            return FakeResponse({
                "LAT": -0.914,   # Lowest Astronomical Tide, a datum!
                "datums": [
                    {"name": "MHW", "value": 2.334},
                    {"name": "MLLW", "value": 0.826},
                    {"name": "NAVD88", "value": 1.548},
                ],
            })
        return FakeResponse({"stations": [
            {"id": "8557380", "lat": 38.7828, "lng": -75.1193}
        ]})

    monkeypatch.setattr(vd, "load_config", lambda: {"vdatum": {
        "datums_api": "https://api.example/stations/{station}/datums.json",
        "station_api": "https://api.example/stations/{station}.json",
    }})
    monkeypatch.setattr(vd.requests, "get", fake_get)

    lon, lat, sep = vd.station_mllw_in_navd88("8557380")
    assert (lon, lat) == (-75.1193, 38.7828)      # NOT the 'LAT' datum
    assert abs(sep - (0.826 - 1.548)) < 1e-12     # MLLW below NAVD88


def test_station_missing_datums_raises(monkeypatch):
    import dem_pipeline.vdatum as vd

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"datums": [{"name": "MHW", "value": 2.3}]}

    monkeypatch.setattr(vd, "load_config", lambda: {"vdatum": {
        "datums_api": "u/{station}/datums.json", "station_api": "u/{station}.json",
    }})
    monkeypatch.setattr(vd.requests, "get",
                        lambda url, params=None, timeout=None: FakeResponse())
    with raises(ValueError):
        vd.station_mllw_in_navd88("0000000")
