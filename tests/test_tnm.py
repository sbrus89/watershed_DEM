"""Tests for the TNM API client (dem_pipeline.tnm): caching + pagination."""

import json

from tests.util import importorskip


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


TEST_CFG = {
    "tnm_api": {"base": "http://tnm.invalid/api/v1", "page_size": 2},
    "download": {"retries": 2, "timeout_s": 5, "workers": 1},
}


def _patch_cache(monkeypatch, tmp_path):
    """Isolate tnm from the real config (live URL, retries) and filesystem
    cache; also neutralize the retry backoff sleep."""
    import dem_pipeline.tnm as tnm

    monkeypatch.setattr(tnm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(tnm, "load_config", lambda: TEST_CFG)
    monkeypatch.setattr(tnm.time, "sleep", lambda s: None)
    return tnm


def test_cached_get_hits_network_once(tmp_path, monkeypatch):
    tnm = _patch_cache(monkeypatch, tmp_path)
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        return _FakeResponse({"total": 1, "items": [{"downloadURL": "u"}]})

    monkeypatch.setattr(tnm.requests, "get", fake_get)
    for _ in range(3):
        data = tnm._cached_get("http://api/products", {"a": 1}, 3, 10)
        assert data["total"] == 1
    assert len(calls) == 1  # served from cache after the first call

    # a different query is a cache miss
    tnm._cached_get("http://api/products", {"a": 2}, 3, 10)
    assert len(calls) == 2


def test_query_products_paginates(tmp_path, monkeypatch):
    tnm = _patch_cache(monkeypatch, tmp_path)
    pages = {
        0: {"total": 3, "items": [{"downloadURL": "a"}, {"downloadURL": "b"}]},
        2: {"total": 3, "items": [{"downloadURL": "c"}]},
    }

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(pages[params["offset"]])

    monkeypatch.setattr(tnm.requests, "get", fake_get)
    items = tnm.query_products("Some Dataset", bbox=(-76, 38, -74, 42))
    assert [i["downloadURL"] for i in items] == ["a", "b", "c"]


def test_cached_get_retries_then_succeeds(tmp_path, monkeypatch):
    import requests as real_requests

    tnm = _patch_cache(monkeypatch, tmp_path)
    calls = []

    def flaky_get(url, params=None, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise real_requests.ConnectionError("transient")
        return _FakeResponse({"total": 0, "items": []})

    monkeypatch.setattr(tnm.requests, "get", flaky_get)
    data = tnm._cached_get("http://api/products", {"q": 1}, retries=2,
                           timeout=5)
    assert data["total"] == 0 and len(calls) == 2


def test_cached_get_exhausts_retries(tmp_path, monkeypatch):
    import requests as real_requests

    tnm = _patch_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(
        tnm.requests, "get",
        lambda url, params=None, timeout=None:
            (_ for _ in ()).throw(real_requests.ConnectionError("down")))
    from tests.util import raises
    with raises(RuntimeError):
        tnm._cached_get("http://api/products", {"q": 2}, retries=2, timeout=5)


def test_simplify_polygon_fits_and_roundtrips():
    shapely = importorskip("shapely")
    from shapely.geometry import Point

    import dem_pipeline.tnm as tnm

    # a detailed ring (~2000 vertices) must simplify to fit the URL budget
    poly = Point(-75.0, 40.0).buffer(1.0, quad_segs=500)
    s = tnm.simplify_polygon_for_api(poly, max_chars=2000)
    assert s is not None and len(s) <= 2000
    pairs = [p.split() for p in s.split(",")]
    assert len(pairs) >= 4
    lon, lat = float(pairs[0][0]), float(pairs[0][1])
    assert -76.5 < lon < -73.5 and 38.5 < lat < 41.5
    del shapely


def test_simplify_polygon_returns_none_when_impossible():
    importorskip("shapely")
    from shapely.geometry import Point

    import dem_pipeline.tnm as tnm

    poly = Point(-75.0, 40.0).buffer(1.0, quad_segs=100)
    assert tnm.simplify_polygon_for_api(poly, max_chars=10) is None
