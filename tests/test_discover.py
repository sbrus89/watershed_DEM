"""Tests for stage 03's EAARL-B ScienceBase discovery.

The trap this guards: ScienceBase SILENTLY IGNORES an unrecognized filter.
`?filter=doi=<doi>` returns HTTP 200 with the entire 19M-item catalog, so the
original code walked 20 arbitrary items, matched no files, and reported
"0 files discovered" -- a broken query that looks exactly like an empty
dataset. Discovery must resolve the DOI to a concrete item id instead.
"""

import importlib.util
from pathlib import Path

from tests.util import importorskip, raises

_SPEC = importlib.util.spec_from_file_location(
    "discover_tiles",
    Path(__file__).resolve().parents[1] / "scripts" / "03_discover_tiles.py",
)
disc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(disc)

ITEM = "5f43b4cf82ce4c3d1222d2a5"


class _Resp:
    def __init__(self, payload=None, url="", ok=True):
        self._payload, self.url, self.ok = payload or {}, url, ok

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def _cfg(tmp_path, **eaarlb):
    base = {"role": "water", "priority": 2, "doi": "10.5066/P97031JU",
            "dir": str(tmp_path / "eaarlb"), "resampling": "average"}
    base.update(eaarlb)
    return {"sources": {"eaarlb": base}}


def _fake_get(calls, files):
    def get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        if "/catalog/item/" in url:
            return _Resp({"title": "EAARL-B", "files": files})
        return _Resp({"items": []})          # child query: no children
    return get


def test_pinned_item_is_used_and_the_catalog_filter_is_never_queried(
        tmp_path, monkeypatch):
    calls = []
    files = [
        {"name": "Topobathy_e470000_n4640000_v1.zip",
         "url": "https://x/t.zip", "size": 27_500_000},
        {"name": "topobathy_voids.zip", "url": "https://x/v.zip", "size": 4096},
        {"name": "perspective.PNG", "url": "https://x/p.png", "size": 1_600_000},
    ]
    import requests
    monkeypatch.setattr(requests, "get", _fake_get(calls, files))

    rows = disc.eaarlb_rows(_cfg(tmp_path, sciencebase_item=ITEM))

    # the pinned id is fetched directly ...
    assert any(f"/catalog/item/{ITEM}" in u for u, _ in calls)
    # ... and the silently-ignored filter query is never issued
    assert not any("doi" in str(p.get("filter", "")) for _, p in calls), calls
    # the void polygons the water mask subtracts must survive the filter,
    # and the preview image must not become a "data" row
    names = [Path(r["dest"]).name for r in rows]
    assert names == ["Topobathy_e470000_n4640000_v1.zip", "topobathy_voids.zip"]
    assert all(r["source"] == "eaarlb" for r in rows)


def test_unresolvable_doi_returns_empty_without_walking_the_catalog(
        tmp_path, monkeypatch):
    """A failed resolution must yield nothing, not 20 arbitrary catalog items."""
    calls = []
    import requests
    monkeypatch.setattr(requests, "get", _fake_get(calls, []))
    monkeypatch.setattr(disc, "_resolve_doi_to_item", lambda doi: None)

    rows = disc.eaarlb_rows(_cfg(tmp_path))
    assert rows == []
    assert calls == []


def test_doi_resolution_reads_the_item_id_out_of_the_redirect(monkeypatch):
    import requests
    monkeypatch.setattr(
        requests, "head",
        lambda url, **kw: _Resp(url=f"https://www.sciencebase.gov/catalog/item/{ITEM}"))
    assert disc._resolve_doi_to_item("10.5066/P97031JU") == ITEM

    # a redirect that is not a ScienceBase item must not yield a bogus id
    monkeypatch.setattr(requests, "head",
                        lambda url, **kw: _Resp(url="https://example.org/nope"))
    assert disc._resolve_doi_to_item("10.5066/P97031JU") is None


def test_eaarlb_tile_bbox_uses_the_nw_corner_convention():
    """Tile names encode the NW corner of a 10 km UTM 18N block.

    Reading it as the SW corner would shift every footprint 10 km north and
    let an estuary-only clip pull (or, worse, drop) the wrong tiles. Verified
    against the extracted raster for e510000_n4560000, whose true bounds are
    lon [-74.881, -74.837], lat [41.146, 41.191].
    """
    importorskip("pyproj")
    w, s, e, n = disc._eaarlb_tile_bbox("Topobathy_e510000_n4560000_v1.zip")
    # the real raster sits inside the nominal block ...
    assert w < -74.881 and e > -74.837, (w, e)
    assert s < 41.146 and n > 41.191, (s, n)
    # ... and the block extends SOUTH of the anchor latitude, not north
    assert s < 41.146 < n
    assert n < 41.30, f"NW-corner block must not reach {n} -- SW misread?"
    # non-tile members carry no e/n and must not be filtered spatially
    assert disc._eaarlb_tile_bbox("topobathy_voids.zip") is None
    assert disc._eaarlb_tile_bbox("Delaware River_topobathy_v1.xml") is None


def test_data_tiles_outside_the_basin_are_skipped_but_metadata_is_kept(
        tmp_path, monkeypatch):
    """clip_bbox is documented to narrow the entire pipeline. Unfiltered,
    an estuary-only run downloaded all 21 upper-river EAARL-B tiles (0.6 GB)
    that no output tile can ever read."""
    importorskip("pyproj")
    shapely = importorskip("shapely")
    from shapely.geometry import box

    files = [
        # lat ~41.15 -- upper Delaware, far north of the estuary
        {"name": "Topobathy_e510000_n4560000_v1.zip", "url": "https://x/a.zip"},
        # lat ~41.19 -- also outside
        {"name": "Topobathy_e520000_n4590000_v1.zip", "url": "https://x/b.zip"},
        {"name": "topobathy_voids.zip", "url": "https://x/v.zip"},
    ]
    import requests
    monkeypatch.setattr(requests, "get", _fake_get([], files))

    estuary = box(-75.60, 38.80, -74.95, 40.10)
    rows = disc.eaarlb_rows(_cfg(tmp_path, sciencebase_item=ITEM), estuary)
    assert [Path(r["dest"]).name for r in rows] == ["topobathy_voids.zip"]

    # the same call with a basin that DOES cover the upper river keeps them,
    # so the filter is spatial rather than a blanket exclusion
    upper = box(-75.0, 41.0, -74.7, 41.4)
    rows = disc.eaarlb_rows(_cfg(tmp_path, sciencebase_item=ITEM), upper)
    assert len(rows) == 3, [Path(r["dest"]).name for r in rows]

    # and no basin at all means no spatial filtering (back-compatible)
    rows = disc.eaarlb_rows(_cfg(tmp_path, sciencebase_item=ITEM))
    assert len(rows) == 3


# ---- eHydro district selection ---------------------------------------------

def _ehydro_cfg(tmp_path, district):
    return {"sources": {"ehydro": {
        "feature_server": "https://fs/query", "district": district,
        "dir": str(tmp_path / "ehydro")}}}


def _ehydro_fake_get(calls, by_district):
    """FeatureServer stub that honours the `where` clause's district list."""
    def get(url, params=None, timeout=None):
        calls.append(dict(params or {}))
        where = params["where"]
        feats = [f for d, fs in by_district.items() if f"'{d}'" in where
                 for f in fs] if params["resultOffset"] == 0 else []
        return _Resp({"type": "FeatureCollection", "features": feats})
    return get


def _survey(name, x):
    return {"type": "Feature",
            "properties": {"sourcedatalocation": f"https://e/{name}.zip",
                           "surveydateend": 1700000000000},
            "geometry": {"type": "Point", "coordinates": [x, 37.5]}}


def test_ehydro_list_of_districts_fetches_surveys_from_every_district(
        tmp_path, monkeypatch):
    """A basin straddling two USACE districts (0208: CENAO + CENAB) must get
    both districts' surveys. The old f-string `usacedistrictcode='{district}'`
    turned a YAML list into `='['CENAO', 'CENAB']'`, which matches nothing --
    an empty survey set that looks like a district with no data."""
    importorskip("shapely")
    from shapely.geometry import box
    import requests
    calls = []
    monkeypatch.setattr(requests, "get", _ehydro_fake_get(calls, {
        "CENAO": [_survey("norfolk", -76.3)],
        "CENAB": [_survey("baltimore", -75.9)],
        "CENAP": [_survey("philly", -75.8)],      # not configured
    }))
    rows = disc.ehydro_rows(_ehydro_cfg(tmp_path, ["CENAO", "CENAB"]),
                            box(-77, 37, -75, 38))
    assert sorted(Path(r["dest"]).name for r in rows) == \
        ["baltimore.zip", "norfolk.zip"]


def test_ehydro_single_district_string_still_works(tmp_path, monkeypatch):
    """Every existing basin config uses a plain string -- it must not be
    iterated character by character into ('C','E','N','A','B')."""
    importorskip("shapely")
    from shapely.geometry import box
    import requests
    calls = []
    monkeypatch.setattr(requests, "get", _ehydro_fake_get(calls, {
        "CENAB": [_survey("baltimore", -75.9)],
        "CENAO": [_survey("norfolk", -76.3)]}))
    rows = disc.ehydro_rows(_ehydro_cfg(tmp_path, "CENAB"),
                            box(-77, 37, -75, 38))
    assert [Path(r["dest"]).name for r in rows] == ["baltimore.zip"]
    assert calls[0]["where"] == "usacedistrictcode IN ('CENAB')"


def test_ehydro_district_rejects_empty_and_malformed_codes():
    """An empty list would query nothing; a stray quote would break (or
    widen) the SQL. Both must fail loudly, not return zero surveys."""
    with raises(ValueError):
        disc._ehydro_district_where([])
    with raises(ValueError):
        disc._ehydro_district_where("CENAB' OR '1'='1")
