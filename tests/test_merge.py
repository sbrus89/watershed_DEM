"""Tests for the region merge (dem_pipeline.merge) on tiny synthetic tiles."""

import json

import numpy as np

from tests.util import importorskip, make_geotiff, raises, skip_unless_cli

NODATA = -999999.0
N = 60                       # px per synthetic tile
PX = 0.25 / N
IDS = {"nodata": 0, "s1m": 1, "cudem19": 13, "filled": 99}
NONREAL = [0, 99]


def _split_alphas(f=5, split=30):
    """Territory weights for basin A owning columns < split, B the rest."""
    from dem_pipeline.merge import territory_alphas

    pad = f + 2
    lab = np.full((N + 2 * pad, N + 2 * pad), 2, "u1")
    lab[:, : split + pad] = 1
    return territory_alphas(lab, [1, 2], f, pad)


def test_blend_feathers_across_territory_boundary():
    """Catches: a hard switch at the basin boundary (a step that would
    contaminate subgrid tables), or weights not reaching exactly 1 deep in a
    territory (owner values altered far from any boundary)."""
    importorskip("scipy")
    from dem_pipeline.merge import blend

    za = np.zeros((N, N), "f4")
    zb = np.full((N, N), 10.0, "f4")
    p = np.ones((N, N), "u1")
    res = blend([za, zb], [p, p], _split_alphas(), NONREAL)
    row = res["z"][0]
    assert np.all(row[:20] == 0.0) and np.all(row[40:] == 10.0)
    # feather_alpha's ramp steps 2/(2f) across the boundary pixel pair: 2 m
    # here; a hard switch would be the full 10 m
    assert np.all(np.diff(row) >= 0) and np.max(np.diff(row)) <= 2.0 + 1e-5
    assert abs(row[29] - 4.0) < 1e-5 and abs(row[30] - 6.0) < 1e-5  # 0.4/0.6
    assert np.all(res["winner"][:, :20] == 0) and np.all(res["winner"][:, 40:] == 1)
    assert res["counts"]["blended"] > 0 and res["band_spread_m"]["max"] == 10.0


def test_blend_filled_never_outweighs_real():
    """Catches: a basin's fillnodata extrapolation past its data edge
    (provenance 99, often tens of metres wrong) being kept because the pixel
    lies in that basin's territory."""
    importorskip("scipy")
    from dem_pipeline.merge import blend

    za = np.full((N, N), 500.0, "f4")
    pa = np.full((N, N), 99, "u1")          # A: filled garbage everywhere
    zb = np.full((N, N), 7.0, "f4")
    pb = np.ones((N, N), "u1")
    res = blend([za, zb], [pa, pb], _split_alphas(), NONREAL)
    assert np.all(res["z"] == 7.0)
    assert np.all(res["prov"] == 1) and np.all(res["winner"] == 1)
    assert res["counts"]["fallback_real"] > 0


def test_blend_fallback_tiers_and_nodata():
    """Catches: owner nodata becoming output nodata although a neighbour has
    data (tier 2), filled data dropped when it is all there is (tier 3), and
    nodata invented where every version is nodata."""
    importorskip("scipy")
    from dem_pipeline.merge import blend

    za = np.full((N, N), np.nan, "f4")
    pa = np.zeros((N, N), "u1")
    zb = np.full((N, N), np.nan, "f4")
    pb = np.zeros((N, N), "u1")
    zb[:, :10] = 3.0
    pb[:, :10] = 1                       # real, deep in A's territory
    zb[:, 10:20] = 4.0
    pb[:, 10:20] = 99                    # filled only
    res = blend([za, zb], [pa, pb], _split_alphas(), NONREAL)
    assert np.all(res["z"][:, :10] == 3.0) and np.all(res["prov"][:, :10] == 1)
    assert np.all(res["z"][:, 10:20] == 4.0) and np.all(res["prov"][:, 10:20] == 99)
    assert np.all(np.isnan(res["z"][:, 20:])) and np.all(res["winner"][:, 20:] == -1)
    assert np.all(res["prov"][:, 20:] == 0)


def test_build_grid_spec_union_and_conflict():
    """Catches: a CUDEM tile with two different audited grids silently taking
    one (breaks the bit-exact contract for the other basin's consumers), and
    the non-CUDEM convention coming from the wrong basin."""
    from dem_pipeline.grid import GridSpec
    from dem_pipeline.merge import build_grid_spec

    gt = [-75.25, PX, 0.0, 40.0, 0.0, -PX]
    a = GridSpec(x_shift_px=-1e-6, cudem_geotransforms={"n40x00_w075x25": gt},
                 cudem_sizes={"n40x00_w075x25": [N, N]})
    b = GridSpec(x_shift_px=-0.00048,
                 cudem_geotransforms={"n40x00_w075x25": list(gt),
                                      "n40x00_w075x00": [-75.0, PX, 0, 40.0, 0, -PX]},
                 cudem_sizes={"n40x00_w075x25": [N, N], "n40x00_w075x00": [N, N]})
    s = build_grid_spec({"A": a, "B": b}, "A")
    assert s.x_shift_px == -1e-6 and len(s.cudem_geotransforms) == 2
    b.cudem_geotransforms["n40x00_w075x25"] = [-75.25 + 1e-9, PX, 0, 40.0, 0, -PX]
    with raises(ValueError):
        build_grid_spec({"A": a, "B": b}, "A")


def test_check_restamp_tolerance():
    """Catches: a real misregistration (a whole or half pixel) being
    relabelled as if it were the ~0.001-px Hudson/Delaware convention gap."""
    from dem_pipeline.merge import check_restamp

    gt = (-75.25, PX, 0.0, 40.0, 0.0, -PX)
    check_restamp((gt[0] + 0.001 * PX,) + gt[1:], (N, N), gt, (N, N), 0.01)
    with raises(ValueError):
        check_restamp((gt[0] + 0.5 * PX,) + gt[1:], (N, N), gt, (N, N), 0.01)
    with raises(ValueError):
        check_restamp(gt, (N, N + 1), gt, (N, N), 0.01)
    with raises(ValueError):    # pixel-size drift accumulating across the tile
        check_restamp(gt[:1] + (PX * 1.001,) + gt[2:], (N, N), gt, (N, N), 0.01)


def test_territories_partition_and_nearest_exterior():
    """Catches: interior pixels assigned to a neighbour (ownership must be the
    exact HUC-4 polygon), and exterior pixels left unowned or given to a
    far basin."""
    importorskip("geopandas")
    import shapely

    from dem_pipeline.merge import build_territories, rasterize_territories

    a = shapely.box(-76.0, 39.0, -75.5, 39.5)
    b = shapely.box(-75.5, 39.0, -75.0, 39.5)
    t = build_territories({1: a, 2: b}, simplify_m=10, spacing_m=500,
                          extend_m=50e3)
    assert t[1][0].equals(a) and t[2][0].equals(b)     # polygons untouched
    step = 0.005
    gt = (-76.4, step, 0.0, 39.9, 0.0, -step)
    lab = rasterize_territories(t, gt, (240, 360))
    assert lab.min() >= 1

    def at(x, y):
        return lab[int((gt[3] - y) / step), int((x - gt[0]) / step)]

    for (x, y), code in (((-75.9, 39.2), 1), ((-75.5125, 39.2), 1),
                         ((-75.4875, 39.2), 2), ((-76.2, 39.25), 1),
                         ((-74.8, 39.25), 2), ((-75.7, 38.8), 1),
                         ((-75.3, 39.7), 2)):
        assert at(x, y) == code, ((x, y), at(x, y))


# ---------------------------------------------------------------- end to end

def _basin(root, key, huc4, name, tiles, gts):
    """A fake basin checkout: config/sources.yaml, tile_grid.json, final tiles."""
    import yaml
    from rasterio.transform import Affine

    from dem_pipeline.grid import GridSpec

    (root / "config").mkdir(parents=True)
    (root / "config" / "sources.yaml").write_text(yaml.safe_dump({
        "project": {"name": name, "version": "v1", "huc4": huc4},
        "output": {"tile_dir": "data/final/tiles"},
        "compositing": {"source_ids": IDS}}))
    GridSpec(cudem_geotransforms=dict(gts),
             cudem_sizes={k: [N, N] for k in gts}).dump(root / "config" / "tile_grid.json")
    td = root / "data" / "final" / "tiles"
    for t, (z, p, gt) in tiles.items():
        make_geotiff(td / f"{name}_{t}_v1.tif", z, Affine.from_gdal(*gt),
                     "EPSG:4269", nodata=NODATA)
        import rasterio
        with rasterio.open(td / f"{name}_{t}_v1_source.tif", "w", driver="GTiff",
                           dtype="uint8", count=1, width=N, height=N, nodata=0,
                           crs="EPSG:4269", transform=Affine.from_gdal(*gt)) as d:
            d.write(p, 1)
    return {"key": key, "huc4": huc4, "root": str(root)}


def test_merge_end_to_end(tmp_path):
    """Catches: single-basin tiles altered by the merge, the restamped tile
    not landing on the merged grid, owner values changed away from the
    boundary, basin-of-origin codes wrong, the skip cache trusting stale
    output after an input changes, and QA not seeing the merged tiles."""
    importorskip("rasterio")
    gpd = importorskip("geopandas")
    skip_unless_cli("gdal_translate")
    import rasterio
    import shapely
    import yaml

    from dem_pipeline import merge

    shared, only_a, only_b = "n40x00_w075x25", "n40x00_w075x50", "n40x00_w075x00"
    gt_s = (-75.25, PX, 0.0, 40.0, 0.0, -PX)
    gt_a = (-75.50, PX, 0.0, 40.0, 0.0, -PX)
    gt_b = (-75.00, PX, 0.0, 40.0, 0.0, -PX)
    gt_b_off = (gt_b[0] + 0.001 * PX,) + gt_b[1:]     # Hudson-style tiny offset
    one = np.ones((N, N), "u1")
    rng = np.random.default_rng(0)
    za_s = rng.normal(0, 1, (N, N)).astype("f4")
    zb_s = (za_s + 10).astype("f4")
    za_only = rng.normal(5, 1, (N, N)).astype("f4")
    zb_only = rng.normal(9, 1, (N, N)).astype("f4")
    zb_only[0, 0] = np.nan

    ba = _basin(tmp_path / "A", "A", "0204", "aaa19",
                {shared: (za_s, one, gt_s), only_a: (za_only, one, gt_a)},
                {shared: gt_s, only_a: gt_a, only_b: gt_b})
    bb = _basin(tmp_path / "B", "B", "0205", "bbb19",
                {shared: (zb_s, one, gt_s), only_b: (zb_only, one, gt_b_off)},
                {shared: gt_s})
    # A owns west of -75.125 (the middle of the shared tile)
    wbd = tmp_path / "wbd.gpkg"
    gpd.GeoDataFrame({"huc4": ["0204", "0205"]},
                     geometry=[shapely.box(-75.5, 39.75, -75.125, 40.0),
                               shapely.box(-75.125, 39.75, -74.75, 40.0)],
                     crs=4269).to_file(wbd, layer="WBDHU4", driver="GPKG")
    cfg_path = tmp_path / "merge.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "project": {"name": "mrg19", "version": "v9", "region_id": "rtest"},
        "output": {"root": str(tmp_path / "out"), "epsg": 4269,
                   "cog": {"compress": "DEFLATE", "predictor": 3,
                           "blocksize": 512, "overview_resampling": "AVERAGE"}},
        "wbd": {"gpkg": str(wbd), "layer": "WBDHU4", "field": "huc4"},
        "merge": {"grid_convention_from": "A", "feather_px": 5,
                  "restamp_tol_px": 0.01},
        "basins": [ba, bb],
        "qa": {"seam_gradient_flag_m": 1000.0, "basin_nodata_edge_buffer_m": 0,
               "basin_nodata_max_px": 0,
               "basin_nodata_accepted_px": {only_b: 1}}}))
    mcfg = merge.load_merge_config(cfg_path)
    state = merge.prepare(mcfg)
    assert state["by_n_basins"] == {"1": 2, "2": 1}
    # an unchanged rerun of prepare must keep the territory digest, or every
    # blended tile's fingerprint (and the resume cache) is invalidated
    assert merge.prepare(mcfg)["territory_digest"] == state["territory_digest"]

    basins, spec, terr, digest, ids, tiles = merge.load_prepared(mcfg)
    for t in tiles:
        merge.merge_tile(t, tiles[t], mcfg, spec, terr, digest, ids)
    out = merge.tiles_dir(mcfg)

    def rd(t, suf=""):
        with rasterio.open(out / f"mrg19_{t}_v9{suf}.tif") as s:
            a = s.read(1).astype("f4")
            if not suf:
                a[a == NODATA] = np.nan
            return a, s.transform.to_gdal()

    z, g = rd(only_a)
    assert np.array_equal(z, za_only) and g == gt_a
    z, g = rd(only_b)
    assert np.array_equal(z, zb_only, equal_nan=True) and g == gt_b  # restamped
    assert json.loads(merge.stats_path(mcfg, only_b).read_text())["mode"] == "restamp"
    z, g = rd(shared)
    assert g == gt_s
    assert np.array_equal(z[:, :20], za_s[:, :20])          # A side, past band
    assert np.array_equal(z[:, 40:], zb_s[:, 40:])          # B side, past band
    bas, _ = rd(shared, "_basin")
    assert np.all(bas[:, :20] == 4) and np.all(bas[:, 40:] == 5)
    bas, _ = rd(only_b, "_basin")
    assert bas[0, 0] == 0 and bas[1, 1] == 5

    # skip cache: unchanged inputs -> untouched; changed input -> rebuilt
    e = out / f"mrg19_{shared}_v9.tif"
    m0 = e.stat().st_mtime_ns
    merge.merge_tile(shared, tiles[shared], mcfg, spec, terr, digest, ids)
    assert e.stat().st_mtime_ns == m0
    src = tmp_path / "A" / "data" / "final" / "tiles" / f"aaa19_{shared}_v1.tif"
    with rasterio.open(src, "r+") as d:
        d.write((za_s + 100).astype("f4"), 1)
    merge.merge_tile(shared, tiles[shared], mcfg, spec, terr, digest, ids)
    z, _ = rd(shared)
    assert np.array_equal(z[:, :20], za_s[:, :20] + 100)

    checks = {c["check"]: c for c in merge.run_qa(mcfg)}
    assert checks["completeness"]["pass"] and checks["grid_conformity"]["pass"]
    assert checks["basin_nodata"]["pass"]
    assert checks["basin_nodata"]["accepted_gaps"] == {only_b: 1}
    assert checks["merge_summary"]["modes"] == {"copy": 1, "restamp": 1, "blend": 1}
