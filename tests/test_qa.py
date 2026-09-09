"""Tests for the QA checks (dem_pipeline.qa) on miniature synthetic tiles."""

import numpy as np

from tests.util import importorskip, make_geotiff

NODATA = -999999.0
PX = 0.25 / 90  # 90x90 px quarter-degree test tiles

TILES = {
    "n40x00_w075x25": [-75.25, PX, 0.0, 40.0, 0.0, -PX],
    "n40x00_w075x00": [-75.00, PX, 0.0, 40.0, 0.0, -PX],  # east neighbor
    "n39x75_w075x25": [-75.25, PX, 0.0, 39.75, 0.0, -PX],  # south neighbor
}


def _sandbox(tmp_path, monkeypatch):
    """Patch qa/grid onto a tmp repo with a small-tile GridSpec."""
    importorskip("rasterio")
    from rasterio.transform import Affine

    import dem_pipeline.grid as grid_mod
    import dem_pipeline.qa as qa
    from dem_pipeline.grid import GridSpec

    (tmp_path / "config").mkdir()
    GridSpec(
        epsg=4269, nodata=NODATA,
        cudem_geotransforms=dict(TILES),
        cudem_sizes={k: [90, 90] for k in TILES},
    ).dump(tmp_path / "config" / "tile_grid.json")

    monkeypatch.setattr(grid_mod, "repo_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(qa, "repo_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(qa, "load_config", lambda: {
        "qa": {"seam_gradient_flag_m": 5.0, "land_overlap_median_max_m": 0.10,
               "quicklook_dir": str(tmp_path / "ql"), "dir": str(tmp_path)},
        "sources": {"cudem19": {"dir": str(tmp_path / "cudem")}},
        "compositing": {"source_ids": {"nodata": 0, "s1m": 1, "cudem19": 13,
                                       "filled": 99}},
    })

    # continuous synthetic terrain z = f(lon, lat) so tile edges agree
    def terrain(gt):
        cols, rows = np.meshgrid(np.arange(90), np.arange(90))
        lon = gt[0] + (cols + 0.5) * gt[1]
        lat = gt[3] + (rows + 0.5) * gt[5]
        return (100 * (lon + 76) + 100 * (lat - 39)).astype("f4")

    tile_dir = tmp_path / "tiles"
    paths = {}
    for name, gt in TILES.items():
        arr = terrain(gt)
        paths[name] = make_geotiff(tile_dir / f"{name}.tif", arr,
                                   Affine.from_gdal(*gt), "EPSG:4269",
                                   nodata=NODATA)
    return qa, paths, terrain


def test_grid_conformity_pass_and_fail(tmp_path, monkeypatch):
    from rasterio.transform import Affine

    qa, paths, terrain = _sandbox(tmp_path, monkeypatch)
    res = qa.check_grid_conformity(paths)
    assert res["pass"] and res["n_tiles"] == 3

    # a tile on the wrong grid must be caught
    bad_gt = [-75.251, PX, 0.0, 40.0, 0.0, -PX]
    make_geotiff(tmp_path / "tiles" / "bad.tif", terrain(bad_gt),
                 Affine.from_gdal(*bad_gt), "EPSG:4269", nodata=NODATA)
    paths_bad = dict(paths)
    paths_bad["n40x00_w075x25"] = tmp_path / "tiles" / "bad.tif"
    res = qa.check_grid_conformity(paths_bad)
    assert not res["pass"] and "n40x00_w075x25" in res["failures"]


def test_seams_continuous_and_step(tmp_path, monkeypatch):
    from rasterio.transform import Affine

    qa, paths, terrain = _sandbox(tmp_path, monkeypatch)
    res = qa.check_seams(paths)
    assert res["pass"]
    assert res["n_edges"] == 2  # east + south of the corner tile
    assert res["flagged"] == {}  # smooth terrain flags nothing

    # introduce a 50 m cliff in the east neighbor -> seam check must flag it
    gt = TILES["n40x00_w075x00"]
    arr = terrain(gt) + 50.0
    make_geotiff(tmp_path / "tiles" / "n40x00_w075x00.tif", arr,
                 Affine.from_gdal(*gt), "EPSG:4269", nodata=NODATA)
    res = qa.check_seams(paths)
    assert not res["pass"] and len(res["flagged"]) >= 1


def test_seams_collar_branch(tmp_path, monkeypatch):
    """With an overlap collar, neighboring tiles share duplicated pixels
    that must agree exactly; disagreement must be flagged."""
    from rasterio.transform import Affine

    import dem_pipeline.grid as grid_mod
    import dem_pipeline.qa as qa
    from dem_pipeline.grid import GridSpec

    collar = 3
    n = 90 + 2 * collar
    names = ["n40x00_w075x25", "n40x00_w075x00"]
    tiles = {nm: [w0 - collar * PX, PX, 0.0, 40.0 + collar * PX, 0.0, -PX]
             for nm, w0 in zip(names, (-75.25, -75.00))}

    (tmp_path / "config").mkdir()
    GridSpec(epsg=4269, nodata=NODATA, collar_px=collar,
             cudem_geotransforms=tiles,
             cudem_sizes={k: [n, n] for k in tiles}).dump(
        tmp_path / "config" / "tile_grid.json")
    monkeypatch.setattr(grid_mod, "repo_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(qa, "repo_path", lambda *p: tmp_path.joinpath(*p))
    monkeypatch.setattr(qa, "load_config", lambda: {
        "qa": {"seam_gradient_flag_m": 5.0,
               "land_overlap_median_max_m": 0.10,
               "quicklook_dir": str(tmp_path / "ql"), "dir": str(tmp_path)},
        "sources": {"cudem19": {"dir": str(tmp_path / "cudem")}},
        "compositing": {"source_ids": {"nodata": 0}},
    })

    def terrain(gt):
        cols, rows = np.meshgrid(np.arange(n), np.arange(n))
        lon = gt[0] + (cols + 0.5) * gt[1]
        lat = gt[3] + (rows + 0.5) * gt[5]
        return (100 * (lon + 76) + 100 * (lat - 39)).astype("f4")

    paths = {nm: make_geotiff(tmp_path / "tiles" / f"{nm}.tif", terrain(gt),
                              Affine.from_gdal(*gt), "EPSG:4269",
                              nodata=NODATA)
             for nm, gt in tiles.items()}

    res = qa.check_seams(paths)
    # duplicated collar pixels are identical samples of the same function
    assert res["pass"] and res["n_edges"] == 1 and res["flagged"] == {}

    # corrupt the east tile's collar: overlapping pixels now disagree by 50 m
    gt = tiles["n40x00_w075x00"]
    bad = terrain(gt)
    bad[:, : 2 * collar] += 50.0
    make_geotiff(tmp_path / "tiles" / "n40x00_w075x00.tif", bad,
                 Affine.from_gdal(*gt), "EPSG:4269", nodata=NODATA)
    res = qa.check_seams(paths)
    assert not res["pass"] and len(res["flagged"]) == 1


def test_seams_inherent_cudem_misalignment_not_flagged(tmp_path, monkeypatch):
    """A cliff whose worst pixel is cudem19 on both sides is the known,
    accepted CUDEM-vintage registration mismatch -- reported separately, and
    must not fail the check."""
    from rasterio.transform import Affine

    qa, paths, terrain = _sandbox(tmp_path, monkeypatch)

    gt = TILES["n40x00_w075x00"]
    arr = terrain(gt) + 50.0
    make_geotiff(tmp_path / "tiles" / "n40x00_w075x00.tif", arr,
                 Affine.from_gdal(*gt), "EPSG:4269", nodata=NODATA)
    for name in ("n40x00_w075x25", "n40x00_w075x00"):
        ids = np.full((90, 90), 13, dtype="f4")  # cudem19 on both sides
        make_geotiff(tmp_path / "tiles" / f"{name}_source.tif", ids,
                     Affine.from_gdal(*TILES[name]), "EPSG:4269")

    res = qa.check_seams(paths)
    assert res["pass"]
    assert res["flagged"] == {}
    assert "n40x00_w075x25|n40x00_w075x00" in res["inherent_misalignment"]


def test_seams_mixed_source_cliff_still_flagged(tmp_path, monkeypatch):
    """Same-magnitude cliff, but the two sides trace to different sources --
    must never be auto-accepted as inherent misalignment, since a
    cudem19/dem13 pairing has also turned out to be a real, undiagnosed bug
    elsewhere in this basin. Only a same-source (cudem19/cudem19) pair is
    trusted enough to auto-classify."""
    from rasterio.transform import Affine

    qa, paths, terrain = _sandbox(tmp_path, monkeypatch)

    gt = TILES["n40x00_w075x00"]
    arr = terrain(gt) + 50.0
    make_geotiff(tmp_path / "tiles" / "n40x00_w075x00.tif", arr,
                 Affine.from_gdal(*gt), "EPSG:4269", nodata=NODATA)
    make_geotiff(tmp_path / "tiles" / "n40x00_w075x25_source.tif",
                 np.full((90, 90), 13, dtype="f4"),  # cudem19
                 Affine.from_gdal(*TILES["n40x00_w075x25"]), "EPSG:4269")
    make_geotiff(tmp_path / "tiles" / "n40x00_w075x00_source.tif",
                 np.full((90, 90), 1, dtype="f4"),  # s1m
                 Affine.from_gdal(*TILES["n40x00_w075x00"]), "EPSG:4269")

    res = qa.check_seams(paths)
    assert not res["pass"]
    assert "n40x00_w075x25|n40x00_w075x00" in res["flagged"]
    assert res["inherent_misalignment"] == {}


def test_cudem_land_overlap_bias_detection(tmp_path, monkeypatch):
    from rasterio.transform import Affine

    qa, paths, terrain = _sandbox(tmp_path, monkeypatch)
    name = "n40x00_w075x25"
    gt = TILES[name]

    # CUDEM twin offset by +2 cm -> pass (median well under 0.10 m)
    make_geotiff(tmp_path / "cudem" / f"ncei19_{name}_2014v1.tif",
                 terrain(gt) + 0.02, Affine.from_gdal(*gt), "EPSG:4269",
                 nodata=NODATA)
    res = qa.check_cudem_land_overlap(paths)
    assert res["pass"]
    assert abs(res["tiles"][name]["median"] + 0.02) < 1e-3

    # 0.5 m systematic offset (datum error) -> fail
    make_geotiff(tmp_path / "cudem" / f"ncei19_{name}_2014v1.tif",
                 terrain(gt) + 0.5, Affine.from_gdal(*gt), "EPSG:4269",
                 nodata=NODATA)
    res = qa.check_cudem_land_overlap(paths)
    assert not res["pass"]


def test_coverage_report(tmp_path, monkeypatch):
    from rasterio.transform import Affine

    qa, paths, _ = _sandbox(tmp_path, monkeypatch)
    name = "n40x00_w075x25"
    ids = np.full((90, 90), 1, dtype="f4")   # make_geotiff writes float32;
    ids[:, 45:] = 13                          # coverage reads values, not dtype
    make_geotiff(tmp_path / "tiles" / f"{name}_source.tif", ids,
                 Affine.from_gdal(*TILES[name]), "EPSG:4269")
    res = qa.coverage_report({name: paths[name]})
    row = res["tiles"][name]
    assert row["s1m"] == 50.0 and row["cudem19"] == 50.0
