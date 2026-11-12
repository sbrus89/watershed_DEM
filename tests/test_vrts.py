"""Tests for stage 05's VRT construction.

The bug these guard: gdalbuildvrt refuses heterogeneous projections by
SILENTLY SKIPPING the offending sources and still exiting 0. Real CUDEM mixes
EPSG:5498 and EPSG:4269 across vintages, so 28 of the Delaware basin's 49
tiles vanished from the water stack with no error anywhere -- and the stage
logged "49 inputs" while writing a 21-source VRT.
"""

import importlib.util
import subprocess
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "build_source_vrts",
    Path(__file__).resolve().parents[1] / "scripts" / "05_build_source_vrts.py",
)
vrts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vrts)


def _stub_gdalbuildvrt(monkeypatch, n_sources):
    """Pretend gdalbuildvrt ran and emitted n_sources, exiting 0 either way."""
    def run(cmd, **kw):
        out = Path(cmd[-1])
        body = "".join(f"<SourceFilename>{i}</SourceFilename>"
                       for i in range(n_sources))
        out.write_text(f"<VRTDataset>{body}</VRTDataset>")
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(vrts.subprocess, "run", run)


def test_dropped_sources_are_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(vrts, "horizontal_epsg", lambda p: 4269)
    _stub_gdalbuildvrt(monkeypatch, n_sources=21)      # 21 of 49 survive
    tifs = [tmp_path / f"t{i}.tif" for i in range(49)]

    try:
        vrts.build_vrt("cudem19", tifs, tmp_path)
    except SystemExit as e:
        assert "21 of 49" in str(e), e
    else:
        raise AssertionError("dropped sources must abort the stage")


def test_complete_vrt_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(vrts, "horizontal_epsg", lambda p: 4269)
    _stub_gdalbuildvrt(monkeypatch, n_sources=49)
    tifs = [tmp_path / f"t{i}.tif" for i in range(49)]
    vrts.build_vrt("cudem19", tifs, tmp_path)          # must not raise
    assert (tmp_path / "cudem19.vrt").exists()


def test_projection_difference_is_allowed_only_within_one_horizontal_frame(
        tmp_path, monkeypatch):
    """EPSG:5498 and 4269 share a horizontal frame, so -allow_projection_
    difference is safe. Genuinely different horizontal CRSs are not: the VRT
    would place those sources at the wrong coordinates."""
    cmds = []

    def run(cmd, **kw):
        cmds.append(cmd)
        Path(cmd[-1]).write_text(
            "<VRTDataset><SourceFilename>a</SourceFilename></VRTDataset>")
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(vrts.subprocess, "run", run)

    # same horizontal frame (5498 and 4269 both reduce to 4269) -> allowed
    monkeypatch.setattr(vrts, "horizontal_epsg", lambda p: 4269)
    vrts.build_vrt("cudem19", [tmp_path / "a.tif"], tmp_path)
    assert "-allow_projection_difference" in cmds[0]

    # different horizontal frames -> must abort, not silently mis-place data
    seen = iter([4269, 26918])
    monkeypatch.setattr(vrts, "horizontal_epsg", lambda p: next(seen))
    try:
        vrts.build_vrt("mixed", [tmp_path / "a.tif", tmp_path / "b.tif"],
                       tmp_path)
    except SystemExit as e:
        assert "horizontal CRS" in str(e), e
    else:
        raise AssertionError("mixed horizontal CRSs must abort the stage")


def _zip_with(path, members):
    import zipfile
    with zipfile.ZipFile(path, "w") as z:
        for name, body in members.items():
            z.writestr(name, body)


def test_zipped_rasters_are_extracted_before_the_glob(tmp_path):
    """EAARL-B ships one GeoTIFF per zip and nothing else unpacked them, so
    stage 05 saw an empty source dir against a 24-row manifest and aborted
    with "stage 04 did not complete" on a download that had in fact finished.
    """
    _zip_with(tmp_path / "Topobathy_e510000_n4560000_v1.zip",
              {"Topobathy_e510000_n4560000_v1.tif": "II*\0raster",
               "Topobathy_e510000_n4560000_v1_FGDC.xml": "<meta/>"})
    # archives with no raster (the 10 km grid index, the void polygons) are
    # left alone -- unpacking them would put shapefiles in the raster glob
    _zip_with(tmp_path / "grid10km_DelRiv.zip", {"grid10km_DelRiv.shp": "shp"})

    assert vrts.extract_archives(tmp_path) == 1
    assert (tmp_path / "Topobathy_e510000_n4560000_v1"
            / "Topobathy_e510000_n4560000_v1.tif").exists()
    assert not (tmp_path / "grid10km_DelRiv").exists()
    assert sorted(p.name for p in tmp_path.rglob("*.tif")) == [
        "Topobathy_e510000_n4560000_v1.tif"]

    # idempotent: a rerun re-extracts nothing and leaves no scratch directory
    assert vrts.extract_archives(tmp_path) == 0
    assert not list(tmp_path.glob("*.tmp_extract"))


def test_incomplete_download_is_told_apart_from_a_rasterless_source(
        tmp_path, monkeypatch):
    """The old guard compared manifest rows to *.tif count, so a source whose
    every byte was downloaded but whose rows are zips/xml/shp looked exactly
    like an interrupted stage 04 and aborted the run."""
    manifest = tmp_path / "downloads.csv"
    present = tmp_path / "have.zip"
    present.write_text("x")
    manifest.write_text(
        "url,dest,size,source,tile,bbox\n"
        f"https://x/a,{present},1,eaarlb,,\n"
        f"https://x/b,{tmp_path / 'gone.tif'},1,s1m,,\n")
    monkeypatch.setattr(vrts.dp, "path",
                        lambda *parts: manifest if parts[-1] == "downloads.csv"
                        else Path(*parts))

    missing = vrts.missing_downloads()
    # eaarlb: every manifest file is on disk -> not an interrupted download
    assert missing.get("eaarlb", 0) == 0
    # s1m: a manifest file is absent -> stage 04 genuinely did not finish
    assert missing["s1m"] == 1


def test_tile_bounds_ignore_out_of_tile_manifest_rows(tmp_path, monkeypatch):
    """Single-tile test slice bug (Susquehanna 0205): stage 04 --tile fetched
    none of dem1m_projects' 51 basin-wide gap tiles because none overlap the
    test tile, and stage 05 -- with no tile filter -- reported all 51 as an
    interrupted download and aborted. Under tile bounds, out-of-tile rows must
    not count, but an in-tile row that is absent still must."""
    manifest = tmp_path / "downloads.csv"
    manifest.write_text(
        "url,dest,size,source,tile,bbox\n"
        # far from the tile: legitimately never downloaded
        f"https://x/a,{tmp_path / 'far.tif'},1,dem1m_projects,,-77.0 41.0 -76.9 41.1\n"
        # overlaps the tile but absent: a real incomplete download
        f"https://x/b,{tmp_path / 'near.tif'},1,s1m,,-76.20 39.60 -76.10 39.70\n")
    monkeypatch.setattr(vrts.dp, "path",
                        lambda *parts: manifest if parts[-1] == "downloads.csv"
                        else Path(*parts))
    tile = (-76.25, 39.5, -76.0, 39.75)     # n39x75_w076x25

    missing = vrts.missing_downloads(tile)
    assert missing.get("dem1m_projects", 0) == 0
    assert missing["s1m"] == 1

    # no bounds (make all): every manifest row is expected on disk
    assert vrts.missing_downloads()["dem1m_projects"] == 1
