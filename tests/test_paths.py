"""Debris from an interrupted writer must never be consumed as product data.

Every raster writer in the pipeline writes <name>.tmp.tif and renames on
success, and stages 06/09/10 glob each other's output directories. A killed
`make all` therefore leaves truncated .tmp.tif files exactly where the next
stage looks, so the shared selector has to filter them out.
"""

import dem_pipeline as dp


def test_product_tifs_excludes_interrupted_writers(tmp_path):
    for name in ("n40x00_w075x25.tif", "n40x25_w075x25.tif",
                 "n40x00_w075x25_source.tif"):
        (tmp_path / name).touch()
    # a compose/translate killed mid-write: header on disk, data truncated
    (tmp_path / "n40x50_w075x25.tmp.tif").touch()
    (tmp_path / "n40x50_w075x25_source.tmp.tif").touch()

    got = [p.name for p in dp.product_tifs(tmp_path)]
    assert got == ["n40x00_w075x25.tif", "n40x00_w075x25_source.tif",
                   "n40x25_w075x25.tif"]


def test_product_tifs_honours_the_pattern(tmp_path):
    # stage 09's mosaic VRT and stage 10's QA both select final tiles by the
    # product-name prefix; a .tmp.tif sorts adjacent to the real tile and would
    # otherwise shadow it in stage 10's tile->path dict.
    (tmp_path / "dewatershed19_n40x00_w075x25_2026v1.tif").touch()
    (tmp_path / "dewatershed19_n40x00_w075x25_2026v1.tmp.tif").touch()
    (tmp_path / "ncei19_n40x00_w075x25_2018.tif").touch()

    got = [p.name for p in dp.product_tifs(tmp_path, "dewatershed19_*.tif")]
    assert got == ["dewatershed19_n40x00_w075x25_2026v1.tif"]
