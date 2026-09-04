"""Consistency checks between config/sources.yaml and the code that uses it."""

import dem_pipeline as dp
from dem_pipeline.composite import LAND_SOURCES, WATER_SOURCES


def test_config_loads():
    cfg = dp.load_config()
    assert cfg["project"]["huc4"] == "0204"
    assert cfg["target_grid"]["cells_per_degree"] == 32400


def test_stacks_match_config_roles_and_priorities():
    cfg = dp.load_config()
    land = sorted((c["priority"], n) for n, c in cfg["sources"].items()
                  if c["role"] == "land")
    water = sorted((c["priority"], n) for n, c in cfg["sources"].items()
                   if c["role"] == "water")
    assert [n for _, n in land] == LAND_SOURCES
    assert [n for _, n in water] == WATER_SOURCES


def test_every_source_has_dir_and_valid_resampling():
    cfg = dp.load_config()
    for name, scfg in cfg["sources"].items():
        assert scfg["resampling"] in ("near", "average", "bilinear"), name
        assert "dir" in scfg, name


def test_source_ids_unique_and_cover_all_sources():
    cfg = dp.load_config()
    ids = cfg["compositing"]["source_ids"]
    assert len(set(ids.values())) == len(ids)
    for name in list(cfg["sources"]) + ["nodata", "filled"]:
        assert name in ids, f"source_ids missing entry for {name}"


def test_coned_disabled_by_default():
    cfg = dp.load_config()
    assert cfg["sources"]["coned"]["enabled"] is False
