"""Delaware basin (HUC-4 0204) seamless 1/9" topobathy DEM pipeline."""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every raster writer in the pipeline writes <name>.tmp.tif and atomically
# renames on success, so a file still carrying this suffix is debris from an
# interrupted run -- never product data.
TMP_TIF_SUFFIX = ".tmp.tif"


@functools.lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else REPO_ROOT / "config" / "sources.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def path(*parts: str) -> Path:
    """Resolve a repo-relative path."""
    return REPO_ROOT.joinpath(*parts)


def product_tifs(directory: str | Path, pattern: str = "*.tif") -> list[Path]:
    """Sorted rasters in *directory*, excluding interrupted writers' scratch
    files.

    Use this instead of a bare glob wherever a stage consumes another stage's
    output directory: a resumed run must not mistake a half-written
    ``<name>.tmp.tif`` for a finished survey grid or tile.
    """
    return sorted(p for p in Path(directory).glob(pattern)
                  if not p.name.endswith(TMP_TIF_SUFFIX))


def setup_logging(level: str = "INFO") -> None:
    import logging

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
