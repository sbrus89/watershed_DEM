"""Shared helpers for the test suite.

Tests avoid importing pytest directly so they also run under the minimal
fallback runner (tests/run_tests.py) in environments without pytest; skips
use unittest.SkipTest, which pytest honors natively.
"""

from __future__ import annotations

import contextlib
import shutil
import unittest


def skip(reason: str):
    raise unittest.SkipTest(reason)


def importorskip(module: str):
    try:
        return __import__(module)
    except ImportError:
        raise unittest.SkipTest(f"requires {module}")


def skip_unless_cli(exe: str):
    if shutil.which(exe) is None:
        raise unittest.SkipTest(f"requires {exe} on PATH")


@contextlib.contextmanager
def raises(exc_type):
    """Minimal pytest.raises stand-in."""
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


def make_geotiff(path, array, transform, crs, nodata=None):
    """Write a small synthetic single-band GeoTIFF."""
    import numpy as np
    import rasterio

    array = np.asarray(array, dtype="float32")
    profile = dict(
        driver="GTiff", dtype="float32", count=1,
        width=array.shape[1], height=array.shape[0],
        transform=transform, crs=crs, nodata=nodata,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    return path
