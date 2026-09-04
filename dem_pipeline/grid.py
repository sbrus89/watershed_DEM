"""Tile lattice math for the CUDEM-compatible 1/9 arc-second grid.

Tile names follow the NCEI ninth-arc-second convention: ``n40x50_w075x25``
decodes to a 0.25 x 0.25 degree tile whose NORTH edge is 40.50 N and whose
WEST edge is 75.25 W, i.e. extent lon [-75.25, -75.00], lat [40.25, 40.50].

The nominal grid is exact: 1/9" = 1/32400 degree, 8100 x 8100 px per tile,
pixel edges on quarter-degree lines. Real ncei19 tiles may carry an overlap
collar or registration offset that varies by vintage, so the authoritative
spec is derived empirically by scripts/02_audit_cudem_grid.py and stored in
config/tile_grid.json; ``GridSpec`` reads that file. Where a CUDEM tile
exists, output geotransforms must match it bit-exactly.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import load_config, path as repo_path

CELLS_PER_DEG = 32400          # 1/9 arc-second
TILE_DEG = 0.25
TILE_PX = 8100                 # nominal, before any collar
PIXEL_DEG = 1.0 / CELLS_PER_DEG

_NAME_RE = re.compile(r"n(?P<nd>\d{2})x(?P<nf>\d{2})_w(?P<wd>\d{3})x(?P<wf>\d{2})")


def tile_name(north_edge: float, west_edge_lon: float) -> str:
    """Name for the tile with the given north edge (deg N) and west edge.

    ``west_edge_lon`` is the (negative) longitude of the west edge; the name
    encodes its absolute value, e.g. (40.50, -75.25) -> 'n40x50_w075x25'.
    """
    w = abs(west_edge_lon)
    nd, nf = int(north_edge), round((north_edge - int(north_edge)) * 100)
    wd, wf = int(w), round((w - int(w)) * 100)
    return f"n{nd:02d}x{nf:02d}_w{wd:03d}x{wf:02d}"


def parse_tile_name(name: str) -> tuple[float, float]:
    """Return (north_edge_deg, west_edge_lon) from a tile name.

    Accepts bare names ('n40x50_w075x25') or full CUDEM/output filenames
    containing one ('ncei19_n40x50_w075x25_2014v1.tif').
    """
    m = _NAME_RE.search(name)
    if not m:
        raise ValueError(f"no tile name found in {name!r}")
    north = int(m["nd"]) + int(m["nf"]) / 100.0
    west = -(int(m["wd"]) + int(m["wf"]) / 100.0)
    return north, west


def nominal_extent(name: str) -> tuple[float, float, float, float]:
    """(west, south, east, north) of the named tile, no collar."""
    north, west = parse_tile_name(name)
    return west, north - TILE_DEG, west + TILE_DEG, north


@dataclass
class GridSpec:
    """Target-grid spec, empirically derived from CUDEM tiles by stage 02."""

    epsg: int = 4269
    nodata: float = -999999.0
    pixel_deg: float = PIXEL_DEG
    collar_px: int = 0          # extra pixels on each edge beyond the 0.25 deg
    # Registration offset of the tile origin relative to the quarter-degree
    # line, in pixels (e.g. 0.5 for half-cell-shifted vintages). Applied to
    # both axes: origin = (west - collar*px + x_shift_px*px,
    #                      north + collar*px + y_shift_px*px).
    x_shift_px: float = 0.0
    y_shift_px: float = 0.0
    # Geotransforms of audited CUDEM tiles keyed by bare tile name; when a
    # tile appears here its geotransform is used verbatim (bit-exact match).
    cudem_geotransforms: dict = field(default_factory=dict)
    cudem_sizes: dict = field(default_factory=dict)

    @classmethod
    def load(cls, json_path: str | Path | None = None) -> "GridSpec":
        p = Path(json_path) if json_path else repo_path("config", "tile_grid.json")
        with open(p) as f:
            d = json.load(f)
        return cls(**d)

    def dump(self, json_path: str | Path | None = None) -> None:
        p = Path(json_path) if json_path else repo_path("config", "tile_grid.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    def tile_geotransform(self, name: str) -> tuple[tuple[float, ...], int, int]:
        """((gt0..gt5), width, height) for the named output tile."""
        if name in self.cudem_geotransforms:
            gt = tuple(self.cudem_geotransforms[name])
            w, h = self.cudem_sizes[name]
            return gt, w, h
        west, south, east, north = nominal_extent(name)
        px = self.pixel_deg
        origin_x = west - self.collar_px * px + self.x_shift_px * px
        origin_y = north + self.collar_px * px + self.y_shift_px * px
        n = TILE_PX + 2 * self.collar_px
        return (origin_x, px, 0.0, origin_y, 0.0, -px), n, n

    def tile_bounds(self, name: str) -> tuple[float, float, float, float]:
        """(west, south, east, north) actual raster bounds incl. collar."""
        gt, w, h = self.tile_geotransform(name)
        west, north = gt[0], gt[3]
        return west, north + h * gt[5], west + w * gt[1], north


def lattice_tiles(bounds: tuple[float, float, float, float]) -> list[str]:
    """All quarter-degree tile names whose nominal extent intersects bounds."""
    w, s, e, n = bounds
    names = []
    lat = math.floor(s / TILE_DEG) * TILE_DEG
    while lat < n:
        lon = math.floor(w / TILE_DEG) * TILE_DEG
        while lon < e:
            names.append(tile_name(lat + TILE_DEG, lon))
            lon += TILE_DEG
        lat += TILE_DEG
    return names


def output_filename(name: str, cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    proj = cfg["project"]
    return f"{proj['name']}_{name}_{proj['version']}.tif"
