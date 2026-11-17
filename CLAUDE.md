# DE_watershed_DEM — Claude session guide

Pipeline that aggregates lidar topography + channel/estuary bathymetry into a
**CUDEM-drop-in-compatible** 1/9 arc-second topobathy DEM for the Delaware
River watershed (HUC-4 0204). Output feeds the user's MPAS-Ocean/E3SM subgrid
storm-surge workflow (Brus et al. 2026, doi:10.1029/2026MS005842), which
currently consumes NOAA CUDEM 1/9″ tiles via bilinear interpolation.

**Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing anything
substantive** — it explains the grid contract, compositing design, datum
handling, and every known gotcha.

## Current status (as of 2026-08-27)

- All infrastructure is written and unit-tested. **The pipeline has NEVER
  been executed** — no data has been downloaded. The `de-dem` env does now
  exist (built 2026-08-27, used only to run the test suite). The user
  explicitly asked to build infrastructure without running it.
- Consequence: everything that touches live services (TNM API, S3 buckets,
  eHydro AGOL, ScienceBase, CO-OPS, VDatum grid zips) is **best-effort
  untested against real payloads**. Expect first-contact fixes in
  `03_discover_tiles.py` (ScienceBase walk, eHydro fields), `06`'s GDB layer
  parsing, and `vdatum.py`'s gtx handling.
- Test suite: **69 pass, 0 fail** in the `de-dem` env under plain `pytest`
  (verified 2026-08-27). A bare env skips more: the 6 download tests need a
  bindable localhost socket, and the rest of the suite needs
  geopandas/rasterio/scipy. A few tests import rasterio directly in the test
  body rather than via `importorskip`, so in an env without it they fail
  rather than skip -- that is the environment, not a regression.
- The suite was hardened by an adversarial multi-agent review (all 10
  verified findings confirmed and fixed). Remaining accepted follow-ups are
  listed in ARCHITECTURE.md §9.
- Resume/interrupt safety is a tested contract as of 2026-08-27: see
  `tests/test_paths.py`, the two resumed-run tests in `tests/test_ehydro.py`,
  and the two rules in ARCHITECTURE.md §6 (never bare-glob another stage's
  output dir; rebuild combined products from per-item state on disk).

## Commands

```bash
mamba env create -f environment.yml && mamba activate de-dem  # once
pytest                          # or: python tests/run_tests.py (no pytest needed)
bash scripts/run_test_slice.sh  # REQUIRED first pipeline run: one Philly tile
make all                        # full basin (~200 GB download, ~1 day)
python scripts/viewer.py        # interactive QA viewer, http://127.0.0.1:8765/
python scripts/merge_region.py all --nproc 1   # region-wide merge of basin products (ARCHITECTURE §11)
python scripts/viewer.py --merge-config config/merge_region02.yaml --port 8778  # view the merged product
```

`scripts/viewer.py` (+ `scripts/viewer_static/`) is a local, read-only Leaflet
viewer over the tile lattice: click a tile to render its hillshade,
provenance, and colorized elevation straight from `data/final/tiles/<name>.tif`
and `_source.tif` (not the precomputed quicklook PNGs -- fresh per request,
cached under `data/final/quicklooks/.viewer_cache/`); toggle a basin-wide
depth (elevation) layer rendered as real georeferenced XYZ tiles windowed
from `data/final/dewatershed19_2026v1.vrt` on demand (this replaced an
`L.imageOverlay` of the static mosaic_topo.png, whose baked-in
title/colorbar/axes never registered correctly against the imagery). Binds
to 127.0.0.1 only; tunnel from a remote host with
`ssh -L 8765:localhost:8765 <host>` -- Perlmutter login
nodes (loginNN) are internal-only, so hop through the public name:
`ssh -J <user>@perlmutter.nersc.gov -L 8765:localhost:8765 <user>@loginNN`. Nothing here feeds the product.

Stages are idempotent (atomic writes, resumable downloads, `.done`
sentinels in `manifests/`). Rerun `make` after any interruption. Single-tile
rerun: `python scripts/08_composite_tile.py --tile n40x00_w075x25`.

## Hard invariants — do not break

1. **Grid contract**: output tiles must match overlapping CUDEM `ncei19_*`
   tiles **bit-exactly** in geotransform/size (the user's subgrid reader
   depends on it). The authoritative spec is `config/tile_grid.json`,
   *derived empirically* by stage 02 from real CUDEM tiles — never hard-code
   grid constants beyond `dem_pipeline/grid.py`'s nominal lattice.
2. **Vertical datum**: everything is NAVD88 meters. eHydro is the only
   exception (MLLW, feet, positive-down) and is converted through a
   benchmark-verified separation raster that **hard-fails** if CO-OPS
   benchmarks disagree by >5 cm. Never weaken `ImplausibleDepthsError` or
   the benchmark assertion — silent datum/unit errors are the primary
   failure mode this pipeline is designed to prevent.
3. **On land the newest 3DEP lidar wins; in water, bathymetry wins.** A pure
   priority stack is wrong (S1M holds valid water-surface elevations over
   the estuary and would delete bathymetry). All source combination goes
   through the masked two-stack + feather design in
   `dem_pipeline/composite.py`.
4. **No basin-scale intermediates**: sources stay as native-CRS VRTs; each
   tile warps windows in memory.

## Testing conventions

- Tests never `import pytest` (skips via `unittest.SkipTest`, helpers in
  `tests/util.py`: `raises`, `importorskip`, `skip`, `make_geotiff`), so the
  identical suite runs under both pytest and `tests/run_tests.py`.
- Tests must not touch the network or the real `config/` and `data/` trees:
  monkeypatch `<module>.load_config` and `<module>.repo_path` (each module
  imports them into its own namespace) plus `grid.repo_path` when
  `GridSpec.load()` is involved. Use tiny tiles by planting small
  `cudem_geotransforms`/`cudem_sizes` entries in a tmp `tile_grid.json` —
  never compose full 8100² tiles in tests.
- Numbered scripts are imported in tests via
  `importlib.util.spec_from_file_location` (see `tests/test_ehydro.py`).
- When adding tests, think mutation-style: name the concrete production bug
  the assertion would catch. Don't re-implement production logic in a test.

## Memory / plan pointers

- Approved implementation plan: `~/.claude/plans/mutable-bubbling-backus.md`
- The Claude Code sandbox in past sessions only allowed network to
  api.anthropic.com — pipeline downloads need user-run commands or sandbox
  host allowances.
