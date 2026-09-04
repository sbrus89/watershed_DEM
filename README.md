# Delaware Basin (HUC-4 0204) Seamless 1/9″ Topobathy DEM

Pipeline that aggregates the best available lidar topography and channel/
estuary bathymetry into a **CUDEM-drop-in-compatible** elevation dataset for
the entire Delaware River watershed plus the adjacent NJ Atlantic coastal
basins: 1/9 arc-second (~3 m) tiles on the CUDEM 0.25° lattice, NAD83
geographic, **NAVD88 meters**, named `dewatershed19_n40x50_w075x25_2026v1.tif`.

Built to feed the MPAS-Ocean/E3SM subgrid storm-surge workflow
(Brus et al. 2026, doi:10.1029/2026MS005842), which currently consumes NOAA
CUDEM 1/9″ tiles — the output uses the identical grid convention, verified
bit-exact against overlapping `ncei19_*` tiles.

Developer documentation: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
(design rationale, contracts, gotchas, open follow-ups) and
[CLAUDE.md](CLAUDE.md) (session-oriented operational summary).

## Data sources (priority order)

| Stack | Priority | Source | Notes |
|---|---|---|---|
| land | 1 | USGS Seamless 1-m DEM (S1M) | COGs, UTM 18N, NAVD88/GEOID18; averaged to 3 m |
| land | 2 | USGS project-based 1-m DEM | fills S1M rollout gaps |
| land | 3 | USGS 1/3″ seamless | last-resort fallback |
| water | 1 | USACE eHydro CENAP multibeam | MLLW ft → NAVD88 m via benchmark-verified VDatum separation |
| water | 2 | USGS EAARL-B Upper Delaware topobathy | channel bathy Hancock NY → Portland PA |
| water | 3 | NOAA CUDEM 1/9″ topobathy | Delaware Bay → Trenton |
| water | 4 | NJ/DE CoNED 1 m (2015) | optional backstop, `coned.enabled: false` |

Land/water composites are blended through a water mask (NHDPlus HR polygons
∪ survey footprints ∪ S1M-void∧CUDEM-valid) with a ~45 m distance-transform
feather, so no artificial steps enter subgrid lookup tables.

**Known limitation:** no published channel bathymetry exists for Portland PA
→ Trenton (~70 river miles) or non-tidal tributaries; those reaches carry
lidar water-surface elevations, flagged in the per-pixel `*_source.tif`
provenance rasters.

## Setup

```bash
mamba env create -f environment.yml
mamba activate de-dem
```

(The env includes `pip install -e .` for the `dem_pipeline` package.)

## Run

Always validate on one tile first (Philadelphia; downloads only that tile's
sources, a few GB):

```bash
bash scripts/run_test_slice.sh            # or: run_test_slice.sh n41x75_w075x00 for the EAARL-B reach
```

Inspect `data/final/tiles/` + `data/final/quicklooks/`, then run the basin:

```bash
make all        # boundary → audit → discover → download (~200 GB) → ... → QA
```

Every stage is idempotent (atomic writes, resumable downloads, `.done`
sentinels); rerun `make` after any interruption. Single-tile reruns:
`python scripts/08_composite_tile.py --tile n40x00_w075x25`.

Requires `jq` and GNU `parallel` for the Makefile composite stage
(`brew install jq parallel`), or run `scripts/08_composite_tile.py --all`.

## Tests

Unit tests (no network, no real data — synthetic rasters throughout):

```bash
make test        # pytest, falling back to python tests/run_tests.py
```

Coverage: tile lattice math and CUDEM-geotransform handling, the stage-02
grid audit (modal convention, collars, half-cell registration, wrong-pixel
rejection), the two-stack compositor end-to-end on a miniature tile
(feathering incl. degenerate all-water/all-land masks, intra-water survey
feathering, vector water mask + void exclusion, provenance, dynamic mask),
the VDatum gtx sign-convention auto-resolution (hard-fail path, benchmark
sampling location, CO-OPS payload parsing), downloader resume/skip/retry/
stale-partial behavior against a local HTTP server plus the real
fetch_manifest bbox/source filters, eHydro production unit/sign conversion +
plausibility gate + XYZ parsing + TIN gridding pinned to absolute
coordinates, TNM client caching/pagination/retries, and the QA checks
(conformity, seams incl. the collar branch, datum-bias detection). The suite
was hardened by an adversarial multi-agent review (mutation-style: every
assertion checked against a concrete production bug it must catch). Tests
avoid importing pytest directly, so `python tests/run_tests.py` works with
only the pipeline's own dependencies; tests needing an optional package
(scipy, rasterio, geopandas) skip cleanly when it is absent.

## Pipeline stages

| # | Script | Output |
|---|---|---|
| 01 | `01_get_boundary.py` | HUC-0204 polygon + 2 km buffer (`data/intermediate/huc4_0204.gpkg`) |
| 02 | `02_audit_cudem_grid.py` | basin CUDEM tiles; empirical grid spec (`config/tile_grid.json`); output tile list (`config/tiles_0204.geojson`) |
| 03 | `03_discover_tiles.py` | download manifest (`manifests/downloads.csv`) incl. S1M gap-fallback chain |
| 04 | `04_download.py` | all raw data (resumable, concurrent) |
| 05 | `05_build_source_vrts.py` | per-source VRTs |
| 06 | `06_process_ehydro.py` | survey grids in NAVD88 + `ehydro.vrt` (newest paints last) |
| 07 | `07_build_water_mask.py` | water-mask vectors (`water_mask.gpkg`) |
| 08 | `08_composite_tile.py` | composed tiles + provenance rasters (`data/intermediate/tiles/`) |
| 09 | `09_make_cogs_vrt.py` | final COGs + mosaic VRT (`data/final/`) |
| 10 | `10_qa_checks.py` | QA report, coverage CSV, hillshade quicklooks |

## QA gates

- Geotransforms bit-exact vs overlapping CUDEM tiles; exact lattice elsewhere.
- Cross-tile seam steps ≤ 5 m flag threshold (expect ~0).
- (ours − CUDEM) land-overlap |median| ≤ 0.10 m (datum sanity).
- eHydro MLLW→NAVD88 separation verified < 5 cm at CO-OPS stations
  Lewes 8557380, Reedy Point 8551910, Philadelphia 8545240 (hard-fail).
- Per-tile source-mix coverage report; hillshade quicklooks for visual review.

## Disk / time budget

~200–300 GB raw (S1M dominates), ~30 GB intermediates, ~10–15 GB final.
Roughly a day wall-clock including downloads on a fast connection.
