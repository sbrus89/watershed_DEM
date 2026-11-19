# Architecture and development guide

Detailed reference for continuing development of the Delaware basin
(HUC-4 0204) seamless 1/9″ topobathy DEM pipeline. Written for a developer
(or Claude session) with no prior context. Companion to the operational
summary in [CLAUDE.md](../CLAUDE.md) and the user-facing [README.md](../README.md).

Contents:
1. [Purpose and consumer](#1-purpose-and-consumer)
2. [The grid contract](#2-the-grid-contract)
3. [Data sources](#3-data-sources)
4. [Compositing design](#4-compositing-design)
5. [Vertical datum handling](#5-vertical-datum-handling)
6. [Pipeline stages](#6-pipeline-stages)
7. [Package reference](#7-package-reference)
8. [Testing](#8-testing)
9. [Known gaps, open follow-ups, first-run risks](#9-known-gaps-open-follow-ups-first-run-risks)
10. [History](#10-history)
11. [Region merge (multi-basin product)](#11-region-merge-multi-basin-product)
12. [Channel burn (optional, on the merged product)](#12-channel-burn-optional-on-the-merged-product)

---

## 1. Purpose and consumer

The user (S. R. Brus) models storm surge with MPAS-Ocean/E3SM using subgrid
corrections (doi:10.1029/2026MS005842). The subgrid preprocessor divides each
model cell into triangles and **bilinearly interpolates a high-resolution DEM**
at triangle centers to build wet-fraction lookup tables. Today it consumes
NOAA **CUDEM 1/9 arc-second** topobathy tiles (NAD83 geographic, NAVD88 m).
CUDEM only covers the coastal strip; expanding the model domain to the whole
Delaware watershed requires this pipeline.

The deliverable is therefore **not** "a nice DEM" but a *drop-in replacement
tile set*: same grid, same datum, same tiling convention, extended inland
with the best available lidar/bathymetry. Product name pattern:
`dewatershed19_<tile>_2026v1.tif` (e.g. `dewatershed19_n40x50_w075x25_2026v1.tif`).

Domain decision (user-confirmed): all of HUC-4 **0204** — the Delaware River
watershed proper (HUC-6 020401 + 020402) **plus** the NJ Atlantic coastal
basins (020403), because surge enters from the Atlantic side. Boundary comes
from the USGS WBD, buffered +2 km.

Consequences that shape everything below:

- Elevation errors that a human eye would never notice (a 0.5 m datum bias,
  a mirrored survey raster, a feather artifact in one corner) silently
  corrupt surge physics. Correctness gates are therefore aggressive and
  hard-failing.
- Steps/discontinuities in the DEM contaminate the subgrid lookup tables
  (they integrate elevation PDFs per cell), hence the feathering design.

## 2. The grid contract

### Nominal lattice

- 1/9 arc-second = **1/32400° exactly** (`grid.PIXEL_DEG`); 0.25° × 0.25°
  tiles = 8100 × 8100 px (`grid.TILE_PX`), pixel *edges* on quarter-degree
  lines; float32; nodata **−999999.0**; horizontal EPSG:4269 (NAD83
  geographic); vertical NAVD88 meters.
- Tile-name decode (NCEI convention, `grid.parse_tile_name`):
  `n40x50_w075x25` → **north edge** 40.50°N, **west edge** 75.25°W → extent
  lon [−75.25, −75.00], lat [40.25, 40.50]. Names parse out of any filename
  containing the pattern (`ncei19_*`, `dewatershed19_*`).

### Why the spec is *derived*, not hard-coded

Real `ncei19` tiles vary by vintage: some carry an overlap **collar** (extra
pixels beyond the quarter-degree), some a **half-cell registration shift**
(pixel centers, not edges, on the lattice). Stage 02
(`scripts/02_audit_cudem_grid.py::audit_tiles`) inspects every downloaded
CUDEM tile and writes the empirical spec to **`config/tile_grid.json`**:

```
epsg, nodata, pixel_deg, collar_px, x_shift_px, y_shift_px,
cudem_geotransforms{name: gt6}, cudem_sizes{name: [w, h]}
```

`grid.GridSpec.tile_geotransform(name)` resolves a tile's exact grid:

- If the tile overlaps a CUDEM tile → **that tile's geotransform verbatim**
  (bit-exact; QA asserts equality).
- Otherwise → the **modal convention** (collar/shift) extended on the
  nominal lattice.

`audit_tiles` hard-fails (SystemExit) on any pixel size ≠ 1/32400°, and
warns (but proceeds with the modal convention) if CUDEM tiles disagree among
themselves. Pixel-size equality uses exact float comparison against
`1/32400` — safe because GeoTIFF geotransforms are doubles and round-trip
exactly; do not "fix" this with a sloppy tolerance.

The output tile list (`config/tiles_0204.geojson`, also from stage 02) is
the 0.25° lattice ∩ buffered basin, unioned with all basin CUDEM tiles
(~90–110 tiles expected).

## 3. Data sources

All endpoints were verified live in Aug 2026 (session research); volumes are
estimates. Everything is configured in `config/sources.yaml` — code reads
URLs/priorities/flags from there, never inline.

### Land stack (priority order — first valid wins)

| # | Source | Details |
|---|---|---|
| 1 | **USGS Seamless 1-m DEM (S1M)** | New product (mid-2025). COGs, 10 km tiles, NAD83(2011) UTM 18N (whole basin is zone 18), NAVD88/GEOID18. TNM dataset tag `Seamless 1-m DEM (S1M)`; S3 `prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/S1M/<n18e16>/<n1800e1600>/S1M_*_<date>.tif`. ~965 tiles in the basin bbox, 150–250 GB. Rollout still in progress → gaps possible. |
| 2 | **Project-based 1-m DEM** | TNM tag `Digital Elevation Model (DEM) 1 meter`; fills S1M gaps; `.../Elevation/1m/Projects/<Project>/TIFF/`. |
| 3 | **1/3″ seamless** | Last resort; 1° tiles at `.../Elevation/13/TIFF/current/n<lat>w<lon>/USGS_13_*.tif`. |

### Water stack (priority order)

| # | Source | Details |
|---|---|---|
| 1 | **USACE eHydro (CENAP)** | Navigation-channel multibeam, 2022–2025. AGOL FeatureServer query (`usacedistrictcode='CENAP'`) → ZIP per survey (GDB SurveyPoint/outline or XYZ text). **MLLW, feet, positive-down** → converted (see §5), TIN-gridded at 3 m UTM 18N, clipped to survey hull. Newest survey wins where overlapping. |
| 1b | **`nos_grids` — pre-converted NAVD88 survey rasters** (optional, added 2026-11-13) | Explicit list in `sources.nos_grids.items` (`url`, `bbox` [W,S,E,N], `name`); basins that omit the source are unaffected (compositor/stage 05 skip unconfigured sources). First use: Hudson (0202/0203) — nine NOAA multibeam surveys 2015–18 (H12872/3/4, F00658, H13022/3, H13092, D00211, H13138), 0.5 m GeoTIFF zips that NYSDEC converted ellipsoid→**NAVD88 m, positive up** and hosts (index: DEC "Benthic_Data_Footprints" FeatureServer). No conversion of ours, so stage 05 checks every raster: values inside `plausible_range_m` **and a negative median** (a positive-down shallow survey passes a range check alone). Ranked below eHydro (newer channel surveys), above CUDEM. Fills the Newburgh→Troy reach that has no CUDEM or BlueTopo. Provenance id 15. |
| 2 | **USGS EAARL-B Upper Delaware topobathy v1** (DOI 10.5066/P97031JU) | Green-lidar channel bathymetry, Hancock NY → Portland PA only. Ships **lidar-void polygons** (deep/turbid pools) which must be excluded from the water mask. Discovered via ScienceBase DOI walk. |
| 3 | **NOAA CUDEM 1/9″** (dataset 8483) | Delaware Bay → ~Trenton, mostly 2014/2018 vintage. S3 `noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/{northeast_sandy,chesapeake_bay}/`; tile index + urllist at `coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/`. Doubles as the grid template (§2). |
| 4 | **NJ/DE CoNED 1 m topobathy (2015)** | `coned.enabled: false` by default (95 GB, sources 1888–2014, footprint stops at 41.2°N). Backstop only if QA shows water gaps. The user explicitly evaluated and rejected CoNED as a primary source. |

### Support data

- **WBD** region-02 GPKG (basin boundary), **NHDPlus HR** 0204 GDB (water
  mask polygons: NHDArea FTypes 460/445/336, NHDWaterbody 390/436/493 ≥1 ha).
- **VDatum regional grids** (`vdatum_DEdelbay.zip`: `tss.gtx`, `mllw.gtx`)
  and **CO-OPS metadata API** for datum benchmarks (§5).

### Known coverage gap (accepted, documented)

No published channel bathymetry exists for **Portland PA → Trenton (~70
river miles)** or any non-tidal tributary — those reaches carry the lidar
water-surface elevation, identifiable in the provenance rasters. Do not
"fix" this by interpolation; the user accepted it as a documented limitation.

## 4. Compositing design

Core module: `dem_pipeline/composite.py`. Per output tile:

```
land  = first_valid(S1M→avg3m, project-1m, 1/3″)                # hard priority
water = feathered_first_valid(eHydro, EAARL-B, CUDEM, CoNED)    # feathered priority
W     = water mask (below)
alpha = feather_alpha(W, feather_px=15)     # EDT ramp, ~45 m wide
z     = alpha·water + (1−alpha)·land        # hard switch where one side NaN
```

### Why not simpler designs (do not regress to these)

- **Pure priority stack**: S1M holds *valid* water-surface elevations over
  the estuary; "newest wins" would delete all bathymetry.
- **Elevation-threshold water detection** (e.g. CUDEM < 0): fails in tidal
  flats, subsided marsh, and diked land — much of the Delaware shoreline
  sits below 0 m NAVD88.
- **No feather**: every source seam becomes a step; the subgrid lookup
  tables integrate elevation PDFs and inherit artificial cliffs.

### The water mask (per tile, on the exact tile grid)

Union of:
1. NHDPlus HR polygons (rasterized from `water_mask.gpkg` layer `water`),
2. eHydro survey footprints + EAARL-B tile footprints (layers
   `ehydro_footprint`, `eaarlb_footprint`) **minus** EAARL-B void polygons
   (layer `eaarlb_voids`),
3. dynamic term: `isnan(S1M) & isfinite(CUDEM)` (catches hydro-flattened /
   void areas NHD misses),
then `&= isfinite(water)` — the mask can only assert water where the water
stack actually has data.

### Feathering details

- `feather_alpha(mask, f)`: signed-EDT ramp, alpha 0 deep on land → 1 deep
  in water over ±f px. **Degenerate masks short-circuit** (all-water → 1.0,
  all-land → 0.0): the EDT of a uniform mask measures distance to the array
  corner and would blend land into open water there (bug found by
  adversarial review — keep the guard).
- Hard switches where one composite is NaN: `alpha[isnan(water)]=0`,
  `alpha[isnan(land)]=1`.
- **Within the water stack**, higher-priority surveys are feathered in from
  their footprint edges (`feather_overlay` / `feathered_first_valid`,
  `survey_feather_px=8`): weight rises 0→1 over f px *inside* the overlay's
  valid footprint, so a freshly dredged eHydro channel doesn't step against
  2014 CUDEM. Where the base is missing, the overlay applies at full
  strength immediately.

### Outputs per tile

- `data/intermediate/tiles/<name>.tif` — float32 elevation, deflate, tiled.
- `data/intermediate/tiles/<name>_source.tif` — uint8 **provenance raster**
  (`compositing.source_ids` in config: s1m=1, dem1m=2, dem13=3, ehydro=11,
  eaarlb=12, cudem19=13, coned=14, nos_grids=15, filled=99, nodata=0). In the feather band
  the id is whichever side has alpha ≥ 0.5. Residual interior holes are
  filled with `rasterio.fill.fillnodata` (max 50 px) and flagged `filled`.
- Stage 09 converts to COG with the final `dewatershed19_*` name.

### Reprojection strategy

Sources stay as native-CRS VRTs (`data/intermediate/vrt/<source>.vrt`);
`warp_to_tile` uses a rasterio `WarpedVRT` window read per tile — **no
basin-scale warped mosaic is ever materialized**. Resampling per source
(config `resampling`): `average` for 1-m sources (area-mean is the right
operator for subgrid statistics), `near` for CUDEM (already on the target
grid), `bilinear` for eHydro grids and 1/3″.

The eHydro VRT is built with inputs sorted **oldest-first** (VRT sources
paint in order, so the newest survey paints last) — see
`06_process_ehydro.py`'s `survey_date` sort. Preserve this if touching VRT
construction.

## 5. Vertical datum handling

Everything is NAVD88 meters except eHydro. GEOID12B-vs-GEOID18 mixing among
lidar vintages is ~1–2 cm in this region — an order below lidar RMSE and
surge-model skill — deliberately **not corrected**, but quantified by QA §3
(land-overlap histograms vs CUDEM).

### eHydro: MLLW feet → NAVD88 meters

Two independently dangerous conversions, each with its own guard:

1. **Units/sign** (`scripts/06_process_ehydro.py::convert_raw_depths`):
   CENAP convention is depth in feet, positive down →
   `z_mllw_m = −depth_ft × 0.3048`. Per-survey overrides live in
   `manifests/ehydro_overrides.yaml`
   (`<survey_stem>: {depth_units: meters, depth_positive_down: false, crs: EPSG:xxxx}`).
   After the full conversion chain, `check_plausible` asserts the NAVD88
   range is within [−35, +5] m; violation raises `ImplausibleDepthsError`
   which is **re-raised past the per-survey exception handler** — a
   units/sign mistake must abort the run, never be skipped. (A survey that
   legitimately violates the window gets an explicit override, not a wider
   window.)
2. **Datum** (`dem_pipeline/vdatum.py`): a separation raster
   `sep_mllw_minus_navd88.tif` = elevation of the MLLW surface in the NAVD88
   frame (≈ −0.5 to −0.9 m in this estuary), applied as
   `z_navd88 = z_mllw + sep`. Built by composing VDatum's `tss.gtx` and
   `mllw.gtx`. Their sign conventions are easy to get backwards, so
   `build_separation_raster` evaluates **all four sign combinations** at
   CO-OPS benchmark stations (Lewes 8557380, Reedy Point 8551910,
   Philadelphia 8545240) and keeps the one reproducing the published datums;
   it hard-fails (RuntimeError) if none fits within 5 cm.

**CO-OPS API trap** (real bug caught by review): the datums endpoint
(`.../stations/{id}/datums.json`) does NOT carry station coordinates — its
top-level `LAT` field is *Lowest Astronomical Tide*. Coordinates come from a
second request to `.../stations/{id}.json` (`stations[0].lat/lng`).
Benchmark truth = `datums[MLLW] − datums[NAVD88]` (datum values are heights
above the station datum).

## 6. Pipeline stages

Orchestrated by the Makefile (sentinels `manifests/<stage>.done`; the
composite stage uses `jq` + GNU `parallel -j6`). Every script is
independently rerunnable and takes `--tile` where meaningful. Idempotency
pattern used everywhere: write to a temp name → atomic rename; skip if the
product exists; downloads resume via HTTP Range.

Two rules make that safe across an interrupted `make all`:

- **Never bare-glob another stage's output directory** — use
  `dem_pipeline.product_tifs()`. A killed writer leaves `<name>.tmp.tif`
  debris exactly where the next stage looks, and stages 06/09/10 would
  otherwise pull a truncated raster into the eHydro VRT, the mosaic VRT, or
  the QA run. (`data/raw/` is exempt: the downloader only ever leaves
  `.part` files, which no glob matches.)
- **Derive a combined product from per-item state on disk, not from the
  items this run happened to process.** A resumed stage skips what it
  already made, so rebuilding a basin-wide artifact from just the loop's
  results silently truncates it. Stage 06 does this via
  `<survey>.hull.geojson` sidecars written beside each survey grid:
  `collect_footprints()` reassembles the full `ehydro_footprints.gpkg`
  every run. A grid without its sidecar is reprocessed — losing a footprint
  drops that channel from stage 07's water mask, which the compositor reads
  as land.

| Stage | Script | Reads | Writes |
|---|---|---|---|
| 01 boundary | `01_get_boundary.py` | WBD region-02 GPKG (downloads) | `data/intermediate/huc4_0204.gpkg` (layers `huc4`, `huc4_buffered`) |
| 02 audit | `02_audit_cudem_grid.py` | CUDEM urllist + tiles (downloads ~25, 4–5 GB) | `config/tile_grid.json`, `config/tiles_0204.geojson`, `manifests/cudem19.csv` |
| 03 discover | `03_discover_tiles.py` | TNM API (polygon query w/ bbox fallback, cached in `manifests/cache/`), eHydro AGOL, ScienceBase | `manifests/downloads.csv` (columns url,dest,size,source,tile,bbox) |
| 04 download | `04_download.py` | downloads.csv | `data/raw/<source>/...` (8–12 workers; `--sources`, `--tile` filters) |
| 05 vrts | `05_build_source_vrts.py` | raw tifs | `data/intermediate/vrt/<source>.vrt` (not ehydro) |
| 06 ehydro | `06_process_ehydro.py` | survey ZIPs, VDatum grids, CO-OPS API | per-survey 3 m grids + `<survey>.hull.geojson` sidecars in `data/intermediate/ehydro_grids/`, `ehydro.vrt` (oldest-first), `ehydro_footprints.gpkg`, separation raster |
| 07 water mask | `07_build_water_mask.py` | NHDPlus HR GDB, footprints, EAARL rasters/voids | `data/intermediate/water_mask/water_mask.gpkg` (layers `water`, `ehydro_footprint`, `eaarlb_footprint`, `eaarlb_voids`) |
| 08 composite | `08_composite_tile.py` (`--tile`/`--all`) | VRTs + mask + tile_grid.json | `data/intermediate/tiles/<name>{,_source}.tif` |
| 09 cog | `09_make_cogs_vrt.py` | intermediate tiles | `data/final/tiles/dewatershed19_*.tif` (COG, DEFLATE, PREDICTOR=3), mosaic VRT + overviews |
| 10 qa | `10_qa_checks.py` | final tiles + CUDEM + provenance | `data/final/qa/qa_report.json`, `coverage.csv`, hillshade quicklooks. Exits nonzero on hard-check failure. |
| 11 overview | `plot_overview.py` | mosaic VRT, `tiles_0204.geojson`, `coverage.csv` (optional) | `data/final/quicklooks/dewatershed19_*_mosaic_topo.png`, `..._tiles_on_imagery.png`. Inspection only, feeds nothing downstream; fetches an Esri World Imagery basemap (cached under `.basemap_cache/`, offline on rerun). |

Stage-03 gap-fallback logic: query S1M → union footprints → subtract from
basin → query project 1-m for gap polygons → any residual gets 1/3″ 1° tiles.

**Always run the test slice first**: `bash scripts/run_test_slice.sh`
(default tile `n40x00_w075x25`, Philadelphia — exercises S1M land, CUDEM
tidal river, eHydro surveys, NHD shoreline, hydro-flattening; downloads only
files whose manifest bbox intersects the tile). Second suggested slice:
`n41x75_w075x00` (Hancock reach) for EAARL-B + voids.

### QA gates (stage 10)

1. Geotransform conformity — bit-exact vs `GridSpec` (and thus vs CUDEM).
2. Seam continuity — facing edge rows/cols (or overlapping collar pixels,
   which must agree exactly) between neighbors; flag > 5 m.
3. Land-overlap vs CUDEM — |median(ours − CUDEM)| ≤ 0.10 m over land
   (>1 m in both); larger bias ⇒ datum handling error.
4. Basin nodata — nodata pixels inside the HUC-4 polygon (shrunk inward by
   `qa.basin_nodata_edge_buffer_m`, default 100 m) must not exceed
   `qa.basin_nodata_max_px` (default 0). Added 2026-11-16 after a 37k-px
   quarry hole passed checks 1–3.
5. Coverage/provenance report per tile; hillshade PNG quicklooks.

## 7. Package reference

`dem_pipeline/` (installed via `pip install -e .`, part of the conda env):

- `__init__.py` — `load_config()` (lru_cached YAML), `path()` (repo-relative
  → absolute; imported into modules as `repo_path`), `setup_logging()`.
  **Pattern**: modules do `from . import load_config, path as repo_path`,
  so tests monkeypatch `<module>.load_config` / `<module>.repo_path`
  per-module. If you add a module, follow the same import style or the test
  sandboxing breaks.
- `grid.py` — pure lattice math + `GridSpec`. No I/O beyond
  `GridSpec.load/dump`. Heavily unit-tested; keep it dependency-light.
- `tnm.py` — TNM API client. Responses cached to `manifests/cache/tnm_*.json`
  keyed by sha256(url+params) — delete cache to force re-discovery.
  `simplify_polygon_for_api` shrinks a shapely ring to fit URL limits
  (returns None → caller falls back to bbox + local filter).
- `download.py` — `fetch` (Range resume, `.part` + atomic rename, 416 branch
  verifies size before accepting a partial as complete),
  `fetch_manifest` (ThreadPool, `sources`/`bounds` filters — bbox-less rows
  always pass the bounds filter), manifest read/write
  (`MANIFEST_FIELDS = url,dest,size,source,tile,bbox`).
- `vdatum.py` — §5. Public: `station_mllw_in_navd88`,
  `download_vdatum_grids`, `build_separation_raster`; `_read_gtx_pair`
  warps mllw.gtx onto the tss.gtx grid and NaNs |v|>100 sentinels.
- `composite.py` — §4. Public pieces are individually testable:
  `first_valid`, `feather_alpha`, `feather_overlay`, `feathered_first_valid`,
  `warp_to_tile`, `compose_tile`. `LAND_SOURCES`/`WATER_SOURCES` order must
  match the `priority` fields in config (asserted by `test_config.py`).
- `qa.py` — stage-10 checks; each returns a dict with a `pass` bool where
  applicable.

Adding a new elevation source = config entry (role/priority/dir/resampling +
a `source_ids` entry) + discovery rows in stage 03 + (only if it needs
preprocessing like eHydro) a processing stage. Compositor and VRT stages are
config-driven and should not need changes.

## 8. Testing

Two runners, one suite:

- `pytest` (in the de-dem env; config in `pyproject.toml`).
- `python tests/run_tests.py [substring…]` — ~120-line stdlib fallback
  providing pytest-compatible `tmp_path`/`monkeypatch`; exists because the
  miniforge base env lacks pytest. It survives module-import failures
  (counts them as FAIL and continues).

Conventions (enforced by construction — keep them):

- **No `import pytest` in test modules.** Skips raise `unittest.SkipTest`
  via `tests/util.py` helpers (`skip`, `importorskip`, `skip_unless_cli`);
  pytest honors SkipTest natively. `raises()` is a local contextmanager.
- **Full isolation from the repo**: patch `<module>.load_config` with a
  minimal dict and `<module>.repo_path`/`grid.repo_path` onto `tmp_path`;
  networkless by patching `<module>.requests.get` or running a local
  Range-supporting HTTP server (`tests/test_download.py::_serve`, which
  skips where sockets can't bind, e.g. the Claude Code sandbox).
- **Small synthetic rasters**: plant tiny `cudem_geotransforms` entries
  (e.g. 90×90 at 0.25°/90 pixels) in a tmp `tile_grid.json` so
  `compose_tile`/QA run in milliseconds. `tests/util.py::make_geotiff`
  writes one-band float32 GeoTIFFs. GeoTIFFs named `.vrt` are fine — GDAL
  identifies drivers by content (used to satisfy `source_vrt`).
- **Mutation mindset**: every assertion should catch a nameable production
  bug. Anti-patterns already purged once by adversarial review — do not
  reintroduce: re-implementing production logic in the test and asserting
  on the copy; symmetric/y-invariant synthetic data (mirror bugs pass);
  range-only assertions where absolute-coordinate sampling is possible.
- Env-limited skips are expected in the base env: 6 socket-binding download
  tests + 1 geopandas compositor test. Everything runs in de-dem.

## 9. Known gaps, open follow-ups, first-run risks

Accepted product limitations (do not "fix" silently):
- No channel bathymetry Portland PA→Trenton and non-tidal tributaries
  (lidar water surface, flagged in provenance).
- EAARL-B deep/turbid voids fall through to lower-priority sources or the
  nodata fill (flagged `filled`).
- Estuary bathymetry is ~3 m CUDEM except where eHydro/EAARL-B are better.
- **EAARL-B land topo overrides newer 3DEP on the Upper Delaware floodplain
  (found 2026-11-18, left as is by user decision).** The stage-07 water mask
  is NHD polygons ∪ survey footprints, and `eaarlb_footprint` is the whole
  EAARL-B tile footprint. The EAARL-B raster also carries merged 2008 PAMAP
  land topo off-channel, so inside that footprint the water stack's EAARL-B
  (provenance 12) wins over the newer 3DEP land stack well beyond the river:
  on `n42x00_w075x25` ~17.7M of its 18.3M EAARL-B pixels lie outside the NHD
  `water` polygons (`n41x00_w075x25`: 4.4M of 4.6M). This contradicts
  invariant 3 ("on land the newest 3DEP lidar wins") for that swath. Seen as
  EAARL above the 3DEP surface on tributaries (e.g. Brodhead Creek). Fix
  options when revisited: restrict the EAARL-B water mask to NHD polygons or
  to its void-free wetted area, or accept EAARL only where it is below the
  3DEP surface. A fix changes the merged product, so the burned product
  (§12) must be rerun afterwards (fingerprints rebuild only affected tiles).
- **`check_seams` cross-tile edges where both sides' worst-disagreement
  pixel is `cudem19` provenance** — an inherent CUDEM-vintage registration
  mismatch, not a pipeline bug. The grid contract (§2) requires each tile to
  reproduce its overlapping audited CUDEM tile's geotransform verbatim
  (`resampling: near`, no blending, to stay bit-exact); two real adjacent
  CUDEM tiles surveyed at different times aren't necessarily mutually
  pixel-aligned, so the "same" boundary location can sample genuinely
  different source pixels on each side, and over a steep real
  seafloor/channel gradient that becomes a multi-meter step. Same root
  cause as the `size_mismatches` edges the check already reports rather
  than papering over. `check_seams` auto-classifies this as
  `inherent_misalignment` (reported, not gating) whenever both sides trace
  to the identical `cudem19` source id — deliberately narrow, since a
  mixed-source edge is never auto-accepted this way (see the open
  follow-up below for why). As of 2026-09-08's full-basin run this covered
  14 of the 21 then-flagged edges plus the 2 known `size_mismatch` edges.
  One further edge — a `cudem19`/`dem13` pairing at a coastal boundary
  (specific tile pair not recorded) — was reviewed manually and found to
  be the same registration-drift mechanism despite the mixed source id;
  accepted, but `check_seams` still reports it as `flagged` since the code
  doesn't special-case it.
- **Window-dependent void fill (0202 `n43x50_w073x25|n43x50_w073x00`,
  ONE pixel, 5.37 m; accepted by the user 2026-11-16).** Both sides are
  provenance `filled` inside a ~120-px lidar void crossing the seam. Not
  the padded-window fill fix failing: recomposing with pad 100 instead of
  50 gives the identical 5.37 m. The pre-fill inputs in the 12-px overlap
  differ by exactly one land pixel on the void's edge, valid in one tile's
  warp and nodata in the other's -- `WarpedVRT` `average` resampling is not
  perfectly window-invariant (elsewhere in the overlap it shows only as
  ≤0.06 m). The fill (radius `fill_max_px` = 50) then sees that pixel from
  one tile only, and the filled value 50 px away differs. Making the warp
  window-invariant would touch every tile in every basin for one pixel.
- **Below-sea-level land is real.** `LAND_PLAUSIBLE_MIN` is -200 m plus a
  2-px halo rule (see `composite.implausible_land_mask`). Until 2026-11-16
  it was -15 m, which deleted aggregate-quarry pit floors in every basin
  (Clinton Point NY -84 m, Richmond VA -99 m, King of Prussia PA -73 m,
  Occoquan, Travilah, Manassas, ...): all outside the water mask, all
  mapped consistently by S1M and dem13. S1M sentinel-blend corruption
  (~-300000 m) is still caught by the floor, and its only moderate values
  are a 1-px partial-blend halo around those pixels (verified on 8
  Delaware coastal tiles), which the halo rule masks.

Open follow-ups from the adversarial review (unverified-tier, real but not
gating):
- `warp_to_tile`'s `average`/`bilinear` paths and metadata-less-nodata
  handling have only light test coverage.
- `tests/test_ehydro.py` and `tests/test_audit.py` execute the numbered
  scripts' module level at import/collection time (harmless today — scripts
  only define things — but a future module-level side effect would run
  during collection).
- `run_tests.py` still imports all test modules even when a name filter is
  given, and its monkeypatch emulation is a subset of pytest's.
- `qa.hillshade_quicklook` shells out to `gdaldem` and is untested.
- **One `check_seams` edge is a real, undiagnosed bug wearing the same
  disguise as the accepted case above**: `n39x25_w074x75|n39x25_w074x50`
  (`cudem19`/`dem13`, 19.45 m step) has the same mixed source-id signature
  as the accepted coastal edge just above, but manual review
  could not attribute it to the same registration-drift mechanism. This is
  exactly why `check_seams` never auto-accepts a mixed-source edge as
  `inherent_misalignment` — same source ids, opposite verdict, so
  provenance alone can't disambiguate them. Not yet root-caused.
- `cudem_land_overlap` also fails (worst |median| 0.256 m vs the 0.10 m
  threshold) — unrelated to the seams work above, signs are mixed across
  tiles (suggests ordinary cross-vintage LIDAR-vs-CUDEM survey
  disagreement rather than one systematic datum/geoid bug), and not yet
  investigated in depth.

First-run risks (nothing network-facing has met real payloads):
- Stage 03: ScienceBase DOI walk for EAARL-B (item tree shape assumed);
  eHydro AGOL field names (`sourcedatalocation`, `surveydateend`) and
  GeoJSON pagination flags; TNM `polygon=` acceptance of the simplified ring.
- Stage 06: GDB layer/field names for SurveyPoint z (`z_depth`/`depth`/…);
  XYZ column order assumption (x, y, z); CRS of XYZ via `.prj` sidecar.
- `vdatum.py`: exact contents/paths inside `vdatum_DEdelbay.zip` (code
  rglobs for `tss.gtx`/`mllw.gtx` anywhere under the extract) and whether
  GDAL reads those gtx grids with a usable CRS.
- S1M coverage may have holes the fallback chain must actually fill — check
  the stage-10 coverage report on the first basin run.
- Disk: ~300 GB raw peak; the Makefile composite stage needs `jq` and GNU
  `parallel` (`brew install jq parallel`) or use `08_composite_tile.py --all`.

## 10. History

- **2026-08-20**: research + plan (approved; archived at
  `~/.claude/plans/mutable-bubbling-backus.md`). Key user decisions: full
  HUC-0204 domain; 1/9″ CUDEM-style output (not 1 m UTM); eHydro + EAARL-B
  bathymetry fusion; CoNED evaluated and rejected as primary (stops at
  41.2°N, sources ≤2014). Infrastructure built the same day; **pipeline
  intentionally never executed**.
- **2026-08-20/21**: test suite added; adversarial multi-agent review (4
  attack dimensions → 10 skeptic-verified findings, all confirmed, all
  fixed). Production bugs found: CO-OPS lon/lat KeyError + `LAT` trap;
  `feather_alpha` degenerate-mask corner artifact; plausibility error
  swallowed by the per-survey handler; `fetch` 416 accepting stale partials;
  unreachable eHydro CRS override; `_read_mask_geoms` swallowing all read
  errors; `survey_feather_px` configured but unimplemented (now
  `feathered_first_valid`). Test bugs: self-verifying conversion test,
  duplicated manifest-filter logic, y-invariant gridding plane, vacuous seam
  assertion, live-config coupling in tnm/download tests.
- **2026-08-27**: interrupt/resume audit of `make all`. Downloads were already
  safe (Range resume + `.part`/atomic rename), but two resume bugs were fixed:
  (1) `.tmp.tif` debris from killed writers was globbed as product data by
  stages 06, 09 and 10 — now filtered through `dem_pipeline.product_tifs()`;
  worst case was stage 10, where a `.tmp.tif` sorts *after* the real tile and
  shadowed it in the QA tile→path dict. (2) A resumed stage 06 rebuilt
  `ehydro_footprints.gpkg` from only the surveys it processed that run,
  truncating the layer and shrinking stage 07's water mask — footprints are now
  persisted as `<survey>.hull.geojson` sidecars and reassembled from disk.
  Also made stage 09's provenance copy atomic and ordered it before the
  elevation rename, so the `dst.exists()` resume check can't skip a
  half-written pair. Tests: `tests/test_paths.py` (new) plus two resumed-run
  tests in `tests/test_ehydro.py`; suite 69 pass under `de-dem`. The `de-dem`
  env was created for the first time; **the pipeline itself is still unrun**.

## 11. Region merge (multi-basin product)

Each HUC-4 basin runs as its own product in its own git worktree (branch off
`master`, own env, own `config/sources.yaml`). `scripts/merge_region.py` +
`dem_pipeline/merge.py`, driven by `config/merge_region02.yaml`, combine the
finished basin tile sets into one region product (`midatlantic19_<tile>_2026v1`
for HUC-2 region 02, written to `output.root`, never into a basin checkout).

```bash
python scripts/merge_region.py all --nproc 1      # prepare, tiles, vrt, qa
python scripts/merge_region.py tiles --tile n39x75_w076x00   # one tile
```

**Why a merge is not a copy.** Neighbouring basins produce the same 0.25° tile
wherever their +2 km buffered domains overlap (region 02: 731 basin tiles →
566 footprints; 141 shared by 2 basins, 12 by 3). Measured 2026-11-17, the
versions differ for three reasons: (1) beyond its data edge a basin has only
50-px `fillnodata` extrapolation (provenance 99), tens of metres wrong;
(2) shared estuaries were composed from each basin's own eHydro/water mask
(1–20 m apart at p99 — C&D Canal, Raritan/NY Harbor, Potomac mouth);
(3) ocean pixels outside every HUC-4 polygon.

**Design (per pixel, finished tiles only — no recomposition):**

- *Territory*: a basin owns its unbuffered WBD HUC-4 polygon; outside all
  polygons, the basin with the nearest polygon (Voronoi of densified boundary
  points, built once in EPSG:5070 by `build_territories`, so adjacent tiles
  agree). Polygons are kept separate from their reprojected exterior pieces and
  burned last (`rasterize_territories`) — unioning them trips GEOS topology
  errors.
- *Feather*: weight = `composite.feather_alpha(territory, feather_px=30)`
  computed on a window padded by `feather_px+2` so overlapping collars match.
- *Real data first*: weight is zeroed where a version is nodata or has
  provenance `filled`/`nodata`. No weight left → the real version with the
  largest territory weight (tier 2), else any finite version (tier 3).
  Ties break on config basin order.
- Outputs per tile: elevation COG (stage-09 COG settings), `_source.tif`
  (provenance of the dominant version) and `_basin.tif` (uint8 HUC-4 % 100,
  e.g. 4 = 0204; 0 = nodata).
- *Single-basin tiles* are byte-copied when their grid already equals the
  merged spec, else **restamped** (same pixels, new geotransform; COG read
  back and compared).

**Grid.** `prepare` unions every basin's audited CUDEM geotransforms (any
disagreement is fatal; region 02 has 144, zero conflicts) and takes the
non-CUDEM collar/shift convention from `merge.grid_convention_from`. The two
Hudson basins (0202/0203) audited a newer CUDEM vintage (EPSG:5498) with a
different modal shift (−0.00048/+0.00048 px vs −1e-6/−0.000928) — about
0.001 px apart — so their non-CUDEM tiles are restamped to the 5-basin
convention. `check_restamp` refuses anything beyond `restamp_tol_px` (0.01),
so a real misregistration can never be relabelled away.

**Resume safety.** Every write is `.tmp.tif` + rename; the per-tile stats JSON
(`manifests/merge_stats/<tile>.json`) is written last and holds a fingerprint
of merge version, parameters, target grid, territory-file digest and each
input's path/size/mtime. A tile is skipped only if that fingerprint matches
and the output passes `GridSpec.tile_matches` — recomposing a basin tile
invalidates exactly the merged tiles that consumed it. Bump
`merge.MERGE_VERSION` when blend semantics change.

**QA** (`qa/qa_report.json`): completeness, grid conformity (merged spec),
cross-tile seams, nodata inside the region polygon, and a merge summary
(modes, pixel tiers, and tiles whose blend-band spread p99 exceeds
`band_spread_flag_m` — report-only, for visual review). Accepted, documented
exceptions live in the merge config: `basin_nodata_accepted_px` (fails if a
known gap grows) and `seams_accepted`.

**Memory.** A 3-basin tile peaks ~6.7 GB (2-basin ~5 GB); the login-node
per-user cgroup cap is 30 GB, so `--nproc 1` beside other work (≈3 h for
region 02), or more on a compute node.

**Not done / options.** Recomposing shared estuary tiles from the union of
all basins' sources would beat blending finished products there; revisit only
if the band-spread review shows visible artifacts.

## 12. Channel burn (optional, on the merged product)

`scripts/burn_region.py` + `dem_pipeline/burn.py`, driven by
`config/burn_region02.yaml`, lower mapped stream channels below the lidar
water surface and write a **new** product (`midatlantic19b_<tile>_2026v1` in
`/pscratch/sd/s/sbrus/Region02_DEM_burned`). The merged product is only
read. Decisions confirmed by the user 2026-11-18: apply to the merged product
only; separate name and directory; burn lidar (1–3) **and** residual fill
(99); never bathymetry (11–15); Delaware (0204) first.

```bash
python scripts/burn_region.py all --nproc 4      # prepare, network, tiles, vrt, qa
python scripts/burn_region.py qa --connectivity  # + ponding check on 3 tiles
python scripts/burn_region.py tiles --tile n42x00_w075x25
```

**Why.** The subgrid model samples the DEM at native resolution and uses edge
statistics; a stream left at its lidar water surface carries no channel
volume, and a road embankment over a culvert acts as a dam in edge stats.
Method and constants are the scratch prototype v3
(`/pscratch/sd/s/sbrus/channel_geom_explore/proto/proto.py`), validated on
two EAARL-B mainstem reaches (Hancock, Water Gap).

**Inputs.** NHDPlus HR flowlines (FType 460/558, in-network, DA ≥ 1 km²,
tidal reaches per STREAM-geo `Tidal_Flag` skipped); Zarrabi et al. 2025
mean-flow depth/width and STREAM-geo RF width, carried from NHDPlusV2.1
COMIDs onto HR lines by a 200 m / DA-ratio match (`prepare` →
`prep/<huc4>/ml.gpkg`); USGS physiographic sections → regional bankfull
mean-depth curves (`curves:` in the config); TIGER roads (all counties
touching the basin) + rails (`prep/<huc4>/roads.gpkg`); USACE National
Inventory of Dams points + crest length (`prep/<huc4>/dams.gpkg`, from the
national CSV); the basin's stage-07 `water` polygons for tier 1.

**Phase A — network (per HUC-4, one process).** The channel profile is a
network property, so it is computed once per basin, not per tile, from
windowed reads of the merged tiles (no basin-scale raster):

- *Tier 1* (DA ≥ 100 km² inside a river polygon): water surface = PAVA
  non-increasing fit of 5×5 medians of lidar-provenance pixels every 15 m,
  capped across junctions; D = 0.5 × Zarrabi mean-flow depth (DA-smoothed).
- *Tier 2 / 1b*: junctions snapped to the lowest pixel within 15 m;
  least-cost path on a local-HAND cost within 40 m of the HR line;
  downstream-monotone cut-only bed; cuts > 2 m only within 15 m of a mapped
  road/rail (else capped and flagged) **and never where the cut segment
  touches an NID dam keep-out** (radius `dam_buffer_m` 150 m + half the
  crest length): a dam crest carries a road, so the culvert rule alone
  trenched F.E. Walter and Downsville 24–47 m (fixed 2026-11-19; reported as
  `seg_capped_at_dam`). NHD's own dam features (FType 343) miss every large
  Delaware dam, hence NID; 1D depressions > 30 m deep are
  quarries (never channel); thalweg = bed − 1.5 D, monotone across the
  network. D = 0.4 × regional curve (min 0.05 m), width = STREAM-geo RF
  (halved below 10 km², floor 2 px); tier 1b (big reach, no polygon) uses
  Zarrabi.
- Output: `spec/<huc4>/<tile>.npz` in **global lattice pixel coordinates**
  plus `manifests/network_<huc4>.json` (fingerprint, per-tile spec digests,
  report). Tier-2 reach table: `prep/<huc4>/tier2_reaches.csv`.

**Phase B — tiles (parallel).** Rasterize tier 1 (parabolic area-preserving
section between river-polygon banks; each pixel takes the water surface of
its **nearest** centerline sample, so `guard_tier1` drops any pixel whose
own merged surface is more than `t1_surface_tol_m` = 2 m above that
surface: the corridor polygon includes a reservoir that touches the reach
below its dam, and before this guard reservoir pools were trenched to the
downstream river, e.g. Cannonsville 349 → 307 m, Swinging Bridge, Blue
Marsh; counted as `n_tier1_surface_guarded`) and tier 2 (parabola over
width + ≥ 2 px 4-connected spine at thalweg), then `apply_burn` (below).
Tier 2 has a matching **lateral guard**: outside a path's minimal 4-connected core -- its pixels
plus one deterministic connector per diagonal step -- a pixel more than
`t2_lateral_tol_m` = 2 m above the local bed, thalweg + 1.5 D, is not
burned; without it the section dug into steep ground beside the path, e.g.
Downsville's dam face cut 46 m beside a path correctly capped at the dam,
and 59-83% of the > 5 m burn pixels on the Downsville / Hancock tiles were
such side-slope cuts. Culvert notches narrow to the 1-2 px core. The core is
explicit because an earlier prototype rule that limited off-path lowering
broke diagonal 4-connectivity (pits 197 -> 4014); validation tiles' residual
ponding moved 1.70/6.53/6.73% -> 1.71/6.59/7.11%. Counted as
`n_tier2_lateral_guarded`. `apply_burn` lowers a pixel only
if merged `_basin.tif` = that network's HUC-4 % 100, it has data, the target
is below it, and its provenance is in `burnable_ids` (lidar, fill) **or** in
`burnable_outside_water_ids` (EAARL-B 12, CoNED 14) and it lies outside the
basin's NHD `water` polygons. The second rule (user decision 2026-11-18)
exists because those survey rasters carry land topo off-channel (see §9):
left unburnable they dammed the tier-2 network on the EAARL tiles
(residual ponding on Hancock 6.9% vs the prototype's 1.1%). Inside water
polygons they stay authoritative bathymetry. Lowered pixels get
provenance **16**. Outputs per tile: elevation COG, `_source.tif`,
`_basin.tif` (copied), `_burndepth.tif` (burned tiles only); tiles with
nothing lowered are byte-copied.

**Lattice.** All merged tiles share pixel size and integer-pixel origins, so
a global (row, col) maps to exactly one pixel in every tile, and both tiles
of a shared collar burn identically. Pixel-size differences between CUDEM
vintages (≤ 0.001 px drift per tile) are tolerated; a tile whose origin is
off the lattice is excluded and copied unburned (region 02:
`n37x00_w076x00`, the half-pixel-shifted bay-mouth CUDEM tile).

**Resume safety.** Network: fingerprint of burn version, parameters,
curves, lattice and every input (HR GDB, water mask, prep files, merged
tiles of the basin). Tiles: fingerprint of version, parameters, grid, the
three merged inputs' size/mtime and the spec digests; stats JSON
(`manifests/burn_stats/<tile>.json`) written last. Specs are read through
the manifest only (a stale `.npz` is ignored; a digest mismatch is fatal).
Rerunning `merge_region.py` therefore invalidates exactly the affected
burned tiles (and the network, which reads the merged tiles). Bump
`burn.BURN_VERSION` when tile semantics change and `burn.NETWORK_VERSION`
when the network pass changes; the network fingerprint ignores apply-only
parameters (`APPLY_ONLY_KEYS`), so changing what may be lowered re-burns
tiles without the ~30 min network rerun.

**QA** (`qa/qa_report.json`): completeness, grid conformity, seams (with
the merge config's accepted exceptions), burn summary (pixels by tier,
fill pixels burned, volume, deepest tiles, network reports) and, with
`--connectivity`, ponding on the snapped network before vs after for
`qa.connectivity_tiles`.

**Known limitations.** Tier-1 depth validated on one river (two reaches);
Water Gap tier-2 residual ponding not investigated; no explicit smooth join
to tidal bathymetry at the head of tide; dams missing from NID (or whose
NID point is > buffer + half-crest from the crest) can still be cut where a
road crosses; tidal reaches are skipped even where a small marsh creek
has no bathymetry; EAARL-vs-3DEP sanity check from the prototype notes not
implemented (EAARL pixels inside water polygons are never lowered); CoNED
bathymetry inside water polygons shallower than a tributary's burned
thalweg leaves a small storage pit at the confluence (Piedmont tile).
