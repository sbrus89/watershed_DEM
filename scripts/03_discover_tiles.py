#!/usr/bin/env python
"""Stage 03: discover all source downloads and write manifests/downloads.csv.

- S1M tiles via the TNM API (polygon query with bbox fallback).
- Gap detection: buffered basin minus S1M footprints -> query project-based
  1-m DEM for the gaps; anything still uncovered gets 1/3" 1-degree tiles.
- NHDPlus HR GDB (water mask) and EAARL-B (ScienceBase via DOI, including
  the lidar-void polygons).
- eHydro CENAP survey ZIPs from the AGOL FeatureServer (all surveys kept in
  the manifest; newest-per-footprint selection happens in stage 06).
"""

import argparse
import logging
import re

import dem_pipeline as dp
from dem_pipeline.download import write_manifest
from dem_pipeline.tnm import query_products, simplify_polygon_for_api

log = logging.getLogger("03_discover_tiles")


def discover_tnm(dataset: str, basin_geom, cfg, dest_dir) -> list[dict]:
    """Query TNM for a dataset, post-filter to the basin, build rows."""
    from shapely.geometry import box

    poly = simplify_polygon_for_api(basin_geom)
    items = query_products(
        dataset,
        bbox=tuple(cfg["project"]["bbox"]) if poly is None else None,
        polygon_wkt=poly,
    )
    rows, seen = [], set()
    for it in items:
        url = it.get("downloadURL") or ""
        if not url.lower().endswith(".tif") or url in seen:
            continue
        bb = it.get("boundingBox") or {}
        if bb:
            fp = box(bb["minX"], bb["minY"], bb["maxX"], bb["maxY"])
            if not basin_geom.intersects(fp):
                continue
        seen.add(url)
        rows.append({
            "url": url,
            "dest": str(dest_dir / url.rsplit("/", 1)[1]),
            "size": it.get("sizeInBytes") or "",
            "source": dest_dir.name,
            "tile": "",
            "bbox": (f"{bb['minX']} {bb['minY']} {bb['maxX']} {bb['maxY']}"
                     if bb else ""),
            "_bbox": bb,
        })
    return rows


def footprint_union(rows):
    from shapely.geometry import box
    from shapely.ops import unary_union

    boxes = [box(r["_bbox"]["minX"], r["_bbox"]["minY"],
                 r["_bbox"]["maxX"], r["_bbox"]["maxY"])
             for r in rows if r.get("_bbox")]
    return unary_union(boxes) if boxes else None


def dem13_rows(gap_geom, cfg) -> list[dict]:
    """1-degree 1/3-arc-second tiles covering any residual gap polygons.

    dem13 is the last-resort land fallback (priority 3, below S1M and the
    project 1-m DEMs): a residual gap is just as likely to be open water
    (no 1/3" land tile exists there at all) as it is a real coverage hole,
    so a 404 here is an expected outcome, not a download failure. Candidates
    are restricted to cells that actually intersect the gap polygon (not
    just its bounding box) and confirmed to exist via HEAD before being
    queued, so stage 04 never has to fail a whole run over a cell that was
    never going to have data.
    """
    import math

    import requests
    from shapely.geometry import box

    dest = dp.path(cfg["sources"]["dem13"]["dir"])
    pattern = cfg["sources"]["dem13"]["url_pattern"]
    timeout = cfg["download"]["timeout_s"]
    w, s, e, n = gap_geom.bounds
    rows = []
    for lat in range(math.floor(s), math.ceil(n)):
        for lon in range(math.floor(w), math.ceil(e)):
            if not gap_geom.intersects(box(lon, lat, lon + 1, lat + 1)):
                continue
            tile = f"n{lat + 1:02d}w{abs(lon):03d}"
            url = pattern.format(tile=tile)
            try:
                exists = requests.head(url, allow_redirects=True,
                                        timeout=timeout).status_code == 200
            except requests.RequestException:
                exists = False
            if not exists:
                log.warning("dem13 %s: no tile at %s -- treating gap cell as "
                            "water/no-land-coverage", tile, url)
                continue
            rows.append({"url": url, "dest": str(dest / url.rsplit("/", 1)[1]),
                         "size": "", "source": "dem13", "tile": tile,
                         "bbox": f"{lon} {lat} {lon + 1} {lat + 1}"})
    return rows


def ehydro_rows(cfg, basin_geom) -> list[dict]:
    import requests

    scfg = cfg["sources"]["ehydro"]
    dest = dp.path(scfg["dir"])
    rows, offset = [], 0
    while True:
        params = {
            "where": f"usacedistrictcode='{scfg['district']}'",
            "outFields": "surveyjobidpk,sdsid,surveytype,channelareaidfk,"
                         "dateuploaded,surveydatestart,surveydateend,"
                         "sourcedatalocation",
            "returnGeometry": "true",
            "outSR": "4269",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": 1000,
        }
        r = requests.get(scfg["feature_server"], params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
        feats = data.get("features", [])
        if not feats:
            break
        from shapely.geometry import shape

        for f in feats:
            props = f["properties"]
            url = props.get("sourcedatalocation") or ""
            if not url:
                continue
            geom = shape(f["geometry"]) if f.get("geometry") else None
            if geom is not None and not basin_geom.intersects(geom):
                continue
            b = geom.bounds if geom is not None else None
            rows.append({
                "url": url,
                "dest": str(dest / url.rsplit("/", 1)[1]),
                "size": "",
                "source": "ehydro",
                "tile": str(props.get("surveydateend") or ""),
                "bbox": f"{b[0]} {b[1]} {b[2]} {b[3]}" if b else "",
            })
        offset += len(feats)
        if not data.get("properties", {}).get("exceededTransferLimit") and \
           len(feats) < 1000:
            break
    log.info("eHydro: %d basin-intersecting surveys", len(rows))
    return rows


def _resolve_doi_to_item(doi: str) -> str | None:
    """ScienceBase item id behind a DOI, via the doi.org redirect."""
    import requests

    try:
        r = requests.head(f"https://doi.org/{doi}", allow_redirects=True,
                          timeout=120)
        m = re.search(r"/catalog/item/([0-9a-f]{24})", r.url)
        return m.group(1) if m else None
    except requests.RequestException as e:
        log.warning("DOI resolution failed for %s: %s", doi, e)
        return None


# EAARL-B data tiles are named for the NW corner of a 10 km UTM 18N block,
# e.g. Topobathy_e510000_n4560000_v1.zip -> [510000, 4550000, 520000, 4560000].
# Verified against the extracted raster: its bounds sit a metre outside the
# nominal corner, hence the pad below.
EAARLB_TILE_RE = re.compile(r"_e(\d{6,7})_n(\d{6,7})_", re.I)
EAARLB_TILE_M = 10000.0
EAARLB_PAD_M = 1000.0


def _eaarlb_tile_bbox(fname: str):
    """Nominal lon/lat bbox of an EAARL-B tile, or None if not a data tile."""
    m = EAARLB_TILE_RE.search(fname)
    if not m:
        return None
    from pyproj import Transformer

    e, n = float(m.group(1)), float(m.group(2))
    tr = Transformer.from_crs("EPSG:26918", "EPSG:4269", always_xy=True)
    xs = [e - EAARLB_PAD_M, e + EAARLB_TILE_M + EAARLB_PAD_M]
    ys = [n - EAARLB_TILE_M - EAARLB_PAD_M, n + EAARLB_PAD_M]
    lons, lats = tr.transform([xs[0], xs[1], xs[0], xs[1]],
                              [ys[0], ys[0], ys[1], ys[1]])
    return min(lons), min(lats), max(lons), max(lats)


def eaarlb_rows(cfg, basin_geom=None) -> list[dict]:
    """Resolve the EAARL-B release on ScienceBase and list its files.

    Data tiles are filtered to the basin: the release covers the whole
    Delaware from Hancock down, so an estuary-only clip_bbox would otherwise
    pull the entire upper-river survey (0.6 GB) that no output tile can read.
    Non-tile members (the FGDC xml, the 10 km grid index, and the lidar-void
    polygons) are small and always kept.
    """
    import requests
    from shapely.geometry import box

    scfg = cfg["sources"]["eaarlb"]
    dest = dp.path(scfg["dir"])
    doi = scfg["doi"]
    rows: list[dict] = []

    # A pinned item id wins; otherwise resolve the DOI. Do NOT go back to
    # ?filter=doi= -- ScienceBase ignores unrecognized filters silently and
    # answers with the entire catalog, so the failure looks like "found
    # nothing" rather than an error.
    root = scfg.get("sciencebase_item") or _resolve_doi_to_item(doi)
    if not root:
        log.warning("could not resolve DOI %s to a ScienceBase item", doi)
        return rows
    # walk the item tree: some releases are parent + child items, this one
    # carries all 24 data files directly on the parent.
    queue = [root]
    seen_items = set()
    n_skipped = [0]
    while queue:
        item_id = queue.pop()
        if item_id in seen_items:
            continue
        seen_items.add(item_id)
        ir = requests.get(
            f"https://www.sciencebase.gov/catalog/item/{item_id}",
            params={"format": "json", "fields": "title,files"},
            timeout=120,
        )
        ir.raise_for_status()
        for f in ir.json().get("files", []) or []:
            fname = f.get("name", "")
            if not fname.lower().endswith((".tif", ".zip", ".shp", ".xml")):
                continue
            bb = _eaarlb_tile_bbox(fname)
            if bb is not None and basin_geom is not None:
                if not basin_geom.intersects(box(*bb)):
                    n_skipped[0] += 1
                    continue
            rows.append({"url": f["url"], "dest": str(dest / fname),
                         "size": f.get("size") or "", "source": "eaarlb",
                         "tile": ""})
        cr = requests.get(
            "https://www.sciencebase.gov/catalog/items",
            params={"parentId": item_id, "format": "json", "fields": "id",
                    "max": 1000},
            timeout=120,
        )
        if cr.ok:
            queue.extend(it["id"] for it in cr.json().get("items", []))
    log.info("EAARL-B: %d files discovered via ScienceBase (%d tiles outside "
             "the basin skipped)", len(rows), n_skipped[0])
    if not rows:
        log.warning("EAARL-B discovery returned nothing -- resolve DOI %s "
                    "manually and add rows to manifests/downloads.csv", doi)
    return rows


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    dp.setup_logging()
    cfg = dp.load_config()

    import geopandas as gpd

    boundary = gpd.read_file(
        dp.path("data", "intermediate", "huc4_0204.gpkg"), layer="huc4_buffered"
    )
    basin_geom = boundary.geometry.union_all()

    rows: list[dict] = []

    # --- land stack -------------------------------------------------------
    s1m = discover_tnm(cfg["sources"]["s1m"]["tnm_dataset"], basin_geom, cfg,
                       dp.path(cfg["sources"]["s1m"]["dir"]))
    log.info("S1M: %d tiles", len(s1m))
    rows += s1m

    covered = footprint_union(s1m)
    gaps = basin_geom.difference(covered.buffer(0)) if covered else basin_geom
    if not gaps.is_empty and gaps.area > 1e-6:
        log.info("S1M gap area: %.4f deg^2 -- querying project 1-m DEMs",
                 gaps.area)
        proj = discover_tnm(cfg["sources"]["dem1m_projects"]["tnm_dataset"],
                            gaps, cfg,
                            dp.path(cfg["sources"]["dem1m_projects"]["dir"]))
        log.info("project 1-m: %d tiles for gaps", len(proj))
        rows += proj
        covered2 = footprint_union(proj)
        residual = gaps.difference(covered2.buffer(0)) if covered2 else gaps
        if not residual.is_empty and residual.area > 1e-6:
            log.warning("residual 1-m gap area %.4f deg^2 -> 1/3\" fallback",
                        residual.area)
            rows += dem13_rows(residual, cfg)

    # --- water stack + masks ---------------------------------------------
    rows += ehydro_rows(cfg, basin_geom)
    rows += eaarlb_rows(cfg, basin_geom)
    nhd_url = cfg["nhdplushr"]["url"]
    rows.append({"url": nhd_url,
                 "dest": str(dp.path("data", "raw", "nhdplushr",
                                     nhd_url.rsplit("/", 1)[1])),
                 "size": "", "source": "nhdplushr", "tile": ""})
    if cfg["sources"]["coned"].get("enabled"):
        import requests
        r = requests.get(cfg["sources"]["coned"]["urllist"], timeout=120)
        r.raise_for_status()
        dest = dp.path(cfg["sources"]["coned"]["dir"])
        rows += [{"url": u.strip(), "dest": str(dest / u.strip().rsplit("/", 1)[1]),
                  "size": "", "source": "coned", "tile": ""}
                 for u in r.text.splitlines() if u.strip().endswith(".tif")]

    for r_ in rows:
        r_.pop("_bbox", None)
    out = dp.path("manifests", "downloads.csv")
    write_manifest(rows, out)
    by_source: dict[str, int] = {}
    for r_ in rows:
        by_source[r_["source"]] = by_source.get(r_["source"], 0) + 1
    log.info("wrote %s: %d files %s", out, len(rows), by_source)


if __name__ == "__main__":
    main()
