#!/usr/bin/env python
"""Stage 07: build water-mask vectors -> data/intermediate/water_mask/water_mask.gpkg.

Layers:
    water             NHDPlus HR water polygons (selected FTypes, dissolved)
    ehydro_footprint  survey hulls from stage 06
    eaarlb_footprint  EAARL-B raster tile footprints
    eaarlb_voids      lidar-void polygons shipped with the EAARL-B release

Rasterization onto each tile grid happens inside the compositor (stage 08);
this stage only prepares clean vectors.
"""

import argparse
import logging
import zipfile

import dem_pipeline as dp

log = logging.getLogger("07_build_water_mask")


def nhd_water_polys(cfg):
    import geopandas as gpd
    import pyogrio

    ncfg = cfg["nhdplushr"]
    zpath = dp.path("data", "raw", "nhdplushr",
                    ncfg["url"].rsplit("/", 1)[1])
    gdb_dir = zpath.parent
    gdbs = sorted(gdb_dir.rglob("*.gdb"))
    if not gdbs:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(gdb_dir)
        gdbs = sorted(gdb_dir.rglob("*.gdb"))
    gdb = gdbs[0]
    layers = [l[0] for l in pyogrio.list_layers(gdb)]

    parts = []
    area_layer = next((l for l in layers if l.lower() == "nhdarea"), None)
    if area_layer:
        area = gpd.read_file(gdb, layer=area_layer,
                             where=f"FType IN {tuple(ncfg['nhdarea_ftypes'])}")
        parts.append(area[["geometry"]])
        log.info("NHDArea: %d polygons", len(area))
    wb_layer = next((l for l in layers if l.lower() == "nhdwaterbody"), None)
    if wb_layer:
        wb = gpd.read_file(gdb, layer=wb_layer,
                           where=f"FType IN {tuple(ncfg['nhdwaterbody_ftypes'])}")
        wb = wb.to_crs(epsg=26918)
        wb = wb[wb.area >= ncfg["min_waterbody_m2"]].to_crs(epsg=4269)
        parts.append(wb[["geometry"]])
        log.info("NHDWaterbody (>= %g m^2): %d polygons",
                 ncfg["min_waterbody_m2"], len(wb))
    if not parts:
        raise SystemExit(f"no NHDArea/NHDWaterbody layers found in {gdb}")

    import pandas as pd
    merged = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), crs=parts[0].crs
    ).to_crs(epsg=4269)
    dissolved = merged.dissolve().explode(index_parts=False).reset_index(drop=True)
    return dissolved


def eaarlb_layers(cfg):
    """(tile footprints, void polygons) from the EAARL-B download dir."""
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box

    raw = dp.path(cfg["sources"]["eaarlb"]["dir"])
    boxes = []
    for tif in sorted(raw.rglob("*.tif")):
        with rasterio.open(tif) as src:
            b = src.bounds
            crs = src.crs
        gdf = gpd.GeoDataFrame(geometry=[box(*b)], crs=crs).to_crs(epsg=4269)
        boxes.append(gdf.geometry.iloc[0])
    footprints = gpd.GeoDataFrame(geometry=boxes, crs="EPSG:4269")

    voids = None
    for shp in list(raw.rglob("*.shp")) + list(raw.rglob("*.gpkg")):
        if "void" in shp.stem.lower():
            voids = gpd.read_file(shp).to_crs(epsg=4269)
            break
    if voids is None:
        # void shapefiles may arrive zipped
        for z in raw.rglob("*.zip"):
            if "void" in z.stem.lower():
                extract = z.parent / z.stem
                if not extract.exists():
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(extract)
                shps = list(extract.rglob("*.shp"))
                if shps:
                    voids = gpd.read_file(shps[0]).to_crs(epsg=4269)
                    break
    return footprints, voids


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    dp.setup_logging()
    cfg = dp.load_config()

    import geopandas as gpd

    out = dp.path("data", "intermediate", "water_mask", "water_mask.gpkg")
    out.parent.mkdir(parents=True, exist_ok=True)

    water = nhd_water_polys(cfg)
    water.to_file(out, layer="water", driver="GPKG")
    log.info("water layer: %d polygons", len(water))

    fp_gpkg = dp.path("data", "intermediate", "ehydro_footprints.gpkg")
    if fp_gpkg.exists():
        gpd.read_file(fp_gpkg, layer="ehydro_footprint").to_file(
            out, layer="ehydro_footprint", driver="GPKG")

    footprints, voids = eaarlb_layers(cfg)
    if len(footprints):
        footprints.to_file(out, layer="eaarlb_footprint", driver="GPKG")
        log.info("eaarlb_footprint: %d tiles", len(footprints))
    if voids is not None and len(voids):
        voids.to_file(out, layer="eaarlb_voids", driver="GPKG")
        log.info("eaarlb_voids: %d polygons", len(voids))

    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
