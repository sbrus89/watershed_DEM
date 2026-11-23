"""Tests for the channel burn (dem_pipeline.burn) on tiny synthetic tiles.

Tiles here are 40 x 64 px at the real 1/9" pixel size (so the metre-based
parameters behave as in production), two of them side by side sharing a
4-px collar, on a GridSpec planted in memory.
"""

import hashlib
import json

import numpy as np

from tests.util import importorskip, raises, skip_unless_cli

RES = 3.08641975308642e-05
H, W, OV = 80, 64, 4                 # rows, cols per tile, collar overlap
X0, Y0 = -75.0, 40.0
TA, TB = "n40x00_w075x25", "n40x00_w075x00"   # A west, B east
NODATA = -999999.0

PARAMS = dict(burn_id=16, burnable_ids=[1, 2, 3, 99], burnable_outside_water_ids=[],
              wse_ids=[1, 2, 3],
              da_min_km2=1.0, skip_tidal=True, t1_da_km2=100.0,
              t1_zarrabi_factor=0.5, t1_corridor_m=500.0, t1_sample_m=15.0,
              t2_factor=0.4, t2_min_depth_m=0.05, t2_narrow_da_km2=10.0,
              node_snap_m=15.0, snap_corridor_m=40.0, window_margin_m=60.0,
              cut_cap_m=2.0, crossing_buffer_m=15.0, pit_deep_m=30.0,
              max_nan_frac=0.2, dam_buffer_m=150.0, t1_surface_tol_m=2.0,
              t2_lateral_tol_m=2.0, conduit_ftypes=[334, 428], conduit_max_m=300.0)


def _spec():
    from dem_pipeline.grid import GridSpec
    gts = {TA: [X0, RES, 0.0, Y0, 0.0, -RES],
           TB: [X0 + (W - OV) * RES, RES, 0.0, Y0, 0.0, -RES]}
    return GridSpec(cudem_geotransforms=gts, cudem_sizes={TA: [W, H], TB: [W, H]})


def _lattice():
    from dem_pipeline.burn import Lattice
    return Lattice(_spec(), [TA, TB])


def _write(path, arr, gt, dtype):
    import rasterio
    from rasterio.transform import Affine
    path.parent.mkdir(parents=True, exist_ok=True)
    nd = NODATA if dtype == "float32" else 0
    a = np.where(np.isfinite(arr), arr, nd).astype(dtype) if dtype == "float32" else arr.astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", dtype=dtype, count=1, width=a.shape[1],
                       height=a.shape[0], crs="EPSG:4269", transform=Affine.from_gdal(*gt),
                       nodata=nd) as d:
        d.write(a, 1)


def _valley(L, embankment=4.0):
    """Global DEM: valley sloping west->east, V cross-section centred on row
    40, an embankment of +embankment m across it at global cols 60-62."""
    nI, nJ = H, 2 * W - OV
    I, J = np.mgrid[0:nI, 0:nJ]
    z = 100.0 - 0.01 * J + 0.5 * np.abs(I - 40)
    z[:, 60:63] += embankment
    return z.astype("f4")


def _plant(tmp_path, L, zg, provg=None, basing=None):
    """Write the global arrays as the two merged input tiles; return paths."""
    provg = np.ones(zg.shape, "u1") if provg is None else provg
    basing = np.full(zg.shape, 4, "u1") if basing is None else basing
    spec = _spec()
    paths = {}
    for t in (TA, TB):
        r0, r1, c0, c1 = L.window(t)
        gt = spec.tile_geotransform(t)[0]
        base = tmp_path / "merged" / "tiles" / f"midatlantic19_{t}_2026v1"
        for kind, arr, dt, sfx in (("z", zg, "float32", ""), ("prov", provg, "uint8", "_source"),
                                   ("basin", basing, "uint8", "_basin")):
            p = base.parent / f"{base.name}{sfx}.tif"
            _write(p, arr[r0:r1, c0:c1], gt, dt)
            paths[(t, kind)] = p
    return paths


def _cfg(tmp_path):
    return {"project": {"name": "midatlantic19b", "version": "2026v1"},
            "output": {"root": str(tmp_path / "burned")},
            "burn": dict(PARAMS), "data": {"metric_epsg": 26918},
            "basins": [{"huc4": "0204"}],
            "_merge": {"project": {"name": "midatlantic19", "version": "2026v1"},
                       "output": {"root": str(tmp_path / "merged"), "epsg": 4269,
                                  "cog": {"compress": "DEFLATE", "predictor": 3,
                                          "blocksize": 512, "overview_resampling": "AVERAGE"}}}}


def _lines(L, cols=(25, 98), row=40):
    """One west->east tier-2 HR flowline along the valley floor."""
    import geopandas as gpd
    import shapely
    (xa, xb), (ya, yb) = L.xy([row, row], list(cols))
    return gpd.GeoDataFrame(
        {"NHDPlusID": [1], "FromNode": [10], "ToNode": [11], "HydroSeq": [5],
         "TotDASqKm": [20.0], "DivDASqKm": [20.0], "DA": [20.0],
         "mf_depth": [np.nan], "mf_width": [np.nan],
         "RF_Width_m": [8.0], "bf_c": [0.204], "bf_e": [0.321], "GNIS_Name": ["x"],
         "in_lake": [False], "conduit": [False]},
        geometry=[shapely.LineString([(xa, ya), (xb, yb)])], crs=4269)


def _net(tmp_path, L, paths, roads=None):
    import geopandas as gpd
    from dem_pipeline import burn
    M = burn.Mosaic(L, lambda t, k: paths[(t, k)])
    net = burn.Net(_cfg(tmp_path), {"huc4": "0204"}, L, M)
    net.roads = roads if roads is not None else gpd.GeoDataFrame(geometry=[], crs=4269)
    return net


# --------------------------------------------------------------- pure pieces

def test_pava_dec_is_the_nonincreasing_least_squares_fit():
    """Catches: an increasing fit, or a fit that does not pool violators
    (a 'monotone' surface that steps up downstream)."""
    from dem_pipeline.burn import pava_dec
    assert np.allclose(pava_dec([3.0, 1.0, 2.0]), [3.0, 1.5, 1.5])
    y = np.array([5.0, 6.0, 4.0, 4.5, 1.0, 2.0, 0.0])
    f = pava_dec(y)
    assert np.all(np.diff(f) <= 1e-12)
    assert abs(f.mean() - y.mean()) < 1e-12           # L2 projection keeps the mean
    mono = np.array([9.0, 7.0, 7.0, 2.0])
    assert np.array_equal(pava_dec(mono), mono)


def test_quarry_is_excluded_and_never_drags_the_bed_down():
    """Catches the 2026-11-17 prototype bug: a -85 m quarry floor on a path
    became the monotone bed for everything downstream (a 110 m trench)."""
    importorskip("scipy")
    from dem_pipeline.burn import path_bed, thalweg
    zp = np.r_[np.linspace(50, 45, 20), np.full(10, 5.0), np.linspace(44, 40, 20)]
    bed, quarry, ign = path_bed(zp, np.inf, pit_deep=30.0)
    assert quarry[20:30].all() and not quarry[:15].any() and not quarry[35:].any()
    assert bed[35:].min() > 39.0                      # downstream not dragged to 5 m
    capped = np.zeros(len(zp), bool)
    thal, _ = thalweg(bed, 0.5, quarry, capped, np.inf, 30.0)
    assert np.isinf(thal[20:30]).all()                # nothing burned in the pit
    assert np.isfinite(thal[35:]).all() and thal[35:].min() > 38.0
    # an upstream cap that came from a pit (far below this path) is ignored
    _, _, ign = path_bed(zp[30:], 5.0, pit_deep=30.0)
    assert ign


def test_level_is_not_carried_across_a_quarry_gap():
    """Catches the 2026-11-20 region-02 bug: the running bed/thalweg level
    was carried across a quarry gap onto the first point beyond it. On an
    HR reach that climbs a ridge (Tioughnioga -> Castle Creek, NY: the whole
    climb is one 'pit') the river's thalweg landed on the hilltop, a single
    pixel cut 147 m; past a real mid-reach pit the exit was cut to the
    pre-pit level."""
    importorskip("scipy")
    from dem_pipeline.burn import path_bed, thalweg
    # (a) a reach that runs uphill, 312 -> 431 m, entered with the
    # downstream river's thalweg cap (283.8 m) at its first node
    zp = np.linspace(312.0, 431.0, 60)
    bed, quarry, _ = path_bed(zp, 286.0, pit_deep=30.0)
    capped = np.zeros(len(zp), bool)
    thal, _ = thalweg(bed, 0.5, quarry, capped, 283.8, 30.0)
    burned = np.isfinite(thal)
    assert quarry[:50].all()                          # the climb is the 'pit'
    assert (zp[burned] - thal[burned]).max() < 2.0    # was 147 m at the end
    # (b) a reach that starts in a deep pit (junction on the pit rim) and
    # exits at 95 -> 90 m, entered with an upstream cap of 80 m: within
    # pit_deep of the rim, so not discarded as a pit cap -- it must still
    # not be carried over the pit onto the exit (old: a 15 m thalweg cut
    # that no cut cap limits)
    zp = np.r_[100.0, 100.0, np.full(15, 0.0), np.linspace(95, 90, 20)]
    bed, quarry, ign = path_bed(zp, 80.0, pit_deep=30.0)
    assert not ign and quarry[2:17].all()
    thal, ign = thalweg(bed, 0.5, quarry, np.zeros(len(zp), bool), 79.25, 30.0)
    post = np.flatnonzero(np.isfinite(thal) & (np.arange(len(zp)) >= 17))
    assert len(post) and (zp[post] - thal[post]).max() < 2.0  # old: 15 m
    assert (zp[post] - bed[post]).max() < 1.0        # bed not dragged to 80 m


def test_uncrossed_deep_cut_is_capped_but_crossing_cut_is_not():
    """Catches: road-embankment (culvert) cuts limited like any other cut,
    or unmapped deep cuts allowed (a trench through a ridge/quarry wall)."""
    importorskip("scipy")
    from dem_pipeline.burn import cap_uncrossed_cuts
    zp = np.array([10.0, 9.9, 13.9, 13.9, 9.7, 9.6])
    bed = np.array([10.0, 9.9, 9.9, 9.9, 9.7, 9.6])
    b1, capped, n_c, n_x, _ = cap_uncrossed_cuts(zp, bed, lambda idx: False, 2.0)
    assert n_c == 1 and n_x == 0 and capped[2:4].all()
    assert np.allclose(b1[2:4], 11.9)                 # lowered by exactly cut_cap
    b2, capped, n_c, n_x, _ = cap_uncrossed_cuts(zp, bed, lambda idx: True, 2.0)
    assert n_c == 0 and n_x == 1 and np.array_equal(b2, bed)


def test_dam_keepout_caps_a_cut_even_at_a_road_crossing():
    """Catches: a dam crest that carries a road being trenched through as
    a culvert (Downsville / F.E. Walter were cut 24-47 m), or the dam rule
    capping shallow cuts / segments that do not touch a dam."""
    importorskip("scipy")
    from dem_pipeline.burn import cap_uncrossed_cuts
    zp = np.array([10.0, 9.9, 13.9, 13.9, 9.7, 9.6, 10.6, 9.4])
    bed = np.array([10.0, 9.9, 9.9, 9.9, 9.7, 9.6, 9.5, 9.4])
    b, capped, n_c, n_x, n_d = cap_uncrossed_cuts(
        zp, bed, lambda idx: True, 2.0, at_dam=lambda idx: True)
    assert n_d == 1 and n_c == 1 and n_x == 0         # the 1.1 m cut is not a dam case
    assert capped[2:4].all() and not capped[6]
    assert np.allclose(b[2:4], 11.9) and b[6] == 9.5
    b, _, n_c, n_x, n_d = cap_uncrossed_cuts(
        zp, bed, lambda idx: True, 2.0, at_dam=lambda idx: False)
    assert (n_c, n_x, n_d) == (0, 1, 0) and np.array_equal(b, bed)


def test_guard_tier1_drops_pixels_on_a_higher_water_surface():
    """Catches: a reservoir pool beside the reach below its dam being
    trenched to the downstream river level (Cannonsville 349 -> 307 m), or
    the guard dropping ordinary river pixels / nodata handling inverted."""
    from dem_pipeline.burn import guard_tier1
    t1 = np.array([[305.5, 305.5, 305.5, 305.5, np.inf]], "f4")
    wse = np.array([[307.0, 307.0, 307.0, 307.0, np.inf]], "f4")
    z = np.array([[307.2, 308.9, 349.3, np.nan, 349.3]], "f4")
    g, n = guard_tier1(t1, wse, z, 2.0)
    assert list(g[0, :2]) == [305.5, 305.5]           # river, and riffle within tol
    assert np.isinf(g[0, 2]) and np.isinf(g[0, 3])    # pool above the dam, nodata
    assert n == 1                                     # only the pool would have been lowered


def test_build_dams_parses_the_nid_csv_and_buffers_by_crest_length(tmp_path):
    """Catches: the NID CSV's leading 'Data Last Updated' line breaking the
    header, dams outside the basin kept, and the keep-out ignoring crest
    length (NID points sit anywhere along a ~1 km crest)."""
    importorskip("geopandas")
    import shapely
    from dem_pipeline.burn import build_dams, dam_buffers
    csv = tmp_path / "nid.csv"
    csv.write_text('Data Last Updated:,2026-11-19\n'
                   '"Dam Name","NID ID","Latitude","Longitude","Dam Length (Ft)","NID Height (Ft)","State"\n'
                   '"Cannonsville Dam","NY00001",42.066,-75.376,2800,175,"NY"\n'
                   '"Far Dam","XX00001",30.0,-90.0,100,10,"LA"\n'
                   '"No Length","NY00002",42.0,-75.0,,12,"NY"\n')
    g = build_dams(csv, shapely.box(-76.0, 41.5, -74.5, 42.5))
    assert sorted(g.nid_id) == ["NY00001", "NY00002"]
    bu = dam_buffers(g, 150.0, 26918).to_crs(26918)
    r = np.sqrt(bu.area.to_numpy() / np.pi)
    i = list(g.nid_id).index("NY00001")
    assert abs(r[i] - (150.0 + 0.5 * 2800 * 0.3048)) < 10
    assert abs(r[1 - i] - 150.0) < 5


def test_apply_burn_guards():
    """Catches: bathymetry lowered (it is the authoritative channel), another
    basin's pixel burned by this network, nodata filled, or a pixel raised."""
    from dem_pipeline.burn import apply_burn
    z = np.array([[5.0, 5.0, 5.0, np.nan, 5.0, 1.0]], "f4")
    prov = np.array([[1, 99, 12, 1, 1, 1]], "u1")
    basin = np.array([[4, 4, 4, 4, 5, 4]], "u1")
    tgt = np.full(z.shape, 2.0, "f4")
    zn, pn, dep = apply_burn(z, prov, basin, {4: tgt}, [1, 2, 3, 99], 16)
    assert zn[0, 0] == 2.0 and zn[0, 1] == 2.0        # lidar and filled burned
    assert zn[0, 2] == 5.0 and pn[0, 2] == 12         # EAARL bathymetry untouched
    assert np.isnan(zn[0, 3])                         # nodata stays nodata
    assert zn[0, 4] == 5.0 and pn[0, 4] == 1          # basin 0205 px untouched
    assert zn[0, 5] == 1.0 and pn[0, 5] == 1          # target above: not raised
    assert list(pn[0]) == [16, 16, 12, 1, 1, 1]
    assert np.allclose(dep[0, [0, 1, 2, 4, 5]], [3, 3, 0, 0, 0]) and np.isnan(dep[0, 3])


def test_apply_burn_survey_topo_only_outside_water():
    """Catches: survey-raster (EAARL) pixels lowered INSIDE the water
    polygons, where they are the authoritative bathymetry, or the
    outside-water allowance leaking to sources not listed / other basins."""
    from dem_pipeline.burn import apply_burn
    z = np.full((1, 5), 5.0, "f4")
    prov = np.array([[12, 12, 13, 14, 12]], "u1")
    basin = np.array([[4, 4, 4, 4, 5]], "u1")
    outside = {4: np.array([[True, False, True, True, True]])}
    tgt = np.full(z.shape, 2.0, "f4")
    zn, pn, _ = apply_burn(z, prov, basin, {4: tgt}, [1, 99], 16, [12, 14], outside)
    assert list(zn[0]) == [2.0, 5.0, 5.0, 2.0, 5.0]
    assert list(pn[0]) == [16, 12, 13, 16, 12]
    zn, _, _ = apply_burn(z, prov, basin, {4: tgt}, [1, 99], 16)   # default: never
    assert np.all(zn == 5.0)


def test_lattice_maps_tiles_and_rejects_misregistration():
    """Catches: a spec stored in global coordinates landing on different
    pixels in two tiles of a shared collar, a misregistered tile being
    silently snapped onto the lattice (it must be excluded instead), or a
    pixel size that drifts across a tile being accepted."""
    from dem_pipeline.burn import Lattice
    from dem_pipeline.grid import GridSpec
    L = _lattice()
    assert L.off[TA] == (0, 0, H, W) and L.off[TB] == (0, W - OV, H, W)
    x, y = L.xy(7, 70)
    I, J = L.rowcol(x, y)
    assert (int(I), int(J)) == (7, 70)
    bad = GridSpec(cudem_geotransforms={TA: [X0, RES, 0, Y0, 0, -RES],
                                        TB: [X0 + (W - OV + 0.3) * RES, RES, 0, Y0, 0, -RES]},
                   cudem_sizes={TA: [W, H], TB: [W, H]})
    Lb = Lattice(bad, [TA, TB])
    assert TB not in Lb.off and abs(Lb.off_lattice[TB][1] - 0.3) < 1e-6
    assert Lb.tiles_overlapping(0, H, 0, 2 * W) == [TA]
    drift = GridSpec(cudem_geotransforms={TA: [X0, RES * (1 + 1e-3), 0, Y0, 0, -RES]},
                     cudem_sizes={TA: [W, H]})
    with raises(ValueError):                          # 0.064 px across 64 px
        Lattice(drift, [TA])


def test_mosaic_reads_across_tiles_without_resampling(tmp_path):
    """Catches: an off-by-one between tiles (a 1-px shift across every tile
    boundary in the network pass) and garbage outside the product."""
    importorskip("rasterio")
    from dem_pipeline.burn import Mosaic
    L = _lattice()
    zg = np.arange(H * (2 * W - OV), dtype="f4").reshape(H, -1)
    paths = _plant(tmp_path, L, zg)
    M = Mosaic(L, lambda t, k: paths[(t, k)])
    a = M.read("z", 5, 15, 50, 80)                    # spans the collar
    assert np.array_equal(a, zg[5:15, 50:80])
    out = M.read("z", -3, 2, -2, 3)
    assert np.isnan(out[:3]).all() and np.isnan(out[:, :2]).all()
    assert np.array_equal(out[3:, 2:], zg[0:2, 0:3])
    M.close()


def test_tier1_section_is_area_preserving():
    """Catches: a section whose mean depth is not D (the calibrated
    quantity is MEAN depth), or deepest-at-bank inversions."""
    importorskip("scipy")
    import shapely
    from dem_pipeline.burn import rasterize_tier1
    L = _lattice()
    xa, yt = L.xy(14, -100)
    xb, yb = L.xy(26, 300)
    poly = shapely.box(xa, yb, xb, yt)                # river band, rows 14..26
    J = np.arange(-50, 200)
    d = dict(t1_I=np.full(len(J), 20), t1_J=J, t1_wse=np.full(len(J), 10.0, "f4"),
             t1_D=np.full(len(J), 1.0, "f4"), t1_lat_ref=np.float64(40.0),
             t1_cpoly=np.frombuffer(shapely.to_wkb(poly), "u1"))
    bed = rasterize_tier1(L, TA, d)
    col = bed[:, 30]
    inside = np.isfinite(col)
    assert inside[14:27].all() and not inside[:13].any() and not inside[28:].any()
    assert col[20] == col[inside].min() and abs(col[20] - 8.5) < 0.05
    assert abs((10.0 - col[inside]).mean() - 1.0) < 0.15


def _diag_path_spec(L, thal=10.0, D=0.5, Wd=20.0):
    """One tier-2 path stepping diagonally (global rows 20..30, cols 20..30)."""
    I = np.arange(20, 31); J = np.arange(20, 31)
    return dict(t2_off=np.array([0, len(I)], "i8"), t2_I=I.astype("i8"), t2_J=J.astype("i8"),
                t2_thal=np.full(len(I), thal, "f4"), t2_D=np.array([D]), t2_Wd=np.array([Wd]),
                t2_lat=np.array([40.0]), t2_id=np.array([1], "i8"))


def test_tier2_lateral_guard_spares_high_ground_but_keeps_the_4connected_core():
    """Catches: the section digging into steep ground beside the path (the
    Downsville dam face was cut 46 m beside a path capped at the dam), and
    -- the prototype's regression -- a guard that drops the diagonal
    connectors so the channel is no longer 4-connected through high ground
    (a culvert embankment re-dams the edge statistics)."""
    importorskip("scipy")
    from scipy import ndimage as ndi
    from dem_pipeline.burn import rasterize_tier2
    L = _lattice()
    d = _diag_path_spec(L)
    z = np.full((H, W), 30.0, "f4")                   # everything is high ground
    t_free = rasterize_tier2(L, TA, d)
    t_g, n = rasterize_tier2(L, TA, d, z, 2.0)
    assert np.isfinite(t_free).sum() > 3 * np.isfinite(t_g).sum()   # width removed
    assert n == np.isfinite(t_free).sum() - np.isfinite(t_g).sum()
    burned = np.isfinite(t_g)
    for k in range(20, 31):
        assert burned[k, k]                            # path px always kept
    lab, nlab = ndi.label(burned, ndi.generate_binary_structure(2, 1))
    assert nlab == 1                                   # one 4-connected channel
    assert np.all(t_g[burned] <= 10.0 + 1e-6)          # core at thalweg level
    z2 = np.full((H, W), 10.6, "f4")                   # flat floodplain at bed level
    t_g2, n2 = rasterize_tier2(L, TA, d, z2, 2.0)
    assert n2 == 0 and np.array_equal(np.isfinite(t_g2), np.isfinite(t_free))


# ------------------------------------------------------------ network pass

def test_attach_ml_takes_divda_only_where_it_matches_v21():
    """Catches the 2026-11-23 region-02 bug: HR TotDASqKm routes the
    Delaware (16,900 km2) through the D&R Canal into the Millstone, so no
    V2.1 reach matched, the river got no Zarrabi depth and burned nothing --
    and the over-correction of using DivDASqKm everywhere, which shrinks
    braids/marsh creeks where V2.1 (and TotDASqKm) is right."""
    importorskip("geopandas")
    import geopandas as gpd
    import shapely
    from dem_pipeline.burn import attach_ml

    def line(k):                        # parallel lines 5 km apart
        x = -75.0 + 0.06 * k
        return shapely.LineString([(x, 40.0), (x, 40.01)])

    #           canal-inflated     braid (minor)      normal          no V2.1 near
    tot = [16905.0, 300.0, 300.0, 50.0]
    div = [650.0, 5.0, 300.0, 50.0]
    fl = gpd.GeoDataFrame({"NHDPlusID": [1, 2, 3, 4], "TotDASqKm": tot, "DivDASqKm": div},
                          geometry=[line(k) for k in range(4)], crs=4269)
    ml = gpd.GeoDataFrame({"COMID": [11, 12, 13], "TotDASqKM": [680.0, 310.0, 310.0],
                           "mf_depth": [1.2, 1.0, 1.0], "mf_width": [30.0, 25.0, 25.0],
                           "RF_Width_m": [28.0, 24.0, 24.0], "Tidal_Flag": [0, 0, 0]},
                          geometry=[line(k) for k in range(3)], crs=4269)
    out = attach_ml(fl, ml, 26918).set_index("NHDPlusID")
    assert out.loc[1, "DA"] == 650.0 and out.loc[1, "COMID"] == 11   # Div matched
    assert np.isfinite(out.loc[1, "mf_depth"])
    assert out.loc[2, "DA"] == 300.0 and out.loc[2, "COMID"] == 12   # Tot kept
    assert out.loc[3, "DA"] == 300.0
    assert out.loc[4, "DA"] == 50.0 and np.isnan(out.loc[4, "mf_depth"])


def test_network_line_mask_keeps_short_conduits_only():
    """Catches the 2026-11-23 road-crossing gaps: NHD culverts (connector
    334 / pipeline 428) were dropped, so the burn stopped at every culvert
    inlet and the embankment dammed the reach (6 m ponds on Plymouth Creek);
    and the over-reach of routing long conduits (buried streams, 54 km
    aqueducts) as surface trenches, or out-of-network lines."""
    import pandas as pd
    from dem_pipeline.burn import Params, network_line_mask
    P = Params.from_config({"burn": dict(PARAMS)})
    fl = pd.DataFrame({"FType": [460, 558, 334, 428, 334, 428, 336, 460],
                       "LengthKM": [1.0, 0.2, 0.012, 0.04, 1.5, 54.0, 0.5, 1.0],
                       "InNetwork": [1, 1, 1, 1, 1, 1, 1, 0]})
    assert network_line_mask(fl, P).tolist() == [True, True, True, True,
                                                 False, False, False, False]


def test_tier1_depth_is_never_nan():
    """Catches: a tier-1 reach whose DA window holds no Zarrabi depth
    getting D = NaN, which rasterizes to nothing (the Raritan went unburned
    silently); and a fallback that also overrides reaches that have one."""
    import pandas as pd
    from dem_pipeline.burn import Params, tier1_depth
    P = Params.from_config({"burn": dict(PARAMS)})
    t1p = pd.DataFrame({"DA": [500.0, 520.0, 5000.0], "mf_depth": [1.0, np.nan, np.nan],
                        "bf_c": [0.204] * 3, "bf_e": [0.321] * 3})
    D, n_fb = tier1_depth(t1p, P)
    assert np.isfinite(D).all() and n_fb == 1
    assert D[0] == D[1] == 0.5                         # smoothed from the neighbour
    assert abs(D[2] - 0.4 * 0.204 * 5000 ** 0.321) < 1e-9   # regional-curve fallback


def test_split_specs_refuses_non_finite_depth():
    """Catches: a NaN depth or width reaching a spec file (it would lower
    nothing, with no error anywhere downstream)."""
    from dem_pipeline.burn import split_specs
    L = _lattice()
    p = dict(id=7, I=np.array([10, 11]), J=np.array([10, 11]), thal=np.zeros(2, "f4"),
             D=float("nan"), Wd=5.0, lat=40.0)
    with raises(ValueError):
        split_specs(L, None, [p], 500.0)
    t1 = {"I": np.array([10]), "J": np.array([10]), "wse": np.array([1.0], "f4"),
          "D": np.array([np.nan], "f4")}
    with raises(ValueError):
        split_specs(L, t1, [], 500.0)


def test_tier2_network_caps_unmapped_embankment_and_cuts_at_road(tmp_path):
    """Catches: the culvert rule inverted or ignored end-to-end (deep cut
    without a mapped crossing / no cut at a road), and a bed that is not
    downstream-monotone across the embankment."""
    importorskip("skimage")
    import geopandas as gpd
    import shapely
    from dem_pipeline.burn import tier2_network
    L = _lattice()
    paths = _plant(tmp_path, L, _valley(L))
    fl = _lines(L)
    net = _net(tmp_path, L, paths)
    p1, cuts = tier2_network(net, fl, None)
    assert len(p1) == 1 and cuts.n_seg_capped.iloc[0] == 1
    assert cuts.max_cut.iloc[0] <= 2.0 + 1e-6
    (xr, yt), (_, yb) = L.xy(0, 61), L.xy(H - 1, 61)
    roads = gpd.GeoDataFrame(geometry=[shapely.LineString([(xr, yt), (xr, yb)])], crs=4269)
    net = _net(tmp_path, L, paths, roads)
    p2, cuts = tier2_network(net, fl, None)
    assert cuts.n_seg_capped.iloc[0] == 0 and cuts.n_seg_at_crossing.iloc[0] == 1
    assert cuts.max_cut.iloc[0] > 3.5
    # the same embankment under a dam keep-out: capped despite the road
    net.dams = gpd.GeoDataFrame(geometry=[shapely.Point(xr, L.xy(40, 61)[1]).buffer(30 * RES)],
                                crs=4269)
    _, cuts_d = tier2_network(net, fl, None)
    assert cuts_d.n_seg_at_dam.iloc[0] == 1 and cuts_d.n_seg_at_crossing.iloc[0] == 0
    assert cuts_d.max_cut.iloc[0] <= 2.0 + 1e-6
    # a mapped culvert (short NHD connector) is its own crossing: no road
    fl_c = fl.assign(conduit=[True])
    net = _net(tmp_path, L, paths)
    _, cuts_c = tier2_network(net, fl_c, None)
    assert cuts_c.n_seg_capped.iloc[0] == 0 and cuts_c.n_seg_at_crossing.iloc[0] == 1
    assert cuts_c.max_cut.iloc[0] > 3.5
    thal = p2[0]["thal"]
    assert np.all(np.diff(thal) <= 1e-5)
    assert np.all(np.abs(p2[0]["I"] - 40) <= 1)       # snapped to the valley floor


# ------------------------------------------------------------- end to end

def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run_e2e(tmp_path):
    from dem_pipeline import burn
    L = _lattice()
    zg = _valley(L, embankment=0.0)
    provg = np.ones(zg.shape, "u1")
    provg[:, 100:110] = 12                            # bathymetry block in tile B
    paths = _plant(tmp_path, L, zg, provg)
    net = _net(tmp_path, L, paths)
    p, _ = burn.tier2_network(net, _lines(L), None)
    specs = burn.split_specs(L, None, p, 500.0)
    b = _cfg(tmp_path)
    man = {}
    for t, sd in specs.items():
        sp = burn.spec_path(b, "0204", t)
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(sp, **sd)
        man[t] = burn._spec_digest(sd)
    burn._atomic_json(burn.network_manifest(b, "0204"),
                      {"fingerprint": "x", "tiles": man, "report": {}})
    return b, L, paths, zg


def test_burn_end_to_end_seam_guard_and_resume(tmp_path):
    """Catches: tiles sharing a collar burned differently (a seam in the
    subgrid edge statistics), the merged product being modified, bathymetry
    lowered, the stats file written before the outputs, and a resumed run
    either redoing finished tiles or trusting a tile whose input changed."""
    importorskip("skimage")
    skip_unless_cli("gdal_translate")
    from dem_pipeline import burn, merge
    b, L, paths, zg = _run_e2e(tmp_path)
    before = {k: _sha(v) for k, v in paths.items()}
    spec = _spec()
    sa = burn.burn_tile(b, TA, spec, L)
    sb = burn.burn_tile(b, TB, spec, L)
    assert sa["mode"] == "burn" and sb["mode"] == "burn"
    assert {k: _sha(v) for k, v in paths.items()} == before
    za = merge._read(burn.out_tile(b, TA), "f4")[0]
    zb = merge._read(burn.out_tile(b, TB), "f4")[0]
    pa = merge._read(burn.out_tile(b, TA, "_source"), "u1")[0]
    pb = merge._read(burn.out_tile(b, TB, "_source"), "u1")[0]
    assert np.array_equal(za[:, W - OV:], zb[:, :OV])          # collar agrees
    assert (za[:, W - OV:] < zg[:, W - OV:W]).any()            # and was burned
    assert np.all(pb[:, 100 - (W - OV):110 - (W - OV)] == 12)  # bathy untouched
    assert np.all(zb[:, 40:50] == zg[:, 100:110])
    assert np.all(za <= zg[:, :W] + 1e-6)
    dep = merge._read(burn.out_tile(b, TA, "_burndepth"), "f4")[0]
    assert np.allclose(dep, zg[:, :W] - za, atol=1e-5)
    assert set(np.unique(pa)) <= {1, 16} and (pa == 16).sum() == sa["n_lowered"]
    # resume: untouched -> skipped (outputs not rewritten)
    m0 = burn.out_tile(b, TA).stat().st_mtime_ns
    assert burn.burn_tile(b, TA, spec, L)["fingerprint"] == sa["fingerprint"]
    assert burn.out_tile(b, TA).stat().st_mtime_ns == m0
    # an input change (merge rerun) invalidates exactly that tile
    import os
    p = paths[(TA, "z")]
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 10**9))
    assert burn.burn_tile(b, TA, spec, L)["fingerprint"] != sa["fingerprint"]


def test_load_specs_rejects_spec_not_matching_manifest(tmp_path):
    """Catches: a spec file left by an interrupted network run being burned
    instead of the one its manifest describes."""
    importorskip("skimage")
    from dem_pipeline import burn
    b, L, paths, zg = _run_e2e(tmp_path)
    assert set(burn.load_specs(b, TA)) == {"0204"}
    sp = burn.spec_path(b, "0204", TA)
    with np.load(sp) as z:
        d = {k: z[k] for k in z.files}
    d["t2_thal"] = d["t2_thal"] - 1.0
    np.savez(sp, **d)
    with raises(RuntimeError):
        burn.load_specs(b, TA)
    # a tile absent from the manifest gets no spec even if a stray file exists
    man = json.loads(burn.network_manifest(b, "0204").read_text())
    man["tiles"].pop(TB)
    burn._atomic_json(burn.network_manifest(b, "0204"), man)
    assert burn.load_specs(b, TB) == {}
