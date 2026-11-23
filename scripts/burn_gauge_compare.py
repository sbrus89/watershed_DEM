#!/usr/bin/env python
"""Compare the channel burn with USGS gauge channel measurements.

For each gauge in a gauges CSV (site_no, lon, lat, huc4, d_obs, w_obs, ...:
the region-wide gauge check's output, observed mean depth A/W and width at
mean annual flow), snap to the nearest burn-spec channel sample within
--snap-m, and report

  model_D   the burn's mean-depth target there (tier 1: t1_D of the nearest
            centreline sample; tier 2: the path's D)  -- same quantity as
            d_obs (mean depth at mean flow below the lidar water surface)
  model_W   tier-2 burn width (tier 1 is the river polygon: not reported)
  max_cut   the deepest burn within --radius-m of the snapped point, from
            the _burndepth.tif sidecar (a parabolic section's maximum is
            1.5 D, so max_cut < 1.5 model_D means guards/caps reduced it)
  status    burned | no_model_depth (the nearest spec sample has D = NaN,
            i.e. the burn has no depth there and lowers nothing) | not_lowered (spec there but nothing lowered, e.g.
            EAARL-B / bathymetry / guarded) | no_network (no spec sample
            within --snap-m: tidal, tier-2 DA cut-off, or off the network)

and write <burn output root>/qa/gauge_compare.geojson, which
scripts/viewer.py --burn-config shows as a point layer. Read-only on the
product; prints a summary of model_D / d_obs.

Usage: python scripts/burn_gauge_compare.py --gauges GAUGES_CSV
           [--config config/burn_region02.yaml] [--snap-m 100] [--radius-m 30]
"""

from __future__ import annotations

import argparse
import json
import logging
import math

import numpy as np
import pandas as pd

import dem_pipeline as dp
from dem_pipeline import burn, merge

log = logging.getLogger("burn_gauge_compare")


def _candidates(d: dict):
    """(I, J, tier, D, W, reach id) of every spec sample in one tile spec."""
    I, J, T, D, W, R = [], [], [], [], [], []
    if "t1_I" in d and len(d["t1_I"]):
        n = len(d["t1_I"])
        I.append(d["t1_I"]); J.append(d["t1_J"]); T.append(np.full(n, 1))
        D.append(d["t1_D"]); W.append(np.full(n, np.nan)); R.append(np.full(n, -1))
    if "t2_off" in d:
        off = d["t2_off"]
        n = np.diff(off)
        I.append(d["t2_I"]); J.append(d["t2_J"]); T.append(np.full(n.sum(), 2))
        D.append(np.repeat(d["t2_D"], n)); W.append(np.repeat(d["t2_Wd"], n))
        R.append(np.repeat(d["t2_id"], n))
    if not I:
        return None
    return tuple(np.concatenate(a) for a in (I, J, T, D, W, R))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(dp.path("config", "burn_region02.yaml")))
    ap.add_argument("--gauges", required=True)
    ap.add_argument("--snap-m", type=float, default=100.0)
    ap.add_argument("--radius-m", type=float, default=30.0)
    args = ap.parse_args()
    dp.setup_logging()

    import rasterio
    from rasterio.windows import Window

    b = burn.load_burn_config(args.config)
    spec, tiles = burn.load_input(b)
    L = burn.Lattice(spec, tiles)
    ids = merge.merged_source_ids(merge.load_basins(b["_merge"]))
    ids = {**ids, "burn": int(b["burn"]["burn_id"])}
    id2name = {v: k for k, v in ids.items()}

    g = pd.read_csv(args.gauges, dtype={"site_no": str, "huc4": str})
    gi, gj = L.rowcol(g["lon"].values, g["lat"].values)
    feats, spec_cache = [], {}
    for k, row in g.iterrows():
        I0, J0 = int(gi[k]), int(gj[k])
        tl = L.tiles_overlapping(I0, I0 + 1, J0, J0 + 1)
        props = {c: (None if pd.isna(row[c]) else row[c]) for c in
                 ("site_no", "name", "huc4", "da_km2", "d_obs", "w_obs", "n_win",
                  "q_mean", "SECTION")}
        props.update(tile=tl[0] if tl else None, status="no_network", tier=None,
                     model_D=None, model_W=None, snap_m=None, max_cut=None,
                     prov=None, diff=None, ratio=None)
        if tl:
            t = tl[0]
            if t not in spec_cache:
                spec_cache[t] = burn.load_specs(b, t)
            specs = spec_cache[t]
            # the gauge's own basin network (a merged pixel is only burned by
            # the network of the basin that owns it); any basin as fallback
            cands = [specs[row["huc4"]]] if row["huc4"] in specs else list(specs.values())
            best = None
            dx, dy = burn.pixel_m(L.res, float(row["lat"]))
            for d in cands:
                c = _candidates(d)
                if c is None:
                    continue
                dist = np.hypot((c[0] - I0) * dy, (c[1] - J0) * dx)
                n = int(np.argmin(dist))
                if best is None or dist[n] < best[0]:
                    best = (float(dist[n]),) + tuple(a[n] for a in c)
            if best is not None and best[0] <= args.snap_m:
                dist, Is, Js, tier, D, W, rid = best
                props.update(tier=int(tier),
                             model_D=round(float(D), 3) if np.isfinite(D) else None,
                             model_W=None if not np.isfinite(W) else round(float(W), 1),
                             snap_m=round(dist, 1), reach_id=int(rid))
                ts = L.tiles_overlapping(Is, Is + 1, Js, Js + 1)[0]
                r0, _, c0, _ = L.window(ts)
                ri, ci = int(Is) - r0, int(Js) - c0
                rad = max(1, math.ceil(args.radius_m / min(dx, dy)))
                with rasterio.open(burn.out_tile(b, ts, "_source")) as src:
                    props["prov"] = id2name.get(
                        int(src.read(1, window=Window(ci, ri, 1, 1))[0, 0]), "?")
                bp = burn.out_tile(b, ts, "_burndepth")
                mc = 0.0
                if bp.exists():
                    with rasterio.open(bp) as src:
                        win = Window(ci - rad, ri - rad, 2 * rad + 1, 2 * rad + 1)
                        a = src.read(1, window=win, boundless=True, fill_value=0)
                        mc = float(np.max(a))
                props["max_cut"] = round(mc, 3)
                props["status"] = ("no_model_depth" if not np.isfinite(D) else
                                   "burned" if mc > 0.01 else "not_lowered")
                if props["d_obs"] and np.isfinite(D):
                    props["diff"] = round(float(D) - props["d_obs"], 3)
                    props["ratio"] = round(float(D) / props["d_obs"], 3)
                props["snap_lon"], props["snap_lat"] = (round(float(v), 6) for v in L.xy(Is, Js))
        props = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in props.items()}
        # browsers' JSON.parse rejects NaN/Infinity (Python's json accepts them)
        props = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                 for k, v in props.items()}
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Point",
                                   "coordinates": [float(row["lon"]), float(row["lat"])]}})

    out = burn.out_root(b) / "qa" / "gauge_compare.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                              allow_nan=False))
    tmp.rename(out)

    P = pd.DataFrame([f["properties"] for f in feats])
    log.info("wrote %s (%d gauges)", out, len(P))
    print(P["status"].value_counts().to_string())
    s = P[P["status"] == "burned"]
    print(f"\nburned gauges: median model_D/d_obs {s['ratio'].median():.2f}, "
          f"median diff {s['diff'].median():+.3f} m, "
          f"log10 sd {np.log10(s['ratio']).std():.2f}")
    for key in ("tier", "huc4"):
        print(s.groupby(key).agg(n=("ratio", "size"), ratio=("ratio", "median"),
                                 diff=("diff", "median"),
                                 d_obs=("d_obs", "median"),
                                 model_D=("model_D", "median")).round(3).to_string())
    print("\nnot_lowered by provenance:")
    print(P[P["status"] == "not_lowered"]["prov"].value_counts().to_string())


if __name__ == "__main__":
    main()
