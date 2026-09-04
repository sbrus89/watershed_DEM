"""Client for The National Map (TNM) Access API product queries.

Responses are cached under manifests/cache/ so discovery reruns are free and
the pipeline stays reproducible; delete the cache to force re-discovery.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import requests

from . import load_config, path as repo_path

log = logging.getLogger(__name__)

CACHE_DIR = repo_path("manifests", "cache")


def _cached_get(url: str, params: dict, retries: int, timeout: int) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        json.dumps([url, sorted(params.items())]).encode()
    ).hexdigest()[:24]
    cache_file = CACHE_DIR / f"tnm_{key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            tmp = cache_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f)
            tmp.rename(cache_file)
            return data
        except (requests.RequestException, ValueError) as e:
            last_err = e
            wait = 2**attempt
            log.warning("TNM request failed (%s), retry in %ds", e, wait)
            time.sleep(wait)
    raise RuntimeError(f"TNM API request failed after {retries} tries: {last_err}")


def query_products(
    dataset: str,
    bbox: tuple[float, float, float, float] | None = None,
    polygon_wkt: str | None = None,
    extra_params: dict | None = None,
) -> list[dict]:
    """Return all product items for a dataset within bbox or polygon.

    ``polygon_wkt`` is preferred when available (server-side clip to the
    basin); the TNM API rejects very long polygon strings, so callers should
    pass a simplified ring and post-filter locally. Falls back to bbox.
    """
    cfg = load_config()
    api = cfg["tnm_api"]
    dl = cfg["download"]
    url = f"{api['base']}/products"
    page = api["page_size"]

    params: dict = {"datasets": dataset, "outputFormat": "JSON", "max": page}
    if polygon_wkt:
        params["polygon"] = polygon_wkt
    elif bbox:
        params["bbox"] = ",".join(str(v) for v in bbox)
    else:
        raise ValueError("bbox or polygon_wkt required")
    if extra_params:
        params.update(extra_params)

    items: list[dict] = []
    offset = 0
    while True:
        data = _cached_get(
            url, {**params, "offset": offset}, dl["retries"], dl["timeout_s"]
        )
        batch = data.get("items", [])
        items.extend(batch)
        total = data.get("total", len(items))
        offset += len(batch)
        log.info("%s: %d/%d items", dataset, len(items), total)
        if not batch or offset >= total:
            break
    return items


def simplify_polygon_for_api(geom, max_chars: int = 6000) -> str | None:
    """Simplify a shapely polygon until its WKT-ish TNM string fits the URL.

    TNM wants 'lon lat,lon lat,...'. Returns None if it can't fit even a
    heavily simplified ring (caller should fall back to bbox).
    """
    from shapely.geometry import MultiPolygon

    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    for tol in (0.001, 0.005, 0.01, 0.02, 0.05):
        ring = geom.simplify(tol).exterior
        s = ",".join(f"{x:.4f} {y:.4f}" for x, y in ring.coords)
        if len(s) <= max_chars:
            return s
    return None
