"""Manifest-driven, resumable, concurrent downloader.

Every fetch writes to <dest>.part and atomically renames on completion, so a
killed run resumes cleanly. Completed files are skipped when the size matches
the manifest (or any size, if the manifest has none), making stage 04 fully
idempotent.
"""

from __future__ import annotations

import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from . import load_config

log = logging.getLogger(__name__)

MANIFEST_FIELDS = ["url", "dest", "size", "source", "tile", "bbox"]


def _object_size(resp) -> int | None:
    """Size of the WHOLE object per the server, or None if unstated.

    Content-Range is authoritative on 206/416 ("bytes 0-99/1000", "bytes
    */1000"). Content-Length only describes the whole object on a 200 -- on a
    206 it is the length of that chunk, so reading it there would truncate.
    """
    cr = resp.headers.get("Content-Range", "")
    if "/" in cr:
        total = cr.rsplit("/", 1)[1].strip()
        if total.isdigit():
            return int(total)
    if resp.status_code == 200:
        cl = resp.headers.get("Content-Length", "")
        if cl.isdigit():
            return int(cl)
    return None


def _head_object_size(url: str, timeout: int) -> int | None:
    try:
        h = requests.head(url, allow_redirects=True, timeout=timeout)
        cl = h.headers.get("Content-Length", "")
        return int(cl) if cl.isdigit() else None
    except requests.RequestException:
        return None


def fetch(url: str, dest: Path, expected_size: int | None = None) -> Path:
    """Download one file with Range-based resume and retries.

    Completeness is judged against the SERVER's byte count, never against
    `expected_size`. Catalog metadata is routinely wrong -- TNM's sizes for the
    S1M tiles are off by tens of bytes in BOTH directions -- and gating on it
    rejects fully-downloaded files forever: a complete 400 MB tile was
    discarded and re-fetched five times before the run gave up. `expected_size`
    is kept as a cross-check and logged on disagreement, so catalog drift stays
    visible instead of becoming fatal.
    """
    cfg = load_config()["download"]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        got = dest.stat().st_size
        if expected_size is None or got == expected_size:
            return dest
        # The manifest disagrees -- but catalog metadata is often wrong, so ask
        # the server before throwing away a file that may be perfectly good.
        total = _head_object_size(url, cfg["timeout_s"])
        if total is not None and got == total:
            log.warning("%s: manifest size %d disagrees with the server's %d "
                        "-- keeping the file on disk (catalog drift)",
                        dest.name, expected_size, total)
            return dest
        log.warning("size mismatch for %s (%d on disk, server reports %s), "
                    "re-downloading", dest.name, got, total)
        dest.unlink()

    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(cfg["retries"]):
        try:
            headers = {}
            mode = "wb"
            if part.exists():
                headers["Range"] = f"bytes={part.stat().st_size}-"
                mode = "ab"
            total = None
            with requests.get(
                url, headers=headers, stream=True, timeout=cfg["timeout_s"]
            ) as r:
                if r.status_code == 416:  # range at/past EOF
                    total = _object_size(r) or _head_object_size(
                        url, cfg["timeout_s"])
                    got = part.stat().st_size
                    if total is None or got == total:
                        part.rename(dest)
                        return dest
                    # genuinely oversized partial: discard and start over
                    part.unlink()
                    raise IOError(
                        f"{url}: partial is {got} bytes but the server reports "
                        f"{total}; restarting")
                if "Range" in headers and r.status_code != 206:
                    # server ignored the range request; start over
                    part.unlink()
                    mode = "wb"
                r.raise_for_status()
                total = _object_size(r)
                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            got = part.stat().st_size
            if total is not None and got != total:
                raise IOError(f"{url}: got {got} bytes, server said {total}")
            if (expected_size is not None and total is not None
                    and expected_size != total):
                log.warning("%s: manifest size %d disagrees with the server's "
                            "%d -- trusting the server (catalog drift)",
                            dest.name, expected_size, total)
            part.rename(dest)
            return dest
        except (requests.RequestException, IOError) as e:
            wait = 2**attempt
            log.warning("download failed (%s), retry in %ds: %s", e, wait, url)
            time.sleep(wait)
    raise RuntimeError(f"download failed after {cfg['retries']} tries: {url}")


def read_manifest(manifest_csv: Path) -> list[dict]:
    with open(manifest_csv, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict], manifest_csv: Path) -> None:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_csv.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
    tmp.rename(manifest_csv)


def fetch_manifest(
    manifest_csv: Path,
    sources: list[str] | None = None,
    workers: int | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> list[Path]:
    """Download manifest rows, optionally filtered by source name and/or a
    (w, s, e, n) bounds intersection test. Rows without a bbox (vector
    bundles like WBD/NHD) always pass the bounds filter."""
    cfg = load_config()["download"]
    rows = read_manifest(manifest_csv)
    if sources:
        rows = [r for r in rows if r["source"] in sources]
    if bounds:
        w, s, e, n = bounds

        def hits(r: dict) -> bool:
            if not r.get("bbox"):
                return True
            bw, bs, be, bn = (float(v) for v in r["bbox"].split())
            return bw < e and be > w and bs < n and bn > s

        rows = [r for r in rows if hits(r)]

    results: list[Path] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers or cfg["workers"]) as ex:
        futs = {
            ex.submit(
                fetch,
                r["url"],
                Path(r["dest"]),
                int(r["size"]) if r.get("size") else None,
            ): r["url"]
            for r in rows
        }
        for fut in tqdm(as_completed(futs), total=len(futs), unit="file"):
            try:
                results.append(fut.result())
            except Exception as e:
                log.error("FAILED %s: %s", futs[fut], e)
                failures.append(futs[fut])
    if failures:
        raise RuntimeError(
            f"{len(failures)} downloads failed (rerun to resume): "
            + "\n".join(failures[:10])
        )
    return results
