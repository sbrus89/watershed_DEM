"""Tests for the resumable downloader and manifest handling.

fetch()/fetch_manifest() read their settings through the module's own
load_config, which every test patches to a self-contained dict so the suite
never depends on (or mutates behavior with) the real config/sources.yaml.
"""

import http.server
import threading
from pathlib import Path

import dem_pipeline.download as dl
from dem_pipeline.download import (MANIFEST_FIELDS, fetch, fetch_manifest,
                                   read_manifest, write_manifest)
from tests.util import raises, skip

TEST_CFG = {"download": {"workers": 2, "retries": 3, "timeout_s": 10}}


def _patch_cfg(monkeypatch):
    monkeypatch.setattr(dl, "load_config", lambda: TEST_CFG)


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Static payload server with Range support; counts requests."""

    payload = b""
    requests_seen = None  # list, injected per test

    def do_GET(self):
        self.requests_seen.append(self.headers.get("Range"))
        rng = self.headers.get("Range")
        body = self.payload
        if rng:
            start = int(rng.split("=")[1].rstrip("-").split("-")[0])
            if start >= len(body):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(body)}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range",
                             f"bytes {start}-{len(body)-1}/{len(body)}")
            body = body[start:]
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()

    def log_message(self, *args):
        pass


def _serve(payload: bytes):
    seen: list = []
    handler = type("H", (_RangeHandler,),
                   {"payload": payload, "requests_seen": seen})
    try:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    except (OSError, PermissionError):
        skip("cannot bind a localhost test server in this environment")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/file.bin"
    return srv, url, seen


def test_fetch_full_download(tmp_path, monkeypatch):
    _patch_cfg(monkeypatch)
    payload = bytes(range(256)) * 400  # 100 KiB
    srv, url, seen = _serve(payload)
    try:
        dest = tmp_path / "file.bin"
        out = fetch(url, dest, expected_size=len(payload))
        assert out.read_bytes() == payload
        assert not dest.with_suffix(".bin.part").exists()
    finally:
        srv.shutdown()


def test_fetch_resumes_partial(tmp_path, monkeypatch):
    _patch_cfg(monkeypatch)
    payload = bytes(range(256)) * 400
    srv, url, seen = _serve(payload)
    try:
        dest = tmp_path / "file.bin"
        part = dest.with_suffix(".bin.part")
        part.write_bytes(payload[:40_000])  # simulate an interrupted run
        out = fetch(url, dest, expected_size=len(payload))
        assert out.read_bytes() == payload
        assert seen[0] == "bytes=40000-"  # resumed, not restarted
    finally:
        srv.shutdown()


def test_fetch_skips_complete_file(tmp_path, monkeypatch):
    _patch_cfg(monkeypatch)
    payload = b"x" * 1000
    srv, url, seen = _serve(payload)
    try:
        dest = tmp_path / "file.bin"
        dest.write_bytes(payload)
        fetch(url, dest, expected_size=len(payload))
        assert seen == []  # no request made at all
    finally:
        srv.shutdown()


def test_fetch_redownloads_on_size_mismatch(tmp_path, monkeypatch):
    _patch_cfg(monkeypatch)
    payload = b"y" * 1000
    srv, url, seen = _serve(payload)
    try:
        dest = tmp_path / "file.bin"
        dest.write_bytes(b"stale")  # wrong size on disk
        out = fetch(url, dest, expected_size=len(payload))
        assert out.read_bytes() == payload and len(seen) == 1
    finally:
        srv.shutdown()


def test_fetch_discards_oversized_stale_partial(tmp_path, monkeypatch):
    """A .part longer than the payload triggers 416; fetch must not accept
    it as complete when the size disagrees -- it restarts from scratch."""
    _patch_cfg(monkeypatch)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    payload = b"z" * 1000
    srv, url, seen = _serve(payload)
    try:
        dest = tmp_path / "file.bin"
        part = dest.with_suffix(".bin.part")
        part.write_bytes(b"w" * 1500)  # stale, larger than the real file
        out = fetch(url, dest, expected_size=len(payload))
        assert out.read_bytes() == payload
        assert seen[0] == "bytes=1500-"   # 416 round trip
        assert seen[-1] is None            # then a clean full download
    finally:
        srv.shutdown()


def test_fetch_416_accepts_complete_partial(tmp_path, monkeypatch):
    _patch_cfg(monkeypatch)
    payload = b"q" * 1000
    srv, url, seen = _serve(payload)
    try:
        dest = tmp_path / "file.bin"
        dest.with_suffix(".bin.part").write_bytes(payload)  # fully downloaded
        out = fetch(url, dest, expected_size=len(payload))
        assert out.read_bytes() == payload and seen == ["bytes=1000-"]
    finally:
        srv.shutdown()


def test_fetch_gives_up_after_retries(tmp_path, monkeypatch):
    _patch_cfg(monkeypatch)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    with raises(RuntimeError):
        fetch("http://127.0.0.1:1/nope.bin", tmp_path / "nope.bin",
              expected_size=10)


def test_manifest_roundtrip(tmp_path):
    rows = [
        {"url": "http://x/a.tif", "dest": "d/a.tif", "size": "10",
         "source": "s1m", "tile": "", "bbox": "-75.3 39.8 -75.2 39.9"},
        {"url": "http://x/wbd.zip", "dest": "d/wbd.zip", "size": "",
         "source": "wbd", "tile": "", "bbox": ""},
    ]
    m = tmp_path / "m.csv"
    write_manifest(rows, m)
    back = read_manifest(m)
    assert [r["url"] for r in back] == [r["url"] for r in rows]
    assert list(back[0].keys()) == MANIFEST_FIELDS


def _filter_manifest(tmp_path):
    """Manifest exercising both filters: bbox rows in/out of the test tile,
    a bbox-less vector bundle, and a second source."""
    rows = [
        {"url": "http://x/in.tif", "dest": str(tmp_path / "d/in.tif"),
         "size": "10", "source": "s1m", "tile": "",
         "bbox": "-75.3 39.8 -75.2 39.9"},          # intersects test tile
        {"url": "http://x/out.tif", "dest": str(tmp_path / "d/out.tif"),
         "size": "", "source": "s1m", "tile": "",
         "bbox": "-74.0 41.0 -73.9 41.1"},          # far away
        {"url": "http://x/wbd.zip", "dest": str(tmp_path / "d/wbd.zip"),
         "size": "", "source": "wbd", "tile": "", "bbox": ""},  # no bbox
        {"url": "http://x/eh.zip", "dest": str(tmp_path / "d/eh.zip"),
         "size": "7", "source": "ehydro", "tile": "",
         "bbox": "-75.3 39.8 -75.2 39.9"},
    ]
    m = tmp_path / "m.csv"
    write_manifest(rows, m)
    return m


def test_fetch_manifest_bounds_filter_calls_production_code(
        tmp_path, monkeypatch):
    """Drive the REAL fetch_manifest filters (not a re-implementation) with
    fetch stubbed to record calls."""
    _patch_cfg(monkeypatch)
    calls = []
    monkeypatch.setattr(
        dl, "fetch",
        lambda url, dest, expected_size=None:
            calls.append((url, str(dest), expected_size)) or Path(dest))

    m = _filter_manifest(tmp_path)
    bounds = (-75.25, 39.75, -75.0, 40.0)  # tile n40x00_w075x25

    fetch_manifest(m, bounds=bounds)
    got = sorted(c[0] for c in calls)
    # bbox intersecting + bbox-less rows pass; the far tile is excluded
    assert got == ["http://x/eh.zip", "http://x/in.tif", "http://x/wbd.zip"]
    # size column maps to expected_size int, empty size to None
    by_url = {c[0]: c[2] for c in calls}
    assert by_url["http://x/in.tif"] == 10
    assert by_url["http://x/wbd.zip"] is None

    calls.clear()
    fetch_manifest(m, sources=["ehydro"], bounds=bounds)
    assert [c[0] for c in calls] == ["http://x/eh.zip"]


def test_fetch_manifest_aggregates_failures(tmp_path, monkeypatch):
    _patch_cfg(monkeypatch)

    def flaky_fetch(url, dest, expected_size=None):
        if "out.tif" in url:
            raise RuntimeError("boom")
        return Path(dest)

    monkeypatch.setattr(dl, "fetch", flaky_fetch)
    m = _filter_manifest(tmp_path)
    with raises(RuntimeError):
        fetch_manifest(m)  # unfiltered: includes the failing row


def test_wrong_manifest_size_does_not_reject_a_good_download(tmp_path,
                                                             monkeypatch):
    """Catalog metadata is not proof of completeness.

    TNM reports S1M tile sizes that are wrong by tens of bytes in both
    directions. Gating on `expected_size` made fetch discard a fully
    downloaded 400 MB tile and re-fetch it until the retries ran out, so
    every S1M download failed. The server's byte count is the authority.
    """
    _patch_cfg(monkeypatch)
    payload = b"x" * 5000
    srv, url, seen = _serve(payload)
    try:
        dest = tmp_path / "tile.tif"
        # manifest claims 37 bytes more than the object actually has
        out = fetch(url, dest, expected_size=len(payload) + 37)
        assert out.read_bytes() == payload
        assert not list(tmp_path.glob("*.part"))
    finally:
        srv.shutdown()


def test_complete_partial_is_kept_when_the_manifest_size_is_wrong(tmp_path,
                                                                  monkeypatch):
    """The exact S1M failure: a .part already at EOF, plus a bad manifest size.

    The resume probe gets a 416; the old code compared the partial against
    `expected_size`, called it stale, deleted it and started over -- throwing
    away a complete file on every retry.
    """
    _patch_cfg(monkeypatch)
    payload = b"y" * 4096
    srv, url, seen = _serve(payload)
    try:
        dest = tmp_path / "tile.tif"
        part = dest.with_suffix(".tif.part")
        part.write_bytes(payload)                     # already complete
        out = fetch(url, dest, expected_size=len(payload) - 12)
        assert out.read_bytes() == payload
        assert not part.exists()
        # kept, not re-downloaded: the only request was the resume probe
        assert len(seen) == 1 and seen[0] is not None
    finally:
        srv.shutdown()


def test_short_transfer_still_fails(tmp_path, monkeypatch):
    """Relaxing the manifest check must not remove integrity checking.

    A server that promises more bytes than it delivers has to be caught, or
    truncated rasters enter the composite as valid data.
    """
    _patch_cfg(monkeypatch)

    class _Lying(_RangeHandler):
        payload = b"z" * 100
        requests_seen = []

        def do_GET(self):
            self.requests_seen.append(None)
            self.send_response(200)
            self.send_header("Content-Length", "999")   # lies
            self.end_headers()
            self.wfile.write(self.payload)

    import http.server
    try:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Lying)
    except (OSError, PermissionError):
        skip("cannot bind a localhost socket in this environment")
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/f.tif"
    try:
        with raises(RuntimeError):
            fetch(url, tmp_path / "f.tif")
        assert not (tmp_path / "f.tif").exists()
    finally:
        srv.shutdown()
