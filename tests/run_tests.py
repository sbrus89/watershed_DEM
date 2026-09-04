#!/usr/bin/env python
"""Minimal fallback test runner for environments without pytest.

Prefer `pytest` (included in the de-dem conda env); this runner exists so the
suite can be executed with nothing but the pipeline's own dependencies:

    python tests/run_tests.py [substring ...]

It discovers tests/test_*.py, runs every test_* function, and supplies
pytest-compatible `tmp_path` / `monkeypatch` arguments. Skips raised as
unittest.SkipTest are honored (tests/util.py uses exactly that, which pytest
also understands natively).
"""

from __future__ import annotations

import importlib
import inspect
import sys
import tempfile
import traceback
import unittest
from pathlib import Path


class MonkeyPatch:
    """Subset of pytest's monkeypatch: setattr with automatic undo."""

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self._undo):
            setattr(obj, name, value)
        self._undo.clear()


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    patterns = argv or [""]

    passed = failed = skipped = 0
    failures: list[str] = []
    for mod_path in sorted((root / "tests").glob("test_*.py")):
        try:
            mod = importlib.import_module(f"tests.{mod_path.stem}")
        except unittest.SkipTest as e:
            print(f"SKIP  {mod_path.stem} (module: {e})")
            skipped += 1
            continue
        except Exception:
            # a broken module (like a pytest collection error) must not take
            # down the rest of the suite
            failed += 1
            failures.append(f"{mod_path.stem} (import)")
            print(f"FAIL  {mod_path.stem} (module import)")
            traceback.print_exc()
            continue
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            label = f"{mod_path.stem}::{name}"
            if not any(p in label for p in patterns):
                continue
            params = inspect.signature(fn).parameters
            kwargs, mp, tmp = {}, None, None
            if "tmp_path" in params:
                tmp = tempfile.TemporaryDirectory()
                kwargs["tmp_path"] = Path(tmp.name)
            if "monkeypatch" in params:
                mp = MonkeyPatch()
                kwargs["monkeypatch"] = mp
            try:
                fn(**kwargs)
                passed += 1
                print(f"PASS  {label}")
            except unittest.SkipTest as e:
                skipped += 1
                print(f"SKIP  {label} ({e})")
            except Exception:
                failed += 1
                failures.append(label)
                print(f"FAIL  {label}")
                traceback.print_exc()
            finally:
                if mp:
                    mp.undo()
                if tmp:
                    tmp.cleanup()

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    if failures:
        print("failed:", *failures, sep="\n  ")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
