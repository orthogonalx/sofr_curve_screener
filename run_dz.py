#!/usr/bin/env python
"""
DataZone terminal launcher.

bbgapi on the desk is usually a Python 3.7 DataZone package. The JupyterLab
shell often starts Python 3.10, where ms.version.addpkg('bbgapi') cannot import.
This launcher re-execs under Python 3.7 when needed, then bootstraps packages.

  python run_dz.py prediction --once --pull-history
  python run_dz.py prediction --once
  python run_dz.py backtest
  python run_dz.py pipeline
"""

from __future__ import annotations

import glob
import os
import runpy
import sys


def _py37_candidates():
    pats = [
        "/ms/dist/python/PROJ/core/3.7*/bin/python",
        "/ms/dist/python/PROJ/core/3.7*/exec/bin/python",
        "/ms/dist/python/PROJ/core/3.7*/bin/python3",
        "/ms/dist/datazone/PROJ/venv*py3.7*/prod/venv/bin/python",
        "/ms/dist/datazone/PROJ/venv*py37*/prod/venv/bin/python",
    ]
    out = []
    for pat in pats:
        out.extend(sorted(glob.glob(pat)))
    # unique, prefer higher patch versions last (then pick last)
    seen = []
    for p in out:
        if p not in seen and os.path.isfile(p) and os.access(p, os.X_OK):
            seen.append(p)
    return seen


def _maybe_reexec_for_bbgapi() -> None:
    """If not on 3.7, switch to a DataZone Python 3.7 that can load bbgapi."""
    if os.environ.get("SOFR_DZ_REEXEC") == "1":
        return
    if sys.version_info[:2] == (3, 7):
        return

    cands = _py37_candidates()
    if not cands:
        print(
            "  [dz] warning: running Python %s.%s; bbgapi usually needs 3.7 and "
            "no py3.7 interpreter was found under /ms/dist/python/PROJ/core/"
            % sys.version_info[:2],
            flush=True,
        )
        return

    py = cands[-1]
    print(
        "  [dz] re-exec under Python 3.7 for bbgapi:\n       %s" % py,
        flush=True,
    )
    os.environ["SOFR_DZ_REEXEC"] = "1"
    os.execv(py, [py, os.path.abspath(__file__)] + sys.argv[1:])


def _bootstrap() -> None:
    try:
        import ms.version
    except ImportError:
        print(
            "ERROR: this python has no ms.version.\n"
            "In a working notebook:  import sys; print(sys.executable)\n"
            "Then:  /that/python run_dz.py prediction --once --pull-history",
            flush=True,
        )
        raise SystemExit(2)

    pkgs = [
        ("pandas", "1.3.4", {}),
        ("numpy", "1.21.2", {}),
        ("pytz", "2022.1", {}),
        ("dateutil", "2.8.0", {}),
        ("six", "1.12.0", {}),
        ("openpyxl", "3.0.9", {}),
        ("urllib3", "1.26.7", {}),
        ("certifi", "2021.10.8", {}),
        ("idna", "3.3", {}),
        ("chardet", "4.0.0", {}),
        ("charset-normalizer", "2.0.7", {}),
        ("requests", "2.26.0", {"shadow": True}),
        ("toml", "0.10.1", {}),
        ("retrying", "1.3.3", {}),
    ]
    for name, ver, kw in pkgs:
        try:
            ms.version.addpkg(name, ver, **kw)
        except Exception as exc:  # noqa: BLE001
            print("  [dz] skip %s: %s" % (name, exc), flush=True)

    # Pre-load bbgapi here so failures are obvious before prediction starts
    try:
        from data import _ensure_bbgapi

        _ensure_bbgapi()
        import bbgapi  # noqa: F401

        print("  [dz] bbgapi OK  file=%s" % getattr(bbgapi, "__file__", "?"), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("  [dz] bbgapi NOT loaded yet: %s" % exc, flush=True)

    print("  [dz] bootstrap ok  python=%s" % sys.executable, flush=True)


def main(argv=None) -> None:
    # Always run from repo root (where this file lives)
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        raise SystemExit(0)

    cmd = argv[0]
    rest = argv[1:]
    mapping = {
        "prediction": "run_prediction.py",
        "predict": "run_prediction.py",
        "backtest": "run_backtest.py",
        "pipeline": "run_pipeline.py",
        "explore": "run_pipeline.py",
    }
    script = mapping.get(cmd)
    if script is None:
        print("Unknown command %r. Use: prediction | backtest | pipeline" % cmd)
        raise SystemExit(2)

    _maybe_reexec_for_bbgapi()
    _bootstrap()
    sys.argv = [script] + rest
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
