#!/usr/bin/env python
"""
DataZone terminal launcher.

Loads pandas/numpy/bbgapi via ms.version (same idea as the jumpstart notebook),
then runs prediction / backtest / pipeline.

Examples (from fwd_screener_v1):

  python run_dz.py prediction --once --pull-history
  python run_dz.py prediction --once
  python run_dz.py prediction
  python run_dz.py backtest
  python run_dz.py pipeline
"""

from __future__ import annotations

import runpy
import sys


def _bootstrap() -> None:
    try:
        import ms.version
    except ImportError:
        print(
            "ERROR: this python has no ms.version.\n"
            "In a working notebook run:  import sys; print(sys.executable)\n"
            "Then:  /that/python run_dz.py prediction --once --pull-history",
            flush=True,
        )
        raise SystemExit(2)

    # Enough for the screener + bbgapi / requests stack (jumpstart-ish)
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
            print(f"  [dz] skip {name}: {exc}", flush=True)

    # bbgapi itself is loaded lazily in data._ensure_bbgapi
    print("  [dz] bootstrap ok  python=%s" % sys.executable, flush=True)


def main(argv=None) -> None:
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

    _bootstrap()
    sys.argv = [script] + rest
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
