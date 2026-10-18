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
  python run_dz.py crosscheck          # python vs BBG USOS/UF structures → Excel
  python run_dz.py correlations        # corr matrix + corr z-score heatmap
  python run_dz.py custom --once --pull-history   # CUSTOM_MODELS prediction
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


def _discover_site(pkg_name):
    """Find site-packages dirs under /ms/dist that contain pkg_name."""
    tag = "python%d.%d" % (sys.version_info[0], sys.version_info[1])
    patterns = [
        "/ms/dist/python/PROJ/%s/*/lib/%s/site-packages" % (pkg_name, tag),
        "/ms/dist/python/PROJ/%s/*/exec/lib/%s/site-packages" % (pkg_name, tag),
        "/ms/dist/python/PROJ/%s/*/%s/site-packages" % (pkg_name, tag),
        "/ms/dist/datazone/PROJ/%s/*/lib/%s/site-packages" % (pkg_name, tag),
        "/ms/dist/datazone/PROJ/%s/*/exec/lib/%s/site-packages" % (pkg_name, tag),
        # broader fallback (any py tag) if exact match missing
        "/ms/dist/python/PROJ/%s/*/lib/python*/site-packages" % pkg_name,
        "/ms/dist/python/PROJ/%s/*/exec/lib/python*/site-packages" % pkg_name,
        "/ms/dist/datazone/PROJ/%s/*/lib/python*/site-packages" % pkg_name,
    ]
    found = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            # prefer exact interpreter tag
            if ("/lib/%s/" % tag) not in d.replace("\\", "/") and "/python*/" not in pat:
                pass
            marker = pkg_name.replace("-", "_") if "-" in pkg_name else pkg_name
            if os.path.isdir(os.path.join(d, marker)) or os.path.isdir(
                os.path.join(d, pkg_name)
            ) or os.path.isfile(os.path.join(d, marker + ".py")):
                if d not in found:
                    found.append(d)
    # put exact-tag matches first
    exact, other = [], []
    needle = "/lib/%s/" % tag
    for d in found:
        (exact if needle in d.replace("\\", "/") else other).append(d)
    return exact + other


def _ensure_import(ms_version, mod_name, versions, kw=None, required=False, pkg_name=None):
    """addpkg + import; try versions then /ms/dist discovery.

    pkg_name: name for addpkg/disk (defaults to mod_name). Use when they differ
    (e.g. pkg_name='charset-normalizer', mod_name='charset_normalizer').
    """
    kw = kw or {}
    pkg = pkg_name or mod_name
    try:
        __import__(mod_name)
        return True
    except ImportError:
        pass

    # Prefer prod/latest first — version pins often missing on the desk
    ordered = []
    for ver in ("prod", "latest") + tuple(versions):
        if ver not in ordered:
            ordered.append(ver)

    for ver in ordered:
        try:
            ms_version.addpkg(pkg, ver, **kw)
        except TypeError:
            try:
                ms_version.addpkg(pkg, ver)
            except Exception as exc:  # noqa: BLE001
                print("  [dz] skip %s %s: %s" % (pkg, ver, exc), flush=True)
                continue
        except Exception as exc:  # noqa: BLE001
            print("  [dz] skip %s %s: %s" % (pkg, ver, exc), flush=True)
            continue
        try:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            __import__(mod_name)
            print("  [dz] %s OK  v=%s" % (pkg, ver), flush=True)
            return True
        except ImportError as exc:
            print("  [dz] %s addpkg ok but import fail v=%s: %s" % (pkg, ver, exc), flush=True)

    for site in _discover_site(pkg) + _discover_site(mod_name):
        if site not in sys.path:
            sys.path.insert(0, site)
        try:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            mod = __import__(mod_name)
            print(
                "  [dz] %s OK from %s  file=%s"
                % (pkg, site, getattr(mod, "__file__", "?")),
                flush=True,
            )
            return True
        except ImportError:
            continue

    msg = "  [dz] could not import %s (tried %s)" % (pkg, ", ".join(ordered))
    if required:
        print("ERROR: " + msg, flush=True)
        raise SystemExit(2)
    print(msg, flush=True)
    return False


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

    # Core scientific stack (best-effort versions)
    for name, vers in (
        ("six", ("1.12.0", "1.16.0")),
        ("pytz", ("2022.1", "2021.3")),
        ("dateutil", ("2.8.0", "2.8.2")),
        ("numpy", ("1.21.2", "1.19.5", "1.18.5")),
        ("pandas", ("1.3.4", "1.1.5", "0.25.3")),
        ("openpyxl", ("3.0.9", "3.0.7")),
        ("matplotlib", ("3.4.3", "3.3.4", "3.1.3", "prod", "latest")),
    ):
        _ensure_import(ms.version, name, vers)

    # requests/bbgapi stack — require the ones that usually break terminal runs
    _ensure_import(
        ms.version,
        "urllib3",
        ("1.26.7", "1.26.6", "1.26.5", "1.26.4", "1.25.11", "1.25.10", "1.24.3"),
        required=True,
    )
    _ensure_import(
        ms.version,
        "certifi",
        (
            "2021.10.8",
            "2021.5.30",
            "2020.12.5",
            "2020.6.20",
            "2019.11.28",
            "2022.12.7",
            "2022.9.24",
            "2022.6.15",
            "2023.7.22",
        ),
        required=True,
    )
    _ensure_import(
        ms.version,
        "idna",
        ("3.3", "3.2", "2.10", "2.9", "2.8"),
        required=True,
    )
    _ensure_import(ms.version, "chardet", ("4.0.0", "3.0.4"))
    _ensure_import(
        ms.version,
        "charset_normalizer",
        ("2.0.7", "2.0.12", "2.0.4"),
        pkg_name="charset-normalizer",
    )
    _ensure_import(ms.version, "toml", ("0.10.1", "0.10.2"))
    _ensure_import(ms.version, "retrying", ("1.3.3",))
    _ensure_import(
        ms.version,
        "requests",
        ("2.26.0", "2.25.1", "2.22.0", "2.21.0"),
        kw={"shadow": True},
        required=True,
    )

    # Pre-load bbgapi here so failures are obvious before prediction starts
    try:
        from data import _ensure_bbgapi

        _ensure_bbgapi()
        import bbgapi  # noqa: F401

        print("  [dz] bbgapi OK  file=%s" % getattr(bbgapi, "__file__", "?"), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("  [dz] bbgapi NOT loaded yet: %s" % exc, flush=True)

    print("  [dz] bootstrap ok  python=%s" % sys.executable, flush=True)


def bootstrap_for_repo(repo_path: str | None = None) -> None:
    """Jupyter: chdir + sys.path + ms.version/bbgapi bootstrap (same as terminal launcher)."""
    if repo_path:
        root = os.path.abspath(repo_path)
    else:
        root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    _bootstrap()


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
        "crosscheck": "crosscheck.py",
        "xcheck": "crosscheck.py",
        "correlations": "correlations.py",
        "corr": "correlations.py",
        "custom": "run_custom_prediction.py",
        "custom-prediction": "run_custom_prediction.py",
    }
    script = mapping.get(cmd)
    if script is None:
        print(
            "Unknown command %r. Use: prediction | backtest | pipeline | crosscheck | "
            "correlations | custom" % cmd
        )
        raise SystemExit(2)

    _maybe_reexec_for_bbgapi()
    _bootstrap()
    sys.argv = [script] + rest
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
