#!/usr/bin/env python
"""Report the spatial dimension names each model in a suite actually uses.

HoloViews links plot axes by DIMENSION NAME, so the grid's shared_axes only
connects panels whose kdims agree. earth2StudioPlot canonicalizes to
latitude/longitude for exactly this reason — this script says whether that
canonicalization is load-bearing for a given suite, and confirms the other
assumptions the loader makes (one non-singleton forecast axis, evenly spaced
grid, latitude direction).

Usage:

    # One suite. QUOTE THE PATH: the timestamp contains colons.
    python scripts/check_dims.py \\
        "/glade/derecho/scratch/pearse/InferStudio_AIFS_Aurora_Pangu_2026_08_25_12:57:01"

    # Every suite under a parent directory
    python scripts/check_dims.py /glade/derecho/scratch/pearse

    # A single model directory
    python scripts/check_dims.py "/glade/.../<suite>/AIFS"

Read-only: opens files lazily and never loads a full field.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Candidate name lists come from the app when it's importable, so this script
# tests the same resolution the loader performs rather than a copy that can
# drift. Falls back to the same defaults when run outside the repo.
try:
    from visualization.earth2StudioVars import LAT_NAMES, LON_NAMES
except ImportError:
    LAT_NAMES = ("latitude", "lat", "y")
    LON_NAMES = ("longitude", "lon", "x")

try:
    from dimensions import PRES_NAME
except ImportError:
    PRES_NAME = "pressure"

CANON_LAT = "latitude"
CANON_LON = "longitude"


def nc_files(model_dir: Path):
    """The .nc files in a model directory, CF-converted ones preferred.

    Mirrors resolve_nc_glob's intent: a directory can hold both the raw
    earth2studio output and its *_cf.nc conversion, and reading both at once
    would merge two different variable-naming schemes.
    """
    cf = sorted(model_dir.glob("*_cf.nc"))
    return cf if cf else sorted(model_dir.glob("*.nc"))


def resolve(names, candidates):
    for c in candidates:
        if c in names:
            return c
    return None


def evenly_spaced(values, rtol=1e-4):
    if values.size < 3:
        return True
    d = np.diff(values.astype("float64"))
    return bool(np.allclose(d, d[0], rtol=rtol, atol=0.0))


def describe(model_dir: Path):
    """Inspect one model directory. Returns a dict, or None if unreadable."""
    files = nc_files(model_dir)
    if not files:
        return None

    # chunks={} keeps this lazy — coordinates are read, field data is not.
    with xr.open_mfdataset(files, engine="netcdf4", data_vars="all",
                           chunks={}) as ds:
        dims = dict(ds.sizes)
        lat = resolve(dims, LAT_NAMES)
        lon = resolve(dims, LON_NAMES)

        info = {
            "dir": model_dir,
            "files": [f.name for f in files],
            "dims": dims,
            "lat": lat,
            "lon": lon,
            "vars": len(ds.data_vars),
            "pressure": (
                sorted(float(v) for v in ds[PRES_NAME].values.ravel())
                if PRES_NAME in ds.coords else None
            ),
        }

        if lat and lat in ds.coords:
            v = ds[lat].values
            info["lat_n"] = int(v.size)
            info["lat_desc"] = bool(v.size > 1 and v[0] > v[-1])
            info["lat_even"] = evenly_spaced(v)
            info["lat_span"] = (float(v.min()), float(v.max()))
        if lon and lon in ds.coords:
            v = ds[lon].values
            info["lon_n"] = int(v.size)
            info["lon_even"] = evenly_spaced(v)
            info["lon_span"] = (float(v.min()), float(v.max()))

        # The loader raises if more than one non-singleton axis survives
        # beyond lat/lon, since it can't tell which one `t` should select.
        # Sample one variable that has both spatial dims.
        sample = next(
            (v for v in ds.data_vars
             if lat in ds[v].dims and lon in ds[v].dims), None)
        if sample:
            da = ds[sample]
            extra = [d for d in da.dims if d not in (lat, lon)]
            info["sample_var"] = sample
            info["extra_dims"] = {d: int(da.sizes[d]) for d in extra}
            info["forecast_axes"] = [
                d for d in extra
                if da.sizes[d] > 1 and d != PRES_NAME
            ]

    return info


def report_model(info):
    d = info
    print(f"  {d['dir'].name}")
    print(f"    files        : {', '.join(d['files'])}")
    print(f"    lat / lon    : {d['lat']!r} / {d['lon']!r}", end="")
    if d["lat"] != CANON_LAT or d["lon"] != CANON_LON:
        print("   <- renamed by the loader")
    else:
        print()

    if "lat_n" in d:
        lo, hi = d["lat_span"]
        print(f"    latitude     : {d['lat_n']} pts, {lo:g}..{hi:g}, "
              f"{'descending' if d['lat_desc'] else 'ascending'}, "
              f"{'even' if d['lat_even'] else 'IRREGULAR -> QuadMesh'}")
    if "lon_n" in d:
        lo, hi = d["lon_span"]
        print(f"    longitude    : {d['lon_n']} pts, {lo:g}..{hi:g}, "
              f"{'even' if d['lon_even'] else 'IRREGULAR -> QuadMesh'}")

    if d.get("pressure"):
        p = d["pressure"]
        print(f"    pressure     : {len(p)} levels, "
              f"{p[0]:g}..{p[-1]:g}  (CF-converted)")
    else:
        print("    pressure     : none as a coord (flattened var names?)")

    if "sample_var" in d:
        print(f"    sample var   : {d['sample_var']}  "
              f"extra dims {d['extra_dims']}")
        n = len(d["forecast_axes"])
        if n == 1:
            print(f"    forecast axis: {d['forecast_axes'][0]}  (ok)")
        elif n == 0:
            print("    forecast axis: NONE — single time step only")
        else:
            print(f"    forecast axis: AMBIGUOUS {d['forecast_axes']} — "
                  f"load_e2s_field will raise")

    print(f"    data_vars    : {d['vars']}")


def check_suite(suite_dir: Path):
    model_dirs = sorted(p for p in suite_dir.iterdir()
                        if p.is_dir() and nc_files(p))
    if not model_dirs:
        return False

    print(f"\n{suite_dir}")
    infos = []
    for md in model_dirs:
        try:
            info = describe(md)
        except Exception as exc:
            print(f"  {md.name}\n    FAILED: {type(exc).__name__}: {exc}")
            continue
        if info:
            infos.append(info)
            report_model(info)

    # The verdict this script exists for.
    pairs = {(i["lat"], i["lon"]) for i in infos}
    print()
    if len(pairs) <= 1:
        #names = next(iter(paires)) if False else next(iter(pairs), None)
        names = next(iter(pairs), None)
        print(f"  VERDICT: all {len(infos)} models agree on {names} — "
              f"shared_axes would link without the rename.")
    else:
        print(f"  VERDICT: {len(pairs)} different naming schemes in one "
              f"suite: {sorted(pairs)}")
        print("           Canonicalization IS required — without it "
              "shared_axes silently fails to link.")

    grids = {(i.get("lat_n"), i.get("lon_n")) for i in infos}
    if len(grids) > 1:
        print(f"  NOTE:    grid sizes differ {sorted(grids)} — expected; "
              f"modelDiff interpolates the mismatched dim by name.")

    if any(not i.get("lat_even", True) or not i.get("lon_even", True)
           for i in infos):
        print("  NOTE:    at least one model has an irregular grid; those "
              "panels render as QuadMesh, not Image.")

    return True


def main(argv):
    targets = [Path(a) for a in argv[1:]] or [
        Path(f"/glade/derecho/scratch/{__import__('os').environ['USER']}")]

    for target in targets:
        if not target.is_dir():
            print(f"not a directory: {target}")
            continue

        # A model directory holds .nc files directly; a suite holds model
        # subdirectories; anything else is treated as a parent of suites.
        if nc_files(target):
            print(f"\n{target.parent}")
            info = describe(target)
            if info:
                report_model(info)
            continue

        if check_suite(target):
            continue

        found = False
        for sub in sorted(p for p in target.iterdir() if p.is_dir()):
            if check_suite(sub):
                found = True
        if not found:
            print(f"no suites with .nc files found under {target}")


if __name__ == "__main__":
    main(sys.argv)
