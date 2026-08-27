"""Convert earth2studio's flattened-variable NetCDF output (e.g. u100,
u850, ...) into a CF-compliant file with a genuine `pressure` dimension,
so cross-section (XZ/YZ) plotting has real vertical structure to slice
through instead of needing to stack many separate flattened variables at
plot time.

Two structural changes are made, in this order:

  1. The (time=1, lead_time=N) pair is collapsed into a single `time`
     dimension of VALID times, with the initialization and offset kept as
     the CF auxiliary coordinates forecast_reference_time and
     forecast_period. earth2studio's layout — initialization as a size-1
     `time` axis, steps as `lead_time` — is not what CF expects, where
     `time` is the valid time and carries the record dimension.

  2. Leveled variables are stacked onto a real `pressure` dimension.

Order matters: the stacking step computes each variable's "leading" dims
and transposes to (*leading, pressure, lat, lon). Collapsing first makes
that leading list just ["time"], so the existing transpose already lands
on CF's recommended ordering. Doing it the other way round would mean
transposing twice.

This only stacks variables InferStudio actually runs (matching the base
names in MODEL_VAR_MAP in earth2StudioRunner.py: u, v, t, q, z). Surface
only fields (t2m, sp, msl, ...) are copied through unchanged, aside from
CF metadata being stamped on if we recognize the variable name.

Deliberately keeps earth2studio's own lowercase variable-name convention
(u, v, t, q, z) rather than mimicking any uppercase convention seen in
other CF-compliant reference files elsewhere — the rest of InferStudio's
codebase (parse_variable_groups, resolve_var_name, the Variable dropdown)
is built entirely around lowercase names already.
"""

from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

# Reuse the SAME level-parsing/dim-resolution logic already used for
# plotting, so this stays in sync automatically rather than duplicating
# the flattened-variable-name regex or lat/lon-dimension detection.
from visualization.earth2StudioVars import parse_variable_groups, _resolve_dim, LAT_NAMES, LON_NAMES

# CF metadata for the base variables InferStudio actually runs. Extend
# this if new base variables are added to MODEL_VAR_MAP in
# earth2StudioRunner.py.
#
# Note the units strings use ECMWF's "m s**-1" style rather than UDUNITS'
# "m s-1". Both parse under UDUNITS; this matches what earth2studio's own
# output and the wider ECMWF toolchain emit, and is what the colorbar
# labels in the app already display.
_CF_VAR_ATTRS = {
    "u": {"standard_name": "eastward_wind", "long_name": "U component of wind", "units": "m s**-1"},
    "v": {"standard_name": "northward_wind", "long_name": "V component of wind", "units": "m s**-1"},
    "t": {"standard_name": "air_temperature", "long_name": "Temperature", "units": "K"},
    "q": {"standard_name": "specific_humidity", "long_name": "Specific humidity", "units": "kg kg**-1"},
    "z": {"standard_name": "geopotential", "long_name": "Geopotential", "units": "m**2 s**-2"},
}

# CF metadata for known surface/single-level fields, applied only if the
# variable is actually present in a given file.
#
# t2m and sp carry a `height` / `surface` coordinate in strict CF (a scalar
# coordinate variable stating the 2 m level). That is not added here — the
# long_name states it, and adding scalar coords would complicate the
# variable-resolution logic downstream for little gain in this application.
_CF_SURFACE_ATTRS = {
    "t2m": {"standard_name": "air_temperature", "long_name": "2 metre temperature", "units": "K"},
    "sp": {"standard_name": "surface_air_pressure", "long_name": "Surface pressure", "units": "Pa"},
    "msl": {"standard_name": "air_pressure_at_mean_sea_level", "long_name": "Mean sea level pressure", "units": "Pa"},
}

# CF attributes for coordinate variables.
#
# `units` uses the plural forms (degrees_north / degrees_east). CF 4.1
# accepts several spellings, but these are canonical and are what most
# readers test for first.
#
# `axis` is what lets a generic reader identify which dimension is which
# without pattern-matching on names, and is the single most useful thing
# to get right here.
#
# `short_name` is not CF — it is an ECMWF/GRIB convention — but the rest of
# this file already emits it, so it stays for consistency.
_LAT_CF_ATTRS = {
    "standard_name": "latitude",
    "long_name": "latitude",
    "short_name": "lat",
    "units": "degrees_north",
    "axis": "Y",
}

_LON_CF_ATTRS = {
    "standard_name": "longitude",
    "long_name": "longitude",
    "short_name": "lon",
    "units": "degrees_east",
    "axis": "X",
}

_PRESSURE_CF_ATTRS = {
    "standard_name": "air_pressure",
    "long_name": "pressure",
    "short_name": "pres",
    "units": "hPa",
    "axis": "Z",
    # CF: "positive" states which direction of the coordinate corresponds
    # to up. Pressure decreases upward, so increasing pressure is downward.
    "positive": "down",
}

_TIME_CF_ATTRS = {
    "standard_name": "time",
    "long_name": "valid time",
    "axis": "T",
}

_FCST_REF_CF_ATTRS = {
    "standard_name": "forecast_reference_time",
    "long_name": "forecast initialization time",
}

_FCST_PERIOD_CF_ATTRS = {
    "standard_name": "forecast_period",
    "long_name": "time since forecast initialization",
    "units": "hours",
}


def _collapse_lead_time(ds):
    """Fold (time=1, lead_time=N) into one `time` dimension of valid times.

    Returns the dataset unchanged if there is no `lead_time` dimension, so
    this is safe to call on already-converted or ERA5-style input.

    The initialization survives as the scalar coordinate
    forecast_reference_time and the offsets as forecast_period(time) —
    both CF standard names — so nothing is lost, it is just relocated to
    where a CF reader expects to find it.
    """
    if "lead_time" not in ds.dims:
        return ds

    if "time" not in ds.coords:
        raise ValueError(
            "file has `lead_time` but no `time` coordinate to take the "
            "forecast initialization from - cannot compute valid times."
        )

    time_vals = np.atleast_1d(ds["time"].values)
    if time_vals.size != 1:
        raise ValueError(
            "expected a size-1 `time` axis holding the initialization, got "
            "%d values - this file is not in earth2studio's forecast layout."
            % time_vals.size
        )

    init = time_vals[0]
    lead = ds["lead_time"].values
    valid = init + lead
    lead_hours = lead / np.timedelta64(1, "h")

    # Drop the size-1 initialization axis before renaming, or the rename
    # would collide with the existing `time` name.
    if "time" in ds.dims:
        ds = ds.isel(time=0, drop=True)
    else:
        ds = ds.drop_vars("time", errors="ignore")

    ds = ds.rename({"lead_time": "time"})
    ds = ds.assign_coords(time=("time", valid))
    ds["time"].attrs = dict(_TIME_CF_ATTRS)

    ds = ds.assign_coords(
        forecast_reference_time=init,
        forecast_period=("time", lead_hours),
    )
    ds["forecast_reference_time"].attrs = dict(_FCST_REF_CF_ATTRS)
    ds["forecast_period"].attrs = dict(_FCST_PERIOD_CF_ATTRS)

    return ds


def _apply_coord_attrs(ds):
    """Stamp CF attributes on every coordinate variable.

    Merged over whatever the source file wrote rather than applied only
    when the existing attrs are empty - earth2studio generally does write
    something, so an emptiness test would skip these entirely and leave
    the coordinates unidentifiable to a CF reader.
    """
    lat_dim = _resolve_dim(ds, *LAT_NAMES)
    lon_dim = _resolve_dim(ds, *LON_NAMES)

    if lat_dim and lat_dim in ds.coords:
        ds[lat_dim].attrs = dict(ds[lat_dim].attrs, **_LAT_CF_ATTRS)

    if lon_dim and lon_dim in ds.coords:
        lon_vals = ds[lon_dim].values
        attrs = dict(ds[lon_dim].attrs, **_LON_CF_ATTRS)
        # No CF attribute states the 0..360 vs -180..180 convention
        # directly, so the actual bounds are the clearest signal a reader
        # has about which one this file uses. These are 0..360.
        attrs["valid_min"] = float(lon_vals.min())
        attrs["valid_max"] = float(lon_vals.max())
        ds[lon_dim].attrs = attrs

    if "pressure" in ds.coords:
        ds["pressure"].attrs = dict(ds["pressure"].attrs, **_PRESSURE_CF_ATTRS)

    # These three may already have been set by _collapse_lead_time, but a
    # file that arrives with a real time axis and no lead_time skips that
    # path entirely and still needs them.
    if "time" in ds.coords:
        ds["time"].attrs = dict(ds["time"].attrs, **_TIME_CF_ATTRS)
    if "forecast_reference_time" in ds.coords:
        ds["forecast_reference_time"].attrs = dict(
            ds["forecast_reference_time"].attrs, **_FCST_REF_CF_ATTRS)
    if "forecast_period" in ds.coords:
        ds["forecast_period"].attrs = dict(
            ds["forecast_period"].attrs, **_FCST_PERIOD_CF_ATTRS)

    return ds


def _build_encoding(ds):
    """Encoding for coordinates and data variables.

    Two things this handles that xarray's defaults get wrong for CF:

    Time is written as hours since the initialization rather than
    nanoseconds since 1970. Both are valid CF, but the former is readable
    in ncdump and self-documenting for a forecast.

    _FillValue is suppressed on every coordinate. xarray adds one by
    default, and CF 2.5.1 says coordinate variables should not have
    missing values - a coordinate with a fill value is a contradiction.
    """
    encoding = {}

    if "time" in ds.coords and np.issubdtype(ds["time"].dtype, np.datetime64):
        if "forecast_reference_time" in ds.coords:
            ref = ds["forecast_reference_time"].values
        else:
            ref = ds["time"].values[0]
        ref_str = pd.Timestamp(ref).strftime("%Y-%m-%d %H:%M:%S")
        units = "hours since " + ref_str

        encoding["time"] = {
            "units": units,
            "calendar": "proleptic_gregorian",
            "dtype": "float64",
            "_FillValue": None,
        }
        if "forecast_reference_time" in ds.coords:
            encoding["forecast_reference_time"] = {
                "units": units,
                "calendar": "proleptic_gregorian",
                "dtype": "float64",
                "_FillValue": None,
            }

    lat_dim = _resolve_dim(ds, *LAT_NAMES)
    lon_dim = _resolve_dim(ds, *LON_NAMES)
    for coord in (lat_dim, lon_dim, "pressure", "forecast_period"):
        if coord and coord in ds.coords and coord not in encoding:
            encoding[coord] = {"_FillValue": None}

    for v in ds.data_vars:
        encoding[v] = {"zlib": True, "complevel": 4, "_FillValue": None}

    return encoding


def make_cf_compliant(nc_path, suffix="_cf"):
    """Read an earth2studio-format NetCDF file at `nc_path` and write a
    CF-compliant version alongside it (same directory, `<stem><suffix>.nc`).
    Returns the path to the new file. The original file is left untouched.
    """
    nc_path = str(nc_path)
    if nc_path.endswith(".nc"):
        out_path = nc_path[:-3] + suffix + ".nc"
    else:
        out_path = nc_path + suffix

    with xr.open_dataset(nc_path) as ds:
        # Collapse first - see the module docstring on why the order of
        # these two transformations is not arbitrary.
        ds = _collapse_lead_time(ds)

        level_vars, surface_vars = parse_variable_groups(list(ds.data_vars))

        new_vars = {}

        # Stack each leveled base variable (u, v, t, q, z, ...) into one
        # array with a genuine `pressure` dimension.
        for base, levels_map in level_vars.items():
            levels_sorted = sorted(levels_map.keys())
            pieces = [ds[levels_map[lev]] for lev in levels_sorted]

            # Determine lat/lon dim names and every other ("leading") dim
            # from the first piece - should be consistent across all levels
            # of the same base variable. Post-collapse this is just `time`.
            lat_dim = _resolve_dim(pieces[0], *LAT_NAMES)
            lon_dim = _resolve_dim(pieces[0], *LON_NAMES)
            leading_dims = [d for d in pieces[0].dims if d not in (lat_dim, lon_dim)]

            stacked = xr.concat(pieces, dim="pressure")
            stacked = stacked.assign_coords(
                pressure=("pressure", np.array(levels_sorted, dtype="float64"))
            )
            # xr.concat prepends the new dim - reorder so leading dims
            # (time) stay first, then pressure, then lat/lon.
            stacked = stacked.transpose(*leading_dims, "pressure", lat_dim, lon_dim)

            attrs = dict(_CF_VAR_ATTRS.get(base, {}))
            attrs.setdefault("short_name", base)
            stacked.attrs = attrs

            new_vars[base] = stacked

        # Surface/single-level fields pass through unchanged, but get CF
        # attrs stamped on if we recognize the variable name.
        for name in surface_vars:
            da = ds[name].copy()
            if name in _CF_SURFACE_ATTRS:
                da.attrs = dict(da.attrs, **_CF_SURFACE_ATTRS[name])
            new_vars[name] = da

        out_ds = xr.Dataset(new_vars, coords=ds.coords)

        out_ds = _apply_coord_attrs(out_ds)

        # Point each data variable at its auxiliary coordinates. Without
        # this a CF reader has no way to know forecast_period applies to
        # these variables rather than being an unrelated array. Dimension
        # coordinates (time, pressure, lat, lon) are found automatically
        # and must NOT be listed here.
        aux = [c for c in ("forecast_reference_time", "forecast_period")
               if c in out_ds.coords]
        if aux:
            for v in out_ds.data_vars:
                out_ds[v].attrs["coordinates"] = " ".join(aux)

        out_ds.attrs["Conventions"] = "CF-1.11"

        out_ds.to_netcdf(out_path, encoding=_build_encoding(out_ds))

    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python cf_convert.py <path_to_netcdf>")
        sys.exit(1)
    result = make_cf_compliant(sys.argv[1])
    print("Wrote CF-compliant file: " + result)
