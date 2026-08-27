"""Compute per-variable differences between two models' CF-compliant
output within the same simulation suite, writing the result to a new
NetCDF file that can be plotted the same way as any other model output
(via load_e2s_field / plot_e2s_field), just with a diverging colormap and
a symmetric value range, since these are signed difference fields.

Cached: once computed for a given model pair, the same file is reused on
subsequent requests rather than recomputing.

Each model pair gets its own subdirectory (cache_dir/<A>_minus_<B>/) -
matching the same "model_dir contains this model's .nc file(s)"
convention used everywhere else in the app (e.g. <sim_dir>/AIFS/AIFS.nc).
The loaders always glob *.nc within whatever directory they're given, so
if multiple diff pairs shared one flat directory, computing a second pair
for the same suite would make that glob match both files at once and try
to merge them together - giving each pair its own directory avoids that
entirely.

The output is written to the same CF conventions cf_convert.py produces
for model output, so a difference file is readable by anything that can
read a model file. That means carrying over the auxiliary coordinates
(forecast_reference_time, forecast_period) and the coordinate attributes,
neither of which survive building a Dataset from bare DataArrays.
"""

import os
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dimensions import resolve_nc_glob
from visualization.earth2StudioVars import _resolve_dim, LAT_NAMES, LON_NAMES
from visualization.earth2StudioPlot import invalidate_dataset, load_e2s_field


# One lock per pair name. The HoloViews grid can fire several panel
# callbacks concurrently off a single parameter change; without this, two
# threads can both see a missing cache file and both start the same
# multi-gigabyte computation.
_PAIR_LOCKS = defaultdict(threading.Lock)
_PAIR_LOCKS_GUARD = threading.Lock()

# Auxiliary (non-dimension) coordinates carried over from model A.
_AUX_COORDS = ("forecast_reference_time", "forecast_period")


def _lock_for(name: str) -> threading.Lock:
    with _PAIR_LOCKS_GUARD:
        return _PAIR_LOCKS[name]


def pair_name(model_a_name: str, model_b_name: str) -> str:
    return f"{model_a_name}_minus_{model_b_name}"


def pair_dir_path(cache_dir, model_a_name, model_b_name) -> Path:
    """Where a given pair's directory lives, whether or not it exists yet."""
    return Path(cache_dir) / pair_name(model_a_name, model_b_name)


def _build_encoding(ds):
    """Encoding matching what cf_convert.py writes for model output.

    Time as hours since the initialization rather than nanoseconds since
    1970, no _FillValue on coordinates (CF 2.5.1: coordinate variables
    should not have missing values), and no _FillValue on data variables
    either - a NaN in a difference field means an interpolation failed and
    should be visible as an anomaly, not silently absorbed as an expected
    fill value.
    """
    encoding = {v: {"zlib": True, "complevel": 4, "_FillValue": None}
                for v in ds.data_vars}

    ref = None
    if "forecast_reference_time" in ds.coords:
        ref = ds["forecast_reference_time"].values
    elif "time" in ds.coords and np.issubdtype(ds["time"].dtype, np.datetime64):
        ref = ds["time"].values.ravel()[0]

    units = None
    if ref is not None:
        units = "hours since " + pd.Timestamp(ref).strftime("%Y-%m-%d %H:%M:%S")

    for c in ds.coords:
        if units and np.issubdtype(ds[c].dtype, np.datetime64):
            encoding[c] = {
                "units": units,
                "calendar": "proleptic_gregorian",
                "dtype": "float64",
                "_FillValue": None,
            }
        else:
            encoding[c] = {"_FillValue": None}

    return encoding


def compute_model_difference(model_a_dir, model_b_dir, cache_dir,
                             model_a_name, model_b_name) -> Path:
    """Compute (model_a - model_b) for every variable present in both
    datasets, writing the result to
    <cache_dir>/<A>_minus_<B>/<A>_minus_<B>.nc.

    Returns the PATH TO THE DIRECTORY containing that file (not the file
    itself) - this is the same "model_dir" contract the loaders expect for
    every other model. If the file already exists, its directory is
    returned immediately without recomputing.
    """
    name = pair_name(model_a_name, model_b_name)
    pair_dir = pair_dir_path(cache_dir, model_a_name, model_b_name)
    out_path = pair_dir / f"{name}.nc"

    # Fast path outside the lock - the overwhelmingly common case once a
    # pair has been computed is that the file is simply there.
    if out_path.exists():
        return pair_dir

    with _lock_for(name):
        # Re-check: another thread may have finished while we waited.
        if out_path.exists():
            return pair_dir

        pair_dir.mkdir(parents=True, exist_ok=True)

        # Write to a temporary name in the same directory, then rename.
        # os.replace is atomic within a filesystem, so a reader either sees
        # no file or a complete one - never a partially flushed NetCDF.
        # This matters now that plot callbacks poll these paths on every
        # parameter change rather than once per explicit user action.
        # The leading dot also keeps the temp file out of the *.nc glob.
        tmp_path = pair_dir / f".{name}.nc.tmp"

        with xr.open_mfdataset(resolve_nc_glob(model_a_dir), engine="netcdf4",
                               data_vars="all", chunks={}) as ds_a, \
             xr.open_mfdataset(resolve_nc_glob(model_b_dir), engine="netcdf4",
                               data_vars="all", chunks={}) as ds_b:

            shared_vars = sorted(set(ds_a.data_vars) & set(ds_b.data_vars))
            if not shared_vars:
                raise ValueError(
                    f"{model_a_name} and {model_b_name} have no variables in "
                    f"common \u2014 cannot compute a difference."
                )

            diff_vars = {}
            skipped = {}
            for var in shared_vars:
                da_a = ds_a[var]
                da_b = ds_b[var]
                try:
                    if da_a.shape != da_b.shape:
                        # Grids don't line up (e.g. AIFS's 721-point latitude
                        # grid vs Aurora's 720-point grid). Interpolate ONLY
                        # the specific dimensions that actually mismatch, by
                        # name - NOT a blanket da_b.interp_like(da_a), which
                        # tries to interpolate every matching coordinate
                        # (including size-1 dims like "time"). Scipy's linear
                        # interpolator needs >=2 points to compute a slope; a
                        # size-1 axis hits an exact 0/0 division, and that
                        # single NaN then propagates through the ENTIRE array
                        # via broadcasting - silently turning a minor,
                        # legitimate lat-grid mismatch into 100% NaN output.
                        lat_dim = _resolve_dim(da_a, *LAT_NAMES)
                        lon_dim = _resolve_dim(da_a, *LON_NAMES)
                        interp_kwargs = {}
                        for dim in (lat_dim, lon_dim):
                            if (
                                dim
                                and dim in da_b.dims
                                and dim in da_a.dims
                                and (
                                    da_a.sizes[dim] != da_b.sizes[dim]
                                    or not np.array_equal(
                                        da_a[dim].values, da_b[dim].values)
                                )
                            ):
                                interp_kwargs[dim] = da_a[dim]
                        if interp_kwargs:
                            da_b = da_b.interp(**interp_kwargs)
                    diff = da_a - da_b
                    diff.attrs = dict(da_a.attrs)
                    diff_vars[var] = diff
                except Exception as e:
                    skipped[var] = str(e)

            if not diff_vars:
                raise ValueError(
                    f"Could not compute a difference for any shared variable "
                    f"between {model_a_name} and {model_b_name}: {skipped}"
                )

            out_ds = xr.Dataset(diff_vars)

            # Carry over the auxiliary coordinates from model A. A Dataset
            # built from DataArrays inherits each array's own DIMENSION
            # coordinates but not dataset-level auxiliary ones, so without
            # this the difference files come out structurally poorer than
            # the model files they came from: no forecast_period, no
            # forecast_reference_time.
            for coord in _AUX_COORDS:
                if coord in ds_a.coords:
                    out_ds = out_ds.assign_coords({coord: ds_a[coord]})
                    out_ds[coord].attrs = dict(ds_a[coord].attrs)

            # Dimension coordinates survive the DataArray round-trip as
            # values but lose their attributes, so restore those too -
            # otherwise the CF metadata cf_convert took care to write is
            # silently dropped from every difference file.
            for coord in out_ds.coords:
                if coord in ds_a.coords and not out_ds[coord].attrs:
                    out_ds[coord].attrs = dict(ds_a[coord].attrs)

            # Variable-level attributes were copied from model A above, but
            # some of them are actively wrong for a difference field. The
            # units carry over correctly (a difference of two fields in
            # K is in K), but standard_name does not: the difference of two
            # eastward_wind fields is not itself an eastward_wind, and a
            # reader trusting that name would apply the wrong valid range
            # and sign conventions.
            for v in out_ds.data_vars:
                attrs = dict(out_ds[v].attrs)
                attrs.pop("standard_name", None)
                base_long = attrs.get("long_name", v)
                attrs["long_name"] = (
                    f"{base_long} difference "
                    f"({model_a_name} minus {model_b_name})")
                if _AUX_COORDS and any(c in out_ds.coords for c in _AUX_COORDS):
                    attrs["coordinates"] = " ".join(
                        c for c in _AUX_COORDS if c in out_ds.coords)
                out_ds[v].attrs = attrs

            out_ds.attrs["Conventions"] = ds_a.attrs.get("Conventions", "CF-1.11")
            out_ds.attrs["history"] = (
                f"Difference: {model_a_name} minus {model_b_name}")
            if skipped:
                out_ds.attrs["skipped_variables"] = ", ".join(sorted(skipped))

            try:
                out_ds.to_netcdf(tmp_path, encoding=_build_encoding(out_ds))
                os.replace(tmp_path, out_path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise

        # A failed or superseded earlier read of this directory may be
        # sitting in the loader's open-dataset cache. Drop it so the next
        # read picks up the file just written.
        invalidate_dataset(pair_dir)

    return pair_dir


def load_diff_field(model_a_dir, model_b_dir, cache_dir,
                    model_a_name, model_b_name, base_or_var, level, t):
    """Convenience wrapper: ensure the diff exists, then load one field
    from it. This is what the plot grid's diff_provider calls."""
    pair_dir = compute_model_difference(
        model_a_dir, model_b_dir, cache_dir, model_a_name, model_b_name)
    return load_e2s_field(pair_dir, base_or_var, level, t)


def symmetric_diff_range(model_a_dir, model_b_dir, cache_dir,
                         model_a_name, model_b_name, base_or_var, level,
                         sample_steps=3):
    """Symmetric (-m, +m) colour limits sampled across a few steps.

    Symmetry is not cosmetic: with coolwarm and an asymmetric range, zero
    lands somewhere off-white and an unbiased difference field reads as
    biased.
    """
    from visualization.earth2StudioPlot import field_range
    pair_dir = compute_model_difference(
        model_a_dir, model_b_dir, cache_dir, model_a_name, model_b_name)
    lo, hi = field_range(pair_dir, base_or_var, level, sample_steps)
    m = max(abs(lo), abs(hi))
    return (-m, m)
