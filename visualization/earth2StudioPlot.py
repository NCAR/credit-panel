"""Data loading and plotting for earth2studio-format model outputs (AIFS,
Aurora, Pangu, and similar). See earth2StudioVars.py for the
naming-convention parsing logic (kept separate so code that only needs
parsing, not plotting, doesn't have to import matplotlib transitively).

Two entry points:

  load_e2s_field  -- returns a 2D DataArray plus metadata. Used by the
                     HoloViews grid (visualization/plotGrid.py), which
                     needs the array itself so Bokeh can re-render it at
                     each zoom level.

  plot_e2s_field  -- renders a PNG. Retained for Export Video and any
                     other consumer that wants a static raster; it is now
                     a thin wrapper over load_e2s_field so the two paths
                     can't drift apart.
"""

import io
import threading
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dimensions import resolve_nc_glob, PRES_NAME
from visualization.earth2StudioVars import (
    LAT_NAMES, LON_NAMES, TIME_NAMES,
    _resolve_dim, parse_variable_groups, available_levels,
    resolve_var_name, get_model_nc_path,
)


# Canonical spatial dimension names for everything this module returns.
#
# HoloViews links axes by DIMENSION NAME, so every panel in the plot grid
# must agree on them or shared_axes silently fails to connect the panels —
# giving per-panel zoom, which looks like it works and doesn't. Model files
# do NOT agree: one may write latitude/longitude where another writes
# lat/lon, which is the whole reason _resolve_dim exists. Renaming here
# means every consumer sees one naming scheme regardless of source.
CANON_LAT = "latitude"
CANON_LON = "longitude"


# ---------------------------------------------------------------------------
# Open-dataset cache
#
# The Matplotlib path opened, read, and closed the file on every call, which
# was fine at one render per user click. The HoloViews grid re-invokes the
# loader on every time-slider tick across six panels, so reopening an
# mfdataset each time becomes the dominant cost. Datasets are held open and
# dask-backed (chunks={}), so only the requested 2D slice is read from disk.
#
# Note the removal of autoclose=True from the open call: it exists precisely
# to drop file handles between reads, which is the opposite of what caching
# wants.
# ---------------------------------------------------------------------------

_DS_CACHE: "OrderedDict[str, xr.Dataset]" = OrderedDict()
_DS_CACHE_MAX = 16
_DS_LOCK = threading.Lock()


def _cache_key(model_dir) -> str:
    return str(Path(model_dir).resolve())


def _open_dataset(model_dir) -> xr.Dataset:
    """Open (or retrieve from cache) the dataset for a model directory."""
    key = _cache_key(model_dir)
    with _DS_LOCK:
        ds = _DS_CACHE.pop(key, None)
        if ds is None:
            ds = xr.open_mfdataset(
                resolve_nc_glob(model_dir),
                engine="netcdf4",
                data_vars="all",
                chunks={},          # dask-backed: defer the actual read
            )
        _DS_CACHE[key] = ds        # reinsert at MRU end
        while len(_DS_CACHE) > _DS_CACHE_MAX:
            _, evicted = _DS_CACHE.popitem(last=False)
            try:
                evicted.close()
            except Exception:
                pass
        return ds


def invalidate_dataset(model_dir) -> None:
    """Drop a directory from the cache. Call after writing new .nc files
    into a directory that may already have been read (notably a diff pair
    directory being recomputed)."""
    key = _cache_key(model_dir)
    with _DS_LOCK:
        ds = _DS_CACHE.pop(key, None)
    if ds is not None:
        try:
            ds.close()
        except Exception:
            pass


def close_dataset_cache() -> None:
    """Close every cached dataset. Wire this to the Panel session's
    on_destroy so a long-lived server doesn't accumulate file handles."""
    with _DS_LOCK:
        items = list(_DS_CACHE.values())
        _DS_CACHE.clear()
    for ds in items:
        try:
            ds.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Field metadata
# ---------------------------------------------------------------------------

class FieldMeta(NamedTuple):
    """Everything about a loaded field that isn't the data itself."""

    var_name: str
    lat_dim: str          # always CANON_LAT (or None if the file had no lat)
    lon_dim: str          # always CANON_LON (or None)
    src_lat_dim: str      # the name as it appeared in the file
    src_lon_dim: str
    valid_time: Optional[np.datetime64]
    lead_hours: Optional[float]
    t_index: int          # index actually used, after clamping
    n_steps: int          # length of the forecast-step axis (1 if none)
    regular_grid: bool    # evenly spaced lat and lon?

    def title(self, prefix: str = "") -> str:
        head = f"{prefix} " if prefix else ""
        if self.valid_time is not None and self.lead_hours is not None:
            valid_str = str(np.datetime64(self.valid_time, "s"))
            return (f"{head}{self.var_name}  |  Valid: {valid_str}"
                    f"  |  Forecast: +{self.lead_hours:.0f}h")
        if self.lead_hours is not None:
            return f"{head}{self.var_name}  |  Forecast: +{self.lead_hours:.0f}h"
        return f"{head}{self.var_name}  (index={self.t_index})"

    def time_label(self) -> str:
        """The valid time: the moment this forecast field is FOR.

        This already includes the lead time — valid_time is computed as
        init_time + lead, so a +144h step of a 2026-08-24T12 initialization
        reports 2026-08-30T12. There is no further arithmetic to do; adding
        the lead again would put the label six days into the future.
        """
        if self.valid_time is not None:
            stamp = str(np.datetime64(self.valid_time, "m"))
            return stamp.replace("T", " ") + " UTC"
        if self.lead_hours is not None:
            # No initialization time in the file, so an absolute timestamp
            # can't be formed — fall back to the offset.
            return f"+{self.lead_hours:.0f}h from initialization"
        return f"Step index: {self.t_index}"

def _is_evenly_spaced(values, rtol=1e-4) -> bool:
    """hv.Image assumes a regular grid; hv.QuadMesh is needed otherwise
    (e.g. a reduced Gaussian latitude axis). Cheap enough to just check."""
    if values.size < 3:
        return True
    d = np.diff(values.astype("float64"))
    return bool(np.allclose(d, d[0], rtol=rtol, atol=0.0))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_e2s_field(model_dir, base_or_var, level, t):
    """Load a single lat/lon field as an in-memory 2D DataArray.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing the model's .nc file(s), e.g. <sim_dir>/AIFS
    base_or_var : str
        Either a surface variable name (t2m, msl, sp, ...) or a leveled
        base name (u, v, t, q, z, ...)
    level : int or None
        Pressure level in hPa, used only when base_or_var is a leveled base.
    t : int
        Time index to select. Clamped to the available range.

    Returns
    -------
    (da, meta) : (xr.DataArray, FieldMeta)
        da is 2D, dims (CANON_LAT, CANON_LON), latitude ascending, values in
        memory.
    """
    ds = _open_dataset(model_dir)

    level_vars, surface_vars = parse_variable_groups(list(ds.data_vars))
    var_name = resolve_var_name(level_vars, surface_vars, base_or_var, level)

    lat_dim = _resolve_dim(ds, *LAT_NAMES)
    lon_dim = _resolve_dim(ds, *LON_NAMES)
    src_lat_dim, src_lon_dim = lat_dim, lon_dim

    da = ds[var_name]

    # CF-compliant files (post cf_convert.py) stack all pressure levels of a
    # variable into one array with a real `pressure` dimension, rather than
    # earth2studio's original flattened per-level variables (u100, u850, ...).
    # If this variable has a real pressure dim, select the requested level
    # from it directly here — resolve_var_name won't have done this, since
    # for a CF file base_or_var already IS the actual variable name (no
    # flattened-name lookup needed).
    if PRES_NAME in da.dims and level is not None:
        da = da.sel({PRES_NAME: level}, method="nearest")

    # Capture the initialization time, needed to turn a lead-time offset
    # into an absolute valid time.
    #
    # Post-cf_convert this lives in forecast_reference_time, because `time`
    # now holds the VALID times rather than a size-1 initialization axis.
    # The `time`-based fallback is for pre-conversion files, which still use
    # earth2studio's original layout.
    if "forecast_reference_time" in ds.coords:
        init_time = ds["forecast_reference_time"].values
    elif "time" in ds.coords and ds["time"].size == 1:
        init_time = ds["time"].values[0]
    else:
        init_time = None

    # Don't rely on the axis actually being named "time" — earth2studio
    # outputs may call it lead_time/step/forecast_time/etc., and there can be
    # more than one extra axis (e.g. a size-1 "time" alongside a size-7
    # "lead_time"). Squeeze singleton dims first, so whatever's left over is
    # the real forecast-step axis to select by `t`.
    extra_dims = [d for d in da.dims if d not in (lat_dim, lon_dim)]

    for d in list(extra_dims):
        if da.sizes[d] == 1:
            da = da.isel({d: 0})
            extra_dims.remove(d)

    select_dim = None
    t_clamped = t
    n_steps = 1
    if len(extra_dims) == 1:
        select_dim = extra_dims[0]
        n_steps = da.sizes[select_dim]
        t_clamped = min(max(t, 0), n_steps - 1)
        da = da.isel({select_dim: t_clamped})
    elif len(extra_dims) > 1:
        raise ValueError(
            f"Variable {var_name!r} has multiple non-singleton extra "
            f"dimensions {extra_dims!r} beyond lat/lon — can't tell which "
            f"one the time index should select."
        )

    # Compute valid time / forecast (lead) time for the title, from whichever
    # coordinate we actually selected by, if possible.
    valid_time = None
    lead_hours = None
    if select_dim is not None and select_dim in ds.coords:
        coord_val = ds[select_dim].values[t_clamped]
        if np.issubdtype(ds[select_dim].dtype, np.timedelta64):
            # select_dim is a lead-time-style offset (e.g. "lead_time")
            lead_hours = coord_val / np.timedelta64(1, "h")
            if init_time is not None:
                valid_time = init_time + coord_val
        elif np.issubdtype(ds[select_dim].dtype, np.datetime64):
            # select_dim IS itself the actual timestamp (ERA5-style)
            valid_time = coord_val
            first_val = ds[select_dim].values[0]
            lead_hours = (coord_val - first_val) / np.timedelta64(1, "h")

    # Realize the slice. Everything up to here was lazy.
    da = da.compute()

    # Drop leftover scalar coords (pressure, time, lead_time). HoloViews will
    # otherwise surface them as extra dimensions and, worse, rasterize can
    # choke trying to aggregate over a length-1 dim it didn't expect.
    da = da.reset_coords(drop=True)

    # forecast_period rides along on the time axis as an auxiliary
    # coordinate, so selecting a step leaves it behind as a scalar coord.
    # reset_coords should have taken it, but HoloViews surfaces stray
    # scalar coords as phantom dimensions and rasterize then fails trying
    # to aggregate over them, so this is worth being explicit about.
    for stray in ("forecast_period", "forecast_reference_time"):
        if stray in da.coords:
            da = da.drop_vars(stray)

    # Most of these models write latitude north-to-south. pcolormesh doesn't
    # care, but hv.Image with a descending y axis renders inverted and
    # datashader's aggregation of a descending coordinate is unreliable.
    # Normalize once here so both consumers see the same orientation.
    if lat_dim and da.sizes.get(lat_dim, 0) > 1:
        lat_vals = da[lat_dim].values
        if lat_vals[0] > lat_vals[-1]:
            da = da.isel({lat_dim: slice(None, None, -1)})

    # Canonicalize the spatial dimension names (see CANON_LAT/CANON_LON).
    # This has to happen before the regular-grid check and the transpose
    # below, both of which index by name.
    renames = {}
    if lat_dim and lat_dim != CANON_LAT:
        renames[lat_dim] = CANON_LAT
    if lon_dim and lon_dim != CANON_LON:
        renames[lon_dim] = CANON_LON
    if renames:
        da = da.rename(renames)
    if lat_dim:
        lat_dim = CANON_LAT
    if lon_dim:
        lon_dim = CANON_LON

    regular = True
    for dim in (lat_dim, lon_dim):
        if dim and dim in da.coords:
            regular = regular and _is_evenly_spaced(da[dim].values)

    # Put dims in (lat, lon) order so downstream code can rely on it.
    if lat_dim and lon_dim and da.dims != (lat_dim, lon_dim):
        da = da.transpose(lat_dim, lon_dim)

    meta = FieldMeta(
        var_name=var_name,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        src_lat_dim=src_lat_dim,
        src_lon_dim=src_lon_dim,
        valid_time=valid_time,
        lead_hours=lead_hours,
        t_index=t_clamped,
        n_steps=n_steps,
        regular_grid=regular,
    )
    return da, meta


def field_step_count(model_dir, base_or_var, level) -> int:
    """Length of the forecast-step axis, for setting the time slider's
    upper bound without loading a field."""
    ds = _open_dataset(model_dir)
    level_vars, surface_vars = parse_variable_groups(list(ds.data_vars))
    var_name = resolve_var_name(level_vars, surface_vars, base_or_var, level)
    lat_dim = _resolve_dim(ds, *LAT_NAMES)
    lon_dim = _resolve_dim(ds, *LON_NAMES)
    da = ds[var_name]
    if PRES_NAME in da.dims:
        da = da.isel({PRES_NAME: 0})
    extra = [d for d in da.dims
             if d not in (lat_dim, lon_dim) and da.sizes[d] > 1]
    return da.sizes[extra[0]] if len(extra) == 1 else 1


def field_range(model_dir, base_or_var, level, sample_steps=3):
    """Min/max sampled across a few evenly spaced forecast steps.

    Used to fix stable colour limits for an animation. Per-frame autoscaling
    makes a time sequence unreadable, because the scale slides underneath
    the data.
    """
    n = field_step_count(model_dir, base_or_var, level)
    idx = np.unique(np.linspace(0, n - 1, min(sample_steps, n), dtype=int))
    lo, hi = np.inf, -np.inf
    for t in idx:
        da, _ = load_e2s_field(model_dir, base_or_var, level, int(t))
        lo = min(lo, float(np.nanmin(da.values)))
        hi = max(hi, float(np.nanmax(da.values)))
    return lo, hi


# ---------------------------------------------------------------------------
# Static PNG rendering (Export Video, thumbnails)
# ---------------------------------------------------------------------------

def plot_e2s_field(model_dir, base_or_var, level, t, cmap="viridis",
                   vmin=None, vmax=None):
    """Render a single lat/lon field to a PNG.

    Signature and return value are unchanged from the pre-HoloViews version,
    so existing callers (Export Video) need no edits.

    Returns
    -------
    (buf, vmin_used, vmax_used) : (io.BytesIO, float, float)
        The rendered PNG, and the actual min/max values used for the
        colorbar (whether passed in explicitly or auto-computed).
    """
    da, meta = load_e2s_field(model_dir, base_or_var, level, t)

    data = da.values
    lats = da[meta.lat_dim].values if meta.lat_dim else np.arange(data.shape[0])
    lons = da[meta.lon_dim].values if meta.lon_dim else np.arange(data.shape[1])

    # Compute the actual min/max explicitly (rather than letting matplotlib
    # silently do it internally) so the values actually used can be reported
    # back to the caller for display.
    vmin_used = float(np.nanmin(data)) if vmin is None else vmin
    vmax_used = float(np.nanmax(data)) if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(7, 3.8))
    mesh = ax.pcolormesh(lons, lats, data, cmap=cmap, shading="auto",
                         vmin=vmin_used, vmax=vmax_used)

    ax.set_title(meta.title(), fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf, vmin_used, vmax_used
