"""Linked, zoomable plot grid for InferStudio.

Replaces the per-model Matplotlib cards with a single HoloViews Layout whose
axes are shared, so a box-zoom on any panel applies to all of them. Panels
are datashader-rasterized, so zooming re-aggregates server-side and resolves
finer structure rather than magnifying pixels.

Layout is (model, model-minus-other) per row:

    +----------------+  +--------------------------+
    | AIFS           |  | AIFS minus Aurora        |
    +----------------+  +--------------------------+
    | Aurora         |  | Aurora minus Pangu       |
    +----------------+  +--------------------------+

Panels are RESPONSIVE: they fill whatever width the browser window gives
them, at a fixed 2:1 aspect.

Instead of Bokeh's per-panel hover tooltip, a single readout above the grid
reports every model's value at the cursor position simultaneously — which is
the comparison the grid exists to support, and which a tooltip showing one
panel at a time cannot give. See _on_pointer.

The per-cell "Compute Difference" selectors moved to the sidebar (see
PlotGrid.diff_selectors) — Panel widgets can't be interleaved inside an
hv.Layout, and the sidebar is where the other field controls already live.

Axis linking depends on every panel sharing dimension NAMES; that is handled
upstream by earth2StudioPlot's CANON_LAT/CANON_LON canonicalization.

Typical use:

    grid = PlotGrid(models, model_dirs, DIFF_CACHE_DIR, state=shared_state)
    grid.set_time_bounds(n_steps)
    sidebar.append(grid.diff_selectors())
    plots_card = grid.card(title=suite_name)
    grid.refresh_clims_async()

    ...later, when the suite changes:
    grid.teardown()
"""

from __future__ import annotations

import threading
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import param
import panel as pn
import holoviews as hv
from holoviews.operation.datashader import rasterize

from visualization.earth2StudioPlot import load_e2s_field, field_range
from visualization.modelDiff import (
    load_diff_field,
    compute_model_difference,
    pair_dir_path,
    pair_name,
    symmetric_diff_range,
)

hv.extension("bokeh")


# Panel geometry.
#
# Panels are responsive rather than fixed-size: width comes from the browser
# window, height follows from ASPECT.
#
# ASPECT is `aspect`, NOT `data_aspect`. data_aspect constrains the axis
# RANGES to preserve a ratio, which fights box_zoom on any region that isn't
# 2:1; aspect constrains the plot's rendered shape and leaves the ranges
# alone.
ASPECT = 2.0
MIN_PANEL_WIDTH = 340
COLORBAR_WIDTH = 12

DIFF_CMAP = "coolwarm"

# Minimum seconds between readout updates. PointerXY fires on every mouse
# move — without a gate that is hundreds of websocket messages per second of
# cursor travel, each one re-rendering an HTML pane. 25 Hz is smooth to the
# eye and roughly an order of magnitude less traffic.
POINTER_MIN_INTERVAL = 0.04

# Fallback extent for placeholder panels drawn before any real field has been
# loaded. These models write longitude on 0..360, not -180..180 — a
# placeholder claiming the wrong range would drag every linked panel to it
# the moment shared_axes reconciles them. __init__ seeds _last_extent from a
# real field, so this is only a fallback if that read fails.
GLOBAL_EXTENT = (0.0, -90.0, 360.0, 90.0)

# "No explicit colour limits" — HoloViews reads (None, None) as autoscale
# from the data.
#
# This is deliberately NOT (nan, nan). A NaN clim produces a colour mapper
# with no usable range: every value maps to the low end of the colormap (a
# flat blue rectangle for coolwarm) and the colorbar is suppressed entirely,
# because there is no range to label. With None the same situation degrades
# to a correctly autoscaled panel instead.
CLIM_UNSET = (None, None)


class PlotGridState(param.Parameterized):
    """Shared state for the grid. Bind the existing sidebar widgets to this
    rather than to the plots directly (see app_layout.link_controls)."""

    # Unbounded deliberately: load_e2s_field clamps t to the available range,
    # so a stale or out-of-range index degrades to the last frame rather than
    # raising. Declaring bounds here would instead raise ValueError from
    # inside a param watcher during a suite switch, when the slider's new
    # value can arrive before set_time_bounds has run.
    time_index = param.Integer(default=0)

    variable = param.String(default="q")
    level = param.Integer(default=500, allow_None=True)

    # Either a colormap name or a matplotlib Colormap object — cmocean's
    # colormaps are not registered globally, so they arrive as objects.
    cmap = param.Parameter(default="viridis")
    cmap_min = param.Number(default=0.0)
    cmap_max = param.Number(default=0.0)

    # Colour limits, recomputed on variable/level change rather than per
    # frame. Per-frame autoscaling makes an animation unreadable, because
    # the scale slides underneath the data.
    field_clim = param.Tuple(default=CLIM_UNSET, length=2)
    diff_clim = param.Tuple(default=CLIM_UNSET, length=2)

    # model name -> other model name (or None for "no difference selected")
    diff_pairs = param.Dict(default={})

    # model name -> "ready" | "computing" | "error: ..."
    diff_status = param.Dict(default={})

    # Valid time of the currently displayed step, published by whichever
    # field panel is nominated first.
    header_text = param.String(default="")

    # Cursor readout rows, as (label, value) pairs in two columns:
    # (left_pairs, right_pairs). Empty until the cursor first enters a
    # panel. Stored as data rather than rendered HTML so that a time-slider
    # tick can re-render the timestamp beside the existing values without
    # waiting for the next mouse move.
    readout_rows = param.Tuple(default=((), ()), length=2)


# Which parameters each half of the grid actually depends on.
#
# Two streams rather than one: nothing in _field_cb reads diff_pairs or
# diff_status, so putting them on a shared stream would make a background
# diff landing, or a selector change, force six 721x1440 reloads of field
# data that hasn't changed.
#
# cmap, cmap_min/max, the clim tuples and readout_html appear in NEITHER
# list — those are restyles or pure display, and must not trigger a re-read.
_FIELD_PARAMS = ["time_index", "variable", "level"]
_DIFF_PARAMS = ["time_index", "variable", "level", "diff_pairs", "diff_status"]


def _schedule(fn):
    """Run fn on the Bokeh document's event loop.

    Param values touched from a worker thread must not be mutated directly:
    Panel needs the document lock to push the resulting change to the
    browser. pn.state.execute handles that when a server session exists,
    and falls through to a direct call in scripts and tests.
    """
    try:
        if pn.state.curdoc is not None:
            pn.state.execute(fn)
            return
    except Exception:
        pass
    fn()

def _fmt(value):
    """Format a field value compactly across the range of variables here.

    %.4g would switch to scientific notation below 1e-4, which for specific
    humidity differences is most values — and those strings are wide enough
    to wrap the readout column. Fixed-point with enough decimals keeps the
    width predictable and the decimal points aligned.
    """
    if value is None or not np.isfinite(value):
        return "—"
    a = abs(value)
    if a >= 1000 or (a > 0 and a < 1e-6):
        return f"{value:.3e}"      # genuinely needs an exponent
    if a >= 1:
        return f"{value:.3f}"
    return f"{value:.7f}"          # 6e-3, -3.4e-5 -> -0.0000344

class PlotGrid(param.Parameterized):
    """Builds and owns the linked plot grid."""

    def __init__(self, models, model_dirs, diff_cache_dir, state=None):
        """
        models         : list[str] — model names, in display order
        model_dirs     : dict[str, Path] — model name -> directory of .nc files
        diff_cache_dir : Path — e.g. /glade/derecho/scratch/pearse/
                                     .inferstudio_diff_cache/
        state          : PlotGridState, or None to create one
        """
        super().__init__()
        self.models = list(models)
        self.model_dirs = dict(model_dirs)
        self.diff_cache_dir = Path(diff_cache_dir)
        self.state = state if state is not None else PlotGridState()

        # Forecast length, set by set_time_bounds. Only frame_spec reads it;
        # the panels themselves rely on load_e2s_field's clamping.
        self.n_steps = 1

        # (kind, model) -> (DataArray, FieldMeta) for whatever each panel is
        # currently displaying. The readout samples these rather than
        # re-reading from disk, which is what makes a per-mouse-move lookup
        # affordable: it is an in-memory .sel on a 2D array.
        self._fields = {}
        self._fields_lock = threading.Lock()

        self._last_pointer = 0.0
        self._pointer_streams = []

        # Extent of the most recently loaded field, so placeholder panels can
        # match it. A placeholder with a different extent would drag every
        # linked panel to its range the moment shared_axes reconciles them —
        # which is why this is seeded from a real field here rather than left
        # on the constant: the first render is exactly when placeholder diff
        # panels sit next to real field panels.
        self._last_extent = GLOBAL_EXTENT
        try:
            da, meta = load_e2s_field(
                self.model_dirs[self.models[0]],
                self.state.variable, self.state.level, 0)
            lon, lat = da[meta.lon_dim].values, da[meta.lat_dim].values
            self._last_extent = (float(lon.min()), float(lat.min()),
                                 float(lon.max()), float(lat.max()))
        except Exception:
            pass   # constant fallback is fine; extent corrects on first render

        # Guards against spawning duplicate background jobs for one pair.
        self._pending = set()
        self._pending_lock = threading.Lock()

        # Set once this grid has been replaced, so any callback still in
        # flight when teardown() runs bails out instead of loading a field
        # from a suite that is no longer on screen.
        self._torn_down = False

        self._hv_pane = None

        self._field_stream = hv.streams.Params(self.state, _FIELD_PARAMS)
        self._diff_stream = hv.streams.Params(self.state, _DIFF_PARAMS)
        self._layout = None

    # -- lifecycle ------------------------------------------------------

    def teardown(self):
        """Detach this grid from the shared state.

        A stream holds a subscription to the PlotGridState for as long as it
        exists, and the state deliberately outlives every grid built against
        it (so the sidebar never needs rebinding). Without this, a replaced
        grid keeps servicing every parameter change off-screen — reloading
        fields from the previous suite's directories on each slider tick, and
        racing the live grid to write header_text.
        """
        self._torn_down = True
        for stream in (self._field_stream, self._diff_stream,
                       *self._pointer_streams):
            try:
                stream.clear()          # drop subscribers
            except Exception:
                pass
            try:
                stream.source = None    # drop the reference
            except Exception:
                pass
        self._pointer_streams = []
        with self._fields_lock:
            self._fields.clear()
        self._layout = None
        self._hv_pane = None

    # -- element construction ------------------------------------------

    def _responsive_opts(self):
        """Sizing opts shared by real and placeholder panels.

        responsive=True and frame_width are mutually exclusive in Bokeh, so
        this is an either/or with a fixed-frame approach, not an addition.
        """
        return dict(
            responsive=True,
            aspect=ASPECT,
            min_width=MIN_PANEL_WIDTH,
        )

    def _element(self, da, meta, title):
        """Wrap a loaded field as the appropriate HoloViews element."""
        lon = da[meta.lon_dim].values
        lat = da[meta.lat_dim].values
        self._last_extent = (
            float(lon.min()), float(lat.min()),
            float(lon.max()), float(lat.max()),
        )

        # hv.Image assumes an evenly spaced grid and will silently misplace
        # data on, say, a reduced Gaussian latitude axis. QuadMesh handles
        # irregular spacing correctly, at some rendering cost.
        cls = hv.Image if meta.regular_grid else hv.QuadMesh

        # kdims come from meta, which earth2StudioPlot has already
        # canonicalized — this is what lets shared_axes connect panels loaded
        # from models that name their dimensions differently on disk.
        #
        # No "hover" in tools: the readout above the grid replaces it, and
        # having both means two things reporting the same number in two
        # places, which invites them to disagree.
        return cls(da, kdims=[meta.lon_dim, meta.lat_dim]).opts(
            title=title,
            colorbar=True,
            colorbar_opts={"width": COLORBAR_WIDTH},
            active_tools=["box_zoom"],
            # framewise=False is what makes zoom survive a time-slider tick:
            # the axis ranges are not recomputed when the data changes.
            framewise=False,
            shared_axes=True,
            xlabel="longitude",
            ylabel="latitude",
            **self._responsive_opts(),
        )

    def _placeholder(self, title):
        """Blank panel that preserves grid geometry and axis ranges.

        Zeros rather than NaN: NaN renders transparent, which would leave the
        panel visually empty rather than showing a grey block.
        """
        left, bottom, right, top = self._last_extent
        return hv.Image(
            np.zeros((2, 2)), bounds=(left, bottom, right, top)
        ).opts(
            title=title,
            cmap=["#f0f0f0"],
            colorbar=False,
            framewise=False,
            shared_axes=True,
            xlabel="longitude",
            ylabel="latitude",
            toolbar=None,
            **self._responsive_opts(),
        )

    # -- callbacks ------------------------------------------------------

    def _field_cb(self, model, is_first, **_):
        if self._torn_down:
            return self._placeholder(model)

        try:
            da, meta = load_e2s_field(
                self.model_dirs[model],
                self.state.variable,
                self.state.level,
                self.state.time_index,
            )
        except Exception as exc:
            # An exception escaping a DynamicMap callback can leave the pane
            # permanently broken, so failures are rendered as a titled blank
            # panel instead. Full traceback goes to the server log.
            traceback.print_exc()
            with self._fields_lock:
                self._fields.pop(("field", model), None)
            return self._placeholder(f"{model} — {type(exc).__name__}: {exc}")

        with self._fields_lock:
            self._fields[("field", model)] = (da, meta)

        if is_first:
            # One panel is nominated to publish the shared valid-time header,
            # avoiding an extra read purely to populate it.
            label = meta.time_label()
            if label != self.state.header_text:
                _schedule(partial(setattr, self.state, "header_text", label))

        return self._element(da, meta, title=model)

    def _diff_cb(self, model, **_):
        if self._torn_down:
            return self._placeholder(model)

        other = self.state.diff_pairs.get(model)
        if not other:
            with self._fields_lock:
                self._fields.pop(("diff", model), None)
            return self._placeholder(f"{model} — no difference selected")

        status = self.state.diff_status.get(model, "")
        if status == "computing":
            return self._placeholder(f"{model} minus {other} — computing…")
        if status.startswith("error"):
            return self._placeholder(f"{model} minus {other} — {status}")

        try:
            da, meta = load_diff_field(
                self.model_dirs[model],
                self.model_dirs[other],
                self.diff_cache_dir,
                model,
                other,
                self.state.variable,
                self.state.level,
                self.state.time_index,
            )
        except Exception as exc:
            traceback.print_exc()
            with self._fields_lock:
                self._fields.pop(("diff", model), None)
            return self._placeholder(
                f"{model} minus {other} — {type(exc).__name__}: {exc}")

        with self._fields_lock:
            self._fields[("diff", model)] = (da, meta)

        return self._element(da, meta, title=f"{model} minus {other}")

    # -- cursor readout --------------------------------------------------

    @staticmethod
    def _sample(da, meta, x, y):
        """Nearest-neighbour lookup at a lon/lat position."""
        try:
            return float(
                da.sel({meta.lon_dim: x, meta.lat_dim: y},
                       method="nearest").item())
        except Exception:
            return None

    def _on_pointer(self, kind, x=None, y=None):
        """Update the readout from a cursor position over a panel.

        `kind` is "field" or "diff", carried in by the per-panel partial —
        it decides whether the readout reports model values or differences,
        so hovering a diff panel shows deltas rather than absolute values.

        Off-plot moves arrive as x=None; the last readout is left in place
        rather than blanked, since a value that vanishes whenever the cursor
        crosses a panel gap is more distracting than a slightly stale one.
        """
        if x is None or y is None or self._torn_down:
            return

        now = time.monotonic()
        if now - self._last_pointer < POINTER_MIN_INTERVAL:
            return
        self._last_pointer = now

        with self._fields_lock:
            entries = [(m, self._fields.get((kind, m))) for m in self.models]

        rows = []
        var_name = None
        for model, entry in entries:
            if entry is None:
                continue
            da, meta = entry
            var_name = var_name or meta.var_name
            rows.append((model, self._sample(da, meta, x, y)))

        if not rows:
            return

        label = f"{var_name} delta" if kind == "diff" else var_name
        self.state.readout_rows = (
            (("Variable", label), ("Lat", f"{y:.2f}"), ("Lon", f"{x:.2f}")),
            tuple((m, _fmt(v)) for m, v in rows),
        )

    @staticmethod
    def _readout_html(stamp, left, right):
        """Lay the readout out on a fixed grid.

        Alignment is the whole point. A plain inline run of spans reflows on
        every mouse move, because "0.0005791" and "-0.006" are different
        widths — so the labels visibly slide around and the readout is
        unreadable while the cursor moves. Two things fix that: a CSS grid
        with `max-content` label columns (label positions are set by the
        widest label once and then never move), and right-aligned values in
        a fixed-width column with tabular-nums and a monospace face, so
        decimal points and minus signs line up.
        """
        left, right = list(left), list(right)
        n = max(len(left), len(right))
        left += [("", "")] * (n - len(left))
        right += [("", "")] * (n - len(right))

        cells = []
        for (l_lab, l_val), (r_lab, r_val) in zip(left, right):
            cells.append(
                f"<div class='rl'>{l_lab}{':' if l_lab else ''}</div>"
                f"<div class='rv'>{l_val}</div>"
                f"<div class='rl'>{r_lab}{':' if r_lab else ''}</div>"
                f"<div class='rv'>{r_val}</div>"
            )

        return (
            "<style>"
            ".readout-wrap{display:flex;align-items:flex-start;gap:32px;}"
            ".readout{display:grid;"
            "grid-template-columns:max-content 4.5em max-content 7em;"
            "column-gap:10px;row-gap:1px;font-size:13px;"
            "width:max-content;}"
            # text-align must be explicit on BOTH classes. Without it the
            # cells inherit whatever alignment the surrounding Panel card
            # applies, which is why the labels came out right-aligned
            # against each other. justify-self pins the grid item to its
            # column edge; text-align positions the text inside that item.
            ".readout .rl{font-weight:600;text-align:left;justify-self:start;}"
            ".readout .rv{text-align:right;justify-self:stretch;"
            "font-variant-numeric:tabular-nums;"
            # nowrap is the important part: a scientific-notation value like
            # -3.443e-05 is wider than the column, and without this it wraps
            # to a second line, which grows the whole grid and pushes every
            # row below it down. Overflowing left is harmless here since the
            # column to the left is the label's, which has slack.
            "white-space:nowrap;"
            "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}"
            ".readout-time{font-size:13px;font-weight:600;white-space:nowrap;"
            "font-variant-numeric:tabular-nums;}"
            "</style>"
            "<div class='readout-wrap'>"
            f"<div class='readout-time'>{stamp}</div>"
            f"<div class='readout'>{''.join(cells)}</div>"
            "</div>"
        )

    # -- layout ---------------------------------------------------------

    def layout(self):
        """The hv.Layout. Built once; the DynamicMaps update in place."""
        if self._layout is not None:
            return self._layout

        panels = []
        for i, model in enumerate(self.models):
            field = hv.DynamicMap(
                partial(self._field_cb, model, i == 0),
                streams=[self._field_stream])
            diff = hv.DynamicMap(
                partial(self._diff_cb, model),
                streams=[self._diff_stream])

            # rasterize() must wrap the DynamicMap rather than being called
            # inside the callback — it is itself a dynamic operation and
            # cannot be returned from one.
            #
            # Styling is applied downstream of rasterize via .apply.opts with
            # param references, so changing the colormap or colour limits
            # restyles the existing render instead of re-reading the field.
            field = rasterize(field, precompute=True).apply.opts(
                clim=self.state.param.field_clim,
                cmap=self.state.param.cmap,
            )
            diff = rasterize(diff, precompute=True).apply.opts(
                clim=self.state.param.diff_clim,
                cmap=DIFF_CMAP,
            )

            # One PointerXY per panel, sourced from the object that actually
            # gets rendered. The `kind` bound into the subscriber is how the
            # readout knows whether the cursor is over a field or a diff.
            for obj, kind in ((field, "field"), (diff, "diff")):
                st = hv.streams.PointerXY(x=None, y=None, source=obj)
                st.add_subscriber(partial(self._on_pointer, kind))
                self._pointer_streams.append(st)

            panels += [field, diff]

        # sizing_mode here is the third of the three places responsive
        # sizing has to be declared: hv.Layout renders as a Bokeh gridplot,
        # and a gridplot does not propagate responsive sizing to its
        # children on its own.
        self._layout = hv.Layout(panels).cols(2).opts(
            shared_axes=True,
            merge_tools=True,
            toolbar="above",
            sizing_mode="stretch_width",
        )
        return self._layout

    def panel(self):
        """Valid time and cursor readout above the grid."""
        # Bound to BOTH the timestamp and the rows, so a slider tick
        # refreshes the time even though the cursor hasn't moved, and a
        # cursor move refreshes the values without dropping the time. Before
        # the cursor first enters a panel, readout_rows is empty and this
        # renders the timestamp alone.
        readout = pn.pane.HTML(
            pn.bind(self._readout_html,
                    self.state.param.header_text,
                    self.state.param.readout_rows.rx()[0],
                    self.state.param.readout_rows.rx()[1]),
            margin=(4, 0, 8, 12),
            min_height=64,
            sizing_mode="stretch_width",
        )

        self._hv_pane = pn.pane.HoloViews(
            self.layout(), sizing_mode="stretch_width")

        return pn.Column(
            readout,
            self._hv_pane,
            sizing_mode="stretch_width",
        )

    def card(self, title="Forecast fields", **kwargs):
        opts = dict(collapsible=False, sizing_mode="stretch_width")
        opts.update(kwargs)
        return pn.Card(self.panel(), title=title, **opts)

    # -- difference selectors (sidebar) ---------------------------------

    def diff_selectors(self, width=200):
        """Column of one Select per model, for the sidebar."""
        widgets = {}
        for model in self.models:
            others = [m for m in self.models if m != model]
            widgets[model] = pn.widgets.Select(
                name=f"{model} minus",
                options=["None"] + others,
                value="None",
                width=width,
            )

        def sync(*_):
            # Rebinding, not mutating: param only fires on assignment, so an
            # in-place dict update would silently fail to trigger a redraw.
            pairs = {
                m: (None if w.value == "None" else w.value)
                for m, w in widgets.items()
            }
            self.state.diff_pairs = pairs
            for a, b in pairs.items():
                if b:
                    self._ensure_diff_async(a, b)

        for w in widgets.values():
            w.param.watch(sync, "value")
        sync()

        self._diff_widgets = widgets
        return pn.Column("### Differences", *widgets.values())

    # -- background computation -----------------------------------------

    def _diff_exists(self, a, b):
        d = pair_dir_path(self.diff_cache_dir, a, b)
        return (d / f"{pair_name(a, b)}.nc").exists()

    def _set_status(self, model, value):
        def apply():
            self.state.diff_status = {**self.state.diff_status, model: value}
        _schedule(apply)

    def _ensure_diff_async(self, a, b):
        """Compute a model pair's difference off the event loop.

        Diffing two full forecast suites takes long enough that doing it
        inside the DynamicMap callback would block the Bokeh server thread
        and freeze the whole app — including the panels that have nothing to
        do with this pair. The callback only ever reads an already-computed
        file; this does the work and flips diff_status when it lands.
        """
        if self._diff_exists(a, b):
            self._set_status(a, "ready")
            # Refresh the diff clim even on a cache hit. Without this, a
            # suite whose diffs were computed in an earlier session never
            # gets diff_clim set at all: the initial refresh_clims runs
            # before any pair is selected and finds nothing, and this early
            # return used to be the only other path.
            self.refresh_diff_clim_async()
            return

        key = (a, b)
        with self._pending_lock:
            if key in self._pending:
                return
            self._pending.add(key)

        self._set_status(a, "computing")

        def work():
            try:
                compute_model_difference(
                    self.model_dirs[a], self.model_dirs[b],
                    self.diff_cache_dir, a, b,
                )
                if self._torn_down:
                    return
                self._set_status(a, "ready")
                self.refresh_diff_clim_async()
            except Exception as exc:
                traceback.print_exc()
                if not self._torn_down:
                    self._set_status(a, f"error: {exc}")
            finally:
                with self._pending_lock:
                    self._pending.discard(key)

        threading.Thread(target=work, daemon=True,
                         name=f"diff-{a}-minus-{b}").start()

    # -- colour limits ---------------------------------------------------

    def refresh_clims(self, sample_steps=3):
        """Recompute field and difference colour limits. Blocking."""
        self._refresh_field_clim(sample_steps)
        self._refresh_diff_clim(sample_steps)

    def _refresh_field_clim(self, sample_steps=3):
        if self._torn_down:
            return

        # An explicit non-zero min/max from the sidebar wins outright.
        if self.state.cmap_min != 0.0 or self.state.cmap_max != 0.0:
            _schedule(partial(setattr, self.state, "field_clim",
                              (self.state.cmap_min, self.state.cmap_max)))
            return

        lo, hi = np.inf, -np.inf
        for model in self.models:
            try:
                a, b = field_range(self.model_dirs[model], self.state.variable,
                                   self.state.level, sample_steps)
            except Exception:
                traceback.print_exc()
                continue
            lo, hi = min(lo, a), max(hi, b)

        if np.isfinite(lo) and np.isfinite(hi) and not self._torn_down:
            _schedule(partial(setattr, self.state, "field_clim", (lo, hi)))

    def _refresh_diff_clim(self, sample_steps=3):
        if self._torn_down:
            return

        m = 0.0
        for a, b in self.state.diff_pairs.items():
            if not b or not self._diff_exists(a, b):
                continue
            try:
                lo, hi = symmetric_diff_range(
                    self.model_dirs[a], self.model_dirs[b],
                    self.diff_cache_dir, a, b,
                    self.state.variable, self.state.level, sample_steps,
                )
            except Exception:
                traceback.print_exc()
                continue
            m = max(m, abs(lo), abs(hi))

        # A shared symmetric range across all difference panels means the
        # panels are directly comparable, and coolwarm's white sits exactly
        # at zero. With an asymmetric range an unbiased field reads as
        # biased, which is actively misleading on a difference plot.
        clim = (-m, m) if m > 0 else CLIM_UNSET
        if not self._torn_down:
            _schedule(partial(setattr, self.state, "diff_clim", clim))

    def refresh_clims_async(self, sample_steps=3):
        threading.Thread(
            target=self.refresh_clims, args=(sample_steps,),
            daemon=True, name="clim-refresh").start()

    def refresh_diff_clim_async(self, sample_steps=3):
        threading.Thread(
            target=self._refresh_diff_clim, args=(sample_steps,),
            daemon=True, name="diff-clim-refresh").start()

    # -- wiring ----------------------------------------------------------

    def set_time_bounds(self, n_steps):
        """Record the forecast length and clamp the current index into it.

        Named for what it used to do. It no longer touches
        param.time_index.bounds — partly because those bounds are gone (see
        PlotGridState.time_index), and partly because the old
        `self.state.param.time_index.bounds = ...` reached the CLASS-level
        Parameter object rather than this instance's.
        """
        self.n_steps = max(1, int(n_steps))
        if self.state.time_index > self.n_steps - 1:
            self.state.time_index = self.n_steps - 1

    def frame_spec(self):
        """What the video exporter needs to render frames itself.

        The grid is Bokeh-rendered client-side, so there is no server-side
        image to capture — the exporter reconstructs each frame through
        earth2StudioPlot.plot_e2s_field instead. Panels are returned in
        display order; each entry feeds
        plot_e2s_field(dir, variable, level, t, cmap=cmap,
                       vmin=clim[0], vmax=clim[1]).
        """
        panels = []
        for model in self.models:
            panels.append({
                "kind": "field",
                "title": model,
                "dir": self.model_dirs[model],
                "cmap": self.state.cmap,
                "clim": self.state.field_clim,
            })
            other = self.state.diff_pairs.get(model)
            if other and self._diff_exists(model, other):
                panels.append({
                    "kind": "diff",
                    "title": f"{model} minus {other}",
                    "dir": pair_dir_path(self.diff_cache_dir, model, other),
                    "cmap": DIFF_CMAP,
                    "clim": self.state.diff_clim,
                })
        return {
            "panels": panels,
            "variable": self.state.variable,
            "level": self.state.level,
            "n_steps": self.n_steps,
            "ncols": 2,
        }
