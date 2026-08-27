#!/usr/bin/env python
"""Round two: find what splits the InferStudio grid's axis linking.

Round one (checkLinking.py) ruled out the obvious suspects. Plain images,
two colormaps, DynamicMap, rasterize, and both the split and uniform
apply.opts chains ALL came back fully linked - one x_range across four
figures. So none of those is the cause.

What round one did NOT reproduce, because it rendered through
hv.renderer('bokeh').get_plot() with fixed frame sizes:

  1. responsive=True + aspect, instead of frame_width/frame_height. The app
     uses responsive sizing, which changes how Bokeh builds the plot.
  2. The Panel wrapper. The app goes hv.Layout -> pn.pane.HoloViews ->
     Column -> Card. Panel may rebuild rather than passing through whatever
     HoloViews constructed.
  3. Real kdims. The app names its dimensions longitude/latitude via
     kdims=[...]; round one used hv.Image(array, bounds=...) defaults.

This script varies exactly those three, holding everything else at the
configuration round one proved innocent.

Usage:

    conda activate creditJun3
    python checkLinking2.py

Still headless - the Panel cases use get_root() on a fresh Document, which
is the same construction path a served app takes, without needing a browser.
"""

import numpy as np
import param
import panel as pn
import holoviews as hv
from holoviews.operation.datashader import rasterize

hv.extension("bokeh")
pn.extension()

from bokeh.document import Document
from bokeh.models import Plot


ASPECT = 2.0
EXTENT = (0.0, -90.0, 360.0, 90.0)
LON = np.linspace(0.0, 357.5, 72)
LAT = np.linspace(-87.5, 87.5, 36)


class _State(param.Parameterized):
    """Stand-in for PlotGridState, carrying only what the opts chain uses."""
    field_clim = param.Tuple(default=(None, None), length=2)
    diff_clim = param.Tuple(default=(None, None), length=2)
    cmap = param.Parameter(default="viridis")
    tick = param.Integer(default=0)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((36, 72))


def _fixed_size():
    return dict(frame_width=200, frame_height=100)


def _responsive_size():
    """What the app actually uses. responsive and frame_* are mutually
    exclusive in Bokeh, so this is an either/or with _fixed_size."""
    return dict(responsive=True, aspect=ASPECT, min_width=340)


def _panel_opts(title, cmap, sizing):
    return dict(
        title=title,
        cmap=cmap,
        colorbar=True,
        default_tools=[],
        tools=["pan", "wheel_zoom", "reset"],
        active_tools=["pan", "wheel_zoom"],
        framewise=False,
        shared_axes=True,
        xlabel="longitude",
        ylabel="latitude",
        **sizing,
    )


def _image(title, seed=0, cmap="viridis", sizing=None, real_kdims=False):
    """A panel. real_kdims builds it the way the app does - an xarray-style
    DataArray with named longitude/latitude dims - rather than a bare array
    with default x/y dimension names."""
    sizing = sizing or _fixed_size()
    if real_kdims:
        import xarray as xr
        da = xr.DataArray(
            _data(seed),
            coords={"latitude": LAT, "longitude": LON},
            dims=["latitude", "longitude"],
        )
        el = hv.Image(da, kdims=["longitude", "latitude"])
    else:
        el = hv.Image(_data(seed), bounds=EXTENT)
    return el.opts(**_panel_opts(title, cmap, sizing))


def _build(state, sizing, real_kdims=False, split_apply=True):
    """The app's panel construction: DynamicMap -> rasterize -> apply.opts,
    with the field/diff split in the opts chain."""
    stream = hv.streams.Params(state, ["tick"])
    panels = []
    for i in range(4):
        dm = hv.DynamicMap(
            lambda i=i, **kw: _image(
                f"p{i}", seed=i,
                cmap="viridis" if i % 2 == 0 else "coolwarm",
                sizing=sizing, real_kdims=real_kdims),
            streams=[stream])
        r = rasterize(dm, precompute=True)
        if split_apply and i % 2 == 1:
            r = r.apply.opts(clim=state.param.diff_clim)
        else:
            r = r.apply.opts(clim=state.param.field_clim,
                             cmap=state.param.cmap)
        panels.append(r)
    return hv.Layout(panels).cols(2).opts(
        shared_axes=True, toolbar=None, sizing_mode="stretch_width")


def _count(figs):
    xs = {id(f.x_range) for f in figs}
    ys = {id(f.y_range) for f in figs}
    return len(figs), len(xs), len(ys)


def via_get_plot(layout):
    """Round one's render path: straight through the HoloViews renderer."""
    plot = hv.renderer("bokeh").get_plot(layout)
    return list(plot.state.select({"type": Plot}))


def via_panel(layout, wrap_card=False):
    """The app's render path: through pn.pane.HoloViews and the layout
    containers, materialised against a real Document the way a served
    session does."""
    pane = pn.pane.HoloViews(layout, sizing_mode="stretch_width")
    obj = pn.Column(pane, sizing_mode="stretch_width")
    if wrap_card:
        obj = pn.Card(obj, title="grid", collapsible=False,
                      sizing_mode="stretch_width")
    doc = Document()
    root = obj.get_root(doc)
    return list(root.select({"type": Plot}))


def report(name, figs):
    try:
        n, nx, ny = _count(figs)
    except Exception as exc:
        print(f"{name:<48} FAILED: {type(exc).__name__}: {exc}")
        return
    verdict = ("all linked" if nx == 1 else
               "PAIRWISE <-- reproduced" if nx == 2 else
               "unlinked" if nx == n else
               "partial")
    print(f"{name:<48} figs={n}  x={nx}  y={ny}   {verdict}")


def safely(name, fn):
    try:
        report(name, fn())
    except Exception as exc:
        import traceback
        print(f"{name:<48} ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()


def main():
    import bokeh
    print(f"holoviews {hv.__version__}   bokeh {bokeh.__version__}   "
          f"panel {pn.__version__}")
    print()
    print(f"{'case':<48} result")
    print("-" * 92)

    st = _State()

    # Control: round one's winning configuration, to confirm this script
    # agrees with the last one before we trust anything else it says.
    safely("control: fixed size, get_plot",
           lambda: via_get_plot(_build(st, _fixed_size())))

    # Variable 1: responsive sizing.
    safely("responsive sizing, get_plot",
           lambda: via_get_plot(_build(st, _responsive_size())))

    # Variable 2: the Panel render path.
    safely("fixed size, via Panel",
           lambda: via_panel(_build(st, _fixed_size())))

    # Both together - this is closest to the app.
    safely("responsive + via Panel",
           lambda: via_panel(_build(st, _responsive_size())))

    # And inside a Card, which is the last container the app adds.
    safely("responsive + via Panel + Card",
           lambda: via_panel(_build(st, _responsive_size()), wrap_card=True))

    # Variable 3: real named kdims rather than default x/y.
    safely("responsive + Panel + real kdims",
           lambda: via_panel(_build(st, _responsive_size(),
                                    real_kdims=True)))

    # If the above reproduces it, does making the opts chain uniform help
    # under those same conditions? Round one could not tell us, because
    # nothing was broken there to begin with.
    safely("^ same, but UNIFORM apply.opts",
           lambda: via_panel(_build(st, _responsive_size(),
                                    real_kdims=True, split_apply=False)))

    print()
    print("The first case that reports PAIRWISE is the ingredient that")
    print("breaks the linking. If the last line comes back 'all linked'")
    print("while the one above it is PAIRWISE, the uniform opts chain is")
    print("the fix. If NOTHING reproduces it, the cause is something else")
    print("in the app entirely and the next step is to dump the range ids")
    print("from a live session rather than guessing here.")


if __name__ == "__main__":
    main()
