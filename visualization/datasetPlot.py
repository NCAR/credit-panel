# Step 1: Load datasets dynamically
import os
from pathlib import Path
import panel as pn
import param

from dimensions import VAR_NAME, TIME_NAME, LEV_NAME, PRES_NAME, LAT_NAME, LON_NAME
from visualization.era5_plot import plot_png, NETCDF_FILE
from visualization.earth2StudioPlot import parse_variable_groups, available_levels, plot_e2s_field
from visualization.modelDiff import compute_model_difference

pn.extension(raw_css=[Path("static/styles.css").read_text()])

# Models whose output uses the earth2studio flattened level-variable
# convention (u100, u850, ... instead of a real `level` dimension).
# TODO: WXFormer/MILES-CREDIT output is ERA5-style with a real level
# dimension and isn't handled by plot_e2s_field yet — it still needs its
# own branch here (probably routing back through era5_plot.plot_png).
EARTH2STUDIO_FORMAT_MODELS = {"AIFS", "Aurora", "Pangu"}


import matplotlib.colors as mcolors

try:
    import cmocean
    _CMOCEAN_AVAILABLE = True
except ImportError:
    _CMOCEAN_AVAILABLE = False


def _get_available_colormaps():
    """Return {name: matplotlib Colormap object} for cmocean's colormaps.
    Falls back to a small set of built-in matplotlib colormaps if cmocean
    isn't installed, so the app still works (install with `pip install
    cmocean` in the active env for the full cmocean set)."""
    if _CMOCEAN_AVAILABLE:
        names = getattr(cmocean.cm, "cmapnames", None)
        if names is None:
            # Fallback: introspect the module for Colormap instances directly
            names = [
                n for n in dir(cmocean.cm)
                if isinstance(getattr(cmocean.cm, n, None), mcolors.Colormap)
            ]
        cmaps = {}
        for name in names:
            obj = getattr(cmocean.cm, name, None)
            if isinstance(obj, mcolors.Colormap):
                cmaps[name] = obj
        if cmaps:
            return cmaps

    # cmocean not installed (or nothing found) — fall back to matplotlib builtins
    import matplotlib.pyplot as plt
    fallback_names = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"]
    return {name: plt.get_cmap(name) for name in fallback_names}


def _cmap_to_hex_swatch(cmap, n=32):
    """Sample a Colormap into a list of hex colors for widget swatch preview."""
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def _dropdown_width(options):
    if not options:
        return 100
    longest = max(len(str(opt)) for opt in options)
    return max(80, (longest * 9) + 40)


class SharedPlotControls(param.Parameterized):
    """Owns the Variable/Level selectors shared across every plot in the
    Visualization tab. Its choices are rebuilt from the union of variables
    across whichever datasets are currently checked in the browser."""

    var_name = param.String(default="")
    level_value = param.Integer(default=0)
    time_index = param.Integer(default=0)
    colormap = param.String(default="")
    # None means "auto" (matplotlib's own per-plot scaling). Once the user
    # actually types into the Min/Max box, this becomes a real number and
    # both models' plots share that exact same fixed color range, since
    # they're both driven by these same shared params. Distinguishing
    # "user typed this" from "code just displayed the live auto value" is
    # handled via the guarded watchers below (_on_cmap_min_input etc.),
    # NOT via a simple .link() — a plain link would treat every
    # programmatic display update as a real user override too, silently
    # turning "auto" into a permanently fixed value after the very first
    # render.
    cmap_min = param.Number(default=None, allow_None=True)
    cmap_max = param.Number(default=None, allow_None=True)

    def __init__(self, **params):
        super().__init__(**params)

        self.level_vars = {}
        self.surface_vars = []
        # Per-variable real pressure levels captured from CF-compliant
        # files (post cf_convert.py) during scanning — see
        # scan_single_dataset's `leveled_vars_cf` in app_layout.py. Takes
        # precedence over name-parsed levels (self.level_vars) whenever a
        # variable has real levels available, since parse_variable_groups
        # can't detect them from a CF file's plain variable names (e.g.
        # "q" has no trailing digit even though it now spans 13 levels).
        self._leveled_vars_cf = {}

        self.time_slider = pn.widgets.IntSlider(
            name="",
            start=0,
            end=1,
            value=0,
            disabled=True,
            show_value=False,
            sizing_mode="stretch_width",
        )
        self.time_slider.link(self, value="time_index")

        self._time_display = pn.pane.HTML(
            pn.bind(lambda v: f"<b>Time:</b> {v}", self.time_slider.param.value),
            styles={'line-height': '30px', 'font-size': '14px', 'white-space': 'nowrap'},
            width=90,
            margin=0,
        )

        # level_selector is created before var_selector/var_name is ever
        # assigned, since _update_level_options (watched on var_name) fires
        # the instant var_name changes and would otherwise reference a
        # not-yet-created widget.
        self.level_selector = pn.widgets.Select(
            name="",
            options=[0],
            value=0,
            disabled=True,
            max_width=150,
            sizing_mode="stretch_width",
        )

        # NOTE: Select widgets reset their own value to None if the current
        # value isn't a member of `options` — and an empty options list
        # means *nothing* is a valid member, including "". That None then
        # propagates through .link() into var_name, a param.String that
        # doesn't allow None, crashing the whole render. Using [""] (a
        # non-empty list containing the empty-string placeholder) instead
        # of [] avoids ever hitting that invalid state.
        self.var_selector = pn.widgets.Select(
            name="",
            options=[""],
            value="",
            disabled=True,
            max_width=150,
            sizing_mode="stretch_width",
        )
        self.var_selector.link(self, value="var_name")
        self.level_selector.link(self, value="level_value")

        # Colormap selector — built from cmocean (or a matplotlib fallback
        # if cmocean isn't installed). self._colormaps maps name -> the
        # actual Colormap object, used to resolve the selected name back to
        # a real colormap when plotting (rather than relying on matplotlib
        # recognizing cmocean's names as globally registered strings).
        self._colormaps = _get_available_colormaps()
        swatch_options = {
            name: _cmap_to_hex_swatch(cmap) for name, cmap in self._colormaps.items()
        }
        default_cmap_name = next(iter(swatch_options), "viridis")

        if hasattr(pn.widgets, "ColorMap"):
            # NOTE: ColorMap's `value` param holds one of the option *values*
            # (the swatch color list here), not its key/name — the widget
            # separately reflects the selected key as `value_name`. So we
            # seed `value` with the actual swatch list, and link on
            # `value_name` (the string) to drive our shared `colormap` param.
            self.colormap_selector = pn.widgets.ColorMap(
                options=swatch_options,
                value=swatch_options.get(default_cmap_name),
                ncols=1,
                swatch_width=150,
                name="",
                sizing_mode="stretch_width",
            )
            self.colormap_selector.link(self, value_name="colormap")
        else:
            # Older Panel version without the ColorMap widget — fall back
            # to a plain text dropdown (no swatch preview).
            self.colormap_selector = pn.widgets.Select(
                options=list(swatch_options.keys()),
                value=default_cmap_name,
                sizing_mode="stretch_width",
            )
            self.colormap_selector.link(self, value="colormap")
        self.colormap = default_cmap_name

        # Colormap Min/Max — always display the actual value currently in
        # effect (whether auto-computed from the data or user-fixed), no
        # placeholder text. Guard flags (_syncing_cmap_min/_max) let code
        # update the DISPLAYED value after each render without that write
        # being mistaken for the user manually overriding the range.
        self._syncing_cmap_min = False
        self._syncing_cmap_max = False

        self.cmap_min_input = pn.widgets.FloatInput(
            name="",
            value=0.0,
            sizing_mode="stretch_width",
        )
        self.cmap_max_input = pn.widgets.FloatInput(
            name="",
            value=0.0,
            sizing_mode="stretch_width",
        )

        def _on_cmap_min_input(event):
            if self._syncing_cmap_min:
                return  # this write came from _set_displayed_min, not the user
            self.cmap_min = event.new

        def _on_cmap_max_input(event):
            if self._syncing_cmap_max:
                return  # this write came from _set_displayed_max, not the user
            self.cmap_max = event.new

        self.cmap_min_input.param.watch(_on_cmap_min_input, 'value')
        self.cmap_max_input.param.watch(_on_cmap_max_input, 'value')

        self._row = pn.Column(
            pn.Row(
                self._time_display,
                self.time_slider,
                align="start",
                sizing_mode="stretch_width",
                css_classes=["widget-row"],
            ),
            pn.Row(
                pn.pane.HTML(
                    "<b>Variable</b>",
                    styles={'line-height': '30px', 'font-size': '14px', 'white-space': 'nowrap'},
                    width=90,
                    margin=0,
                ),
                self.var_selector,
                align="start",
                sizing_mode="stretch_width",
                css_classes=["widget-row"],
            ),
            pn.Row(
                pn.pane.HTML(
                    "<b>Level (hPa)</b>",
                    styles={'line-height': '30px', 'font-size': '14px', 'white-space': 'nowrap'},
                    width=90,
                    margin=0,
                ),
                self.level_selector,
                align="start",
                sizing_mode="stretch_width",
                css_classes=["widget-row"],
            ),
            pn.Column(
                pn.pane.HTML(
                    "<b>Colormap</b>",
                    styles={'line-height': '20px', 'font-size': '14px', 'white-space': 'nowrap'},
                    margin=0,
                ),
                self.colormap_selector,
                sizing_mode="stretch_width",
            ),
            pn.Row(
                pn.pane.HTML(
                    "<b>Colormap Min</b>",
                    styles={'line-height': '30px', 'font-size': '14px', 'white-space': 'nowrap'},
                    width=90,
                    margin=0,
                ),
                self.cmap_min_input,
                align="start",
                sizing_mode="stretch_width",
                css_classes=["widget-row"],
            ),
            pn.Row(
                pn.pane.HTML(
                    "<b>Colormap Max</b>",
                    styles={'line-height': '30px', 'font-size': '14px', 'white-space': 'nowrap'},
                    width=90,
                    margin=0,
                ),
                self.cmap_max_input,
                align="start",
                sizing_mode="stretch_width",
                css_classes=["widget-row"],
            ),
            sizing_mode="stretch_width",
        )

    def _set_displayed_min(self, value):
        """Update the Colormap Min box's displayed value without it being
        mistaken for the user manually fixing a range."""
        self._syncing_cmap_min = True
        try:
            self.cmap_min_input.value = value
        finally:
            self._syncing_cmap_min = False

    def _set_displayed_max(self, value):
        """Update the Colormap Max box's displayed value without it being
        mistaken for the user manually fixing a range."""
        self._syncing_cmap_max = True
        try:
            self.cmap_max_input.value = value
        finally:
            self._syncing_cmap_max = False

    def update_choices(self, dataset_keys, dataset_metadata):
        """Recompute variable/level/time choices from the union of
        vars2d+vars3d and the max ntime across the given dataset keys
        (typically browser.checked_items)."""
        all_vars = set()
        max_ntime = 0
        leveled_vars_cf = {}
        for key in dataset_keys:
            meta = dataset_metadata.get(key) or {}
            all_vars.update(meta.get("vars2d") or [])
            all_vars.update(meta.get("vars3d") or [])
            ntime = meta.get("ntime") or 0
            if ntime > max_ntime:
                max_ntime = ntime
            for var, levels in (meta.get("leveled_vars_cf") or {}).items():
                leveled_vars_cf.setdefault(var, set()).update(levels)
        self._leveled_vars_cf = {k: sorted(v) for k, v in leveled_vars_cf.items()}

        # Time: use the longest checked dataset's range. Shorter datasets
        # get their time index clamped automatically in plot_e2s_field, so
        # scrubbing past a shorter dataset's end just holds its last step.
        end = max(max_ntime - 1, 0)
        if end <= 0:
            end = 1  # avoid Bokeh's zero-width slider error; stays disabled below regardless
        self.time_slider.end = end
        self.time_slider.disabled = not bool(dataset_keys) or max_ntime <= 0
        if self.time_index > end:
            self.time_slider.value = 0
            self.time_index = 0

        self.level_vars, self.surface_vars = parse_variable_groups(sorted(all_vars))

        # Variables with real CF pressure levels should be treated as
        # "leveled" (base) choices even though their name has no trailing
        # digit — remove them from surface_vars if name-parsing put them
        # there, since parse_variable_groups can't detect this from the
        # name alone.
        for var in self._leveled_vars_cf:
            if var in self.surface_vars:
                self.surface_vars.remove(var)

        base_choices = sorted(set(self.level_vars.keys()) | set(self._leveled_vars_cf.keys()))
        surface_choices = sorted(self.surface_vars)
        var_choices = base_choices + surface_choices

        self.var_selector.options = var_choices if var_choices else [""]
        self.var_selector.max_width = _dropdown_width(var_choices) if var_choices else 150
        self.var_selector.disabled = not bool(var_choices)

        if not var_choices:
            self.var_selector.value = ""
            self.var_name = ""
            self._update_level_options()
            return

        # Keep the current selection if it's still valid for the new set of
        # checked datasets; otherwise fall back to the first choice.
        if self.var_name in var_choices:
            # value unchanged -> the var_name watcher below won't fire on
            # its own, so refresh level options explicitly.
            self._update_level_options()
        else:
            self.var_selector.value = var_choices[0]  # triggers var_name -> _update_level_options

    @param.depends('var_name', watch=True)
    def _update_level_options(self):
        # Prefer real CF pressure-coordinate levels when available for
        # this variable (post cf_convert.py); fall back to the old
        # flattened-variable-name parsing for legacy (pre-conversion)
        # files, where a variable like "u500" encodes its level in the
        # name rather than a real dimension.
        cf_levels = self._leveled_vars_cf.get(self.var_name)
        if cf_levels:
            levels = [int(round(lv)) for lv in cf_levels]
        else:
            levels = available_levels(self.level_vars, self.var_name)

        if levels:
            self.level_selector.options = levels
            self.level_selector.disabled = False
            if self.level_value not in levels:
                default_level = 500 if 500 in levels else levels[0]
                self.level_selector.value = default_level
                self.level_value = default_level
        else:
            self.level_selector.options = [0]
            self.level_selector.disabled = True
            self.level_selector.value = 0
            self.level_value = 0

    def panel(self):
        return self._row


class DatasetPlot2(param.Parameterized):
    dataset = param.String()
    metadata = param.Dict(default={})

    def __init__(self, controls, **params):
        super().__init__(**params)

        self.controls = controls
        self.metadata = self.metadata[self.dataset]

        # Suite entries (from scan_simulation_suite) carry a "models" dict
        # with each model's own path; older single-directory entries don't.
        if "models" in self.metadata:
            self.models = sorted(self.metadata["models"].keys())
            self.model_paths = {m: self.metadata["models"][m]["path"] for m in self.models}
        else:
            self.models = [self.dataset]
            self.model_paths = {self.dataset: self.metadata["path"]}

        # Per-model reactive state. Built once here (not rebuilt on every
        # variable/level/time change) so a model's "Compute Difference"
        # dropdown selection survives ordinary control changes rather
        # than resetting back to "no diff" on every render.
        self._diff_selectors = {}       # model -> Select widget
        self._diff_slots = {}           # model -> Column that holds the diff card (empty when none selected)
        # model -> cached difference dataset dir, for whichever diffs are
        # currently displayed. compute_model_difference already returns
        # this path; retaining it lets export_panels() below describe the
        # diff cards without recomputing anything.
        self._diff_paths = {}

        self._sections = pn.Column(sizing_mode="stretch_width")
        for model in self.models:
            self._sections.append(self._build_model_section(model))

    def _model_label(self, text):
        return pn.pane.HTML(
            f"<div style='text-align:center; font-size:20px; font-weight:bold; margin:0; padding:0;'>{text}</div>",
            sizing_mode="stretch_width",
            margin=0,
        )

    def _build_model_section(self, model):
        if model not in EARTH2STUDIO_FORMAT_MODELS:
            return pn.Column(
                self._model_label(model),
                pn.pane.Markdown(f"*Plotting for {model} output format isn't wired up yet.*"),
                align="center",
                sizing_mode="stretch_width",
                margin=0,
                css_classes=["plot-container"],
            )

        # The model's own reactive plot, bound to the shared controls —
        # same pn.bind pattern used for the time label in SharedPlotControls.
        model_view = pn.bind(
            self._render_model,
            model,
            self.controls.param.var_name,
            self.controls.param.level_value,
            self.controls.param.time_index,
            self.controls.param.colormap,
            self.controls.param.cmap_min,
            self.controls.param.cmap_max,
        )

        other_models = [m for m in self.models if m != model and m in EARTH2STUDIO_FORMAT_MODELS]
        diff_options = ["\u2014"] + [f"{model} minus {other}" for other in other_models]
        diff_selector = pn.widgets.Select(
            name="Compute Difference",
            options=diff_options,
            value="\u2014",
            width=220,
        )
        self._diff_selectors[model] = diff_selector

        diff_slot = pn.Column(
            sizing_mode="stretch_width",
            styles={"min-width": "0"},
        )  # empty until a diff is picked
        self._diff_slots[model] = diff_slot

        def _on_diff_change(event, model=model):
            self._on_diff_selected(model, event.new)

        diff_selector.param.watch(_on_diff_change, 'value')

        model_card = pn.Column(
            self._model_label(model),
            pn.panel(model_view, sizing_mode="stretch_width"),
            pn.Row(
                pn.Spacer(sizing_mode="stretch_width"),
                diff_selector,
                sizing_mode="stretch_width",
            ),
            align="center",
            sizing_mode="stretch_width",
            margin=0,
            css_classes=["plot-container"],
            styles={"min-width": "0"},
        )

        return pn.Row(
            model_card,
            diff_slot,
            sizing_mode="stretch_width",
            styles={
                "display": "grid",
                "grid-template-columns": "1fr 1fr",
                "gap": "10px",
                "width": "100%",
            },
        )

    def _render_model(self, model, var_name, level_value, time_index, colormap_name, cmap_min, cmap_max):
        if not var_name:
            return pn.pane.Markdown("*No variable selected*")

        cmap = self.controls._colormaps.get(colormap_name, "viridis")
        model_path = self.model_paths.get(model)

        try:
            buf, vmin_used, vmax_used = plot_e2s_field(
                model_dir=model_path,
                base_or_var=var_name,
                level=level_value,
                t=time_index,
                cmap=cmap,
                vmin=cmap_min,
                vmax=cmap_max,
            )
            return pn.pane.PNG(
                buf,
                sizing_mode="scale_width",
                align="center",
                height=None,
                min_height=None,
                max_height=None,
                margin=0,
            )
        except Exception as e:
            return pn.pane.Markdown(f"*Error plotting {model}: {e}*")

    def _on_diff_selected(self, model, selected_option):
        slot = self._diff_slots[model]

        if selected_option == "\u2014":
            slot.objects = []
            self._diff_paths.pop(model, None)   # no diff card on screen anymore
            return

        # "AIFS minus Aurora" -> other = "Aurora"
        other = selected_option.split(" minus ", 1)[1]

        slot.objects = [
            pn.Column(
                pn.indicators.LoadingSpinner(value=True, width=30, height=30, align="center"),
                pn.pane.Markdown("*Computing difference...*", align="center"),
                align="center",
                sizing_mode="stretch_width",
            )
        ]

        sim_dir = Path(self.metadata["path"])
        # Deliberately NOT a sibling of sim_dir (e.g. sim_dir.parent /
        # f"{sim_dir.name}_diffs") — that lands inside data_dir itself,
        # which scan_datasets walks directly, so it would show up as a
        # spurious extra dataset in the Datasets list. Using a dedicated,
        # clearly-separate cache root avoids that class of bug entirely.
        cache_dir = Path(f"/glade/derecho/scratch/{os.environ['USER']}/.inferstudio_diff_cache") / sim_dir.name

        try:
            diff_path = compute_model_difference(
                self.model_paths[model], self.model_paths[other],
                cache_dir, model, other,
            )
        except Exception as e:
            slot.objects = [pn.pane.Markdown(f"*Error computing difference: {e}*")]
            self._diff_paths.pop(model, None)   # nothing renderable to export
            return

        self._diff_paths[model] = diff_path

        label = f"{model} minus {other}"
        diff_view = pn.bind(
            self._render_diff,
            diff_path,
            self.controls.param.var_name,
            self.controls.param.level_value,
            self.controls.param.time_index,
        )
        diff_card = pn.Column(
            self._model_label(label),
            pn.panel(diff_view, sizing_mode="stretch_width"),
            align="center",
            sizing_mode="stretch_width",
            margin=0,
            css_classes=["plot-container"],
        )
        slot.objects = [diff_card]

    def _render_diff(self, diff_path, var_name, level_value, time_index):
        if not var_name:
            return pn.pane.Markdown("*No variable selected*")

        try:
            # First pass: auto-range, purely to discover the actual
            # min/max of this difference field.
            _, vmin_auto, vmax_auto = plot_e2s_field(
                model_dir=diff_path,
                base_or_var=var_name,
                level=level_value,
                t=time_index,
                cmap="coolwarm",
                vmin=None,
                vmax=None,
            )
            # Second pass: symmetric range around zero, so the diverging
            # colormap's center (white/neutral) actually lands on zero
            # difference rather than on some arbitrary asymmetric midpoint.
            max_abs = max(abs(vmin_auto), abs(vmax_auto))
            buf, _, _ = plot_e2s_field(
                model_dir=diff_path,
                base_or_var=var_name,
                level=level_value,
                t=time_index,
                cmap="coolwarm",
                vmin=-max_abs,
                vmax=max_abs,
            )
            return pn.pane.PNG(
                buf,
                sizing_mode="scale_width",
                align="center",
                height=None,
                min_height=None,
                max_height=None,
                margin=0,
            )
        except Exception as e:
            return pn.pane.Markdown(f"*Error plotting difference: {e}*")

    def export_panels(self):
        """Describe what this plot grid is currently rendering.

        Returns a list of rows, each a list of (label, model_dir, is_diff)
        tuples, mirroring the on-screen layout: one row per plottable
        model, with its difference card alongside when one is selected.
        Models without a wired-up plotting branch are omitted, since they
        have nothing to render into a frame.

        Deliberately returns plain tuples rather than any videoExport type,
        so this module stays independent of the export machinery.
        """
        rows = []
        for model in self.models:
            if model not in EARTH2STUDIO_FORMAT_MODELS:
                continue
            row = [(model, self.model_paths[model], False)]
            diff_path = self._diff_paths.get(model)
            if diff_path:
                row.append((self._diff_selectors[model].value, diff_path, True))
            rows.append(row)
        return rows

    def panel(self):
        return pn.Column(
            pn.pane.Markdown(f"### {self.dataset}"),
            self._sections,
            align="center",
            sizing_mode="stretch_width",
        )
