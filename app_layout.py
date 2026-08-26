# app_layout.py
import os
import base64
import warnings
import panel as pn
import xarray as xr
from pathlib import Path
from functools import lru_cache

from dimensions import LEV_NAME, PRES_NAME, LAT_NAME, LON_NAME, resolve_nc_glob

from visualization.datasetSelector2 import DatasetBrowser
from visualization.metadata import DatasetMetadata
from visualization.datasetPlot import DatasetPlot2, SharedPlotControls
from visualization.forecastStatsPanel import ForecastStatsPanel
from visualization.loadSuiteDialog import LoadSuiteDialog
from visualization.videoExport import VideoExportPanel

from inference.commandRunner import CommandRunner
from inference.inferenceTab import InferenceTab

# --- Static asset locations ------------------------------------------------
# Resolved relative to THIS module, not the process working directory, so the
# paths hold regardless of where `panel serve` is launched from (OOD's
# script.sh.erb does not necessarily cd into the repo root).
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_LOGO_DIR = _STATIC_DIR / "logo"

_MIME = {".png": "image/png", ".ico": "image/x-icon", ".svg": "image/svg+xml"}

@lru_cache(maxsize=None)
def logo_uri(name: str) -> str:
    """Embed a file from static/logo as a data URI (proxy-prefix safe).

    The trailing "#<filename>" fragment exists for Panel: its
    _get_favicon_type() dispatches on the string suffix of the favicon
    value and raises "favicon type not supported" for a bare data URI.
    The fragment satisfies that check and is discarded by the browser
    before the base64 payload is decoded.
    """
    path = _LOGO_DIR / name
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_MIME[path.suffix.lower()]};base64,{data}#{path.name}"

def _resolve_dim(ds, *candidates):
    """Return the first candidate name that exists as a dimension in ds."""
    for name in candidates:
        if name in ds.sizes:
            return name
    return None

def scan_single_dataset(dataset_dir: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with xr.open_mfdataset(resolve_nc_glob(dataset_dir), engine="netcdf4", autoclose=True, data_vars='all') as ds:
            lat_dim = _resolve_dim(ds, LAT_NAME, "lat")
            lon_dim = _resolve_dim(ds, LON_NAME, "lon")

            # Forecast-style earth2studio output (AIFS/Aurora/Pangu/...) has
            # a size-1 `time` dim (the init/cycle time) plus a separate
            # `lead_time` dim holding the actual forecast steps. ERA5-style
            # files have no `lead_time` and step through `time` directly.
            # Prefer `lead_time` as the "number of forecast steps" dimension
            # whenever it's present and non-trivial, instead of always
            # reading the (possibly size-1) init-time dim.
            has_lead_time = "lead_time" in ds.sizes and ds.sizes["lead_time"] > 1

            if has_lead_time:
                ntime = int(ds.sizes["lead_time"])
                init_time = ds.time.values[0]
                lead_times = ds.lead_time.values
                stime = str((init_time + lead_times.min()).astype("datetime64[s]"))
                etime = str((init_time + lead_times.max()).astype("datetime64[s]"))
            else:
                ntime = len(ds.time)
                stime = str(ds.time.values[0].astype("datetime64[s]"))
                etime = str(ds.time.values[-1].astype("datetime64[s]"))

            # CF-compliant files (post cf_convert.py) stack pressure levels
            # into a real `pressure` dimension on each leveled variable,
            # rather than earth2studio's original flattened per-level
            # variable names (u100, u850, ...). Capture which variables
            # actually have this real dimension, and the real coordinate
            # values, so the Level (hPa) dropdown can be populated
            # correctly even though these variable names no longer end in
            # a digit (parse_variable_groups can't detect levels from the
            # name alone for these).
            leveled_vars_cf = {}
            if PRES_NAME in ds.coords:
                pressure_values = sorted(float(x) for x in ds[PRES_NAME].values.tolist())
                for v in ds.data_vars:
                    if PRES_NAME in ds[v].dims:
                        leveled_vars_cf[v] = pressure_values

            return {
                "path": str(dataset_dir),
                "ntime": ntime,
                "nlev": len(ds.get(LEV_NAME, [])),
                "nplev": int(ds.sizes.get(PRES_NAME, 0)),
                "nlat": int(ds.sizes[lat_dim]) if lat_dim else 0,
                "nlon": int(ds.sizes[lon_dim]) if lon_dim else 0,
                "stime": stime,
                "etime": etime,
                "vars2d": [v for v in ds.data_vars if len(ds[v].dims) <= 3],
                "vars3d": [v for v in ds.data_vars if len(ds[v].dims) > 3],
                "leveled_vars_cf": leveled_vars_cf,
            }

def scan_simulation_suite(sim_dir: Path) -> dict:
    """Scan every model subdirectory under a simulation dir and combine
    them into a single metadata entry representing the whole suite."""
    model_meta = {}
    errors = {}
    for model_dir in sorted(p for p in sim_dir.iterdir() if p.is_dir()):
        try:
            model_meta[model_dir.name] = scan_single_dataset(model_dir)
        except Exception as e:
            errors[model_dir.name] = str(e)

    if not model_meta:
        raise RuntimeError(f"No scannable model outputs found under {sim_dir}")

    any_model = next(iter(model_meta.values()))
    vars2d = sorted(set().union(*(m["vars2d"] for m in model_meta.values())))
    vars3d = sorted(set().union(*(m["vars3d"] for m in model_meta.values())))

    leveled_vars_cf = {}
    for m in model_meta.values():
        for var, levels in m.get("leveled_vars_cf", {}).items():
            leveled_vars_cf.setdefault(var, set()).update(levels)
    leveled_vars_cf = {k: sorted(v) for k, v in leveled_vars_cf.items()}

    return {
        "path": str(sim_dir),
        "models": model_meta,      # per-model breakdown: {model_name: {...}}
        "model_errors": errors,
        "ntime": any_model["ntime"],
        "nlev": any_model["nlev"],
        "nplev": any_model["nplev"],
        "nlat": any_model["nlat"],
        "nlon": any_model["nlon"],
        "stime": any_model["stime"],
        "etime": any_model["etime"],
        "vars2d": vars2d,
        "vars3d": vars3d,
        "leveled_vars_cf": leveled_vars_cf,
    }

def scan_datasets(data_dir):
    metadata = {}
    for d in data_dir.iterdir():
        if not d.is_dir():
            continue
        subdirs_with_nc = [
            sub for sub in d.iterdir()
            if sub.is_dir() and any(sub.glob("*.nc"))
        ]
        try:
            if subdirs_with_nc:
                metadata[d.name] = scan_simulation_suite(d)
            else:
                metadata[d.name] = scan_single_dataset(d)
        except Exception as e:
            print(f"Skipping {d.name}: {e}")
            continue
    return metadata


def build_app(data_dir):
    dataset_metadata = scan_datasets(data_dir)
    datasets = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    browser = DatasetBrowser(datasets=datasets)
    meta_panel = DatasetMetadata(metadata=dataset_metadata)

    controls = SharedPlotControls()
    controls.update_choices(browser.checked_items, dataset_metadata)

    def sync_active(event):
        meta_panel.active_key = event.new

    browser.param.watch(sync_active, 'active_dataset')

    def sync_controls(event):
        controls.update_choices(event.new, dataset_metadata)

    browser.param.watch(sync_controls, 'checked_items')

    DEFAULT_DATASET = "ExampleDataset"
    if DEFAULT_DATASET in dataset_metadata:
        browser.checked_items = [DEFAULT_DATASET]
        browser.active_dataset = DEFAULT_DATASET
    elif DEFAULT_DATASET and DEFAULT_DATASET != "REPLACE_WITH_YOUR_FOLDER_NAME":
        print(f"Warning: default dataset {DEFAULT_DATASET!r} not found under {data_dir}")

    inference_tab = InferenceTab()

    def _on_new_output(event):
        sim_dir = Path(event.new)
        if not sim_dir.is_dir():
            if pn.state.notifications:
                pn.state.notifications.error(f"sim_dir not found: {sim_dir}", duration=0)
            return

        key = sim_dir.name
        try:
            dataset_metadata[key] = scan_simulation_suite(sim_dir)
        except Exception as e:
            if pn.state.notifications:
                pn.state.notifications.error(f"Could not scan {key}: {e}", duration=0)
            with open('/tmp/debug.log', 'a') as f:
                f.write(f"scan_simulation_suite failed for {key}: {e}\n")
            return
        with open('/tmp/debug.log', 'a') as f:
            f.write(f"scanned {key}: vars2d={dataset_metadata[key]['vars2d']} vars3d={dataset_metadata[key]['vars3d']} models={list(dataset_metadata[key].get('models', {}).keys())}\n")
        for model, err in dataset_metadata[key].get("model_errors", {}).items():
            if pn.state.notifications:
                pn.state.notifications.error(f"Could not scan {key}/{model}: {err}", duration=0)
        meta_panel.metadata = dict(dataset_metadata)
        browser.add_datasets([key])
        if browser.checked_items != [key]:
            browser.checked_items = [key]
        browser.active_dataset = key
        tabs.active = 0
        if pn.state.notifications:
            pn.state.notifications.info(f"browser now has: {browser.datasets}", duration=0)

    inference_tab.param.watch(_on_new_output, 'outputDirectory')

    # --- Load Existing Suite (Visualization tab) ---------------------- #
    # Lets a user browse to and select a simulation suite directory from
    # a PREVIOUS InferStudio session (rather than only ever seeing suites
    # produced in the current session), reusing the exact same
    # scan_simulation_suite/AIFS-Aurora-etc. scanning logic used for
    # freshly-completed inference runs above.
    def _on_load_existing_suite(path_str):
        sim_dir = Path(path_str)

        if not sim_dir.is_dir():
            load_suite_dialog.report_error(f"{sim_dir} is not a directory.")
            return

        key = sim_dir.name
        try:
            dataset_metadata[key] = scan_simulation_suite(sim_dir)
        except Exception as e:
            # scan_simulation_suite raises RuntimeError specifically when
            # no supported model (AIFS, Aurora, ...) output was found —
            # this is also where any other scan failure surfaces (e.g.
            # unreadable/corrupt files). Full detail (including the full
            # path) goes in the dialog's own inline error; the toast
            # notification is kept short deliberately, since Notyf-style
            # toasts have a fixed size and don't wrap/expand for long
            # text — putting the full path + exception text in the toast
            # was getting visually clipped.
            load_suite_dialog.report_error(
                f"Could not load a simulation suite from {sim_dir}: {e}"
            )
            if pn.state.notifications:
                pn.state.notifications.error(
                    f"Could not load suite from {key} \u2014 see dialog for details.",
                    duration=0,
                )
            return

        for model, err in dataset_metadata[key].get("model_errors", {}).items():
            if pn.state.notifications:
                pn.state.notifications.error(f"Could not scan {key}/{model}: {err}", duration=0)

        meta_panel.metadata = dict(dataset_metadata)
        browser.add_datasets([key])
        if browser.checked_items != [key]:
            browser.checked_items = [key]
        browser.active_dataset = key
        load_suite_dialog.close()
        if pn.state.notifications:
            pn.state.notifications.info(f"Loaded suite: {key}", duration=0)

    load_suite_dialog = LoadSuiteDialog(
        start_path=Path(f"/glade/derecho/scratch/{os.environ['USER']}"),
        on_select=_on_load_existing_suite,
    )
    # Match the Datasets checkbox panel's width exactly: that panel is
    # sizing_mode="stretch_width" with margin=(0, 10, 0, 0) (a 10px right
    # margin — see DatasetBrowser._column in datasetSelector2.py), so
    # giving this button the identical sizing_mode + right margin makes
    # both stretch to the exact same effective width within the sidebar.
    load_suite_dialog.open_button.sizing_mode = "stretch_width"
    load_suite_dialog.open_button.margin = (10, 10, 0, 0)

    # Holds the DatasetPlot2 instance currently on screen, so the video
    # exporter can ask it what it's rendering. A dict rather than a bare
    # local because plot_grid (a closure) needs to rebind it on every
    # dataset change, and the exporter needs to see that new value.
    _active_plot = {"obj": None}

    @pn.depends(browser.param.checked_items)
    def plot_grid(datasets):
        if not datasets:
            _active_plot["obj"] = None
            return pn.pane.Markdown("### Select one or more datasets")
        ds = datasets[0]
        plot = DatasetPlot2(controls=controls, dataset=ds, metadata=dataset_metadata)
        _active_plot["obj"] = plot
        return plot.panel()

    @pn.depends(browser.param.checked_items)
    def stats_panel(datasets):
        if not datasets:
            return pn.pane.Markdown("")
        ds = datasets[0]
        return ForecastStatsPanel(controls=controls, dataset_key=ds, metadata=dataset_metadata).panel()

    # Export Video — sweeps the shared time index across the full forecast
    # and encodes the current rendering (all model cards, plus any active
    # difference cards) to MP4 with ffmpeg.
    video_export = VideoExportPanel(controls, lambda: _active_plot["obj"])

    sidebar = pn.Column(
        pn.pane.HTML("<h2 style='margin: 5px 0; font-size: 14px; font-weight: bold;'>Datasets</h2>"),
        browser.panel,
        load_suite_dialog.open_button,
        load_suite_dialog.modal,
        controls.panel(),
        video_export.open_button,
        video_export.modal,
        pn.pane.HTML("<h2 style='margin: 5px 0; font-size: 14px; font-weight: bold;'>Metadata</h2>"),
        meta_panel.panel,
        width=250,
    )
    main = pn.Column(
        pn.panel(plot_grid, sizing_mode="stretch_width"),
        pn.panel(stats_panel, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
        css_classes=["main-content"],
    )
    vis = pn.Row(sidebar, main, sizing_mode="stretch_both", styles={"height": "100vh"})
    inference = pn.Column(
        inference_tab.panel(),
        sizing_mode="stretch_both",
        styles={"height": "100vh", "overflow": "auto"},
    )
    tabs = pn.Tabs(
        ("Visualization", vis),
        ("Inference", inference),
        stylesheets=["""
            .bk-tab { background: #f0f0f0; border-radius: 4px 4px 0 0; font-size: 14px; padding: 8px 16px; }
            .bk-tab.bk-active { background: white; border-top: 2px solid #007bff; font-weight: bold; }
            .bk-tabs-header { background: #e8e8e8; }
            .bk-tabs-content { border: 1px solid #ccc; padding: 10px; }
        """],
    )
    # `title` now only drives the browser tab text — the header title text is
    # supplied by the wordmark image below, so it is no longer set to "".
    template = pn.template.BootstrapTemplate(
        title="InferStudio",
        favicon=logo_uri("favicon.ico"),
        header_background="#091422",
        busy_indicator=None,
    )
    # InferStudio wordmark, replacing the plain-text HTML title pane. The
    # HSpacer takes over the layout job the old pane's stretch_width was
    # doing — pushing the spinner and NSF NCAR logo to the right edge.
    template.header.append(
        pn.pane.PNG(
            str(_LOGO_DIR / "wordmark_dark.png"),
            height=64,
            width=161,
            sizing_mode="fixed",
            margin=(5, 0, 5, 10),
        )
    )
    template.header.append(pn.layout.HSpacer())
    busy_spinner = pn.indicators.LoadingSpinner(
        value=False, width=20, height=20, color="light",
        margin=(10, 10, 10, 0),
    )
    # Documentation link
    template.header.append(
        pn.pane.HTML(
            '<a href="https://inferstudio.readthedocs.io/" target="_blank" '
            'rel="noopener" title="InferStudio documentation" '
            'style="display:inline-flex;align-items:center;gap:7px;'
            'color:#DFEFF6;font-size:14px;font-weight:500;'
            'text-decoration:none;white-space:nowrap;">'
            '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" '
            'stroke="currentColor" stroke-width="1.4" stroke-linecap="round" '
            'stroke-linejoin="round" aria-hidden="true">'
            '<path d="M2 2.5h4a2 2 0 0 1 2 2v9a1.5 1.5 0 0 0-1.5-1.5H2z"/>'
            '<path d="M14 2.5h-4a2 2 0 0 0-2 2v9a1.5 1.5 0 0 1 1.5-1.5H14z"/>'
            '</svg>Docs</a>',
            styles={"display": "flex", "align-items": "center", "height": "100%"},
            margin=(0, 40, 0, 0),
        )
    )
    pn.state.sync_busy(busy_spinner)
    template.header.append(busy_spinner)
    template.header.append(
        pn.pane.PNG(
            str(_STATIC_DIR / "nsf_ncar_logo_padded.png"),
            height=45,
            width=534,
            sizing_mode="fixed",
            margin=(5, 0, 5, 0),
        )
    )
    template.main[:] = [pn.Column(tabs, sizing_mode="stretch_both")]

    # Deferred via pn.state.onload rather than called directly here:
    # pn.state.notifications requires the browser session to be fully
    # connected before it can actually display anything client-side.
    # Calling .info(...) synchronously at this point in build_app() runs
    # before that connection is guaranteed to be live, so the message was
    # being silently dropped — this is why the welcome message never
    # appeared on initial launch, while the (unrelated) notification fired
    # from _on_new_output above worked fine, since by the time an
    # inference run completes the session has obviously been live for a
    # while already.
    def _show_welcome():
        if pn.state.notifications:
            pn.state.notifications.info(
                "Welcome to InferStudio.<br><br>"
                "You are currently viewing information from "
                "an example dataset. To run your own AI weather model inference, go "
                "to the Inference tab.<br><br> Then select your desired parameters, click "
                "\"Run Inference,\" and your simulation suite will be viewable from here."
                "<br><br><br>",
                duration=0,
            )

    pn.state.onload(_show_welcome)

    return template

# To drop jupyter in the future:
if __name__ == "__main__": build_app(DATA_DIR).servable()
