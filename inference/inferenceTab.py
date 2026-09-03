import panel as pn
import param
import subprocess
import threading
import os
import time
import signal

from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

from inference.outputParams import OutputParams
from inference.timePicker import TimePicker
from inference.commandRunner import CommandRunner

from inference.milesCreditRunner import MilesCreditRunner
from inference.earth2StudioRunner import Earth2StudioRunner

MILES_CREDIT_MODEL_LIST = ['WXFormer']
MILES_CREDIT_MODELS = frozenset(MILES_CREDIT_MODEL_LIST)
EARTH2STUDIO_MODEL_LIST = ['AIFS', 'Aurora', 'Pangu', 'FourCastNet3']
EARTH2STUDIO_MODELS = frozenset(EARTH2STUDIO_MODEL_LIST)
#EARTH2STUDIO_MODELS = {'AIFS', 'Aurora', 'Pangu', 'FourCastNet3', 'GraphCast', 'SFNO'}

# Short, hover-friendly descriptions shown as a tooltip on each model button.
MODEL_DESCRIPTIONS = {
    'WXFormer': (
        "NCAR MILES-CREDIT's own transformer-based weather model. Writes "
        "ERA5-style output with a real vertical-level dimension."
    ),
    'AIFS': (
        "ECMWF's AI Forecasting System, run through Earth2Studio (PyTorch). "
        "Requires a source-compiled flash_attn."
    ),
    'Aurora': (
        "Microsoft's Aurora foundation model for weather forecasting, run "
        "through Earth2Studio (PyTorch)."
    ),
    'Pangu': (
        "Huawei's Pangu-Weather model, run through Earth2Studio on "
        "GPU-enabled ONNX Runtime. Fast, and a reasonable baseline."
    ),
    'FourCastNet3': (
        "NVIDIA's FourCastNet 3, a spherical Fourier Neural Operator model, "
        "run through Earth2Studio (PyTorch + makani)."
    ),
}


class ModelPicker(param.Parameterized):
    """Multi-select button group where each button shows a hover tooltip
    describing the model it selects."""

    value = param.List(default=[])

    def __init__(self, options, descriptions=None, **params):
        super().__init__(**params)
        descriptions = descriptions or {}
        self._buttons = {}
        for name in options:
            button = pn.widgets.Button(
                name=name,
                button_type='primary',
                button_style='solid' if name in self.value else 'outline',
                description=descriptions.get(name, ''),
                margin=(0, 5, 5, 0),
            )
            button.on_click(partial(self._on_click, name))
            self._buttons[name] = button
        self.row = pn.Row(*self._buttons.values(), margin=(0, 5, 5, 0))

    def _on_click(self, name, event):
        selected = set(self.value)
        if name in selected:
            selected.discard(name)
            self._buttons[name].button_style = 'outline'
        else:
            selected.add(name)
            self._buttons[name].button_style = 'solid'
        # Preserve the original option ordering rather than click order.
        self.value = [m for m in self._buttons if m in selected]

    def panel(self):
        return self.row


class InferenceTab(param.Parameterized):
    startDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0))
    endDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=72))
    outputDirectory = param.String(default="")

    def __init__(self, **params):
        super().__init__(**params)

        self.modelPicker = ModelPicker(
            value=['AIFS', 'Aurora'],
            options=MILES_CREDIT_MODEL_LIST + EARTH2STUDIO_MODEL_LIST,
            descriptions=MODEL_DESCRIPTIONS,
        )

        self.outputParams = OutputParams(start_path=Path(f"/glade/derecho/scratch/{os.environ['USER']}"))
        self.outputParams.simulationNamePicker.param.watch(
            lambda e: self._clear_highlight(self.outputParams.simulationNamePicker),
            'value_input'
        )
        self.modelPicker.param.watch(self._update_default_sim_name, 'value')
        self._update_default_sim_name()

        self.timePicker = TimePicker()

        self.inferenceButton = pn.widgets.Button(
            name="Run Inference",
            button_type="success",
            button_style="outline"
        )
        self.inferenceButton.on_click(self._on_run_click)

        self.cancelButton = pn.widgets.Button(
            name="Cancel",
            button_type="danger",
            disabled=True,
        )
        self.cancelButton.on_click(self._on_cancel_click)

        self.spinner = pn.indicators.LoadingSpinner(
            width=30, height=30, value=False, color="primary", visible=False
        )

        self.elapsedLabel = pn.widgets.StaticText(name="Elapsed", value="N/A")
        self.completionLabel = pn.widgets.StaticText(name="Completed at", value="N/A")
        self.completionPathLabel = pn.widgets.StaticText(name="", value="")

        self.commandRunner = CommandRunner()

        # Per-model state; populated fresh on each Run click
        self._processes = {}        # model -> Popen
        self._log_widgets = {}      # model -> TextAreaInput
        self._spinners = {}         # model -> LoadingSpinner
        self._status_widgets = {}   # model -> HTML pane
        self._timer_running = False
        self._active_count = 0
        self._active_lock = threading.Lock()
        self._cancel_event = threading.Event()

        self.outputTabs = pn.Tabs(sizing_mode="stretch_width", height=350)
        self.statusRow = pn.Row(pn.pane.Markdown("", margin=0))

    # ------------------------------------------------------------------ #
    #  Runner helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_runners(self):
        """Return list of (model_name, runner) for every selected model."""
        runners = []
        for model in self.modelPicker.value:
            if model in MILES_CREDIT_MODELS:
                runners.append((model, MilesCreditRunner()))
            elif model in EARTH2STUDIO_MODELS:
                runners.append((model, Earth2StudioRunner()))
        return runners

    def _build_config(self, model: str) -> dict:
        return {
            "simulation_name": self.outputParams.simulationNamePicker.value_input,
            "start_time":      self.timePicker.startDatePicker.value,
            "end_time":        self.timePicker.end_date,
            "timestep":        self.timePicker.incrementButtons.value,
            "ua_vars":         self.outputParams.UAVars.value,
            "surface_vars":    self.outputParams.surfaceVars.value,
            "output_path":     str(
                Path(self.outputParams.pathDisplay.value)
                / self.outputParams.simulationNamePicker.value_input
            ),
            "output_dir":      self.outputParams.current_path_val,
            "model":           model,
        }

    # ------------------------------------------------------------------ #
    #  UI event handlers                                                   #
    # ------------------------------------------------------------------ #

    def _on_run_click(self, event):
        # Fresh debug log each run
        try:
            open('/tmp/debug.log', 'w').close()
        except Exception:
            pass

        runners = self._get_runners()

        if not runners:
            self._set_single_log("Error: No recognized model selected.")
            return

        self._clear_highlight(self.outputParams.simulationNamePicker)

        # Pre-validate all runners before touching the UI
        for model, runner in runners:
            config = self._build_config(model)
            error = runner.validate(config)
            if error:
                self._set_single_log(f"[{model}] {error}")
                if not config["simulation_name"]:
                    self._highlight_error(self.outputParams.simulationNamePicker)
                return

        # Build tabs, spinners, and status indicators — one per model
        self._log_widgets = {}
        self._spinners = {}
        self._status_widgets = {}
        self._processes = {}
        self.outputTabs.objects = []
        self.statusRow.objects = []

        for model, _ in runners:
            widget = pn.widgets.TextAreaInput(
                name=model,
                value="",
                sizing_mode="stretch_width",
                height=300,
            )
            spinner = pn.indicators.LoadingSpinner(
                width=25, height=25, value=True, color="primary", visible=True
            )
            pane = pn.pane.HTML(self._status_html(model, "pending"), width=100)
            self._log_widgets[model] = widget
            self._spinners[model] = spinner
            self._status_widgets[model] = pane
            self.statusRow.append(pane)
            self.outputTabs.append((model, pn.Column(spinner, widget, sizing_mode="stretch_width", height=330)))

        self.spinner.value = True
        self.spinner.visible = True
        self.inferenceButton.disabled = True
        self.cancelButton.disabled = False
        self.elapsedLabel.value = ""
        self.completionLabel.value = ""
        self.completionPathLabel.value = ""
        self._cancel_event.clear()

        # Shared elapsed timer
        overall_start = time.time()
        self._doc = pn.state.curdoc

        def _tick_update():
            elapsed = time.time() - overall_start
            mins, secs = divmod(int(elapsed), 60)
            self.elapsedLabel.value = f"{mins}m {secs}s"

        self._periodic_cb = pn.state.add_periodic_callback(_tick_update, period=1000)

        with self._active_lock:
            self._active_count = len(runners)

        def _all_done():
            with open('/tmp/debug.log', 'a') as f:
                f.write("_all_done called\n")
            elapsed = time.time() - overall_start
            mins, secs = divmod(int(elapsed), 60)

            sim_dir = Path(self.outputParams.pathDisplay.value) / self.outputParams.simulationNamePicker.value_input

            def _finish():
                try:
                    with open('/tmp/debug.log', 'a') as f:
                        f.write("_finish running on doc thread\n")
                    if self._periodic_cb is not None:
                        self._periodic_cb.stop()
                    self.elapsedLabel.value = f"{mins}m {secs}s"
                    self.completionLabel.value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.completionPathLabel.value = f"Files written to {sim_dir}"
                    self.outputDirectory = str(sim_dir)
                    self.param.trigger('outputDirectory')
                    self.spinner.value = False
                    self.spinner.visible = False
                    self.inferenceButton.disabled = False
                    self.cancelButton.disabled = True
                except Exception as e:
                    with open('/tmp/debug.log', 'a') as f:
                        f.write(f"_finish died: {e!r}\n")

            if self._doc is not None:
                self._doc.add_next_tick_callback(_finish)
            else:
                _finish()

        def _run_all_sequential():
            try:
                for model, runner in runners:
                    if self._cancel_event.is_set():
                        self._append_log(model, "Skipped (cancelled by user).\n")

                        def _mark_skipped(model=model):
                            pane = self._status_widgets.get(model)
                            if pane is not None:
                                pane.object = self._status_html(model, "cancelled")
                            spinner = self._spinners.get(model)
                            if spinner is not None:
                                spinner.value = False
                                spinner.visible = False

                        self._ui(_mark_skipped)
                        continue
                    self._run_model(model, runner, lambda: None)
                _all_done()
            except Exception:
                import traceback
                tb = traceback.format_exc()

                def _report():
                    for model in self._log_widgets:
                        self._append_log(model, f"\n\n[INTERNAL ERROR]\n{tb}\n")
                    self.spinner.value = False
                    self.spinner.visible = False
                    self.inferenceButton.disabled = False
                    self.cancelButton.disabled = True

                pn.state.execute(_report)

        threading.Thread(target=_run_all_sequential, daemon=True).start()

    def _on_cancel_click(self, event):
        self._cancel_event.set()
        for model, proc in list(self._processes.items()):
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
                self._append_log(model, "\n\nCancelled by user.")
        self.cancelButton.disabled = True

    # ------------------------------------------------------------------ #
    #  Per-model execution                                                 #
    # ------------------------------------------------------------------ #

    def _ui(self, fn):
        """Schedule a UI-mutating callable safely on the document's own thread."""
        if self._doc is not None:
            self._doc.add_next_tick_callback(fn)
        else:
            fn()

    def _append_log(self, model: str, text: str):
        widget = self._log_widgets.get(model)
        if widget is None:
            return
        def _do_append():
            widget.value += text
        if self._doc is not None:
            self._doc.add_next_tick_callback(_do_append)
        else:
            _do_append()

    def _run_model(self, model: str, runner, on_done):
        """Prepare, build, and execute a single model in its own thread."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        def _update_start():
            spinner = self._spinners.get(model)
            if spinner is not None:
                spinner.value = True
                spinner.visible = True
            pane = self._status_widgets.get(model)
            if pane is not None:
                pane.object = self._status_html(model, "running")
            self.outputTabs.active = list(self._log_widgets.keys()).index(model)

        self._ui(_update_start)

        config = self._build_config(model)
        model_output = Path(config["output_path"]) / model
        model_output.mkdir(parents=True, exist_ok=True)
        config["output_path"] = str(model_output)

        proc = None
        try:
            prepared = runner.prepare(config)
            config.update(prepared)
            cmd = runner.build_cmd(config)
        except Exception as e:
            self._append_log(model, f"Error during setup: {e}\n")
            def _update_setup_error():
                pane = self._status_widgets.get(model)
                if pane is not None:
                    pane.object = self._status_html(model, "error")
                spinner = self._spinners.get(model)
                if spinner is not None:
                    spinner.value = False
                    spinner.visible = False
            self._ui(_update_setup_error)
            on_done()
            return

        try:
            proc = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, start_new_session=True,
            )
            self._processes[model] = proc

            # --- batched log flushing ---
            buf = []
            buf_lock = threading.Lock()
            last_flush = time.time()
            FLUSH_INTERVAL = 0.3  # seconds

            def _flush(force=False):
                nonlocal last_flush
                now = time.time()
                if not force and (now - last_flush) < FLUSH_INTERVAL:
                    return
                with buf_lock:
                    if not buf:
                        return
                    text = "".join(buf)
                    buf.clear()
                last_flush = now
                self._append_log(model, text)

            for line in proc.stdout:
                with buf_lock:
                    buf.append(line)
                _flush()

            _flush(force=True)  # catch any remainder
            proc.wait()

            if proc.returncode != 0:
                self._append_log(model, f"\nExited with code {proc.returncode}\n")
            else:
                self._append_log(model, "\nDone.\n")

        except Exception as e:
            self._append_log(model, f"Error: {e}\n")
        finally:
            self._processes.pop(model, None)
            try:
                state = "done" if proc.returncode == 0 else "error"
            except Exception:
                state = "error"

            def _update_finish(state=state):
                spinner = self._spinners.get(model)
                if spinner is not None:
                    spinner.value = False
                    spinner.visible = False
                pane = self._status_widgets.get(model)
                if pane is not None:
                    pane.object = self._status_html(model, state)

            self._ui(_update_finish)
            on_done()


    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _set_single_log(self, message: str):
        """Show a plain error message before tabs have been built."""
        widget = pn.widgets.TextAreaInput(
            name="Log", value=message, sizing_mode="stretch_both"
        )
        self.outputTabs.objects = []
        self.outputTabs.append(("Log", widget))

    def _status_html(self, model: str, state: str) -> str:
        symbol = {"pending": "◷", "running": "⟳", "done": "✓", "error": "✗", "cancelled": "⊘"}[state]
        color  = {"pending": "#cccccc", "running": "#888888", "done": "#2ecc71", "error": "#e74c3c", "cancelled": "#999999"}[state]
        return (
            f'<div style="text-align:center;line-height:1.4">'
            f'<span style="font-size:11px;color:#555">{model}</span><br>'
            f'<span style="font-size:20px;color:{color}">{symbol}</span>'
            f'</div>'
        )

    def _update_default_sim_name(self, event=None):
        self.outputParams.set_default_simulation_name(self.modelPicker.value)

    def _highlight_error(self, widget):
        styles = dict(widget.styles or {})
        styles.update({'border': '2px solid #e74c3c', 'border-radius': '4px'})
        widget.styles = styles

    def _clear_highlight(self, widget):
        styles = dict(widget.styles or {})
        styles.pop('border', None)
        styles.pop('border-radius', None)
        widget.styles = styles

    # ------------------------------------------------------------------ #
    #  Layout                                                              #
    # ------------------------------------------------------------------ #

    def panel(self):
        return pn.Column(
            pn.WidgetBox(
                '# AI Model',
                self.modelPicker.panel(),
                sizing_mode='stretch_width',
            ),
            self.outputParams.panel(),
            self.timePicker.panel,
            pn.WidgetBox(
                "# Launcher",
                pn.Row(
                    self.inferenceButton, self.cancelButton,
                    self.spinner,
                    pn.Column(self.elapsedLabel, self.completionLabel, self.completionPathLabel)
                ),
                self.statusRow,
                self.outputTabs,
                sizing_mode='stretch_width',
            ),
            sizing_mode='stretch_both',
            min_height=300,
        )
