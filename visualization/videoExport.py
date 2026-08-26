"""
visualization/videoExport.py

Export the Visualization tab's current rendering as an MP4, sweeping the
shared time index from the first forecast step to the last.

Frames are produced by calling plot_e2s_field directly -- the same function
that backs the on-screen panes -- rather than by driving the live widgets, so
the export never fights the session for control of the plot.

The user-facing knob is "renderings per second": how fast the forecast
advances in wall-clock time. That is deliberately distinct from the MP4
container frame rate. ffmpeg reads the PNG sequence at the rendering rate
(-framerate) and duplicates frames up to the output rate (-r), so 2
renderings/s still yields a smooth 30 fps file rather than a 2 fps one that
some players stutter on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import panel as pn
import param
from PIL import Image, ImageDraw

from visualization.earth2StudioPlot import plot_e2s_field

DEFAULT_RENDERINGS_PER_SECOND = 2.0
MP4_FRAME_RATE = 30

RANGE_GLOBAL = "Global (scan all steps)"
RANGE_FIRST = "First step only (faster)"

_PAD = 10
_LABEL_H = 30
_HEADER_H = 34
_BG = (255, 255, 255)
_INK = (20, 20, 20)

_EXPORT_DIR = Path(
    f"/glade/derecho/scratch/{os.environ.get('USER', 'nobody')}/.inferstudio_exports"
)

# plot_e2s_field renders through matplotlib's pyplot state machine, which is
# not thread safe. The export worker must not race the live session's own
# renders, or frames come back garbled/blank.
_RENDER_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _ffmpeg_exe() -> str:
    """Locate a working ffmpeg.

    Prefers imageio-ffmpeg's statically-linked binary, which carries its own
    libx264 and so is immune to the conda env's shared-library state -- the
    conda-forge ffmpeg in creditJun3 is a stale 2.8.6 build that fails at
    load time looking for libx264.so.138 against an installed .so.164.
    Whichever candidate is chosen is verified by actually running it, since
    a path from shutil.which proves only that a file exists, not that it
    can link.
    """
    candidates = []
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    on_path = shutil.which("ffmpeg")
    if on_path:
        candidates.append(on_path)

    problems = []
    for exe in candidates:
        try:
            proc = subprocess.run([exe, "-version"], capture_output=True,
                                  text=True, timeout=20)
            if proc.returncode == 0:
                return exe
            first = (proc.stderr.strip().splitlines() or ["non-zero exit"])[0]
            problems.append(f"{exe}: {first}")
        except Exception as e:
            problems.append(f"{exe}: {e}")

    detail = ("\n" + "\n".join(problems)) if problems else ""
    raise RuntimeError(
        "No working ffmpeg found. Install the self-contained build with "
        "`pip install imageio-ffmpeg` in the active env." + detail
    )


def _ui(fn, *args, **kwargs):
    """Apply a widget mutation on the Bokeh document thread.

    Worker threads must not touch widget state directly; pn.state.execute
    defers the call onto the session's event loop. Falls back to a direct
    call when there is no live session (headless testing).
    """
    try:
        pn.state.execute(lambda: fn(*args, **kwargs))
    except Exception:
        fn(*args, **kwargs)


@lru_cache(maxsize=8)
def _font(size):
    """A real TrueType font at `size`, falling back to PIL's bitmap default.

    PIL's load_default() ignores size entirely, so labels would come out
    tiny on a 1200px-wide composite; matplotlib always ships DejaVu Sans,
    so findfont is a reliable source for a scalable face.
    """
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


class _Tile:
    """One panel of the composite frame, with a color range fixed for the
    whole sweep.

    Mirrors one card of the on-screen grid: either a model's field or a
    model-minus-model difference. The interactive plot re-derives its range
    on every step, which is fine when scrubbing but makes a video's colorbar
    breathe and its field flicker, so the range is resolved once up front
    (see scan/finalize) and then held.
    """

    def __init__(self, label, model_dir, is_diff, user_vmin=None, user_vmax=None):
        self.label = label
        self.model_dir = model_dir
        self.is_diff = is_diff
        # A range the user typed into Colormap Min/Max, or None for auto.
        # Never applies to difference tiles -- those have their own
        # symmetric-about-zero scale, exactly as _render_diff does on screen.
        self.user_vmin = None if is_diff else user_vmin
        self.user_vmax = None if is_diff else user_vmax
        self.vmin = None
        self.vmax = None
        self._scan_lo = None
        self._scan_hi = None
        self.size = None      # fixed at the first rendered frame; libx264
                              # requires constant dimensions

    @property
    def needs_scan(self):
        """False only when the user pinned both ends of a non-diff range."""
        return self.user_vmin is None or self.user_vmax is None

    def _cmap_for(self, cmap):
        return "coolwarm" if self.is_diff else cmap

    def scan(self, var_name, level, t, cmap):
        """Record this step's data extrema without keeping the image.

        plot_e2s_field reports the range it actually used, so an auto-ranged
        call doubles as a probe. There is no cheaper path to the extrema
        through the current plotting interface.
        """
        _, lo, hi = plot_e2s_field(
            model_dir=self.model_dir, base_or_var=var_name, level=level, t=t,
            cmap=self._cmap_for(cmap), vmin=None, vmax=None,
        )
        self._scan_lo = lo if self._scan_lo is None else min(self._scan_lo, lo)
        self._scan_hi = hi if self._scan_hi is None else max(self._scan_hi, hi)

    def finalize(self):
        """Resolve the fixed range from whatever scanning found."""
        if self.is_diff:
            # Symmetric about zero so coolwarm's neutral midpoint lands on
            # zero difference -- same reasoning as DatasetPlot2._render_diff.
            #
            # Scanning the FULL series matters most here. At t=0 both models
            # start from the same analysis, so the difference is nothing but
            # floating-point noise (~1e-10). Locking that as the range makes
            # every later step -- where genuine divergence is many orders of
            # magnitude larger -- saturate to solid red and blue.
            lo = self._scan_lo if self._scan_lo is not None else 0.0
            hi = self._scan_hi if self._scan_hi is not None else 0.0
            max_abs = max(abs(lo), abs(hi))
            if max_abs == 0.0:
                max_abs = 1e-12    # a degenerate all-zero field: avoid vmin==vmax
            self.vmin, self.vmax = -max_abs, max_abs
        else:
            self.vmin = self.user_vmin if self.user_vmin is not None else self._scan_lo
            self.vmax = self.user_vmax if self.user_vmax is not None else self._scan_hi
            if self.vmin is not None and self.vmin == self.vmax:
                self.vmin, self.vmax = self.vmin - 1e-12, self.vmax + 1e-12

    def render(self, var_name, level, t, cmap):
        """Return a PIL image of this panel at time step t."""
        buf, _, _ = plot_e2s_field(
            model_dir=self.model_dir, base_or_var=var_name, level=level, t=t,
            cmap=self._cmap_for(cmap), vmin=self.vmin, vmax=self.vmax,
        )
        img = Image.open(buf).convert("RGB")
        if self.size is None:
            self.size = img.size
        elif img.size != self.size:
            # Colorbar tick labels can change width between steps (e.g. "5"
            # vs "-12.5"), nudging the figure's pixel dimensions. H.264
            # requires every frame to be identical in size.
            img = img.resize(self.size, Image.LANCZOS)
        return img


def _compose(rows, header):
    """Tile rendered panels into one frame, mirroring the on-screen grid.

    `rows` is a list of rows, each a list of (label, PIL.Image) -- one row
    per model, with its difference card alongside when active.
    """
    row_dims = []
    for row in rows:
        w = sum(im.width for _, im in row) + _PAD * (len(row) + 1)
        h = max(im.height for _, im in row) + _LABEL_H + _PAD
        row_dims.append((w, h))

    width = max(w for w, _ in row_dims)
    height = _HEADER_H + sum(h for _, h in row_dims) + _PAD
    # libx264 with yuv420p chroma subsampling requires even dimensions.
    width += width % 2
    height += height % 2

    canvas = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((_PAD, 8), header, fill=_INK, font=_font(17))

    y = _HEADER_H
    for row, (row_w, row_h) in zip(rows, row_dims):
        x = (width - row_w) // 2 + _PAD
        for label, im in row:
            draw.text((x + im.width // 2, y + 6), label,
                      fill=_INK, font=_font(19), anchor="ma")
            canvas.paste(im, (x, y + _LABEL_H))
            x += im.width + _PAD
        y += row_h
    return canvas


# --------------------------------------------------------------------------
# modal
# --------------------------------------------------------------------------

class VideoExportPanel(param.Parameterized):
    """Export Video button + options modal.

    Follows the LoadSuiteDialog convention: exposes `.open_button` and
    `.modal`, both of which must be appended to the sidebar Column.
    """

    renderings_per_second = param.Number(
        default=DEFAULT_RENDERINGS_PER_SECOND,
        bounds=(0.25, 24.0),
        step=0.25,
        label="Renderings per second",
        doc="How many forecast time steps are shown per second of playback.",
    )

    color_range = param.Selector(
        default=RANGE_GLOBAL,
        objects=[RANGE_GLOBAL, RANGE_FIRST],
        label="Color range",
    )

    def __init__(self, controls, active_plot_fn, output_dir=_EXPORT_DIR, **params):
        super().__init__(**params)
        self.controls = controls
        self._active_plot_fn = active_plot_fn
        self._output_dir = Path(output_dir)
        self._cancel = threading.Event()
        self._thread = None

        # Sized/margined to match load_suite_dialog.open_button exactly, so
        # both stretch to the same effective width inside the 250px sidebar.
        self.open_button = pn.widgets.Button(
            name="Export Video",
            button_type="primary",
            sizing_mode="stretch_width",
            margin=(10, 10, 0, 0),
        )
        self.open_button.on_click(self._open)

        self._rps = pn.widgets.FloatSlider.from_param(
            self.param.renderings_per_second, sizing_mode="stretch_width",
        )
        self._range = pn.widgets.RadioBoxGroup.from_param(
            self.param.color_range, inline=False,
        )
        self._range_help = pn.pane.Markdown(
            "_Global scans every step first so the colorbar stays fixed and "
            "nothing clips. Difference panels need this: at the initial time "
            "both models share the same analysis, so their difference is "
            "numerical noise._",
            styles={"font-size": "12px"},
            sizing_mode="stretch_width",
        )
        self._summary = pn.pane.Markdown("", sizing_mode="stretch_width")
        self._progress = pn.indicators.Progress(
            value=0, max=100, sizing_mode="stretch_width", visible=False,
        )
        self._status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self._download = pn.widgets.FileDownload(
            label="Download MP4", button_type="success",
            visible=False, sizing_mode="stretch_width",
        )
        self._export_btn = pn.widgets.Button(name="Export", button_type="primary", width=110)
        self._cancel_btn = pn.widgets.Button(
            name="Cancel", button_type="light", width=110, disabled=True,
        )
        self._export_btn.on_click(self._start)
        self._cancel_btn.on_click(self._request_cancel)

        self.modal = pn.Modal(
            pn.Column(
                pn.pane.Markdown("### Export Video"),
                pn.pane.Markdown(
                    "Renders the current visualization across every forecast "
                    "time step and encodes it to MP4."
                ),
                pn.layout.Divider(),
                self._rps,
                pn.pane.Markdown("**Color range**", margin=(10, 0, 0, 0)),
                self._range,
                self._range_help,
                self._summary,
                pn.layout.Divider(),
                pn.Row(self._export_btn, self._cancel_btn),
                self._progress,
                self._status,
                self._download,
                width=460,
                sizing_mode="stretch_width",
            ),
            name="export_video_modal",
            show_close_button=True,
            background_close=False,
        )

    # -- state -------------------------------------------------------------

    def _nframes(self):
        """Forecast step count, taken from the shared time slider's range.

        SharedPlotControls.update_choices already resolved the
        lead_time-vs-time distinction and clamped to the longest checked
        dataset, so there is no need to re-derive it from the metadata.
        """
        return int(self.controls.time_slider.end) + 1

    def _rows(self):
        """What the Visualization tab is currently rendering, or None."""
        plot = self._active_plot_fn()
        if plot is None or not self.controls.var_name:
            return None
        try:
            rows = plot.export_panels()
        except Exception:
            return None
        return rows or None

    # -- callbacks ---------------------------------------------------------

    def _open(self, _event=None):
        self._download.visible = False
        self._status.object = ""
        self._progress.visible = False
        self._refresh_summary()
        self.modal.open = True

    @param.depends("renderings_per_second", "color_range", watch=True)
    def _refresh_summary(self):
        rows = self._rows()
        if rows is None or self.controls.time_slider.disabled:
            self._summary.object = "_Nothing rendered to export._"
            self._export_btn.disabled = True
            return
        n = self._nframes()
        npanels = sum(len(r) for r in rows)
        extra = " Scanning roughly doubles render time." \
            if self.color_range == RANGE_GLOBAL else ""
        self._summary.object = (
            f"**{n}** time steps \u00d7 **{npanels}** panel(s) \u2192 "
            f"**{n / self.renderings_per_second:.1f} s** of video "
            f"at {MP4_FRAME_RATE} fps.{extra}"
        )
        self._export_btn.disabled = False

    def _request_cancel(self, _event=None):
        self._cancel.set()
        self._status.object = "_Cancelling\u2026_"

    def _start(self, _event=None):
        if self._thread and self._thread.is_alive():
            return
        rows = self._rows()
        if rows is None:
            self._status.object = "**Nothing to export** \u2014 render something first."
            return

        # Snapshot the controls now. The user is free to keep scrubbing the
        # live session while the export runs in the background; the video
        # reflects the state at the moment Export was pressed, not whatever
        # the widgets drift to mid-encode.
        snapshot = dict(
            var_name=self.controls.var_name,
            level=self.controls.level_value,
            cmap=self.controls._colormaps.get(self.controls.colormap, "viridis"),
            vmin=self.controls.cmap_min,
            vmax=self.controls.cmap_max,
            nframes=self._nframes(),
            rps=self.renderings_per_second,
            scan_all=(self.color_range == RANGE_GLOBAL),
        )

        self._cancel.clear()
        self._export_btn.disabled = True
        self._cancel_btn.disabled = False
        self._download.visible = False
        self._progress.value = 0
        self._progress.visible = True
        self._status.object = "_Starting\u2026_"
        self._thread = threading.Thread(
            target=self._worker, args=(rows, snapshot),
            daemon=True, name="video-export",
        )
        self._thread.start()

    # -- worker ------------------------------------------------------------

    def _worker(self, rows, snap):
        try:
            exe = _ffmpeg_exe()

            tiles = [[_Tile(label, path, is_diff,
                            user_vmin=snap["vmin"], user_vmax=snap["vmax"])
                      for label, path, is_diff in row]
                     for row in rows]
            flat = [t for row in tiles for t in row]

            total = snap["nframes"]
            var, level, cmap = snap["var_name"], snap["level"], snap["cmap"]

            # --- pass 1: resolve color ranges --------------------------------
            scan_steps = range(total) if snap["scan_all"] else [0]
            scanning = [t for t in flat if t.needs_scan]
            if scanning:
                nscan = len(list(scan_steps))
                for i, t in enumerate(scan_steps):
                    if self._cancel.is_set():
                        _ui(setattr, self._status, "object", "**Export cancelled.**")
                        return
                    with _RENDER_LOCK:
                        for tile in scanning:
                            tile.scan(var, level, t, cmap)
                    _ui(setattr, self._progress, "value", int(45 * (i + 1) / nscan))
                    _ui(setattr, self._status, "object",
                        f"_Scanning color range {i + 1}/{nscan}_")
            for tile in flat:
                tile.finalize()

            # --- pass 2: render frames ---------------------------------------
            self._output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            out_path = self._output_dir / f"inferstudio_{var}_{stamp}.mp4"
            level_txt = f" @ {level} hPa" if level else ""

            with tempfile.TemporaryDirectory(prefix="inferstudio_frames_") as tmp:
                tmpdir = Path(tmp)
                for t in range(total):
                    if self._cancel.is_set():
                        _ui(setattr, self._status, "object", "**Export cancelled.**")
                        return

                    with _RENDER_LOCK:
                        composed = [
                            [(tile.label, tile.render(var, level, t, cmap))
                             for tile in row]
                            for row in tiles
                        ]

                    header = f"{var}{level_txt}   \u2014   step {t + 1} of {total}"
                    _compose(composed, header).save(tmpdir / f"frame_{t:05d}.png")

                    _ui(setattr, self._progress, "value",
                        45 + int(45 * (t + 1) / total))
                    _ui(setattr, self._status, "object", f"_Rendered {t + 1}/{total}_")

                # --- pass 3: encode -------------------------------------------
                _ui(setattr, self._status, "object", "_Encoding with ffmpeg\u2026_")
                cmd = [
                    exe, "-y",
                    "-framerate", f"{snap['rps']:g}",   # input rate = renderings/s
                    "-i", str(tmpdir / "frame_%05d.png"),
                    "-vf", "format=yuv420p",            # broad player compatibility
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-r", str(MP4_FRAME_RATE),          # output container rate
                    "-movflags", "+faststart",
                    str(out_path),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
                    raise RuntimeError(f"ffmpeg failed:\n```\n{tail}\n```")

            mb = out_path.stat().st_size / 1e6
            _ui(setattr, self._progress, "value", 100)
            _ui(setattr, self._status, "object",
                f"**Done** \u2014 {total} renderings, {mb:.1f} MB\n\n`{out_path}`")
            _ui(self._download.param.update,
                file=str(out_path), filename=out_path.name, visible=True)

        except Exception as exc:
            _ui(setattr, self._status, "object", f"**Export failed:** {exc}")
        finally:
            _ui(setattr, self._export_btn, "disabled", False)
            _ui(setattr, self._cancel_btn, "disabled", True)
