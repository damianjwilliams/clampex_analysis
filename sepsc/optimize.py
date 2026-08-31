"""
Interactive parameter-optimization tool for the classical Mini-Analysis-
style detector (minianalysis.py) -- the Mini Analysis Program tutorial's
own calibration workflow (this_method.pdf p.4: "Optimizing Detection
Parameters" -- "First use mouse-click to detect. Examine the location of
the X's and dots. Examine the amplitude and area."), as a live three-panel
window instead of a one-shot batch run:

    - Left panel: every detection parameter (PDF p.2), editable, read fresh
      on each click -- no Apply button needed.
    - Top-right panel: the full trace, pan/zoom in both X and Y
      (pyqtgraph, so a multi-million-sample recording stays responsive).
    - Bottom-right panel: click any peak in the trace above and it's
      measured HERE, right now, with whatever parameter values are
      currently in the left panel -- baseline/amplitude/onset/decay/area,
      annotated exactly like sepsc.inspect's detail view, including WHY a
      candidate would be rejected (which threshold/search-window it failed)
      if it would be. Edit a parameter and click the same peak again to see
      the new measurement immediately.

A "Run full detection with these parameters" button commits the current
left-panel values to an actual sepsc.minianalysis run (same detect_events
+ events_frame + CSV/params-sidecar output as `python -m sepsc
minianalysis`), once you're happy with what single-click testing shows.

Deliberately simpler than a full detect_events scan for the single-click
measurement itself: baseline is ALWAYS the plain (d)/(e) window average
(see evaluate_candidate), never the overlap-baseline extrapolation
(minianalysis.DetectionParams.adjust_overlapping_baseline) -- that needs
sequential state from scanning the WHOLE trace in order (which candidate
came before this one, and its fitted decay), which an isolated ad-hoc
click has no way to know. "Run full detection" DOES apply it if enabled,
so the two can disagree slightly on events immediately following another
close one -- by design, not a bug: single-click testing is for dialing in
thresholds/windows quickly, not for reproducing every event's exact final
number.

Usage
-----
    python -m sepsc.optimize path\\to\\recording.abf
    python -m sepsc.optimize recording.abf --amplitude-threshold 8  # initial value, still editable
    python -m sepsc.optimize recording.abf --filter  # load the same 3kHz/10kHz filtered trace
        # minianalysis.py's --filter produces (see sepsc.preprocess) -- fixed for this session,
        # since filtering is a trace-preprocessing step, not something to redo on every click
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyabf
from scipy.signal import find_peaks

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from .gui_utils import make_field_widget, read_field_widget, safe_callback
from .inspect import (
    COLOR_AMPLITUDE, COLOR_AREA_FAIL, COLOR_AREA_PASS, COLOR_BASELINE_LINE, COLOR_BASELINE_WINDOW,
    COLOR_BEFORE_PEAK, COLOR_DECAY_LEVEL, COLOR_DECAY_WINDOW, COLOR_LOCAL_MAX, COLOR_ONSET,
    COLOR_ONSET_WINDOW, COLOR_PEAK, _LABEL_BBOX, _load_params,
)
from .minianalysis import (
    DetectionParams, _PARAM_FIELD_SPECS, _samples,
    detect_events, events_frame,
)
from .preprocess import DEFAULT_CUTOFF_HZ, DEFAULT_ORDER, DEFAULT_TARGET_RATE_HZ, bessel_lowpass, downsample, get_hardware_filter_hz
from .style import GRID, INK, MUTED, SURFACE, TRACE

__all__ = ["CandidateResult", "evaluate_candidate", "plot_candidate_detail", "OptimizerWindow"]

# Hard cap on how many samples the trace panel ever hands to pyqtgraph at
# once, manually re-sliced/decimated to the current view on every pan/zoom.
# NOT pyqtgraph's own autoDownsample/clipToView: on this project's pinned
# pyqtgraph version, an internal AttributeError (view.autoRangeEnabled(),
# printed to stderr -- see sepsc.view's own comment about it) fires on
# EVERY view-range change, not just once at startup, which silently stops
# that feature from ever updating the curve again -- the trace visually
# freezes in its initial view no matter how you pan or zoom. Redrawing by
# hand here sidesteps that codepath entirely.
DISPLAY_MAX_POINTS = 200_000


@dataclass
class CandidateResult:
    """Everything about one ad-hoc, single-click detection test -- unlike
    minianalysis.Event, this exists whether or not the candidate would
    actually be ACCEPTED, so a rejected click can still be shown and
    explained (which is the whole point of this tool)."""
    peak_idx: int
    peak_time_s: float
    accepted: bool
    reasons: list = field(default_factory=list)  # human-readable failure reasons, empty if accepted
    baseline: Optional[float] = None
    peak_v: Optional[float] = None
    amplitude: Optional[float] = None
    onset_idx: Optional[int] = None
    decay_idx: Optional[int] = None
    rise_time_ms: Optional[float] = None
    decay_time_ms: Optional[float] = None
    area: Optional[float] = None
    # window geometry, for plot_candidate_detail -- mirrors sepsc.inspect._event_geometry's dict
    b0: int = 0
    b1: int = 0
    p0: int = 0
    p1: int = 0
    local_max_lo: int = 0
    local_max_hi: int = 0
    decay_search_end: int = 0
    onset_search_start: int = 0


def evaluate_candidate(v: np.ndarray, dt: float, click_idx: int, params: DetectionParams) -> CandidateResult:
    """Snap `click_idx` to the nearest local maximum, then run the same
    per-candidate math as minianalysis.detect_events' steps 2-6 (baseline,
    amplitude, onset, decay, area), collecting EVERY failure reason instead
    of stopping at the first one -- so a single click shows the complete
    picture, not just whichever check happens to fail first.
    """
    pol = -1.0 if params.direction == "negative" else 1.0
    sv = pol * v

    search_n = _samples(params.search_local_max_ms, dt)
    before_n = _samples(params.baseline_before_ms, dt)
    avg_n = _samples(params.baseline_avg_ms, dt)
    decay_n = _samples(params.decay_search_ms, dt)
    onset_search_n = _samples(params.onset_search_ms, dt)
    n_avg_peak = max(1, params.n_avg_peak)

    # Step 1: snap the click to the NEAREST of detect_events' own candidate
    # peaks (find_peaks(sv, distance=search_n) over the whole trace -- same
    # call, so this always agrees with what a full detect_events scan would
    # have proposed at this location). Deliberately nearest, not tallest: a
    # window search here would readily jump to a taller neighboring event
    # instead of the one actually clicked, since two genuine candidates can
    # legally sit as few as search_n samples apart.
    candidate_idxs, _ = find_peaks(sv, distance=search_n)
    if len(candidate_idxs) == 0:
        peak_idx = click_idx
    else:
        peak_idx = int(candidate_idxs[np.argmin(np.abs(candidate_idxs - click_idx))])
    peak_time_s = peak_idx * dt

    c_half = max(1, search_n // 2)
    b0 = peak_idx - before_n - avg_n
    b1 = peak_idx - before_n
    result = CandidateResult(
        peak_idx=peak_idx, peak_time_s=peak_time_s, accepted=False,
        b0=max(0, b0), b1=max(0, b1),
        local_max_lo=max(0, peak_idx - c_half), local_max_hi=min(len(v), peak_idx + c_half),
    )
    if b0 < 0:
        result.reasons.append(
            f"baseline window (d)+(e) starts before the recording (needs {(before_n + avg_n) * dt * 1e3:.1f}ms "
            f"before the peak) -- click a peak later in the trace, or shorten (d)/(e)")
        return result

    # Step 2: baseline -- ALWAYS the plain (d)/(e) window average here, see
    # module docstring for why (no overlap-baseline extrapolation for an
    # isolated ad-hoc click).
    baseline = float(np.mean(v[b0:b1]))
    result.baseline = baseline

    # Step 3: amplitude vs. threshold.
    p0 = max(0, peak_idx - n_avg_peak // 2)
    p1 = min(len(v), p0 + n_avg_peak)
    peak_v = float(np.mean(v[p0:p1]))
    amplitude = peak_v - baseline
    result.p0, result.p1, result.peak_v, result.amplitude = p0, p1, peak_v, amplitude
    if abs(amplitude) < params.amplitude_threshold:
        result.reasons.append(f"|amplitude|={abs(amplitude):.2f} < threshold (a)={params.amplitude_threshold}")

    s_baseline, s_peak = pol * baseline, pol * peak_v
    span = s_peak - s_baseline
    if span <= 0:
        result.reasons.append("baseline is on the wrong side of the peak (degenerate) -- "
                               "this is very unlikely to be a real event at this click location")
        return result

    # Step 4: onset (rise crossing), within onset_search_ms.
    onset_level = s_baseline + params.onset_fraction * span
    onset_search_start = max(0, peak_idx - onset_search_n)
    result.onset_search_start = onset_search_start
    onset_idx = None
    for k in range(peak_idx, onset_search_start - 1, -1):
        if sv[k] < onset_level:
            onset_idx = k
            break
    if onset_idx is None:
        result.reasons.append(f"onset (fraction {params.onset_fraction}) not reached within "
                               f"onset_search_ms={params.onset_search_ms}")
    else:
        result.onset_idx = onset_idx
        result.rise_time_ms = (peak_idx - onset_idx) * dt * 1e3

    # Step 5: decay, within decay_search_ms (f).
    decay_level = s_baseline + (1.0 - params.decay_fraction) * span
    decay_search_end = min(len(sv), peak_idx + decay_n)
    result.decay_search_end = decay_search_end
    decay_idx = None
    for k in range(peak_idx, decay_search_end):
        if sv[k] <= decay_level:
            decay_idx = k
            break
    if decay_idx is None:
        result.reasons.append(f"decay (fraction (g)={params.decay_fraction}) not reached within "
                               f"decay_search_ms (f)={params.decay_search_ms}")
    else:
        result.decay_idx = decay_idx
        result.decay_time_ms = (decay_idx - peak_idx) * dt * 1e3

    # Step 6: area, only computable once both endpoints are known.
    if onset_idx is not None and decay_idx is not None:
        area = float(_trapz(v[onset_idx:decay_idx + 1] - baseline, dx=dt * 1e3))
        result.area = area
        if abs(area) < params.area_threshold:
            result.reasons.append(f"|area|={abs(area):.2f} < threshold (b)={params.area_threshold}")

    result.accepted = len(result.reasons) == 0
    return result


def plot_candidate_detail(ax, t: np.ndarray, v: np.ndarray, dt: float, result: CandidateResult,
                           params: DetectionParams, y_unit: str, pre_ms: float, post_ms: float):
    """Same annotated-trace style as sepsc.inspect.plot_event_detail, but
    tolerant of a REJECTED candidate's missing pieces (no onset_idx and/or
    decay_idx, no area) -- draws whatever was actually computed, and the
    title states every reason the click failed."""
    sign = -1.0 if params.direction == "negative" else 1.0
    peak_idx = result.peak_idx
    pre_n = _samples(pre_ms, dt)
    post_n = _samples(post_ms, dt)
    i0, i1 = max(0, peak_idx - pre_n), min(len(v), peak_idx + post_n)
    t_ms = (np.arange(i0, i1) - peak_idx) * dt * 1e3

    def rel(idx):
        return (idx - peak_idx) * dt * 1e3

    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.plot(t_ms, v[i0:i1], color=TRACE, lw=1.0, zorder=3)
    ax.plot(0, v[peak_idx], "x", color=COLOR_PEAK, ms=10, mew=2, zorder=5)

    if result.baseline is None:
        ax.set_xlabel("time from peak (ms)", color=INK, fontsize=10)
        ax.set_ylabel(f"current ({y_unit})", color=INK, fontsize=10)
        ax.tick_params(colors=MUTED)
        ax.set_title(f"t={result.peak_time_s:.3f}s   REJECTED: {result.reasons[0]}",
                     color="crimson", fontsize=10, fontweight="bold", wrap=True)
        return

    baseline, amplitude = result.baseline, result.amplitude
    ax.axvspan(rel(result.b0), rel(result.b1), color=COLOR_BASELINE_WINDOW, alpha=0.35, zorder=1)
    ax.axvspan(rel(result.b1), rel(peak_idx), color=COLOR_BEFORE_PEAK, alpha=0.35, zorder=1)
    ax.axhline(baseline, color=COLOR_BASELINE_LINE, ls="--", lw=1.2, zorder=2)
    ax.axvline(rel(result.onset_search_start), color=COLOR_ONSET_WINDOW, ls="--", lw=1.4, zorder=2)
    ax.axvspan(rel(peak_idx), rel(result.decay_search_end), color=COLOR_DECAY_WINDOW, alpha=0.20, zorder=1)
    ax.axvspan(rel(result.p0), rel(result.p1), color=COLOR_PEAK, alpha=0.15, zorder=1)

    y_c = baseline + sign * 0.08 * abs(amplitude if amplitude else 1.0)
    x_c_lo, x_c_hi = rel(result.local_max_lo), rel(result.local_max_hi)
    ax.annotate("", xy=(x_c_lo, y_c), xytext=(x_c_hi, y_c),
                arrowprops=dict(arrowstyle="<->", color=COLOR_LOCAL_MAX, lw=1.4))

    thresh_v = baseline + sign * params.amplitude_threshold
    x_a = t_ms[0] * 0.5
    ax.annotate("", xy=(x_a, baseline), xytext=(x_a, thresh_v),
                arrowprops=dict(arrowstyle="<->", color=COLOR_AMPLITUDE, lw=1.4))
    ax.text(x_a, (baseline + thresh_v) / 2, f" a={params.amplitude_threshold}", color=COLOR_AMPLITUDE,
             fontsize=9, fontweight="bold", va="center", ha="left", bbox=_LABEL_BBOX, zorder=6)
    ax.axhline(thresh_v, color=COLOR_AMPLITUDE, ls=":", lw=1.0, zorder=1)

    decay_level = baseline + (1.0 - params.decay_fraction) * amplitude
    ax.axhline(decay_level, color=COLOR_DECAY_LEVEL, ls=":", lw=1.2, zorder=1)
    ax.text(rel(result.decay_search_end), decay_level, f" g={params.decay_fraction}", color=COLOR_DECAY_LEVEL,
             fontsize=9, fontweight="bold", ha="left", va="center", bbox=_LABEL_BBOX, zorder=6)

    if result.onset_idx is not None:
        ax.plot(rel(result.onset_idx), v[result.onset_idx], "o", color=COLOR_ONSET, ms=6,
                 mec=INK, mew=0.5, zorder=4)
    if result.decay_idx is not None:
        ax.plot(rel(result.decay_idx), v[result.decay_idx], "o", color=COLOR_DECAY_LEVEL, ms=6,
                 mec=INK, mew=0.5, zorder=4)
    if result.onset_idx is not None and result.decay_idx is not None:
        area_ok = result.area is not None and abs(result.area) >= params.area_threshold
        fill_color = COLOR_AREA_PASS if area_ok else COLOR_AREA_FAIL
        onset_i, decay_i = result.onset_idx, result.decay_idx + 1
        ax.fill_between(rel(np.arange(onset_i, decay_i)), v[onset_i:decay_i], baseline,
                         color=fill_color, alpha=0.35, zorder=2)

    ax.set_xlabel("time from peak (ms)", color=INK, fontsize=10)
    ax.set_ylabel(f"current ({y_unit})", color=INK, fontsize=10)
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=0.15, color=GRID)

    verdict = "ACCEPTED" if result.accepted else "REJECTED"
    verdict_color = "#2a8f2a" if result.accepted else "crimson"
    amp_str = f"{amplitude:.2f}" if amplitude is not None else "n/a"
    area_str = f"{result.area:.2f}" if result.area is not None else "n/a"
    ax.set_title(
        f"t={result.peak_time_s:.3f}s   amplitude={amp_str} {y_unit} (thr {params.amplitude_threshold})   "
        f"area={area_str} (thr {params.area_threshold})   [{verdict}]",
        color=verdict_color, fontsize=10, fontweight="bold",
    )


def _hline() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    return line


class ParamPanel(QtWidgets.QWidget):
    """Every DetectionParams field as a live-editable widget -- read fresh
    on each trace click via read_params(), no separate Apply step needed."""

    def __init__(self, initial: DetectionParams, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Detection parameters")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        note = QtWidgets.QLabel("Edit a value, then click a peak (or the same one again) in the trace.")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QtWidgets.QFormLayout()
        self.widgets = {}
        for attr, label, kind, kwargs in _PARAM_FIELD_SPECS:
            w = make_field_widget(kind, kwargs, getattr(initial, attr))
            self.widgets[attr] = w
            form.addRow(label + ":", w)
        layout.addLayout(form)
        layout.addStretch(1)

    def read_params(self) -> DetectionParams:
        values = {attr: read_field_widget(self.widgets[attr], kind) for attr, _, kind, _ in _PARAM_FIELD_SPECS}
        return DetectionParams(**values)


class FilterPanel(QtWidgets.QWidget):
    """Bessel low-pass + resample settings (sepsc.preprocess), live-editable
    like ParamPanel -- but filtering the WHOLE trace is comparatively
    expensive (a zero-phase filtfilt over possibly millions of samples), so
    unlike detection parameters this has its own explicit "Apply" button
    rather than re-filtering on every click."""

    def __init__(self, filter_info: dict, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Filter / resample")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        note = QtWidgets.QLabel(
            "Edit these, then click Apply -- unlike the detection parameters above, filtering the "
            "whole trace is too expensive to redo on every click.")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QtWidgets.QFormLayout()
        self.enabled_w = QtWidgets.QCheckBox()
        self.enabled_w.setChecked(filter_info["enabled"])
        form.addRow("Apply Bessel low-pass + resample:", self.enabled_w)

        self.cutoff_w = QtWidgets.QDoubleSpinBox()
        self.cutoff_w.setRange(1.0, 1e6)
        self.cutoff_w.setDecimals(1)
        self.cutoff_w.setSingleStep(100.0)
        self.cutoff_w.setValue(filter_info["cutoff_hz"])
        form.addRow("Bessel low-pass cutoff, Hz:", self.cutoff_w)

        self.rate_w = QtWidgets.QDoubleSpinBox()
        self.rate_w.setRange(1.0, 1e7)
        self.rate_w.setDecimals(1)
        self.rate_w.setSingleStep(1000.0)
        self.rate_w.setValue(filter_info["target_rate_hz"])
        form.addRow("Resample to, Hz:", self.rate_w)

        self.order_w = QtWidgets.QSpinBox()
        self.order_w.setRange(1, 16)
        self.order_w.setValue(filter_info["filter_order"])
        form.addRow("Bessel filter order:", self.order_w)

        self._filter_only_widgets = (self.cutoff_w, self.rate_w, self.order_w)
        for w in self._filter_only_widgets:
            w.setEnabled(filter_info["enabled"])
        self.enabled_w.toggled.connect(self._on_enabled_toggled)

        layout.addLayout(form)
        self.apply_btn = QtWidgets.QPushButton("Apply filter settings")
        layout.addWidget(self.apply_btn)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def _on_enabled_toggled(self, checked: bool):
        for w in self._filter_only_widgets:
            w.setEnabled(checked)

    def read_filter_info(self) -> dict:
        return dict(enabled=self.enabled_w.isChecked(), cutoff_hz=self.cutoff_w.value(),
                    target_rate_hz=self.rate_w.value(), filter_order=self.order_w.value())


class OptimizerWindow(QtWidgets.QMainWindow):
    def __init__(self, raw_t: np.ndarray, raw_v: np.ndarray, raw_dt: float, y_unit: str,
                 initial_params: DetectionParams, abf_path: str, filter_info: dict):
        super().__init__()
        self.raw_t, self.raw_v, self.raw_dt = raw_t, raw_v, raw_dt
        self.y_unit = y_unit
        self.abf_path = abf_path
        self.filter_info = dict(filter_info)
        self.t, self.v, self.dt = self._compute_filtered_trace(self.filter_info)

        self.setWindowTitle(f"sEPSC parameter optimizer -- {os.path.basename(abf_path)}")
        self.resize(1750, 950)

        # -- left panel: detection params + filter settings + "run" --------
        self.param_panel = ParamPanel(initial_params)
        self.filter_panel = FilterPanel(self.filter_info)
        self.filter_panel.apply_btn.clicked.connect(self.apply_filter_settings)

        left_scroll_widget = QtWidgets.QWidget()
        left_scroll_layout = QtWidgets.QVBoxLayout(left_scroll_widget)
        left_scroll_layout.addWidget(self.param_panel)
        left_scroll_layout.addWidget(_hline())
        left_scroll_layout.addWidget(self.filter_panel)
        param_scroll = QtWidgets.QScrollArea()
        param_scroll.setWidget(left_scroll_widget)
        param_scroll.setWidgetResizable(True)
        # Minimum only, deliberately no setMaximumWidth: a max width pins
        # this pane's on-screen size regardless of the splitter's own
        # proportional resize, so the panel would stay a fixed width no
        # matter how the main window is resized. Leaving only a minimum
        # keeps it from collapsing when the window shrinks, while still
        # letting the splitter grow/shrink it with everything else.
        param_scroll.setMinimumWidth(320)

        run_btn = QtWidgets.QPushButton("Run full detection with these parameters")
        run_btn.clicked.connect(self.run_full_detection)
        self.run_status = QtWidgets.QLabel("")
        self.run_status.setWordWrap(True)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)

        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(param_scroll)
        left_layout.addWidget(run_btn)
        left_layout.addWidget(self.run_status)
        left_layout.addWidget(close_btn)

        # -- top-right: full trace, pan/zoom in x and y (pyqtgraph) --------
        pg.setConfigOption("background", SURFACE)
        pg.setConfigOption("foreground", INK)
        pg.setConfigOptions(antialias=False)
        self.trace_plot = pg.PlotWidget()
        plot_item = self.trace_plot.getPlotItem()
        plot_item.setLabel("bottom", "time", units="s")
        plot_item.setLabel("left", f"current ({y_unit})")
        plot_item.showGrid(x=True, y=True, alpha=0.15)
        vb = plot_item.getViewBox()
        vb.setMouseEnabled(x=True, y=True)  # explicit: zoom/pan on BOTH axes
        # NOT autoDownsample/clipToView -- see DISPLAY_MAX_POINTS' comment.
        # Manual re-slice/decimate to the visible range on every X-range
        # change instead (_redraw_visible_trace), which sidesteps that bug
        # entirely by never touching the codepath that trips it.
        self.plot_curve = plot_item.plot([], [], pen=pg.mkPen(TRACE, width=0))
        self.marker = pg.ScatterPlotItem(size=13, symbol="o", pen=pg.mkPen("gold", width=2.5),
                                          brush=pg.mkBrush(None))
        plot_item.addItem(self.marker)
        self.trace_plot.scene().sigMouseClicked.connect(self.on_trace_click)
        vb.sigXRangeChanged.connect(self._on_view_range_changed)

        instructions = QtWidgets.QLabel(
            "Click near a peak below to measure it with the current parameters (snaps to the "
            "nearest local maximum). Scroll to zoom X+Y together; drag on an axis to zoom just "
            "that axis; drag inside the plot to pan.")
        instructions.setWordWrap(True)

        top_widget = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_widget)
        top_layout.setContentsMargins(4, 4, 4, 4)
        top_layout.addWidget(instructions)
        top_layout.addWidget(self.trace_plot)

        # -- bottom-right: measured-event detail (matplotlib) --------------
        self.figure = Figure(figsize=(9, 5))
        self.figure.patch.set_facecolor(SURFACE)
        self.canvas = FigureCanvas(self.figure)
        self.ax_detail = self.figure.subplots()
        self.ax_detail.set_facecolor(SURFACE)
        self.status_label = QtWidgets.QLabel("Click a peak in the trace above to measure it.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-family: Consolas, monospace;")

        bottom_widget = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(4, 4, 4, 4)
        bottom_layout.addWidget(self.canvas)
        bottom_layout.addWidget(self.status_label)

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_splitter.addWidget(top_widget)
        right_splitter.addWidget(bottom_widget)
        right_splitter.setSizes([520, 430])

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(left_container)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([330, 1420])
        self.setCentralWidget(main_splitter)

        self.last_result: Optional[CandidateResult] = None
        self._reset_trace_view()

    def _compute_filtered_trace(self, filter_info: dict):
        """(t, v, dt) for the current filter settings, always derived fresh
        from the RAW trace (never re-filtering an already-filtered one)."""
        if not filter_info.get("enabled"):
            return self.raw_t, self.raw_v, self.raw_dt
        raw_fs = 1.0 / self.raw_dt
        v_filt = bessel_lowpass(self.raw_v, raw_fs, cutoff_hz=filter_info["cutoff_hz"],
                                 order=filter_info["filter_order"])
        v_ds, fs_ds = downsample(v_filt, raw_fs, target_hz=filter_info["target_rate_hz"])
        dt_ds = 1.0 / fs_ds
        t_ds = np.arange(len(v_ds)) * dt_ds
        return t_ds, v_ds, dt_ds

    def apply_filter_settings(self):
        new_filter_info = self.filter_panel.read_filter_info()
        self.filter_panel.status_label.setText("Filtering...")
        QtWidgets.QApplication.processEvents()
        try:
            t, v, dt = self._compute_filtered_trace(new_filter_info)
        except Exception as exc:
            self.filter_panel.status_label.setText(f"FAILED: {exc}")
            QtWidgets.QMessageBox.critical(self, "Filter error", f"Could not apply filter settings:\n\n{exc}")
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            return

        self.filter_info = new_filter_info
        self.t, self.v, self.dt = t, v, dt

        # Any earlier click's sample indices belonged to the OLD trace
        # (different length/dt once filtering changes) -- they no longer
        # mean anything, so clear them rather than show a stale/misaligned
        # measurement.
        self.last_result = None
        self.ax_detail.clear()
        self.ax_detail.set_facecolor(SURFACE)
        self.canvas.draw_idle()
        self.marker.setData([], [])
        self.status_label.setText("Filter settings changed -- click a peak in the trace above to measure it.")

        fs = 1.0 / dt if dt else 0.0
        if new_filter_info["enabled"]:
            self.filter_panel.status_label.setText(
                f"Active: {len(v)} samples @ {fs:.0f} Hz (from {1.0/self.raw_dt:.0f} Hz raw)")
        else:
            self.filter_panel.status_label.setText(f"Active: {len(v)} samples @ {fs:.0f} Hz (raw, unfiltered)")

        self._reset_trace_view()

    def _reset_trace_view(self):
        """(Re)plot self.t/self.v as the curve's base data and reset the
        view to show the whole trace -- called on init and whenever the
        trace itself changes (filter settings applied)."""
        plot_item = self.trace_plot.getPlotItem()
        if len(self.t):
            plot_item.getViewBox().setRange(xRange=(self.t[0], self.t[-1]), padding=0.02)
        self._redraw_visible_trace()

    def _on_view_range_changed(self, *_args):
        self._redraw_visible_trace()

    def _redraw_visible_trace(self):
        """Manually decimate self.t/self.v to whatever's currently visible
        -- see DISPLAY_MAX_POINTS' comment for why this isn't left to
        pyqtgraph's own autoDownsample/clipToView."""
        if len(self.t) == 0:
            self.plot_curve.setData([], [])
            return
        (xmin, xmax), _ = self.trace_plot.getPlotItem().getViewBox().viewRange()
        i0 = max(0, int(np.searchsorted(self.t, xmin)) - 1)
        i1 = min(len(self.t), int(np.searchsorted(self.t, xmax)) + 1)
        t_slice = self.t[i0:i1]
        v_slice = self.v[i0:i1]
        if len(t_slice) > DISPLAY_MAX_POINTS:
            stride = max(1, len(t_slice) // DISPLAY_MAX_POINTS)
            t_slice = t_slice[::stride]
            v_slice = v_slice[::stride]
        self.plot_curve.setData(t_slice, v_slice)

    @safe_callback
    def on_trace_click(self, mouse_click_event):
        if mouse_click_event.button() != QtCore.Qt.LeftButton:
            return
        plot_item = self.trace_plot.getPlotItem()
        if not plot_item.sceneBoundingRect().contains(mouse_click_event.scenePos()):
            return  # click landed outside the plot (e.g. on an axis label)
        data_pos = plot_item.getViewBox().mapSceneToView(mouse_click_event.scenePos())
        click_idx = int(round(data_pos.x() / self.dt))
        click_idx = max(0, min(len(self.v) - 1, click_idx))
        self.measure_at(click_idx)

    def measure_at(self, click_idx: int):
        params = self.param_panel.read_params()
        result = evaluate_candidate(self.v, self.dt, click_idx, params)
        self.last_result = result

        pre_ms = params.baseline_before_ms + max(params.baseline_avg_ms, params.onset_search_ms) + 3.0
        post_ms = params.decay_search_ms + 3.0
        plot_candidate_detail(self.ax_detail, self.t, self.v, self.dt, result, params,
                               self.y_unit, pre_ms, post_ms)
        self.canvas.draw_idle()

        self.marker.setData([result.peak_time_s], [self.v[result.peak_idx]])

        lines = [f"Peak @ t={result.peak_time_s:.4f}s (sample {result.peak_idx})",
                 f"Verdict: {'ACCEPTED' if result.accepted else 'REJECTED'}"]
        if result.amplitude is not None:
            lines.append(f"amplitude={result.amplitude:.2f} {self.y_unit}   "
                         f"rise={result.rise_time_ms if result.rise_time_ms is not None else 'n/a'}"
                         f"{'ms' if result.rise_time_ms is not None else ''}   "
                         f"decay={result.decay_time_ms if result.decay_time_ms is not None else 'n/a'}"
                         f"{'ms' if result.decay_time_ms is not None else ''}   "
                         f"area={result.area if result.area is not None else 'n/a'}")
        if result.reasons:
            lines.append("Reasons:")
            lines += [f"  - {r}" for r in result.reasons]
        self.status_label.setText("\n".join(lines))

    def run_full_detection(self):
        # Deliberately NOT @safe_callback here: that decorator only prints
        # to stderr, which is invisible in a windowed/no-console launch --
        # exactly the shape of "I clicked the button and nothing happened".
        # Every outcome (success or failure) gets an unmissable QMessageBox
        # here instead, plus immediate "Running..." feedback before the
        # (possibly multi-second, for a large trace) detection call so the
        # click itself is never left looking unacknowledged.
        self.run_status.setText("Running full detection...")
        QtWidgets.QApplication.processEvents()
        try:
            params = self.param_panel.read_params()
            events = detect_events(self.t, self.v, self.dt, params)
            df = events_frame(events)

            stem = os.path.splitext(self.abf_path)[0]
            if self.filter_info.get("enabled"):
                stem += f"_filt{int(self.filter_info['cutoff_hz'])}Hz{int(self.filter_info['target_rate_hz'])}Hz"
            out_csv = f"{stem}_minianalysis_events.csv"
            df.to_csv(out_csv, index=False)
            params_path = f"{stem}_minianalysis_params.json"
            with open(params_path, "w") as fh:
                json.dump(dataclasses.asdict(params), fh, indent=2)
        except Exception as exc:
            self.run_status.setText(f"FAILED: {exc}")
            QtWidgets.QMessageBox.critical(self, "Detection failed", f"{exc}\n\n(full traceback printed to console)")
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            return

        rate_hz = len(events) / (len(self.v) * self.dt) if len(self.v) else 0.0
        msg = f"Detected {len(events)} events ({rate_hz:.3f} Hz)\nSaved -> {os.path.basename(out_csv)}"
        self.run_status.setText(msg)
        QtWidgets.QMessageBox.information(self, "Detection complete", msg)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", help="Path to the gap-free voltage-clamp .abf file")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--params-json", default=None,
                   help="initial parameters sidecar to load (default: <abf>_minianalysis_params.json "
                        "if it exists next to the .abf, else built-in defaults)")
    # Same override flags as minianalysis.py/inspect.py -- INITIAL values only, all still editable
    # live in the left panel afterward.
    p.add_argument("--direction", choices=["negative", "positive"], default=None)
    p.add_argument("--amplitude-threshold", type=float, default=None)
    p.add_argument("--area-threshold", type=float, default=None)
    p.add_argument("--n-avg-peak", type=int, default=None)
    p.add_argument("--search-local-max-ms", type=float, default=None)
    p.add_argument("--baseline-before-ms", type=float, default=None)
    p.add_argument("--baseline-avg-ms", type=float, default=None)
    p.add_argument("--decay-search-ms", type=float, default=None)
    p.add_argument("--decay-fraction", type=float, default=None)
    p.add_argument("--onset-fraction", type=float, default=None)
    p.add_argument("--onset-search-ms", type=float, default=None)
    p.add_argument("--filter", action="store_true",
                   help="start with the Bessel-filtered + resampled trace (see sepsc.preprocess) -- "
                        "still fully editable afterward in the 'Filter / resample' panel")
    p.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ,
                   help=f"only with --filter: initial Bessel low-pass cutoff, Hz (default {DEFAULT_CUTOFF_HZ:.0f})")
    p.add_argument("--target-rate-hz", type=float, default=DEFAULT_TARGET_RATE_HZ,
                   help=f"only with --filter: initial output sampling rate, Hz (default {DEFAULT_TARGET_RATE_HZ:.0f})")
    p.add_argument("--filter-order", type=int, default=DEFAULT_ORDER,
                   help=f"only with --filter: initial Bessel filter order (default {DEFAULT_ORDER})")
    args = p.parse_args(argv)

    stem = os.path.splitext(args.abf)[0]
    overrides = dict(
        direction=args.direction, amplitude_threshold=args.amplitude_threshold,
        area_threshold=args.area_threshold, n_avg_peak=args.n_avg_peak,
        search_local_max_ms=args.search_local_max_ms, baseline_before_ms=args.baseline_before_ms,
        baseline_avg_ms=args.baseline_avg_ms, decay_search_ms=args.decay_search_ms,
        decay_fraction=args.decay_fraction, onset_fraction=args.onset_fraction,
        onset_search_ms=args.onset_search_ms,
    )
    params = _load_params(args.params_json, stem, overrides)

    abf = pyabf.ABF(args.abf)
    abf.setSweep(0, channel=args.channel)
    y_unit = abf.adcUnits[args.channel]

    # ALWAYS load the raw trace -- OptimizerWindow keeps it and derives the
    # active (possibly filtered) one from it, so filter settings can be
    # changed live later without re-reading the file.
    raw_t = np.asarray(abf.sweepX, float)
    raw_v = np.asarray(abf.sweepY, float)
    raw_dt = 1.0 / abf.dataRate

    filter_info = dict(enabled=args.filter, cutoff_hz=args.cutoff_hz,
                        target_rate_hz=args.target_rate_hz, filter_order=args.filter_order)
    if args.filter:
        hardware_filter_hz = get_hardware_filter_hz(abf, args.channel)
        if hardware_filter_hz is not None and hardware_filter_hz <= args.cutoff_hz:
            print(f"NOTE: channel {args.channel} is already hardware-filtered at "
                  f"{hardware_filter_hz:.0f} Hz, at or below the requested {args.cutoff_hz:.0f} Hz.")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = OptimizerWindow(raw_t, raw_v, raw_dt, y_unit, params, args.abf, filter_info)
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
