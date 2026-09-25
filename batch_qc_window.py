"""PyQt5 window that shows, at a glance, whether a batch of recordings is any
good: the F-I curves of every cell overlaid, the cell-health measurements side
by side against QC guides, and the raw traces behind them -- each cell at
rheobase, at half its maximum firing rate, and at the maximum.

The half-max sweep is the one worth looking hardest at: it sits on the steep
part of the F-I curve, where a cell that is unhealthy or poorly clamped
departs from a healthy one soonest.

Access resistance and capacitance are shown twice per cell -- from the
membrane test before the step recordings and the one after -- joined by a
line, so a cell that ran down during the recording reads as a slope rather
than as a single number that happens to look acceptable.

Opened automatically at the end of a ``batch_cell_analysis.py`` run, and
available on its own for a folder that has already been analysed:

    python batch_cell_analysis.py DATA_ROOT --view

Everything is read back from the files the run wrote -- ``batch_cell_summary.csv``
and each cell's ``*_AP_sweeps.csv`` -- plus the .abf files themselves for the
traces, so the window works just as well days later as it does immediately
after the run.

The QC guides below are the conventional thresholds for patch-clamp cell
health, not measurements: they are drawn as dashed lines and used only to
colour a point red and count it as flagged. Edit them to match what your
experiment actually tolerates. Nothing here alters the analysis or the saved
summary; a flagged cell is a prompt to look at its traces, not a verdict.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

import batch_sepsc

# (column before, column after, label, unit, threshold, which side is bad)
# The "after" column is None for everything measured from the step recording
# itself, which happens once and so has no before/after to show.
QC_GUIDES = [
    ("mt_Ra_MOhm",      "mt_post_Ra_MOhm", "Access resistance",  "MOhm", 25.0,  "above"),
    ("mt_Cm_pF",        "mt_post_Cm_pF",   "Capacitance",        "pF",   None,  None),
    ("resting_Vm_mV",   None,              "Resting Vm",         "mV",   -55.0, "above"),
    ("rheobase_pA",     None,              "Rheobase",           "pA",   None,  None),
    ("AP_amplitude_mV", None,              "AP amplitude",       "mV",   40.0,  "below"),
    ("AP_halfwidth_ms", None,              "AP half-width",      "ms",   None,  None),
]

# Ra moving by more than this across the recordings flags the cell, however
# healthy both ends look on their own: it is the change, not the value, that
# says the access was unstable while the F-I curve was being measured.
RA_DRIFT_PCT = 20.0

# sEPSC guides, in the same spirit: conventions, not measurements.
# Below this ratio of median amplitude to noise SD the events being counted
# are close enough to the noise floor that the RATE is the first thing to
# become unreliable -- the detector finds whatever the threshold lets through.
SEPSC_MIN_SNR = 3.0
# Holding current moving by more than this over the recording means the
# amplitudes at one end are not comparable with the other.
SEPSC_HOLD_DRIFT_PA = 20.0

_app_ref = None      # keeps the QApplication alive when this module made it


def qt_app() -> QtWidgets.QApplication:
    """The QApplication, made with high-DPI scaling on if it doesn't exist yet.

    Windows display scaling (150% here) makes the screen report a logical DPI
    of 144 while Qt still reports a device pixel ratio of 1, so widget chrome
    is laid out large while the matplotlib canvases render at 100 dpi -- the
    figures come out cramped and their text oversized relative to the axes.
    These attributes have to be set BEFORE the QApplication exists, so a
    process that already made one keeps whatever it chose.
    """
    global _app_ref
    app = QtWidgets.QApplication.instance()
    if app is not None:
        return app
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    # Qt 5 rounds a scale factor by default, so Windows' 150% becomes 200% and
    # a 1920x1200 display shrinks to 960x600 of usable layout space. Pass the
    # fractional factor through instead, so the window matches the size
    # everything else on the desktop is drawn at.
    try:
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except AttributeError:          # Qt < 5.14
        pass
    _app_ref = app = QtWidgets.QApplication(sys.argv[:1])
    return app


# ---------------------------------------------------------------------------
# Loading a finished run
# ---------------------------------------------------------------------------
def load_run(root: Path, summary: Optional[pd.DataFrame] = None) -> dict:
    """Summary table, per-cell F-I sweeps, and where each cell's recording is.

    A cell whose sweep table is missing simply has no F-I curve -- the rest of
    the window still works, since a partly failed run is exactly when someone
    wants to look at it.
    """
    root = Path(root)
    if summary is None:
        csv = root / "batch_cell_summary.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"{csv} not found -- run the analysis on this folder first.")
        summary = pd.read_csv(csv)

    sweeps: dict[str, pd.DataFrame] = {}
    steps_abf: dict[str, Path] = {}
    sepsc_events: dict[str, pd.DataFrame] = {}
    sepsc_waves: dict[str, dict] = {}
    for _i, row in summary.iterrows():
        name = str(row.get("folder_name", ""))
        folder = Path(str(row.get("folder", root / name)))

        # sEPSCs are optional twice over: the cell may have no gap-free
        # recording, and a run made before this stage existed has neither the
        # column nor the files. Both simply leave the tab out.
        sepsc = str(row.get("file_sepsc", "") or "")
        if sepsc:
            events = batch_sepsc.load_events(folder / sepsc)
            waves = batch_sepsc.load_waveforms(folder / sepsc)
            if events is not None and not events.empty:
                sepsc_events[name] = events
            if waves is not None and waves.get("waveforms", np.zeros(0)).size:
                sepsc_waves[name] = waves

        steps = str(row.get("file_steps", "") or "")
        if not steps:
            continue
        abf = folder / steps
        if abf.exists():
            steps_abf[name] = abf
        csv = folder / f"{Path(steps).stem}_AP_sweeps.csv"
        if csv.exists():
            try:
                sweeps[name] = pd.read_csv(csv)
            except Exception:
                pass
    return {"root": root, "summary": summary, "sweeps": sweeps, "abf": steps_abf,
            "sepsc_events": sepsc_events, "sepsc_waves": sepsc_waves}


def _num(row, key) -> float:
    """A row value as a float, NaN when missing or non-numeric.

    pandas hands back numpy scalars, and np.int64 is NOT a subclass of int
    (unlike np.float64, which is a float), so an isinstance check would
    silently drop every integer column -- sweep numbers included.
    """
    try:
        v = float(row.get(key, np.nan))
    except (TypeError, ValueError):
        return float("nan")
    return v


def _short(name: str) -> str:
    """A cell folder name without its leading YYMMDD_.

    With one recording date per run the date is identical on every cell, so it
    is pure width -- whether on an axis tick or a three-across panel title.
    """
    return re.sub(r"^\d{6}_", "", name)


def _flagged(summary: pd.DataFrame) -> dict[str, list[str]]:
    """Which cells trip a QC guide, and on what.

    A paired measurement is checked at both ends -- a cell whose Ra started
    fine and ended at 40 MOhm is flagged on the closing value."""
    out: dict[str, list[str]] = {}

    def flag(row, reason):
        out.setdefault(str(row.get("folder_name", "")), []).append(reason)

    for col, post_col, label, unit, thresh, side in QC_GUIDES:
        if thresh is None:
            continue
        for when, c in (("", col), (" after", post_col)):
            if not c or c not in summary:
                continue
            for _i, row in summary.iterrows():
                v = _num(row, c)
                if not np.isfinite(v):
                    continue
                if v > thresh if side == "above" else v < thresh:
                    flag(row, f"{label}{when} {v:.1f} {unit}")

    if "mt_Ra_change_pct" in summary:
        for _i, row in summary.iterrows():
            d = _num(row, "mt_Ra_change_pct")
            if np.isfinite(d) and abs(d) > RA_DRIFT_PCT:
                flag(row, f"Ra drifted {d:+.0f}%")

    for _i, row in summary.iterrows():
        snr = _num(row, "sepsc_snr")
        if np.isfinite(snr) and snr < SEPSC_MIN_SNR:
            flag(row, f"sEPSC SNR {snr:.1f}")
        hold = _num(row, "sepsc_holding_drift_pA")
        if np.isfinite(hold) and abs(hold) > SEPSC_HOLD_DRIFT_PA:
            flag(row, f"holding drifted {hold:+.0f} pA")
        if str(row.get("sepsc_kernel_source", "")) == "analytic-fallback":
            flag(row, "sEPSC kernel not estimated from the data")
    return out


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------
class QCWindow(QtWidgets.QMainWindow):
    def __init__(self, run: dict, parent=None):
        super().__init__(parent)
        self.run = run
        self.summary = run["summary"]
        self.names = [str(n) for n in self.summary.get("folder_name", [])]
        cmap = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
        self.colors = {n: cmap[i % len(cmap)] for i, n in enumerate(self.names)}
        self.flags = _flagged(self.summary)

        self.setWindowTitle(f"Recording QC  -  {run['root']}")
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._wrap(self._fi_figure()), "F-I curves")
        tabs.addTab(self._wrap(self._health_figure()), "Cell health")
        tabs.addTab(self._traces_tab(), "Traces")
        # Only when something was actually detected: a batch of current-clamp
        # recordings has no gap-free files and should not get an empty tab.
        if run.get("sepsc_events") or run.get("sepsc_waves"):
            tabs.addTab(self._wrap(self._sepsc_figure()), "sEPSCs")
        self.setCentralWidget(tabs)

        n_flagged = len(self.flags)
        msg = f"{len(self.names)} cell(s)."
        if n_flagged:
            msg += "  Flagged: " + "; ".join(
                f"{name} ({', '.join(reasons)})" for name, reasons in self.flags.items())
        else:
            msg += "  No cell trips a QC guide."
        self.statusBar().showMessage(msg)

        # Fit the screen rather than assuming one: at 150% scaling a 1920x1200
        # display is only 1280x800 logical pixels, so a hardcoded 1250x820
        # window is taller than the desktop and the status bar falls off it.
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1250, 820)
        self.resize(min(1250, int(avail.width() * 0.95)),
                    min(860, int(avail.height() * 0.92)))

    def _wrap(self, fig: Figure) -> QtWidgets.QWidget:
        """A figure plus the matplotlib toolbar, so anything can be zoomed and
        saved out of the window."""
        page = QtWidgets.QWidget()
        canvas = FigureCanvas(fig)
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(NavigationToolbar(canvas, page))
        layout.addWidget(canvas)
        return page

    # -- tab 1: F-I curves ---------------------------------------------
    def _fi_figure(self) -> Figure:
        fig = Figure(figsize=(12, 7.2), tight_layout=True)
        ax = fig.add_subplot(111)
        plotted = 0
        for _i, row in self.summary.iterrows():
            name = str(row.get("folder_name", ""))
            sw = self.run["sweeps"].get(name)
            if sw is None or "stim_amp_pA" not in sw or "avg_rate" not in sw:
                continue
            fi = sw.dropna(subset=["stim_amp_pA"]).sort_values("stim_amp_pA")
            rheo = _num(row, "rheobase_pA")
            # Prefer the Vm measured from this recording's own 0 pA sweep; fall
            # back to the RMP typed into the folder name at the rig, marked as
            # such, since the two are taken minutes apart and do drift.
            rmp, rmp_note = _num(row, "resting_Vm_mV"), ""
            if not np.isfinite(rmp):
                rmp, rmp_note = _num(row, "rmp_folder_mV"), " rig"

            bias = _num(row, "bias_current_pA")

            bits = []
            if np.isfinite(rheo):
                bits.append(f"rheo {rheo:.0f} pA")
            if np.isfinite(rmp):
                bits.append(f"RMP {rmp:.1f} mV{rmp_note}")
            if np.isfinite(bias):
                bits.append(f"bias {bias:+.1f} pA")
            label = name + (f"  ({', '.join(bits)})" if bits else "")
            ax.plot(fi["stim_amp_pA"], fi["avg_rate"], "o-", ms=3.5, lw=1.4,
                    color=self.colors[name], label=label, alpha=0.9)

            if np.isfinite(rheo):
                y = fi.loc[fi["stim_amp_pA"] == rheo, "avg_rate"].max()
                if np.isfinite(y):
                    ax.plot([rheo], [y], "v", ms=9, color=self.colors[name], zorder=5)
                    # The RMP rides on the curve at its rheobase, where the
                    # cells are furthest apart; at the right-hand end they
                    # converge and the labels would sit on top of each other.
                    tag = []
                    if np.isfinite(rmp):
                        tag.append(f"{rmp:.1f} mV{rmp_note}")
                    if np.isfinite(bias):
                        tag.append(f"bias {bias:+.1f} pA")
                    if tag:
                        # Above and to the LEFT of the rheobase marker: the
                        # curve is flat at 0 Hz there, whereas below it would
                        # clip against the axis and to the right it would sit
                        # on the steeply rising limb.
                        ax.annotate("\n".join(tag), (rheo, y),
                                    textcoords="offset points", xytext=(-9, 9),
                                    fontsize=7.5, color=self.colors[name],
                                    ha="right", va="bottom", zorder=6, linespacing=1.3,
                                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
            plotted += 1

        ax.set_xlabel("injected current (pA)")
        ax.set_ylabel("firing rate (Hz)")
        # Two short lines: one long title is the first thing to clip when the
        # window is narrower than the machine it was laid out on.
        ax.set_title("F-I curves, all cells overlaid\n"
                     "marker = rheobase, label = resting Vm at 0 pA "
                     "and holding current", fontsize=10)
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        else:
            ax.text(0.5, 0.5, "no per-sweep tables found\n(run the analysis first)",
                    ha="center", va="center", transform=ax.transAxes, color="#b00020")
        return fig

    # -- tab 2: cell health --------------------------------------------
    def _column(self, col: Optional[str]) -> pd.Series:
        """One summary column as floats, all-NaN when the run never wrote it."""
        if not col or col not in self.summary:
            return pd.Series([np.nan] * len(self.names), dtype=float)
        return pd.to_numeric(self.summary[col], errors="coerce")

    def _health_figure(self) -> Figure:
        fig = Figure(figsize=(12, 7.2), tight_layout=True)
        x = np.arange(len(self.names))
        # How far the before/after markers of one cell sit either side of its
        # tick. Wide enough to read as a pair, narrow enough not to be mistaken
        # for the neighbouring cell.
        dx = 0.16
        for k, (col, post_col, label, unit, thresh, side) in enumerate(QC_GUIDES):
            ax = fig.add_subplot(2, 3, k + 1)
            pre_vals, post_vals = self._column(col), self._column(post_col)
            paired = bool(post_col) and bool(np.isfinite(post_vals.to_numpy(float)).any())

            def trips(v):
                return thresh is not None and (v > thresh if side == "above" else v < thresh)

            for i, name in enumerate(self.names):
                a = pre_vals.iloc[i] if i < len(pre_vals) else np.nan
                b = post_vals.iloc[i] if paired and i < len(post_vals) else np.nan
                if np.isfinite(a) and np.isfinite(b):
                    # The line IS the measurement here: its slope is the
                    # run-down over the course of the recording.
                    ax.plot([i - dx, i + dx], [a, b], "-", lw=1.2, alpha=0.65,
                            color=self.colors[name], zorder=2)
                # Circle before, square after; the text leans away from its own
                # marker so the two numbers of a pair can never collide.
                marks = (((a, -dx, "o", -3, "right"), (b, dx, "s", 3, "left"))
                         if paired else ((a, 0.0, "o", 0, "center"),))
                for v, off, marker, tx, ha in marks:
                    if not np.isfinite(v):
                        continue
                    bad = trips(v)
                    ax.plot([i + off], [v], marker, ms=10,
                            color="#d62728" if bad else self.colors[name],
                            markeredgecolor="black" if bad else "none", zorder=3)
                    ax.annotate(f"{v:.1f}", (i + off, v), textcoords="offset points",
                                xytext=(tx, 10), ha=ha, fontsize=7.5)

            if thresh is not None:
                ax.axhline(thresh, color="#d62728", ls="--", lw=1.0, alpha=0.8,
                           label=f"QC guide {thresh:g} {unit}")
                ax.legend(fontsize=7, loc="best")
            ax.set_xticks(x)
            ax.set_xticklabels([_short(n) for n in self.names],
                               rotation=30, ha="right", fontsize=7.5)
            title = f"{label} ({unit})"
            if paired:
                title += "\nbefore (circle) to after (square)"
            ax.set_title(title, fontsize=10)
            if len(x):
                # Paired panels need more room at the edges: the outermost
                # value labels lean outward, and at the default limits the
                # leftmost one is drawn over the y-axis tick labels.
                pad = 0.9 if paired else 0.6
                ax.set_xlim(-pad, len(x) - 1 + pad)
            ax.margins(y=0.25)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Cell health across the batch   (red = trips a QC guide; "
                     "guides are conventions, editable in batch_qc_window.py)",
                     fontsize=10)
        return fig

    # -- tab 4: sEPSCs --------------------------------------------------
    #: Top row of the sEPSC tab. Chosen to answer "is this recording usable",
    #: not "what is this cell doing".
    #: (column, label, unit, threshold, which side is bad)
    SEPSC_METRICS = [
        ("sepsc_rate_Hz",          "Event rate",       "Hz", None, None),
        ("sepsc_amp_median_pA",    "Median amplitude", "pA", None, None),
        ("sepsc_snr",              "SNR",              "",   SEPSC_MIN_SNR, "below"),
        ("sepsc_holding_drift_pA", "Holding drift",    "pA", SEPSC_HOLD_DRIFT_PA, "abs"),
    ]

    #: Individual events drawn per cell in the overlay panels. The mean is
    #: computed from all of them; this only limits what is drawn behind it.
    SEPSC_OVERLAY_N = 80

    def _sepsc_figure(self) -> Figure:
        fig = Figure(figsize=(13, 8.6), tight_layout=True)
        # Four metrics across the top, three plots across the bottom: a
        # 12-column grid divides evenly into both.
        grid = fig.add_gridspec(2, 12)
        waves_by_cell = self.run.get("sepsc_waves", {})
        events_by_cell = self.run.get("sepsc_events", {})

        for k, (col, label, unit, thresh, side) in enumerate(self.SEPSC_METRICS):
            ax = fig.add_subplot(grid[0, 3 * k:3 * k + 3])
            self._sepsc_metric(ax, col, label, unit, thresh, side)

        ax_raw = fig.add_subplot(grid[1, 0:4])
        ax_scaled = fig.add_subplot(grid[1, 4:8])
        ax_amp = fig.add_subplot(grid[1, 8:12])

        for name in self.names:
            wf = waves_by_cell.get(name)
            if wf is None:
                continue
            waves = np.asarray(wf["waveforms"], dtype=float)
            t = np.asarray(wf["t_ms"], dtype=float)
            if waves.size == 0:
                continue
            colour = self.colors[name]
            step = max(1, waves.shape[0] // self.SEPSC_OVERLAY_N)
            for w in waves[::step]:
                ax_raw.plot(t, w, lw=0.4, color=colour, alpha=0.12, zorder=1)
            ax_raw.plot(t, waves.mean(axis=0), lw=1.8, color=colour, zorder=3,
                        label=f"{_short(name)} (n={waves.shape[0]})")

            # Scaled by each event's own measured amplitude, so every trace
            # peaks at -1 and only the kinetics are left to differ. A cell
            # whose events are a different shape shows up here even when its
            # amplitudes look ordinary.
            amps = np.abs(np.asarray(wf["amplitude"], dtype=float))
            ok = amps > 0
            if ok.any():
                scaled = waves[ok] / amps[ok][:, None]
                for w in scaled[::step]:
                    ax_scaled.plot(t, w, lw=0.4, color=colour, alpha=0.12, zorder=1)
                ax_scaled.plot(t, scaled.mean(axis=0), lw=1.8, color=colour, zorder=3)

        # Scaled traces sit at -1 at the peak by construction, so the useful
        # range is small and fixed. Without a limit one event riding on a
        # neighbour's decay -- scaled by a small amplitude, so enormous --
        # flattens every other trace into the top of the panel.
        ax_scaled.set_ylim(-3.0, 1.0)

        for ax, title, ylabel in (
                (ax_raw, "Detected events, overlaid", "current (pA)"),
                (ax_scaled, "The same events, scaled to their own peak", "peak = -1")):
            ax.axhline(0.0, color="#bbbbbb", lw=0.8, ls=":", zorder=2)
            ax.axvline(0.0, color="#bbbbbb", lw=0.8, ls=":", zorder=2)
            ax.set_xlabel("time from peak (ms)", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(title, fontsize=9.5)
            ax.tick_params(labelsize=7.5)
            ax.grid(alpha=0.2)
        if waves_by_cell:
            ax_raw.legend(fontsize=7, loc="lower right", framealpha=0.9)

        self._sepsc_amplitude_scatter(ax_amp, events_by_cell)

        fig.suptitle("Spontaneous EPSCs, detected by deconvolution   "
                     "(red = trips a QC guide; guides are conventions, "
                     "editable in batch_qc_window.py)", fontsize=10)
        return fig

    def _sepsc_metric(self, ax, col, label, unit, thresh, side) -> None:
        """One per-cell metric panel, with its guide and the noise floor."""
        vals = self._column(col)
        limits = self._column("sepsc_detection_limit_pA")
        show_limit = col == "sepsc_amp_median_pA"

        for i, name in enumerate(self.names):
            v = vals.iloc[i] if i < len(vals) else np.nan
            if not np.isfinite(v):
                continue
            if thresh is None:
                bad = False
            elif side == "abs":
                bad = abs(v) > thresh
            else:
                bad = v > thresh if side == "above" else v < thresh
            # The gap between a cell's median event and its own noise floor is
            # the headroom the rate depends on, and it is per cell -- one
            # shared guide line could not show it.
            if show_limit and i < len(limits) and np.isfinite(limits.iloc[i]):
                ax.vlines(i, limits.iloc[i], v, color=self.colors[name],
                          lw=1.2, alpha=0.45, zorder=2)
                ax.plot([i], [limits.iloc[i]], "_", ms=13, color="#777777", zorder=3)
            ax.plot([i], [v], "o", ms=10,
                    color="#d62728" if bad else self.colors[name],
                    markeredgecolor="black" if bad else "none", zorder=4)
            ax.annotate(f"{v:.1f}", (i, v), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=7.5)

        if show_limit:
            ax.plot([], [], "_", color="#777777", label="detection limit (4x noise)")
            ax.legend(fontsize=7, loc="best")
        elif thresh is not None:
            for level in ((thresh, -thresh) if side == "abs" else (thresh,)):
                ax.axhline(level, color="#d62728", ls="--", lw=1.0, alpha=0.8)
            ax.plot([], [], ls="--", color="#d62728",
                    label=f"QC guide {thresh:g}{' ' + unit if unit else ''}")
            ax.legend(fontsize=7, loc="best")

        ax.set_xticks(np.arange(len(self.names)))
        ax.set_xticklabels([_short(n) for n in self.names],
                           rotation=30, ha="right", fontsize=7.5)
        ax.set_title(f"{label} ({unit})" if unit else label, fontsize=10)
        if len(self.names):
            ax.set_xlim(-0.6, len(self.names) - 0.4)
        ax.margins(y=0.25)
        ax.grid(axis="y", alpha=0.25)

    def _sepsc_amplitude_scatter(self, ax, events_by_cell: dict) -> None:
        """Every event's amplitude against when it happened.

        The point of plotting all of them rather than a summary is drift: a
        cell whose amplitudes sag through the recording has a median that
        still looks perfectly reasonable. The binned median line is there so
        that trend is readable through the scatter.
        """
        for name in self.names:
            ev = events_by_cell.get(name)
            if ev is None or ev.empty:
                continue
            t = pd.to_numeric(ev["peak_time_s"], errors="coerce").to_numpy(float)
            a = np.abs(pd.to_numeric(ev["amplitude"], errors="coerce").to_numpy(float))
            ok = np.isfinite(t) & np.isfinite(a)
            t, a = t[ok], a[ok]
            if t.size == 0:
                continue
            ax.plot(t, a, ".", ms=2.5, alpha=0.35, color=self.colors[name], zorder=1)

            n_bins = min(20, max(2, t.size // 15))
            edges = np.linspace(t.min(), t.max(), n_bins + 1)
            which = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
            centres, medians = [], []
            for b in range(n_bins):
                in_bin = a[which == b]
                if in_bin.size:
                    centres.append(0.5 * (edges[b] + edges[b + 1]))
                    medians.append(float(np.median(in_bin)))
            if centres:
                ax.plot(centres, medians, "-", lw=1.8, color=self.colors[name],
                        zorder=3, label=f"{_short(name)} median")

        ax.set_xlabel("time in recording (s)", fontsize=8)
        ax.set_ylabel("event amplitude (pA)", fontsize=8)
        ax.set_title("Amplitude against time\n(line = binned median; a sag is run-down)",
                     fontsize=9.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.2)
        if events_by_cell:
            ax.legend(fontsize=7, loc="best", framealpha=0.9)

    # -- tab 3: traces --------------------------------------------------
    def _traces_tab(self) -> QtWidgets.QWidget:
        fig = self._traces_figure()
        page = self._wrap(fig)
        # One row per cell, so the figure grows past the window: scroll it
        # rather than squeezing ten cells into 800 px.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        # Tie the scrollable height to the screen's real DPI, not a guessed 96
        page.setMinimumHeight(int(fig.get_figheight() * max(72, self.logicalDpiY())))
        scroll.setWidget(page)
        return scroll

    # Left to right, the order the cell passes through them as the injected
    # current grows: the first sweep that fires at all, the one firing at half
    # its maximum rate, and the maximum itself. The half-max sweep is the one
    # worth looking hardest at -- it is on the steep part of the F-I curve,
    # where a cell that is unhealthy or badly clamped shows it first.
    # (sweep column, label, current column, second number, its format)
    TRACE_COLUMNS = [
        ("rheobase_sweep", "rheobase", "rheobase_pA", None, ""),
        ("half_max_sweep", "half-max firing", "half_max_stim_amp_pA",
         "half_max_n_spikes", "{:.0f} APs"),
        ("max_firing_sweep", "max firing", "max_firing_rate_at_pA",
         "max_firing_rate_Hz", "{:.0f} Hz"),
    ]

    def _traces_figure(self) -> Figure:
        rows = max(1, len(self.names))
        ncols = len(self.TRACE_COLUMNS)
        fig = Figure(figsize=(12, 2.6 * rows), tight_layout=True)
        for i, name in enumerate(self.names):
            row = self.summary.iloc[i]
            abf_path = self.run["abf"].get(name)
            axes = [fig.add_subplot(rows, ncols, ncols * i + k + 1) for k in range(ncols)]
            for ax in axes:
                ax.set_xlabel("time (s)", fontsize=8)
                ax.set_ylabel("Vm (mV)", fontsize=8)
                ax.tick_params(labelsize=7.5)
            if abf_path is None:
                axes[0].text(0.5, 0.5, f"{name}: recording not found",
                             ha="center", va="center", transform=axes[0].transAxes,
                             color="#b00020")
                continue
            try:
                import pyabf
                abf = pyabf.ABF(str(abf_path))
                channel = next((c for c in abf.channelList
                                if abf.adcUnits[c].strip().lower() == "mv"), 0)
            except Exception as exc:                     # pragma: no cover
                axes[0].text(0.5, 0.5, f"{name}: {exc}", ha="center", va="center",
                             transform=axes[0].transAxes, color="#b00020")
                continue

            for ax, (key, what, amp_key, extra_key, extra_fmt) in zip(axes, self.TRACE_COLUMNS):
                sw = _num(row, key)
                if not np.isfinite(sw):
                    ax.text(0.5, 0.5, f"no {what} sweep", ha="center", va="center",
                            transform=ax.transAxes, color="#777777")
                    continue
                abf.setSweep(int(sw), channel=channel)
                ax.axhline(0.0, color="#bbbbbb", lw=0.8, ls=":", zorder=1)
                ax.plot(abf.sweepX, abf.sweepY, lw=0.6, color=self.colors[name], zorder=2)
                amp = _num(row, amp_key)
                # Each panel carries the number that defines it: the current
                # for rheobase, the spike count for half-max, the rate for the
                # maximum -- so the three are comparable without the F-I tab.
                bits = [f"{amp:.0f} pA" if np.isfinite(amp) else "?"]
                extra = _num(row, extra_key) if extra_key else float("nan")
                if np.isfinite(extra):
                    bits.append(extra_fmt.format(extra))
                # Three panels across leave no room for the repeated YYMMDD_,
                # which the Cell health tab drops from its tick labels too.
                ax.set_title(f"{_short(name)}  -  {what} sweep {int(sw)} "
                             f"({', '.join(bits)})", fontsize=8.5)
        return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def show_qc_window(root: Path, summary: Optional[pd.DataFrame] = None,
                   block: bool = True) -> Optional[QCWindow]:
    """Open the QC window for a finished run. Blocks until it is closed when
    `block`, so a command-line run ends when the user is done looking."""
    run = load_run(Path(root), summary)
    app = qt_app()
    win = QCWindow(run)
    win.show()
    win.raise_()
    win.activateWindow()
    if block:
        app.exec_()
    else:
        QtCore.QCoreApplication.processEvents()
    return win


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python batch_qc_window.py DATA_ROOT")
    show_qc_window(Path(sys.argv[1]))
