"""
Build a hand-labeled sEPSC training set by clicking event peaks directly on
the raw trace. Meant for collecting enough real examples of two kinetically
distinct populations (a fast-decaying one and a slower one) to train/fine-
tune an event classifier -- NOT for exhaustively annotating a whole
recording (use review.py to vet miniML's own detections instead).

You page through the gap-free trace a few seconds at a time. Clicking near
an event peak does NOT record the raw click position -- it snaps to the
true local extremum, estimates a local baseline, and walks outward to a
rough onset (fast rise) and decay-back point (slower decay) using simple
threshold crossings. A click that doesn't land near a real deflection
(amplitude below --min-amp) is ignored.

    LEFT click  = fast-decaying population
    RIGHT click = slow-decaying population

Because the two populations decay on different timescales, the saved
window length is class-specific (see --post-ms-fast / --post-ms-slow) so
fast events aren't padded with flat baseline and slow events aren't
truncated mid-decay.

Output (written next to the source .abf, autosaved after every click):
    <name>_training_events.csv    one row per labeled event: label, peak
                                   time/index, baseline, amplitude, rough
                                   onset/decay timing
    <name>_training_windows.npz   raw current snippets, class-specific
                                   fixed length, keyed 'fast_windows' /
                                   'slow_windows' (each shape
                                   [n_events, window_len]), plus
                                   'fast_dt'/'slow_dt' (the sample interval)
                                   for reconstructing a time axis later

Usage
-----
    python -m sepsc.label path\\to\\recording.abf
    python -m sepsc.label recording.abf --chunk-s 3 --min-amp 8

Controls: left click = label fast event, right click = label slow event,
Undo button (or 'z') removes the last labeled event, Prev/Next (or
PageUp/PageDown) page through the trace. X Zoom +/- buttons (mouse scroll,
or +/- keys) change how much TIME is visible, centered on the cursor when
scrolling -- useful for seeing an event's time course before you click it.
Y Zoom +/- buttons (or Up/Down keys) change the visible AMPLITUDE range;
unlike the x-axis, the y-axis never autoscales on its own, so the trace
doesn't jump vertically while you pan/zoom in time.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyabf
from scipy.ndimage import uniform_filter1d

import matplotlib
matplotlib.rcParams["toolbar"] = "None"  # raw clicks must reach our handler, not pan/zoom
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from .gui_utils import safe_callback
from .style import COLORS, MARKERS

LABEL_BUTTONS = {1: "fast", 3: "slow"}  # matplotlib mouse button codes: 1=left, 3=right
EVENT_COLUMNS = ["label", "peak_idx", "peak_time_s", "onset_time_s", "decay_time_s",
                  "rise_ms", "decay_ms", "baseline", "amplitude"]


def analyze_click(v: np.ndarray, v_smooth: np.ndarray, dt: float, click_idx: int,
                   direction: str, snap_ms: float, baseline_ms: float, onset_frac: float,
                   decay_frac: float, max_decay_ms: float, min_amp: float):
    """Snap a rough click to the true peak and characterize a rough onset /
    decay-back point. Returns None if nothing event-like is near the click.

    Peak/baseline/onset/decay are all located on `v_smooth` (a lightly
    boxcar-smoothed copy of the trace) rather than raw `v`, so a single
    noisy sample near the click doesn't get mistaken for the peak the way
    it would on raw 100 kHz data -- this mirrors miniML's own use of a
    smoothing convolution before peak-picking. `v` is only used by the
    caller afterwards, to slice out the actual raw window to save.

    All threshold logic is done on a polarity-normalized copy of the
    smoothed trace (`sv`, always "peak = large positive") so the same code
    handles inward (negative-going) and outward (positive-going) events
    without branching.
    """
    pol = -1.0 if direction == "negative" else 1.0
    sv = pol * v_smooth

    snap_n = max(1, int(round(snap_ms / 1000.0 / dt)))
    i0, i1 = max(0, click_idx - snap_n), min(len(v_smooth), click_idx + snap_n + 1)
    if i1 - i0 < 2:
        return None
    peak_idx = i0 + int(np.argmax(sv[i0:i1]))

    baseline_n = max(1, int(round(baseline_ms / 1000.0 / dt)))
    b1 = max(0, peak_idx - snap_n)
    b0 = max(0, b1 - baseline_n)
    if b1 <= b0:
        return None
    baseline = float(np.median(v_smooth[b0:b1]))
    peak_v = float(v_smooth[peak_idx])
    amplitude = peak_v - baseline  # signed: negative for inward events
    if abs(amplitude) < min_amp:
        return None

    s_baseline, s_peak = pol * baseline, pol * peak_v
    onset_level = s_baseline + onset_frac * (s_peak - s_baseline)
    decay_level = s_baseline + decay_frac * (s_peak - s_baseline)

    max_onset_n = baseline_n + snap_n
    onset_idx = max(0, peak_idx - max_onset_n)
    for k in range(peak_idx, max(0, peak_idx - max_onset_n) - 1, -1):
        if sv[k] < onset_level:
            onset_idx = k
            break

    max_decay_n = max(1, int(round(max_decay_ms / 1000.0 / dt)))
    decay_idx = min(len(v_smooth) - 1, peak_idx + max_decay_n)
    for k in range(peak_idx, min(len(v_smooth), peak_idx + max_decay_n)):
        if sv[k] < decay_level:
            decay_idx = k
            break

    return dict(peak_idx=peak_idx, onset_idx=onset_idx, decay_idx=decay_idx,
                baseline=baseline, amplitude=amplitude)


class LabelSession:
    def __init__(self, abf: pyabf.ABF, channel: int, chunk_s: float,
                 direction: str, snap_ms: float, baseline_ms: float,
                 onset_frac: float, decay_frac: float,
                 pre_ms: float, post_ms_fast: float, post_ms_slow: float,
                 min_amp: float, min_sep_ms: float, smooth_samples: int,
                 min_chunk_s: float, max_chunk_s: float,
                 csv_path: str, npz_path: str):
        abf.setSweep(0, channel=channel)
        self.t = np.asarray(abf.sweepX, float)
        self.v = np.asarray(abf.sweepY, float)
        self.v_smooth = uniform_filter1d(self.v, size=max(1, smooth_samples))
        self.y_unit = abf.adcUnits[channel]
        self.dt = self.t[1] - self.t[0]
        self.duration_s = float(self.t[-1])

        self.chunk_s = chunk_s
        self.direction = direction
        self.snap_ms, self.baseline_ms = snap_ms, baseline_ms
        self.onset_frac, self.decay_frac = onset_frac, decay_frac
        self.pre_ms = pre_ms
        self.post_ms = {"fast": post_ms_fast, "slow": post_ms_slow}
        self.min_amp = min_amp
        self.min_sep_ms = min_sep_ms
        self.csv_path, self.npz_path = csv_path, npz_path

        self.events: list[dict] = []       # metadata, one dict per labeled event
        self.windows: dict[str, list] = {"fast": [], "slow": []}
        self._load_existing()

        self.view_start = 0.0
        self.min_chunk_s = min_chunk_s
        self.max_chunk_s = min(max_chunk_s, self.duration_s)
        self.zoom_factor = 1.5
        self._busy = False

        # fixed y-axis range: set once here, changed only by the Y Zoom
        # buttons -- NOT re-derived from whatever happens to be on screen,
        # so panning/zooming in time doesn't make the trace jump vertically
        y_lo, y_hi = np.percentile(self.v, [0.5, 99.5])
        pad = 0.1 * (y_hi - y_lo)
        self.y_lo, self.y_hi = float(y_lo - pad), float(y_hi + pad)
        self.min_y_range = 2.0  # pA; guards against zooming the y-axis to nothing

        self.fig, self.ax = plt.subplots(figsize=(11, 6.2))
        self.fig.canvas.manager.set_window_title("sEPSC training-event labeler")
        plt.subplots_adjust(bottom=0.28)

        # row 1 (bottom): navigation + undo
        ax_prev = plt.axes([0.03, 0.05, 0.14, 0.075])
        ax_undo = plt.axes([0.43, 0.05, 0.14, 0.075])
        ax_next = plt.axes([0.83, 0.05, 0.14, 0.075])
        self.btn_prev = Button(ax_prev, "<< Prev")
        self.btn_undo = Button(ax_undo, "Undo", color="#f2d0a0", hovercolor="#e5b873")
        self.btn_next = Button(ax_next, "Next >>")
        self.btn_prev.on_clicked(self.prev_chunk)
        self.btn_undo.on_clicked(self.undo)
        self.btn_next.on_clicked(self.next_chunk)

        # row 2 (above it): time (X) zoom and amplitude (Y) zoom, kept apart
        # so they're never confused for each other
        ax_xzoomout = plt.axes([0.05, 0.15, 0.14, 0.06])
        ax_xzoomin = plt.axes([0.22, 0.15, 0.14, 0.06])
        ax_yzoomout = plt.axes([0.64, 0.15, 0.14, 0.06])
        ax_yzoomin = plt.axes([0.81, 0.15, 0.14, 0.06])
        self.btn_xzoomout = Button(ax_xzoomout, "X Zoom −")
        self.btn_xzoomin = Button(ax_xzoomin, "X Zoom +")
        self.btn_yzoomout = Button(ax_yzoomout, "Y Zoom −")
        self.btn_yzoomin = Button(ax_yzoomin, "Y Zoom +")
        self.btn_xzoomout.on_clicked(self.zoom_out)
        self.btn_xzoomin.on_clicked(self.zoom_in)
        self.btn_yzoomout.on_clicked(self.y_zoom_out)
        self.btn_yzoomin.on_clicked(self.y_zoom_in)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        self.render()

    # -- persistence ------------------------------------------------------
    def _load_existing(self):
        if not os.path.exists(self.csv_path):
            return
        try:
            df = pd.read_csv(self.csv_path)
        except pd.errors.EmptyDataError:
            return  # a prior 0-event session wrote a headerless file; nothing to resume
        self.events = df.to_dict("records")
        if os.path.exists(self.npz_path):
            npz = np.load(self.npz_path)
            for label in ("fast", "slow"):
                key = f"{label}_windows"
                if key in npz:
                    self.windows[label] = [row for row in npz[key]]
        n_fast = sum(e["label"] == "fast" for e in self.events)
        n_slow = sum(e["label"] == "slow" for e in self.events)
        print(f"Resuming: {len(self.events)} events already labeled "
              f"({n_fast} fast, {n_slow} slow) from {os.path.basename(self.csv_path)}.",
              flush=True)

    def _save(self):
        df = pd.DataFrame(self.events) if self.events else pd.DataFrame(columns=EVENT_COLUMNS)
        df.to_csv(self.csv_path, index=False)
        payload = {}
        for label in ("fast", "slow"):
            if self.windows[label]:
                payload[f"{label}_windows"] = np.stack(self.windows[label])
                payload[f"{label}_dt"] = np.array(self.dt)
        np.savez(self.npz_path, **payload)

    # -- click -> labeled event --------------------------------------------
    @safe_callback
    def on_click(self, event):
        if self._busy or event.inaxes != self.ax or event.button not in LABEL_BUTTONS:
            return
        label = LABEL_BUTTONS[event.button]
        click_idx = int(round(event.xdata / self.dt))
        if not (0 <= click_idx < len(self.v)):
            return

        self._busy = True
        try:
            result = analyze_click(
                self.v, self.v_smooth, self.dt, click_idx, self.direction,
                self.snap_ms, self.baseline_ms, self.onset_frac, self.decay_frac,
                max_decay_ms=self.post_ms[label], min_amp=self.min_amp,
            )
            if result is None:
                print(f"  no event found near t={event.xdata:.3f}s -- ignored", flush=True)
                return

            peak_idx = result["peak_idx"]
            min_sep_n = int(round(self.min_sep_ms / 1000.0 / self.dt))
            if any(abs(e["peak_idx"] - peak_idx) < min_sep_n for e in self.events):
                print(f"  event at t={peak_idx * self.dt:.3f}s already labeled -- ignored", flush=True)
                return

            pre_n = int(round(self.pre_ms / 1000.0 / self.dt))
            post_n = int(round(self.post_ms[label] / 1000.0 / self.dt))
            w0, w1 = peak_idx - pre_n, peak_idx + post_n
            if w0 < 0 or w1 > len(self.v):
                print(f"  event at t={peak_idx * self.dt:.3f}s too close to recording edge -- ignored", flush=True)
                return

            self.events.append(dict(
                label=label, peak_idx=peak_idx, peak_time_s=peak_idx * self.dt,
                onset_time_s=result["onset_idx"] * self.dt,
                decay_time_s=result["decay_idx"] * self.dt,
                rise_ms=(peak_idx - result["onset_idx"]) * self.dt * 1e3,
                decay_ms=(result["decay_idx"] - peak_idx) * self.dt * 1e3,
                baseline=result["baseline"], amplitude=result["amplitude"],
            ))
            self.windows[label].append(self.v[w0:w1].copy())
            self._save()
            self.render()
        finally:
            self._busy = False

    @safe_callback
    def undo(self, _event):
        if not self.events:
            return
        removed = self.events.pop()
        self.windows[removed["label"]].pop()
        self._save()
        print(f"Undid last event: {removed['label']} @ t={removed['peak_time_s']:.3f}s", flush=True)
        self.render()

    # -- navigation ---------------------------------------------------------
    def _clamp_view(self):
        max_start = max(0.0, self.duration_s - self.chunk_s)
        self.view_start = min(max(0.0, self.view_start), max_start)

    @safe_callback
    def prev_chunk(self, _event):
        self.view_start -= self.chunk_s
        self._clamp_view()
        self.render()

    @safe_callback
    def next_chunk(self, _event):
        self.view_start += self.chunk_s
        self._clamp_view()
        self.render()

    # -- zoom (visible-duration control) -------------------------------------
    def _zoom(self, factor: float, center_s: float | None = None):
        if center_s is None:
            center_s = self.view_start + self.chunk_s / 2.0
        self.chunk_s = float(np.clip(self.chunk_s * factor, self.min_chunk_s, self.max_chunk_s))
        self.view_start = center_s - self.chunk_s / 2.0
        self._clamp_view()
        self.render()

    @safe_callback
    def zoom_in(self, _event):
        self._zoom(1.0 / self.zoom_factor)

    @safe_callback
    def zoom_out(self, _event):
        self._zoom(self.zoom_factor)

    # -- zoom (amplitude / y-axis control) -----------------------------------
    def _y_zoom(self, factor: float):
        center = (self.y_lo + self.y_hi) / 2.0
        half_range = max(self.min_y_range, (self.y_hi - self.y_lo) / 2.0 * factor)
        self.y_lo, self.y_hi = center - half_range, center + half_range
        self.render()

    @safe_callback
    def y_zoom_in(self, _event):
        self._y_zoom(1.0 / self.zoom_factor)

    @safe_callback
    def y_zoom_out(self, _event):
        self._y_zoom(self.zoom_factor)

    @safe_callback
    def on_scroll(self, scroll_event):
        if scroll_event.inaxes != self.ax or scroll_event.xdata is None:
            return
        factor = (1.0 / self.zoom_factor) if scroll_event.button == "up" else self.zoom_factor
        self._zoom(factor, center_s=scroll_event.xdata)

    @safe_callback
    def on_key(self, key_event):
        if key_event.key in ("pageup", "left"):
            self.prev_chunk(key_event)
        elif key_event.key in ("pagedown", "right"):
            self.next_chunk(key_event)
        elif key_event.key == "z":
            self.undo(key_event)
        elif key_event.key in ("+", "="):
            self.zoom_in(key_event)
        elif key_event.key in ("-", "_"):
            self.zoom_out(key_event)
        elif key_event.key == "up":
            self.y_zoom_in(key_event)
        elif key_event.key == "down":
            self.y_zoom_out(key_event)

    # -- drawing --------------------------------------------------------------
    def render(self):
        i0 = max(0, int(self.view_start / self.dt))
        i1 = min(len(self.v), int((self.view_start + self.chunk_s) / self.dt))

        self.ax.clear()
        self.ax.plot(self.t[i0:i1], self.v[i0:i1], color="k", lw=0.6)
        self.ax.set_xlim(self.t[i0], self.t[max(i0, i1 - 1)])
        self.ax.set_ylim(self.y_lo, self.y_hi)  # fixed -- never autoscaled, only Y Zoom changes this
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel(f"current ({self.y_unit})")

        for e in self.events:
            if i0 <= e["peak_idx"] < i1:
                self.ax.plot(e["peak_time_s"], self.v[e["peak_idx"]],
                              marker=MARKERS[e["label"]], color=COLORS[e["label"]],
                              ms=9, mec="k", mew=0.6, zorder=5)

        n_fast = sum(e["label"] == "fast" for e in self.events)
        n_slow = sum(e["label"] == "slow" for e in self.events)
        # more decimals once the visible window is short enough that .1f would
        # round start/end to the same value
        prec = 1 if self.chunk_s >= 1.0 else (3 if self.chunk_s >= 0.02 else 4)
        self.ax.set_title(
            f"t = {self.view_start:.{prec}f}-{self.view_start + self.chunk_s:.{prec}f}s "
            f"of {self.duration_s:.1f}s  (X window {self.chunk_s * 1e3:.0f} ms, "
            f"Y range {self.y_hi - self.y_lo:.0f} {self.y_unit})   |   "
            f"LEFT-click = fast (▲ {n_fast})   RIGHT-click = slow (■ {n_slow})   "
            f"total {len(self.events)}"
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    @safe_callback
    def on_close(self, _event):
        n_fast = sum(e["label"] == "fast" for e in self.events)
        n_slow = sum(e["label"] == "slow" for e in self.events)
        print(f"Session ended: {len(self.events)} events labeled ({n_fast} fast, {n_slow} slow).", flush=True)
        print(f"Saved -> {self.csv_path}", flush=True)
        print(f"Saved -> {self.npz_path}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", help="Path to the gap-free voltage-clamp .abf file")
    p.add_argument("--channel", type=int, default=0, help="ADC channel to display (default 0)")
    p.add_argument("--direction", choices=["negative", "positive"], default="negative",
                   help="event polarity; sEPSCs in voltage clamp are inward/negative (default)")
    p.add_argument("--chunk-s", type=float, default=4.0, help="visible trace window, in seconds (default 4)")
    p.add_argument("--snap-ms", type=float, default=4.0,
                   help="+/- window around the click to search for the true peak, in ms (default 4)")
    p.add_argument("--baseline-ms", type=float, default=15.0,
                   help="window just before the rise used to estimate baseline, in ms (default 15)")
    p.add_argument("--onset-frac", type=float, default=0.1,
                   help="fraction of peak amplitude (from baseline) marking rise onset (default 0.1)")
    p.add_argument("--decay-frac", type=float, default=0.25,
                   help="fraction of peak amplitude (from baseline) marking decay end (default 0.25)")
    p.add_argument("--pre-ms", type=float, default=10.0,
                   help="baseline/rise captured before the peak in saved windows, in ms (default 10)")
    p.add_argument("--post-ms-fast", type=float, default=25.0,
                   help="decay captured after the peak for FAST events, in ms (default 25)")
    p.add_argument("--post-ms-slow", type=float, default=80.0,
                   help="decay captured after the peak for SLOW events, in ms (default 80)")
    p.add_argument("--min-amp", type=float, default=5.0,
                   help="minimum |amplitude| (pA) to accept a click as a real event (default 5)")
    p.add_argument("--min-sep-ms", type=float, default=2.0,
                   help="ignore a click landing within this many ms of an already-labeled peak (default 2)")
    p.add_argument("--smooth-samples", type=int, default=15,
                   help="boxcar smoothing window (samples) used only for peak/onset/decay "
                        "detection, not for the saved raw windows (default 15, matches "
                        "sepsc.detect's convolve_win)")
    p.add_argument("--min-chunk-s", type=float, default=0.02,
                   help="shortest visible window zooming in can reach, in seconds (default 0.02 = 20ms)")
    p.add_argument("--max-chunk-s", type=float, default=20.0,
                   help="longest visible window zooming out can reach, in seconds (default 20)")
    args = p.parse_args(argv)

    stem = os.path.splitext(args.abf)[0]
    csv_path = f"{stem}_training_events.csv"
    npz_path = f"{stem}_training_windows.npz"

    abf = pyabf.ABF(args.abf)
    print(f"Labeling training events in {os.path.basename(args.abf)} "
          f"({abf.sweepLengthSec:.1f}s @ {abf.dataRate:.0f} Hz)", flush=True)
    print("LEFT click = fast event, RIGHT click = slow event. "
          "Undo (or 'z') removes the last one. X Zoom +/- (or scroll) changes the time "
          "window; Y Zoom +/- (or Up/Down) changes the amplitude range -- the y-axis "
          "is fixed and never autoscales on its own.", flush=True)

    LabelSession(
        abf, args.channel, args.chunk_s, args.direction,
        args.snap_ms, args.baseline_ms, args.onset_frac, args.decay_frac,
        args.pre_ms, args.post_ms_fast, args.post_ms_slow,
        args.min_amp, args.min_sep_ms, args.smooth_samples,
        args.min_chunk_s, args.max_chunk_s, csv_path, npz_path,
    )
    plt.show()


if __name__ == "__main__":
    main()
