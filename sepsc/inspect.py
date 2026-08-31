"""
Interactive click-to-verify/curate tool for ANY of the three sEPSC
detectors this pipeline has, chosen with --source (default minianalysis):

`--source minianalysis` (the classical Mini-Analysis-style detector,
minianalysis.py): click a detected event and see, drawn directly on that
event's own trace, every window/threshold the 6-step detection sequence
actually used to accept it -- the Mini Analysis Program tutorial's own
"Optimizing Detection Parameters" workflow (this_method.pdf p.4-5: "First
use mouse-click to detect... Examine the location of the X's and dots...
Examine the amplitude and area") applied to THIS reimplementation. Shows,
both as text and as annotated spans/lines on the trace, all 8 of the
tutorial's named detection-window parameters (PDF p.2) plus the rise-side
onset search window (this reimplementation's own addition, mirroring (f) on
the decay side): Amplitude threshold (a), Area threshold (b), Number of
points to average peak, Period to search local maximum (c), Time before
peak for baseline (d), Period to average baseline (e), Period to search
decay time (f), Fraction to find decay (g), Period to search onset before
peak. Reads the exact DetectionParams a `python -m sepsc minianalysis` run
used from its `<stem>_minianalysis_params.json` sidecar (saved
automatically by that command) -- so what you see here is guaranteed to
match what actually produced the events CSV, not just today's CLI
defaults. If no sidecar is found (events from an older run), falls back to
DetectionParams()'s defaults, overridable with the same
--amplitude-threshold/--area-threshold/etc. flags minianalysis.py itself
takes.

`--source miniml` (the CNN-LSTM deep-learning detector, detect.py): no
windowed-detection-parameter concept to annotate (there's no baseline/
onset/decay search window the way the classical detector has -- the model
just outputs a peak location), so the detail view is simpler: the raw
snippet around the model's own peak location, plus whichever of
score/amplitude/charge/risetime/decaytime/halfwidth are present in that
event's row (see sepsc.detect's own *_miniML_individual.csv columns).

`--source fastmini` (the per-recording MLP peel-off detector, fastmini.py):
also no baseline-averaging window, local-max search span, or amplitude/area
THRESHOLD to annotate (its "threshold" is the peel-off confidence curve's
own peak-prominence cutoff, not a per-event window) -- but unlike miniml,
its *_fastmini_events.csv DOES persist a real baseline plus rise_time_ms/
decay_time_ms/area for every event (see fastmini.py's own _measure_event),
so the onset/decay markers and shaded area ARE reconstructed exactly (peak
-+ rise_time_ms/decay_time_ms), just without the search-window geometry
around them. fastmini.py always preprocesses internally at a fixed 3 kHz
Bessel / 10 kHz resample (no raw-trace mode, no --filter of its own) -- this
tool loads that same fixed trace automatically for this source, ignoring
--filter/--cutoff-hz/--target-rate-hz/--filter-order (a NOTE is printed if
you pass them anyway).

Also doubles as a QC/curation tool: Accept/Reject each event (buttons or
keys), or Accept All Remaining in one click, same accepted/rejected output
as sepsc.review -- `<stem>_<source>_reviewed.csv` (accepted events, same
schema review.py writes) and `<stem>_<source>_review_progress.csv` (every
decided event, with a `decision` column: 1=accepted, 0=rejected) --
autosaved after every decision, and resumed automatically if either file
already exists (from a prior run of THIS tool or of sepsc.review, for
whichever --source -- the two are fully interchangeable). Events you
haven't decided on yet just don't appear in the progress file; they aren't
"no" by default.

Usage
-----
    python -m sepsc.inspect path\\to\\recording.abf
    python -m sepsc.inspect recording.abf --csv custom_minianalysis_events.csv
    python -m sepsc.inspect recording.abf --amplitude-threshold 8  # override one param
    python -m sepsc.inspect recording.abf --filter  # if minianalysis.py was also run with --filter
        # (10 kHz resample + 3 kHz Bessel low-pass by default -- see sepsc.preprocess); MUST match
        # whatever --filter settings that detection run used, since location values in the events
        # CSV are indices into that filtered/resampled trace, not the raw one
    python -m sepsc.inspect recording.abf --source miniml  # detect.py's output instead
    python -m sepsc.inspect recording.abf --source fastmini  # fastmini.py's output instead

Controls:
    click a red X in the top overview -- inspect that event
    Right / N            -- next event
    Left / P              -- previous event
    A / Up                -- accept this event, then advance
    R / Down              -- reject this event, then advance
    "Accept All Remaining" button -- accept every still-undecided event
    scroll wheel over the top (full-trace) panel -- zoom in/out, centered
        on the cursor; time (X) and current (Y) together by default, hold
        Ctrl to zoom time only or Shift to zoom current only
    left-drag inside the top panel -- pan (both X and Y together); no
        toolbar tool needs to be selected first, and a plain click (no
        drag) still selects the nearest event as usual
    toolbar's Pan/Zoom-rectangle buttons -- optional, for the toolbar's own
        constrained/box-zoom behavior on the top panel (hold x or y while
        dragging to pan or zoom just one axis); while either is toggled on
        it takes over the drag instead of the always-on pan above. Home
        resets to the original view; Back/Forward step through previous
        views.
    The bottom (per-event detail) panel has no independent pan/zoom of its
        own -- it's always redrawn fresh, auto-fit around the selected
        peak, exactly as before.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import numpy as np
import pandas as pd
import pyabf

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from .gui_utils import safe_callback
from .minianalysis import DetectionParams, _samples
from .preprocess import DEFAULT_CUTOFF_HZ, DEFAULT_ORDER, DEFAULT_TARGET_RATE_HZ, get_hardware_filter_hz, load_filtered_trace
from .style import COLORS, GRID, INK, MUTED, SURFACE, TRACE

# Deliberately NOT `from .review import SOURCES`: review.py is now also
# built directly on PyQt5/Figure (see the comment below), so importing it
# no longer risks the pyplot-backend-clobbering problem that comment
# describes -- but it would still pull in review.py's whole argparse/
# QApplication-construction code for the sake of a handful of filename
# strings. Hardcoded here instead. NOTE the asymmetry with review.py's OWN
# SOURCES dict: that one only ever has "minianalysis"/"miniml" (compare.py
# and view.py both hardcode that 2-source assumption -- see their own
# SOURCE_STYLE_KEY/overrides dicts -- so it can't grow a third "fastmini"
# key without updating those too, out of scope here); review.py imports
# THIS dict instead for its own --source fastmini support, since it (via
# EventInspector) doesn't share that 2-source assumption. Still MUST stay
# in sync with review.py's SOURCES dict on the two keys they share
# (out_suffix/progress_suffix here are individual_suffix/out_suffix there),
# and with the schema _kept()/_all_reviewed_with_decisions() there
# write (location/location_s columns, decision 1/0) -- see
# _autosave/_load_prior_decisions below.
SOURCE_CFG = {
    "minianalysis": dict(
        label="Mini-Analysis-style",
        events_suffix="_minianalysis_events.csv",
        reviewed_suffix="_minianalysis_reviewed.csv",
        progress_suffix="_minianalysis_review_progress.csv",
    ),
    "miniml": dict(
        label="miniML",
        events_suffix="_miniML_individual.csv",
        reviewed_suffix="_miniML_reviewed.csv",
        progress_suffix="_miniML_review_progress.csv",
    ),
    "fastmini": dict(
        label="fastmini (MLP peel-off)",
        events_suffix="_fastmini_events.csv",
        reviewed_suffix="_fastmini_reviewed.csv",
        progress_suffix="_fastmini_review_progress.csv",
    ),
}

# Built directly on a PyQt5 QApplication/QMainWindow -- like view.py -- via
# matplotlib's Qt Figure/FigureCanvas, NOT pyplot's own TkAgg mainloop:
# minianalysis.py forces matplotlib.use("Agg") at its own module level (for
# its headless CLI plotting), and since pyplot's backend is a single global,
# importing it here (for DetectionParams/_samples) silently downgrades
# ANY pyplot-based window in this same process back to the non-interactive
# Agg backend regardless of import order. Building the Qt window explicitly
# sidesteps that global entirely.

OVERVIEW_MAX_POINTS = 200_000  # stride-decimated for display speed only; detail view always uses full-res data


def _load_params(params_json: str | None, stem: str, overrides: dict) -> DetectionParams:
    """Sidecar JSON (if present) as the base, with any explicitly-passed
    CLI flags (non-None in `overrides`) applied on top -- see module
    docstring."""
    path = params_json or f"{stem}_minianalysis_params.json"
    if os.path.exists(path):
        with open(path) as fh:
            base = DetectionParams(**json.load(fh))
        print(f"Loaded detection parameters -> {path}")
    else:
        base = DetectionParams()
        print(f"No params sidecar found ({path}) -- using DetectionParams() defaults "
              f"(pass --amplitude-threshold etc. to override, or re-run minianalysis to "
              f"regenerate the sidecar).")
    explicit = {k: v for k, v in overrides.items() if v is not None}
    return dataclasses.replace(base, **explicit) if explicit else base


def _load_miniml_events(csv_path: str, t: np.ndarray) -> pd.DataFrame:
    """*_miniML_individual.csv (rows=features, cols=event_N, as written by
    miniML's EventDetection.save_to_csv: location/score/amplitude/charge/
    risetime/decaytime/halfwidth/interval) as a tidy one-row-per-event
    DataFrame on the same peak_idx/peak_time_s columns the rest of this
    module uses -- mirrors sepsc.review.load_miniml_events, reimplemented
    locally rather than imported (see the module-level comment above
    SOURCE_CFG for why). peak_time_s is read directly from `t` at each
    peak_idx (not recomputed from a sample rate) so it's automatically
    correct whether `t` is the raw or --filter'd/resampled trace."""
    raw = pd.read_csv(csv_path, index_col=0)
    events = raw.T.reset_index(drop=True).rename(columns={"location": "peak_idx"})
    events["peak_idx"] = events["peak_idx"].astype(int)
    events["peak_time_s"] = t[events["peak_idx"].to_numpy()]
    return events


def _load_fastmini_events(csv_path: str, t: np.ndarray) -> pd.DataFrame:
    """*_fastmini_events.csv (already tidy: location/location_s/baseline/
    amplitude/rise_time_ms/decay_time_ms/area -- see fastmini.py's own
    _measure_event/main()) onto the same peak_idx/peak_time_s columns the
    rest of this module uses. peak_time_s is read directly from `t` (like
    _load_miniml_events), not trusted from the CSV's own location_s --
    same reasoning, kept uniform across all three sources even though
    fastmini.py computes location_s identically (peak_location * dt
    against the same fixed trace this module always loads for this
    source -- see main()'s --source fastmini handling)."""
    events = pd.read_csv(csv_path).rename(columns={"location": "peak_idx", "location_s": "peak_time_s"})
    events["peak_idx"] = events["peak_idx"].astype(int)
    events["peak_time_s"] = t[events["peak_idx"].to_numpy()]
    return events


def load_source_events(csv_path: str, source: str, t: np.ndarray) -> pd.DataFrame:
    """SOURCE_CFG-dispatching event loader -- minianalysis's own CSV is
    already tidy with peak_idx/peak_time_s columns (pd.read_csv as-is);
    miniml's needs _load_miniml_events' transpose; fastmini's needs
    _load_fastmini_events' location->peak_idx rename. Shared by main()
    (below) and by sepsc.review, which reuses this EventInspector window
    wholesale (see review.py's own module docstring) rather than
    re-loading events its own way."""
    if source == "miniml":
        return _load_miniml_events(csv_path, t)
    if source == "fastmini":
        return _load_fastmini_events(csv_path, t)
    return pd.read_csv(csv_path)


def _event_geometry(v: np.ndarray, dt: float, peak_idx: int, baseline: float, amplitude: float,
                     params: DetectionParams) -> dict:
    """Recompute the index boundaries detect_events' steps 2/4/5 used for
    this peak (b0/b1 baseline window, p0/p1 peak-average window, onset_idx,
    decay_idx, and an illustrative local-max search span around the peak).

    Deliberately takes `baseline`/`amplitude` from the event's own CSV row
    rather than recomputing them: those are the exact (possibly
    overlap-adjusted, see minianalysis._overlap_adjusted_baseline) values
    actually used at detection time, and re-deriving them here without that
    same sequential state could silently disagree with what really produced
    this event. Only the WINDOW GEOMETRY is recomputed -- purely a function
    of peak_idx/params, safe to redo standalone.
    """
    pol = -1.0 if params.direction == "negative" else 1.0
    before_n = _samples(params.baseline_before_ms, dt)
    avg_n = _samples(params.baseline_avg_ms, dt)
    decay_n = _samples(params.decay_search_ms, dt)
    onset_search_n = _samples(params.onset_search_ms, dt)
    search_n = _samples(params.search_local_max_ms, dt)
    n_avg_peak = max(1, params.n_avg_peak)

    b0 = max(0, peak_idx - before_n - avg_n)
    b1 = max(0, peak_idx - before_n)
    p0 = max(0, peak_idx - n_avg_peak // 2)
    p1 = min(len(v), p0 + n_avg_peak)

    peak_v = baseline + amplitude
    s_baseline, s_peak = pol * baseline, pol * peak_v
    span = s_peak - s_baseline
    sv = pol * v

    # Mirrors detect_events' own onset search exactly (peak_idx -
    # onset_search_n : peak_idx), NOT capped at b1 -- an accepted event is
    # guaranteed to have a real crossing in this window, so the fallback
    # (onset_search_start) only matters for degenerate/edge cases.
    onset_level = s_baseline + params.onset_fraction * span
    onset_search_start = max(0, peak_idx - onset_search_n)
    onset_idx = onset_search_start
    for k in range(peak_idx, onset_search_start - 1, -1):
        if sv[k] < onset_level:
            onset_idx = k
            break

    decay_level = s_baseline + (1.0 - params.decay_fraction) * span
    decay_search_end = min(len(sv), peak_idx + decay_n)
    decay_idx = decay_search_end - 1
    for k in range(peak_idx, decay_search_end):
        if sv[k] <= decay_level:
            decay_idx = k
            break

    c_half = max(1, search_n // 2)
    return dict(b0=b0, b1=b1, p0=p0, p1=p1, onset_idx=onset_idx, decay_idx=decay_idx,
                decay_search_end=decay_search_end, onset_search_start=onset_search_start,
                local_max_lo=max(0, peak_idx - c_half), local_max_hi=min(len(v), peak_idx + c_half))


PARAM_LINES = [
    ("Amplitude threshold (a)", "amplitude_threshold", ""),
    ("Area threshold (b)", "area_threshold", ""),
    ("Number of points to average peak", "n_avg_peak", ""),
    ("Period to search local maximum (c)", "search_local_max_ms", " ms"),
    ("Time before peak for baseline (d)", "baseline_before_ms", " ms"),
    ("Period to average baseline (e)", "baseline_avg_ms", " ms"),
    ("Period to search decay time (f)", "decay_search_ms", " ms"),
    ("Fraction to find decay (g)", "decay_fraction", ""),
    ("Period to search onset before peak", "onset_search_ms", " ms"),
]


def params_text(params: DetectionParams) -> str:
    return "\n".join(f"{label}: {getattr(params, field)}{unit}" for label, field, unit in PARAM_LINES)


# Colors used consistently between plot_event_detail's annotations and
# build_legend_handles' key, so the two never drift apart.
COLOR_BASELINE_WINDOW = "#7fbf7f"
COLOR_BEFORE_PEAK = "#c9c9c9"
COLOR_BASELINE_LINE = "#2a8f2a"
COLOR_LOCAL_MAX = "#8a6dc9"
COLOR_PEAK = "crimson"
COLOR_AMPLITUDE = "#c97a2a"
COLOR_DECAY_WINDOW = "#f0c674"
COLOR_DECAY_LEVEL = "#b8860b"
COLOR_ONSET = "#2a6dc9"
COLOR_ONSET_WINDOW = "#9fc9e8"
COLOR_AREA_PASS = "#4a90d9"
COLOR_AREA_FAIL = "#e0555a"

# Light, high-contrast backing so inline trace labels stay legible over the
# raw signal and colored spans, instead of just floating text.
_LABEL_BBOX = dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=MUTED, lw=0.5, alpha=0.9)

# QC accept/reject color language -- shared between the overview's halo
# markers (accepted_overlay/rejected_overlay below) and the Accept/Reject
# buttons' own stylesheets (see EventInspector.__init__), so a decision
# reads the same way in both places. Kept as its own pair, separate from
# the per-event annotation colors above, even though COLOR_ACCEPTED happens
# to equal COLOR_BASELINE_LINE -- QC state and detection-window annotations
# are conceptually unrelated. Deliberately NOT reusing COLOR_PEAK (crimson,
# already the plain "undecided" candidate marker's own color) for rejected
# -- a rejected halo needs to read as "reject" without being confusable
# with an undecided marker.
COLOR_ACCEPTED = "#2a8f2a"
COLOR_REJECTED = "#c0392b"


def build_legend_handles():
    """Proxy artists explaining every annotation plot_event_detail draws --
    the 'key' shown once in the inspector window, built here so its colors
    can never drift out of sync with the trace annotations themselves."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    return [
        (Patch(fc=COLOR_BASELINE_WINDOW, alpha=0.35), "Baseline averaging window (e)"),
        (Line2D([], [], color=COLOR_BASELINE_LINE, ls="--", lw=1.2), "Baseline level used"),
        (Patch(fc=COLOR_BEFORE_PEAK, alpha=0.35), "Time before peak for baseline (d)"),
        (Line2D([], [], color=COLOR_LOCAL_MAX, lw=1.2, marker=">", ms=5), "Local-max search span (c)"),
        (Line2D([], [], color=COLOR_ONSET_WINDOW, ls="--", lw=1.4), "Onset search window boundary"),
        (Line2D([], [], color=COLOR_PEAK, marker="x", ls="None", mew=2, ms=9), "Detected peak"),
        (Patch(fc=COLOR_PEAK, alpha=0.15), "Peak-averaging window (n points)"),
        (Line2D([], [], color=COLOR_AMPLITUDE, lw=1.2, marker=">", ms=5), "Amplitude threshold (a)"),
        (Patch(fc=COLOR_DECAY_WINDOW, alpha=0.35), "Decay search window (f)"),
        (Line2D([], [], color=COLOR_DECAY_LEVEL, ls=":", lw=1.4), "Decay-fraction level (g)"),
        (Line2D([], [], color=COLOR_ONSET, marker="o", ls="None", ms=6), "Onset (rise crossing)"),
        (Line2D([], [], color=COLOR_DECAY_LEVEL, marker="o", ls="None", ms=6), "Decay point"),
        (Patch(fc=COLOR_AREA_PASS, alpha=0.35), "Measured area -- PASS (b)"),
        (Patch(fc=COLOR_AREA_FAIL, alpha=0.35), "Measured area -- FAIL (b)"),
    ]


def plot_event_detail(ax, t: np.ndarray, v: np.ndarray, dt: float, row: pd.Series,
                       params: DetectionParams, y_unit: str, pre_ms: float, post_ms: float):
    """Draw one detected event with every detection-parameter window/
    threshold annotated on it -- the interactive counterpart of
    this_method.pdf p.2's "Detection Parameters" diagram, but on the real
    trace that actually produced this event. See build_legend_handles() for
    what each color/marker means."""
    peak_idx = int(row["peak_idx"])
    baseline, amplitude = float(row["baseline"]), float(row["amplitude"])
    sign = -1.0 if params.direction == "negative" else 1.0
    peak_v = baseline + amplitude

    geo = _event_geometry(v, dt, peak_idx, baseline, amplitude, params)
    pre_n = _samples(pre_ms, dt)
    post_n = _samples(post_ms, dt)
    i0, i1 = max(0, peak_idx - pre_n), min(len(v), peak_idx + post_n)
    t_ms = (np.arange(i0, i1) - peak_idx) * dt * 1e3

    def rel(idx):
        return (idx - peak_idx) * dt * 1e3

    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.plot(t_ms, v[i0:i1], color=TRACE, lw=1.0, zorder=3)

    # (e) baseline-averaging window + (d) time-before-peak-for-baseline gap
    ax.axvspan(rel(geo["b0"]), rel(geo["b1"]), color=COLOR_BASELINE_WINDOW, alpha=0.35, zorder=1)
    ax.axvspan(rel(geo["b1"]), rel(peak_idx), color=COLOR_BEFORE_PEAK, alpha=0.35, zorder=1)
    ax.axhline(baseline, color=COLOR_BASELINE_LINE, ls="--", lw=1.2, zorder=2)

    # Onset search window (rise-side counterpart of decay_search_ms (f)) --
    # a boundary line, not a filled span, since it can extend further back
    # than (or overlap) the (d)/(e) bands above and a second overlapping
    # fill would just muddy them. (See the annotation key for what this is.)
    ax.axvline(rel(geo["onset_search_start"]), color=COLOR_ONSET_WINDOW, ls="--", lw=1.4, zorder=2)

    # (c) illustrative local-maximum search span, centered on the peak
    y_c = baseline + sign * 0.08 * abs(amplitude if amplitude else 1.0)
    x_c_lo, x_c_hi = rel(geo["local_max_lo"]), rel(geo["local_max_hi"])
    ax.annotate("", xy=(x_c_lo, y_c), xytext=(x_c_hi, y_c),
                arrowprops=dict(arrowstyle="<->", color=COLOR_LOCAL_MAX, lw=1.4))
    ax.text((x_c_lo + x_c_hi) / 2, y_c, "(c)", color=COLOR_LOCAL_MAX, fontsize=9, fontweight="bold",
             ha="center", va="bottom", bbox=_LABEL_BBOX, zorder=6)

    # peak marker + n_avg_peak averaging window
    ax.plot(0, peak_v, "x", color=COLOR_PEAK, ms=10, mew=2, zorder=5)
    ax.axvspan(rel(geo["p0"]), rel(geo["p1"]), color=COLOR_PEAK, alpha=0.15, zorder=1)

    # (a) amplitude threshold, drawn as a bracket from baseline
    thresh_v = baseline + sign * params.amplitude_threshold
    x_a = t_ms[0] * 0.5
    ax.annotate("", xy=(x_a, baseline), xytext=(x_a, thresh_v),
                arrowprops=dict(arrowstyle="<->", color=COLOR_AMPLITUDE, lw=1.4))
    ax.text(x_a, (baseline + thresh_v) / 2, f" a={params.amplitude_threshold}", color=COLOR_AMPLITUDE,
             fontsize=9, fontweight="bold", va="center", ha="left", bbox=_LABEL_BBOX, zorder=6)
    pass_amp = abs(amplitude) >= params.amplitude_threshold
    ax.axhline(thresh_v, color=COLOR_AMPLITUDE, ls=":", lw=1.0, zorder=1)

    # (f) decay-search window, (g) decay-fraction level, onset/decay markers
    ax.axvspan(rel(peak_idx), rel(geo["decay_search_end"]), color=COLOR_DECAY_WINDOW, alpha=0.20, zorder=1)
    # NOTE: no `sign` factor here, unlike thresh_v above -- `amplitude` (unlike
    # params.amplitude_threshold) is already signed, so baseline + amplitude
    # == peak_v; multiplying by sign again would flip this to the wrong side
    # of baseline entirely instead of landing between baseline and the peak.
    decay_level = baseline + (1.0 - params.decay_fraction) * amplitude
    ax.axhline(decay_level, color=COLOR_DECAY_LEVEL, ls=":", lw=1.2, zorder=1)
    ax.text(rel(geo["decay_search_end"]), decay_level, f" g={params.decay_fraction}", color=COLOR_DECAY_LEVEL,
             fontsize=9, fontweight="bold", ha="left", va="center", bbox=_LABEL_BBOX, zorder=6)
    ax.plot(rel(geo["onset_idx"]), v[geo["onset_idx"]], "o", color=COLOR_ONSET, ms=6,
             mec=INK, mew=0.5, zorder=4)
    ax.plot(rel(geo["decay_idx"]), v[geo["decay_idx"]], "o", color=COLOR_DECAY_LEVEL, ms=6,
             mec=INK, mew=0.5, zorder=4)

    # (b) area under the curve between onset and decay, shaded pass/fail
    # (blue/red, deliberately distinct from the green baseline-window span
    # above -- the two can sit close together near the peak)
    area_ok = abs(row["area"]) >= params.area_threshold
    fill_color = COLOR_AREA_PASS if area_ok else COLOR_AREA_FAIL
    onset_i, decay_i = geo["onset_idx"], geo["decay_idx"] + 1
    ax.fill_between(rel(np.arange(onset_i, decay_i)), v[onset_i:decay_i], baseline,
                     color=fill_color, alpha=0.35, zorder=2)

    ax.set_xlabel("time from peak (ms)", color=INK, fontsize=10)
    ax.set_ylabel(f"current ({y_unit})", color=INK, fontsize=10)
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=0.15, color=GRID)

    verdict = f"amp {'PASS' if pass_amp else 'FAIL'} / area {'PASS' if area_ok else 'FAIL'}"
    ax.set_title(
        f"t={row['peak_time_s']:.3f}s   amplitude={amplitude:.2f} {y_unit} (thr {params.amplitude_threshold})   "
        f"area={row['area']:.2f} (thr {params.area_threshold})   [{verdict}]",
        color=INK, fontsize=10,
    )


def build_legend_handles_miniml():
    """--source miniml counterpart of build_legend_handles: miniML has no
    windowed detection-parameters to annotate (see module docstring), so
    just the two things plot_event_detail_miniml actually draws."""
    from matplotlib.lines import Line2D
    return [
        (Line2D([], [], color=COLOR_BASELINE_LINE, ls="--", lw=1.2), "Baseline (peak - amplitude)"),
        (Line2D([], [], color=COLOR_PEAK, marker="x", ls="None", mew=2, ms=9), "Detected peak (model location)"),
    ]


# event-row column -> (label, value scale factor, unit) for the miniML detail
# view's stat line -- risetime/decaytime/halfwidth/interval are stored in
# seconds (see miniml.core.event.EventDetection: decaytimes *= trace.sampling,
# and the tutorial's own mean(...) * 1000 for a ms display), so scale=1000
# converts them to ms for display; charge and score are shown as saved.
MINIML_STAT_COLUMNS = [
    ("score", "score", 1.0, ""),
    ("charge", "charge", 1.0, " {y_unit}·s"),
    ("risetime", "10-90% rise", 1000.0, " ms"),
    ("decaytime", "half-decay", 1000.0, " ms"),
    ("halfwidth", "half-width", 1000.0, " ms"),
    ("interval", "IEI", 1000.0, " ms"),
]


def plot_event_detail_miniml(ax, t: np.ndarray, v: np.ndarray, dt: float, row: pd.Series,
                              y_unit: str, pre_ms: float, post_ms: float):
    """--source miniml counterpart of plot_event_detail: the CNN-LSTM
    detector has no windowed baseline/onset/decay-search concept the way
    the classical detector does (miniML doesn't persist those internal
    per-event positions to its *_individual.csv either -- only location/
    score/amplitude/charge/risetime/decaytime/halfwidth/interval), so this
    just draws the raw snippet around the model's own peak location with
    the peak marked and a derived baseline line (baseline = peak - amplitude,
    both already sign-adjusted the same way minianalysis's events are -- see
    _load_miniml_events / the *_individual.csv's own amplitude column), plus
    a text line of whichever stat columns are present in this event's row.
    See build_legend_handles_miniml() for what the two annotations mean."""
    peak_idx = int(row["peak_idx"])
    amplitude = float(row["amplitude"])
    peak_v = v[peak_idx]
    baseline = peak_v - amplitude

    pre_n = _samples(pre_ms, dt)
    post_n = _samples(post_ms, dt)
    i0, i1 = max(0, peak_idx - pre_n), min(len(v), peak_idx + post_n)
    t_ms = (np.arange(i0, i1) - peak_idx) * dt * 1e3

    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.plot(t_ms, v[i0:i1], color=TRACE, lw=1.0, zorder=3)
    ax.axhline(baseline, color=COLOR_BASELINE_LINE, ls="--", lw=1.2, zorder=2)
    ax.plot(0, peak_v, "x", color=COLOR_PEAK, ms=10, mew=2, zorder=5)

    ax.set_xlabel("time from peak (ms)", color=INK, fontsize=10)
    ax.set_ylabel(f"current ({y_unit})", color=INK, fontsize=10)
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=0.15, color=GRID)

    bits = [f"amplitude={amplitude:.2f} {y_unit}"]
    for col, label, scale, unit in MINIML_STAT_COLUMNS:
        if col in row.index and pd.notna(row[col]):
            bits.append(f"{label}={float(row[col]) * scale:.2f}{unit.format(y_unit=y_unit)}")
    ax.set_title(f"t={row['peak_time_s']:.3f}s   " + "   ".join(bits), color=INK, fontsize=9)


def build_legend_handles_fastmini():
    """--source fastmini counterpart of build_legend_handles: unlike miniml,
    fastmini's own *_fastmini_events.csv DOES persist a real onset/decay
    (via rise_time_ms/decay_time_ms, see plot_event_detail_fastmini), so
    this gets three annotations instead of miniml's two -- still nothing
    for a baseline-averaging window, local-max search span, or amplitude/
    area threshold, since fastmini has no per-event concept of any of
    those (see module docstring)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    return [
        (Line2D([], [], color=COLOR_BASELINE_LINE, ls="--", lw=1.2), "Baseline"),
        (Line2D([], [], color=COLOR_PEAK, marker="x", ls="None", mew=2, ms=9), "Detected peak"),
        (Line2D([], [], color=COLOR_ONSET, marker="o", ls="None", ms=6), "Onset (peak - rise_time_ms)"),
        (Line2D([], [], color=COLOR_DECAY_LEVEL, marker="o", ls="None", ms=6), "Decay point (peak + decay_time_ms)"),
        (Patch(fc=COLOR_AREA_PASS, alpha=0.35), "Measured area (onset-to-decay)"),
    ]


def plot_event_detail_fastmini(ax, t: np.ndarray, v: np.ndarray, dt: float, row: pd.Series,
                                y_unit: str, pre_ms: float, post_ms: float):
    """--source fastmini counterpart of plot_event_detail: like miniml,
    fastmini has no baseline-averaging window, local-max search span, or
    amplitude/area THRESHOLD to annotate (its peel-off/MLP confidence
    threshold isn't a per-event window the way minianalysis's is) -- but
    UNLIKE miniml, its *_fastmini_events.csv already carries a real
    baseline plus rise_time_ms/decay_time_ms/area for every event (see
    fastmini.py's own _measure_event, which measures these with the exact
    same onset/decay threshold-crossing convention minianalysis's
    detect_events steps 4-6 use), so the onset/decay POINTS and the area
    between them are reconstructed exactly here (peak_idx -+ rise_time_ms/
    decay_time_ms converted to samples), not just a derived baseline line
    the way plot_event_detail_miniml's is. See build_legend_handles_fastmini()."""
    peak_idx = int(row["peak_idx"])
    baseline = float(row["baseline"])
    amplitude = float(row["amplitude"])
    peak_v = baseline + amplitude

    onset_n = max(0, round(float(row["rise_time_ms"]) / 1000.0 / dt))
    decay_n = max(0, round(float(row["decay_time_ms"]) / 1000.0 / dt))
    onset_idx = max(0, peak_idx - onset_n)
    decay_idx = min(len(v) - 1, peak_idx + decay_n)

    pre_n = _samples(pre_ms, dt)
    post_n = _samples(post_ms, dt)
    i0, i1 = max(0, peak_idx - pre_n), min(len(v), peak_idx + post_n)
    t_ms = (np.arange(i0, i1) - peak_idx) * dt * 1e3

    def rel(idx):
        return (idx - peak_idx) * dt * 1e3

    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.plot(t_ms, v[i0:i1], color=TRACE, lw=1.0, zorder=3)
    ax.axhline(baseline, color=COLOR_BASELINE_LINE, ls="--", lw=1.2, zorder=2)
    ax.plot(0, peak_v, "x", color=COLOR_PEAK, ms=10, mew=2, zorder=5)
    ax.plot(rel(onset_idx), v[onset_idx], "o", color=COLOR_ONSET, ms=6, mec=INK, mew=0.5, zorder=4)
    ax.plot(rel(decay_idx), v[decay_idx], "o", color=COLOR_DECAY_LEVEL, ms=6, mec=INK, mew=0.5, zorder=4)
    ax.fill_between(rel(np.arange(onset_idx, decay_idx + 1)), v[onset_idx:decay_idx + 1], baseline,
                     color=COLOR_AREA_PASS, alpha=0.35, zorder=2)

    ax.set_xlabel("time from peak (ms)", color=INK, fontsize=10)
    ax.set_ylabel(f"current ({y_unit})", color=INK, fontsize=10)
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=0.15, color=GRID)

    ax.set_title(
        f"t={row['peak_time_s']:.3f}s   amplitude={amplitude:.2f} {y_unit}   "
        f"rise={row['rise_time_ms']:.2f} ms   decay={row['decay_time_ms']:.2f} ms   area={row['area']:.2f}",
        color=INK, fontsize=9,
    )


def _load_prior_decisions(reviewed_path: str, progress_path: str) -> dict:
    """{peak_idx: accepted_bool} from an existing progress file (this tool's
    own, or sepsc.review's -- identical schema, either resumes the other),
    or from a reviewed-only file (all True) if that's all that exists.
    Mirrors sepsc.review.ReviewSession._load_prior_decisions exactly."""
    if os.path.exists(progress_path):
        prior = pd.read_csv(progress_path)
        return dict(zip(prior["location"].astype(int), prior["decision"] == 1))
    if os.path.exists(reviewed_path):
        prior = pd.read_csv(reviewed_path)
        return dict(zip(prior["location"].astype(int), [True] * len(prior)))
    return {}


class EventInspector(QtWidgets.QMainWindow):
    def __init__(self, t: np.ndarray, v: np.ndarray, dt: float, df: pd.DataFrame,
                 params: DetectionParams | None, y_unit: str, title: str, pre_ms: float, post_ms: float,
                 reviewed_path: str, progress_path: str, source: str = "minianalysis",
                 compare_events: pd.DataFrame | None = None, compare_label: str | None = None,
                 compare_tolerance_s: float = 0.002):
        super().__init__()
        self.t, self.v, self.dt = t, v, dt
        self.reviewed_path, self.progress_path = reviewed_path, progress_path
        self.decisions: dict = _load_prior_decisions(reviewed_path, progress_path)
        self.df = df.reset_index(drop=True)
        self.params = params
        self.source = source
        self.y_unit = y_unit
        self.pre_ms, self.post_ms = pre_ms, post_ms
        # Read-only overlay of the OTHER detector's candidates (sepsc.review's
        # own feature -- see its --compare-csv), needs only peak_idx/
        # peak_time_s columns (both loaders here produce those). None/empty
        # -> no overlay marker, no agreement text; see show_event/__init__'s
        # marker block below.
        self.compare_events = compare_events if compare_events is not None and not compare_events.empty else None
        self.compare_label = compare_label
        self.compare_tolerance_s = compare_tolerance_s
        self.idx = 0
        self._pan_ax = None  # set while a left-button drag-to-pan is in progress; see _on_press

        self.setWindowTitle(title)
        self.resize(1600, 880)

        self.figure = Figure(figsize=(15, 8.5))
        self.figure.patch.set_facecolor(SURFACE)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        # Standard matplotlib pan/zoom-rectangle/home toolbar -- gives both
        # panels independent click-drag pan and box-zoom (each Axes keeps
        # its own view/history), on top of the scroll-wheel zoom wired up
        # below via _on_scroll.
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # -- QC row: Accept/Reject this event, or Accept All Remaining -----
        qc_row = QtWidgets.QHBoxLayout()
        self.accept_btn = QtWidgets.QPushButton("Accept  (A)")
        self.reject_btn = QtWidgets.QPushButton("Reject  (R)")
        self.accept_all_btn = QtWidgets.QPushButton("Accept All Remaining")
        for btn in (self.accept_btn, self.reject_btn, self.accept_all_btn):
            btn.setFocusPolicy(QtCore.Qt.NoFocus)  # keep keyboard focus on the canvas, not the button
        # Same green/red accept/reject language as COLOR_ACCEPTED/
        # COLOR_REJECTED's overview halos above (and the same hex values
        # sepsc.review's pre-EventInspector Tk buttons used to use, before
        # review.py started sharing this window -- restored here so that
        # color feedback isn't lost for either tool).
        self.accept_btn.setStyleSheet(
            "QPushButton { background-color: #a0d8a0; font-weight: bold; }"
            "QPushButton:hover { background-color: #66bb6a; }"
            "QPushButton:pressed { background-color: #4c9950; }")
        self.reject_btn.setStyleSheet(
            "QPushButton { background-color: #f2a0a0; font-weight: bold; }"
            "QPushButton:hover { background-color: #e57373; }"
            "QPushButton:pressed { background-color: #c85a5a; }")
        self.accept_all_btn.setStyleSheet(
            "QPushButton { background-color: #c8e6c9; }"
            "QPushButton:hover { background-color: #a5d6a7; }")
        self.accept_btn.clicked.connect(self.accept_current)
        self.reject_btn.clicked.connect(self.reject_current)
        self.accept_all_btn.clicked.connect(self.accept_all_remaining)
        qc_row.addWidget(self.accept_btn)
        qc_row.addWidget(self.reject_btn)
        qc_row.addWidget(self.accept_all_btn)
        qc_row.addStretch(1)
        self.qc_status_label = QtWidgets.QLabel("")
        self.qc_status_label.setStyleSheet("font-family: Consolas, monospace;")
        self.qc_status_label.setTextFormat(QtCore.Qt.RichText)
        layout.addLayout(qc_row)
        layout.addWidget(self.qc_status_label)

        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "Click a red X in the top panel to inspect that event, or use Right/N, Left/P to step. "
            "A/Up = accept, R/Down = reject.")

        self.ax_over, self.ax_detail = self.figure.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 2]})
        # Pan/zoom (scroll, drag, and the toolbar's own Pan/Zoom-rectangle
        # tool) is for the full-trace overview only -- the detail view
        # below it is a per-event snapshot that plot_event_detail always
        # redraws fresh (auto-fit to pre_ms/post_ms around the selected
        # peak) on every navigation/QC action, so it stays exactly as
        # before: no independent view of its own.
        self.ax_detail.set_navigate(False)

        stride = max(1, len(v) // OVERVIEW_MAX_POINTS)
        self.ax_over.set_facecolor(SURFACE)
        self.ax_over.plot(t[::stride], v[::stride], color=TRACE, lw=0.3, zorder=1)
        # QC halos, drawn BEHIND the x markers (zorder 2 < 3) so an
        # undecided event still just shows its plain crimson x, while a
        # decided one gets a colored ring around it.
        self.accepted_overlay, = self.ax_over.plot(
            [], [], "o", mfc=COLOR_ACCEPTED, mec="none", ms=9, zorder=2, label="accepted")
        self.rejected_overlay, = self.ax_over.plot(
            [], [], "o", mfc=COLOR_REJECTED, mec="none", ms=9, zorder=2, label="rejected")
        self.markers, = self.ax_over.plot(
            self.df["peak_time_s"], v[self.df["peak_idx"].to_numpy()], "x", color="crimson",
            ms=6, mew=1.2, zorder=3, picker=5, label=f"{len(self.df)} detected events (click one)")
        # Read-only comparison overlay (sepsc.review's --compare-csv) -- NOT
        # pickable (no picker= kwarg), so on_pick's own artist check never
        # needs to special-case it: only self.markers' clicks select an event.
        if self.compare_events is not None:
            self.ax_over.plot(
                self.compare_events["peak_time_s"], v[self.compare_events["peak_idx"].to_numpy()],
                "^", color=COLORS["fast"], mec=INK, mew=0.4, ms=6, ls="None", alpha=0.7, zorder=2,
                label=f"{len(self.compare_events)} {self.compare_label} candidates (comparison)")
        self.selected_marker, = self.ax_over.plot([], [], "o", color="gold", ms=12, mfc="none",
                                                     mew=2, zorder=4)
        self.ax_over.set_xlabel("time (s)", color=INK)
        self.ax_over.set_ylabel(f"current ({y_unit})", color=INK)
        self.ax_over.tick_params(colors=MUTED)
        self.ax_over.legend(loc="upper right", fontsize=8, labelcolor=INK)
        self._refresh_overview_markers()

        # Sidebar reserved at figure fraction x >= 0.66 (5.1in wide at this
        # figure size) -- wide enough for the longest parameter line at this
        # font size, so the box's left edge (it's left-aligned, unlike the
        # old right-aligned version that could spill past its own margin)
        # never crosses into the plot area.
        self.figure.subplots_adjust(right=0.65, hspace=0.35)
        SIDEBAR_X = 0.675

        # y-anchors: params box starts near the figure's own top (it lives in
        # figure coordinates, disjoint in x from both subplots above, so
        # this is safe regardless of where ax_over/ax_detail sit) and the
        # legend starts far enough below it to clear all 9 parameter lines
        # at this font size -- 8 lines used to fit under the old y=0.60/0.42
        # split, but the 9th (onset_search_ms) made the box tall enough to
        # overlap the legend below it.
        if self.source == "miniml":
            sidebar_body = (
                "miniML (CNN-LSTM) detector\n" + "-" * 26 +
                "\nNo windowed detection-parameters --\nthe model outputs a peak location\ndirectly "
                "(see module docstring).\nThreshold used is not saved to a\nsidecar; check the "
                "`detect` run's\nown --threshold."
            )
        elif self.source == "fastmini":
            sidebar_body = (
                "fastmini (MLP peel-off) detector\n" + "-" * 26 +
                "\nNo baseline-window, local-max-\nsearch, or amplitude/area threshold\n"
                "concept (the peel-off confidence\ncurve's own prominence cutoff isn't\na "
                "per-event window). Onset/decay\nmarkers ARE this event's own\nmeasured "
                "rise_time_ms/decay_time_ms\n(see module docstring)."
            )
        else:
            sidebar_body = "Detection parameters\n" + "-" * 26 + "\n" + params_text(params)
        self.param_text = self.figure.text(
            SIDEBAR_X, 0.97, sidebar_body,
            transform=self.figure.transFigure, ha="left", va="top", fontsize=9,
            family="monospace", color=INK,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f2f1ea", ec=MUTED, alpha=0.95))

        if self.source == "miniml":
            key_handles, key_labels = zip(*build_legend_handles_miniml())
        elif self.source == "fastmini":
            key_handles, key_labels = zip(*build_legend_handles_fastmini())
        else:
            key_handles, key_labels = zip(*build_legend_handles())
        self.figure.legend(
            key_handles, key_labels, loc="upper left", bbox_to_anchor=(SIDEBAR_X, 0.68),
            bbox_transform=self.figure.transFigure, fontsize=8.5, labelcolor=INK,
            title="Annotation key", title_fontsize=9.5, frameon=True,
            facecolor="#f2f1ea", edgecolor=MUTED, framealpha=0.95, handlelength=1.8)

        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas.mpl_connect("key_press_event", self.on_key)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_pan_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_pan_motion)
        self.canvas.mpl_connect("button_release_event", self.on_pan_release)
        self.show_event()
        self.canvas.setFocus()

    def show_event(self):
        row = self.df.iloc[self.idx]
        self.selected_marker.set_data([row["peak_time_s"]], [self.v[int(row["peak_idx"])]])
        if self.source == "miniml":
            plot_event_detail_miniml(self.ax_detail, self.t, self.v, self.dt, row,
                                      self.y_unit, self.pre_ms, self.post_ms)
        elif self.source == "fastmini":
            plot_event_detail_fastmini(self.ax_detail, self.t, self.v, self.dt, row,
                                        self.y_unit, self.pre_ms, self.post_ms)
        else:
            plot_event_detail(self.ax_detail, self.t, self.v, self.dt, row, self.params,
                               self.y_unit, self.pre_ms, self.post_ms)

        peak_idx = int(row["peak_idx"])
        qc = self.decisions.get(peak_idx)
        qc_word = "ACCEPTED" if qc is True else "REJECTED" if qc is False else "undecided"
        agreement = ""
        if self.compare_events is not None:
            also_flagged = bool(np.any(np.abs(
                self.compare_events["peak_time_s"].to_numpy() - row["peak_time_s"]) <= self.compare_tolerance_s))
            agreement = (f"   [ALSO flagged by {self.compare_label}]" if also_flagged
                         else f"   [NOT flagged by {self.compare_label}]")
        self.ax_detail.set_title(
            f"Event {self.idx + 1}/{len(self.df)}   QC: {qc_word}   " + self.ax_detail.get_title() + agreement,
            fontsize=10, color=INK)
        self.canvas.draw_idle()

        n_decided = len(self.decisions)
        n_accepted = sum(1 for ok in self.decisions.values() if ok)
        n_rejected = n_decided - n_accepted
        qc_color = {"ACCEPTED": COLOR_ACCEPTED, "REJECTED": COLOR_REJECTED, "undecided": MUTED}[qc_word]
        self.qc_status_label.setText(
            f"This event: <span style='color:{qc_color}; font-weight:bold;'>{qc_word}</span>"
            f"    |    Reviewed: {n_decided}/{len(self.df)} "
            f"(<span style='color:{COLOR_ACCEPTED};'>{n_accepted} accepted</span>, "
            f"<span style='color:{COLOR_REJECTED};'>{n_rejected} rejected</span>, "
            f"{len(self.df) - n_decided} undecided)")

    def _refresh_overview_markers(self):
        accepted_idxs = [i for i, ok in self.decisions.items() if ok]
        rejected_idxs = [i for i, ok in self.decisions.items() if not ok]
        acc_rows = self.df[self.df["peak_idx"].isin(accepted_idxs)]
        rej_rows = self.df[self.df["peak_idx"].isin(rejected_idxs)]
        self.accepted_overlay.set_data(acc_rows["peak_time_s"], self.v[acc_rows["peak_idx"].to_numpy()])
        self.rejected_overlay.set_data(rej_rows["peak_time_s"], self.v[rej_rows["peak_idx"].to_numpy()])

    def _autosave(self):
        """Write <stem>_minianalysis_reviewed.csv (accepted events, same
        columns sepsc.review writes) and <stem>_minianalysis_review_progress.csv
        (every decided event + a decision column, 1=accepted/0=rejected) --
        called after every single decision, so nothing is ever lost."""
        if not self.decisions:
            return
        decided = self.df[self.df["peak_idx"].isin(self.decisions.keys())].copy()
        decided["decision"] = decided["peak_idx"].map(lambda i: int(self.decisions[i]))
        decided = decided.rename(columns={"peak_idx": "location", "peak_time_s": "location_s"})
        decided.to_csv(self.progress_path, index=False)
        decided[decided["decision"] == 1].drop(columns="decision").to_csv(self.reviewed_path, index=False)

    def _set_decision(self, peak_idx: int, accepted: bool):
        self.decisions[peak_idx] = accepted
        self._autosave()
        self._refresh_overview_markers()

    @safe_callback
    def accept_current(self, *_qt_args):
        self._set_decision(int(self.df.iloc[self.idx]["peak_idx"]), True)
        self.idx = min(self.idx + 1, len(self.df) - 1)
        self.show_event()
        self.canvas.setFocus()

    @safe_callback
    def reject_current(self, *_qt_args):
        self._set_decision(int(self.df.iloc[self.idx]["peak_idx"]), False)
        self.idx = min(self.idx + 1, len(self.df) - 1)
        self.show_event()
        self.canvas.setFocus()

    @safe_callback
    def accept_all_remaining(self, *_qt_args):
        newly = 0
        for peak_idx in self.df["peak_idx"]:
            peak_idx = int(peak_idx)
            if peak_idx not in self.decisions:
                self.decisions[peak_idx] = True
                newly += 1
        self._autosave()
        self._refresh_overview_markers()
        self.show_event()
        self.statusBar().showMessage(f"Accepted {newly} remaining undecided event(s).", 5000)
        self.canvas.setFocus()

    @safe_callback
    def on_scroll(self, scroll_event):
        """Zoom the full-trace overview, centered on the cursor's data
        position. Plain scroll zooms time (X) and current (Y) together;
        hold Ctrl to zoom time only, Shift to zoom current only. Overview
        only, deliberately: the detail view below it is a per-event
        snapshot (plot_event_detail always redraws it fresh, auto-fit to
        pre_ms/post_ms around the selected peak), not something meant to
        be independently panned/zoomed.

        Modifier state is read directly from Qt (QApplication.
        keyboardModifiers()), NOT scroll_event.key: the Qt backend's own
        wheelEvent (backend_qt.FigureCanvasQT.wheelEvent) builds its
        MouseEvent with only x/y/step, never a key= kwarg, so
        scroll_event.key is always None for a real scroll -- matplotlib's
        usual event.key modifier check silently never fires here."""
        ax = scroll_event.inaxes
        if ax is not self.ax_over or scroll_event.xdata is None or scroll_event.ydata is None:
            return
        scale = 1 / 1.2 if scroll_event.button == "up" else 1.2
        xdata, ydata = scroll_event.xdata, scroll_event.ydata

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        ctrl = bool(modifiers & QtCore.Qt.ControlModifier)
        shift = bool(modifiers & QtCore.Qt.ShiftModifier)
        zoom_x, zoom_y = (True, False) if (ctrl and not shift) else \
                         (False, True) if (shift and not ctrl) else (True, True)

        if zoom_x:
            x0, x1 = ax.get_xlim()
            ax.set_xlim(xdata - (xdata - x0) * scale, xdata + (x1 - xdata) * scale)
        if zoom_y:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(ydata - (ydata - y0) * scale, ydata + (y1 - ydata) * scale)
        self.canvas.draw_idle()

    @safe_callback
    def on_pan_press(self, press_event):
        """Start of a left-drag pan on the full-trace overview (see
        on_scroll for why the detail view is excluded) -- deliberately NOT
        gated on distance from on_pick's marker picking: pick_event fires
        independently on the same press (a plain click still selects an
        event as usual), this only starts tracking a possible drag.
        Deferred entirely to the toolbar's own Pan/Zoom-rectangle tool
        while either is toggled on (self.toolbar.mode is non-empty then),
        so the two never fight over the same drag."""
        if press_event.button != 1 or press_event.inaxes is not self.ax_over or self.toolbar.mode != "":
            return
        self._pan_ax = press_event.inaxes
        self._pan_x0_px, self._pan_y0_px = press_event.x, press_event.y
        self._pan_xlim0 = press_event.inaxes.get_xlim()
        self._pan_ylim0 = press_event.inaxes.get_ylim()

    @safe_callback
    def on_pan_motion(self, motion_event):
        if self._pan_ax is None or motion_event.x is None or motion_event.y is None:
            return
        bbox = self._pan_ax.get_window_extent()
        if bbox.width <= 0 or bbox.height <= 0:
            return
        dx_px = motion_event.x - self._pan_x0_px
        dy_px = motion_event.y - self._pan_y0_px
        x0, x1 = self._pan_xlim0
        y0, y1 = self._pan_ylim0
        dx_data = dx_px / bbox.width * (x1 - x0)
        dy_data = dy_px / bbox.height * (y1 - y0)
        self._pan_ax.set_xlim(x0 - dx_data, x1 - dx_data)
        self._pan_ax.set_ylim(y0 - dy_data, y1 - dy_data)
        self.canvas.draw_idle()

    @safe_callback
    def on_pan_release(self, release_event):
        self._pan_ax = None

    @safe_callback
    def on_pick(self, pick_event):
        if pick_event.artist is not self.markers or len(pick_event.ind) == 0:
            return
        self.idx = int(pick_event.ind[0])
        self.show_event()

    @safe_callback
    def on_key(self, key_event):
        if key_event.key in ("right", "n"):
            self.idx = min(self.idx + 1, len(self.df) - 1)
            self.show_event()
        elif key_event.key in ("left", "p"):
            self.idx = max(self.idx - 1, 0)
            self.show_event()
        elif key_event.key in ("a", "up"):
            self.accept_current()
        elif key_event.key in ("r", "down"):
            self.reject_current()


def resolve_source_filter(source: str, filter_flag: bool, cutoff_hz: float, target_rate_hz: float,
                           filter_order: int) -> tuple[bool, float, float, int]:
    """--source fastmini special case, factored out of load_trace_for_source
    so callers can resolve it (and the stem it implies) BEFORE doing any
    file I/O -- see load_trace_for_source's own docstring for why fastmini
    needs this at all. Returns the (possibly overridden) effective
    (filter_flag, cutoff_hz, target_rate_hz, filter_order)."""
    if source != "fastmini":
        return filter_flag, cutoff_hz, target_rate_hz, filter_order
    if filter_flag and (cutoff_hz != DEFAULT_CUTOFF_HZ or target_rate_hz != DEFAULT_TARGET_RATE_HZ
                         or filter_order != DEFAULT_ORDER):
        print(f"NOTE: --source fastmini always preprocesses internally at a fixed "
              f"{DEFAULT_CUTOFF_HZ:.0f} Hz/{DEFAULT_TARGET_RATE_HZ:.0f} Hz Bessel filter "
              f"(order {DEFAULT_ORDER}) -- your --cutoff-hz/--target-rate-hz/--filter-order "
              f"are ignored for this source.", flush=True)
    return True, DEFAULT_CUTOFF_HZ, DEFAULT_TARGET_RATE_HZ, DEFAULT_ORDER


def resolve_source_stem(abf_path: str, source: str, filter_flag: bool, cutoff_hz: float,
                         target_rate_hz: float, filter_order: int) -> tuple[str, bool, float, float, int]:
    """The `stem` a source's CSV/output paths are built from -- pure string
    logic, no file I/O, so callers can compute this (and fail fast on a
    missing CSV) BEFORE the potentially-expensive Bessel-filter/downsample
    load_trace_for_source below does. Returns
    (stem, effective_filter_flag, cutoff_hz, target_rate_hz, filter_order)."""
    filter_flag, cutoff_hz, target_rate_hz, filter_order = resolve_source_filter(
        source, filter_flag, cutoff_hz, target_rate_hz, filter_order)
    stem = os.path.splitext(abf_path)[0]
    if filter_flag and source != "fastmini":
        # Same suffix minianalysis.py's/detect.py's own --filter appends to
        # `stem` -- fastmini.py never appends one (see resolve_source_filter),
        # so its stem is always the plain abf stem regardless of filter_flag.
        stem += f"_filt{int(cutoff_hz)}Hz{int(target_rate_hz)}Hz"
    return stem, filter_flag, cutoff_hz, target_rate_hz, filter_order


def load_display_trace(abf_path: str, channel: int, filter_flag: bool, cutoff_hz: float,
                        target_rate_hz: float, filter_order: int):
    """Load (t, v, dt, y_unit) -- the raw trace, or the same Bessel-filtered
    + resampled one minianalysis.py's/detect.py's/fastmini.py's own
    filtering produces, if filter_flag. Source-AGNOSTIC and does no
    --source fastmini special-casing of its own -- callers resolve that via
    resolve_source_stem/resolve_source_filter first (so a missing CSV can
    fail fast before this does any filtering) and pass the already-resolved
    filter_flag/cutoff_hz/target_rate_hz/filter_order straight through, so
    the fastmini NOTE (if any) prints exactly once, not twice. Shared by
    this module's own main() and sepsc.review's (which reuses EventInspector
    wholesale, see its module docstring), so the two can't drift out of
    sync on this logic again the way they already have once this session (a
    raw-vs-filtered trace mismatch was a real, repeated bug here before
    review.py started sharing this)."""
    abf = pyabf.ABF(abf_path)
    abf.setSweep(0, channel=channel)
    y_unit = abf.adcUnits[channel]

    if filter_flag:
        hardware_filter_hz = get_hardware_filter_hz(abf, channel)
        if hardware_filter_hz is not None and hardware_filter_hz <= cutoff_hz:
            print(f"NOTE: channel {channel} is already hardware-filtered at "
                  f"{hardware_filter_hz:.0f} Hz, at or below the requested {cutoff_hz:.0f} Hz.")
        t, v, fs, _ = load_filtered_trace(abf_path, channel=channel, cutoff_hz=cutoff_hz,
                                           target_hz=target_rate_hz, order=filter_order)
        dt = 1.0 / fs
        print(f"Filtered: {abf.dataRate:.0f} Hz raw -> {cutoff_hz:.0f} Hz Bessel "
              f"(order {filter_order}, zero-phase) -> {fs:.0f} Hz", flush=True)
    else:
        t = np.asarray(abf.sweepX, float)
        v = np.asarray(abf.sweepY, float)
        dt = 1.0 / abf.dataRate

    return t, v, dt, y_unit


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", help="Path to the gap-free voltage-clamp .abf file")
    p.add_argument("--source", choices=list(SOURCE_CFG), default="minianalysis",
                   help="which detector's finished output to inspect (default: minianalysis)")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--csv", default=None,
                   help="Path to the source's events CSV (default: <abf><source's own _events/_individual "
                        "suffix>.csv)")
    p.add_argument("--params-json", default=None,
                   help="minianalysis only: path to the params sidecar (default: "
                        "<abf>_minianalysis_params.json)")
    p.add_argument("--pre-ms", type=float, default=None,
                   help="detail-view window before the peak, ms (default: auto -- for minianalysis, "
                        "from the baseline params with headroom; a fixed 15/8 ms for miniml/fastmini)")
    p.add_argument("--post-ms", type=float, default=None,
                   help="detail-view window after the peak, ms (default: auto -- for minianalysis, "
                        "from decay_search_ms with headroom; a fixed 25/23 ms for miniml/fastmini)")
    # Same override flags as minianalysis.py, applied on top of the sidecar (or its own defaults)
    # only when explicitly passed -- see _load_params.
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
                   help="minianalysis/miniml only: load the SAME Bessel-filtered + resampled trace "
                        "minianalysis.py's/detect.py's --filter produced (must match whatever --filter "
                        "settings that detection run used, since event peak_idx values are indices into "
                        "that filtered/resampled trace, not the raw one). Ignored for --source fastmini, "
                        "which always preprocesses at a fixed rate regardless of this flag.")
    p.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ,
                   help=f"only with --filter: Bessel low-pass cutoff, Hz (default {DEFAULT_CUTOFF_HZ:.0f})")
    p.add_argument("--target-rate-hz", type=float, default=DEFAULT_TARGET_RATE_HZ,
                   help=f"only with --filter: output sampling rate, Hz (default {DEFAULT_TARGET_RATE_HZ:.0f})")
    p.add_argument("--filter-order", type=int, default=DEFAULT_ORDER,
                   help=f"only with --filter: Bessel filter order (default {DEFAULT_ORDER})")
    args = p.parse_args(argv)

    cfg = SOURCE_CFG[args.source]
    stem, filter_flag, cutoff_hz, target_rate_hz, filter_order = resolve_source_stem(
        args.abf, args.source, args.filter, args.cutoff_hz, args.target_rate_hz, args.filter_order)

    csv_path = args.csv or f"{stem}{cfg['events_suffix']}"
    if not os.path.exists(csv_path):
        detect_hint = {"minianalysis": "python -m sepsc minianalysis", "fastmini": "python -m sepsc fastmini",
                        "miniml": "<clampex_miniml env>\\python.exe -m sepsc detect"}[args.source]
        p.error(f"events CSV not found: {csv_path!r} (run `{detect_hint} {args.abf}"
                 f"{' --filter' if args.filter and args.source != 'fastmini' else ''}` first, or pass --csv)")

    t, v, dt, y_unit = load_display_trace(args.abf, args.channel, filter_flag, cutoff_hz,
                                           target_rate_hz, filter_order)

    df = load_source_events(csv_path, args.source, t)
    if args.source == "miniml":
        params = None
        pre_ms = args.pre_ms if args.pre_ms is not None else 15.0
        post_ms = args.post_ms if args.post_ms is not None else 25.0
    elif args.source == "fastmini":
        params = None
        # +3ms headroom beyond fastmini.py's own _measure_event pre_ms=5.0/
        # post_ms=20.0 measurement window, same convention as minianalysis's
        # own auto-default below.
        pre_ms = args.pre_ms if args.pre_ms is not None else 8.0
        post_ms = args.post_ms if args.post_ms is not None else 23.0
    else:
        overrides = dict(
            direction=args.direction, amplitude_threshold=args.amplitude_threshold,
            area_threshold=args.area_threshold, n_avg_peak=args.n_avg_peak,
            search_local_max_ms=args.search_local_max_ms, baseline_before_ms=args.baseline_before_ms,
            baseline_avg_ms=args.baseline_avg_ms, decay_search_ms=args.decay_search_ms,
            decay_fraction=args.decay_fraction, onset_fraction=args.onset_fraction,
            onset_search_ms=args.onset_search_ms,
        )
        params = _load_params(args.params_json, stem, overrides)
        pre_ms = args.pre_ms if args.pre_ms is not None else params.baseline_before_ms + params.baseline_avg_ms + 3.0
        post_ms = args.post_ms if args.post_ms is not None else params.decay_search_ms + 3.0
    if df.empty:
        p.error("no events in that CSV")

    reviewed_path = f"{stem}{cfg['reviewed_suffix']}"
    progress_path = f"{stem}{cfg['progress_suffix']}"

    print(f"{len(df)} {cfg['label']} events from {os.path.basename(csv_path)}", flush=True)
    print("Click a red X in the top panel to inspect that event, or use Right/N, Left/P to step. "
          "A/Up = accept, R/Down = reject.", flush=True)
    if os.path.exists(progress_path) or os.path.exists(reviewed_path):
        print(f"Resuming QC from {os.path.basename(progress_path)}"
              f"{' / ' + os.path.basename(reviewed_path) if os.path.exists(reviewed_path) else ''}", flush=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = EventInspector(t, v, dt, df, params, y_unit,
                             title=f"sEPSC detection inspector [{cfg['label']}] -- {os.path.basename(args.abf)}",
                             pre_ms=pre_ms, post_ms=post_ms,
                             reviewed_path=reviewed_path, progress_path=progress_path,
                             source=args.source)
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
