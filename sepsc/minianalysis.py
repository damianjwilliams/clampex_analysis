"""
Classical local-maximum + baseline + amplitude/area-threshold synaptic
event detector AND its downstream analysis stage, reimplementing the
method described in the Synaptosoft "Mini Analysis Program" tutorial
(C. Justin Lee, PhD; recovered from archive.org -- see
C:\\Users\\damia\\synaptosoft_mirror\\synaptosoft_tutorial_recovered\\final,
slides 0006/0007/0017-0019/0022, and
C:\\Users\\damia\\OneDrive - Columbia University Irving Medical Center\\Desktop\\this_method.pdf,
pages 1-13, which covers the same detector plus the analysis stage below).

This is an independent reconstruction from that tutorial's published
method description, not a copy of Synaptosoft's source (which isn't
available) -- a handful of implementation details it doesn't specify
(the baseline statistic, the exact onset-crossing definition used for
"time to peak", the exact skewness/kurtosis estimator) are reasonable,
clearly-flagged choices below rather than verified originals. Everything
the tutorial DOES specify is implemented as described.

DETECTION -- sequence of peak detection (slide 0017 / PDF p.3):
    1. find a local maximum
    2. find a baseline
    3. compare amplitude to threshold
    4. compute time to peak -- if the trace never rises past onset_fraction
       within onset_search_ms before the peak, the candidate is REJECTED
       outright (same hard search-limit semantics as step 5's
       decay_search_ms, just on the rise side)
    5. compute time to decay -- if the trace never reaches decay_fraction
       (g) of the peak within decay_search_ms (f), the candidate is
       REJECTED outright (decay_search_ms is a hard search limit on
       whether an event counts at all, not just a display/measurement cap)
    6. compute area, compare to threshold

Detection parameters (slide 0019 / PDF p.2), all fields of DetectionParams:
    amplitude threshold (a), area threshold (b), peak direction,
    number of points to average peak, period to search local maximum (c),
    time before peak for baseline (d), period to average baseline (e),
    period to search decay time (f), fraction to find decay (g)

For closely-spaced/overlapping events, slide 0007 / PDF p.6 describes
adjusting the baseline by extrapolating the PREVIOUS event's decay as a
single exponential (Y = A*e^(-x/tau)) rather than trusting a baseline
window that's still contaminated by that decay -- see
_overlap_adjusted_baseline. OFF by default (--adjust-overlapping-baseline
to opt in): baseline is otherwise ALWAYS the plain (d)/(e) window average,
exactly what those two named parameters describe and nothing else.

ANALYSIS -- once events are detected, PDF pages 7-13 describe a further
analysis stage built on two structures (p.7 "Analysis I: Grouping and Data
Array"):
    - Grouping: selecting a subset of detected events for focused analysis
      (p.8 "Different Ways to Group Events" -- by criteria search, random
      selection, or episode) -- see group_by_criteria / group_random.
    - Data Arrays: merging/combining event data for further analysis (p.9
      "Data Arrays: Combining and Manipulating Data" -- inter-event
      intervals, +,-,x,/,sqr,sqrt) -- see events_frame / combine_data_arrays.
That feeds two further analyses:
    - Descriptive Analysis (p.10): column statistics (5 moments of a
      distribution), frequency/cumulative histograms, running average,
      auto-/cross-correlation histograms -- see column_statistics,
      frequency_histogram, cumulative_histogram, running_average,
      autocorrelation_histogram, cross_correlation_histogram.
    - Group Analysis (p.11-13): display of grouped traces (average/
      superimposed, raw or scaled) and single/double exponential decay
      fitting via a Simplex (Nelder-Mead) minimization of sum-of-squares
      over a selectable fit range -- see extract_event_traces,
      scale_traces, fit_exponential_decay.
The action-potential-waveform-analysis, amperometric-peak-analysis, and
random-walk-modeling items also named on p.11 are for entirely different
signal types (spiking traces, amperometry) outside this project's scope
(gap-free voltage-clamp sEPSC recordings) and aren't implemented.

Usage
-----
    python -m sepsc.minianalysis path\\to\\recording.abf
        # ^ by default this ALSO opens a dialog to review/edit the 9 detection parameters plus the
        # filter/resample settings before running -- edit or leave them, then click OK to detect.
    python -m sepsc.minianalysis recording.abf --no-gui --amplitude-threshold 8 --area-threshold 15
        # ^ --no-gui skips the dialog and runs immediately from these flags/defaults (for scripts)
    python -m sepsc.minianalysis recording.abf --no-gui --filter --cutoff-hz 3000 --target-rate-hz 10000
    python -m sepsc.minianalysis recording.abf --no-gui --stats --histogram-column amplitude \\
        --autocorr --group-analysis --fit-decay peak_to_end
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pyabf
from scipy import stats as spstats
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
from scipy.signal import find_peaks

from .gui_utils import FILTER_FIELD_SPECS, make_field_widget, read_field_widget

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .preprocess import DEFAULT_CUTOFF_HZ, DEFAULT_ORDER, DEFAULT_TARGET_RATE_HZ, get_hardware_filter_hz, load_filtered_trace
from .style import SURFACE, TRACE, INK, MUTED

__all__ = [
    "DetectionParams", "Event", "detect_events",
    "events_frame", "group_by_criteria", "group_random", "combine_data_arrays",
    "ColumnStatistics", "column_statistics", "frequency_histogram", "cumulative_histogram",
    "running_average", "autocorrelation_histogram", "cross_correlation_histogram",
    "extract_event_traces", "scale_traces", "DecayFit", "fit_exponential_decay",
]


@dataclass(frozen=True)
class DetectionParams:
    """The 9 Mini-Analysis-style detection parameters (tutorial slide 0019).

    Defaults are a reasonable starting point for sEPSCs, not calibrated
    values -- the tutorial itself (slide 0018, "Optimizing Detection
    Parameters") is explicit that these are meant to be tuned per
    recording by inspecting detected vs. missed events, not left fixed.
    """
    amplitude_threshold: float = 5.0                          # (a) pA
    area_threshold: float = 10.0                                # (b) pA*ms
    direction: Literal["negative", "positive"] = "negative"
    n_avg_peak: int = 3            # points averaged to read the peak value
    search_local_max_ms: float = 3.0                              # (c)
    """Must be at least comparable to a real event's decay duration --
    too short (e.g. 1ms against a ~3ms decay tau) lets per-sample noise
    riding on a single event's own decaying tail masquerade as several
    additional "local maxima", fragmenting one event into several spurious
    detections. Tune this against your own events' typical decay time."""
    baseline_before_ms: float = 2.0                                # (d)
    baseline_avg_ms: float = 3.0                                    # (e)
    decay_search_ms: float = 20.0                                    # (f)
    decay_fraction: float = 0.5                                       # (g)
    onset_fraction: float = 0.1
    """Not one of the tutorial's 9 named parameters -- needed to define
    "time to peak" (step 4), which the tutorial doesn't give a formula
    for. Mirrors decay_fraction's role but on the rising side: the onset
    is where the trace first rises past this fraction of the peak
    amplitude (above baseline) and stays there."""
    onset_search_ms: float = 5.0
    """Also not one of the tutorial's 9 -- the rise-side counterpart of
    decay_search_ms (f): how far before the peak to search for the
    onset_fraction crossing. If the trace never rises past onset_fraction
    within this window, the candidate is REJECTED outright (same hard
    search-limit semantics as decay_search_ms), not just measured with a
    truncated/unreached onset."""
    adjust_overlapping_baseline: bool = False
    """Off by default: baseline is then ALWAYS the plain mean of the (d)/
    (e) window (peak_idx - before_n - avg_n : peak_idx - before_n), exactly
    the two named parameters describe -- nothing else. Set True to opt into
    slide 0007/PDF p.6's extra step for closely-spaced events, which
    REPLACES that window average with an extrapolated value (Y=A*e^(-x/tau)
    from the previous event's own decay) when the window would otherwise
    still be contaminated by it -- i.e. a baseline computed a different way
    than (d)/(e) for those specific events, not a bug when it happens, but
    also not what (d)/(e) alone would produce, hence opt-in."""


@dataclass
class Event:
    peak_idx: int
    peak_time_s: float
    baseline: float
    amplitude: float
    rise_time_ms: float
    decay_time_ms: float
    area: float


def _samples(ms: float, dt: float) -> int:
    return max(1, int(round(ms / 1000.0 / dt)))


def _overlap_adjusted_baseline(resting_baseline: float, prev_root_event: Event,
                                prev_tau_s: float, peak_time_s: float) -> Optional[float]:
    """Slide 0007: predict what the trace would read at THIS candidate's
    own peak time if the previous (root) event's decay is still ongoing --
    Y = A*e^(-x/tau) extrapolated from that event's fitted single-
    exponential decay -- so amplitude can be measured against the
    instantaneous decay trajectory instead of a stale, contaminated
    baseline-window average.

    Deliberately evaluated at the CANDIDATE'S OWN peak time, not the
    baseline window's time: those differ by several ms whenever a decay is
    still in progress, and using the wrong one under-corrects baseline by
    nearly the full amplitude of the earlier event (the two times aren't
    interchangeable the way they are for a flat, unchanging baseline).

    Returns None (caller falls back to the flat window average) if the
    residual is negligible or `peak_time_s` doesn't come after the root
    event.
    """
    x_s = peak_time_s - prev_root_event.peak_time_s
    if x_s <= 0 or prev_tau_s <= 0:
        return None
    residual = prev_root_event.amplitude * float(np.exp(-x_s / prev_tau_s))
    if abs(residual) < 0.02 * abs(prev_root_event.amplitude):
        return None  # decayed away; nothing to correct for
    return resting_baseline + residual


def detect_events(t: np.ndarray, v: np.ndarray, dt: float,
                   params: DetectionParams = DetectionParams()) -> list[Event]:
    """Scan a raw trace for synaptic events using the Mini-Analysis-style
    6-step sequence (slide 0017). `t`/`v` are the full trace (seconds /
    trace units); `dt` is the sample interval in seconds.
    """
    pol = -1.0 if params.direction == "negative" else 1.0
    sv = pol * v

    search_n = _samples(params.search_local_max_ms, dt)
    before_n = _samples(params.baseline_before_ms, dt)
    avg_n = _samples(params.baseline_avg_ms, dt)
    decay_n = _samples(params.decay_search_ms, dt)
    onset_search_n = _samples(params.onset_search_ms, dt)
    n_avg_peak = max(1, params.n_avg_peak)

    # Step 1: local maxima at least search_n samples apart -- a point only
    # counts as a candidate peak if nothing taller exists within that
    # window (tutorial's "period to search local maximum").
    candidate_idxs, _ = find_peaks(sv, distance=search_n)

    events: list[Event] = []
    prev_root_event: Optional[Event] = None  # last ACCEPTED event, for extrapolation
    prev_tau_s: Optional[float] = None
    resting_baseline: Optional[float] = None  # last known uncontaminated baseline reading

    for peak_idx in candidate_idxs:
        peak_idx = int(peak_idx)
        peak_time_s = peak_idx * dt

        # Step 2: baseline, from a window ending `baseline_before_ms`
        # before the peak and spanning `baseline_avg_ms`.
        b0 = peak_idx - before_n - avg_n
        b1 = peak_idx - before_n
        if b0 < 0:
            continue
        flat_baseline = float(np.mean(v[b0:b1]))
        baseline = flat_baseline
        adjusted = None
        if params.adjust_overlapping_baseline and prev_root_event is not None and prev_tau_s is not None:
            adjusted = _overlap_adjusted_baseline(
                resting_baseline if resting_baseline is not None else flat_baseline,
                prev_root_event, prev_tau_s, peak_time_s)
        if adjusted is not None:
            baseline = adjusted
        else:
            resting_baseline = flat_baseline  # this window wasn't contaminated; a clean reading

        # Step 3: amplitude vs. threshold.
        p0 = max(0, peak_idx - n_avg_peak // 2)
        p1 = min(len(v), p0 + n_avg_peak)
        peak_v = float(np.mean(v[p0:p1]))
        amplitude = peak_v - baseline
        if abs(amplitude) < params.amplitude_threshold:
            continue

        s_baseline, s_peak = pol * baseline, pol * peak_v
        span = s_peak - s_baseline
        if span <= 0:
            continue  # degenerate; can happen if baseline adjustment overshoots

        # Step 4: time to peak -- walk backward from the peak, within
        # onset_search_ms, to where the trace first rises past
        # onset_fraction of the amplitude and stays above it. If it never
        # gets there within the window, this candidate is rejected outright
        # (onset_search_ms is a hard search limit, same as decay_search_ms
        # (f) on the decay side -- not just a display/measurement cap).
        onset_level = s_baseline + params.onset_fraction * span
        onset_search_start = max(0, peak_idx - onset_search_n)
        onset_idx = None
        for k in range(peak_idx, onset_search_start - 1, -1):
            if sv[k] < onset_level:
                onset_idx = k
                break
        if onset_idx is None:
            continue
        rise_time_ms = (peak_idx - onset_idx) * dt * 1e3

        # Step 5: time to decay -- walk forward within decay_search_ms for
        # where the trace decays to decay_fraction of the peak amplitude.
        # If it never gets there within the window, this candidate is
        # rejected outright (not counted as an event with a truncated/
        # unreached decay point) -- decay_search_ms (f) is a hard search
        # limit, not a display cap.
        decay_level = s_baseline + (1.0 - params.decay_fraction) * span
        search_end = min(len(sv), peak_idx + decay_n)
        decay_idx = None
        for k in range(peak_idx, search_end):
            if sv[k] <= decay_level:
                decay_idx = k
                break
        if decay_idx is None:
            continue
        decay_time_ms = (decay_idx - peak_idx) * dt * 1e3

        # Step 6: area (baseline-subtracted, onset to decay point) vs.
        # threshold.
        area = float(_trapz(v[onset_idx:decay_idx + 1] - baseline, dx=dt * 1e3))
        if abs(area) < params.area_threshold:
            continue

        event = Event(
            peak_idx=peak_idx, peak_time_s=float(peak_idx * dt),
            baseline=baseline, amplitude=amplitude,
            rise_time_ms=rise_time_ms, decay_time_ms=decay_time_ms, area=area,
        )
        events.append(event)

        # Fit this event's own decay as a single exponential (slide 0007's
        # Y = A*e^(-x/tau)) so the NEXT candidate can extrapolate it if
        # it's close enough to still be riding this one's tail. tau is
        # solved analytically from decay_fraction/decay_time_ms, which are
        # already exactly known -- no extra sampling needed.
        #
        # Only do this when THIS event's own baseline was clean (adjusted
        # is None): an event accepted while still riding a previous decay
        # has its own decay_time_ms measured on a signal that's the SUM of
        # both events' decays, not a clean read of its own kinetics alone,
        # so fitting tau from it and chaining forward would compound the
        # error into every subsequent candidate. Keep extrapolating from
        # the last genuinely clean event instead -- a documented
        # simplification for 3+ closely chained/overlapping events (this
        # single-exponential correction is a first-order approximation per
        # the tutorial's own description, not full multi-event
        # deconvolution).
        if adjusted is None:
            if decay_time_ms > 0 and 0.0 < params.decay_fraction < 1.0:
                prev_tau_s = -(decay_time_ms / 1000.0) / np.log(1.0 - params.decay_fraction)
            else:
                prev_tau_s = None
            prev_root_event = event

    return events


# ---------------------------------------------------------------------------
# Analysis I: Grouping and Data Arrays (PDF p.7-9)
# ---------------------------------------------------------------------------

def events_frame(events: list[Event]) -> pd.DataFrame:
    """Event list -> tidy DataFrame, sorted by peak time, with an
    inter-event interval column added (p.9 "Data Arrays": "By calculating
    inter-event intervals"). The first event's interval is NaN -- there's
    no preceding event to measure it from."""
    df = pd.DataFrame([vars(e) for e in events])
    if df.empty:
        df["inter_event_interval_ms"] = pd.Series(dtype=float)
        return df
    df = df.sort_values("peak_time_s").reset_index(drop=True)
    df["inter_event_interval_ms"] = df["peak_time_s"].diff() * 1e3
    return df


def group_by_criteria(df: pd.DataFrame, **ranges: tuple[Optional[float], Optional[float]]) -> np.ndarray:
    """Boolean mask selecting rows whose columns fall within given
    (min, max) ranges, e.g. group_by_criteria(df, amplitude=(10, 50),
    decay_time_ms=(0, 15)) -- p.8 "Different Ways to Group Events": "Group
    by Criteria Search". Either bound may be None for an open end."""
    mask = np.ones(len(df), dtype=bool)
    for column, (lo, hi) in ranges.items():
        values = df[column].to_numpy(dtype=float)
        if lo is not None:
            mask &= values >= lo
        if hi is not None:
            mask &= values <= hi
    return mask


def group_random(n_total: int, n_select: int, seed: Optional[int] = None) -> np.ndarray:
    """Boolean mask selecting a random subset of n_total rows -- p.8
    "Different Ways to Group Events": "Group by Random Selection"."""
    rng = np.random.default_rng(seed)
    n_select = max(0, min(n_select, n_total))
    idx = rng.choice(n_total, size=n_select, replace=False)
    mask = np.zeros(n_total, dtype=bool)
    mask[idx] = True
    return mask


def combine_data_arrays(a, b, op: Literal["+", "-", "x", "/", "sqr", "sqrt"]) -> np.ndarray:
    """The mathematical operations p.9 "Data Arrays" lists for combining/
    manipulating data arrays: (+,-,x,/,sqr,sqrt). `b` is ignored for
    sqr/sqrt (unary)."""
    a = np.asarray(a, dtype=float)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "x":
        return a * b
    if op == "/":
        return a / b
    if op == "sqr":
        return a ** 2
    if op == "sqrt":
        return np.sqrt(a)
    raise ValueError(f"unknown operation {op!r} (expected one of +,-,x,/,sqr,sqrt)")


# ---------------------------------------------------------------------------
# Analysis II: Descriptive Analysis (PDF p.10)
# ---------------------------------------------------------------------------

@dataclass
class ColumnStatistics:
    """"Column Statistics (5 moments of a distribution: mean, variance,
    etc.)" (p.10) -- n plus the 5 moments: mean, variance, sd, skewness,
    kurtosis (excess, i.e. 0 for a normal distribution)."""
    n: int
    mean: float
    variance: float
    sd: float
    skewness: float
    kurtosis: float


def column_statistics(values) -> ColumnStatistics:
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return ColumnStatistics(0, np.nan, np.nan, np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=1)) if n > 1 else 0.0
    sd = float(np.sqrt(variance))
    skewness = float(spstats.skew(x, bias=False)) if n > 2 else np.nan
    kurtosis = float(spstats.kurtosis(x, bias=False)) if n > 3 else np.nan
    return ColumnStatistics(n, mean, variance, sd, skewness, kurtosis)


def frequency_histogram(values, bin_size: Optional[float] = None) -> pd.DataFrame:
    """Frequency histogram (p.10): counts per bin_size-wide bin over the
    data's own range. bin_size=None auto-picks range/30 (or 1.0 if the
    data is degenerate/empty)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return pd.DataFrame({"bin_start": [], "bin_center": [], "count": []})
    data_range = float(x.max() - x.min())
    if bin_size is None:
        bin_size = data_range / 30.0 if data_range > 0 else 1.0
    lo = np.floor(x.min() / bin_size) * bin_size
    hi = np.ceil(x.max() / bin_size) * bin_size + bin_size
    edges = np.arange(lo, hi, bin_size)
    counts, edges = np.histogram(x, bins=edges)
    return pd.DataFrame({"bin_start": edges[:-1], "bin_center": edges[:-1] + bin_size / 2, "count": counts})


def cumulative_histogram(values, bin_size: Optional[float] = None) -> pd.DataFrame:
    """Cumulative fraction histogram (p.10)."""
    hist = frequency_histogram(values, bin_size)
    hist = hist.copy()
    total = hist["count"].sum()
    hist["cumulative_fraction"] = hist["count"].cumsum() / total if total > 0 else hist["count"].astype(float)
    return hist


def running_average(values, window: int) -> np.ndarray:
    """Running average histogram (p.10)."""
    x = np.asarray(values, dtype=float)
    return uniform_filter1d(x, size=max(1, window), mode="nearest")


def autocorrelation_histogram(event_times_s, max_lag_ms: float, bin_ms: float) -> pd.DataFrame:
    """Auto-correlation histogram for periodicity (p.10): histogram of
    every pairwise time difference between events (both signs) out to
    +/-max_lag_ms."""
    t = np.sort(np.asarray(event_times_s, dtype=float)) * 1e3  # ms
    diffs: list[float] = []
    for i in range(len(t)):
        j = i + 1
        while j < len(t) and t[j] - t[i] <= max_lag_ms:
            d = t[j] - t[i]
            diffs.append(d)
            diffs.append(-d)
            j += 1
    edges = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms)
    counts, edges = np.histogram(diffs, bins=edges)
    return pd.DataFrame({"lag_start_ms": edges[:-1], "lag_center_ms": edges[:-1] + bin_ms / 2, "count": counts})


def cross_correlation_histogram(times_a_s, times_b_s, max_lag_ms: float, bin_ms: float) -> pd.DataFrame:
    """Cross-correlation histogram for synaptic connection between a pair
    (p.10): histogram of every t_b - t_a pairwise time difference (events B
    relative to events A) out to +/-max_lag_ms."""
    a = np.asarray(times_a_s, dtype=float) * 1e3
    b = np.sort(np.asarray(times_b_s, dtype=float)) * 1e3
    diffs: list[np.ndarray] = []
    for ta in a:
        i0, i1 = np.searchsorted(b, [ta - max_lag_ms, ta + max_lag_ms])
        diffs.append(b[i0:i1] - ta)
    all_diffs = np.concatenate(diffs) if diffs else np.array([])
    edges = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms)
    counts, edges = np.histogram(all_diffs, bins=edges)
    return pd.DataFrame({"lag_start_ms": edges[:-1], "lag_center_ms": edges[:-1] + bin_ms / 2, "count": counts})


# ---------------------------------------------------------------------------
# Analysis III: Group Analysis -- display of grouped traces (PDF p.11-12)
# ---------------------------------------------------------------------------

def extract_event_traces(v: np.ndarray, dt: float, events: list[Event], pre_ms: float, post_ms: float):
    """Peak-centered raw windows for a group of events -- p.11 "Analysis
    III: Group Analysis": "Display of grouped traces (raw or scaled)".
    Returns (t_ms, traces, baselines, amplitudes); events too close to
    either recording edge are silently dropped, so the returned arrays may
    have fewer rows than `events`. baselines/amplitudes come from each
    Event's own already-computed detection values, not re-derived here."""
    pre_n = _samples(pre_ms, dt)
    post_n = _samples(post_ms, dt)
    t_ms = np.arange(-pre_n, post_n) * dt * 1e3
    rows, baselines, amplitudes = [], [], []
    for e in events:
        i0, i1 = e.peak_idx - pre_n, e.peak_idx + post_n
        if i0 < 0 or i1 > len(v):
            continue
        rows.append(v[i0:i1])
        baselines.append(e.baseline)
        amplitudes.append(e.amplitude)
    traces = np.array(rows) if rows else np.empty((0, pre_n + post_n))
    return t_ms, traces, np.array(baselines), np.array(amplitudes)


def scale_traces(traces: np.ndarray, baselines: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
    """Baseline-subtract each trace and divide by its OWN detected
    amplitude, so events of different sizes overlay at unit peak
    deflection -- p.12 "Display of Grouped Traces": "Scaled Superimposed" /
    "Scaled Averaged"."""
    safe_amp = np.where(amplitudes == 0, 1.0, amplitudes)
    return (traces - baselines[:, None]) / safe_amp[:, None]


# ---------------------------------------------------------------------------
# Exponential Decay Fitting (PDF p.13): Simplex (Nelder-Mead) minimization
# of sum-of-squares, single or double exponential, selectable fit range.
# ---------------------------------------------------------------------------

@dataclass
class DecayFit:
    n_exp: int
    amplitudes: list  # [A1] or [A1, A2]
    taus_ms: list      # [tau1] or [tau1, tau2], ms
    offset: float       # C, steady-state asymptote
    residual_sd: float
    n_iterations: int
    fit_range_ms: tuple  # (start_ms, end_ms), relative to the trace's own t_ms[0]

    def equation_str(self) -> str:
        terms = "+".join(f"({a:.4g})*exp(-t/{tau:.4g}ms)" for a, tau in zip(self.amplitudes, self.taus_ms))
        return f"y={terms}+({self.offset:.4g})"

    def predict(self, t_ms: np.ndarray) -> np.ndarray:
        y = np.full_like(np.asarray(t_ms, dtype=float), self.offset, dtype=float)
        for a, tau in zip(self.amplitudes, self.taus_ms):
            y = y + a * np.exp(-np.asarray(t_ms, dtype=float) / tau)
        return y


def _decay_fit_bounds(t_ms: np.ndarray, y: np.ndarray, mode: str,
                       custom_start_ms: Optional[float] = None, custom_end_ms: Optional[float] = None,
                       direction: str = "negative") -> tuple[int, int]:
    """Resolve a named fit range to (start_idx, end_idx) into t_ms/y, where
    t_ms[0] is the peak -- p.13 "Flexible fitting range": "Peak to end /
    Decay 10-90 / Decay 20-80 / Custom range"."""
    pol = -1.0 if direction == "negative" else 1.0
    sy = pol * y
    speak = float(sy[0]) if len(sy) else 0.0

    if mode == "peak_to_end":
        return 0, len(t_ms) - 1
    if mode == "custom":
        if custom_start_ms is None or custom_end_ms is None:
            raise ValueError("fit_range='custom' requires custom_start_ms and custom_end_ms")
        i0 = int(np.searchsorted(t_ms, custom_start_ms))
        i1 = int(np.searchsorted(t_ms, custom_end_ms))
        i0 = max(0, min(i0, len(t_ms) - 2))
        i1 = max(i0 + 1, min(i1, len(t_ms) - 1))
        return i0, i1
    if mode in ("decay_10_90", "decay_20_80"):
        hi_frac, lo_frac = (0.9, 0.1) if mode == "decay_10_90" else (0.8, 0.2)
        hi_level, lo_level = hi_frac * speak, lo_frac * speak
        i0 = i1 = None
        for k in range(len(sy)):
            if i0 is None and sy[k] <= hi_level:
                i0 = k
            if i0 is not None and sy[k] <= lo_level:
                i1 = k
                break
        i0 = 0 if i0 is None else i0
        i1 = (len(t_ms) - 1) if i1 is None else i1
        return i0, max(i0 + 1, i1)
    raise ValueError(f"unknown fit_range {mode!r}")


def fit_exponential_decay(t_ms: np.ndarray, y: np.ndarray, n_exp: int = 1,
                           fit_range: Literal["peak_to_end", "decay_10_90", "decay_20_80", "custom"] = "peak_to_end",
                           custom_start_ms: Optional[float] = None, custom_end_ms: Optional[float] = None,
                           direction: Literal["negative", "positive"] = "negative",
                           p0: Optional["DecayFit"] = None) -> DecayFit:
    """Fit y(t) ~= sum_i A_i*exp(-t/tau_i) + C to the peak-to-tail portion
    of one (typically averaged) event trace -- p.13 "Exponential Decay
    Fitting": "Using Simplex fitting algorithm. Minimization of
    sum-of-squares", "Single and double exponential fitting", "Flexible
    fitting range", and "Ability to use last fit results as initial
    guesses" (pass a previous DecayFit as `p0`).

    `t_ms`/`y` must be peak-aligned (t_ms[0] == 0, e.g. from
    extract_event_traces's averaged output). Uses
    scipy.optimize.minimize(method="Nelder-Mead") -- the Nelder-Mead
    downhill Simplex the tutorial names -- on raw sum-of-squared residuals,
    not a gradient-based least-squares solver.
    """
    if n_exp not in (1, 2):
        raise ValueError("n_exp must be 1 or 2")
    i0, i1 = _decay_fit_bounds(t_ms, y, fit_range, custom_start_ms, custom_end_ms, direction)
    tt = np.asarray(t_ms[i0:i1 + 1], dtype=float) - t_ms[i0]
    yy = np.asarray(y[i0:i1 + 1], dtype=float)

    peak_amp = float(yy[0])
    tail = float(np.median(yy[-max(1, len(yy) // 10):]))
    span = max(tt[-1], 1e-3)

    if p0 is not None and p0.n_exp == n_exp:
        x0 = np.array([*p0.amplitudes, *p0.taus_ms, p0.offset], dtype=float)
    elif n_exp == 1:
        x0 = np.array([peak_amp - tail, span / 3.0, tail], dtype=float)
    else:
        x0 = np.array([0.7 * (peak_amp - tail), span / 6.0, 0.3 * (peak_amp - tail), span / 2.0, tail], dtype=float)

    def unpack(x):
        amps = x[0:n_exp]
        taus = np.abs(x[n_exp:2 * n_exp]) + 1e-6
        offset = x[-1]
        return amps, taus, offset

    def sse(x):
        amps, taus, offset = unpack(x)
        pred = np.full_like(tt, offset, dtype=float)
        for a, tau in zip(amps, taus):
            pred = pred + a * np.exp(-tt / tau)
        return float(np.sum((yy - pred) ** 2))

    result = minimize(sse, x0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000, "maxfev": 5000})
    amps, taus, offset = unpack(result.x)
    n, k = len(yy), 2 * n_exp + 1
    residual_sd = float(np.sqrt(result.fun / max(1, n - k)))
    return DecayFit(n_exp=n_exp, amplitudes=amps.tolist(), taus_ms=taus.tolist(), offset=float(offset),
                     residual_sd=residual_sd, n_iterations=int(result.nit),
                     fit_range_ms=(float(t_ms[i0]), float(t_ms[i1])))


# ---------------------------------------------------------------------------
# --no-gui (default OFF, i.e. the dialog shows unless disabled): a modal
# parameters dialog, pre-filled with defaults (or whatever --amplitude-
# threshold/--filter/etc. were also passed), shown before detection runs --
# the tutorial's own "enter parameters, click OK" workflow (PDF p.4: "First
# use mouse-click to detect"). Covers both the 9 detection parameters AND
# the filter/resample settings (PDF's Bessel low-pass step doesn't have a
# tutorial page of its own, but belongs in the same up-front dialog since
# it's the other thing you'd want to set before detection runs). PyQt5 is
# imported lazily, INSIDE this function, so `--no-gui` runs (scripts,
# automation) never need PyQt5 installed at all.
# ---------------------------------------------------------------------------

_PARAM_FIELD_SPECS = [
    # (attr, label, widget kind, kwargs for the spin box)
    ("amplitude_threshold", "Amplitude threshold (a)", "double", dict(minimum=0.0, maximum=1e6, decimals=2, singleStep=0.5)),
    ("area_threshold", "Area threshold (b)", "double", dict(minimum=0.0, maximum=1e7, decimals=2, singleStep=0.5)),
    ("direction", "Peak direction", "choice", dict(choices=["negative", "positive"])),
    ("n_avg_peak", "Number of points to average peak", "int", dict(minimum=1, maximum=1000)),
    ("search_local_max_ms", "Period to search local maximum (c), ms", "double", dict(minimum=0.01, maximum=100000.0, decimals=2, singleStep=0.5)),
    ("baseline_before_ms", "Time before peak for baseline (d), ms", "double", dict(minimum=0.0, maximum=100000.0, decimals=2, singleStep=0.5)),
    ("baseline_avg_ms", "Period to average baseline (e), ms", "double", dict(minimum=0.01, maximum=100000.0, decimals=2, singleStep=0.5)),
    ("decay_search_ms", "Period to search decay time (f), ms", "double", dict(minimum=0.01, maximum=100000.0, decimals=2, singleStep=1.0)),
    ("decay_fraction", "Fraction to find decay (g)", "double", dict(minimum=0.01, maximum=0.99, decimals=2, singleStep=0.05)),
    ("onset_fraction", "Onset fraction (rise crossing)", "double", dict(minimum=0.01, maximum=0.99, decimals=2, singleStep=0.05)),
    ("onset_search_ms", "Period to search onset before peak, ms", "double", dict(minimum=0.01, maximum=100000.0, decimals=2, singleStep=0.5)),
    ("adjust_overlapping_baseline", "Adjust baseline for overlapping events", "bool", dict()),
]



@dataclass
class GuiSettings:
    params: DetectionParams
    filter_enabled: bool
    cutoff_hz: float
    target_rate_hz: float
    filter_order: int


def _prompt_for_settings(defaults: DetectionParams, filter_enabled: bool, cutoff_hz: float,
                          target_rate_hz: float, filter_order: int) -> Optional[GuiSettings]:
    """Modal dialog covering both the 9 detection parameters and the
    filter/resample settings, pre-filled from the arguments above --
    returns a GuiSettings on OK, or None if the user cancelled (caller
    should abort without running detection)."""
    from PyQt5 import QtWidgets

    # Must keep a live reference -- an unassigned QApplication([]) here can
    # be garbage-collected before the QDialog below is constructed (no
    # QApplication existed yet in a plain `python -m sepsc.minianalysis ...`
    # run, unlike in tests that pre-create one), which crashes with "QWidget:
    # Must construct a QApplication before a QWidget".
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Mini Analysis -- Detection Parameters")
    layout = QtWidgets.QVBoxLayout(dialog)

    note = QtWidgets.QLabel(
        "Values are pre-filled with defaults (or any --flags already passed).\n"
        "Leave a field as-is to use that default.")
    note.setWordWrap(True)
    layout.addWidget(note)

    param_group = QtWidgets.QGroupBox("Detection parameters")
    param_form = QtWidgets.QFormLayout(param_group)
    param_widgets = {}
    for attr, label, kind, kwargs in _PARAM_FIELD_SPECS:
        w = make_field_widget(kind, kwargs, getattr(defaults, attr))
        param_widgets[attr] = w
        param_form.addRow(label + ":", w)
    layout.addWidget(param_group)

    filter_group = QtWidgets.QGroupBox("Filter / resample (sepsc.preprocess)")
    filter_form = QtWidgets.QFormLayout(filter_group)
    enabled_w = QtWidgets.QCheckBox()
    enabled_w.setChecked(filter_enabled)
    filter_form.addRow("Apply Bessel low-pass + resample:", enabled_w)
    filter_widgets = {}
    filter_current = dict(cutoff_hz=cutoff_hz, target_rate_hz=target_rate_hz, filter_order=filter_order)
    for attr, label, kind, kwargs in FILTER_FIELD_SPECS:
        w = make_field_widget(kind, kwargs, filter_current[attr])
        w.setEnabled(filter_enabled)
        enabled_w.toggled.connect(w.setEnabled)
        filter_widgets[attr] = w
        filter_form.addRow(label + ":", w)
    layout.addWidget(filter_group)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(440)

    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None

    params = DetectionParams(**{
        attr: read_field_widget(param_widgets[attr], kind) for attr, _, kind, _ in _PARAM_FIELD_SPECS
    })
    filter_values = {attr: read_field_widget(filter_widgets[attr], kind) for attr, _, kind, _ in FILTER_FIELD_SPECS}
    return GuiSettings(params=params, filter_enabled=enabled_w.isChecked(), **filter_values)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", help="Path to the gap-free voltage-clamp .abf file")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--direction", choices=["negative", "positive"], default="negative",
                   help="peak direction; sEPSCs in voltage clamp are inward/negative (default)")
    p.add_argument("--amplitude-threshold", type=float, default=5.0,
                   help="(a) minimum |amplitude| to accept, in trace units (default 5)")
    p.add_argument("--area-threshold", type=float, default=10.0,
                   help="(b) minimum |area| to accept, in trace-units*ms (default 10)")
    p.add_argument("--n-avg-peak", type=int, default=3,
                   help="points averaged to read the peak value (default 3)")
    p.add_argument("--search-local-max-ms", type=float, default=3.0,
                   help="(c) period to search for a local maximum, in ms (default 3 -- should be "
                        "at least comparable to your events' decay duration, or a single event's "
                        "noisy tail can fragment into multiple spurious detections)")
    p.add_argument("--baseline-before-ms", type=float, default=2.0,
                   help="(d) time before the peak where the baseline window ends, in ms (default 2)")
    p.add_argument("--baseline-avg-ms", type=float, default=3.0,
                   help="(e) duration of the baseline-averaging window, in ms (default 3)")
    p.add_argument("--decay-search-ms", type=float, default=20.0,
                   help="(f) how far past the peak to search for the decay point, in ms (default 20)")
    p.add_argument("--decay-fraction", type=float, default=0.5,
                   help="(g) fraction of peak amplitude defining the decay point (default 0.5 = half-decay)")
    p.add_argument("--onset-fraction", type=float, default=0.1,
                   help="fraction of peak amplitude defining rise onset, for time-to-peak (default 0.1)")
    p.add_argument("--onset-search-ms", type=float, default=5.0,
                   help="how far before the peak to search for the onset_fraction crossing, in ms "
                        "(default 5) -- the rise-side counterpart of decay_search_ms (f); if the "
                        "onset amplitude is never reached within this window, the candidate is "
                        "rejected outright, same as an unreached decay point")
    p.add_argument("--adjust-overlapping-baseline", action="store_true",
                   help="opt into slide 0007's exponential-decay baseline extrapolation for "
                        "closely-spaced events. Off by default: baseline is then ALWAYS the plain "
                        "(d)/(e) window average as those two parameters describe, nothing else; "
                        "turning this on REPLACES that window average with an extrapolated value "
                        "for events still riding a previous one's decay tail")
    p.add_argument("--filter", action="store_true",
                   help="Bessel low-pass + downsample the trace (see sepsc.preprocess) before "
                        "detection, instead of detecting on the raw trace -- lower-noise input "
                        "can reduce spurious local-maximum detections on a noisy recording, at "
                        "the cost of attenuating genuinely fast/small events")
    p.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ,
                   help=f"only with --filter: Bessel low-pass cutoff, Hz (default {DEFAULT_CUTOFF_HZ:.0f})")
    p.add_argument("--target-rate-hz", type=float, default=DEFAULT_TARGET_RATE_HZ,
                   help=f"only with --filter: output sampling rate, Hz (default {DEFAULT_TARGET_RATE_HZ:.0f})")
    p.add_argument("--filter-order", type=int, default=DEFAULT_ORDER,
                   help=f"only with --filter: Bessel filter order (default {DEFAULT_ORDER})")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-gui", dest="gui", action="store_false", default=True,
                   help="skip the parameters dialog and run immediately from the flags/defaults "
                        "above -- for scripts/automation. By default (no --no-gui) a dialog shows "
                        "to review/edit the 9 detection parameters (PDF p.2) plus the filter/resample "
                        "settings, pre-filled with the values above (or their defaults), before "
                        "running -- the tutorial's own 'enter parameters, click OK' workflow "
                        "(this_method.pdf p.4). Cancel aborts without detecting.")

    analysis = p.add_argument_group(
        "Analysis (PDF p.7-13)", "grouping/data-array, descriptive, and group analyses on the detected events")
    analysis.add_argument("--stats", action="store_true",
                   help="save column statistics (n, mean, variance, sd, skewness, kurtosis -- p.10) "
                        "for amplitude/area/rise_time_ms/decay_time_ms/inter_event_interval_ms")
    analysis.add_argument("--histogram-column",
                   choices=["amplitude", "area", "rise_time_ms", "decay_time_ms", "inter_event_interval_ms"],
                   default=None, help="save a frequency + cumulative histogram (p.10) for this column")
    analysis.add_argument("--hist-bin-size", type=float, default=None,
                   help="bin width for --histogram-column, in that column's own units "
                        "(default: auto, data range / 30)")
    analysis.add_argument("--autocorr", action="store_true",
                   help="save an auto-correlation histogram of event times, for periodicity (p.10)")
    analysis.add_argument("--cross-corr-csv", default=None,
                   help="path to another *_events.csv (must have a peak_time_s or location_s column) "
                        "-- save a cross-correlation histogram against THIS run's events (p.10)")
    analysis.add_argument("--corr-max-lag-ms", type=float, default=500.0,
                   help="+/- window for --autocorr/--cross-corr-csv, ms (default 500)")
    analysis.add_argument("--corr-bin-ms", type=float, default=5.0,
                   help="bin width for --autocorr/--cross-corr-csv, ms (default 5)")
    analysis.add_argument("--group-analysis", action="store_true",
                   help="save a superimposed/averaged (raw and scaled) trace plot across all "
                        "detected events (p.11-12)")
    analysis.add_argument("--group-pre-ms", type=float, default=5.0,
                   help="only with --group-analysis: window before each peak, ms (default 5)")
    analysis.add_argument("--group-post-ms", type=float, default=20.0,
                   help="only with --group-analysis: window after each peak, ms (default 20)")
    analysis.add_argument("--fit-decay", choices=["peak_to_end", "decay_10_90", "decay_20_80", "custom"],
                   default=None, help="fit a single/double exponential decay (Simplex, p.13) to the "
                        "averaged trace from --group-analysis (implies --group-analysis)")
    analysis.add_argument("--fit-n-exp", type=int, choices=[1, 2], default=1,
                   help="only with --fit-decay: number of exponential components (default 1)")
    analysis.add_argument("--fit-custom-start-ms", type=float, default=None,
                   help="only with --fit-decay custom: fit-window start, ms relative to the peak")
    analysis.add_argument("--fit-custom-end-ms", type=float, default=None,
                   help="only with --fit-decay custom: fit-window end, ms relative to the peak")
    args = p.parse_args(argv)
    if args.fit_decay is not None:
        args.group_analysis = True

    params = DetectionParams(
        amplitude_threshold=args.amplitude_threshold,
        area_threshold=args.area_threshold,
        direction=args.direction,
        n_avg_peak=args.n_avg_peak,
        search_local_max_ms=args.search_local_max_ms,
        baseline_before_ms=args.baseline_before_ms,
        baseline_avg_ms=args.baseline_avg_ms,
        decay_search_ms=args.decay_search_ms,
        decay_fraction=args.decay_fraction,
        onset_fraction=args.onset_fraction,
        onset_search_ms=args.onset_search_ms,
        adjust_overlapping_baseline=args.adjust_overlapping_baseline,
    )

    if args.gui:
        settings = _prompt_for_settings(params, args.filter, args.cutoff_hz,
                                         args.target_rate_hz, args.filter_order)
        if settings is None:
            print("Cancelled in the parameters dialog -- no detection run.")
            return
        params = settings.params
        args.filter = settings.filter_enabled
        args.cutoff_hz = settings.cutoff_hz
        args.target_rate_hz = settings.target_rate_hz
        args.filter_order = settings.filter_order

    abf = pyabf.ABF(args.abf)
    abf.setSweep(0, channel=args.channel)

    if args.filter:
        hardware_filter_hz = get_hardware_filter_hz(abf, args.channel)
        if hardware_filter_hz is not None and hardware_filter_hz <= args.cutoff_hz:
            print(f"NOTE: channel {args.channel} is already hardware-filtered at "
                  f"{hardware_filter_hz:.0f} Hz (amplifier's telegraphed setting), at or below "
                  f"the requested {args.cutoff_hz:.0f} Hz -- filtering further is a no-op.")
        t, v, fs, hardware_filter_hz = load_filtered_trace(
            args.abf, channel=args.channel, cutoff_hz=args.cutoff_hz,
            target_hz=args.target_rate_hz, order=args.filter_order)
        dt = 1.0 / fs
        sweep_len_s = len(v) * dt
        print(f"Filtered: {abf.dataRate:.0f} Hz raw -> {args.cutoff_hz:.0f} Hz Bessel "
              f"(order {args.filter_order}, zero-phase) -> {fs:.0f} Hz", flush=True)

        # Robust (MAD-based, so a few genuinely large events don't inflate
        # it) noise-floor estimate on the FILTERED trace -- filtering
        # correlates adjacent samples, so a filtered noise bump survives
        # this detector's n_avg_peak averaging and area check far more
        # easily than a raw noise spike does. amplitude_threshold/
        # area_threshold tuned against raw-trace noise silently stop being
        # a meaningful bar once the trace is filtered; only warn (never
        # override -- the right value is a real scientific choice, not
        # something to guess on the caller's behalf).
        noise_sd = float(np.median(np.abs(v - np.median(v))) * 1.4826)
        if args.amplitude_threshold < 3 * noise_sd:
            print(f"WARNING: --amplitude-threshold {args.amplitude_threshold:.1f} pA is only "
                  f"{args.amplitude_threshold / noise_sd:.1f}x the filtered trace's noise SD "
                  f"(~{noise_sd:.1f} pA, robust estimate) -- thresholds tuned for the RAW trace "
                  f"don't transfer to filtered data (filtering correlates adjacent noise samples, "
                  f"so noise survives the amplitude/area checks more easily, not less). Consider "
                  f"--amplitude-threshold >= {3 * noise_sd:.0f} (3x SD) as a starting point, and "
                  f"re-tune --area-threshold too.", flush=True)
    else:
        t = np.asarray(abf.sweepX, float)
        v = np.asarray(abf.sweepY, float)
        dt = 1.0 / abf.dataRate
        fs = abf.dataRate
        sweep_len_s = abf.sweepLengthSec

    print(f"Detecting events in {os.path.basename(args.abf)} "
          f"({sweep_len_s:.1f}s @ {fs:.0f} Hz) "
          f"using the Mini-Analysis-style detector", flush=True)

    events = detect_events(t, v, dt, params)

    stem = os.path.splitext(args.abf)[0]
    if args.filter:
        stem += f"_filt{int(args.cutoff_hz)}Hz{int(args.target_rate_hz)}Hz"
    out_csv = f"{stem}_minianalysis_events.csv"
    df = events_frame(events)  # adds inter_event_interval_ms (p.9 "Data Arrays")
    df.to_csv(out_csv, index=False)

    # Sidecar with the exact DetectionParams used -- sepsc.inspect reads this
    # so its click-to-verify view shows the REAL parameters behind these
    # events, not just today's argparse defaults.
    params_path = f"{stem}_minianalysis_params.json"
    with open(params_path, "w") as fh:
        json.dump(asdict(params), fh, indent=2)

    if events:
        rate_hz = len(events) / sweep_len_s
        amps = np.array([e.amplitude for e in events])
        print(f"Detected {len(events)} events ({rate_hz:.3f} Hz), "
              f"mean amplitude {amps.mean():.2f} {abf.adcUnits[args.channel]}, "
              f"median {np.median(amps):.2f}")
    else:
        print("Detected 0 events -- check direction/thresholds.")
    print(f"Saved -> {out_csv}")

    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(16, 4.5))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        ax.plot(t, v, color=TRACE, lw=0.3, zorder=1)
        if events:
            peak_t = np.array([e.peak_time_s for e in events])
            peak_v = v[np.array([e.peak_idx for e in events])]
            ax.plot(peak_t, peak_v, "x", color="crimson", ms=6, mew=1.2, zorder=3,
                     label=f"{len(events)} detected events")
            ax.legend(loc="upper right", fontsize=9, labelcolor=INK)
        ax.set_xlabel("time (s)", color=INK)
        ax.set_ylabel(f"current ({abf.adcUnits[args.channel]})", color=INK)
        ax.set_title(f"{os.path.basename(args.abf)} -- Mini-Analysis-style detection", color=INK)
        ax.tick_params(colors=MUTED)
        fig.tight_layout()
        out_png = f"{stem}_minianalysis_trace.png"
        fig.savefig(out_png, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved -> {out_png}")

    # -- Analysis stage (PDF p.7-13) -------------------------------------
    if not events and (args.stats or args.histogram_column or args.autocorr
                        or args.cross_corr_csv or args.group_analysis):
        print("No events detected -- skipping requested analysis steps.", flush=True)
        return

    if args.stats:
        rows = {}
        for col in ["amplitude", "area", "rise_time_ms", "decay_time_ms", "inter_event_interval_ms"]:
            rows[col] = vars(column_statistics(df[col]))
        stats_df = pd.DataFrame(rows).T
        stats_df.index.name = "metric"
        out = f"{stem}_minianalysis_stats.csv"
        stats_df.to_csv(out)
        print(f"Saved -> {out}")
        print(stats_df.round(3).to_string())

    if args.histogram_column:
        col = args.histogram_column
        freq = frequency_histogram(df[col], args.hist_bin_size)
        cum = cumulative_histogram(df[col], args.hist_bin_size)
        out_csv2 = f"{stem}_minianalysis_hist_{col}.csv"
        freq.assign(cumulative_fraction=cum["cumulative_fraction"]).to_csv(out_csv2, index=False)
        print(f"Saved -> {out_csv2}")
        if not args.no_plot and len(freq):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            for ax in (ax1, ax2):
                fig.patch.set_facecolor(SURFACE)
                ax.set_facecolor(SURFACE)
                ax.tick_params(colors=MUTED)
            bin_w = float(freq["bin_center"].diff().median()) if len(freq) > 1 else 1.0
            ax1.bar(freq["bin_center"], freq["count"], width=bin_w, color=TRACE)
            ax1.set_xlabel(col, color=INK)
            ax1.set_ylabel("count", color=INK)
            ax1.set_title("Frequency histogram", color=INK)
            ax2.plot(cum["bin_center"], cum["cumulative_fraction"], color=TRACE)
            ax2.set_xlabel(col, color=INK)
            ax2.set_ylabel("cumulative fraction", color=INK)
            ax2.set_title("Cumulative histogram", color=INK)
            fig.tight_layout()
            out_png2 = f"{stem}_minianalysis_hist_{col}.png"
            fig.savefig(out_png2, dpi=130, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Saved -> {out_png2}")

    if args.autocorr:
        ac = autocorrelation_histogram(df["peak_time_s"], args.corr_max_lag_ms, args.corr_bin_ms)
        out_csv3 = f"{stem}_minianalysis_autocorr.csv"
        ac.to_csv(out_csv3, index=False)
        print(f"Saved -> {out_csv3}")
        if not args.no_plot and len(ac):
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor(SURFACE)
            ax.set_facecolor(SURFACE)
            ax.bar(ac["lag_center_ms"], ac["count"], width=args.corr_bin_ms, color=TRACE)
            ax.set_xlabel("lag (ms)", color=INK)
            ax.set_ylabel("count", color=INK)
            ax.set_title("Auto-correlation histogram", color=INK)
            ax.tick_params(colors=MUTED)
            fig.tight_layout()
            out_png3 = f"{stem}_minianalysis_autocorr.png"
            fig.savefig(out_png3, dpi=130, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Saved -> {out_png3}")

    if args.cross_corr_csv:
        other = pd.read_csv(args.cross_corr_csv)
        time_col = "peak_time_s" if "peak_time_s" in other.columns else "location_s"
        cc = cross_correlation_histogram(df["peak_time_s"], other[time_col],
                                          args.corr_max_lag_ms, args.corr_bin_ms)
        out_csv4 = f"{stem}_minianalysis_crosscorr.csv"
        cc.to_csv(out_csv4, index=False)
        print(f"Saved -> {out_csv4}")
        if not args.no_plot and len(cc):
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor(SURFACE)
            ax.set_facecolor(SURFACE)
            ax.bar(cc["lag_center_ms"], cc["count"], width=args.corr_bin_ms, color=TRACE)
            ax.set_xlabel("lag, other - this (ms)", color=INK)
            ax.set_ylabel("count", color=INK)
            ax.set_title(f"Cross-correlation vs. {os.path.basename(args.cross_corr_csv)}", color=INK)
            ax.tick_params(colors=MUTED)
            fig.tight_layout()
            out_png4 = f"{stem}_minianalysis_crosscorr.png"
            fig.savefig(out_png4, dpi=130, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Saved -> {out_png4}")

    if args.group_analysis:
        t_ms, traces, baselines, amplitudes = extract_event_traces(
            v, dt, events, args.group_pre_ms, args.group_post_ms)
        if len(traces) == 0:
            print("Group analysis: no events far enough from the recording edges to extract -- skipped.")
        else:
            scaled = scale_traces(traces, baselines, amplitudes)
            avg_raw, avg_scaled = traces.mean(axis=0), scaled.mean(axis=0)
            print(f"Group analysis: {len(traces)} traces "
                  f"({args.group_pre_ms:.1f}ms before / {args.group_post_ms:.1f}ms after peak)")

            decay_fit = None
            if args.fit_decay is not None:
                pre_n = _samples(args.group_pre_ms, dt)
                t_decay, y_decay = t_ms[pre_n:], avg_raw[pre_n:]
                decay_fit = fit_exponential_decay(
                    t_decay, y_decay, n_exp=args.fit_n_exp, fit_range=args.fit_decay,
                    custom_start_ms=args.fit_custom_start_ms, custom_end_ms=args.fit_custom_end_ms,
                    direction=args.direction)
                summary = (
                    f"Number of traces averaged: {len(traces)}\n"
                    f"# of exponentials: {decay_fit.n_exp}   Fit range: {args.fit_decay} "
                    f"({decay_fit.fit_range_ms[0]:.2f} to {decay_fit.fit_range_ms[1]:.2f} ms post-peak)\n"
                    f"{decay_fit.equation_str()}\n"
                    f"Std deviation: {decay_fit.residual_sd:.4g}   Iterations: {decay_fit.n_iterations}")
                print(summary)
                out_txt = f"{stem}_minianalysis_decayfit.txt"
                with open(out_txt, "w") as fh:
                    fh.write(summary + "\n")
                print(f"Saved -> {out_txt}")

            if not args.no_plot:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
                fig.patch.set_facecolor(SURFACE)
                titles = [["Superimposed", "Scaled Superimposed"], ["Averaged", "Scaled Averaged"]]
                data = [[traces, scaled], [avg_raw[None, :], avg_scaled[None, :]]]
                for r in range(2):
                    for c in range(2):
                        ax = axes[r, c]
                        ax.set_facecolor(SURFACE)
                        for row in data[r][c]:
                            ax.plot(t_ms, row, color=TRACE, lw=(0.3 if r == 0 else 1.2), alpha=(0.3 if r == 0 else 1.0))
                        ax.set_title(titles[r][c], color=INK)
                        ax.tick_params(colors=MUTED)
                        if r == 1:
                            ax.set_xlabel("time from peak (ms)", color=INK)
                if decay_fit is not None:
                    pre_n = _samples(args.group_pre_ms, dt)
                    fit_t = t_ms[pre_n:]
                    axes[1, 0].plot(fit_t, decay_fit.predict(fit_t - fit_t[0]), color="crimson", lw=1.0,
                                     label=f"{decay_fit.n_exp}-exp fit")
                    axes[1, 0].legend(fontsize=8, labelcolor=INK)
                fig.suptitle(f"{os.path.basename(args.abf)} -- group analysis ({len(traces)} events)", color=INK)
                fig.tight_layout()
                out_png5 = f"{stem}_minianalysis_group.png"
                fig.savefig(out_png5, dpi=130, facecolor=fig.get_facecolor())
                plt.close(fig)
                print(f"Saved -> {out_png5}")


if __name__ == "__main__":
    main()
