#!/usr/bin/env python3
"""
batch_sepsc.py
==============

Automated sEPSC detection for the batch pipeline, by deconvolution.

One gap-free voltage-clamp recording per cell goes in; a table of detected
events, a set of aligned event waveforms, and a row of summary measurements
come out. ``batch_cell_analysis.py`` runs this as one of its per-cell stages,
and ``batch_qc_window.py`` reads what it writes.

The detection is not implemented here
-------------------------------------
It is ``sEPSC_multimethod``'s deconvolution detector (Pernia-Andrade et al.
2012, Biophys J 103:1429), imported from the sibling package rather than
reimplemented, so a count from a batch run and a count from the standalone
tool are the same number rather than two numbers that ought to agree. That
detector estimates the event waveform from the recording itself and thresholds
in units of measured noise, which is what makes it usable unattended: there is
nothing to tune per cell.

This module is therefore a thin wrapper that decides what to run it on, what
to write, and which measurements describe the QUALITY of a recording rather
than its biology.

What is written, per cell
-------------------------
    <sepsc>_sEPSC_events.csv       one row per detected event
    <sepsc>_sEPSC_waveforms.npz    aligned waveforms for the QC window

Deliberately NOT the ``_deconvolution_events.csv`` name the standalone
multimethod tool writes. The two tools would otherwise overwrite each other's
output for the same recording, silently, and the reviewed output of a careful
manual pass is not something a batch re-run should be able to destroy.

Quality, not biology
--------------------
The summary columns are chosen to answer "is this recording usable?" rather
than "what is this cell doing?":

  * ``sepsc_noise_pA`` and ``sepsc_detection_limit_pA`` -- the noise floor, and
    4x it, below which an event cannot be distinguished from noise at all
    (Greger & Watson). A cell whose median event sits near its own detection
    limit has a rate that means very little.
  * ``sepsc_snr`` -- median event amplitude over that noise. The single most
    useful number here.
  * ``sepsc_holding_drift_pA`` -- how far the holding current moved from the
    start of the recording to the end. A seal that is failing shows here
    first, and it invalidates the amplitudes either side of it.
  * ``sepsc_amp_trend_pct`` -- how far the median amplitude moved over the
    same span, which is what run-down looks like in the events themselves.
  * ``sepsc_kernel_source`` -- whether the detector estimated the event
    waveform from this recording or fell back to its analytic one. A fallback
    means too few isolated events were found, and the detection should be
    treated as provisional.

Amplitudes in the summary are magnitudes in pA (positive). The events CSV
keeps the detector's signed convention, negative for inward currents.

Usage
-----
    python batch_sepsc.py RECORDING.abf         # detect, write, report

Requires: numpy, pandas, scipy, pyabf, and the ``sEPSC_multimethod`` package
in the same folder as this script. It is a git submodule, so a fresh clone
has the folder but not its contents until::

    git submodule update --init

Without it every other stage still runs and this one reports what is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

#: The sibling package that does the actual detection.
MULTIMETHOD_DIR = Path(__file__).resolve().parent / "sEPSC_multimethod"

#: Window kept around each event peak for the overlay panels, ms.
WAVEFORM_PRE_MS = 5.0
WAVEFORM_POST_MS = 30.0

#: Most waveforms stored per cell. Enough to see the spread, small enough that
#: the QC window opens instantly and the .npz stays a few hundred kB.
WAVEFORM_MAX = 300

#: Fraction of the recording averaged at each end for the drift measurements.
DRIFT_FRACTION = 0.1


def _import_multimethod():
    """The sibling detection package: (trace, deconvolution, events) modules.

    Imported on demand rather than at module import, so that a machine without
    the package still runs every other stage -- the sEPSC stage then fails for
    that cell with a message saying what is missing, which is the same way
    every other stage failure is handled.
    """
    if not MULTIMETHOD_DIR.is_dir():
        raise ImportError(
            f"sEPSC detection needs the sEPSC_multimethod package at "
            f"{MULTIMETHOD_DIR}, which does not exist.")
    if str(MULTIMETHOD_DIR) not in sys.path:
        sys.path.insert(0, str(MULTIMETHOD_DIR))
    from multimethod import events, trace
    from multimethod.detectors import deconvolution
    return trace, deconvolution, events


def current_channel(path: Path) -> int:
    """The recorded current channel of an .abf, or 0 if none is labelled pA.

    Header-only: the sweep data is loaded once, later, by the detector.
    """
    import pyabf

    abf = pyabf.ABF(str(path), loadData=False)
    for ch in abf.channelList:
        if abf.adcUnits[ch].strip().lower() in ("pa", "na", "a"):
            return int(ch)
    return 0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect(path: Path, channel: Optional[int] = None,
           thresh_sd: Optional[float] = None) -> dict:
    """Detect and measure every event in one gap-free recording.

    Returns the preprocessed trace, the events table, the detector's own
    diagnostics and the noise estimate -- everything the summary and the
    waveforms are derived from, so the recording is read and filtered once.
    """
    trace_mod, deconv_mod, events_mod = _import_multimethod()

    if channel is None:
        channel = current_channel(path)
    tr = trace_mod.load(str(path), channel)

    params = deconv_mod.DeconvolutionParams()
    if thresh_sd is not None:
        params = deconv_mod.DeconvolutionParams(thresh_sd=float(thresh_sd))

    detector = deconv_mod.DeconvolutionDetector()
    idx = detector.find(tr, params)
    evs, n_unmeasurable = events_mod.measure_all(
        tr, idx, events_mod.MeasurementParams(), "deconvolution")

    diagnostics = dict(getattr(detector, "last_diagnostics", {}))
    # The kernel is an array, not a number; it has no place in a summary row.
    diagnostics.pop("kernel", None)
    return {
        "path": Path(path),
        "trace": tr,
        "events": events_mod.events_frame(evs),
        "n_unmeasurable": int(n_unmeasurable),
        "diagnostics": diagnostics,
        "noise_pA": float(trace_mod.noise_sd(tr)),
        "thresh_sd": float(params.thresh_sd),
    }


# ---------------------------------------------------------------------------
# Event waveforms
# ---------------------------------------------------------------------------
def event_waveforms(trace, events: pd.DataFrame,
                    pre_ms: float = WAVEFORM_PRE_MS,
                    post_ms: float = WAVEFORM_POST_MS,
                    limit: int = WAVEFORM_MAX) -> dict:
    """Baseline-subtracted waveforms, aligned on their peaks.

    The drift baseline is subtracted before the event's own baseline, because
    that is what the event baselines in the events table were measured
    against. Skipping it leaves the holding current in every waveform -- tens
    of pA of constant offset, enough that no overlay sits on zero and every
    scaled trace is wrong by the same factor.

    When there are more events than `limit`, the ones kept are spread evenly
    through the recording rather than taken from the front. That matters for
    exactly the recordings this is meant to catch: a cell that ran down looks
    perfectly healthy in its first minute, and a first-`limit` sample would
    show only that minute.

    Events too close to either end to fill the window are skipped -- a short
    waveform padded with zeros would drag the mean towards baseline.
    """
    trace_mod, _deconv_mod, events_mod = _import_multimethod()
    mp = events_mod.MeasurementParams()
    v = np.asarray(trace.v, dtype=float)
    dt = float(trace.dt)
    if mp.subtract_drift:
        pct = 80.0 if mp.direction == "negative" else 20.0
        v = v - trace_mod.track_baseline(v, dt, mp.drift_win_ms, pct)
    pre_n = max(1, int(round(pre_ms / 1000.0 / dt)))
    post_n = max(2, int(round(post_ms / 1000.0 / dt)))

    if events is None or events.empty:
        return {"waveforms": np.zeros((0, pre_n + post_n)),
                "t_ms": (np.arange(pre_n + post_n) - pre_n) * dt * 1e3,
                "amplitude": np.zeros(0), "peak_time_s": np.zeros(0)}

    idx = events["peak_idx"].to_numpy(dtype=np.int64)
    keep = np.nonzero((idx >= pre_n) & (idx + post_n < v.size))[0]
    if keep.size > limit:
        keep = keep[np.linspace(0, keep.size - 1, limit).round().astype(int)]

    baselines = events["baseline"].to_numpy(dtype=float)
    waves = np.empty((keep.size, pre_n + post_n), dtype=np.float32)
    for k, j in enumerate(keep):
        i = int(idx[j])
        waves[k] = v[i - pre_n: i + post_n] - baselines[j]

    return {
        "waveforms": waves,
        "t_ms": (np.arange(pre_n + post_n) - pre_n) * dt * 1e3,
        "amplitude": events["amplitude"].to_numpy(dtype=float)[keep],
        "peak_time_s": events["peak_time_s"].to_numpy(dtype=float)[keep],
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _median(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if len(s) else float("nan")


def _ends(values: np.ndarray, fraction: float = DRIFT_FRACTION) -> tuple[float, float]:
    """Medians of the first and last `fraction` of a series, in order.

    NaN either side when there is not enough of it to be worth quoting.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    k = int(round(n * fraction))
    if n < 4 or k < 1:
        return float("nan"), float("nan")
    return float(np.median(values[:k])), float(np.median(values[-k:]))


def summarise(det: dict) -> dict:
    """The summary row: what says whether this recording is usable."""
    trace, events = det["trace"], det["events"]
    diag = det.get("diagnostics", {})
    noise = float(det.get("noise_pA", float("nan")))
    duration = float(trace.duration_s)

    amps = np.abs(pd.to_numeric(events.get("amplitude"), errors="coerce")
                  .dropna().to_numpy(float)) if len(events) else np.zeros(0)

    row = {
        "sepsc_n_events": int(len(events)),
        "sepsc_duration_s": duration,
        "sepsc_rate_Hz": len(events) / duration if duration > 0 else float("nan"),
        "sepsc_amp_median_pA": float(np.median(amps)) if amps.size else float("nan"),
        "sepsc_amp_iqr_pA": (float(np.subtract(*np.percentile(amps, [75, 25])))
                             if amps.size else float("nan")),
        "sepsc_rise_ms": _median(events.get("rise_time_ms", pd.Series(dtype=float))),
        "sepsc_decay_ms": _median(events.get("decay_time_ms", pd.Series(dtype=float))),
        "sepsc_iei_median_ms": _median(
            events.get("inter_event_interval_ms", pd.Series(dtype=float))),
        "sepsc_noise_pA": noise,
        # 4x the noise SD: below this an event cannot be told from noise at
        # all, whatever the detector's threshold happens to be set to.
        "sepsc_detection_limit_pA": 4.0 * noise if np.isfinite(noise) else float("nan"),
        "sepsc_unmeasurable": int(det.get("n_unmeasurable", 0)),
        "sepsc_thresh_sd": float(det.get("thresh_sd", float("nan"))),
        "sepsc_kernel_source": str(diag.get("kernel_source", "")),
        "sepsc_kernel_ms": float(diag.get("kernel_ms", float("nan"))),
        "sepsc_n_isolated": int(diag.get("n_isolated", 0)),
    }
    med = row["sepsc_amp_median_pA"]
    row["sepsc_snr"] = med / noise if (np.isfinite(med) and noise > 0) else float("nan")

    # Stability across the recording. The holding current comes from the trace
    # itself and is there whether or not anything was detected; the amplitude
    # trend needs enough events at both ends to have a median.
    v = np.asarray(trace.v, dtype=float)
    hold_first, hold_last = _ends(v)
    row["sepsc_holding_pA"] = float(np.median(v)) if v.size else float("nan")
    row["sepsc_holding_drift_pA"] = hold_last - hold_first

    amp_first, amp_last = _ends(amps) if amps.size >= 20 else (float("nan"),) * 2
    row["sepsc_amp_trend_pct"] = (100.0 * (amp_last - amp_first) / amp_first
                                  if np.isfinite(amp_first) and amp_first > 0
                                  else float("nan"))
    return row


# ---------------------------------------------------------------------------
# Stage entry point and the files it writes
# ---------------------------------------------------------------------------
def events_path(abf_path) -> Path:
    return Path(abf_path).with_name(f"{Path(abf_path).stem}_sEPSC_events.csv")


def waveforms_path(abf_path) -> Path:
    return Path(abf_path).with_name(f"{Path(abf_path).stem}_sEPSC_waveforms.npz")


def analyze_sepsc(path: Path, opts=None) -> tuple[dict, list[str]]:
    """Detect, measure and write. The per-cell stage `batch_cell_analysis` runs.

    Signature matches the other stages: (summary row, files written).
    """
    det = detect(Path(path), thresh_sd=getattr(opts, "sepsc_thresh_sd", None))
    row = summarise(det)

    csv = events_path(path)
    det["events"].to_csv(csv, index=False)

    waves = event_waveforms(det["trace"], det["events"])
    npz = waveforms_path(path)
    np.savez_compressed(npz, **waves)
    return row, [str(csv), str(npz)]


def load_waveforms(abf_path) -> Optional[dict]:
    """Read back what `analyze_sepsc` stored, or None if it was never run.

    Never raises on a damaged or half-written file: the QC window opening at
    all matters more than one cell's overlay panel.
    """
    npz = waveforms_path(abf_path)
    if not npz.exists():
        return None
    try:
        with np.load(npz) as data:
            return {k: data[k] for k in data.files}
    except Exception:
        return None


def load_events(abf_path) -> Optional[pd.DataFrame]:
    """The events table written for this recording, or None."""
    csv = events_path(abf_path)
    if not csv.exists():
        return None
    try:
        return pd.read_csv(csv, float_precision="round_trip")
    except Exception:
        return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 1
    path = Path(argv[0])
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 1

    row, written = analyze_sepsc(path)
    width = max(len(k) for k in row)
    for key, value in row.items():
        shown = f"{value:.3f}" if isinstance(value, float) else value
        print(f"  {key:<{width}}  {shown}")
    print("\n".join(["", "wrote:"] + [f"  {w}" for w in written]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
