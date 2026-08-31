"""
Shared event feature extraction -- used by train.py (to build the training
matrix), classify.py (to score newly detected events the same way), and
overlay.py (to peak-align/normalize for plotting). Keeping this in one
place means the classifier and every consumer of it always agree on what
"amplitude", "peak", etc. mean for a given raw window.

All functions here take a raw, peak-centered current window (as saved by
label.py, or sliced by classify.py around a miniML-reported location) plus
its sample interval -- no file I/O, no argparse, no plotting.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

FEATURE_NAMES = ["amplitude_pA", "rise_10_90_ms", "half_decay_ms", "tau_decay_ms", "charge_pA_ms"]


def _exp_decay(t, amp, tau):
    return amp * np.exp(-t / tau)


def _first_sustained(sw: np.ndarray, start: int, stop: int, level: float,
                      above: bool, hold_n: int) -> int | None:
    """First index in sw[start:stop] where the condition (>=level if `above`
    else <=level) holds for at least `hold_n` consecutive samples.

    A single-sample crossing isn't enough: near-threshold, low-amplitude
    events have enough per-sample noise that the smoothed trace can dip
    across a 50% threshold for one sample without the event having actually
    decayed there, which otherwise silently produces spuriously tiny
    half-decay times. Requiring the crossing to hold for a short stretch
    filters that out. Returns None if no such run exists before `stop`.
    """
    run = 0
    for k in range(start, stop):
        ok = (sw[k] >= level) if above else (sw[k] <= level)
        run = run + 1 if ok else 0
        if run >= hold_n:
            return k - hold_n + 1
    return None


def locate_peak_and_baseline(window: np.ndarray, dt: float, pre_ms: float,
                              direction: str = "negative", smooth_samples: int = 15):
    """Find (baseline, peak_idx, peak_v, w_smooth) for one peak-centered raw
    window -- shared by extract_features and anything else (e.g. plotting)
    that needs the same peak location.

    The labeled peak sits at a KNOWN, fixed offset in the window (it was
    sliced as v[peak_idx-pre_n : peak_idx+post_n] when saved) -- anchor to
    that instead of re-searching the whole window with argmax. Slow-event
    windows extend up to 80ms past the peak, long enough (mean inter-event
    interval here is ~255ms) that a subsequent overlapping event can land
    inside the window and hijack a whole-window argmax, silently measuring
    the wrong event's rise/decay. Only a small local window is re-searched,
    to correct for pre_n's sample rounding, not to relocate to a different
    event entirely. Peak/baseline are located on a lightly boxcar-smoothed
    copy of the window, not raw data, since slow events have a broad,
    near-flat peak plateau where a single noisy raw sample can look like a
    (wrong) local extremum.
    """
    pol = -1.0 if direction == "negative" else 1.0
    w = window.astype(float)
    w_smooth = uniform_filter1d(w, size=max(1, smooth_samples))
    sw = pol * w_smooth

    baseline_n = max(2, min(int(round(0.7 * pre_ms / 1000.0 / dt)), len(window) // 4))
    baseline = float(np.median(window[:baseline_n]))

    anchor = max(2, int(round(pre_ms / 1000.0 / dt)))
    anchor = min(anchor, len(sw) - 2)
    refine_n = max(1, int(round(1.0 / 1000.0 / dt)))  # +/-1ms
    r0, r1 = max(0, anchor - refine_n), min(len(sw), anchor + refine_n + 1)
    peak_idx = r0 + int(np.argmax(sw[r0:r1]))
    peak_v = float(w_smooth[peak_idx])
    return baseline, peak_idx, peak_v, w_smooth


def extract_features(window: np.ndarray, dt: float, pre_ms: float,
                      direction: str = "negative", smooth_samples: int = 15
                      ) -> tuple[list[float], bool]:
    """Compute [amplitude, rise_10_90_ms, half_decay_ms, tau_decay_ms,
    charge_pA_ms] for one peak-centered raw window.

    Rise/half-decay crossings are LOCATED on the lightly boxcar-smoothed
    copy of the window (see locate_peak_and_baseline), not raw data: left
    unsmoothed, per-sample noise can make a slow event's broad peak
    plateau look like it decays as fast as a sharp fast-event peak.
    Amplitude, tau, and charge (all aggregate measures that already
    average over many points) are computed from the raw window.

    Returns (features, decay_censored) -- censored=True means the trace
    never got back down to 50% of peak within the saved window, so
    half_decay_ms is a lower bound, not the true value (expected for some
    slow events if the window wasn't generous enough).
    """
    pol = -1.0 if direction == "negative" else 1.0
    baseline, peak_idx, peak_v, w_smooth = locate_peak_and_baseline(
        window, dt, pre_ms, direction, smooth_samples)
    sw = pol * w_smooth

    amplitude = peak_v - baseline
    if amplitude == 0 or peak_idx < 2:
        return [np.nan] * 5, True

    s_baseline, s_peak = pol * baseline, pol * peak_v
    span = s_peak - s_baseline
    hold_n = max(1, int(round(0.3 / 1000.0 / dt)))  # require 0.3ms sustained past threshold

    # 10-90% rise time: walk from window start to the peak
    lvl10, lvl90 = s_baseline + 0.1 * span, s_baseline + 0.9 * span
    i10 = _first_sustained(sw, 0, peak_idx + 1, lvl10, above=True, hold_n=hold_n) or 0
    i90 = _first_sustained(sw, i10, peak_idx + 1, lvl90, above=True, hold_n=hold_n)
    if i90 is None:
        i90 = peak_idx
    rise_10_90_ms = (i90 - i10) * dt * 1e3

    # half-decay time: walk forward from the peak to the end of the window,
    # requiring the trace to STAY at/below 50% of peak, not just touch it
    lvl50 = s_baseline + 0.5 * span
    half_idx = _first_sustained(sw, peak_idx, len(sw), lvl50, above=False, hold_n=hold_n)
    censored = half_idx is None
    if censored:
        half_idx = len(sw) - 1
    half_decay_ms = (half_idx - peak_idx) * dt * 1e3

    # single-exponential decay tau: fit on the SMOOTHED (not raw) decay --
    # for the smaller/noisier events here (many under 20pA), curve_fit on
    # raw single-sample data chases point noise and produces wild outlier
    # taus; the smoothed trace is a far more stable fit target. Upper bound
    # is capped relative to the observed window (a tau many times longer
    # than what's actually visible is unidentifiable from this data, not a
    # real measurement).
    decay_seg = w_smooth[peak_idx:] - baseline
    tt = np.arange(len(decay_seg)) * dt * 1e3
    tau_decay_ms = np.nan
    if len(decay_seg) >= 5:
        try:
            tau0 = max(half_decay_ms / np.log(2), dt * 1e3)
            popt, _ = curve_fit(_exp_decay, tt, decay_seg, p0=[amplitude, tau0],
                                 bounds=([-np.inf, dt * 1e3 * 0.5], [np.inf, tt[-1] * 3]),
                                 maxfev=2000)
            tau_decay_ms = float(popt[1])
        except (RuntimeError, ValueError):
            pass

    charge_pA_ms = float(np.trapezoid(window - baseline, dx=dt * 1e3))

    return [amplitude, rise_10_90_ms, half_decay_ms, tau_decay_ms, charge_pA_ms], censored
