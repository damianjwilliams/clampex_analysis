# ARCHIVED 2026-08-27: superseded by sepsc/train.py (its feature extraction
# moved to sepsc/features.py) -- see sepsc/__init__.py for the combined
# pipeline (python -m sepsc train ...). Kept here only for reference; not
# maintained.
"""
train_sEPSC_classifier.py
===========================

Train a fast-vs-slow sEPSC classifier from events hand-labeled with
label_sEPSC_training_events_gui.py.

This script only ever READS the *_training_events.csv / *_training_windows.npz
files produced by that tool -- it never modifies or deletes them, so they
stay available to add more labels later or retrain from scratch.

For each saved raw window, proper kinetic features are computed directly
from the trace (independent of the rough onset/decay heuristic the labeling
tool used only to size the window):
    amplitude        peak - baseline (pA)
    rise_10_90_ms     10%-90% rise time
    half_decay_ms     time from peak to 50%-of-peak decay (censored at the
                       window edge for slow events whose decay outlasts it
                       -- see the printed warning)
    tau_decay_ms      single-exponential decay time constant (curve fit;
                       NaN if the fit fails, imputed with the column median)
    charge_pA_ms      area under the event (baseline-subtracted)

These features feed a logistic regression (balanced class weights,
standardized + median-imputed features) evaluated with stratified 5-fold
cross-validation and a held-out test split.

Accepts one or more .abf paths whose *_training_windows.npz / _events.csv
companions should be combined -- so relabeling more files later and
retraining on everything together is a one-line rerun.

Output (next to the first input .abf, unless --out-dir is given):
    sEPSC_fast_slow_classifier.joblib        the fitted sklearn Pipeline
    sEPSC_fast_slow_classifier_meta.json     feature names, source files,
                                              performance metrics, versions

Usage
-----
    python train_sEPSC_classifier.py path\\to\\recording.abf
    python train_sEPSC_classifier.py rec1.abf rec2.abf --out-dir models\\
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib
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


def load_dataset(abf_paths: list[str], pre_ms: float, direction: str, smooth_samples: int = 15):
    """Load and combine *_training_windows.npz (+ matching _events.csv, for
    the printed per-file summary only) for every given .abf path. Read-only:
    nothing here writes to those files."""
    X, y, sources = [], [], []
    n_censored = 0
    for abf_path in abf_paths:
        stem = os.path.splitext(abf_path)[0]
        npz_path = f"{stem}_training_windows.npz"
        csv_path = f"{stem}_training_events.csv"
        if not os.path.exists(npz_path):
            print(f"WARNING: {npz_path!r} not found -- skipping {abf_path!r}", file=sys.stderr)
            continue

        npz = np.load(npz_path)
        dt = float(npz["fast_dt"]) if "fast_dt" in npz else float(npz["slow_dt"])
        n_fast = len(npz["fast_windows"]) if "fast_windows" in npz else 0
        n_slow = len(npz["slow_windows"]) if "slow_windows" in npz else 0
        print(f"  {os.path.basename(npz_path)}: {n_fast} fast, {n_slow} slow "
              f"(from {os.path.basename(csv_path) if os.path.exists(csv_path) else '<no csv found>'}, read-only)")

        for label, key in (("fast", "fast_windows"), ("slow", "slow_windows")):
            if key not in npz:
                continue
            for window in npz[key]:
                feats, censored = extract_features(window, dt, pre_ms, direction, smooth_samples)
                if any(np.isnan(f) for f in (feats[0], feats[1], feats[2])):
                    continue  # amplitude/rise/half-decay must be valid; tau alone may be NaN (imputed)
                X.append(feats)
                y.append(label)
                n_censored += int(censored)
        sources.append(os.path.abspath(npz_path))

    if n_censored:
        print(f"Note: {n_censored} event(s) never decayed back to 50% of peak within their "
              f"saved window -- half_decay_ms is a lower bound for those (tau_decay_ms is "
              f"usually still meaningful).")
    return np.array(X, dtype=float), np.array(y), sources


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", nargs="+", help="One or more .abf files whose "
                                          "*_training_windows.npz/_events.csv to train on")
    p.add_argument("--pre-ms", type=float, default=10.0,
                   help="pre-peak baseline length used when the windows were saved "
                        "(must match label_sEPSC_training_events_gui.py's --pre-ms, default 10)")
    p.add_argument("--direction", choices=["negative", "positive"], default="negative")
    p.add_argument("--smooth-samples", type=int, default=15,
                   help="boxcar smoothing window (samples) used only to locate the peak/rise/"
                        "decay crossings, not for amplitude/tau/charge (default 15, matches "
                        "label_sEPSC_training_events_gui.py's --smooth-samples)")
    p.add_argument("--test-size", type=float, default=0.2, help="held-out test fraction (default 0.2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None,
                   help="where to save the trained model (default: alongside the first .abf)")
    args = p.parse_args(argv)

    print(f"Loading labeled events from {len(args.abf)} file(s) (read-only):")
    X, y, sources = load_dataset(args.abf, args.pre_ms, args.direction, args.smooth_samples)
    if len(X) == 0:
        p.error("no usable labeled events found")
    n_fast, n_slow = int((y == "fast").sum()), int((y == "slow").sum())
    print(f"Total usable events: {len(X)} ({n_fast} fast, {n_slow} slow)")
    if min(n_fast, n_slow) < 20:
        print("WARNING: fewer than 20 examples in one class -- results will be noisy. "
              "Label more events before trusting this model.", file=sys.stderr)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"\n5-fold CV accuracy on training split: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_acc = float((y_pred == y_test).mean())
    print(f"Held-out test accuracy ({len(y_test)} events): {test_acc:.3f}\n")
    print(classification_report(y_test, y_pred, digits=3))

    labels_sorted = sorted(pipeline.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print(f"Confusion matrix (rows=true, cols=predicted, order={labels_sorted}):")
    print(cm)

    coefs = pipeline.named_steps["clf"].coef_[0]
    print("\nStandardized feature weights (positive -> pushes toward "
          f"'{pipeline.classes_[1]}', negative -> toward '{pipeline.classes_[0]}'):")
    for name, coef in sorted(zip(FEATURE_NAMES, coefs), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<16} {coef:+.3f}")

    # refit on ALL available data (train+test) for the final saved model,
    # now that test accuracy has already been honestly measured above
    pipeline.fit(X, y)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.abf[0]))
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "sEPSC_fast_slow_classifier.joblib")
    meta_path = os.path.join(out_dir, "sEPSC_fast_slow_classifier_meta.json")

    joblib.dump(pipeline, model_path)
    meta = dict(
        trained_at=datetime.now(timezone.utc).isoformat(),
        feature_names=FEATURE_NAMES,
        classes=list(pipeline.classes_),
        n_events=dict(fast=n_fast, slow=n_slow),
        cv_accuracy_mean=float(cv_scores.mean()),
        cv_accuracy_std=float(cv_scores.std()),
        held_out_test_accuracy=test_acc,
        source_npz_files=sources,
        pre_ms=args.pre_ms,
        smooth_samples=args.smooth_samples,
        direction=args.direction,
        sklearn_version=sklearn.__version__,
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model -> {model_path}")
    print(f"Saved metadata -> {meta_path}")
    print("Source *_training_events.csv / *_training_windows.npz files were only read, never modified.")


if __name__ == "__main__":
    main()
