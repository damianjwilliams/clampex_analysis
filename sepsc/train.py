"""
Train a fast-vs-slow sEPSC classifier from events hand-labeled with label.py.

This script only ever READS the *_training_events.csv / *_training_windows.npz
files produced by that tool -- it never modifies or deletes them, so they
stay available to add more labels later or retrain from scratch.

For each saved raw window, proper kinetic features are computed directly
from the trace via features.extract_features (independent of the rough
onset/decay heuristic label.py used only to size the window):
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
    python -m sepsc.train path\\to\\recording.abf
    python -m sepsc.train rec1.abf rec2.abf --out-dir models\\
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

from .features import FEATURE_NAMES, extract_features


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
                        "(must match label.py's --pre-ms, default 10)")
    p.add_argument("--direction", choices=["negative", "positive"], default="negative")
    p.add_argument("--smooth-samples", type=int, default=15,
                   help="boxcar smoothing window (samples) used only to locate the peak/rise/"
                        "decay crossings, not for amplitude/tau/charge (default 15, matches "
                        "label.py's --smooth-samples)")
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
