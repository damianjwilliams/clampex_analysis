# ARCHIVED 2026-08-27: superseded by sepsc/classify.py -- see
# sepsc/__init__.py for the combined pipeline (python -m sepsc classify ...).
# Kept here only for reference; not maintained.
"""
classify_sEPSCs.py
=====================

Apply the fast/slow classifier (train_sEPSC_classifier.py) to every event
miniML already detected in a recording (detect_sEPSCs_miniML.py's
*_miniML_individual.csv), labeling each one fast or slow.

For each detected event, a window is sliced from the raw trace around
miniML's reported peak location (generous enough -- pre_ms before, post_ms
after -- to cover either kinetic population), the same 5 kinetic features
used in training are computed (train_sEPSC_classifier.extract_features:
amplitude, 10-90% rise time, half-decay time, exponential decay tau,
charge), and the trained sklearn pipeline predicts a label + probability.

Only ever READS the .abf, the *_miniML_individual.csv, and the classifier
model -- nothing here writes to any of those.

Output (next to the source .abf):
    <name>_miniML_classified.csv    every detected event: miniML's own
                                     columns + this script's own re-derived
                                     features + predicted_label + prob_fast
                                     + prob_slow
    <name>_miniML_classified.png    raw trace with each event marked
                                     fast/slow (colors match
                                     label_sEPSC_training_events_gui.py /
                                     plot_fast_slow_overlay.py)

Usage
-----
    python classify_sEPSCs.py path\\to\\recording.abf
    python classify_sEPSCs.py recording.abf --model path\\to\\model.joblib
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyabf
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_sEPSC_classifier import extract_features, FEATURE_NAMES

COLORS = {"fast": "#2a78d6", "slow": "#eb6834"}
MARKERS = {"fast": "^", "slow": "s"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", help="Path to the gap-free voltage-clamp .abf file")
    p.add_argument("--individual-csv", default=None,
                   help="Path to the *_miniML_individual.csv (default: <abf>_miniML_individual.csv)")
    p.add_argument("--model", default=None,
                   help="Path to the trained classifier .joblib "
                        "(default: sEPSC_fast_slow_classifier.joblib next to the .abf)")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--pre-ms", type=float, default=10.0,
                   help="baseline captured before each event's peak, in ms (default 10, "
                        "should match what the model was trained on)")
    p.add_argument("--post-ms", type=float, default=80.0,
                   help="decay captured after each event's peak, in ms (default 80 -- generous "
                        "enough to cover either population; unlike training, one shared window "
                        "is used here since the class isn't known yet)")
    p.add_argument("--direction", choices=["negative", "positive"], default="negative")
    p.add_argument("--smooth-samples", type=int, default=15)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    stem = os.path.splitext(args.abf)[0]
    csv_path = args.individual_csv or f"{stem}_miniML_individual.csv"
    model_path = args.model or os.path.join(os.path.dirname(os.path.abspath(args.abf)),
                                              "sEPSC_fast_slow_classifier.joblib")
    if not os.path.exists(csv_path):
        p.error(f"{csv_path!r} not found -- run detect_sEPSCs_miniML.py first")
    if not os.path.exists(model_path):
        p.error(f"{model_path!r} not found -- run train_sEPSC_classifier.py first, or pass --model")

    abf = pyabf.ABF(args.abf)
    abf.setSweep(0, channel=args.channel)
    v = np.asarray(abf.sweepY, float)
    t = np.asarray(abf.sweepX, float)
    dt = 1.0 / abf.dataRate

    raw = pd.read_csv(csv_path, index_col=0)   # rows=features, cols=event_N (miniML's own format)
    events = raw.T.reset_index(drop=True)
    print(f"Loaded {len(events)} detected events from {os.path.basename(csv_path)} (read-only)")

    pipeline = joblib.load(model_path)
    print(f"Loaded classifier -> {os.path.basename(model_path)} (read-only)")

    pre_n = int(round(args.pre_ms / 1000.0 / dt))
    post_n = int(round(args.post_ms / 1000.0 / dt))

    feats, skipped = [], 0
    for loc in events["location"]:
        loc = int(round(loc))
        w0, w1 = loc - pre_n, loc + post_n
        if w0 < 0 or w1 > len(v):
            feats.append([np.nan] * len(FEATURE_NAMES))
            skipped += 1
            continue
        window = v[w0:w1]
        f, _censored = extract_features(window, dt, args.pre_ms, args.direction, args.smooth_samples)
        feats.append(f)

    feat_df = pd.DataFrame(feats, columns=FEATURE_NAMES)
    if skipped:
        print(f"Note: {skipped} event(s) too close to the recording edge for a full window -- "
              f"left unclassified.")

    valid = feat_df.notna().all(axis=1)
    predicted = pd.Series(pd.NA, index=feat_df.index, dtype=object)
    prob_fast = pd.Series(np.nan, index=feat_df.index)
    prob_slow = pd.Series(np.nan, index=feat_df.index)

    if valid.any():
        X = feat_df.loc[valid].values
        pred = pipeline.predict(X)
        proba = pipeline.predict_proba(X)
        classes = list(pipeline.classes_)
        predicted.loc[valid] = pred
        prob_fast.loc[valid] = proba[:, classes.index("fast")]
        prob_slow.loc[valid] = proba[:, classes.index("slow")]

    out = pd.concat([events, feat_df], axis=1)
    out["predicted_label"] = predicted
    out["prob_fast"] = prob_fast
    out["prob_slow"] = prob_slow

    out_csv = f"{stem}_miniML_classified.csv"
    out.to_csv(out_csv, index=False)

    n_fast = int((out["predicted_label"] == "fast").sum())
    n_slow = int((out["predicted_label"] == "slow").sum())
    n_unclassified = int(out["predicted_label"].isna().sum())
    print(f"\nClassified {n_fast + n_slow}/{len(out)} events: "
          f"{n_fast} fast, {n_slow} slow"
          f"{f', {n_unclassified} unclassified' if n_unclassified else ''}")
    print(f"Saved -> {out_csv}")

    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(16, 4.5))
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        ax.plot(t, v, color="#333333", lw=0.3, zorder=1)
        for label in ("fast", "slow"):
            sel = out[out["predicted_label"] == label]
            if sel.empty:
                continue
            peak_t = sel["location"].to_numpy() * dt
            peak_v = v[sel["location"].to_numpy().astype(int)]
            ax.plot(peak_t, peak_v, marker=MARKERS[label], color=COLORS[label], ls="None",
                     ms=5, mec="k", mew=0.3, zorder=3, label=f"{label} (n={len(sel)})")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"current ({abf.adcUnits[args.channel]})")
        ax.set_title(f"{os.path.basename(args.abf)} -- {n_fast} fast / {n_slow} slow sEPSCs classified")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        out_png = f"{stem}_miniML_classified.png"
        fig.savefig(out_png, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved -> {out_png}")

    print("Source .abf / individual CSV / classifier model were only read, never modified.")


if __name__ == "__main__":
    main()
