"""
Compare two detectors' amplitude distributions on the same recording --
miniML (detect.py) vs. the classical Mini-Analysis-style detector
(minianalysis.py) -- with three standard, complementary views of a
single-variable distribution:

    1. cumulative frequency (empirical CDF): what fraction of each
       method's events fall at or below a given |amplitude|
    2. amplitude histogram: the same variable's density, shared bins
       across both methods for a fair comparison
    3. box-and-whisker + dot plot: median/IQR/range plus every individual
       event, side by side per method

...plus a second figure comparing each method's average event TIME COURSE
(not just amplitude), peak-aligned the same way overlay.py aligns fast vs
slow events:

    4. unscaled average waveform (real pA, baseline=0) -- shows both
       kinetics AND relative amplitude between methods
    5. peak-normalized average waveform (peak=-1) -- isolates kinetics
       (rise/decay shape) from amplitude, so a slower/faster timecourse
       between methods is visible even if their amplitudes differ

Only ever READS each detector's own events CSV and the raw .abf trace --
nothing here writes to either.

Output (next to the source .abf, unless --out/--trace-out are given):
    <name>_method_comparison.png
    <name>_method_avg_trace.png

Usage
-----
    python -m sepsc.compare path\\to\\recording.abf
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyabf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .features import locate_peak_and_baseline
from .overlay import resample_to_grid
from .review import SOURCES
from .style import COLORS, SURFACE, INK, MUTED, GRID

METHOD_COLORS = {"miniML": COLORS["fast"], "Mini-Analysis-style": COLORS["slow"]}


def load_events(csv_path: str, source_key: str, data_rate_hz: float) -> pd.DataFrame:
    """Full tidy events table for one detector's output, via the same
    loaders sepsc.review already uses -- keeps both formats read the same
    way everywhere in the package."""
    return SOURCES[source_key]["loader"](csv_path, data_rate_hz)


def extract_aligned_traces(v: np.ndarray, dt: float, df: pd.DataFrame,
                            pre_ms: float, post_ms: float, direction: str,
                            smooth_samples: int):
    """Slice a pre_ms/post_ms window around each event's 'location' out of
    the raw trace, then peak-locate/baseline within it the same way
    overlay.py does for labeled fast/slow events -- so this works
    regardless of whether a detector's 'location' is the true peak
    (minianalysis) or an onset near it (miniML), rather than trusting each
    detector's own convention.

    Returns (all_t_ms, all_unscaled, all_scaled): per-event lists, each
    entry still on its own native time axis (t=0 at that event's peak).
    """
    pre_n = int(round(pre_ms / 1000.0 / dt))
    post_n = int(round(post_ms / 1000.0 / dt))
    all_t, all_unscaled, all_scaled = [], [], []
    for loc in df["location"].to_numpy(dtype=float):
        center = int(round(loc))
        i0, i1 = center - pre_n, center + post_n
        if i0 < 0 or i1 > len(v):
            continue
        window = v[i0:i1]
        baseline, peak_idx, peak_v, _ = locate_peak_and_baseline(
            window, dt, pre_ms, direction, smooth_samples)
        amplitude = peak_v - baseline
        if amplitude == 0:
            continue
        t_ms = (np.arange(len(window)) - peak_idx) * dt * 1e3
        unscaled = window - baseline
        all_t.append(t_ms)
        all_unscaled.append(unscaled)
        all_scaled.append(unscaled / abs(amplitude))
    return all_t, all_unscaled, all_scaled


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", help="Path to the gap-free voltage-clamp .abf file")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--miniml-csv", default=None,
                   help="Path to the *_miniML_individual.csv (default: <abf>_miniML_individual.csv)")
    p.add_argument("--minianalysis-csv", default=None,
                   help="Path to the *_minianalysis_events.csv (default: <abf>_minianalysis_events.csv)")
    p.add_argument("--bins", type=int, default=30, help="histogram bin count (default 30)")
    p.add_argument("--pre-ms", type=float, default=5.0,
                   help="average-waveform window before each event's location (default 5)")
    p.add_argument("--post-ms", type=float, default=20.0,
                   help="average-waveform window after each event's location (default 20)")
    p.add_argument("--direction", choices=["negative", "positive"], default="negative")
    p.add_argument("--smooth-samples", type=int, default=15,
                   help="boxcar smoothing window (samples) used only to locate each event's "
                        "peak within its window (default 15)")
    p.add_argument("--out", default=None, help="amplitude-comparison PNG path (default: alongside the .abf)")
    p.add_argument("--trace-out", default=None,
                   help="average-waveform PNG path (default: alongside the .abf)")
    args = p.parse_args(argv)

    stem = os.path.splitext(args.abf)[0]
    abf = pyabf.ABF(args.abf)
    abf.setSweep(0, channel=args.channel)
    trace_t = np.asarray(abf.sweepX, dtype=float)
    trace_v = np.asarray(abf.sweepY, dtype=float)
    trace_dt = trace_t[1] - trace_t[0]

    sources = {
        "miniML": args.miniml_csv or f"{stem}_miniML_individual.csv",
        "Mini-Analysis-style": args.minianalysis_csv or f"{stem}_minianalysis_events.csv",
    }
    source_keys = {"miniML": "miniml", "Mini-Analysis-style": "minianalysis"}

    amplitudes, events = {}, {}
    for label, csv_path in sources.items():
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path!r} not found -- skipping {label}")
            continue
        df = load_events(csv_path, source_keys[label], abf.dataRate)
        events[label] = df
        amps = np.abs(df["amplitude"].to_numpy(dtype=float))
        amplitudes[label] = amps
        print(f"Loaded {len(amps)} {label} events (read-only): "
              f"median |amp| {np.median(amps):.1f} pA, mean {amps.mean():.1f} pA")

    if len(amplitudes) < 2:
        p.error("need both detectors' output to compare -- run `python -m sepsc detect` and "
                "`python -m sepsc minianalysis` on this file first")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    # Panel 1: cumulative frequency (empirical CDF), normalized to a
    # fraction (0-1) so methods with very different n are shape-comparable
    # instead of the smaller-n curve looking artificially "lower".
    ax = axes[0]
    for label, amps in amplitudes.items():
        sorted_amps = np.sort(amps)
        ax.step(sorted_amps, np.arange(1, len(sorted_amps) + 1) / len(sorted_amps), where="post",
                 color=METHOD_COLORS[label], lw=2, label=f"{label} (n={len(amps)})")
    ax.set_xlabel("|amplitude| (pA)")
    ax.set_ylabel("cumulative fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title("Cumulative frequency")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    # Panel 2: amplitude histogram, shared bins for a fair comparison
    ax = axes[1]
    all_amps = np.concatenate(list(amplitudes.values()))
    bin_max = np.percentile(all_amps, 99.5)
    bins = np.linspace(0, bin_max, args.bins)
    for label, amps in amplitudes.items():
        ax.hist(amps, bins=bins, color=METHOD_COLORS[label], alpha=0.5,
                 edgecolor=METHOD_COLORS[label], label=label)
    ax.set_xlabel("|amplitude| (pA)")
    ax.set_ylabel("count")
    ax.set_title("Amplitude histogram")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # Panel 3: box-and-whisker + jittered dot plot
    ax = axes[2]
    labels = list(amplitudes.keys())
    positions = np.arange(1, len(labels) + 1)
    bp = ax.boxplot([amplitudes[l] for l in labels], positions=positions, widths=0.35,
                      showfliers=False, patch_artist=True, medianprops=dict(color=INK, lw=1.5),
                      zorder=5)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor("none")
        patch.set_edgecolor(METHOD_COLORS[label])
        patch.set_linewidth(1.5)
    for part in ("whiskers", "caps"):
        for artist in bp[part]:
            artist.set_color(MUTED)

    rng = np.random.default_rng(0)
    for pos, label in zip(positions, labels):
        y = amplitudes[label]
        x = pos + rng.uniform(-0.12, 0.12, size=len(y))
        ax.scatter(x, y, color=METHOD_COLORS[label], alpha=0.3, s=10, zorder=3, edgecolors="none")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("|amplitude| (pA)")
    ax.set_title("Amplitude distribution")

    for ax in axes:
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.title.set_color(INK)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)

    fig.suptitle(f"{os.path.basename(args.abf)}  --  detection method comparison", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = args.out or f"{stem}_method_comparison.png"
    fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {out_path}")

    # Second figure: peak-aligned average waveform per method, unscaled and
    # peak-normalized. Both methods share ONE time grid (like overlay.py's
    # fast/slow comparison) so the two mean curves are directly
    # point-for-point comparable, not just visually similar-looking.
    raw_traces = {}
    for label, df in events.items():
        all_t, all_unscaled, all_scaled = extract_aligned_traces(
            trace_v, trace_dt, df, args.pre_ms, args.post_ms, args.direction, args.smooth_samples)
        if not all_t:
            print(f"WARNING: no usable {label} waveforms extracted -- skipping from average-trace plot")
            continue
        raw_traces[label] = (all_t, all_unscaled, all_scaled)

    if len(raw_traces) < 2:
        print("Need both methods' waveforms to plot average time courses -- skipping trace figure.")
    else:
        t_min = max(t[0] for all_t, _, _ in raw_traces.values() for t in all_t)
        t_max = min(t[-1] for all_t, _, _ in raw_traces.values() for t in all_t)
        common_t = np.linspace(t_min, t_max, 400)

        fig2, (ax_u, ax_s) = plt.subplots(1, 2, figsize=(12, 5.5))
        fig2.patch.set_facecolor(SURFACE)
        for ax in (ax_u, ax_s):
            ax.set_facecolor(SURFACE)

        for label, (all_t, all_unscaled, all_scaled) in raw_traces.items():
            _, mean_unscaled = resample_to_grid(all_t, all_unscaled, common_t)
            _, mean_scaled = resample_to_grid(all_t, all_scaled, common_t)
            n = len(all_t)
            ax_u.plot(common_t, mean_unscaled, color=METHOD_COLORS[label], lw=2.5,
                       label=f"{label} (n={n})")
            ax_s.plot(common_t, mean_scaled, color=METHOD_COLORS[label], lw=2.5,
                       label=f"{label} (n={n})")

        ax_u.axhline(0.0, color=GRID, lw=1.0, zorder=1)
        ax_u.set_ylabel("current (pA), baseline = 0")
        ax_u.set_title("Average waveform -- unscaled")

        ax_s.axhline(0.0, color=GRID, lw=1.0, zorder=1)
        ax_s.axhline(-1.0, color=GRID, lw=1.0, ls="--", zorder=1)
        ax_s.set_ylabel("normalized current (peak = -1)")
        ax_s.set_title("Average waveform -- peak-normalized")

        for ax in (ax_u, ax_s):
            ax.axvline(0.0, color=MUTED, lw=1.0, ls="--", zorder=1)
            ax.set_xlabel("time from peak (ms)")
            ax.tick_params(colors=MUTED)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            ax.title.set_color(INK)
            ax.xaxis.label.set_color(INK)
            ax.yaxis.label.set_color(INK)
            ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK)

        fig2.suptitle(f"{os.path.basename(args.abf)}  --  average event time course", color=INK)
        fig2.tight_layout(rect=[0, 0, 1, 0.95])

        trace_out_path = args.trace_out or f"{stem}_method_avg_trace.png"
        fig2.savefig(trace_out_path, dpi=140, facecolor=fig2.get_facecolor())
        plt.close(fig2)
        print(f"Saved -> {trace_out_path}")

    print("Source event CSVs and the raw .abf trace were only read, never modified.")


if __name__ == "__main__":
    main()
