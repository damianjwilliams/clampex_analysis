"""
Overlay the peak-aligned, amplitude-normalized time course of every hand-
labeled fast and slow sEPSC (from label.py), so the two populations'
kinetics can be compared visually.

Every event is:
  * peak-located the same way train.py does (features.locate_peak_and_baseline
    -- a lightly smoothed copy of the window, anchored to the known peak
    offset), so this plot and the trained classifier agree on where each
    event's peak actually is;
  * shifted so its peak sits at t=0;
  * scaled by its own |amplitude| so every peak lands at -1 (baseline = 0),
    which is what makes shape (kinetics) comparable across events of very
    different absolute size.

Both classes are resampled onto ONE shared time range (the intersection of
however much post-peak data each class actually has), so fast and slow are
never shown over different-length windows just because one was captured
with a longer post-peak duration during labeling.

Each class is drawn as many thin, low-opacity individual traces (so
variability is visible) plus one bold mean trace on top.

Only ever READS the *_training_windows.npz files -- nothing here writes to
them.

Output (next to the first input .abf, unless --out is given):
    sEPSC_fast_slow_overlay.png

Usage
-----
    python -m sepsc.overlay path\\to\\recording.abf
    python -m sepsc.overlay rec1.abf rec2.abf --out combined_overlay.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from .features import locate_peak_and_baseline
from .style import COLORS, INK, MUTED, GRID, SURFACE


def peak_aligned_normalized(window: np.ndarray, dt: float, pre_ms: float,
                             direction: str, smooth_samples: int):
    """Return (t_ms, normalized) with t=0 at the peak and the peak at -1
    (baseline=0), or None if the event has no usable amplitude."""
    baseline, peak_idx, peak_v, _ = locate_peak_and_baseline(
        window, dt, pre_ms, direction, smooth_samples)
    amplitude = peak_v - baseline
    if amplitude == 0:
        return None
    t_ms = (np.arange(len(window)) - peak_idx) * dt * 1e3
    normalized = (window - baseline) / abs(amplitude)
    return t_ms, normalized


def collect_class_traces(npz_list, key: str, pre_ms: float, direction: str, smooth_samples: int):
    """Peak-align and amplitude-normalize every window under `key` (e.g.
    'fast_windows') across the given npz files. Returns lists (all_t,
    all_norm), one entry per event, each still on its own native time axis
    (not yet resampled to a shared grid)."""
    dt_key = key.replace("windows", "dt")
    all_t, all_norm = [], []
    for npz in npz_list:
        if key not in npz:
            continue
        dt = float(npz[dt_key])
        for window in npz[key]:
            result = peak_aligned_normalized(window, dt, pre_ms, direction, smooth_samples)
            if result is None:
                continue
            t_ms, normalized = result
            all_t.append(t_ms)
            all_norm.append(normalized)
    return all_t, all_norm


def resample_to_grid(all_t, all_norm, common_t):
    """Interpolate every (t, normalized) pair onto the shared common_t grid.

    Returns (individual_traces [n_events x n_grid], mean_trace), or
    (None, None) if there are no events.
    """
    if not all_t:
        return None, None
    stacked = np.array([np.interp(common_t, t, n) for t, n in zip(all_t, all_norm)])
    return stacked, stacked.mean(axis=0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", nargs="+", help="One or more .abf files whose "
                                          "*_training_windows.npz to overlay")
    p.add_argument("--pre-ms", type=float, default=10.0,
                   help="pre-peak baseline length used when the windows were saved (default 10)")
    p.add_argument("--direction", choices=["negative", "positive"], default="negative")
    p.add_argument("--smooth-samples", type=int, default=15,
                   help="boxcar smoothing window (samples) used only to locate peaks (default 15)")
    p.add_argument("--max-overlay-traces", type=int, default=120,
                   help="cap on individual traces drawn per class, for render speed/clarity "
                        "(the mean is always computed from ALL events regardless) (default 120)")
    p.add_argument("--display-smooth-ms", type=float, default=1.5,
                   help="extra light smoothing (ms) applied ONLY to the plotted traces, for "
                        "legibility -- amplitude-normalizing small events amplifies their raw "
                        "per-sample noise a lot; has no effect on peak detection or any saved "
                        "numbers (default 1.5, use 0 to disable)")
    p.add_argument("--out", default=None, help="output PNG path (default: alongside the first .abf)")
    args = p.parse_args(argv)

    npz_list = []
    for abf_path in args.abf:
        stem = os.path.splitext(abf_path)[0]
        npz_path = f"{stem}_training_windows.npz"
        if not os.path.exists(npz_path):
            print(f"WARNING: {npz_path!r} not found -- skipping {abf_path!r}", file=sys.stderr)
            continue
        npz_list.append(np.load(npz_path))
        print(f"  loaded {os.path.basename(npz_path)} (read-only)")

    if not npz_list:
        p.error("no *_training_windows.npz files found")

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # collect both classes first so a SINGLE shared time range can be used
    # for both -- fast events were only labeled with a 25ms post-peak window
    # vs slow's 80ms, so without this fast would visibly cut off early
    # simply because less was captured, not because of anything about the
    # events themselves
    raw = {}
    for label in ("fast", "slow"):
        all_t, all_norm = collect_class_traces(
            npz_list, f"{label}_windows", args.pre_ms, args.direction, args.smooth_samples)
        if not all_t:
            print(f"WARNING: no usable '{label}' events found", file=sys.stderr)
            continue
        raw[label] = (all_t, all_norm)

    if not raw:
        p.error("no usable labeled events found")

    t_min = max(t[0] for all_t, _ in raw.values() for t in all_t)
    t_max = min(t[-1] for all_t, _ in raw.values() for t in all_t)
    common_t = np.linspace(t_min, t_max, 400)
    print(f"Shared display window: {t_min:.1f} to {t_max:.1f} ms "
          f"(limited by whichever class was captured with a shorter post-peak window)")

    counts, all_display = {}, []
    for label, (all_t, all_norm) in raw.items():
        stacked, mean_trace = resample_to_grid(all_t, all_norm, common_t)
        counts[label] = len(stacked)

        # a little extra smoothing purely for the plotted lines: normalizing
        # a small-amplitude event by its own tiny |amplitude| amplifies that
        # event's raw per-sample noise a lot, which otherwise buries the
        # actual shape under noise spikes -- doesn't touch peak detection or
        # any saved feature/model.
        grid_dt_ms = common_t[1] - common_t[0]
        smooth_grid_n = max(1, int(round(args.display_smooth_ms / grid_dt_ms))) if args.display_smooth_ms > 0 else 1
        display = uniform_filter1d(stacked, size=smooth_grid_n, axis=1) if smooth_grid_n > 1 else stacked
        display_mean = uniform_filter1d(mean_trace, size=smooth_grid_n) if smooth_grid_n > 1 else mean_trace
        all_display.append(display)

        show = display
        if len(display) > args.max_overlay_traces:
            idx = np.random.default_rng(0).choice(len(display), args.max_overlay_traces, replace=False)
            show = display[idx]
        for trace in show:
            ax.plot(common_t, trace, color=COLORS[label], lw=0.5, alpha=0.05, zorder=2)

        ax.plot(common_t, display_mean, color=COLORS[label], lw=2.5, zorder=5,
                label=f"{label}  (n={counts[label]})")

    ax.axhline(0.0, color=GRID, lw=1.0, zorder=1)
    ax.axhline(-1.0, color=GRID, lw=1.0, ls="--", zorder=1)
    ax.axvline(0.0, color=MUTED, lw=1.0, ls="--", zorder=1)

    if all_display:
        y_lo, y_hi = np.percentile(np.concatenate([d.ravel() for d in all_display]), [1, 99])
        pad = 0.15 * (y_hi - y_lo)
        ax.set_ylim(y_lo - pad, max(0.3, y_hi + pad))

    ax.set_xlabel("time from peak (ms)", color=INK)
    ax.set_ylabel("normalized current (peak = -1, baseline = 0)", color=INK)
    ax.set_title("Fast vs slow sEPSC time course -- peak-aligned, amplitude-normalized", color=INK)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK)

    fig.tight_layout()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.abf[0])), "sEPSC_fast_slow_overlay.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n{', '.join(f'{k}: {v}' for k, v in counts.items())} events plotted")
    print(f"Saved -> {out_path}")
    print("Source *_training_windows.npz files were only read, never modified.")


if __name__ == "__main__":
    main()
