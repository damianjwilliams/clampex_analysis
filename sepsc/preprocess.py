"""
Shared trace preprocessing: zero-phase Bessel low-pass filter + downsample,
for any gap-free voltage-clamp .abf channel.

Before filtering, checks the recording's OWN header for the amplifier's
telegraphed hardware low-pass setting (fTelegraphFilter -- what the
amplifier's front panel/software was actually set to during acquisition,
not a guess) so a redundant re-filter below what's already been applied is
never silently applied without saying so.

Order matters: the Bessel low-pass is applied at the trace's native
sampling rate FIRST, then the result is decimated down to the target rate.
Filtering before decimating is what prevents aliasing -- downsampling first
would let content between the new Nyquist and the old one fold back into
the passband.

Usage
-----
    python -m sepsc.preprocess path\\to\\recording.abf
    python -m sepsc.preprocess path\\to\\recording.abf --cutoff-hz 3000 --target-rate-hz 10000
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pyabf
from scipy.signal import bessel, sosfiltfilt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import SURFACE, INK, MUTED, GRID, TRACE

DEFAULT_CUTOFF_HZ = 3000.0
DEFAULT_TARGET_RATE_HZ = 10_000.0
DEFAULT_ORDER = 8


def get_hardware_filter_hz(abf: pyabf.ABF, channel: int) -> float | None:
    """The amplifier's own telegraphed low-pass filter setting (Hz) for
    this channel, straight from the ABF header -- None if telegraphing
    wasn't enabled for this channel (older/non-telegraphing amplifiers)."""
    adc = abf._adcSection
    enabled = getattr(adc, "nTelegraphEnable", None)
    cutoffs = getattr(adc, "fTelegraphFilter", None)
    if not enabled or not cutoffs or not enabled[channel]:
        return None
    return float(cutoffs[channel])


def bessel_lowpass(v: np.ndarray, fs: float, cutoff_hz: float = DEFAULT_CUTOFF_HZ,
                    order: int = DEFAULT_ORDER) -> np.ndarray:
    """Zero-phase (filtfilt) Bessel low-pass -- Bessel for its maximally
    flat group delay (preserves event timecourse/shape, unlike a
    Butterworth's steeper but phase-distorting rolloff), filtfilt so the
    zero-phase result doesn't shift event peak times."""
    sos = bessel(order, cutoff_hz, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, v)


def downsample(v: np.ndarray, fs: float, target_hz: float = DEFAULT_TARGET_RATE_HZ):
    """Decimate to as close to target_hz as an integer factor allows.

    Assumes v has ALREADY been low-pass filtered below target_hz/2 (this
    module's own load_filtered_trace guarantees that) -- with aliasing
    already prevented upstream, plain strided decimation is exact and
    avoids scipy.signal.decimate's own redundant internal anti-alias
    filter. Returns (v_downsampled, actual_output_rate_hz).
    """
    factor = max(1, round(fs / target_hz))
    return v[::factor], fs / factor


def load_filtered_trace(abf_path: str, channel: int = 0,
                         cutoff_hz: float = DEFAULT_CUTOFF_HZ,
                         target_hz: float = DEFAULT_TARGET_RATE_HZ,
                         order: int = DEFAULT_ORDER):
    """Load one channel, Bessel-filter at cutoff_hz, downsample to
    target_hz. Returns (t, v, fs_out, hardware_filter_hz) -- t/v are the
    filtered+downsampled trace, hardware_filter_hz is what the amplifier
    itself already applied (or None if unknown), for the caller to report.
    """
    abf = pyabf.ABF(abf_path)
    abf.setSweep(0, channel=channel)
    v_raw = np.asarray(abf.sweepY, dtype=float)
    fs = abf.dataRate

    hardware_filter_hz = get_hardware_filter_hz(abf, channel)

    v_filt = bessel_lowpass(v_raw, fs, cutoff_hz, order)
    v_out, fs_out = downsample(v_filt, fs, target_hz)
    t_out = np.arange(len(v_out)) / fs_out
    return t_out, v_out, fs_out, hardware_filter_hz


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", help="Path to the gap-free voltage-clamp .abf file")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ,
                   help=f"Bessel low-pass cutoff, Hz (default {DEFAULT_CUTOFF_HZ:.0f})")
    p.add_argument("--target-rate-hz", type=float, default=DEFAULT_TARGET_RATE_HZ,
                   help=f"output sampling rate, Hz (default {DEFAULT_TARGET_RATE_HZ:.0f})")
    p.add_argument("--order", type=int, default=DEFAULT_ORDER, help=f"Bessel filter order (default {DEFAULT_ORDER})")
    p.add_argument("--plot-window-s", type=float, default=2.0,
                   help="length of the before/after comparison plot window, seconds (default 2)")
    p.add_argument("--out", default=None, help="output .npz path (default: alongside the .abf)")
    args = p.parse_args(argv)

    stem = os.path.splitext(args.abf)[0]
    abf = pyabf.ABF(args.abf)
    abf.setSweep(0, channel=args.channel)
    t_raw = np.asarray(abf.sweepX, dtype=float)
    v_raw = np.asarray(abf.sweepY, dtype=float)
    fs_raw = abf.dataRate

    hardware_filter_hz = get_hardware_filter_hz(abf, args.channel)
    if hardware_filter_hz is None:
        print(f"Channel {args.channel}: no telegraphed hardware filter setting found in the "
              f"ABF header (older/non-telegraphing amplifier) -- proceeding with the requested "
              f"{args.cutoff_hz:.0f} Hz filter regardless.")
    elif hardware_filter_hz <= args.cutoff_hz:
        print(f"Channel {args.channel}: recording is ALREADY hardware-filtered at "
              f"{hardware_filter_hz:.0f} Hz (per the amplifier's own telegraphed setting), which "
              f"is at or below the requested {args.cutoff_hz:.0f} Hz cutoff -- applying "
              f"{args.cutoff_hz:.0f} Hz here would be redundant (can't recover detail the "
              f"amplifier already removed). Filtering anyway, but treat this as a no-op check.")
    else:
        print(f"Channel {args.channel}: hardware-filtered at {hardware_filter_hz:.0f} Hz "
              f"(amplifier's telegraphed setting) -- well above the requested {args.cutoff_hz:.0f} Hz, "
              f"so digitally filtering further is meaningful, not redundant.")

    v_filt = bessel_lowpass(v_raw, fs_raw, args.cutoff_hz, args.order)
    v_out, fs_out = downsample(v_filt, fs_raw, args.target_rate_hz)
    t_out = np.arange(len(v_out)) / fs_out
    print(f"{os.path.basename(args.abf)}: {fs_raw:.0f} Hz -> {args.cutoff_hz:.0f} Hz Bessel "
          f"(order {args.order}, zero-phase) -> downsampled to {fs_out:.0f} Hz "
          f"({len(v_raw)} -> {len(v_out)} samples)")

    out_path = args.out or f"{stem}_filtered_{int(args.cutoff_hz)}Hz_{int(fs_out)}Hz.npz"
    np.savez(out_path, t=t_out, v=v_out, fs=fs_out, channel=args.channel,
             cutoff_hz=args.cutoff_hz, order=args.order,
             hardware_filter_hz=hardware_filter_hz if hardware_filter_hz is not None else np.nan,
             source_abf=os.path.basename(args.abf))
    print(f"Saved -> {out_path}")

    n_plot_raw = min(len(v_raw), int(args.plot_window_s * fs_raw))
    n_plot_out = min(len(v_out), int(args.plot_window_s * fs_out))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(t_raw[:n_plot_raw], v_raw[:n_plot_raw], color=TRACE, lw=0.6, alpha=0.5,
             label=f"raw ({fs_raw:.0f} Hz)")
    ax.plot(t_out[:n_plot_out], v_out[:n_plot_out], color="#2a78d6", lw=1.2,
             label=f"{args.cutoff_hz:.0f} Hz Bessel + {fs_out:.0f} Hz ({fs_out:.0f} Hz)")
    ax.set_xlabel("time (s)", color=INK)
    ax.set_ylabel(f"current ({abf.adcUnits[args.channel]})", color=INK)
    ax.set_title(f"{os.path.basename(args.abf)} -- first {args.plot_window_s:.0f}s, raw vs filtered/downsampled",
                  color=INK)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()

    plot_path = f"{stem}_filtered_{int(args.cutoff_hz)}Hz_{int(fs_out)}Hz_preview.png"
    fig.savefig(plot_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {plot_path}")


if __name__ == "__main__":
    main()
