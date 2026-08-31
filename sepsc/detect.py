"""
Detect spontaneous EPSCs in a gap-free voltage-clamp .abf recording using
miniML (Delvendahl lab, https://github.com/delvendahl/miniML), a CNN-LSTM
deep-learning event detector (O'Neill et al. 2025, eLife).

Must be run with the 'clampex_miniml' conda environment -- TensorFlow/miniML
aren't installed in the main project venv the rest of this package uses.
E.g. from the project root:
    C:\\Users\\damia\\anaconda3\\envs\\clampex_miniml\\python.exe -m sepsc.detect <abf_path>
or via the CLI dispatcher (sepsc.cli prints this same hint automatically if
the wrong interpreter is used):
    C:\\Users\\damia\\anaconda3\\envs\\clampex_miniml\\python.exe -m sepsc detect <abf_path>

Usage:
    python -m sepsc.detect path\\to\\recording.abf [--channel 0] [--threshold 0.5] [--plot]
    python -m sepsc.detect recording.abf --filter --cutoff-hz 3000 --target-rate-hz 10000
"""

from __future__ import annotations

import argparse
import os

import pyabf
from miniml import EventDetection
from miniml.core.trace import MiniTrace
from miniml.fileio.trace_loader import TraceLoader
from miniml.resources.util import get_resource_file_path

from .preprocess import DEFAULT_CUTOFF_HZ, DEFAULT_ORDER, DEFAULT_TARGET_RATE_HZ, get_hardware_filter_hz, load_filtered_trace

DEFAULT_MODEL = "models/GC_lstm_model.h5"  # granule-cell mEPSC/sEPSC CNN-LSTM model

# The GC mEPSC model was trained on 50 kHz recordings with a 600-sample
# (12 ms) detection window. window_size is in raw trace samples, so at any
# other sampling rate it must be rescaled to keep that 12 ms real-time span --
# miniML then resamples window_size raw samples down to 600 for inference
# (see EventDetection's resample_to_600). Passing a fixed 600 regardless of
# sampling rate silently shrinks the window at rates above 50 kHz (e.g. 6 ms
# at 100 kHz), which starves the model of enough of the event to recognize it
# and can drive detections to zero with no error.
MODEL_NATIVE_RATE_HZ = 50_000
MODEL_NATIVE_WINDOW = 600

# Below this rescaled window_size, miniML's OWN get_event_baseline (called
# with use_legacy_baseline_method=True below) can crash with "Baseline could
# not be determined": for closely-spaced events it computes
# int(int(window_size * 0.1) / 10) // 2 as a half-width, which floors to 0
# once window_size gets small enough, producing an empty baseline slice ->
# np.mean of that is NaN. Confirmed on a real recording: --target-rate-hz
# 10000 (window_size=120) crashes this way; --target-rate-hz 25000
# (window_size=300) does not. 200 is a safety margin above that observed
# boundary, not the exact miniML cutoff.
MIN_SAFE_WINDOW_SIZE = 200

# Hann-filter widths used to smooth the raw trace and its gradient when
# locating event onsets/peaks (miniML's _make_smth_gradient) -- also raw
# trace samples, also tuned in miniML's own tutorial for 50 kHz data, and
# NOT rescaled internally by miniML the way peak_w is. Left fixed, they
# under-smooth at higher sampling rates, which can make real CNN-flagged
# candidates fail the noisier trace-domain refinement step and get dropped
# before ever reaching event_stats -- a second, independent way sampling
# rate can silently suppress detections beyond just window_size.
MODEL_NATIVE_CONVOLVE_WIN = 15
MODEL_NATIVE_GRADIENT_CONVOLVE_WIN = 40


def detect(
    abf_path: str,
    channel: int = 0,
    threshold: float = 0.5,
    window_size: int | None = None,
    model_rel_path: str = DEFAULT_MODEL,
    plot: bool = False,
    filter_trace: bool = False,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    target_rate_hz: float = DEFAULT_TARGET_RATE_HZ,
    filter_order: int = DEFAULT_ORDER,
) -> EventDetection:
    if filter_trace:
        abf = pyabf.ABF(abf_path)
        hardware_filter_hz = get_hardware_filter_hz(abf, channel)
        if hardware_filter_hz is not None and hardware_filter_hz <= cutoff_hz:
            print(f"NOTE: channel {channel} is already hardware-filtered at "
                  f"{hardware_filter_hz:.0f} Hz (amplifier's telegraphed setting), at or below "
                  f"the requested {cutoff_hz:.0f} Hz -- filtering further is a no-op.")
        _t, v, fs_out, _ = load_filtered_trace(
            abf_path, channel=channel, cutoff_hz=cutoff_hz, target_hz=target_rate_hz, order=filter_order)
        print(f"Filtered: {abf.dataRate:.0f} Hz raw -> {cutoff_hz:.0f} Hz Bessel "
              f"(order {filter_order}, zero-phase) -> {fs_out:.0f} Hz", flush=True)
        trace = MiniTrace(data=v, sampling_interval=1.0 / fs_out, y_unit=abf.adcUnits[channel],
                           filename=os.path.basename(abf_path))
    else:
        trace = TraceLoader.from_axon_file(filename=abf_path, channel=channel, scaling=1.0, unit="pA")

    rate_ratio = trace.sampling_rate / MODEL_NATIVE_RATE_HZ
    if window_size is None:
        window_size = round(MODEL_NATIVE_WINDOW * rate_ratio)
    convolve_win = round(MODEL_NATIVE_CONVOLVE_WIN * rate_ratio)
    gradient_convolve_win = round(MODEL_NATIVE_GRADIENT_CONVOLVE_WIN * rate_ratio)
    print(f"Trace sampled at {trace.sampling_rate:.0f} Hz (ratio {rate_ratio:.2f}x native "
          f"{MODEL_NATIVE_RATE_HZ} Hz) -> window_size={window_size}, "
          f"convolve_win={convolve_win}, gradient_convolve_win={gradient_convolve_win}")
    if window_size < MIN_SAFE_WINDOW_SIZE:
        min_safe_rate = MIN_SAFE_WINDOW_SIZE * MODEL_NATIVE_RATE_HZ / MODEL_NATIVE_WINDOW
        print(f"WARNING: window_size={window_size} is small enough that miniML's own baseline "
              f"routine can crash on closely-spaced events (see MIN_SAFE_WINDOW_SIZE's comment) "
              f"-- if detect_events raises 'Baseline could not be determined', re-run with a "
              f"higher --target-rate-hz (roughly >= {min_safe_rate:.0f} Hz) or without --filter.")

    model_path = get_resource_file_path(model_rel_path)

    detection = EventDetection(
        data=trace,
        model_path=model_path,
        window_size=window_size,
        event_direction="negative",  # sEPSCs are inward (negative) currents in voltage clamp
        model_threshold=threshold,
        batch_size=512,
        compile_model=True,
        verbose=1,
    )

    detection.detect_events(
        eval=True,
        peak_w=5,
        rel_prom_cutoff=0.25,
        convolve_win=convolve_win,
        gradient_convolve_win=gradient_convolve_win,
        use_legacy_baseline_method=True,
    )

    if detection.events_present():
        detection.event_stats.print()
    else:
        print("No events detected above threshold.")

    # Same filt<cutoff>Hz<rate>Hz suffix minianalysis.py appends under
    # --filter -- without it, a filtered and an unfiltered run (or two
    # filtered runs at different settings) would silently overwrite the
    # same *_miniML_avgs.csv/_individual.csv, and sepsc.inspect's own
    # --source miniml --filter stem logic (mirroring minianalysis's)
    # wouldn't be able to find the right one.
    stem = os.path.splitext(abf_path)[0]
    if filter_trace:
        stem += f"_filt{int(cutoff_hz)}Hz{int(target_rate_hz)}Hz"
    out_base = stem + "_miniML"
    detection.save_to_csv(out_base)

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(trace.time_axis, trace.data, linewidth=0.5, color="k")
        if detection.events_present():
            ax.scatter(
                detection.event_peak_times,
                detection.event_peak_values,
                color="r",
                s=10,
                zorder=3,
                label=f"{detection.event_stats.event_count} detected sEPSCs",
            )
            ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Current ({trace.y_unit})")
        fig.tight_layout()
        fig.savefig(out_base + "_trace.png", dpi=150)
        print(f"Trace plot saved to {out_base}_trace.png")

    return detection


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect sEPSCs in a gap-free .abf using miniML")
    parser.add_argument("abf", help="Path to gap-free voltage-clamp .abf file")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5, help="Model detection threshold (0-1)")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--filter", action="store_true",
                         help="Bessel low-pass + downsample the trace (see sepsc.preprocess) before "
                              "detection, instead of feeding miniML the raw trace")
    parser.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ,
                         help=f"only with --filter: Bessel low-pass cutoff, Hz (default {DEFAULT_CUTOFF_HZ:.0f})")
    parser.add_argument("--target-rate-hz", type=float, default=DEFAULT_TARGET_RATE_HZ,
                         help=f"only with --filter: output sampling rate, Hz (default {DEFAULT_TARGET_RATE_HZ:.0f})")
    parser.add_argument("--filter-order", type=int, default=DEFAULT_ORDER,
                         help=f"only with --filter: Bessel filter order (default {DEFAULT_ORDER})")
    args = parser.parse_args(argv)

    detect(args.abf, channel=args.channel, threshold=args.threshold, plot=args.plot,
           filter_trace=args.filter, cutoff_hz=args.cutoff_hz,
           target_rate_hz=args.target_rate_hz, filter_order=args.filter_order)


if __name__ == "__main__":
    main()
