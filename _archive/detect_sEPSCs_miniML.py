# ARCHIVED 2026-08-27: superseded by sepsc/detect.py -- see sepsc/__init__.py
# for the combined pipeline (python -m sepsc detect ...). Kept here only for
# reference; not maintained.
"""
Detect spontaneous EPSCs in a gap-free voltage-clamp .abf recording using
miniML (Delvendahl lab, https://github.com/delvendahl/miniML), a CNN-LSTM
deep-learning event detector (O'Neill et al. 2025, eLife).

Must be run with the 'clampex_miniml' conda environment, e.g.:
    C:\\Users\\damia\\anaconda3\\envs\\clampex_miniml\\python.exe detect_sEPSCs_miniML.py <abf_path>

Usage:
    python detect_sEPSCs_miniML.py path\\to\\recording.abf [--channel 0] [--threshold 0.5] [--plot]
"""

from __future__ import annotations

import argparse
import os

from miniml import EventDetection
from miniml.fileio.trace_loader import TraceLoader
from miniml.resources.util import get_resource_file_path

DEFAULT_MODEL = "models/GC_lstm_model.h5"  # granule-cell mEPSC/sEPSC CNN-LSTM model


def detect(
    abf_path: str,
    channel: int = 0,
    threshold: float = 0.5,
    window_size: int = 600,
    model_rel_path: str = DEFAULT_MODEL,
    plot: bool = False,
) -> EventDetection:
    trace = TraceLoader.from_axon_file(filename=abf_path, channel=channel, scaling=1.0, unit="pA")

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
        convolve_win=15,
        gradient_convolve_win=40,
        use_legacy_baseline_method=True,
    )

    detection.event_stats.print()

    out_base = os.path.splitext(abf_path)[0] + "_miniML"
    detection.save_to_csv(out_base)

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(trace.time_axis, trace.data, linewidth=0.5, color="k")
        ax.scatter(
            detection.event_peak_times,
            detection.event_peak_values,
            color="r",
            s=10,
            zorder=3,
            label=f"{detection.event_stats.event_count} detected sEPSCs",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Current ({trace.y_unit})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_base + "_trace.png", dpi=150)
        print(f"Trace plot saved to {out_base}_trace.png")

    return detection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect sEPSCs in a gap-free .abf using miniML")
    parser.add_argument("abf", help="Path to gap-free voltage-clamp .abf file")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5, help="Model detection threshold (0-1)")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    detect(args.abf, channel=args.channel, threshold=args.threshold, plot=args.plot)
