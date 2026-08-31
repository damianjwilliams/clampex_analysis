"""
Interactive full-trace viewer: the whole gap-free recording with every
detector's events marked on it, in a PyQt/pyqtgraph window with mouse-wheel
zoom and click-drag pan/scroll -- unlike review.py's one-event-at-a-time
queue, this is for eyeballing detection coverage across the entire trace at
a glance (missed regions, event clusters, the handful of huge non-canonical
transients seen in this recording).

Only ever READS event CSVs -- nothing here writes to any of them.

For each source (miniml, minianalysis) prefers the reviewed/accepted CSV if
one exists next to the .abf, falling back to the raw detector output
otherwise, so this always shows the best curation available without extra
flags.

Usage
-----
    python -m sepsc.view path\\to\\recording.abf
    python -m sepsc.view path\\to\\recording.abf --channel 0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyabf

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from .review import SOURCES
from .style import COLORS, SURFACE, INK, MUTED, GRID, TRACE

# pyqtgraph scatter symbol codes -- not the same alphabet as matplotlib's
# MARKERS in style.py, so translated once here rather than growing style.py
# a second, GUI-toolkit-specific marker mapping.
SYMBOLS = {"fast": "t1", "slow": "s"}
SOURCE_STYLE_KEY = {"miniml": "fast", "minianalysis": "slow"}


def _resolve_csv(stem: str, cfg: dict) -> tuple[str | None, bool]:
    """Prefer the reviewed/accepted CSV; fall back to the raw detector
    output. Returns (path, is_reviewed) or (None, False) if neither exists."""
    reviewed = f"{stem}{cfg['out_suffix']}"
    if os.path.exists(reviewed):
        return reviewed, True
    raw = f"{stem}{cfg['individual_suffix']}"
    if os.path.exists(raw):
        return raw, False
    return None, False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("abf", help="Path to the gap-free voltage-clamp .abf file")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--miniml-csv", default=None,
                         help="Override: path to a miniML events CSV (default: auto-detected next to the .abf)")
    parser.add_argument("--minianalysis-csv", default=None,
                         help="Override: path to a minianalysis events CSV (default: auto-detected next to the .abf)")
    args = parser.parse_args(argv)

    stem = os.path.splitext(args.abf)[0]
    abf = pyabf.ABF(args.abf)
    abf.setSweep(0, channel=args.channel)
    t = np.asarray(abf.sweepX, dtype=float)
    v = np.asarray(abf.sweepY, dtype=float)
    y_unit = abf.adcUnits[args.channel]

    overrides = {"miniml": args.miniml_csv, "minianalysis": args.minianalysis_csv}
    event_sets = []  # (label, style_key, df, is_reviewed)
    for source_key, cfg in SOURCES.items():
        csv_path = overrides[source_key]
        is_reviewed = False
        if csv_path is None:
            csv_path, is_reviewed = _resolve_csv(stem, cfg)
        elif csv_path.endswith(cfg["out_suffix"]):
            is_reviewed = True
        if csv_path is None or not os.path.exists(csv_path):
            print(f"No {cfg['label']} events found next to the .abf -- skipping "
                  f"(run `{cfg['detect_hint']}` first if you want them shown)")
            continue
        # Reviewed/accepted CSVs are already tidy (review.py writes back the
        # same location/location_s/amplitude schema the loader below
        # produces) -- only the raw detector output needs source-specific
        # parsing (e.g. miniML's transposed rows=features format).
        df = pd.read_csv(csv_path) if is_reviewed else cfg["loader"](csv_path, abf.dataRate)
        event_sets.append((cfg["label"], SOURCE_STYLE_KEY[source_key], df, is_reviewed))
        kind = "reviewed/accepted" if is_reviewed else "raw candidates"
        print(f"Loaded {len(df)} {cfg['label']} events ({kind}) from {os.path.basename(csv_path)}")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOption("background", SURFACE)
    pg.setConfigOption("foreground", INK)
    # Antialiasing forces Qt's slow per-frame line-smoothing path; at 12M
    # points that's the difference between instant and laggy pan/zoom.
    # (useOpenGL was tried too, but pyqtgraph 0.14's OpenGL viewport breaks
    # PlotWidget's view re-parenting -- AttributeError: autoRangeEnabled --
    # so left off; antialias=False + a cosmetic pen already do most of the
    # work without that instability.)
    pg.setConfigOptions(antialias=False)

    # NOTE: on show, pyqtgraph 0.14 prints one harmless
    # "AttributeError: autoRangeEnabled" traceback to stderr from inside
    # PlotDataItem's autoDownsample view-range handler (confirmed to happen
    # identically with GraphicsLayoutWidget/addPlot too, so it's not a
    # PlotWidget-specific issue) -- it's caught internally and doesn't
    # affect the window, so left as PlotWidget for simplicity.
    win = pg.PlotWidget()
    win.setWindowTitle(f"sEPSC trace -- {os.path.basename(args.abf)}")
    win.resize(1400, 700)
    plot_item = win.getPlotItem()
    plot_item.setLabel("bottom", "time", units="s")
    plot_item.setLabel("left", f"current ({y_unit})")
    plot_item.showGrid(x=True, y=True, alpha=0.15)
    plot_item.addLegend(offset=(10, 10))

    # autoDownsample + clipToView: only the samples relevant to the current
    # view are ever drawn, so a 12M-point trace still pans/zooms smoothly --
    # this is the whole reason for pyqtgraph over matplotlib here. width=0 is
    # a Qt "cosmetic" pen (always 1px, ignores transforms) -- any nonzero
    # width switches Qt to a much slower stroked-path renderer per frame.
    plot_item.plot(t, v, pen=pg.mkPen(TRACE, width=0), autoDownsample=True,
                     downsampleMethod="peak", clipToView=True, name="raw trace")

    for label, style_key, df, is_reviewed in event_sets:
        loc = df["location"].to_numpy(dtype=np.int64)
        loc = np.clip(loc, 0, len(v) - 1)
        x = df["location_s"].to_numpy(dtype=float)
        y = v[loc]
        suffix = " (reviewed)" if is_reviewed else " (raw)"
        scatter = pg.ScatterPlotItem(
            x=x, y=y, size=9, symbol=SYMBOLS[style_key],
            pen=pg.mkPen(INK, width=0.5), brush=pg.mkBrush(COLORS[style_key]),
            name=f"{label}{suffix} (n={len(df)})",
        )
        plot_item.addItem(scatter)

    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
