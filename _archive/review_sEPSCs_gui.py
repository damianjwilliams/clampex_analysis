# ARCHIVED 2026-08-27: superseded by sepsc/review.py -- see sepsc/__init__.py
# for the combined pipeline (python -m sepsc review ...). Kept here only for
# reference; not maintained.
"""
review_sEPSCs_gui.py
=====================

Manually accept/reject miniML-detected sEPSC events by eye.

Loads the raw gap-free .abf trace and the companion *_miniML_individual.csv
table produced by detect_sEPSCs_miniML.py, then steps through every
detected event one at a time in a short window centered on its peak.
Click "Accept" to keep it or "Reject" to discard it -- either way the tool
immediately advances to the next event. Progress is autosaved after every
decision, and the final kept-events table is written when review finishes
(or the window is closed early).

Output (written next to the source .abf):
    <name>_miniML_reviewed.csv          one row per ACCEPTED event (same
                                         feature columns as the input CSV,
                                         plus location_s)
    <name>_miniML_review_progress.csv   every reviewed event (accepted or
                                         rejected) -- re-running the tool on
                                         the same file resumes from the first
                                         un-reviewed event using this file.

Usage
-----
    python review_sEPSCs_gui.py path\\to\\recording.abf
    python review_sEPSCs_gui.py path\\to\\recording.abf --window 40
    python review_sEPSCs_gui.py path\\to\\recording.abf --csv custom_individual.csv

Controls: click Accept / Reject, or use the Right/A arrow key = accept,
Left/R arrow key = reject.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import traceback

import numpy as np
import pandas as pd
import pyabf

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def load_events(csv_path: str, data_rate_hz: float) -> pd.DataFrame:
    """Read a *_miniML_individual.csv (rows=features, cols=event_N) into a
    tidy one-row-per-event DataFrame with an added location_s column."""
    raw = pd.read_csv(csv_path, index_col=0)
    events = raw.T.reset_index(drop=True)
    events["location_s"] = events["location"] / data_rate_hz
    return events


def safe_callback(fn):
    """Never let an exception escape a Tk/matplotlib callback silently.

    An uncaught exception inside a button/key callback can leave the Tk
    mainloop in a state where the window stops updating but still answers
    the OS "are you responding" ping -- it looks frozen forever with no
    error printed anywhere. Catch, print (flushed immediately, since stdout
    is fully buffered once redirected to a file/pipe), and re-raise nothing.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            print(f"ERROR in {fn.__name__}:", file=sys.stderr, flush=True)
            traceback.print_exc()
            sys.stderr.flush()
    return wrapper


class ReviewSession:
    def __init__(self, abf: pyabf.ABF, channel: int, events: pd.DataFrame,
                 window_ms: float, out_path: str, progress_path: str):
        self.out_path = out_path
        self.progress_path = progress_path
        self.events = events.reset_index(drop=True)
        self.window_s = window_ms / 1000.0
        self.decisions = [None] * len(self.events)  # True=accept, False=reject
        self._busy = False  # reentrancy guard: ignore clicks while one is in flight

        self.idx = self._load_progress()

        abf.setSweep(0, channel=channel)
        self.t = np.asarray(abf.sweepX, float)
        self.v = np.asarray(abf.sweepY, float)
        self.y_unit = abf.adcUnits[channel]
        self.dt = self.t[1] - self.t[0]

        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self.fig.canvas.manager.set_window_title("sEPSC review")
        plt.subplots_adjust(bottom=0.22)
        ax_reject = plt.axes([0.28, 0.05, 0.18, 0.09])
        ax_accept = plt.axes([0.54, 0.05, 0.18, 0.09])
        self.btn_reject = Button(ax_reject, "Reject", color="#f2a0a0", hovercolor="#e57373")
        self.btn_accept = Button(ax_accept, "Accept", color="#a0d8a0", hovercolor="#66bb6a")
        self.btn_reject.on_clicked(self.reject)
        self.btn_accept.on_clicked(self.accept)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        self.show_event()

    def _load_progress(self) -> int:
        """Resume from a prior *_miniML_review_progress.csv if one exists.

        Matches saved rows back to self.events by 'location' (the sample
        index miniML assigned the event -- stable across runs) rather than
        by row position, so this is safe even if the individual-events CSV
        was regenerated with a different event count.

        One-time migration: an older version of this script only ever wrote
        the accepted-events file (out_path), with no progress file. If that's
        all that exists, treat those rows as already-accepted so a restart
        after a crash/freeze doesn't force re-reviewing them.
        """
        source_path = self.progress_path
        if not os.path.exists(self.progress_path):
            if os.path.exists(self.out_path):
                prior = pd.read_csv(self.out_path)
                loc_to_decision = dict(zip(prior["location"], [True] * len(prior)))
                source_path = self.out_path
            else:
                return 0
        else:
            prior = pd.read_csv(self.progress_path)
            loc_to_decision = dict(zip(prior["location"], prior["decision"] == 1))
        n_restored = 0
        for i, loc in enumerate(self.events["location"]):
            if loc in loc_to_decision:
                self.decisions[i] = bool(loc_to_decision[loc])
                n_restored += 1
        if n_restored:
            print(f"Resuming: {n_restored}/{len(self.events)} events already reviewed "
                  f"(from {os.path.basename(source_path)}).", flush=True)
        next_idx = next((i for i, d in enumerate(self.decisions) if d is None), len(self.events))
        return next_idx

    @safe_callback
    def on_key(self, key_event):
        if key_event.key in ("right", "a", "y"):
            self.accept(key_event)
        elif key_event.key in ("left", "r", "n"):
            self.reject(key_event)

    def show_event(self):
        if self.idx >= len(self.events):
            self.finish()
            return

        row = self.events.iloc[self.idx]
        center_s = row["location_s"]
        i0 = max(0, int((center_s - self.window_s) / self.dt))
        i1 = min(len(self.t), int((center_s + self.window_s) / self.dt))

        self.ax.clear()
        self.ax.plot(self.t[i0:i1] * 1e3, self.v[i0:i1], color="k", lw=0.8)
        self.ax.axvline(center_s * 1e3, color="crimson", ls="--", lw=1, label="detected peak")
        self.ax.set_xlabel("time (ms)")
        self.ax.set_ylabel(f"current ({self.y_unit})")

        n_reviewed = sum(d is not None for d in self.decisions)
        self.ax.set_title(
            f"Event {self.idx + 1}/{len(self.events)}   t={center_s:.3f}s   "
            f"amp={row['amplitude']:.1f} {self.y_unit}   score={row['score']:.3f}\n"
            f"reviewed so far: {n_reviewed}/{len(self.events)}"
        )
        self.ax.legend(loc="upper right", fontsize=8)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _advance(self, accepted: bool):
        # Reentrancy guard: a fast double-click (or a key repeat while the
        # button's own click is still being processed) can fire this twice
        # before the first call returns. Without this guard the second call
        # clears/redraws the same Axes mid-draw, which can wedge Tk's event
        # loop -- the window then reports "responding" to Windows (its
        # message pump is technically alive) but never repaints again.
        if self._busy or self.idx >= len(self.events):
            return
        self._busy = True
        try:
            self.decisions[self.idx] = accepted
            self._autosave()
            self.idx += 1
            self.show_event()
        finally:
            self._busy = False

    @safe_callback
    def accept(self, _event):
        self._advance(True)

    @safe_callback
    def reject(self, _event):
        self._advance(False)

    def _kept(self) -> pd.DataFrame:
        return self.events[[d is True for d in self.decisions]].copy()

    def _reviewed_with_decisions(self) -> pd.DataFrame:
        mask = [d is not None for d in self.decisions]
        reviewed = self.events[mask].copy()
        reviewed["decision"] = [int(d) for d in self.decisions if d is not None]
        return reviewed

    def _autosave(self):
        self._kept().to_csv(self.out_path, index=False)
        self._reviewed_with_decisions().to_csv(self.progress_path, index=False)

    @safe_callback
    def on_close(self, _event):
        n_reviewed = sum(d is not None for d in self.decisions)
        if n_reviewed and n_reviewed < len(self.events):
            print(f"Window closed early: {n_reviewed}/{len(self.events)} events reviewed.", flush=True)
            print(f"Partial results ({int(sum(d is True for d in self.decisions))} accepted) saved -> {self.out_path}", flush=True)
            print(f"Re-run the same command to resume from event {n_reviewed + 1}.", flush=True)

    def finish(self):
        kept = self._kept()
        kept.to_csv(self.out_path, index=False)

        self.ax.clear()
        self.ax.axis("off")
        self.ax.text(
            0.5, 0.5,
            f"Review complete.\n{len(kept)}/{len(self.events)} accepted.\nSaved to:\n{self.out_path}",
            ha="center", va="center", transform=self.ax.transAxes, fontsize=11,
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        print(f"Review complete: {len(kept)}/{len(self.events)} events accepted.", flush=True)
        print(f"Saved -> {self.out_path}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", help="Path to the gap-free voltage-clamp .abf file")
    p.add_argument("--csv", default=None,
                   help="Path to the *_miniML_individual.csv (default: <abf>_miniML_individual.csv)")
    p.add_argument("--channel", type=int, default=0, help="ADC channel to display (default 0)")
    p.add_argument("--window", type=float, default=50.0,
                   help="half-window shown around each event peak, in ms (default 50)")
    args = p.parse_args(argv)

    stem = os.path.splitext(args.abf)[0]
    csv_path = args.csv or f"{stem}_miniML_individual.csv"
    if not os.path.exists(csv_path):
        p.error(f"events CSV not found: {csv_path!r} (run detect_sEPSCs_miniML.py first, or pass --csv)")

    abf = pyabf.ABF(args.abf)
    events = load_events(csv_path, abf.dataRate)
    if events.empty:
        p.error("no events found in the CSV")

    out_path = f"{stem}_miniML_reviewed.csv"
    progress_path = f"{stem}_miniML_review_progress.csv"
    print(f"Reviewing {len(events)} events from {os.path.basename(csv_path)}", flush=True)
    print("Accept = keep, Reject = discard. Arrow keys also work (Right=accept, Left=reject).", flush=True)
    ReviewSession(abf, args.channel, events, window_ms=args.window,
                  out_path=out_path, progress_path=progress_path)
    plt.show()


if __name__ == "__main__":
    main()
