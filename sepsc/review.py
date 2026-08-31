"""
Manually accept/reject detected sEPSC events by eye -- works with any of
this pipeline's three sEPSC detectors (miniML's detect.py, the classical
Mini-Analysis-style minianalysis.py, or the per-recording MLP peel-off
fastmini.py), chosen with --source.

Same window as `sepsc inspect` (this module builds directly on inspect.py's
own EventInspector, not a separate window of its own) -- a zoomable overview
of the WHOLE trace with every candidate marked, click one (or Right/N,
Left/P) to see it in the detail panel below (the same annotated
detection-window view for --source minianalysis, the same simpler peak+stats
view for --source miniml -- see inspect.py's own module docstring), then
Accept/Reject it (buttons or A/R keys) or "Accept All Remaining". Progress is
autosaved after every decision (identical output schema to `inspect`'s own
QC output -- the two are fully interchangeable, see below), and unlike a
plain queue, already-decided events stay visible (with a colored halo) and
clickable, so revisiting one is just clicking it again.

The one thing `inspect` doesn't have: for --source miniml/minianalysis (NOT
fastmini -- see below), if the OTHER one's output also exists next to the
.abf, it's loaded read-only for comparison -- its candidates are marked on
the overview too (a distinct triangle marker), and each event's detail
title states whether it was ALSO flagged by that other method -- exactly
what you need to build ground truth for a false-positive-rate comparison
between the two detectors. --source fastmini has no such automatic
comparison (there's no single natural "other" detector to diff a third
method against) -- it reviews on its own, --compare-csv/--no-compare are
ignored for it.

Output (written next to the source .abf, autosaved after every decision;
identical to `sepsc inspect`'s own -- reviewing/inspecting the same source
resumes each other's progress):
    <name>_<source>_reviewed.csv          one row per ACCEPTED event (same
                                           feature columns as the input
                                           CSV, plus location_s)
    <name>_<source>_review_progress.csv   every reviewed event (accepted
                                           or rejected) -- re-running the
                                           tool on the same file/source
                                           resumes from the first
                                           un-reviewed event using this
                                           file.
    (<source> is "miniML" or "minianalysis" -- reviewing one source never
    touches the other's review files.)

Usage
-----
    python -m sepsc.review path\\to\\recording.abf
    python -m sepsc.review path\\to\\recording.abf --source minianalysis
    python -m sepsc.review path\\to\\recording.abf --csv custom_individual.csv
    python -m sepsc.review path\\to\\recording.abf --filter  # if that CSV came from a --filter'd
        # minianalysis/detect run -- MUST match those --cutoff-hz/--target-rate-hz settings, since
        # 'location'/'peak_idx' values are indices into that filtered/resampled trace, not the raw
        # one; also switches the default --csv/--compare-csv/output paths to that run's own
        # <stem>_filt<cutoff>Hz<rate>Hz-suffixed files. Passing --filter when the CSV is actually a
        # raw-trace run's output (or vice versa) is caught up front with a clear error, rather than
        # silently reviewing events in the wrong place on the trace.
    python -m sepsc.review path\\to\\recording.abf --shuffle --sample 50  # optional: only show/step
        # through a random 50-event subset this session (the overview only marks that subset too) --
        # everything else about the window is unaffected.

Controls: same as `sepsc inspect` -- click a marker in the top panel to
inspect that event, Right/N and Left/P to step, A/Up = accept, R/Down =
reject, "Accept All Remaining" button, scroll/drag/toolbar to pan+zoom the
overview.
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd

from PyQt5 import QtWidgets

from .inspect import EventInspector, _load_params, load_source_events, resolve_source_stem, load_display_trace
from .inspect import SOURCE_CFG as INSPECT_SOURCE_CFG
from .preprocess import DEFAULT_CUTOFF_HZ, DEFAULT_ORDER, DEFAULT_TARGET_RATE_HZ

# Matches the "_filt<cutoff>Hz<rate>Hz" stem suffix minianalysis.py/detect.py
# append under their own --filter -- used by main() to catch someone passing
# a filtered-detection CSV (via --csv, or a stem that happens to already
# carry this suffix) without also passing --filter here, which would load
# this tool's own trace at the wrong sample rate for that CSV's `location`
# indices. See the --filter mismatch check in main().
_FILT_STEM_RE = re.compile(r"_filt\d+Hz\d+Hz")


# main() below builds its window on inspect.py's EventInspector, which uses
# its own load_source_events()/SOURCE_CFG (peak_idx/peak_time_s columns)
# instead of these two -- kept here (location/location_s columns) only
# because compare.py and view.py still import SOURCES/these loaders
# directly for their own, non-EventInspector uses. Don't remove without
# updating those.
def load_miniml_events(csv_path: str, sample_rate_hz: float) -> pd.DataFrame:
    """Read a *_miniML_individual.csv (rows=features, cols=event_N) into a
    tidy one-row-per-event DataFrame with an added location_s column.

    `sample_rate_hz` MUST be the sample rate of the trace whose sample grid
    `location` is an index into -- the recording's native rate normally, or
    the --filter'd target rate if `detect` was run with --filter. Passing
    the wrong one silently produces a valid-looking but wrong location_s for
    every event.
    """
    raw = pd.read_csv(csv_path, index_col=0)
    events = raw.T.reset_index(drop=True)
    events["location_s"] = events["location"] / sample_rate_hz
    return events


def load_minianalysis_events(csv_path: str, sample_rate_hz: float) -> pd.DataFrame:
    """Read a *_minianalysis_events.csv (already tidy, one row per event)
    and rename its columns onto the same location/location_s/amplitude
    schema load_miniml_events produces, so compare.py/view.py can treat
    either source uniformly. `sample_rate_hz` is accepted only for a uniform
    loader signature; unlike miniML's, this CSV already carries peak_time_s
    directly (computed by minianalysis.py against whatever trace -- raw or
    --filter'd -- it actually used, so it's already a correct absolute time
    either way).
    """
    df = pd.read_csv(csv_path)
    return df.rename(columns={"peak_idx": "location", "peak_time_s": "location_s"})


SOURCES = {
    "miniml": dict(
        label="miniML",
        individual_suffix="_miniML_individual.csv",
        out_suffix="_miniML_reviewed.csv",
        progress_suffix="_miniML_review_progress.csv",
        detect_hint="python -m sepsc detect",
        loader=load_miniml_events,
    ),
    "minianalysis": dict(
        label="Mini-Analysis-style",
        individual_suffix="_minianalysis_events.csv",
        out_suffix="_minianalysis_reviewed.csv",
        progress_suffix="_minianalysis_review_progress.csv",
        detect_hint="python -m sepsc minianalysis",
        loader=load_minianalysis_events,
    ),
}


def _check_filter_mismatch(p: argparse.ArgumentParser, csv_path: str, filter_flag: bool, what: str):
    """Catch the (silently-wrong-results) case of a filtered-detection CSV
    reviewed without --filter, or vice versa -- see _FILT_STEM_RE. Filename-
    pattern based, so it only catches the two source/compare CSVs actually
    used here, not e.g. a renamed file -- see the location-bounds check in
    main() below for a second, content-based backstop."""
    looks_filtered = bool(_FILT_STEM_RE.search(os.path.basename(csv_path)))
    if looks_filtered and not filter_flag:
        p.error(
            f"{what} {csv_path!r} looks like output from a --filter'd detection run (its filename "
            f"contains '_filt<cutoff>Hz<rate>Hz'), but --filter wasn't passed to this review command. "
            f"Its 'location'/'peak_idx' column is an index into that filtered/resampled trace, not the "
            f"raw one this tool would otherwise load -- re-run with --filter (and matching --cutoff-hz/"
            f"--target-rate-hz) so the trace loaded here matches.")
    elif filter_flag and not looks_filtered:
        p.error(
            f"--filter was passed, but {what} {csv_path!r} doesn't look like a --filter'd detection "
            f"run's output (no '_filt<cutoff>Hz<rate>Hz' in its filename) -- its 'location'/'peak_idx' "
            f"column is probably an index into the RAW trace, not the filtered/resampled one --filter "
            f"would load here. Drop --filter, or pass the correct --csv/--compare-csv.")


def _check_location_bounds(p: argparse.ArgumentParser, events: pd.DataFrame, n_samples: int,
                            csv_path: str, filter_flag: bool):
    """Content-based backstop for the same raw/filtered trace mismatch
    _check_filter_mismatch guards against by filename -- catches it even for
    a CSV whose name doesn't happen to match _FILT_STEM_RE (a manually
    renamed file, say). `events` here is inspect.load_source_events' own
    output, so the column is 'peak_idx', not 'location'."""
    max_loc = int(events["peak_idx"].max())
    if max_loc >= n_samples:
        p.error(
            f"event location index {max_loc} (from {os.path.basename(csv_path)}) is out of range for "
            f"the loaded trace ({n_samples} samples) -- this usually means the trace loaded here "
            f"doesn't match the sample grid that CSV's location column is indexed into. Check that "
            f"--filter{'' if filter_flag else ' (currently NOT passed)'} and --cutoff-hz/--target-rate-hz "
            f"match whatever run produced it.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", help="Path to the gap-free voltage-clamp .abf file")
    p.add_argument("--source", choices=list(INSPECT_SOURCE_CFG), default="miniml",
                   help="which detector's candidates to review (default: miniml). --source fastmini "
                        "has no automatic comparison overlay -- see --compare-csv below")
    p.add_argument("--csv", default=None,
                   help="Path to the source's events CSV (default: <abf>_<source's own suffix>)")
    p.add_argument("--compare-csv", default=None,
                   help="Path to the OTHER detector's events CSV for the comparison overlay "
                        "(default: auto-detected next to the .abf if present)")
    p.add_argument("--no-compare", action="store_true",
                   help="disable loading the other detector's output for comparison")
    p.add_argument("--compare-tolerance-ms", type=float, default=2.0,
                   help="max time difference (ms) to count as 'also flagged by' the other "
                        "detector (default 2)")
    p.add_argument("--channel", type=int, default=0, help="ADC channel to display (default 0)")
    p.add_argument("--pre-ms", type=float, default=None,
                   help="detail-view window before the peak, ms (default: same auto-default "
                        "`inspect` uses per --source)")
    p.add_argument("--post-ms", type=float, default=None,
                   help="detail-view window after the peak, ms (default: same auto-default "
                        "`inspect` uses per --source)")
    p.add_argument("--window", type=float, default=None,
                   help="convenience shorthand for --pre-ms/--post-ms together (a single "
                        "symmetric half-window, ms) -- --pre-ms/--post-ms individually take "
                        "precedence over this if also passed")
    p.add_argument("--filter", action="store_true",
                   help="load the SAME Bessel-filtered + resampled trace minianalysis.py's/detect.py's "
                        "own --filter produced (must match whatever --filter settings that detection run "
                        "used, since event 'location'/'peak_idx' values are indices into that "
                        "filtered/resampled trace, not the raw one) -- also changes the default "
                        "--csv/--compare-csv/output paths to that run's own "
                        "<stem>_filt<cutoff>Hz<rate>Hz-suffixed files")
    p.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ,
                   help=f"only with --filter: Bessel low-pass cutoff, Hz (default {DEFAULT_CUTOFF_HZ:.0f})")
    p.add_argument("--target-rate-hz", type=float, default=DEFAULT_TARGET_RATE_HZ,
                   help=f"only with --filter: output sampling rate, Hz (default {DEFAULT_TARGET_RATE_HZ:.0f})")
    p.add_argument("--filter-order", type=int, default=DEFAULT_ORDER,
                   help=f"only with --filter: Bessel filter order (default {DEFAULT_ORDER})")
    p.add_argument("--shuffle", action="store_true",
                   help="show/navigate a randomly-reordered subset (see --sample) instead of "
                        "sequential time order -- avoids biasing a partial review toward "
                        "whatever happens to be early in the recording")
    p.add_argument("--seed", type=int, default=0, help="random seed for --shuffle (default 0)")
    p.add_argument("--sample", type=int, default=None,
                   help="only show/step through this many randomly-chosen events this session "
                        "(most useful with --shuffle, to draw a fixed-size random sample) -- "
                        "the overview only marks this subset, not the full candidate set")
    args = p.parse_args(argv)

    cfg = INSPECT_SOURCE_CFG[args.source]
    stem, filter_flag, cutoff_hz, target_rate_hz, filter_order = resolve_source_stem(
        args.abf, args.source, args.filter, args.cutoff_hz, args.target_rate_hz, args.filter_order)

    csv_path = args.csv or f"{stem}{cfg['events_suffix']}"
    if not os.path.exists(csv_path):
        detect_hint = {"minianalysis": "python -m sepsc minianalysis", "fastmini": "python -m sepsc fastmini",
                        "miniml": "<clampex_miniml env>\\python.exe -m sepsc detect"}[args.source]
        p.error(f"events CSV not found: {csv_path!r} (run `{detect_hint} {args.abf}"
                 f"{' --filter' if args.filter and args.source != 'fastmini' else ''}` first, or pass --csv)")
    # fastmini's own convention (always implicitly filtered, never a
    # _filt<..>-suffixed filename -- see resolve_source_stem) doesn't fit
    # this filename-pattern check at all, so it's skipped for that source;
    # the content-based _check_location_bounds below still applies.
    if args.source != "fastmini":
        _check_filter_mismatch(p, csv_path, args.filter, "the events CSV")

    t, v, dt, y_unit = load_display_trace(args.abf, args.channel, filter_flag, cutoff_hz,
                                           target_rate_hz, filter_order)

    # inspect.py's own loader (peak_idx/peak_time_s columns, needed by
    # EventInspector below) -- not SOURCES[...]['loader'] above, which is
    # kept only for compare.py/view.py's location/location_s schema.
    df = load_source_events(csv_path, args.source, t)
    if df.empty:
        p.error("no events found in the CSV")
    _check_location_bounds(p, df, len(v), csv_path, args.filter)

    if args.source == "minianalysis":
        params = _load_params(None, stem, {})
        default_pre_ms = params.baseline_before_ms + params.baseline_avg_ms + 3.0
        default_post_ms = params.decay_search_ms + 3.0
    elif args.source == "fastmini":
        params = None
        default_pre_ms, default_post_ms = 8.0, 23.0
    else:
        params = None
        default_pre_ms, default_post_ms = 15.0, 25.0
    pre_ms = args.pre_ms if args.pre_ms is not None else (args.window if args.window is not None else default_pre_ms)
    post_ms = args.post_ms if args.post_ms is not None else (args.window if args.window is not None else default_post_ms)

    # The automatic "other detector" comparison overlay only makes sense
    # between the original two methods (miniML vs minianalysis, SOURCES'
    # only two keys) -- fastmini has no single natural "other" to diff
    # against, so it's skipped entirely for that source (module docstring).
    compare_events, compare_label = None, None
    if not args.no_compare and args.source != "fastmini":
        other_source = next(k for k in SOURCES if k != args.source)
        other_cfg = SOURCES[other_source]
        compare_csv = args.compare_csv or f"{stem}{other_cfg['individual_suffix']}"
        if os.path.exists(compare_csv):
            _check_filter_mismatch(p, compare_csv, args.filter, "the comparison events CSV")
            compare_events = load_source_events(compare_csv, other_source, t)
            _check_location_bounds(p, compare_events, len(v), compare_csv, args.filter)
            compare_label = other_cfg["label"]
            print(f"Loaded {len(compare_events)} {compare_label} candidates for comparison "
                  f"(from {os.path.basename(compare_csv)}, read-only)", flush=True)
        else:
            print(f"No {other_cfg['label']} output found ({os.path.basename(compare_csv)}) "
                  f"-- reviewing without a comparison overlay.", flush=True)
    elif args.source == "fastmini" and (not args.no_compare) and (args.compare_csv is not None):
        print("NOTE: --compare-csv is ignored for --source fastmini (no automatic comparison "
              "overlay for this source -- see module docstring).", flush=True)

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        print(f"Shuffled event order (seed={args.seed}).", flush=True)
    if args.sample is not None:
        df = df.iloc[:args.sample].reset_index(drop=True)
        print(f"Limited to a {len(df)}-event sample this session "
              f"({'shuffled' if args.shuffle else 'in CSV order'}).", flush=True)

    reviewed_path = f"{stem}{cfg['reviewed_suffix']}"
    progress_path = f"{stem}{cfg['progress_suffix']}"
    print(f"{len(df)} {cfg['label']} events from {os.path.basename(csv_path)}", flush=True)
    print("Click a red X in the top panel to inspect that event, or use Right/N, Left/P to step. "
          "A/Up = accept, R/Down = reject.", flush=True)
    if os.path.exists(progress_path) or os.path.exists(reviewed_path):
        print(f"Resuming QC from {os.path.basename(progress_path)}"
              f"{' / ' + os.path.basename(reviewed_path) if os.path.exists(reviewed_path) else ''}", flush=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = EventInspector(t, v, dt, df, params, y_unit,
                             title=f"sEPSC review [{cfg['label']}] -- {os.path.basename(args.abf)}",
                             pre_ms=pre_ms, post_ms=post_ms,
                             reviewed_path=reviewed_path, progress_path=progress_path,
                             source=args.source, compare_events=compare_events,
                             compare_label=compare_label,
                             compare_tolerance_s=args.compare_tolerance_ms / 1000.0)
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
