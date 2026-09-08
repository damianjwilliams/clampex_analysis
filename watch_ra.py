#!/usr/bin/env python3
"""
watch_ra.py
===========

Watch a folder and, whenever a new .abf file is saved whose size is close to a
target (default ~801 KB), automatically run the membrane-test / access-resistance
script on it.

Why the size filter matters twice
---------------------------------
1. It selects the right recordings. A fixed-length Membrane-Test protocol always
   produces files of very nearly the same size, so "~801 KB" is a simple, robust
   way to pick out those recordings and ignore everything else (gap-free records,
   other protocols, etc.).
2. It confirms the file is finished. Clampex creates the .abf on disk and then
   streams data into it, so a file can appear at a few KB and grow. This watcher
   therefore waits until the size stops changing (stable for a few reads) AND
   matches the target before it runs anything -- so the analysis never sees a
   half-written file.

Which access-resistance method is used
--------------------------------------
Ra is not computed here -- this watcher only launches the analysis script. That
script offers two methods that give DIFFERENT numbers, so rather than leaning on
its default, the watcher always passes the choice explicitly:

    exact       Ra = tau / Cm_charge -- the exact, non-iterative solution for
    (default)   the series model Ra + (Rm||Cm).
    quadratic   the Molecular Devices note's Newton-Raphson root of
                Ra^2 - Rt*Ra + Rt*(tau/Cm) -- biased, and has no real root once
                Ra > Rt/4.

Choose with --method. Because it is always forwarded explicitly, the method
appears in the startup banner and in every launched command line this watcher
logs, so a run's console output on its own says which one produced the Ra
figure. See the analysis script's module docstring for the derivation of both.

Usage
-----
    # watch the Clampex data folder, run the analysis with your usual flags:
    python watch_ra.py "C:\\Users\\Set0\\Documents\\Molecular Devices\\pCLAMP\\Data" ^
        --cross-check --plot-average --show

    # the published quadratic method instead of the default exact one:
    python watch_ra.py . --method quadratic

    # anything this script does not recognise is forwarded to the analysis
    # script, so --cross-check / --plot / --show / --channel etc. all work.

    # tune the size gate (values are in KB as shown by Windows Explorer):
    python watch_ra.py . --size-kb 801 --tol-kb 40

    # test the watcher without launching the analysis:
    python watch_ra.py . --dry-run

    # process the next matching file and then exit:
    python watch_ra.py . --once

By default only files that appear AFTER the watcher starts are processed; pass
--process-existing to also run on matching files already in the folder.

This script has no third-party dependencies (pure standard library); it launches
the analysis script as a subprocess, so an interactive --show window behaves
exactly as it would if you ran the analysis by hand.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Locating the analysis script
# ----------------------------------------------------------------------------
_ANALYSIS_CANDIDATES = (
    "real_time_ra_analysis.py",
    "membrane_test.py",
    "calculate_Ra.py",
)

# One-line reminder of what each --method value actually computes, so the
# banner names the formula and not just the label. Kept in sync with the
# analysis script's "Solving for Ra" docstring section.
_METHOD_FORMULA = {
    "exact": "Ra = tau / Cm_charge",
    "quadratic": "Newton-Raphson root of Ra^2 - Rt*Ra + Rt*(tau/Cm)",
}


def find_analysis_script(explicit: str | None) -> str:
    """Return a path to the membrane-test script, or exit with a clear error."""
    if explicit:
        if os.path.isfile(explicit):
            return os.path.abspath(explicit)
        sys.exit(f"--script not found: {explicit}")
    here = os.path.dirname(os.path.abspath(__file__))
    for name in _ANALYSIS_CANDIDATES:
        cand = os.path.join(here, name)
        if os.path.isfile(cand):
            return cand
    sys.exit("Could not find the analysis script next to watch_ra.py. Tried: "
             + ", ".join(_ANALYSIS_CANDIDATES)
             + ".\nPass its path explicitly with --script PATH.")


# ----------------------------------------------------------------------------
# File helpers
# ----------------------------------------------------------------------------
def list_abfs(folder: str) -> set[str]:
    try:
        return {os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(".abf")}
    except FileNotFoundError:
        sys.exit(f"Folder does not exist: {folder}")


def wait_until_stable(path: str, interval: float, needed: int,
                      timeout: float) -> int | None:
    """
    Block until `path`'s size is unchanged for `needed` consecutive reads.

    Returns the final size in bytes, or None if the file vanished or the
    timeout elapsed while it was still growing.
    """
    last = -1
    stable = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sz = os.path.getsize(path)
        except OSError:
            return None                      # file disappeared / not readable yet
        if sz == last and sz > 0:
            stable += 1
            if stable >= needed:
                return sz
        else:
            stable = 0
            last = sz
        time.sleep(interval)
    return None                              # still changing when time ran out


def is_readable(path: str) -> bool:
    """True if the file can be opened for reading (not locked by the amplifier SW)."""
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


# ----------------------------------------------------------------------------
# Running the analysis
# ----------------------------------------------------------------------------
def run_analysis(script: str, abf: str, passthrough: list[str],
                 dry_run: bool) -> None:
    cmd = [sys.executable, script, abf] + passthrough
    printable = " ".join(f'"{a}"' if " " in a else a for a in cmd)
    if dry_run:
        log(f"[dry-run] would run: {printable}")
        return
    log(f"running: {printable}")
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:                 # noqa: BLE001 - report and keep watching
        log(f"ERROR launching analysis: {exc}")
    else:
        log("analysis finished.")


# ----------------------------------------------------------------------------
# Main watch loop
# ----------------------------------------------------------------------------
def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", default=".",
                   help="folder to watch (default: current directory)")
    p.add_argument("--script", default=None,
                   help="path to the membrane-test script (default: auto-detect "
                        "next to this file)")
    p.add_argument("--size-kb", type=float, default=801.0,
                   help="target file size in KB (Windows-Explorer KB = 1024 B; "
                        "default 801)")
    p.add_argument("--tol-kb", type=float, default=40.0,
                   help="+/- tolerance around the target, in KB (default 40)")
    p.add_argument("--any-size", action="store_true",
                   help="run on every new .abf regardless of size")
    p.add_argument("--interval", type=float, default=2.0,
                   help="seconds between folder scans / size checks (default 2)")
    p.add_argument("--stable-checks", type=int, default=3,
                   help="consecutive equal-size reads before a file is judged "
                        "finished (default 3)")
    p.add_argument("--stable-timeout", type=float, default=120.0,
                   help="give up waiting for a file to stop growing after this "
                        "many seconds (default 120)")
    p.add_argument("--process-existing", action="store_true",
                   help="also process matching files already present at startup")
    p.add_argument("--once", action="store_true",
                   help="process the next matching file, then exit")
    p.add_argument("--dry-run", action="store_true",
                   help="log what would run, but do not launch the analysis")
    p.add_argument("--method", choices=tuple(_METHOD_FORMULA), default="exact",
                   help="access-resistance method for the analysis script: "
                        "'exact' (Ra = tau/Cm_charge, default) or 'quadratic' "
                        "(the published Newton-Raphson root). Always forwarded "
                        "explicitly, so the log records which one ran")
    # everything else is forwarded verbatim to the analysis script
    args, passthrough = p.parse_known_args(argv)

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        sys.exit(f"Not a folder: {folder}")
    script = find_analysis_script(args.script)

    target_b = args.size_kb * 1024.0
    tol_b = args.tol_kb * 1024.0

    def size_matches(sz: int) -> bool:
        return args.any_size or (abs(sz - target_b) <= tol_b)

    # Prepended, not appended: an explicit --method the user also typed would
    # have been captured by our own parser, so this cannot double up, and
    # putting it first keeps it visible at the head of every logged command.
    analysis_args = ["--method", args.method] + passthrough

    log(f"watching folder : {folder}")
    log(f"analysis script : {script}")
    log(f"Ra method       : {args.method} ({_METHOD_FORMULA[args.method]})")
    if args.any_size:
        log("size filter     : OFF (any .abf)")
    else:
        log(f"size filter     : {args.size_kb:.0f} +/- {args.tol_kb:.0f} KB "
            f"({target_b/1024:.0f} KB target)")
    if passthrough:
        log(f"forwarding args : {' '.join(passthrough)}")
    log("Ctrl+C to stop.")

    handled: set[str] = set()
    if not args.process_existing:
        handled |= list_abfs(folder)        # ignore whatever is already here
        if handled:
            log(f"ignoring {len(handled)} pre-existing .abf file(s); "
                f"waiting for new ones.")

    try:
        while True:
            for path in sorted(list_abfs(folder) - handled):
                name = os.path.basename(path)
                log(f"new file detected: {name} -- waiting for it to finish "
                    f"writing...")
                final = wait_until_stable(path, args.interval,
                                          args.stable_checks, args.stable_timeout)
                handled.add(path)            # decided once; don't re-evaluate
                if final is None:
                    log(f"  {name}: never stabilised (still growing or removed); "
                        f"skipping.")
                    continue
                kb = final / 1024.0
                if not size_matches(final):
                    log(f"  {name}: {kb:.1f} KB is outside "
                        f"{args.size_kb:.0f}+/-{args.tol_kb:.0f} KB -- not a "
                        f"membrane test; skipping.")
                    continue
                if not is_readable(path):
                    log(f"  {name}: {kb:.1f} KB matches, but the file is locked "
                        f"(still open in the amplifier software?); skipping.")
                    continue
                log(f"  {name}: {kb:.1f} KB -- match. Launching analysis.")
                run_analysis(script, path, analysis_args, args.dry_run)
                if args.once:
                    log("--once given; exiting.")
                    return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log("stopped.")


if __name__ == "__main__":
    main()
