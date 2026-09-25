#!/usr/bin/env python3
"""
batch_cell_analysis.py
======================

Batch-analyse a whole experiment: one folder per cell, one summary row per
cell.

Each cell lives in its own subfolder named

    YYMMDD_<slice>_<cell>_<RMP>

with optional ``s``/``c`` prefixes on the numbers and an optional trailing
label, so all of these parse:

    260817_s1_c1_-67        260901_3_7_-68.5        260901_3_7_-68.5_wt

Folders are found recursively under the chosen root, so a date-per-folder
layout works too.

Inside a cell folder each recording is identified from its .abf header --
the Clampex protocol name and the recorded units -- and measured according
to what it is:

  * the voltage-clamp membrane tests (protocol ``DW_Ra``) that BRACKET those
    recordings give access resistance, membrane resistance, capacitance and
    tau, via ``calculate_Ra.py``, once before and once after. The rig runs an
    Ra either side of each measurement, so a cell normally has two; when the
    step recordings are run back to back the same pair brackets both of them.
    A cell with only one Ra has it reported on whichever side it was actually
    recorded, rather than being assumed to be the initial one, and
    ``mt_Ra_change_pct`` says how far Ra moved across the recordings;

  * the current-clamp step file with the widest injected-current range
    (``DW_IC``, ``DW_IC_IV``, ``IC_IV``, ... -- matched by having a mV
    channel and a varying step, not by name) gives the action-potential
    measurements via ``autorun_AP_analysis.py``: rheobase, F-I curve, AP
    threshold / amplitude / half-width / AHP;

  * its hyperpolarizing sweeps -- or a dedicated hyperpolarizing-step file
    if the folder has one -- give sag at a fixed membrane potential and
    input resistance measured where Ih cannot distort it (below).

Any of these may be missing. Whatever is present is analysed, and the
summary row says what was found, what was not, and what failed.

Sag-free input resistance
-------------------------
Ih (the sag current) makes the steady-state voltage response sublinear, so a
V-I slope taken over large hyperpolarizing steps underestimates Rin. Only
sweeps that stay out of the Ih-activation range are therefore fitted:

  * the peak (most hyperpolarized) voltage must stay above ``--rin-v-floor``
    (default -80 mV), and
  * the sweep's sag ratio must be at or below ``--rin-max-sag`` (default
    0.05, i.e. 5% relaxation),

with the 0 pA sweep always kept as the (0 pA, V_rest) anchor. Rin is the
slope of steady-state V vs injected I across those sweeps. The
peak-deflection slope over the same sweeps is reported alongside as a
cross-check: the peak is reached long before Ih develops, so the two
agreeing is evidence that the steady-state fit really is Ih-free. If fewer
than two sweeps qualify, the smallest hyperpolarizing steps are fitted
instead and the row is flagged in ``Rin_note``.

Sag is quantified per sweep as ipfx defines it,

    sag ratio = (V_peak - V_steady) / (V_peak - V_baseline)

and reported at one fixed membrane potential (-85 mV by default) by linear
interpolation between the two sweeps whose peaks bracket it -- cells differ
in rest and in Rin, so the same current step lands at different voltages,
and quoting sag at a fixed potential is what makes it comparable between
cells. A cell whose sweeps never reach the target is reported at its nearest
peak voltage with ``sag_bracketed`` = False.

Output
------
Per cell, written next to the recordings under the same names the
single-file tools use:

    <ra>_membrane_test.txt         membrane-test report
    <ra>_membrane_test_qc.png      membrane-test QC figure
    <steps>_subthreshold.csv       per-sweep sag / V-I table
    <steps>_subthreshold_qc.png    sag + Rin QC figure
    <steps>_AP_spikes.csv          one row per detected AP
    <steps>_AP_sweeps.csv          one row per sweep
    <steps>_AP_qc.png, <steps>_AP_rheobase.png

and one row per cell at the root:

    batch_cell_summary.csv

Animal details
--------------
Run with no folder argument and a dialog collects both the experiment folder
and the animal it came from -- date of birth, sex, genotype, strain -- and
says how many cell folders it found before anything is analysed. Those
details are written to every row of the summary, and the date of birth gives
each cell an ``age_days`` (its folder's date minus the DOB), so cells from
different litters stay distinguishable once the summaries are pooled. The
same fields are available as flags (``--dob``, ``--sex``, ``--genotype``,
``--strain``) for scripted runs, and ``--no-gui`` skips the dialog.

When the run finishes, a QC window opens (``batch_qc_window.py``): the F-I
curves of every cell overlaid, the cell-health measurements side by side
against QC guides -- Ra and Cm shown before and after the recordings, joined
by a line, so a cell that ran down is obvious -- and each cell's rheobase and
max-firing traces.
``--no-window`` skips it; ``--view`` reopens it later for a folder that has
already been analysed, without re-running anything.

Usage
-----
    python batch_cell_analysis.py                     # dialog: folder + animal
    python batch_cell_analysis.py DATA_ROOT
    python batch_cell_analysis.py DATA_ROOT --dob 2026-06-01 --sex male \
        --genotype "Sst-Cre;Ai14" --strain C57BL/6J
    python batch_cell_analysis.py DATA_ROOT --no-plot
    python batch_cell_analysis.py DATA_ROOT --sag-target -85 --rin-v-floor -78
    python batch_cell_analysis.py DATA_ROOT --view    # just show the QC window
    python batch_cell_analysis.py --selftest          # synthetic validation

Requires: numpy, pandas, matplotlib, pyabf, ipfx, scipy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import autorun_AP_analysis as ap
import calculate_Ra as mt
# The per-sweep stimulus window comes from the ABF epoch table; reuse the AP
# script's detector rather than re-deriving it, so the subthreshold and
# suprathreshold measurements are windowed by identical rules.
from autorun_AP_analysis import _stim_window as stim_window


# ---------------------------------------------------------------------------
# Cell folders
# ---------------------------------------------------------------------------
FOLDER_RE = re.compile(
    r"^(?P<date>\d{6})_s?(?P<slice>\d+)_c?(?P<cell>\d+)_"
    r"(?P<rmp>[-+]?\d+(?:\.\d+)?)\s*(?:mv)?(?:_(?P<label>.*))?$",
    re.IGNORECASE,
)


def parse_cell_folder(name: str) -> Optional[dict]:
    """Metadata encoded in a cell folder's name, or None if it doesn't match."""
    m = FOLDER_RE.match(name.strip())
    if not m:
        return None
    return {
        "date": m.group("date"),
        "slice": int(m.group("slice")),
        "cell": int(m.group("cell")),
        "rmp_folder_mV": float(m.group("rmp")),
        "label": (m.group("label") or "").strip(),
    }


def age_days(dob: str, folder_date: str) -> Optional[int]:
    """Age in days at recording: a YYMMDD folder date minus an ISO date of birth.

    None if either date is missing or unparseable -- a blank age column is
    better than a fabricated one.
    """
    if not dob or not folder_date:
        return None
    try:
        born = dt.datetime.strptime(str(dob), "%Y-%m-%d").date()
        recorded = dt.datetime.strptime(str(folder_date), "%y%m%d").date()
    except (ValueError, TypeError):
        return None
    return (recorded - born).days


def find_cell_folders(root: Path, include_unparsed: bool = False) -> list[tuple[Path, dict]]:
    """(folder, metadata) for every cell folder under `root`, sorted by name.

    A folder qualifies if its name matches the YYMMDD_slice_cell_RMP pattern
    and it holds at least one .abf. With `include_unparsed`, any folder
    holding .abf files is analysed with blank metadata -- which rescues a
    folder whose name got typed differently, at the cost of also sweeping up
    folders that were never meant to be cells.
    """
    found: list[tuple[Path, dict]] = []
    for folder in [root] + sorted(p for p in root.rglob("*") if p.is_dir()):
        try:
            has_abf = any(p.suffix.lower() == ".abf" for p in folder.iterdir() if p.is_file())
        except OSError:
            continue
        if not has_abf:
            continue
        meta = parse_cell_folder(folder.name)
        if meta is None:
            if not include_unparsed:
                continue
            meta = {"date": "", "slice": "", "cell": "",
                    "rmp_folder_mV": float("nan"), "label": "", "unparsed_name": True}
        found.append((folder, meta))
    found.sort(key=lambda fm: str(fm[0]).lower())
    return found


# ---------------------------------------------------------------------------
# Reading ABF headers / sweeps
# ---------------------------------------------------------------------------
def read_header(path: Path) -> dict:
    """Protocol name, units and recording time from an .abf header.

    Deliberately not imported from cell_metadata_entry.read_abf_protocol:
    that module pulls in PyQt5 at import time, and this is a headless batch
    job. Header-only read -- the sweep data is loaded later, and only for the
    files that turn out to matter.
    """
    info = {"path": path, "protocol": "", "units": [], "when": None, "sweep_count": 0}
    try:
        import pyabf
        abf = pyabf.ABF(str(path), loadData=False)
        info["protocol"] = abf.protocol or ""
        info["units"] = [abf.adcUnits[c].strip() for c in abf.channelList]
        info["when"] = abf.abfDateTime
        info["sweep_count"] = int(abf.sweepCount)
    except Exception as exc:
        info["error"] = str(exc)
    if info["when"] is None:
        try:
            info["when"] = dt.datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            pass
    return info


def scan_folder(folder: Path) -> list[dict]:
    """Every .abf in the folder with its header facts, oldest recording first."""
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".abf"]
    scanned = [read_header(p) for p in sorted(files, key=lambda p: p.name.lower())]
    scanned.sort(key=lambda c: (c["when"] or dt.datetime.min, c["path"].name))
    return scanned


def _has_units(info: dict, units: tuple[str, ...]) -> bool:
    return any(u.lower() in units for u in info["units"])


def _channel_with_units(abf, units: tuple[str, ...]) -> Optional[int]:
    for ch in abf.channelList:
        if abf.adcUnits[ch].strip().lower() in units:
            return ch
    return None


def load_cc_sweeps(path: Path, channel: Optional[int] = None) -> tuple[list[dict], dict]:
    """Every sweep of a current-clamp ABF, with its stimulus window.

    Each sweep is a dict of t (s), v (mV), i (pA command), start/end (s) and
    stim_pA -- the step amplitude relative to that sweep's baseline epoch, so
    a protocol run from a holding current still reports the injected step.
    """
    import pyabf
    abf = pyabf.ABF(str(path))
    if channel is None:
        channel = _channel_with_units(abf, ("mv",))
    if channel is None:
        raise ValueError(f"no current-clamp (mV) channel in {path.name} "
                         f"(units: {[abf.adcUnits[c] for c in abf.channelList]})")

    sweeps = []
    for sw in range(abf.sweepCount):
        start, end, stim = stim_window(abf, sw, channel)
        abf.setSweep(sw, channel=channel)
        sweeps.append({
            "sweep": sw,
            "t": np.asarray(abf.sweepX, float),
            "v": np.asarray(abf.sweepY, float),
            "i": np.asarray(abf.sweepC, float),
            "start": float(start), "end": float(end), "stim_pA": float(stim),
        })
    meta = {"path": str(path), "channel": channel, "protocol": abf.protocol,
            "sweep_count": abf.sweepCount, "sample_rate_hz": abf.dataRate}
    return sweeps, meta


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------
def membrane_test_candidates(scanned: list[dict], protocol: str = "DW_Ra",
                             ) -> tuple[list[dict], list[str]]:
    """Every recording in the folder that could be a membrane test, oldest first.

    Matched on protocol name, falling back to voltage-clamp files if the name
    never appears. Voltage clamp is judged on the PRIMARY channel's units, not
    on the file having a pA channel anywhere: a current-clamp step file records
    the current monitor alongside the membrane potential, so "has a pA channel"
    is true of every recording the rig makes. The primary channel is the
    amplifier's telegraphed output, so it is pA exactly when the cell was
    voltage-clamped. Single-sweep files are excluded as well -- a gap-free
    sEPSC recording is voltage clamp too, and has no test pulse to measure.
    """
    want = protocol.strip().lower()
    named = [c for c in scanned if want and want in (c["protocol"] or "").lower()]
    if named:
        return named, []
    vc = [c for c in scanned
          if (c["units"] or [""])[0].strip().lower() in ("pa", "na", "a")
          and int(c.get("sweep_count") or 0) > 1]
    if vc:
        return vc, [f"no {protocol} recording; used the voltage-clamp file(s) "
                    f"{', '.join(c['path'].name for c in vc)} as membrane tests"]
    return [], []


def pick_membrane_tests(scanned: list[dict], targets: list[Optional[Path]],
                        protocol: str = "DW_Ra",
                        ) -> tuple[Optional[Path], Optional[Path], list[str]]:
    """(before, after, notes): the membrane tests bracketing `targets`.

    The rig runs an Ra either side of each measurement, so which one is the
    "before" is a fact about WHEN the step recordings were made, not about
    which Ra file happens to come first in the folder -- a cell whose only Ra
    was recorded after the F-I curve would otherwise have a run-down Ra
    reported as its initial one, which is the opposite of the truth.

    `targets` are the recordings being bracketed (the step file, and the
    hyperpolarizing file when that is a separate recording). Run back to back,
    as they usually are, they share one pair: the Ra before the first and the
    Ra after the last. Either side may be missing, and is then simply None.
    """
    cands, notes = membrane_test_candidates(scanned, protocol)
    if not cands:
        return None, None, notes

    times = [c["when"] for c in scanned
             if c["path"] in {t for t in targets if t is not None} and c["when"]]
    if not times:
        # Nothing datable to bracket (a header with no timestamp, or no step
        # recording at all): fall back to the folder's first and last.
        first, last = cands[0]["path"], cands[-1]["path"]
        return first, (last if last != first else None), notes

    t0, t1 = min(times), max(times)
    before = [c for c in cands if c["when"] and c["when"] <= t0]
    after = [c for c in cands if c["when"] and c["when"] >= t1]
    pre = before[-1]["path"] if before else None            # the nearest one before
    post = after[0]["path"] if after else None              # the nearest one after
    if post is not None and post == pre:
        post = None
    return pre, post, notes


def pick_step_files(scanned: list[dict], protocol: str = "",
                    hyperpol_protocol: str = "hyperpol") -> tuple[Optional[Path], Optional[Path], list[str]]:
    """(steps file, hyperpolarizing file, notes).

    Current-clamp step files are recognised by having a mV channel and a
    stimulus that actually varies between sweeps -- not by protocol name,
    because the same measurements live under DW_IC, DW_IC_IV and IC_IV
    depending on when the protocol was written, and a name match would also
    accept a fixed-amplitude test (e.g. 20 identical +10 pA sweeps) that
    carries no F-I curve at all. Of the real step files, the one spanning the
    widest current range is used: it is the one that reaches both rheobase
    and, if the protocol goes there, the hyperpolarizing range.

    A dedicated hyperpolarizing-steps file, if the folder has one, is
    returned separately and takes precedence for the sag/Rin measurements.
    """
    notes: list[str] = []
    cc = [c for c in scanned if _has_units(c, ("mv",))]
    if protocol.strip():
        want = protocol.strip().lower()
        cc = [c for c in cc if want in (c["protocol"] or "").lower()]

    ranges: list[tuple[float, float, dict]] = []
    for info in cc:
        try:
            sweeps, _meta = load_cc_sweeps(info["path"])
        except Exception as exc:
            notes.append(f"could not read {info['path'].name}: {exc}")
            continue
        stims = np.array([s["stim_pA"] for s in sweeps], float)
        if stims.size < 2 or not np.isfinite(stims).any() or np.ptp(stims[np.isfinite(stims)]) == 0:
            notes.append(f"{info['path'].name} ({info['protocol']}) has no varying "
                         f"current step; not a step protocol")
            continue
        ranges.append((float(np.ptp(stims)), float(np.min(stims)), info))

    if not ranges:
        return None, None, notes

    ranges.sort(key=lambda r: r[0])
    steps = ranges[-1][2]["path"]

    want_hyp = hyperpol_protocol.strip().lower()
    hyper = [info for _r, _lo, info in ranges
             if want_hyp and want_hyp in (info["protocol"] or "").lower()]
    if hyper:
        return steps, hyper[-1]["path"], notes
    # No dedicated file: the step protocol's own negative sweeps carry the
    # sag, if it goes there at all.
    has_negative = any(lo < 0 for _r, lo, info in ranges if info["path"] == steps)
    return steps, (steps if has_negative else None), notes


# ---------------------------------------------------------------------------
# Subthreshold measurements: sag and (sag-free) input resistance
# ---------------------------------------------------------------------------
def _mean_over(t: np.ndarray, v: np.ndarray, t0: float, t1: float) -> float:
    i0 = int(np.searchsorted(t, t0, side="left"))
    i1 = int(np.searchsorted(t, t1, side="right"))
    i0 = max(0, min(i0, len(t) - 1))
    i1 = max(i0 + 1, min(i1, len(t)))
    return float(np.mean(v[i0:i1]))


def subthreshold_sweep_table(
    sweeps: list[dict],
    baseline_interval: float = 0.1,
    ss_interval: Optional[float] = None,
    peak_width: float = 0.005,
    spike_thresh_mV: float = -20.0,
    min_deflection_mV: float = 0.5,
) -> pd.DataFrame:
    """Per-sweep baseline / peak / steady-state voltage and sag ratio.

    sag ratio = (V_peak - V_steady) / (V_peak - V_baseline), the definition
    ipfx.subthresh_features.sag uses, left as NaN when the deflection is
    smaller than `min_deflection_mV` -- below that the ratio is dominated by
    baseline noise rather than by the cell.
    """
    rows = []
    for s in sweeps:
        t, v = s["t"], s["v"]
        start, end = s["start"], s["end"]
        dur = max(end - start, 1e-6)
        bi = min(baseline_interval, max(start - float(t[0]), 1e-3))
        ssi = ss_interval if ss_interval else min(0.1, max(0.25 * dur, 1e-3))

        v_base = _mean_over(t, v, start - bi, start)
        i0 = int(np.searchsorted(t, start, side="left"))
        i1 = max(i0 + 1, int(np.searchsorted(t, end, side="right")))
        win = v[i0:i1]
        pk = i0 + int(np.argmin(win))
        v_peak = _mean_over(t, v, t[pk] - peak_width / 2.0, t[pk] + peak_width / 2.0)
        v_ss = _mean_over(t, v, end - ssi, end)

        denom = v_peak - v_base
        sag_ratio = (float((v_peak - v_ss) / denom)
                     if abs(denom) >= min_deflection_mV else float("nan"))

        rows.append({
            "sweep": s["sweep"],
            "stim_pA": s["stim_pA"],
            "stim_start_s": start,
            "stim_end_s": end,
            "v_baseline_mV": v_base,
            "v_peak_mV": v_peak,
            "v_steady_mV": v_ss,
            "peak_t_s": float(t[pk]),
            "dV_peak_mV": v_peak - v_base,
            "dV_steady_mV": v_ss - v_base,
            "sag_ratio": sag_ratio,
            "sag_amplitude_mV": v_ss - v_peak,
            "spiked": bool(np.max(win) > spike_thresh_mV),
        })
    return pd.DataFrame(rows)


def sag_at_potential(table: pd.DataFrame, target_v: float = -85.0) -> dict:
    """Sag interpolated to one fixed peak membrane potential."""
    out = {"sag_target_mV": target_v, "sag_ratio": np.nan, "sag_amplitude_mV": np.nan,
           "sag_v_peak_mV": np.nan, "sag_stim_pA": np.nan, "sag_bracketed": False,
           "sag_n_sweeps": 0, "sag_note": ""}
    ok = table[(table["stim_pA"] < 0) & (~table["spiked"])
               & table["sag_ratio"].notna() & table["v_peak_mV"].notna()]
    if ok.empty:
        out["sag_note"] = "no hyperpolarizing sweeps with a measurable deflection"
        return out

    ok = ok.sort_values("v_peak_mV")
    x = ok["v_peak_mV"].to_numpy(float)
    out["sag_n_sweeps"] = int(len(ok))

    if x.min() <= target_v <= x.max():
        out["sag_bracketed"] = True
        out["sag_v_peak_mV"] = float(target_v)
        for col, key in (("sag_ratio", "sag_ratio"),
                         ("sag_amplitude_mV", "sag_amplitude_mV"),
                         ("stim_pA", "sag_stim_pA")):
            out[key] = float(np.interp(target_v, x, ok[col].to_numpy(float)))
    else:
        near = ok.iloc[int(np.argmin(np.abs(x - target_v)))]
        out.update({"sag_v_peak_mV": float(near["v_peak_mV"]),
                    "sag_ratio": float(near["sag_ratio"]),
                    "sag_amplitude_mV": float(near["sag_amplitude_mV"]),
                    "sag_stim_pA": float(near["stim_pA"]),
                    "sag_note": f"no sweep peaked at {target_v:g} mV; reported at the "
                                f"nearest peak ({float(near['v_peak_mV']):.1f} mV)"})
    return out


def _fit_slope(i_pA: np.ndarray, v_mV: np.ndarray) -> tuple[float, float]:
    """(slope in mV/pA, r^2) of a straight line through the V-I points."""
    if len(i_pA) < 2 or np.ptp(i_pA) == 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(i_pA, v_mV, 1)
    pred = slope * i_pA + intercept
    ss_res = float(np.sum((v_mV - pred) ** 2))
    ss_tot = float(np.sum((v_mV - v_mV.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)


def input_resistance(table: pd.DataFrame, v_floor: float = -80.0,
                     max_sag: float = 0.05) -> dict:
    """Input resistance from the sweeps where Ih cannot distort it.

    Included: non-spiking sweeps at or below 0 pA whose peak voltage stays
    above `v_floor` and whose sag ratio is <= `max_sag`. The 0 pA sweep is
    always kept -- it is the (0 pA, V_rest) anchor of the fit, and has no sag
    by construction.

    Rin is the steady-state V-I slope over those sweeps (1 mV/pA = 1000
    MOhm). `Rin_peak_MOhm` is the same fit on the peak deflection, which is
    reached before Ih has had time to develop: close agreement between the
    two is evidence that the steady-state fit really is Ih-free.
    """
    out = {"Rin_MOhm": np.nan, "Rin_r2": np.nan, "Rin_peak_MOhm": np.nan,
           "Rin_n_sweeps": 0, "Rin_I_min_pA": np.nan, "Rin_I_max_pA": np.nan,
           "Rin_V_min_mV": np.nan, "Rin_V_max_mV": np.nan,
           "Rin_sweeps_used": "", "Rin_note": ""}

    usable = table[(table["stim_pA"] <= 0) & (~table["spiked"])
                   & table["v_steady_mV"].notna()]
    if usable.empty or (usable["stim_pA"] < 0).sum() == 0:
        out["Rin_note"] = "no subthreshold hyperpolarizing sweeps"
        return out

    sag_ok = ~(usable["sag_ratio"] > max_sag)   # NaN (deflection too small) counts as OK
    keep = usable[(usable["stim_pA"] == 0)
                  | ((usable["v_peak_mV"] >= v_floor) & sag_ok)]

    if keep["stim_pA"].nunique() < 2:
        # Rest already below the floor, or Ih active at the smallest step:
        # fit the smallest hyperpolarizing steps so the cell still gets a
        # number, and say in the row that it may be Ih-contaminated.
        hyper = usable[usable["stim_pA"] < 0]
        hyper = hyper.reindex(hyper["stim_pA"].abs().sort_values().index)
        keep = pd.concat([usable[usable["stim_pA"] == 0], hyper.head(2)])
        out["Rin_note"] = (f"fewer than 2 sweeps stayed above {v_floor:g} mV with sag "
                           f"<= {max_sag:g}; fitted the smallest steps instead "
                           f"(Ih contamination possible)")
        if keep["stim_pA"].nunique() < 2:
            out["Rin_note"] = "too few usable sweeps to fit a V-I slope"
            return out

    i = keep["stim_pA"].to_numpy(float)
    slope_ss, r2 = _fit_slope(i, keep["v_steady_mV"].to_numpy(float))
    slope_pk, _ = _fit_slope(i, keep["v_peak_mV"].to_numpy(float))
    out.update({
        "Rin_MOhm": slope_ss * 1e3,          # mV/pA -> MOhm
        "Rin_r2": r2,
        "Rin_peak_MOhm": slope_pk * 1e3,
        "Rin_n_sweeps": int(len(keep)),
        "Rin_I_min_pA": float(i.min()), "Rin_I_max_pA": float(i.max()),
        "Rin_V_min_mV": float(keep["v_steady_mV"].min()),
        "Rin_V_max_mV": float(keep["v_steady_mV"].max()),
        "Rin_sweeps_used": ",".join(str(int(s)) for s in keep["sweep"]),
    })
    return out


def subthreshold_summary(table: pd.DataFrame, target_v: float = -85.0,
                         v_floor: float = -80.0, max_sag: float = 0.05) -> dict:
    """Everything the hyperpolarizing sweeps contribute to the summary row."""
    out: dict = {}
    rest = table[table["stim_pA"] == 0]["v_baseline_mV"]
    if rest.empty:
        rest = table["v_baseline_mV"]
    out["Vrest_measured_mV"] = float(rest.dropna().mean()) if rest.notna().any() else np.nan
    out.update(sag_at_potential(table, target_v=target_v))
    out.update(input_resistance(table, v_floor=v_floor, max_sag=max_sag))
    return out


# ---------------------------------------------------------------------------
# Subthreshold QC figure
# ---------------------------------------------------------------------------
def plot_subthreshold_qc(sweeps: list[dict], table: pd.DataFrame, summary: dict,
                         out_path: str, title: str = "") -> str:
    """Left: the subthreshold sweeps with every measurement point marked.
    Right: the V-I plot, showing which sweeps the Rin fit was allowed to use
    and which were excluded for sitting in the Ih range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    used = {int(s) for s in str(summary.get("Rin_sweeps_used", "")).split(",") if s.strip()}
    sub = table[(table["stim_pA"] <= 0) & (~table["spiked"])]
    by_sweep = {int(r["sweep"]): r for _k, r in sub.iterrows()}

    fig, (ax, axi) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(title or "subthreshold QC", fontsize=11)

    for s in sweeps:
        row = by_sweep.get(int(s["sweep"]))
        if row is None:
            continue
        in_fit = int(s["sweep"]) in used
        ax.plot(s["t"], s["v"], lw=0.9 if in_fit else 0.5,
                color="#1f77b4" if in_fit else "#c8c8c8", zorder=3 if in_fit else 1)
        ax.plot(row["peak_t_s"], row["v_peak_mV"], "v", ms=5,
                color="crimson" if in_fit else "#dddddd", zorder=4)
        ax.hlines(row["v_steady_mV"], row["stim_end_s"] - 0.1, row["stim_end_s"],
                  color="#ff7f0e" if in_fit else "#e2e2e2", lw=1.5, zorder=4)

    target = summary.get("sag_target_mV", -85.0)
    ax.axhline(target, color="purple", ls="--", lw=1.0, label=f"sag target {target:g} mV")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Vm (mV)")
    ax.set_title("blue = fitted for Rin, grey = excluded (Ih range)\n"
                 "v = peak deflection, bar = steady state", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")

    fit = sub[sub["sweep"].isin(used)]
    rest = sub[~sub["sweep"].isin(used)]
    axi.plot(rest["stim_pA"], rest["v_steady_mV"], "o", color="#c8c8c8", ms=6,
             label="steady state (excluded)")
    axi.plot(fit["stim_pA"], fit["v_steady_mV"], "o", color="#1f77b4", ms=7,
             label="steady state (fitted)")
    axi.plot(sub["stim_pA"], sub["v_peak_mV"], "^", color="crimson", ms=5, alpha=0.6,
             label="peak deflection")
    if np.isfinite(summary.get("Rin_MOhm", np.nan)) and not fit.empty:
        i = fit["stim_pA"].to_numpy(float)
        slope = summary["Rin_MOhm"] / 1e3
        intercept = float(np.mean(fit["v_steady_mV"].to_numpy(float) - slope * i))
        xs = np.linspace(min(float(i.min()), 0.0), max(float(i.max()), 0.0), 10)
        axi.plot(xs, slope * xs + intercept, "-", color="#1f77b4", lw=1.4,
                 label=f"Rin = {summary['Rin_MOhm']:.0f} M$\\Omega$ "
                       f"(r$^2$ = {summary.get('Rin_r2', float('nan')):.3f})")
    axi.set_xlabel("injected current (pA)")
    axi.set_ylabel("Vm (mV)")
    axi.set_title(f"sag at {summary.get('sag_v_peak_mV', float('nan')):.1f} mV: "
                  f"ratio {summary.get('sag_ratio', float('nan')):.3f}, "
                  f"{summary.get('sag_amplitude_mV', float('nan')):.2f} mV"
                  + ("" if summary.get("sag_bracketed") else "   [target not bracketed]"),
                  fontsize=9)
    axi.legend(fontsize=7.5, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Per-cell stages
# ---------------------------------------------------------------------------
def analyze_membrane_test(path: Path, opts) -> tuple[dict, list[str]]:
    """Ra / Rm / Cm / tau from the initial voltage-clamp membrane test."""
    import pyabf
    abf = pyabf.ABF(str(path), loadData=False)
    channel = _channel_with_units(abf, ("pa", "na", "a"))
    if channel is None:
        raise ValueError(f"no voltage-clamp (pA) channel in {path.name} "
                         f"(units: {[abf.adcUnits[c] for c in abf.channelList]})")

    res = mt.membrane_test_from_abf(str(path), channel=channel)
    s = res.summary()
    row = {
        "mt_Ra_MOhm": s["Ra"]["mean"], "mt_Ra_sd": s["Ra"]["sd"],
        "mt_Rm_MOhm": s["Rm"]["mean"], "mt_Rt_MOhm": s["Rt"]["mean"],
        "mt_Cm_pF": s["Cm"]["mean"], "mt_Cm_sd": s["Cm"]["sd"],
        "mt_tau_ms": s["Tau"]["mean"], "mt_Iss_pA": s["Iss"]["mean"],
        "mt_n_transients": s["Ra"]["n"],
    }
    written = []
    txt = mt.default_report_path(str(path))
    mt.write_report(res, txt, settings={"method": "exact", "t1": 0.20,
                                        "fit": "10-80% of peak", "min_r2": 0.80})
    written.append(str(txt))
    if not opts.no_plot and res.first_sweep:
        fs = res.first_sweep
        png = path.with_name(f"{path.stem}_membrane_test_qc.png")
        mt.plot_membrane_test(fs["time_s"], fs["current_A"], fs["command_V"],
                              fs["result"], out_path=str(png),
                              title=f"{path.name}  -  membrane test (sweep 0)")
        written.append(str(png))
    return row, written


def analyze_membrane_test_post(path: Path, opts) -> tuple[dict, list[str]]:
    """The closing membrane test, under ``mt_post_*`` keys.

    Measured exactly like the opening one -- the pair is only comparable if
    nothing differs but the recording."""
    row, written = analyze_membrane_test(path, opts)
    return {f"mt_post_{k[len('mt_'):]}": v for k, v in row.items()}, written


def analyze_subthreshold(path: Path, opts) -> tuple[dict, list[str]]:
    """Sag and sag-free input resistance from the hyperpolarizing sweeps."""
    sweeps, _meta = load_cc_sweeps(path)
    table = subthreshold_sweep_table(sweeps)
    summary = subthreshold_summary(table, target_v=opts.sag_target,
                                   v_floor=opts.rin_v_floor, max_sag=opts.rin_max_sag)
    written = []
    csv = path.with_name(f"{path.stem}_subthreshold.csv")
    table.to_csv(csv, index=False)
    written.append(str(csv))
    if not opts.no_plot:
        png = path.with_name(f"{path.stem}_subthreshold_qc.png")
        plot_subthreshold_qc(sweeps, table, summary, str(png),
                             title=f"{path.name}  -  sag / input resistance")
        written.append(str(png))
    return summary, written


def holding_current(path: Path, max_sweeps: int = 10) -> dict:
    """The bias (holding) current the amplifier was applying, in pA.

    A bias current dialed in on the amplifier is added downstream of the
    Clampex command, so it never appears in the command waveform (sweepC) or
    in stim_amp_pA. It can only be seen on the recorded current-monitor
    channel -- the MultiClamp's secondary output, telegraphed as a pA channel.
    Measured over the pre-step baseline, where the protocol commands nothing,
    and averaged over the first `max_sweeps` sweeps (the holding current does
    not change between them, so this is just noise averaging).

    The monitor carries its own offset, so a couple of pA is indistinguishable
    from zero: this answers "was the cell held with a bias current", not "what
    current flowed" to the last picoamp.
    """
    import pyabf
    abf = pyabf.ABF(str(path))
    v_ch = _channel_with_units(abf, ("mv",))
    i_ch = _channel_with_units(abf, ("pa", "na", "a"))
    if v_ch is None or i_ch is None:
        return {}

    scale = {"pa": 1.0, "na": 1e3, "a": 1e12}
    vals = []
    for sw in range(min(abf.sweepCount, max_sweeps)):
        start, _end, _amp = stim_window(abf, sw, v_ch)
        abf.setSweep(sw, channel=i_ch)
        t = np.asarray(abf.sweepX, float)
        if not np.isfinite(start) or start <= float(t[0]) + 1e-3:
            continue          # gap-free or no pre-step baseline to average
        i = np.asarray(abf.sweepY, float) * scale.get(
            abf.sweepUnitsY.strip().lower(), 1.0)
        vals.append(_mean_over(t, i, start - min(0.1, start - float(t[0])), start))

    if not vals:
        return {}
    return {"bias_current_pA": float(np.mean(vals)),
            "bias_current_sd_pA": float(np.std(vals))}


def analyze_action_potentials(path: Path, opts) -> tuple[dict, list[str]]:
    """Rheobase / F-I / AP waveform, via autorun_AP_analysis."""
    result = ap.analyze_abf(str(path), dv_cutoff=opts.dv_cutoff,
                            min_peak_mV=opts.min_peak)
    summary = ap.cell_summary(result)
    summary["n_spikes_total"] = int(len(result["spikes"]))
    # Measured from the same recording the F-I curve comes from, so the two
    # always describe the cell in the same state.
    try:
        summary.update(holding_current(path))
    except Exception as exc:
        print(f"      note: holding current not measured ({exc})", file=sys.stderr)
    stem = str(path.with_suffix(""))
    result["spikes"].to_csv(f"{stem}_AP_spikes.csv", index=False)
    result["sweeps"].to_csv(f"{stem}_AP_sweeps.csv", index=False)
    written = [f"{stem}_AP_spikes.csv", f"{stem}_AP_sweeps.csv"]
    if not opts.no_plot:
        for fn, suffix in ((ap.plot_ap_qc, "_AP_qc.png"),
                           (ap.plot_rheobase_detail, "_AP_rheobase.png")):
            try:
                fn(result, summary, f"{stem}{suffix}")
                written.append(f"{stem}{suffix}")
            except Exception as exc:
                print(f"      note: {suffix.lstrip('_')} not built ({exc})",
                      file=sys.stderr)
    return summary, written


def _as_float(value) -> float:
    """A summary value as a float, NaN when absent or non-numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def ra_drift(row: dict) -> dict:
    """How far access resistance moved between the two membrane tests.

    As a percentage of where it started, which is what a run-down judgement is
    actually made on: +8 MOhm means something different on a 12 MOhm seal than
    on a 30 MOhm one. Empty unless the cell has both an opening and a closing
    membrane test.
    """
    pre, post = _as_float(row.get("mt_Ra_MOhm")), _as_float(row.get("mt_post_Ra_MOhm"))
    if not (np.isfinite(pre) and np.isfinite(post)) or pre == 0:
        return {}
    return {"mt_Ra_change_MOhm": post - pre,
            "mt_Ra_change_pct": 100.0 * (post - pre) / pre}


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------
def analyze_cell(folder: Path, meta: dict, opts) -> dict:
    """One summary row. A failure in any one stage is recorded in `notes` and
    leaves the other stages' numbers intact -- one unreadable file must not
    cost a whole batch run."""
    row: dict = {"folder_name": folder.name, "folder": str(folder)}
    row.update({k: v for k, v in meta.items() if k != "unparsed_name"})
    notes: list[str] = []

    # Animal details are per run; age is per cell, since a root folder can hold
    # more than one recording date.
    animal = getattr(opts, "animal", None) or {}
    row.update(animal)
    age = age_days(animal.get("animal_dob", ""), meta.get("date", ""))
    row["age_days"] = age if age is not None else ""
    if age is not None and age < 0:
        notes.append(f"recording date precedes the date of birth by {-age} days")

    if meta.get("unparsed_name"):
        notes.append("folder name does not match YYMMDD_slice_cell_RMP")
    rmp = meta.get("rmp_folder_mV", float("nan"))
    if isinstance(rmp, float) and np.isfinite(rmp) and rmp > 0:
        notes.append("RMP in the folder name is positive; recorded as written")

    scanned = scan_folder(folder)
    for info in scanned:
        if info.get("error"):
            notes.append(f"unreadable: {info['path'].name} ({info['error']})")

    # The step files are chosen first because the membrane tests are located
    # relative to them: "before" and "after" are times, not file positions.
    steps_path, hyper_path, pick_notes = pick_step_files(
        scanned, opts.protocol_steps, opts.protocol_hyperpol)
    notes += pick_notes
    ra_pre, ra_post, ra_notes = pick_membrane_tests(
        scanned, [steps_path, hyper_path], opts.protocol_ra)
    notes += ra_notes

    row["file_membrane_test"] = ra_pre.name if ra_pre else ""
    row["file_membrane_test_post"] = ra_post.name if ra_post else ""
    row["file_steps"] = steps_path.name if steps_path else ""
    row["file_hyperpol"] = hyper_path.name if hyper_path else ""
    used = {p for p in (ra_pre, ra_post, steps_path, hyper_path) if p is not None}
    row["files_unused"] = ";".join(c["path"].name for c in scanned if c["path"] not in used)

    stages = [
        ("Ra before", ra_pre, analyze_membrane_test,
         f"no {opts.protocol_ra} (or any voltage-clamp) recording before the steps"),
        ("Ra after", ra_post, analyze_membrane_test_post,
         f"no {opts.protocol_ra} recording after the steps"),
        ("sag / Rin", hyper_path, analyze_subthreshold,
         "no hyperpolarizing sweeps in this folder"),
        ("AP analysis", steps_path, analyze_action_potentials,
         "no current-clamp step recording"),
    ]
    for label, path, fn, missing in stages:
        if path is None:
            notes.append(f"{label} skipped: {missing}")
            print(f"    {label:<15} -- skipped ({missing})")
            continue
        try:
            out, written = fn(path, opts)
            row.update(out)
            print(f"    {label:<15} {path.name}  ->  {len(written)} file(s)")
        except (Exception, SystemExit) as exc:
            notes.append(f"{label} failed on {path.name}: {exc}")
            print(f"    {label:<15} {path.name}  ->  FAILED: {exc}", file=sys.stderr)
            if opts.traceback:
                traceback.print_exc()

    row.update(ra_drift(row))
    row["notes"] = " | ".join(notes)
    return row


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
COLUMN_ORDER = [
    "folder_name", "date", "slice", "cell", "rmp_folder_mV", "label",
    # animal (one set of values per run; age is per recording date)
    "genotype", "strain", "animal_sex", "animal_dob", "age_days",
    "file_membrane_test", "file_membrane_test_post", "file_steps", "file_hyperpol",
    # membrane test (voltage clamp), before the step recordings ...
    "mt_Ra_MOhm", "mt_Ra_sd", "mt_Rm_MOhm", "mt_Rt_MOhm", "mt_Cm_pF", "mt_Cm_sd",
    "mt_tau_ms", "mt_Iss_pA", "mt_n_transients",
    # ... and after them, plus how far Ra moved in between
    "mt_post_Ra_MOhm", "mt_post_Ra_sd", "mt_post_Rm_MOhm", "mt_post_Rt_MOhm",
    "mt_post_Cm_pF", "mt_post_Cm_sd", "mt_post_tau_ms", "mt_post_Iss_pA",
    "mt_post_n_transients", "mt_Ra_change_MOhm", "mt_Ra_change_pct",
    # subthreshold current clamp
    "Vrest_measured_mV", "bias_current_pA", "bias_current_sd_pA", "sag_target_mV", "sag_ratio", "sag_amplitude_mV",
    "sag_v_peak_mV", "sag_stim_pA", "sag_bracketed", "sag_n_sweeps", "sag_note",
    "Rin_MOhm", "Rin_r2", "Rin_peak_MOhm", "Rin_n_sweeps",
    "Rin_I_min_pA", "Rin_I_max_pA", "Rin_V_min_mV", "Rin_V_max_mV",
    "Rin_sweeps_used", "Rin_note",
    # action potentials
    "resting_Vm_mV", "rheobase_pA", "rheobase_sweep", "n_spikes_total",
    "AP_threshold_mV", "AP_peak_mV", "AP_amplitude_mV", "AP_halfwidth_ms",
    "AP_max_upstroke_mV_per_ms", "AP_max_downstroke_mV_per_ms", "AP_AHP_mV",
    "AP_latency_ms", "max_firing_rate_Hz", "max_firing_rate_at_pA",
    "max_firing_sweep", "adaptation_index_at_max_rate",
    "half_max_sweep", "half_max_n_spikes", "half_max_stim_amp_pA",
    "files_unused", "notes", "folder",
]


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLUMN_ORDER if c in df.columns]
    return df[cols + [c for c in df.columns if c not in cols]]


def show_qc_window(root: Path, summary: "Optional[pd.DataFrame]" = None) -> bool:
    """Open the end-of-run QC window. Never fatal: a headless machine, a
    missing PyQt5 or a half-finished run must not lose the analysis that has
    already been written to disk."""
    try:
        import batch_qc_window
        batch_qc_window.show_qc_window(root, summary)
        return True
    except Exception as exc:
        print(f"NOTE: could not open the QC window ({exc}). The summary and the "
              f"per-cell files are still written; `--view` reopens it later.",
              file=sys.stderr)
        return False


def pick_root_folder() -> Optional[str]:
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title="Select the folder holding the cell subfolders")
    root.destroy()
    return path or None


# ---------------------------------------------------------------------------
# Synthetic validation
# ---------------------------------------------------------------------------
_SYNTH = dict(rin_MOhm=200.0, tau_ms=30.0, v_rest=-65.0, ih_onset_mV=-80.0,
              ih_gain=0.35, tau_h_ms=150.0,
              currents=(0, -20, -40, -60, -80, -100, -120, -150),
              pre=0.3, dur=1.0, post=0.3)


def _synth_step(I: float, t: np.ndarray, rin_MOhm, tau_ms, v_rest, ih_onset_mV,
                ih_gain, tau_h_ms, **_ignored) -> np.ndarray:
    """Membrane potential during a current step: a passive RC charge plus an
    Ih-like sag that only switches on below `ih_onset_mV`. The true Rin is
    therefore exactly recoverable from the small steps and deliberately
    unrecoverable from the large ones."""
    dv = I * rin_MOhm / 1e3                                   # pA * MOhm -> mV
    sag = ih_gain * max(0.0, ih_onset_mV - (v_rest + dv))      # mV of relaxation
    return (v_rest + dv * (1 - np.exp(-t / (tau_ms / 1e3)))
            + sag * (1 - np.exp(-t / (tau_h_ms / 1e3))))


def _synthetic_hyperpol_sweeps(fs=20_000.0, noise=0.05, seed=0, **params) -> list[dict]:
    """Sampled, noisy sweeps of the model, in the shape load_cc_sweeps returns."""
    p = {**_SYNTH, **params}
    rng = np.random.default_rng(seed)
    pre, dur, post = p["pre"], p["dur"], p["post"]
    n_pre, n_dur, n_post = int(pre * fs), int(dur * fs), int(post * fs)
    t = np.arange(n_pre + n_dur + n_post) / fs
    sweeps = []
    for k, I in enumerate(p["currents"]):
        step = _synth_step(I, np.arange(n_dur) / fs, **p)
        v = np.full(t.size, p["v_rest"])
        v[n_pre:n_pre + n_dur] = step
        v[n_pre + n_dur:] = p["v_rest"] + (step[-1] - p["v_rest"]) * np.exp(
            -(np.arange(n_post) / fs) / (p["tau_ms"] / 1e3))
        v = v + rng.normal(0, noise, v.size)
        i = np.zeros(t.size)
        i[n_pre:n_pre + n_dur] = I
        sweeps.append({"sweep": k, "t": t, "v": v, "i": i,
                       "start": pre, "end": pre + dur, "stim_pA": float(I)})
    return sweeps


def _synthetic_truth(target_v: float = -85.0, ss_interval: float = 0.1,
                     **params) -> dict:
    """What the model's own sag is, evaluated in closed form.

    Deliberately independent of the measurement code: the noiseless step is
    evaluated on a fine grid and reduced with plain numpy, using the same
    conventions the measurement uses (peak = minimum, steady state = mean
    over the last `ss_interval`). So a disagreement is the measurement's
    numerics -- windowing, peak averaging, noise, interpolation -- and not a
    difference of definition.
    """
    p = {**_SYNTH, **params}
    t = np.linspace(0.0, p["dur"], 200_001)
    peaks, ratios, amps = [], [], []
    for I in p["currents"]:
        if I >= 0:
            continue
        step = _synth_step(I, t, **p)
        v_peak = float(step.min())
        v_ss = float(step[t >= p["dur"] - ss_interval].mean())
        peaks.append(v_peak)
        amps.append(v_ss - v_peak)
        ratios.append((v_peak - v_ss) / (v_peak - p["v_rest"]))
    order = np.argsort(peaks)
    x = np.asarray(peaks)[order]
    return {
        "sag_ratio": float(np.interp(target_v, x, np.asarray(ratios)[order])),
        "sag_amplitude_mV": float(np.interp(target_v, x, np.asarray(amps)[order])),
        "rin_MOhm": p["rin_MOhm"], "v_rest": p["v_rest"],
    }


def _selftest() -> int:
    sweeps = _synthetic_hyperpol_sweeps()
    table = subthreshold_sweep_table(sweeps)
    summary = subthreshold_summary(table, target_v=-85.0, v_floor=-80.0, max_sag=0.05)
    truth = _synthetic_truth(target_v=-85.0)
    naive_slope, _ = _fit_slope(table["stim_pA"].to_numpy(float),
                                table["v_steady_mV"].to_numpy(float))

    print("Synthetic subthreshold validation (passive RC + Ih-like sag)\n")
    checks = [
        ("Rin (sag-free fit)", summary["Rin_MOhm"], truth["rin_MOhm"], "MOhm", 0.02),
        ("Rin (peak deflection)", summary["Rin_peak_MOhm"], truth["rin_MOhm"], "MOhm", 0.02),
        ("sag ratio at -85 mV", summary["sag_ratio"], truth["sag_ratio"], "", 0.05),
        ("sag amplitude at -85 mV", summary["sag_amplitude_mV"],
         truth["sag_amplitude_mV"], "mV", 0.05),
        ("resting Vm", summary["Vrest_measured_mV"], truth["v_rest"], "mV", 0.01),
    ]
    hdr = f"{'measurement':<26} {'measured':>11} {'model':>11} {'rel.err':>9}  result"
    print(hdr)
    print("-" * len(hdr))
    ok = True
    for label, got, want, unit, tol in checks:
        err = abs(got - want) / abs(want) if want else abs(got - want)
        passed = bool(err <= tol)
        ok &= passed
        print(f"{label:<26} {got:11.3f} {want:11.3f} {err:9.4f}  "
              f"{'PASS' if passed else 'FAIL'}  {unit}")
    print(f"\n  sweeps fitted: {summary['Rin_n_sweeps']} "
          f"(sweep {summary.get('Rin_sweeps_used', '')}), "
          f"I from {summary['Rin_I_min_pA']:.0f} to {summary['Rin_I_max_pA']:.0f} pA")
    print(f"  for contrast, fitting ALL sweeps gives Rin = {naive_slope * 1e3:.1f} MOhm: "
          f"Ih makes the large steps sublinear, which is exactly what the sweep "
          f"selection exists to avoid.")
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", help="folder holding the cell subfolders "
                                           "(omit to pick one via a dialog)")
    p.add_argument("--out", default=None, metavar="CSV",
                   help="summary table (default: <root>/batch_cell_summary.csv)")
    p.add_argument("--no-plot", action="store_true", help="skip every QC figure")
    p.add_argument("--include-unparsed", action="store_true",
                   help="also analyse folders whose name doesn't match "
                        "YYMMDD_slice_cell_RMP (metadata left blank)")
    p.add_argument("--sag-target", type=float, default=-85.0, metavar="MV",
                   help="membrane potential to report sag at (default -85)")
    p.add_argument("--rin-v-floor", type=float, default=-80.0, metavar="MV",
                   help="sweeps whose peak goes below this are excluded from the "
                        "Rin fit, as Ih distorts them (default -80)")
    p.add_argument("--rin-max-sag", type=float, default=0.05, metavar="RATIO",
                   help="also exclude sweeps whose sag ratio exceeds this "
                        "(default 0.05)")
    p.add_argument("--protocol-ra", default="DW_Ra",
                   help="protocol name of the membrane test (default DW_Ra); "
                        "falls back to any voltage-clamp file")
    p.add_argument("--protocol-steps", default="",
                   help="restrict the current-clamp step file to protocols whose "
                        "name contains this (default: any varying-step CC file)")
    p.add_argument("--protocol-hyperpol", default="hyperpol",
                   help="protocol name of a dedicated hyperpolarizing-steps file, "
                        "if the rig records one separately (default 'hyperpol')")
    p.add_argument("--dv-cutoff", type=float, default=20.0,
                   help="dV/dt spike-detection threshold in mV/ms (default 20)")
    p.add_argument("--min-peak", type=float, default=-30.0,
                   help="minimum AP peak voltage in mV (default -30)")
    p.add_argument("--dob", default="", metavar="YYYY-MM-DD",
                   help="animal's date of birth; gives each cell an age in days")
    p.add_argument("--sex", default="", help="animal's sex (male / female / unknown)")
    p.add_argument("--genotype", default="", help="animal's genotype")
    p.add_argument("--strain", default="", help="animal's strain, e.g. C57BL/6J")
    p.add_argument("--gui", action="store_true",
                   help="open the dialog even when a folder is given, pre-filled "
                        "with it -- e.g. to add the animal details to a re-run")
    p.add_argument("--no-gui", action="store_true",
                   help="skip the animal-details dialog and just pick a folder")
    p.add_argument("--no-window", action="store_true",
                   help="don't open the QC window when the run finishes")
    p.add_argument("--view", action="store_true",
                   help="open the QC window for a folder already analysed, "
                        "without re-running anything")
    p.add_argument("--traceback", action="store_true",
                   help="print a full traceback for each failing stage")
    p.add_argument("--selftest", action="store_true",
                   help="validate the sag / Rin maths on synthetic data and exit")
    args = p.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.dob and age_days(args.dob, "000101") is None:
        p.error(f"--dob must be a calendar date as YYYY-MM-DD, not {args.dob!r}")
    args.animal = {"genotype": args.genotype.strip(), "strain": args.strain.strip(),
                   "animal_sex": args.sex.strip(), "animal_dob": args.dob.strip()}

    # With no folder argument, the dialog collects the folder AND the animal
    # details in one step; --no-gui (or a missing PyQt5) falls back to a plain
    # folder picker, leaving the animal columns to the flags above.
    root: Optional[Path] = Path(args.root) if args.root else None
    if (root is None or args.gui) and not args.no_gui:
        try:
            import batch_run_dialog
        except ImportError as exc:
            print(f"NOTE: the run dialog needs PyQt5 ({exc}); falling back to a "
                  f"plain folder picker.", file=sys.stderr)
        else:
            chosen = batch_run_dialog.prompt_for_run(root)
            if chosen is None:
                print("Cancelled.")
                return 1
            root = Path(chosen.pop("root"))
            args.animal.update(chosen)
    if root is None:
        picked = pick_root_folder()
        if not picked:
            p.error("no folder selected")
        root = Path(picked)
    if not root.is_dir():
        p.error(f"not a folder: {root}")

    if args.view:
        return 0 if show_qc_window(root) else 1

    cells = find_cell_folders(root, include_unparsed=args.include_unparsed)
    if not cells:
        print(f"No cell folders with .abf files found under {root}.\n"
              f"Expected subfolders named YYMMDD_<slice>_<cell>_<RMP>, e.g. "
              f"260817_s1_c1_-67 (use --include-unparsed to analyse differently "
              f"named folders anyway).", file=sys.stderr)
        return 1

    print(f"{len(cells)} cell folder(s) under {root}\n")
    rows = []
    for n, (folder, meta) in enumerate(cells, 1):
        print(f"[{n}/{len(cells)}] {folder.name}")
        rows.append(analyze_cell(folder, meta, args))

    df = order_columns(pd.DataFrame(rows))
    out = Path(args.out) if args.out else root / "batch_cell_summary.csv"
    df.to_csv(out, index=False)

    failed = int(df["notes"].astype(str).str.contains("failed").sum())
    print(f"\nSummary ({len(df)} cells, {failed} with a failed stage)  ->  {out}")

    # The table in memory, not the CSV just written: identical content, but it
    # means a run whose summary landed somewhere else via --out still opens.
    if not args.no_window:
        show_qc_window(root, df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
