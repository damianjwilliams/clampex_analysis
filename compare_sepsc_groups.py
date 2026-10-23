#!/usr/bin/env python3
"""
compare_sepsc_groups.py
=======================

Compare sEPSC frequency and amplitude between groups of cells (genotype,
region, cell type...) whose recordings have different noise levels.

Why a plain comparison is biased
--------------------------------
The detector thresholds each recording in units of its OWN noise, so a quiet
recording detects smaller events than a noisy one. The small end of every
cell's amplitude distribution is therefore cut off at a different place --
the histogram peaks just above each cell's detection limit (4x noise SD). A
quiet cell's rate is inflated by events a noisy cell could not have seen, and
its median amplitude is pulled down by the same events. If one group is
noisier on average (a genotype with larger cells, a worse slice batch), the
groups will differ in rate and amplitude for no biological reason at all.

What this script does about it
------------------------------
  1. One amplitude cutoff for every cell compared: by default the largest
     detection limit among the included cells, rounded up to the next pA, so
     every cell detects events of that size completely. ``--min-amp`` sets it
     by hand; ``--max-limit`` drops the noisiest cells first, so one bad
     recording does not raise the cutoff for everybody.
  2. Rate, amplitude and inter-event interval re-computed per cell from only
     the events at or above the cutoff, from the ``*_sEPSC_events.csv`` the
     batch run wrote -- nothing is re-detected.
  3. Statistics on one number per CELL (Mann-Whitney for two groups,
     Kruskal-Wallis for more), never on pooled events: a cell with 2000
     events must not outvote a cell with 200.
  4. The noise itself compared between groups. If it differs, the cutoff is
     doing real work and the uncorrected numbers should not be reported.
  5. The comparison repeated at higher cutoffs. A real difference survives a
     few pA either way; one that appears or flips with the cutoff is noise.

Cells from the same animal are not independent. The per-cell table carries
``mouse_number`` and ``cage_number`` so the numbers can go into a mixed model
(animal as a random effect) when there are enough animals per group.

Outputs (in ``--out``, default the folder given):
    sepsc_group_cells.csv        one row per cell, corrected and raw
    sepsc_group_summary.csv      one row per group, plus the test results
    sepsc_group_cutoffs.csv      the group comparison at several cutoffs
    sepsc_group_comparison.png   per-cell rate, amplitude, noise and the
                                 cumulative amplitude distribution per group

Usage
-----
    python compare_sepsc_groups.py D:\\data\\benita_prelimbic
    python compare_sepsc_groups.py DATA_DIR --group genotype --min-amp 12
    python compare_sepsc_groups.py DIR1 DIR2 --group brain_region --max-limit 10

Each folder is searched (recursively) for ``batch_cell_summary.csv`` files.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

#: Extra cutoffs, in pA above the main one, for the sensitivity table.
CUTOFF_STEPS = (0.0, 2.0, 5.0, 10.0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def find_summaries(folders: list[Path]) -> list[Path]:
    found = []
    for folder in folders:
        if folder.is_file():
            found.append(folder)
        else:
            found.extend(sorted(folder.rglob("batch_cell_summary.csv")))
    return sorted(set(found))


def events_file(summary_csv: Path, row: pd.Series) -> Optional[Path]:
    """The cell's events CSV, found next to the summary rather than at the
    path stored in it: a summary written on another machine records a drive
    letter that may not exist here."""
    stem = Path(str(row.get("file_sepsc", ""))).stem
    if not stem:
        return None
    name = f"{stem}_sEPSC_events.csv"
    direct = summary_csv.parent / str(row["folder_name"]) / name
    if direct.is_file():
        return direct
    hits = list(summary_csv.parent.rglob(name))
    return hits[0] if hits else None


def load_cells(summaries: list[Path], group_col: str) -> pd.DataFrame:
    """Every cell with an sEPSC recording, with its event amplitudes attached."""
    rows = []
    for csv in summaries:
        df = pd.read_csv(csv)
        if "sepsc_n_events" not in df.columns:
            continue
        for _, r in df.dropna(subset=["sepsc_n_events"]).iterrows():
            ev_path = events_file(csv, r)
            rec = {
                "folder_name": r["folder_name"],
                "experiment": csv.parent.name,
                "group": str(r.get(group_col, "")) if pd.notna(r.get(group_col)) else "",
                "mouse_number": r.get("mouse_number", ""),
                "cage_number": r.get("cage_number", ""),
                "noise_pA": float(r.get("sepsc_noise_pA", np.nan)),
                "detection_limit_pA": float(r.get("sepsc_detection_limit_pA", np.nan)),
                "duration_s": float(r.get("sepsc_duration_s", np.nan)),
                "n_events_all": int(r["sepsc_n_events"]),
                "rate_all_Hz": float(r.get("sepsc_rate_Hz", np.nan)),
                "amp_median_all_pA": float(r.get("sepsc_amp_median_pA", np.nan)),
                "snr": float(r.get("sepsc_snr", np.nan)),
                "kernel_source": r.get("sepsc_kernel_source", ""),
                "events_file": str(ev_path) if ev_path else "",
                "_amps": None, "_times": None,
            }
            if ev_path is not None:
                ev = pd.read_csv(ev_path)
                rec["_amps"] = np.abs(pd.to_numeric(ev["amplitude"], errors="coerce").to_numpy(float))
                rec["_times"] = pd.to_numeric(ev["peak_time_s"], errors="coerce").to_numpy(float)
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-cell measurements above a common cutoff
# ---------------------------------------------------------------------------
def above_cutoff(amps: np.ndarray, times: np.ndarray, duration_s: float,
                 cutoff: float) -> dict:
    keep = np.isfinite(amps) & (amps >= cutoff)
    a, t = amps[keep], np.sort(times[keep])
    iei = np.diff(t) * 1e3
    return {
        "n_events": int(a.size),
        "rate_Hz": a.size / duration_s if duration_s > 0 else np.nan,
        "amp_median_pA": float(np.median(a)) if a.size else np.nan,
        "amp_mean_pA": float(np.mean(a)) if a.size else np.nan,
        "iei_median_ms": float(np.median(iei)) if iei.size else np.nan,
    }


def choose_cutoff(cells: pd.DataFrame, min_amp: Optional[float]) -> float:
    if min_amp is not None:
        return float(min_amp)
    worst = cells["detection_limit_pA"].max()
    return float(math.ceil(worst)) if np.isfinite(worst) else 0.0


def mark_excluded(cells: pd.DataFrame, max_limit: Optional[float],
                  cutoff_fn) -> tuple[pd.DataFrame, float]:
    cells = cells.copy()
    cells["excluded"] = ""
    cells.loc[cells["_amps"].isna(), "excluded"] = "events file not found"
    cells.loc[cells["group"] == "", "excluded"] = "no group value"
    if max_limit is not None:
        noisy = cells["detection_limit_pA"] > max_limit
        cells.loc[noisy & (cells["excluded"] == ""), "excluded"] = (
            f"detection limit above --max-limit {max_limit:g} pA")
    cutoff = cutoff_fn(cells[cells["excluded"] == ""])
    # A hand-set cutoff below a cell's own limit would let its incomplete
    # small-event range back in -- exactly what the cutoff is there to stop.
    under = (cells["detection_limit_pA"] > cutoff) & (cells["excluded"] == "")
    cells.loc[under, "excluded"] = f"detection limit above the {cutoff:g} pA cutoff"
    return cells, cutoff


def measure(cells: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    out = cells.copy()
    for col in ("n_events", "rate_Hz", "amp_median_pA", "amp_mean_pA", "iei_median_ms"):
        out[col] = np.nan
    for i, r in out.iterrows():
        if r["excluded"]:
            continue
        m = above_cutoff(r["_amps"], r["_times"], r["duration_s"], cutoff)
        for k, v in m.items():
            out.at[i, k] = v
    out["cutoff_pA"] = cutoff
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def group_test(df: pd.DataFrame, col: str) -> tuple[str, float]:
    from scipy import stats
    samples = [g[col].dropna().to_numpy() for _, g in df.groupby("group")]
    samples = [s for s in samples if s.size]
    if len(samples) < 2 or min(s.size for s in samples) < 2:
        return "too few cells", np.nan
    if len(samples) == 2:
        return "Mann-Whitney U", float(stats.mannwhitneyu(*samples, alternative="two-sided").pvalue)
    return "Kruskal-Wallis", float(stats.kruskal(*samples).pvalue)


SUMMARY_COLS = [("rate_Hz", "rate"), ("amp_median_pA", "amplitude"),
                ("iei_median_ms", "IEI"), ("noise_pA", "noise"),
                ("rate_all_Hz", "rate, uncorrected"),
                ("amp_median_all_pA", "amplitude, uncorrected")]


def summarise_groups(cells: pd.DataFrame) -> pd.DataFrame:
    inc = cells[cells["excluded"] == ""]
    rows = []
    for group, g in inc.groupby("group"):
        # Mouse numbers are only unique within a cage.
        animals = g["cage_number"].astype(str) + "/" + g["mouse_number"].astype(str)
        row = {"group": group, "n_cells": len(g), "n_animals": animals.nunique()}
        for col, _label in SUMMARY_COLS:
            v = g[col].dropna()
            row[f"{col}_mean"] = v.mean()
            row[f"{col}_sem"] = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    tests = {"group": "test p-value"}
    for col, _label in SUMMARY_COLS:
        name, p = group_test(inc, col)
        tests[f"{col}_mean"] = p
        tests[f"{col}_sem"] = name
    return pd.concat([summary, pd.DataFrame([tests])], ignore_index=True)


def cutoff_sensitivity(cells: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    rows = []
    for step in CUTOFF_STEPS:
        c = cutoff + step
        m = measure(cells, c)
        inc = m[m["excluded"] == ""]
        row = {"cutoff_pA": c}
        for group, g in inc.groupby("group"):
            row[f"{group}_rate_Hz"] = g["rate_Hz"].mean()
            row[f"{group}_amp_median_pA"] = g["amp_median_pA"].mean()
        row["rate_p"] = group_test(inc, "rate_Hz")[1]
        row["amp_p"] = group_test(inc, "amp_median_pA")[1]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot(cells: pd.DataFrame, cutoff: float, group_col: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inc = cells[cells["excluded"] == ""]
    groups = sorted(inc["group"].unique())
    colours = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    col_of = {g: colours[i % len(colours)] for i, g in enumerate(groups)}
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), tight_layout=True)
    panels = [("rate_Hz", f"Frequency, events >= {cutoff:g} pA (Hz)"),
              ("amp_median_pA", f"Median amplitude, events >= {cutoff:g} pA (pA)"),
              ("noise_pA", "Noise SD (pA)"),
              ("rate_all_Hz", "Frequency, uncorrected (Hz)"),
              ("amp_median_all_pA", "Median amplitude, uncorrected (pA)")]
    for ax, (col, title) in zip(axes.flat, panels):
        for k, g in enumerate(groups):
            v = inc.loc[inc["group"] == g, col].dropna().to_numpy()
            x = k + rng.uniform(-0.12, 0.12, v.size)
            ax.scatter(x, v, color=col_of[g], s=36, alpha=0.85, zorder=3)
            if v.size:
                m = v.mean()
                sem = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0
                ax.errorbar(k + 0.3, m, yerr=sem, fmt="_", color="k", ms=18, capsize=4)
        _name, p = group_test(inc, col)
        ax.set_title(f"{title}\np = {p:.3g}" if np.isfinite(p) else title, fontsize=9.5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}\n(n={int((inc['group'] == g).sum())})" for g in groups],
                           fontsize=8.5)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 2].axhline(cutoff / 4.0, color="#888888", ls="--", lw=0.9,
                       label=f"cutoff / 4 = noise the cutoff tolerates")
    axes[0, 2].legend(fontsize=7.5, loc="best")

    # Cumulative amplitude distribution: each cell's own CDF, averaged within
    # the group, so every cell weighs the same however many events it had.
    ax = axes[1, 2]
    grid = np.linspace(cutoff, max(cutoff + 1, np.nanpercentile(
        np.concatenate([a for a in inc["_amps"]]) if len(inc) else [cutoff + 1], 99)), 200)
    for g in groups:
        cdfs = []
        for amps in inc.loc[inc["group"] == g, "_amps"]:
            a = np.sort(amps[np.isfinite(amps) & (amps >= cutoff)])
            if a.size:
                cdfs.append(np.searchsorted(a, grid, side="right") / a.size)
        if cdfs:
            ax.plot(grid, np.mean(cdfs, axis=0), color=col_of[g], lw=2,
                    label=f"{g} (mean of {len(cdfs)} cells)")
    ax.set_xlabel("amplitude (pA)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title(f"Amplitude distribution, events >= {cutoff:g} pA", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"sEPSCs by {group_col}: one point per cell, mean +/- SEM", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+", type=Path,
                   help="folders to search for batch_cell_summary.csv (or the CSVs)")
    p.add_argument("--group", default="genotype", metavar="COLUMN",
                   help="summary column that defines the groups (default genotype)")
    p.add_argument("--min-amp", type=float, default=None, metavar="PA",
                   help="common amplitude cutoff; default the largest detection "
                        "limit among included cells, rounded up")
    p.add_argument("--max-limit", type=float, default=None, metavar="PA",
                   help="exclude cells whose detection limit (4x noise) is above this")
    p.add_argument("--out", type=Path, default=None,
                   help="output folder (default the first folder given)")
    args = p.parse_args(argv)

    summaries = find_summaries(args.folders)
    if not summaries:
        p.error("no batch_cell_summary.csv found")
    cells = load_cells(summaries, args.group)
    if cells.empty:
        print("No cells with sEPSC results in those summaries.", file=sys.stderr)
        return 1

    cells, cutoff = mark_excluded(cells, args.max_limit,
                                  lambda inc: choose_cutoff(inc, args.min_amp))
    cells = measure(cells, cutoff)
    out = args.out or (args.folders[0] if args.folders[0].is_dir() else args.folders[0].parent)
    out.mkdir(parents=True, exist_ok=True)

    public = [c for c in cells.columns if not c.startswith("_")]
    cells[public].to_csv(out / "sepsc_group_cells.csv", index=False)
    summary = summarise_groups(cells)
    summary.to_csv(out / "sepsc_group_summary.csv", index=False)
    sens = cutoff_sensitivity(cells, cutoff)
    sens.to_csv(out / "sepsc_group_cutoffs.csv", index=False)
    plot(cells, cutoff, args.group, out / "sepsc_group_comparison.png")

    inc = cells[cells["excluded"] == ""]
    print(f"{len(summaries)} summaries, {len(cells)} cells with sEPSCs, "
          f"{len(inc)} compared by {args.group}.")
    print(f"Common amplitude cutoff: {cutoff:g} pA"
          + ("" if args.min_amp is not None else " (largest detection limit, rounded up)"))
    for _, r in cells[cells["excluded"] != ""].iterrows():
        print(f"  excluded {r['folder_name']}: {r['excluded']}")
    print()
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:.3g}".format):
        print(inc[["folder_name", "group", "noise_pA", "detection_limit_pA",
                   "rate_all_Hz", "rate_Hz", "amp_median_all_pA", "amp_median_pA"]]
              .to_string(index=False))
        print()
        print(sens.to_string(index=False))
    noise_p = group_test(inc, "noise_pA")[1]
    if np.isfinite(noise_p) and noise_p < 0.05:
        print(f"\nNoise differs between groups (p = {noise_p:.3g}): report the "
              f"cutoff-corrected numbers, not the uncorrected ones.")
    print(f"\nWritten to {out}: sepsc_group_cells.csv, sepsc_group_summary.csv, "
          f"sepsc_group_cutoffs.csv, sepsc_group_comparison.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
