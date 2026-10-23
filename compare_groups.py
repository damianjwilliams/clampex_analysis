#!/usr/bin/env python3
"""
compare_groups.py
=================

Compare two (or more) groups of cells -- by default WT against Het -- on
action-potential firing and sEPSCs, from the files a batch run already wrote.
Nothing is re-analysed.

Groups come from the ``genotype`` column of each ``batch_cell_summary.csv``,
which the batch run filled from that experiment folder's ``run_metadata.txt``.
``--groups`` maps genotype labels onto groups (case-insensitive), so several
labels can be pooled:

    --groups "WT=WT" "Het=Het,mutant"      (the default)

A cell whose genotype matches no group is left out and listed. Every
experiment folder is printed with its genotype first, so the grouping can be
checked before reading any result.

Firing
------
  * Average F-I curve per group: firing rate (spikes / s during the 1 s step,
    ``avg_rate`` in each cell's ``*_AP_sweeps.csv``) against injected
    current, mean +- SEM across cells, with every cell drawn faintly behind.
    The cells must share one step protocol; currents only some cells reached
    are drawn with the n they rest on.
  * Per-cell measurements (rheobase, max rate, F-I area, AP threshold,
    amplitude, half-width, AHP, upstroke, resting Vm, Rin, sag, Ra) compared
    by Mann-Whitney (Kruskal-Wallis for 3+ groups), one number per cell.
  * The F-I curves themselves compared current by current (Mann-Whitney,
    Holm-corrected across currents) -- in the CSV, not as the headline result.

sEPSCs
------
Handed to ``compare_sepsc_groups``: one amplitude cutoff for every cell (the
largest detection limit), rate / amplitude / IEI re-computed above it, and the
comparison repeated at higher cutoffs. See that module's docstring for why.

Cells from the same animal are not independent; ``n_animals`` is reported
beside ``n_cells`` for that reason.

Outputs (in ``--out``, default ``<first folder>/group_comparison``):
    groups_experiments.csv     every experiment folder, its genotype, its group
    fi_curves_cells.csv        one row per cell per current step
    fi_curves_mean.csv         mean / SEM / n per group per current, + tests
    fi_anova.csv               mixed ANOVA of the F-I curves: genotype x current,
                               Greenhouse-Geisser and permutation p values
    firing_cells.csv           per-cell firing measurements, with group
    firing_stats.csv           group means and the test per measurement,
                               with its category (AP firing / Passive)
    fi_curves.png              the average F-I curves
    fi_curves_individual.png   every cell's F-I curve, per group and overlaid
    rheobase_vs_rmp.png        rheobase against RMP, input resistance and the
                               threshold - RMP distance, per cell
    rheobase_models.csv        linear models: rheobase ~ genotype, + RMP,
                               + RMP + Rin, genotype x RMP
    rheobase_correlations.csv  Spearman correlations behind rheobase_vs_rmp
    block_examples.png         example top-step traces: sustained firing vs
                               depolarization block, per group
    ap_firing_comparison.png   per-cell AP firing characteristics (box plots)
    passive_properties_comparison.png
                               per-cell passive membrane properties (box plots)
    firing_regularity.png      ISI CV, CV2 and adaptation at each cell's
                               half-max firing step
    depolarization_block.png   current at max APs, max APs, and depolarization
                               block (drop in APs to the top step, firing
                               duration in the top step; cells entering block)
    sepsc_boxplots.png         per-cell sEPSC measurements by group (box plots)
    sepsc_amplitude_distribution.csv/.png
                               amplitude-distribution tests: permutation test on
                               the mean-of-cells CDFs and per-cell percentiles
                               (cells as the unit), plus KS / Anderson-Darling on
                               equal-N pooled events (--events-per-cell)
    sepsc_complete_figure.png/.pdf
                               everything in one figure: average event,
                               per-cell box plots, cumulative amplitude / IEI
                               curves and the amplitude histogram
    sepsc_average_event.png/.pdf
                               average event per genotype, raw and peak-scaled,
                               with per-cell rise time and decay tau
    sepsc_figure.png/.pdf      publication-style figure: cumulative amplitude and
                               IEI curves with per-cell insets, and an amplitude
                               histogram (--hist-bin pA bins)
    sepsc_group_*.csv/.png     the sEPSC comparison (compare_sepsc_groups)

Usage
-----
    python compare_groups.py E:\\SNF_electrophysiology_data\\benita_prelimbic
    python compare_groups.py DIR --groups "WT=WT" "Het=Het" "mutant=mutant"
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import compare_sepsc_groups as cs

DEFAULT_GROUPS = ["WT=WT", "Het=Het,mutant"]

#: (summary column, label, unit) compared per cell. Ra is included as a
#: control: a group difference there would make every other one suspect.
#: RMP is the one written in the cell folder's name at the rig; input
#: resistance, capacitance and Ra come from the cell's FIRST voltage-clamp
#: membrane test (see first_membrane_test).
AP_MEASURES = [
    ("rheobase_pA", "Rheobase", "pA"),
    ("max_firing_rate_Hz", "Max firing rate", "Hz"),
    ("fi_area", "F-I area", "spikes"),
    ("AP_threshold_mV", "AP threshold", "mV"),
    ("AP_amplitude_mV", "AP amplitude", "mV"),
    ("AP_halfwidth_ms", "AP half-width", "ms"),
    ("AP_AHP_mV", "AHP", "mV"),
    ("AP_max_upstroke_mV_per_ms", "Max upstroke", "mV/ms"),
]
PASSIVE_MEASURES = [
    ("rmp_folder_mV", "RMP (folder name)", "mV"),
    ("vc_Rin_MOhm", "Input resistance (VC)", "MOhm"),
    ("vc_Cm_pF", "Capacitance (VC)", "pF"),
    ("sag_ratio", "Sag ratio", ""),
    ("vc_Ra_MOhm", "Ra (control)", "MOhm"),
]
#: Depolarization block, from each cell's own F-I curve (see block_measures).
BLOCK_MEASURES = [
    ("i_max_aps_pA", "Current at max APs", "pA"),
    ("max_aps", "Max APs per step", "APs"),
    ("block_drop_pct", "Drop in APs, peak to top step", "%"),
    ("top_step_firing_frac", "Top-step firing duration", "fraction"),
]
#: Firing regularity at the half-max firing step (see regularity_measures).
REGULARITY_MEASURES = [
    ("half_max_stim_amp_pA", "Half-max step current", "pA"),
    ("hm_isi_cv", "ISI CV", ""),
    ("hm_cv2", "CV2 (local)", ""),
    ("hm_adapt_ratio", "Adaptation (last/first 3 ISIs)", ""),
]
#: The same near rheobase: the first step at or above rheobase with enough
#: ISIs, since rheobase itself is usually a single spike.
RHEO_REGULARITY_MEASURES = [
    ("aps_at_rheobase", "APs at rheobase", "APs"),
    ("nr_above_rheo_pA", "Step used, above rheobase", "pA"),
    ("nr_isi_cv", "ISI CV", ""),
    ("nr_cv2", "CV2 (local)", ""),
    ("nr_adapt_ratio", "Adaptation (last/first 3 ISIs)", ""),
]
#: Fewest ISIs in a step for the regularity measures to be computed.
MIN_ISIS = 4

#: A cell counts as entering depolarization block when, at a current above
#: its peak, it fires at most this fraction of its peak number of APs.
BLOCK_FRACTION = 0.5

#: (category, measures, figure file, figure title, columns in the figure)
MEASURE_SETS = [
    ("AP firing", AP_MEASURES, "ap_firing_comparison.png",
     "AP firing characteristics", 4),
    ("Passive", PASSIVE_MEASURES, "passive_properties_comparison.png",
     "Passive membrane properties", 5),
    ("Depolarization block", BLOCK_MEASURES, "depolarization_block.png",
     "Maximum firing and depolarization block", 4),
    ("Regularity", REGULARITY_MEASURES, "firing_regularity.png",
     "Firing regularity at the half-max firing step", 4),
    ("Regularity near rheobase", RHEO_REGULARITY_MEASURES,
     "firing_regularity_rheobase.png",
     "Firing regularity near rheobase (first step at or above rheobase with "
     f"≥ {MIN_ISIS} ISIs)", 5),
]
FIRING_MEASURES = (AP_MEASURES + PASSIVE_MEASURES + BLOCK_MEASURES + REGULARITY_MEASURES
                   + RHEO_REGULARITY_MEASURES)

#: A group mean at one current is drawn solid only if it rests on this many cells.
MIN_CELLS_PER_POINT = 3

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def save_fig(fig, path: Path, **kwargs) -> None:
    """fig.savefig, retried for a few seconds if the file is briefly held open.

    On this rig's data drive a just-written PNG is sometimes still open in
    another process (image viewer, Explorer preview, antivirus) when the next
    run rewrites it, and Windows then refuses the write with EINVAL.
    """
    for attempt in range(10):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError:
            if attempt == 9:
                raise
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------
def parse_groups(specs: list[str]) -> dict[str, str]:
    """{genotype label (lower case): group name} from "Group=label1,label2"."""
    mapping: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--groups entry {spec!r} is not GROUP=label[,label]")
        group, labels = spec.split("=", 1)
        for label in labels.split(","):
            if label.strip():
                mapping[label.strip().lower()] = group.strip()
    return mapping


def group_order(specs: list[str]) -> list[str]:
    return list(dict.fromkeys(s.split("=", 1)[0].strip() for s in specs))


def load_summaries(summaries: list[Path], mapping: dict[str, str]) -> pd.DataFrame:
    """Every cell of every summary, with ``experiment`` and ``group`` added."""
    frames = []
    for csv in summaries:
        df = pd.read_csv(csv)
        df["experiment"] = csv.parent.name
        df["_summary"] = str(csv)
        geno = df.get("genotype", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
        df["genotype"] = geno
        df["group"] = geno.str.lower().map(mapping).fillna("")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def print_experiments(cells: pd.DataFrame) -> pd.DataFrame:
    """The date / genotype / group of every experiment folder, on the terminal."""
    rows = []
    for exp, d in cells.groupby("experiment", sort=True):
        genos = sorted(set(d["genotype"]) - {""}) or ["(blank)"]
        groups = sorted(set(d["group"]) - {""}) or ["(not compared)"]
        rows.append({"experiment": exp, "genotype": "/".join(genos),
                     "group": "/".join(groups), "n_cells": len(d),
                     "n_fi": int(d["rheobase_pA"].notna().sum()) if "rheobase_pA" in d else 0,
                     "n_sepsc": int(d["sepsc_n_events"].notna().sum())
                     if "sepsc_n_events" in d else 0})
    table = pd.DataFrame(rows)
    print(f"{'date':<12}{'genotype':<12}{'group':<16}{'cells':>6}{'F-I':>6}{'sEPSC':>7}")
    for r in table.itertuples():
        print(f"{r.experiment:<12}{r.genotype:<12}{r.group:<16}{r.n_cells:>6}"
              f"{r.n_fi:>6}{r.n_sepsc:>7}")
    print()
    return table


# ---------------------------------------------------------------------------
# F-I curves
# ---------------------------------------------------------------------------
def sweeps_file(row: pd.Series) -> Optional[Path]:
    """The cell's ``*_AP_sweeps.csv``, found next to its summary."""
    stem = Path(str(row.get("file_steps", ""))).stem
    if not stem or stem == "nan":
        return None
    folder = Path(row["_summary"]).parent / str(row["folder_name"])
    hit = folder / f"{stem}_AP_sweeps.csv"
    if hit.is_file():
        return hit
    hits = list(Path(row["_summary"]).parent.rglob(f"{stem}_AP_sweeps.csv"))
    return hits[0] if hits else None


def animal_id(row: pd.Series) -> str:
    """Which animal a cell came from: cage + mouse number when recorded, else
    the experiment folder (one animal per run). Keyed on the animal, not the
    day, so a mouse recorded on two days counts once."""
    def clean(v):
        return "" if pd.isna(v) else str(v).strip().removesuffix(".0")
    mouse, cage = clean(row.get("mouse_number")), clean(row.get("cage_number"))
    return f"cage {cage} mouse {mouse}" if mouse else f"run {row['experiment']}"


def first_membrane_test(row: pd.Series, protocol: str = "DW_Ra") -> dict:
    """Rin / Cm / Ra from the first membrane-test recording in the cell folder.

    "First" is by recording time from the .abf headers. The batch summary
    already holds the measurements of the test before the steps (mt_*) and
    after them (mt_post_*); whichever of the two is the first file is used.
    A first file that is neither -- e.g. an extra test recorded before the
    one the batch paired with the steps -- is measured here directly.

    Input resistance is the membrane resistance Rm = Rt - Ra: the total
    steady-state resistance of the test pulse with the access resistance in
    series with it taken out. Cm is the bias-corrected capacitance.
    """
    import batch_cell_analysis as B

    out = {"vc_file": "", "vc_Rin_MOhm": np.nan, "vc_Cm_pF": np.nan, "vc_Ra_MOhm": np.nan}
    folder = Path(row["_summary"]).parent / str(row["folder_name"])
    if not folder.is_dir():
        return out
    cands, _notes = B.membrane_test_candidates(B.scan_folder(folder), protocol)
    if not cands:
        return out
    dated = [c for c in cands if c["when"]]
    first = (min(dated, key=lambda c: c["when"]) if dated else cands[0])["path"]
    out["vc_file"] = first.name
    for prefix, file_col in (("mt_", "file_membrane_test"),
                             ("mt_post_", "file_membrane_test_post")):
        if str(row.get(file_col, "")) == first.name:
            out.update(vc_Rin_MOhm=_num_or_nan(row.get(f"{prefix}Rm_MOhm")),
                       vc_Cm_pF=_num_or_nan(row.get(f"{prefix}Cm_pF")),
                       vc_Ra_MOhm=_num_or_nan(row.get(f"{prefix}Ra_MOhm")))
            return out
    try:
        import calculate_Ra as mt
        s = mt.membrane_test_from_abf(str(first), channel=_vc_channel(first)).summary()
        out.update(vc_Rin_MOhm=s["Rm"]["mean"], vc_Cm_pF=s["Cm"]["mean"],
                   vc_Ra_MOhm=s["Ra"]["mean"])
    except (Exception, SystemExit) as exc:
        print(f"  note: membrane test not measured for {row['folder_name']} ({exc})")
    return out


def _num_or_nan(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _vc_channel(path: Path) -> int:
    import pyabf
    abf = pyabf.ABF(str(path), loadData=False)
    for ch in abf.channelList:
        if abf.adcUnits[ch].strip().lower() in ("pa", "na", "a"):
            return ch
    return 0


def load_fi(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per cell per current step: group, cell, stim_pA, rate_Hz."""
    rows = []
    for _, r in cells.iterrows():
        path = sweeps_file(r)
        if path is None:
            continue
        sw = pd.read_csv(path)
        for _, s in sw.iterrows():
            rows.append({"group": r["group"], "experiment": r["experiment"],
                         "folder_name": r["folder_name"],
                         "animal": animal_id(r),
                         "stim_pA": round(float(s["stim_amp_pA"]), 1),
                         "rate_Hz": float(s["avg_rate"]),
                         "n_spikes": int(s["n_spikes"])})
    return pd.DataFrame(rows)


def block_measures(row: pd.Series) -> dict:
    """Maximum firing and depolarization block from one cell's step recording.

    From the cell's *_AP_sweeps.csv (APs per step) and *_AP_spikes.csv (spike
    times):
      i_max_aps_pA          current of the step with the most APs (lowest
                            such current on a tie)
      max_aps               that number of APs
      block_drop_pct        100 * (max - APs at the top step) / max: 0 = still
                            firing at its peak at the highest current, 100 =
                            silent there
      top_step_firing_frac  time of the last spike in the top step, as a
                            fraction of the step: a cell going into block
                            fires at the start of the step and then stops
      block_current_pA      the lowest current above the peak at which the
                            cell fires <= BLOCK_FRACTION of its peak (NaN if
                            it never does) -- and enters_block, True/False
    """
    out = {"i_max_aps_pA": np.nan, "max_aps": np.nan, "block_drop_pct": np.nan,
           "top_step_firing_frac": np.nan, "block_current_pA": np.nan,
           "enters_block": np.nan}
    path = sweeps_file(row)
    if path is None:
        return out
    sw = pd.read_csv(path).sort_values("stim_amp_pA").reset_index(drop=True)
    n = sw["n_spikes"].to_numpy(float)
    stim = sw["stim_amp_pA"].to_numpy(float)
    if not n.size or np.nanmax(n) <= 0:
        return out
    k = int(np.nanargmax(n))                    # first (lowest-current) maximum
    out.update(i_max_aps_pA=stim[k], max_aps=n[k],
               block_drop_pct=100.0 * (n[k] - n[-1]) / n[k])
    above = np.nonzero((np.arange(n.size) > k) & (n <= BLOCK_FRACTION * n[k]))[0]
    out["enters_block"] = bool(above.size)
    if above.size:
        out["block_current_pA"] = stim[above[0]]

    top = sw.iloc[-1]
    dur = float(top["stim_end_s"] - top["stim_start_s"])
    spikes_path = path.with_name(path.name.replace("_AP_sweeps.csv", "_AP_spikes.csv"))
    if spikes_path.is_file() and dur > 0:
        sp = pd.read_csv(spikes_path, usecols=["sweep", "peak_t"])
        t = sp.loc[sp["sweep"] == top["sweep"], "peak_t"].to_numpy(float)
        out["top_step_firing_frac"] = (
            float(np.clip((t.max() - top["stim_start_s"]) / dur, 0, 1)) if t.size else 0.0)
    return out


def _step_isis(sw: pd.DataFrame, sp: pd.DataFrame, sweep: int) -> np.ndarray:
    """ISIs (ms) between the spike peaks inside one step's stimulus window."""
    step = sw[sw["sweep"] == int(sweep)]
    if step.empty:
        return np.array([])
    t0, t1 = float(step["stim_start_s"].iloc[0]), float(step["stim_end_s"].iloc[0])
    t = np.sort(sp.loc[(sp["sweep"] == int(sweep)) & (sp["peak_t"] >= t0)
                       & (sp["peak_t"] <= t1), "peak_t"].to_numpy(float))
    return np.diff(t) * 1e3


def _regularity(isi: np.ndarray, prefix: str) -> dict:
    """ISI CV, CV2 and adaptation ratio of one spike train, keyed with `prefix`."""
    out = {f"{prefix}_n_isis": int(isi.size), f"{prefix}_isi_cv": np.nan,
           f"{prefix}_cv2": np.nan, f"{prefix}_adapt_ratio": np.nan}
    if isi.size < MIN_ISIS:
        return out
    out[f"{prefix}_isi_cv"] = float(isi.std(ddof=1) / isi.mean())
    out[f"{prefix}_cv2"] = float(np.mean(2 * np.abs(np.diff(isi)) / (isi[1:] + isi[:-1])))
    k = min(3, isi.size // 2)
    out[f"{prefix}_adapt_ratio"] = float(isi[-k:].mean() / isi[:k].mean())
    return out


def regularity_measures(row: pd.Series) -> dict:
    """How regular the spike train is, at two points on the cell's F-I curve.

    Half-max step (``hm_*``): ``half_max_sweep``, chosen by the AP analysis --
    the step whose spike count is closest to half the cell's maximum, on the
    rising limb of the F-I curve, so the same point on every cell's own
    curve, clear of threshold and of depolarization block.

    Near rheobase (``nr_*``): the rheobase step itself usually carries a
    single spike (median 1 in this dataset), which has no ISIs at all, so
    the first step AT OR ABOVE rheobase with at least MIN_ISIS ISIs is used;
    ``nr_above_rheo_pA`` says how far above rheobase that was, and
    ``aps_at_rheobase`` how many spikes rheobase itself gave.

    For each spike train:
      *_isi_cv       SD / mean of the ISIs: 0 = perfectly regular; slow
                     adaptation alone also raises it
      *_cv2          mean of 2|ISI(i+1) - ISI(i)| / (ISI(i+1) + ISI(i))
                     (Holt et al. 1996): spike-to-spike irregularity,
                     insensitive to slow adaptation
      *_adapt_ratio  mean of the last 3 ISIs / mean of the first 3:
                     1 = no adaptation, > 1 = firing slows during the step
    NaN when the train has fewer than MIN_ISIS ISIs.
    """
    out = {**_regularity(np.array([]), "hm"), **_regularity(np.array([]), "nr"),
           "nr_step_pA": np.nan, "nr_above_rheo_pA": np.nan, "aps_at_rheobase": np.nan}
    path = sweeps_file(row)
    if path is None:
        return out
    spikes_path = path.with_name(path.name.replace("_AP_sweeps.csv", "_AP_spikes.csv"))
    if not spikes_path.is_file():
        return out
    sw = pd.read_csv(path).sort_values("stim_amp_pA")
    sp = pd.read_csv(spikes_path, usecols=["sweep", "peak_t"])

    if pd.notna(row.get("half_max_sweep")):
        out.update(_regularity(_step_isis(sw, sp, int(row["half_max_sweep"])), "hm"))

    if pd.notna(row.get("rheobase_sweep")):
        rheo = sw[sw["sweep"] == int(row["rheobase_sweep"])]
        if len(rheo):
            rheo_pA = float(rheo["stim_amp_pA"].iloc[0])
            out["aps_at_rheobase"] = int(rheo["n_spikes"].iloc[0])
            for _, step in sw[sw["stim_amp_pA"] >= rheo_pA].iterrows():
                isi = _step_isis(sw, sp, int(step["sweep"]))
                if isi.size >= MIN_ISIS:
                    out.update(_regularity(isi, "nr"))
                    out["nr_step_pA"] = float(step["stim_amp_pA"])
                    out["nr_above_rheo_pA"] = float(step["stim_amp_pA"]) - rheo_pA
                    break
    return out


def block_proportions(cells: pd.DataFrame, groups: list[str]) -> tuple[dict, float]:
    """Cells entering block per group, and Fisher's exact p (2 groups)."""
    from scipy import stats
    counts = {}
    for g in groups:
        v = cells.loc[cells["group"] == g, "enters_block"].dropna().astype(bool)
        counts[g] = (int(v.sum()), int(v.size))
    p = np.nan
    if len(groups) == 2 and all(n for _k, n in counts.values()):
        (a, na), (b, nb) = counts[groups[0]], counts[groups[1]]
        p = float(stats.fisher_exact([[a, na - a], [b, nb - b]])[1])
    return counts, p


def fi_area(fi: pd.DataFrame) -> pd.Series:
    """Area under each cell's F-I curve (Hz x pA / 1000 = spikes per nA over
    the 1 s step), over the current range every cell reached -- so a cell is
    not credited for steps it was never given."""
    top = fi.groupby(["experiment", "folder_name"])["stim_pA"].max().min()
    sub = fi[fi["stim_pA"] <= top].sort_values("stim_pA")
    return sub.groupby(["experiment", "folder_name"]).apply(
        lambda d: float(np.trapezoid(d["rate_Hz"], d["stim_pA"]) / 1000.0),
        include_groups=False)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p values (NaNs left alone)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def fi_means(fi: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for stim, d in fi.groupby("stim_pA"):
        row = {"stim_pA": stim}
        for g in groups:
            v = d.loc[d["group"] == g, "rate_Hz"].to_numpy(float)
            row[f"{g}_mean_Hz"] = v.mean() if v.size else np.nan
            row[f"{g}_sem_Hz"] = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
            row[f"{g}_n"] = int(v.size)
        test, p = cs.group_test(d.rename(columns={"rate_Hz": "x"}), "x")
        row["test"], row["p"] = test, p
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy(float))
    return out


def _mixed_anova_F(Y: np.ndarray, labels: np.ndarray) -> dict:
    """Sums of squares and F for a split-plot ANOVA.

    Y is cells x levels (every cell measured at every level), labels the
    between-cell group of each row. Returns SS, df and F for the group
    (between), level (within) and group x level effects, with their error
    terms: cells-within-groups for the group effect, level x
    cells-within-groups for the two within effects.
    """
    n, k = Y.shape
    grand = Y.mean()
    groups = np.unique(labels)
    cell_means = Y.mean(axis=1)
    level_means = Y.mean(axis=0)
    ss_total = ((Y - grand) ** 2).sum()
    ss_group = sum(k * (labels == g).sum() * (Y[labels == g].mean() - grand) ** 2
                   for g in groups)
    ss_cells = k * ((cell_means - grand) ** 2).sum()          # all between-cell variance
    ss_err_between = ss_cells - ss_group
    ss_level = n * ((level_means - grand) ** 2).sum()
    ss_cells_levels = sum((labels == g).sum() * ((Y[labels == g].mean(axis=0)
                                                  - grand) ** 2).sum() for g in groups)
    ss_inter = ss_cells_levels - ss_group - ss_level
    ss_err_within = ss_total - ss_cells - ss_level - ss_inter
    a = len(groups)
    df_group, df_err_b = a - 1, n - a
    df_level, df_inter, df_err_w = k - 1, (a - 1) * (k - 1), (n - a) * (k - 1)
    ms = lambda ss, df: ss / df if df > 0 else np.nan
    return {
        "group": (ss_group, df_group, ms(ss_group, df_group) / ms(ss_err_between, df_err_b)),
        "level": (ss_level, df_level, ms(ss_level, df_level) / ms(ss_err_within, df_err_w)),
        "inter": (ss_inter, df_inter, ms(ss_inter, df_inter) / ms(ss_err_within, df_err_w)),
        "err_b": (ss_err_between, df_err_b, np.nan),
        "err_w": (ss_err_within, df_err_w, np.nan),
    }


def _greenhouse_geisser(Y: np.ndarray, labels: np.ndarray) -> float:
    """Greenhouse-Geisser epsilon from the pooled within-group covariance of
    the repeated levels. 1 = sphericity holds; 1/(k-1) = maximal violation."""
    k = Y.shape[1]
    S = np.zeros((k, k))
    dof = 0
    for g in np.unique(labels):
        Yg = Y[labels == g]
        if len(Yg) > 1:
            S += np.cov(Yg, rowvar=False) * (len(Yg) - 1)
            dof += len(Yg) - 1
    S /= max(dof, 1)
    C = np.eye(k) - np.ones((k, k)) / k
    Sc = C @ S @ C                                         # double-centred
    denom = (k - 1) * np.sum(Sc ** 2)
    return float(np.trace(Sc) ** 2 / denom) if denom > 0 else np.nan


def fi_anova(fi: pd.DataFrame, groups: list[str], n_perm: int = 5000,
             seed: int = 0) -> pd.DataFrame:
    """Mixed-design ANOVA of the F-I curves: genotype (between cells) x
    injected current (within cells), on firing rate.

      genotype            are the curves different in overall height?
      current             does rate depend on current? (trivially yes)
      genotype x current  do the curves differ in SHAPE -- e.g. one shifted
                          along the current axis, or blocking sooner?

    Uses only the currents every cell was given, so the design is balanced.
    Sphericity cannot hold across this many current steps, so the within
    effects carry Greenhouse-Geisser corrected p values (p_gg). p_perm comes
    from shuffling the genotype labels across cells n_perm times -- no
    normality or sphericity assumption at all; it is the p to trust when
    the two disagree.
    """
    from scipy import stats

    wide = fi.pivot_table(index=["group", "experiment", "folder_name"],
                          columns="stim_pA", values="rate_Hz")
    wide = wide.dropna(axis=1)                      # currents every cell reached
    wide = wide[wide.index.get_level_values("group").isin(groups)]
    Y = wide.to_numpy(float)
    labels = np.asarray(wide.index.get_level_values("group"))
    res = _mixed_anova_F(Y, labels)
    eps = _greenhouse_geisser(Y, labels)

    rng = np.random.default_rng(seed)
    perm_F = {"group": [], "inter": []}
    for _ in range(n_perm):
        r = _mixed_anova_F(Y, rng.permutation(labels))
        perm_F["group"].append(r["group"][2])
        perm_F["inter"].append(r["inter"][2])

    rows = []
    names = {"group": "genotype", "level": "current", "inter": "genotype x current",
             "err_b": "error (cells within genotype)",
             "err_w": "error (current x cells within genotype)"}
    for key in ("group", "level", "inter", "err_b", "err_w"):
        ss, df, F = res[key]
        row = {"effect": names[key], "SS": ss, "df": df, "MS": ss / df if df else np.nan,
               "F": F, "p": np.nan, "gg_epsilon": np.nan, "p_gg": np.nan, "p_perm": np.nan}
        if key in ("group", "level", "inter"):
            df_err = res["err_b"][1] if key == "group" else res["err_w"][1]
            row["p"] = float(stats.f.sf(F, df, df_err))
            if key != "group":
                row["gg_epsilon"] = eps
                row["p_gg"] = float(stats.f.sf(F, df * eps, df_err * eps))
            else:
                row["p_gg"] = row["p"]                # between effect: no correction
            if key in perm_F:
                row["p_perm"] = (1 + np.sum(np.array(perm_F[key]) >= F)) / (n_perm + 1)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["n_cells"] = {g: int((labels == g).sum()) for g in groups}
    out.attrs["currents"] = (float(wide.columns.min()), float(wide.columns.max()),
                             int(wide.shape[1]))
    return out


def anova_text(av: pd.DataFrame) -> str:
    """One line for the F-I figure: the genotype and interaction effects."""
    def one(effect, label):
        r = av[av["effect"] == effect].iloc[0]
        return f"{label} F = {r['F']:.2f}, {p_text(r['p_gg'])} (perm. {p_text(r['p_perm'])})"
    return ("Mixed ANOVA: " + one("genotype", "genotype") + ";  "
            + one("genotype x current", "genotype x current (GG-corrected)"))


def plot_fi(fi: pd.DataFrame, means: pd.DataFrame, groups: list[str], path: Path,
            subtitle: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Black and white: groups differ by marker fill and line style, not colour.
    # (marker, face, line style) per group, in --groups order.
    styles = [("o", "black", "-"), ("o", "white", "--"), ("s", "black", ":"),
              ("s", "white", "-."), ("^", "black", "-")]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for k, g in enumerate(groups):
        marker, face, ls = styles[k % len(styles)]
        m = means[means[f"{g}_n"] > 0]
        enough = (m[f"{g}_n"] >= MIN_CELLS_PER_POINT).to_numpy()
        n_cells = fi.loc[fi["group"] == g, ["experiment", "folder_name"]].drop_duplicates().shape[0]
        n_animals = fi.loc[fi["group"] == g, "animal"].nunique()
        x, y, e = (m["stim_pA"].to_numpy(), m[f"{g}_mean_Hz"].to_numpy(),
                   m[f"{g}_sem_Hz"].to_numpy())
        # a small sideways offset keeps the two groups' error bars apart
        dx = (k - (len(groups) - 1) / 2) * 3.0
        ax.errorbar(x[enough] + dx, y[enough], yerr=e[enough], fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=2, zorder=2)
        ax.plot(x[enough] + dx, y[enough], ls=ls, color="black", lw=1.4,
                marker=marker, ms=5, mfc=face, mec="black", mew=1.0, zorder=3,
                label=f"{g}  (n = {n_cells} cells, {n_animals} animals)")
        if (~enough).any():
            ax.plot(x[~enough] + dx, y[~enough], ls="none", marker=marker, ms=4,
                    mfc=face, mec="#888888", mew=0.8, zorder=3)
    sig = means[means["p_holm"] < 0.05]
    if len(sig):
        ytop = ax.get_ylim()[1]
        ax.plot(sig["stim_pA"], [ytop * 0.97] * len(sig), "k*", ms=6, ls="none",
                label="p < 0.05 at that current (Holm-corrected)")
    ax.set_xlabel("Injected current (pA)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Average F-I curve   (mean ± SEM)"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 8))
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    fig.tight_layout()
    save_fig(fig, path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_fi_individual(fi: pd.DataFrame, groups: list[str], path: Path) -> None:
    """Every cell's F-I curve: one panel per group, then all groups overlaid,
    each with its group mean drawn bold on top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = [("-", "black", "#555555"), ("--", "#6e6e6e", "#9a9a9a")]
    panels = [[g] for g in groups] + [list(groups)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6),
                             sharey=True)
    for ax, members in zip(np.atleast_1d(axes), panels):
        for g in members:
            k = groups.index(g)
            ls, mean_col, cell_col = styles[k % len(styles)]
            sub = fi[fi["group"] == g]
            n_cells = 0
            for _cell, d in sub.groupby(["experiment", "folder_name"]):
                d = d.sort_values("stim_pA")
                ax.plot(d["stim_pA"], d["rate_Hz"], ls=ls, color=cell_col, lw=0.7,
                        alpha=0.75, zorder=1)
                n_cells += 1
            m = sub.groupby("stim_pA")["rate_Hz"].mean()
            ax.plot(m.index, m.to_numpy(), ls=ls, color=mean_col, lw=2.6, zorder=3,
                    label=f"{g} mean  ({n_cells} cells)")
        ax.set_xlabel("Injected current (pA)")
        ax.set_title(" and ".join(members) if len(members) > 1 else
                     f"{members[0]}: every cell", fontsize=10)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_position(("outward", 6))
    np.atleast_1d(axes)[0].set_ylabel("Firing rate (Hz)")
    fig.suptitle("Individual F-I curves   (thin = one cell; bold = group mean)",
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, path, dpi=150, facecolor="white")
    plt.close(fig)


def pick_block_examples(cells: pd.DataFrame, groups: list[str]) -> list[tuple]:
    """(group, kind, row) for one sustained-firing and one blocking cell per
    group. Sustained: still firing at the end of the top step (duration >=
    0.95) with the smallest drop in APs; blocking: firing stopped within the
    first 20% of the top step. Of the candidates, the cell whose peak number
    of APs is closest to its group's median is taken, so the example is a
    typical cell of its kind rather than the most extreme one."""
    out = []
    for g in groups:
        sub = cells[(cells["group"] == g) & cells["top_step_firing_frac"].notna()]
        if sub.empty:
            continue
        med = sub["max_aps"].median()
        for kind, cand in (
                ("sustained", sub[(sub["top_step_firing_frac"] >= 0.95)
                                  & (sub["block_drop_pct"] <= 20)]),
                ("depolarization block", sub[sub["top_step_firing_frac"] <= 0.2])):
            if cand.empty:
                continue
            row = cand.loc[(cand["max_aps"] - med).abs().idxmin()]
            out.append((g, kind, row))
    return out


def plot_block_examples(cells: pd.DataFrame, groups: list[str], path: Path) -> None:
    """Example traces: the top current step (black) of a sustained-firing and a
    blocking cell per group, with the same cell's peak-firing step (grey)
    drawn above it for comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import batch_cell_analysis as B

    examples = pick_block_examples(cells, groups)
    if not examples:
        return
    kinds = ["sustained", "depolarization block"]
    fig, axes = plt.subplots(len(groups), 2, figsize=(13, 3.6 * len(groups)),
                             squeeze=False, sharex=True)
    for ax in axes.ravel():
        ax.axis("off")
    for g, kind, row in examples:
        ax = axes[groups.index(g), kinds.index(kind)]
        ax.axis("on")
        path_abf = Path(row["_summary"]).parent / str(row["folder_name"]) / str(row["file_steps"])
        try:
            sweeps, _meta = B.load_cc_sweeps(path_abf)
        except Exception as exc:                  # noqa: BLE001 - leave the panel empty
            ax.text(0.5, 0.5, f"could not read {path_abf.name}: {exc}",
                    transform=ax.transAxes, ha="center", fontsize=8)
            continue
        by_stim = sorted(sweeps, key=lambda s: s["stim_pA"])
        top = by_stim[-1]
        peak = min(by_stim, key=lambda s: abs(s["stim_pA"] - row["i_max_aps_pA"]))
        t0 = top["start"]
        offset = 130.0                            # mV, peak sweep drawn above
        win = lambda s: (s["t"] >= t0 - 0.1) & (s["t"] <= s["end"] + 0.2)
        mp, mt = win(peak), win(top)
        ax.plot((peak["t"][mp] - t0) * 1e3, peak["v"][mp] + offset, color="#9a9a9a",
                lw=0.5, label=f"{peak['stim_pA']:.0f} pA: {row['max_aps']:.0f} APs (peak)")
        ax.plot((top["t"][mt] - t0) * 1e3, top["v"][mt], color="black", lw=0.5,
                label=f"{top['stim_pA']:.0f} pA (top step)")
        dur_ms = (top["end"] - top["start"]) * 1e3
        ax.plot([0, dur_ms], [-105, -105], color="black", lw=3, solid_capstyle="butt")
        frac = row["top_step_firing_frac"]
        if np.isfinite(frac) and frac > 0:
            ax.axvline(frac * dur_ms, color="black", ls=":", lw=1)
        ax.set_title(f"{g}, {kind}:  {row['folder_name']} ({row['experiment']})\n"
                     f"top-step firing duration {frac:.2f},  drop in APs "
                     f"{row['block_drop_pct']:.0f}%", fontsize=9)
        ax.set_ylabel("Vm (mV)")
        ax.set_ylim(-115, offset + 60)
        ax.legend(loc="upper right", fontsize=7.5, frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[-1]:
        if ax.axison:
            ax.set_xlabel("Time from step onset (ms)")
    fig.suptitle("Top-step firing duration: example cells   (black = highest current "
                 "step, bar = step; grey = the same cell's peak-firing step, offset "
                 "+130 mV; dotted = last spike)", fontsize=9.5)
    fig.tight_layout()
    save_fig(fig, path, dpi=200, facecolor="white")
    plt.close(fig)


def ols(y: np.ndarray, X: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Ordinary least squares with an intercept: coefficient, SE, t, p per
    term, plus n and R^2 on every row. numpy/scipy only, so the script does
    not need statsmodels."""
    from scipy import stats
    X = np.column_stack([np.ones(len(y)), X])
    names = ["intercept"] + names
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = beta / se
    r2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)
    return pd.DataFrame({"term": names, "coef": beta, "se": se, "t": t,
                         "p": 2 * stats.t.sf(np.abs(t), dof), "n": len(y), "r2": r2})


def rheobase_models(cells: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Does RMP (or input resistance) account for the rheobase difference?

    Linear models of rheobase on genotype (the second group coded 1), alone
    and adjusted for RMP, for RMP and input resistance, and with a
    genotype x RMP interaction -- the last asks whether RMP's effect on
    rheobase differs between the genotypes. Fitted on the cells that have
    every value the largest model needs, so all models use the same cells.
    """
    if len(groups) != 2:
        return pd.DataFrame()
    d = cells[["group", "rheobase_pA", "rmp_folder_mV", "vc_Rin_MOhm"]].copy()
    for c in ("rheobase_pA", "rmp_folder_mV", "vc_Rin_MOhm"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    y = d["rheobase_pA"].to_numpy(float)
    geno = (d["group"] == groups[1]).to_numpy(float)
    rmp = d["rmp_folder_mV"].to_numpy(float)
    rin = d["vc_Rin_MOhm"].to_numpy(float)
    gname = f"genotype ({groups[1]} vs {groups[0]})"
    models = [
        ("rheobase ~ genotype", np.column_stack([geno]), [gname]),
        ("rheobase ~ genotype + RMP", np.column_stack([geno, rmp]), [gname, "RMP (mV)"]),
        ("rheobase ~ genotype + RMP + Rin", np.column_stack([geno, rmp, rin]),
         [gname, "RMP (mV)", "Rin (MOhm)"]),
        ("rheobase ~ genotype * RMP", np.column_stack([geno, rmp, geno * rmp]),
         [gname, "RMP (mV)", "genotype x RMP"]),
    ]
    frames = []
    for name, X, names in models:
        f = ols(y, X, names)
        f.insert(0, "model", name)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def plot_rheobase_vs_rmp(cells: pd.DataFrame, groups: list[str], path: Path) -> pd.DataFrame:
    """Rheobase against RMP, input resistance, and the distance from RMP to
    AP threshold, one point per cell, a least-squares line per group, and
    Spearman's correlation per group and pooled. Returns the correlations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    d = cells.copy()
    for c in ("rheobase_pA", "rmp_folder_mV", "vc_Rin_MOhm", "AP_threshold_mV"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["thr_minus_rmp_mV"] = d["AP_threshold_mV"] - d["rmp_folder_mV"]
    panels = [("rmp_folder_mV", "RMP from folder name (mV)"),
              ("vc_Rin_MOhm", "Input resistance, VC (MOhm)"),
              ("thr_minus_rmp_mV", "AP threshold - RMP (mV)")]
    styles = [("o", "black", "-"), ("o", "white", "--")]
    rows = []
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), sharey=True)
    for ax, (col, xlabel) in zip(axes, panels):
        for k, g in enumerate(groups):
            marker, face, ls = styles[k % len(styles)]
            sub = d.loc[d["group"] == g, [col, "rheobase_pA"]].dropna()
            x, y = sub[col].to_numpy(float), sub["rheobase_pA"].to_numpy(float)
            rho, pr = stats.spearmanr(x, y) if len(x) > 2 else (np.nan, np.nan)
            rows.append({"x": col, "group": g, "n": len(x), "spearman_r": rho, "p": pr})
            ax.scatter(x, y, s=30, marker=marker, facecolor=face, edgecolor="black",
                       lw=0.9, zorder=3,
                       label=f"{g}  r = {rho:.2f}, {p_text(pr)}  (n={len(x)})")
            if len(x) > 2 and np.ptp(x) > 0:
                slope, icpt = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, slope * xs + icpt, ls=ls, color="black", lw=1.4, zorder=2)
        both = d[[col, "rheobase_pA"]].dropna()
        rho, pr = stats.spearmanr(both[col], both["rheobase_pA"])
        rows.append({"x": col, "group": "all", "n": len(both), "spearman_r": rho, "p": pr})
        ax.set_title(f"all cells: r = {rho:.2f}, {p_text(pr)}", fontsize=9.5,
                     fontweight="bold" if pr < 0.05 else "normal")
        ax.set_xlabel(xlabel)
        # below the axes: the points fill the whole panel
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8,
                  frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_position(("outward", 6))
    axes[0].set_ylabel("Rheobase (pA)")
    fig.suptitle("What sets rheobase?   (one point per cell; lines = least-squares "
                 "fit per group; r = Spearman)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, path, dpi=150, facecolor="white")
    plt.close(fig)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-cell firing measurements
# ---------------------------------------------------------------------------
def firing_stats(cells: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    category = {c: cat for cat, measures, *_rest in MEASURE_SETS for c, _l, _u in measures}
    for col, label, unit in FIRING_MEASURES:
        if col not in cells:
            continue
        d = cells[["group", col]].rename(columns={col: "x"})
        d["x"] = pd.to_numeric(d["x"], errors="coerce")
        d = d.dropna()
        row = {"category": category.get(col, ""), "measure": col, "label": label,
               "unit": unit}
        for g in groups:
            v = d.loc[d["group"] == g, "x"].to_numpy(float)
            row[f"{g}_mean"] = v.mean() if v.size else np.nan
            row[f"{g}_sem"] = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
            row[f"{g}_n"] = int(v.size)
        row["test"], row["p"] = cs.group_test(d, "x")
        rows.append(row)
    return pd.DataFrame(rows)


def box_points(ax, values: list[np.ndarray], labels: list[str], seed: int = 0) -> None:
    """R-style box plot in black and white, every cell drawn on top.

    White boxes with a thick median, dashed whiskers to the furthest point
    within 1.5 IQR with capped ends, and one jittered black dot per cell
    (so no separate outlier markers). Axes are detached from the data, as in
    base R: no top/right frame, left and bottom spines offset outward.
    """
    rng = np.random.default_rng(seed)
    pos = np.arange(1, len(values) + 1)
    data = [v if v.size else np.array([np.nan]) for v in values]
    ax.boxplot(data, positions=pos, widths=0.6, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.0),
               medianprops=dict(color="black", linewidth=3.0),
               whiskerprops=dict(color="black", linewidth=1.0, linestyle=(0, (4, 3))),
               capprops=dict(color="black", linewidth=1.0),
               capwidths=0.4)
    for x, v in zip(pos, values):
        if v.size:
            ax.scatter(x + rng.uniform(-0.12, 0.12, v.size), v, s=22, color="black",
                       zorder=3, linewidths=0)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{l}\n(n={v.size})" for l, v in zip(labels, values)])
    ax.set_xlim(0.4, len(values) + 0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 8))
    ax.spines["bottom"].set_bounds(pos[0], pos[-1])
    ax.tick_params(colors="black")


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p = n/a"
    return f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"


def box_grid(panels: list[tuple[str, list[np.ndarray], float]], groups: list[str],
             title: str, path: Path, ncol: int = 4) -> None:
    """One box_points panel per (y label, values per group, p)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.4 * nrow), squeeze=False)
    for k, (ax, (ylabel, values, p)) in enumerate(zip(axes.ravel(), panels)):
        box_points(ax, values, groups, seed=k)
        ax.set_ylabel(ylabel)
        ax.set_title(p_text(p), fontsize=9,
                     fontweight="bold" if np.isfinite(p) and p < 0.05 else "normal")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save_fig(fig, path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_firing(cells: pd.DataFrame, stats: pd.DataFrame, groups: list[str],
                path: Path, title: str, ncol: int = 4, note: str = "") -> None:
    panels = []
    for st in stats.itertuples():
        values = [pd.to_numeric(cells.loc[cells["group"] == g, st.measure],
                                errors="coerce").dropna().to_numpy(float) for g in groups]
        panels.append((st.label + (f" ({st.unit})" if st.unit else ""), values, st.p))
    box_grid(panels, groups, f"{title}, one point per cell   "
             "(Mann-Whitney, uncorrected; bold = p < 0.05)"
             + (f"\n{note}" if note else ""), path, ncol=ncol)


def sepsc_box_panels(sc: pd.DataFrame, cutoff: float,
                     groups: list[str]) -> list[tuple[str, list[np.ndarray], float]]:
    """(y label, values per group, p) for rate / amplitude / IEI above the
    common cutoff, plus the noise."""
    inc = sc[sc["excluded"] == ""]
    panels = []
    for col, label in (("rate_Hz", f"Frequency, events ≥ {cutoff:g} pA (Hz)"),
                       ("amp_median_pA", "Median amplitude (pA)"),
                       ("iei_median_ms", "Median inter-event interval (ms)"),
                       ("noise_pA", "Noise SD (pA)")):
        if col not in inc:
            continue
        values = [pd.to_numeric(inc.loc[inc["group"] == g, col], errors="coerce")
                  .dropna().to_numpy(float) for g in groups]
        panels.append((label, values, cs.group_test(inc, col)[1]))
    return panels


def draw_box_panel(ax, label: str, values: list[np.ndarray], p: float,
                   groups: list[str], seed: int = 0) -> None:
    """One box_points panel with its y label and p value."""
    box_points(ax, values, groups, seed=seed)
    ax.set_ylabel(label)
    ax.set_title(p_text(p), fontsize=9,
                 fontweight="bold" if np.isfinite(p) and p < 0.05 else "normal")


def plot_sepsc_box(sc: pd.DataFrame, cutoff: float, groups: list[str], path: Path) -> None:
    box_grid(sepsc_box_panels(sc, cutoff, groups), groups,
             f"sEPSCs, one point per cell   (events ≥ {cutoff:g} pA; "
             "Mann-Whitney; bold = p < 0.05)", path)


# ---------------------------------------------------------------------------
# sEPSC amplitude distributions
# ---------------------------------------------------------------------------
#: Per-cell amplitude percentiles compared between groups.
PERCENTILES = (10, 25, 50, 75, 90)


def cell_amplitudes(sc: pd.DataFrame, cutoff: float) -> list[tuple[str, np.ndarray]]:
    """(group, sorted amplitudes >= cutoff) for every included cell."""
    out = []
    for _, r in sc[sc["excluded"] == ""].iterrows():
        a = np.asarray(r["_amps"], float)
        a = np.sort(a[np.isfinite(a) & (a >= cutoff)])
        if a.size:
            out.append((r["group"], a))
    return out


def mean_cdf(amps: list[np.ndarray], grid: np.ndarray) -> np.ndarray:
    """Each cell's own CDF on `grid`, averaged: every cell weighs the same."""
    return np.mean([np.searchsorted(a, grid, side="right") / a.size for a in amps], axis=0)


def amplitude_distribution(sc: pd.DataFrame, cutoff: float, groups: list[str],
                           n_per_cell: int, n_perm: int = 10000,
                           n_sub: int = 1000, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Tests for a difference in the amplitude DISTRIBUTION between two groups.

    Cell-level (cells are the independent unit):
      * permutation test on the KS distance between the groups' mean CDFs,
        shuffling the group labels across cells;
      * Mann-Whitney on each cell's 10/25/50/75/90th amplitude percentile.
    Pooled events, equal N per cell (the conventional analysis):
      * n_per_cell random events from every cell with at least that many,
        pooled per group, then two-sample KS and k-sample Anderson-Darling,
        repeated over n_sub random subsamples. Events are not independent
        (they share a cell), so these p values are far too small; they are
        reported because reviewers ask for them, not as the result.
    """
    from scipy import stats

    cells = cell_amplitudes(sc, cutoff)
    rows: list[dict] = []
    extra: dict = {"cells": cells, "n_per_cell": n_per_cell}
    if len(groups) != 2:
        return pd.DataFrame([{"test": "distribution tests need exactly 2 groups"}]), extra
    ga, gb = groups
    labels = np.array([g for g, _a in cells])
    amps = [a for _g, a in cells]
    if not ((labels == ga).any() and (labels == gb).any()):
        return pd.DataFrame([{"test": "a group has no sEPSC cells"}]), extra
    hi = np.nanpercentile(np.concatenate(amps), 99.5)
    grid = np.linspace(cutoff, max(cutoff + 1, hi), 400)
    extra["grid"] = grid

    def ks_of(lab):
        ca = mean_cdf([a for a, l in zip(amps, lab) if l == ga], grid)
        cb = mean_cdf([a for a, l in zip(amps, lab) if l == gb], grid)
        return float(np.max(np.abs(ca - cb)))

    rng = np.random.default_rng(seed)
    d_obs = ks_of(labels)
    perm = np.array([ks_of(rng.permutation(labels)) for _ in range(n_perm)])
    rows.append({"test": "permutation, KS distance between mean-of-cells CDFs",
                 "unit": "cell", "n": f"{(labels == ga).sum()} vs {(labels == gb).sum()} cells",
                 "statistic": d_obs,
                 "p": (1 + np.sum(perm >= d_obs)) / (n_perm + 1)})

    for q in PERCENTILES:
        per_cell = np.array([np.percentile(a, q) for a in amps])
        va, vb = per_cell[labels == ga], per_cell[labels == gb]
        res = stats.mannwhitneyu(va, vb, alternative="two-sided")
        rows.append({"test": f"Mann-Whitney, per-cell {q}th percentile",
                     "unit": "cell", "n": f"{va.size} vs {vb.size} cells",
                     "statistic": float(res.statistic), "p": float(res.pvalue),
                     f"{ga}_mean": va.mean(), f"{gb}_mean": vb.mean()})
        extra.setdefault("percentiles", {})[q] = (va, vb)

    big = [(l, a) for l, a in zip(labels, amps) if a.size >= n_per_cell]
    dropped = len(amps) - len(big)
    extra["dropped"] = dropped
    na = sum(l == ga for l, _a in big)
    nb = sum(l == gb for l, _a in big)
    if na and nb:
        ks_d, ks_p, ad_p = [], [], []
        for _ in range(n_sub):
            pa = np.concatenate([rng.choice(a, n_per_cell, replace=False)
                                 for l, a in big if l == ga])
            pb = np.concatenate([rng.choice(a, n_per_cell, replace=False)
                                 for l, a in big if l == gb])
            k = stats.ks_2samp(pa, pb)
            ks_d.append(k.statistic)
            ks_p.append(k.pvalue)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                # anderson_ksamp warns whenever p is outside its 0.001-0.25 table
                warnings.simplefilter("ignore")
                ad_p.append(stats.anderson_ksamp([pa, pb]).significance_level)
        extra["pooled_example"] = (pa, pb)
        note = (f"{n_per_cell} random events per cell, {na} vs {nb} cells"
                + (f" ({dropped} cell(s) with fewer events left out)" if dropped else ""))
        rows.append({"test": "KS two-sample, pooled equal-N events", "unit": "event",
                     "n": note, "statistic": float(np.median(ks_d)),
                     "p": float(np.median(ks_p)),
                     "frac_p_below_0.05": float(np.mean(np.array(ks_p) < 0.05))})
        rows.append({"test": "Anderson-Darling k-sample, pooled equal-N events",
                     "unit": "event", "n": note, "statistic": np.nan,
                     "p": float(np.median(ad_p)),
                     "frac_p_below_0.05": float(np.mean(np.array(ad_p) < 0.05))})
    return pd.DataFrame(rows), extra


def plot_amplitude_distribution(extra: dict, tests: pd.DataFrame, groups: list[str],
                                cutoff: float, path: Path) -> None:
    """Black and white: mean-of-cells CDF, pooled equal-N CDF, percentiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "grid" not in extra:
        return
    styles = {groups[0]: ("-", "black"), groups[1]: ("--", "white")}
    grid, cells = extra["grid"], extra["cells"]

    def p_of(prefix):
        hit = tests[tests["test"].str.startswith(prefix)]
        return float(hit["p"].iloc[0]) if len(hit) else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for g in groups:
        ls, _face = styles[g]
        mine = [a for l, a in cells if l == g]
        for a in mine:
            ax.plot(grid, np.searchsorted(a, grid, side="right") / a.size, ls=ls,
                    color="#bbbbbb", lw=0.6, zorder=1)
        ax.plot(grid, mean_cdf(mine, grid), ls=ls, color="black", lw=2.2, zorder=3,
                label=f"{g}  (mean of {len(mine)} cells)")
    ax.set_title(f"Mean of each cell's CDF   (permutation {p_text(p_of('permutation'))})",
                 fontsize=9.5)
    ax.set_ylabel("Cumulative fraction")

    ax = axes[1]
    if "pooled_example" in extra:
        for g, pooled in zip(groups, extra["pooled_example"]):
            ls, _face = styles[g]
            s = np.sort(pooled)
            ax.plot(s, np.arange(1, s.size + 1) / s.size, ls=ls, color="black", lw=2.0,
                    label=f"{g}  ({s.size} events)")
        ax.set_title(f"Pooled, {extra['n_per_cell']} events per cell   "
                     f"(KS median {p_text(p_of('KS'))}; events not independent)",
                     fontsize=9.5)
    for a in axes[:2]:
        a.set_xlabel("Amplitude (pA)")
        a.set_xlim(cutoff, grid[-1])
        a.legend(fontsize=8.5, loc="lower right", frameon=False)

    ax = axes[2]
    pcts = extra.get("percentiles", {})
    qs = list(pcts)
    for k, g in enumerate(groups):
        ls, face = styles[g]
        mean = [pcts[q][k].mean() for q in qs]
        sem = [pcts[q][k].std(ddof=1) / np.sqrt(pcts[q][k].size) for q in qs]
        ax.errorbar(np.array(qs) + (k - 0.5) * 1.2, mean, yerr=sem, ls=ls, color="black",
                    marker="o", mfc=face, mec="black", ms=6, capsize=3, lw=1.4, label=g)
    ymax = ax.get_ylim()[1]
    for q in qs:
        p = p_of(f"Mann-Whitney, per-cell {q}th")
        ax.annotate(p_text(p).replace("p = ", "p="), (q, ymax), ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold" if p < 0.05 else "normal")
    ax.set_xticks(qs)
    ax.set_xlabel("Percentile of each cell's amplitudes")
    ax.set_ylabel("Amplitude (pA)")
    ax.set_title("Per-cell percentiles, mean ± SEM (Mann-Whitney)", fontsize=9.5, pad=16)
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_position(("outward", 6))
    fig.suptitle(f"sEPSC amplitude distributions, events ≥ {cutoff:g} pA", fontsize=10.5)
    fig.tight_layout()
    save_fig(fig, path, dpi=150, facecolor="white")
    plt.close(fig)


def _inset_bars(ax, values: list[np.ndarray], groups: list[str], ylabel: str,
                p: float, seed: int = 0) -> None:
    """Bar = mean, error bar = SEM, grey dot per cell; first group filled black."""
    rng = np.random.default_rng(seed)
    for k, (g, v) in enumerate(zip(groups, values)):
        if not v.size:
            continue
        face = "black" if k == 0 else "white"
        sem = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
        ax.bar(k, v.mean(), width=0.62, color=face, edgecolor="black", lw=1.0, zorder=1)
        ax.errorbar(k, v.mean(), yerr=sem, fmt="none", ecolor="black", capsize=3,
                    elinewidth=1.0, zorder=3)
        ax.scatter(k + rng.uniform(-0.15, 0.15, v.size), v, s=7, color="#8a8a8a",
                   zorder=2, linewidths=0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_title(p_text(p), fontsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def per_cell_events(sc: pd.DataFrame, cutoff: float,
                    groups: list[str]) -> dict[str, dict[str, list[np.ndarray]]]:
    """{group: {"amps": [...], "ieis": [...]}}, one array per included cell,
    events >= cutoff only, in time order; IEIs in ms between those events."""
    per = {g: {"amps": [], "ieis": []} for g in groups}
    for _, r in sc[sc["excluded"] == ""].iterrows():
        if r["group"] not in per:
            continue
        a = np.asarray(r["_amps"], float)
        t = np.asarray(r["_times"], float)
        keep = np.isfinite(a) & np.isfinite(t) & (a >= cutoff)
        order = np.argsort(t[keep])
        per[r["group"]]["amps"].append(a[keep][order])
        per[r["group"]]["ieis"].append(np.diff(t[keep][order]) * 1e3)
    return per


def cell_level_p(tests: pd.DataFrame, prefix: str) -> float:
    """p of the first distribution test whose name starts with `prefix`."""
    if "test" not in tests:
        return np.nan
    hit = tests[tests["test"].astype(str).str.startswith(prefix)]
    return float(hit["p"].iloc[0]) if len(hit) else np.nan


def draw_cumulative(ax, per: dict, key: str, groups: list[str], cutoff: float,
                    n_per_cell: int, rng, p_main: float = np.nan) -> None:
    """Cumulative probability of `key` ("amps" or "ieis"): n_per_cell random
    events from every cell that has that many, pooled per group."""
    styles = ["-", "--"]
    xmax = 0.0
    for k, g in enumerate(groups):
        arrs = per[g][key]
        big = [x for x in arrs if x.size >= n_per_cell]
        if not big:
            continue
        v = np.sort(np.concatenate([rng.choice(x, n_per_cell, replace=False) for x in big]))
        ax.plot(v, np.arange(1, v.size + 1) / v.size, ls=styles[k % 2], color="black",
                lw=2.0, label=f"{g}  ({len(big)} cells)")
        xmax = max(xmax, np.percentile(v, 99))
    ax.set_xlim(cutoff if key == "amps" else 0, xmax or 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Amplitude (pA)" if key == "amps" else "Inter-event interval (ms)")
    ax.set_ylabel("Cumulative probability")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if np.isfinite(p_main):
        ax.set_title(f"cells as unit: permutation {p_text(p_main)}", fontsize=8.5)


def draw_histogram(ax, per: dict, groups: list[str], cutoff: float,
                   bin_width: float) -> None:
    """Amplitude histogram: each cell's events as % of its own total per bin,
    averaged across cells (mean +- SEM), so every cell counts equally."""
    faces = ["black", "white"]
    all_amps = np.concatenate([x for g in groups for x in per[g]["amps"]])
    top = np.ceil(np.percentile(all_amps, 97) / bin_width) * bin_width
    edges = np.arange(cutoff, top + bin_width, bin_width)
    centres = edges[:-1] + bin_width / 2
    w = bin_width * 0.42
    for k, g in enumerate(groups):
        pct = np.array([np.histogram(x, edges)[0] / x.size * 100 for x in per[g]["amps"]
                        if x.size])
        if not len(pct):
            continue
        mean = pct.mean(axis=0)
        sem = pct.std(axis=0, ddof=1) / np.sqrt(len(pct)) if len(pct) > 1 else 0 * mean
        x = centres + (k - 0.5) * w
        ax.bar(x, mean, width=w, color=faces[k % 2], edgecolor="black", lw=0.7,
               label=f"{g}  ({len(pct)} cells)")
        ax.errorbar(x, mean, yerr=sem, fmt="none", ecolor="black", elinewidth=0.6,
                    capsize=1.2)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(f"Amplitude (pA), {bin_width:g} pA bins")
    ax.set_ylabel("Events (% of each cell's total)")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.set_title("mean ± SEM across cells", fontsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_paper_figure(sc: pd.DataFrame, cutoff: float, groups: list[str], path: Path,
                      tests: pd.DataFrame, n_per_cell: int = 100,
                      bin_width: float = 10.0, seed: int = 0) -> None:
    """The figure sEPSC papers usually show, in black and white.

    A  cumulative probability of amplitude, n_per_cell random events per cell
       pooled per group (the convention), with an inset of per-cell median
       amplitude. B  the same for inter-event interval, inset per-cell
       frequency. C  amplitude histogram in `bin_width` pA bins: each cell's
       events as % of that cell's total, averaged across cells (mean +- SEM),
       so every cell counts equally.
    The p values printed are the cell-level ones (permutation on the CDFs,
    Mann-Whitney on the inset values) -- the pooled curves are for display.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(groups) != 2:
        return
    inc = sc[sc["excluded"] == ""]
    rng = np.random.default_rng(seed)
    per = per_cell_events(sc, cutoff, groups)

    fig = plt.figure(figsize=(14, 4.4))
    gsp = fig.add_gridspec(1, 3, wspace=0.32)
    for col, (key, inset_col, inset_label, p_main) in enumerate((
            ("amps", "amp_median_pA", "Median amp. (pA)", cell_level_p(tests, "permutation")),
            ("ieis", "rate_Hz", "Frequency (Hz)", np.nan))):
        ax = fig.add_subplot(gsp[0, col])
        draw_cumulative(ax, per, key, groups, cutoff, n_per_cell, rng, p_main)
        ins = ax.inset_axes([0.55, 0.30, 0.38, 0.42])
        vals = [pd.to_numeric(inc.loc[inc["group"] == g, inset_col], errors="coerce")
                .dropna().to_numpy(float) for g in groups]
        _inset_bars(ins, vals, groups, inset_label, cs.group_test(inc, inset_col)[1], seed=col)
    ax = fig.add_subplot(gsp[0, 2])
    draw_histogram(ax, per, groups, cutoff, bin_width)

    fig.suptitle(f"sEPSCs, events ≥ {cutoff:g} pA   (cumulative curves: "
                 f"{n_per_cell} random events per cell; insets: one point per cell, "
                 f"Mann-Whitney)", fontsize=9.5, y=1.02)
    save_fig(fig, path, dpi=200, facecolor="white", bbox_inches="tight")
    save_fig(fig, path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


#: An event is averaged only if no other event peaks this close to it (ms
#: before, ms after): a neighbour in the window distorts the rise or the decay.
ISOLATION_MS = (5.0, 25.0)


def _kinetics(t_ms: np.ndarray, w: np.ndarray) -> dict:
    """10-90% rise time and single-exponential decay tau of a peak-aligned,
    inward (negative) average waveform. NaN where it cannot be measured."""
    from scipy.optimize import curve_fit

    y = -np.asarray(w, float)                     # upward-going for the maths
    pk = int(np.argmax(y))
    amp = y[pk]
    out = {"rise_10_90_ms": np.nan, "decay_tau_ms": np.nan, "peak_pA": -amp}
    if not np.isfinite(amp) or amp <= 0:
        return out
    pre = y[:pk + 1]
    try:
        i10 = np.nonzero(pre < 0.1 * amp)[0][-1]
        i90 = np.nonzero(pre < 0.9 * amp)[0][-1]
        # linear interpolation to the exact crossings
        x10 = np.interp(0.1 * amp, pre[i10:i10 + 2], t_ms[i10:i10 + 2])
        x90 = np.interp(0.9 * amp, pre[i90:i90 + 2], t_ms[i90:i90 + 2])
        out["rise_10_90_ms"] = float(x90 - x10)
    except IndexError:
        pass
    td, yd = t_ms[pk:] - t_ms[pk], y[pk:]
    try:
        p, _ = curve_fit(lambda t, a, tau, c: a * np.exp(-t / tau) + c, td, yd,
                         p0=[amp, 3.0, 0.0], bounds=([0, 0.05, -amp], [3 * amp, 100, amp]),
                         maxfev=10000)
        out["decay_tau_ms"] = float(p[1])
    except Exception:                        # noqa: BLE001 - no fit, no number
        pass
    return out


def average_events(sc: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Each included cell's average event (isolated events >= cutoff only),
    from its saved *_sEPSC_waveforms.npz, plus that average's kinetics."""
    rows = []
    for _, r in sc[sc["excluded"] == ""].iterrows():
        ev_path = Path(str(r.get("events_file", "")))
        npz = ev_path.with_name(ev_path.name.replace("_sEPSC_events.csv",
                                                     "_sEPSC_waveforms.npz"))
        if not npz.is_file():
            continue
        z = np.load(npz)
        wav, t_ms = z["waveforms"].astype(float), z["t_ms"]
        amp, peak_t = np.abs(z["amplitude"]), z["peak_time_s"]
        # isolation is judged against EVERY detected event, not just the saved ones
        all_t = np.sort(np.asarray(r["_times"], float))
        pre_s, post_s = ISOLATION_MS[0] / 1e3, ISOLATION_MS[1] / 1e3
        lo = np.searchsorted(all_t, peak_t - pre_s, side="left")
        hi = np.searchsorted(all_t, peak_t + post_s, side="right")
        isolated = (hi - lo) <= 1                # only the event itself
        keep = isolated & (amp >= cutoff) & np.isfinite(amp)
        if keep.sum() < 5:
            continue
        mean_w = wav[keep].mean(axis=0)
        rows.append({"group": r["group"], "experiment": r["experiment"],
                     "folder_name": r["folder_name"], "n_waveforms_saved": len(amp),
                     "n_averaged": int(keep.sum()), "t_ms": t_ms, "mean_w": mean_w,
                     **_kinetics(t_ms, mean_w)})
    return pd.DataFrame(rows)


def draw_average_trace(ax, avg: pd.DataFrame, groups: list[str], scaled: bool) -> None:
    """Average event per genotype (mean +- SEM across cells' averages); with
    `scaled`, each cell's average is first divided by its own peak."""
    styles = [("-", "#9a9a9a"), ("--", "#d0d0d0")]
    t_ms = avg["t_ms"].iloc[0]
    for k, g in enumerate(groups):
        ls, shade = styles[k % len(styles)]
        ws = np.stack(avg.loc[avg["group"] == g, "mean_w"].to_list())
        if scaled:
            ws = ws / np.abs(ws.min(axis=1, keepdims=True))
        m = ws.mean(axis=0)
        sem = ws.std(axis=0, ddof=1) / np.sqrt(len(ws)) if len(ws) > 1 else 0 * m
        ax.fill_between(t_ms, m - sem, m + sem, color=shade, alpha=0.6, lw=0, zorder=1)
        n_ev = int(avg.loc[avg["group"] == g, "n_averaged"].sum())
        ax.plot(t_ms, m, ls=ls, color="black", lw=1.8, zorder=2,
                label=f"{g}  ({len(ws)} cells, {n_ev} events)")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_xlabel("Time from peak (ms)")
    ax.set_ylabel("Normalised current (peak = -1)" if scaled else "Current (pA)")
    ax.set_title("Scaled to peak" if scaled else "Average event", fontsize=10)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def kinetics_panels(avg: pd.DataFrame, groups: list[str]
                    ) -> tuple[list[tuple[str, list[np.ndarray], float]], pd.DataFrame]:
    """Rise / decay of each cell's average: box panels and the statistics."""
    panels, rows = [], []
    for key, label in (("rise_10_90_ms", "10-90% rise time (ms)"),
                       ("decay_tau_ms", "Decay tau (ms)")):
        d = avg[["group", key]].rename(columns={key: "x"}).dropna()
        test, p = cs.group_test(d, "x")
        values = [d.loc[d["group"] == g, "x"].to_numpy(float) for g in groups]
        panels.append((label, values, p))
        row = {"measure": key, "test": test, "p": p}
        for g, v in zip(groups, values):
            row[f"{g}_mean"] = v.mean() if v.size else np.nan
            row[f"{g}_sem"] = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
            row[f"{g}_n"] = v.size
        rows.append(row)
    return panels, pd.DataFrame(rows)


def _average_event_caption(cutoff: float) -> str:
    return (f"isolated events ≥ {cutoff:g} pA (no other event {ISOLATION_MS[0]:g} ms "
            f"before / {ISOLATION_MS[1]:g} ms after), averaged per cell, then mean ± SEM "
            f"across cells")


def plot_average_events(avg: pd.DataFrame, groups: list[str], cutoff: float,
                        path: Path) -> pd.DataFrame:
    """A  average event per genotype, overlaid (mean +- SEM across cells).
    B  the same with every cell's average scaled to its own peak first, so
    the timecourse can be compared independent of amplitude. C, D  rise and
    decay of each cell's average. Returns the kinetics statistics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels, stats = kinetics_panels(avg, groups)
    fig = plt.figure(figsize=(15, 4.3))
    gsp = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.35, 0.7, 0.7], wspace=0.38)
    for col, scaled in enumerate((False, True)):
        ax = fig.add_subplot(gsp[0, col])
        draw_average_trace(ax, avg, groups, scaled)
    for col, (label, values, p) in enumerate(panels):
        ax = fig.add_subplot(gsp[0, 2 + col])
        draw_box_panel(ax, label, values, p, groups, seed=col)
    fig.suptitle("Average sEPSC per genotype: " + _average_event_caption(cutoff),
                 fontsize=9.5, y=1.03)
    save_fig(fig, path, dpi=200, facecolor="white", bbox_inches="tight")
    save_fig(fig, path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return stats


def plot_sepsc_complete(sc: pd.DataFrame, avg: pd.DataFrame, cutoff: float,
                        groups: list[str], path: Path, tests: pd.DataFrame,
                        n_per_cell: int = 100, bin_width: float = 10.0,
                        seed: int = 0) -> None:
    """One figure, three rows: the average event (raw, peak-scaled, rise,
    decay); the per-cell frequency / amplitude / IEI / noise box plots; and
    the cumulative amplitude and IEI distributions with the amplitude
    histogram. The bottom row is drawn by the same code, with the same random
    seed, as sepsc_figure, so the two figures show identical curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 13.5))
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.45)
    top = outer[0].subgridspec(1, 4, width_ratios=[1.35, 1.35, 0.7, 0.7], wspace=0.38)
    bottom = outer[1].subgridspec(1, 4, wspace=0.45)
    dist = outer[2].subgridspec(1, 3, wspace=0.32)

    for col, scaled in enumerate((False, True)):
        ax = fig.add_subplot(top[0, col])
        draw_average_trace(ax, avg, groups, scaled)
    kin, _stats = kinetics_panels(avg, groups)
    for col, (label, values, p) in enumerate(kin):
        ax = fig.add_subplot(top[0, 2 + col])
        draw_box_panel(ax, label, values, p, groups, seed=col)
    for col, (label, values, p) in enumerate(sepsc_box_panels(sc, cutoff, groups)):
        ax = fig.add_subplot(bottom[0, col])
        draw_box_panel(ax, label, values, p, groups, seed=10 + col)

    rng = np.random.default_rng(seed)
    per = per_cell_events(sc, cutoff, groups)
    for col, (key, p_main) in enumerate((("amps", cell_level_p(tests, "permutation")),
                                         ("ieis", np.nan))):
        ax = fig.add_subplot(dist[0, col])
        draw_cumulative(ax, per, key, groups, cutoff, n_per_cell, rng, p_main)
    ax = fig.add_subplot(dist[0, 2])
    draw_histogram(ax, per, groups, cutoff, bin_width)

    fig.suptitle(f"sEPSCs, {' vs '.join(groups)}   (top row: {_average_event_caption(cutoff)};  "
                 f"middle row: events ≥ {cutoff:g} pA, one point per cell, Mann-Whitney, "
                 f"bold = p < 0.05;\nbottom row: cumulative curves from {n_per_cell} random "
                 f"events per cell, pooled per group; histogram in {bin_width:g} pA bins, % of "
                 f"each cell's events, mean ± SEM across cells)", fontsize=9, y=0.995)
    save_fig(fig, path, dpi=200, facecolor="white", bbox_inches="tight")
    save_fig(fig, path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def run_sepsc(summaries: list[Path], mapping: dict[str, str], groups: list[str],
              out: Path, min_amp: Optional[float], max_limit: Optional[float],
              events_per_cell: int = 100, hist_bin: float = 10.0) -> None:
    sc = cs.load_cells(summaries, "genotype")
    if sc.empty:
        print("No cells with sEPSC results.\n")
        return
    sc["genotype"] = sc["group"]
    sc["group"] = sc["group"].str.strip().str.lower().map(mapping).fillna("")
    sc = sc[sc["group"] != ""].reset_index(drop=True)
    sc, cutoff = cs.mark_excluded(sc, max_limit, lambda inc: cs.choose_cutoff(inc, min_amp))
    sc = cs.measure(sc, cutoff)
    public = [c for c in sc.columns if not c.startswith("_")]
    sc[public].to_csv(out / "sepsc_group_cells.csv", index=False)
    summary = cs.summarise_groups(sc)
    summary.to_csv(out / "sepsc_group_summary.csv", index=False)
    sens = cs.cutoff_sensitivity(sc, cutoff)
    sens.to_csv(out / "sepsc_group_cutoffs.csv", index=False)
    cs.plot(sc, cutoff, "group", out / "sepsc_group_comparison.png")
    plot_sepsc_box(sc, cutoff, groups, out / "sepsc_boxplots.png")
    tests, extra = amplitude_distribution(sc, cutoff, groups, events_per_cell)
    tests.to_csv(out / "sepsc_amplitude_distribution.csv", index=False)
    plot_amplitude_distribution(extra, tests, groups, cutoff,
                                out / "sepsc_amplitude_distribution.png")
    plot_paper_figure(sc, cutoff, groups, out / "sepsc_figure.png", tests,
                      n_per_cell=events_per_cell, bin_width=hist_bin)
    avg = average_events(sc, cutoff)
    if not avg.empty:
        kin = plot_average_events(avg, groups, cutoff, out / "sepsc_average_event.png")
        plot_sepsc_complete(sc, avg, cutoff, groups, out / "sepsc_complete_figure.png",
                            tests, n_per_cell=events_per_cell, bin_width=hist_bin)
        avg.drop(columns=["t_ms", "mean_w"]).to_csv(out / "sepsc_average_event_cells.csv",
                                                     index=False)
        kin.to_csv(out / "sepsc_kinetics_stats.csv", index=False)

    inc = sc[sc["excluded"] == ""]
    print(f"sEPSCs: {len(inc)} cells compared, common amplitude cutoff {cutoff:g} pA"
          + ("" if min_amp is not None else " (largest detection limit, rounded up)"))
    for _, r in sc[sc["excluded"] != ""].iterrows():
        print(f"  excluded {r['experiment']}/{r['folder_name']}: {r['excluded']}")
    with pd.option_context("display.width", 200, "display.max_columns", 40,
                           "display.float_format", "{:.3g}".format):
        cols = [c for c in summary.columns
                if c in ("group", "n_cells", "n_animals") or c.endswith(("_mean", "_sem"))
                if not c.startswith(("rate_all", "amp_median_all"))]
        print(summary[cols].to_string(index=False))
        print()
        print(sens.to_string(index=False))
    noise_p = cs.group_test(inc, "noise_pA")[1]
    if np.isfinite(noise_p) and noise_p < 0.05:
        print(f"Noise differs between groups (p = {noise_p:.3g}): use the "
              f"cutoff-corrected numbers.")
    print()
    print(f"Amplitude distribution, events >= {cutoff:g} pA")
    for t in tests.itertuples():
        line = f"  {t.test:<54} {p_text(t.p):>11}   [{t.unit}: {t.n}]"
        if "frac_p_below_0.05" in tests and pd.notna(tests.loc[t.Index, "frac_p_below_0.05"]):
            line += (f"  p<0.05 in {100 * tests.loc[t.Index, 'frac_p_below_0.05']:.0f}% "
                     f"of subsamples")
        print(line)
    print("  Pooled-event p values treat events as independent and overstate "
          "significance; the cell-level tests are the ones to report.")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+", type=Path,
                   help="folders to search (recursively) for batch_cell_summary.csv")
    p.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS, metavar="GROUP=LABELS",
                   help='genotype labels per group, e.g. "WT=WT" "Het=Het,mutant" '
                        "(the default)")
    p.add_argument("--min-amp", type=float, default=None, metavar="PA",
                   help="sEPSC amplitude cutoff (default: largest detection limit)")
    p.add_argument("--max-limit", type=float, default=None, metavar="PA",
                   help="exclude sEPSC cells whose detection limit is above this")
    p.add_argument("--out", type=Path, default=None,
                   help="output folder (default <first folder>/group_comparison)")
    p.add_argument("--events-per-cell", type=int, default=100, metavar="N",
                   help="random events taken from each cell for the pooled KS / "
                        "Anderson-Darling tests; cells with fewer are left out (default 100)")
    p.add_argument("--hist-bin", type=float, default=10.0, metavar="PA",
                   help="amplitude histogram bin width in sepsc_figure (default 10 pA)")
    p.add_argument("--no-sepsc", action="store_true", help="firing only")
    args = p.parse_args(argv)

    mapping = parse_groups(args.groups)
    groups = group_order(args.groups)
    summaries = cs.find_summaries(args.folders)
    if not summaries:
        p.error("no batch_cell_summary.csv found")
    out = args.out or (args.folders[0] / "group_comparison")
    out.mkdir(parents=True, exist_ok=True)

    cells = load_summaries(summaries, mapping)
    exps = print_experiments(cells)
    exps.to_csv(out / "groups_experiments.csv", index=False)
    left_out = cells[cells["group"] == ""]
    if len(left_out):
        labels = sorted(set(left_out["genotype"]) or {"(blank)"})
        print(f"Left out: {len(left_out)} cell(s) with genotype "
              f"{', '.join(repr(l) for l in labels)} (no group in --groups)\n")
    cells = cells[cells["group"] != ""].reset_index(drop=True)
    vc = pd.DataFrame([first_membrane_test(r) for _, r in cells.iterrows()])
    cells = pd.concat([cells.drop(columns=[c for c in vc.columns if c in cells]), vc], axis=1)
    blk = pd.DataFrame([block_measures(r) for _, r in cells.iterrows()])
    cells = pd.concat([cells.drop(columns=[c for c in blk.columns if c in cells]), blk], axis=1)
    reg = pd.DataFrame([regularity_measures(r) for _, r in cells.iterrows()])
    cells = pd.concat([cells.drop(columns=[c for c in reg.columns if c in cells]), reg], axis=1)

    # ---- firing ----------------------------------------------------------
    fi = load_fi(cells)
    if fi.empty:
        print("No *_AP_sweeps.csv found; no F-I comparison.\n")
    else:
        protocols = fi.groupby(["experiment", "folder_name"])["stim_pA"].apply(tuple)
        if protocols.nunique() > 1:
            print(f"NOTE: {protocols.nunique()} different step protocols; the "
                  f"F-I area uses only currents every cell reached.\n")
        area = fi_area(fi).rename("fi_area").reset_index()
        cells = cells.merge(area, on=["experiment", "folder_name"], how="left")
        means = fi_means(fi, groups)
        fi.to_csv(out / "fi_curves_cells.csv", index=False)
        means.to_csv(out / "fi_curves_mean.csv", index=False)
        av = fi_anova(fi, groups) if len(groups) >= 2 else pd.DataFrame()
        if len(av):
            av.to_csv(out / "fi_anova.csv", index=False)
        plot_fi(fi, means, groups, out / "fi_curves.png",
                subtitle=anova_text(av) if len(av) else "")
        plot_fi_individual(fi, groups, out / "fi_curves_individual.png")

    stats = firing_stats(cells, groups)
    cells["animal"] = cells.apply(animal_id, axis=1)
    keep = ["experiment", "folder_name", "group", "genotype", "animal", "vc_file",
            "block_current_pA", "enters_block", "half_max_sweep", "half_max_n_spikes",
            "hm_n_isis", "rheobase_sweep", "nr_step_pA", "nr_n_isis"] + \
        [c for c, _l, _u in FIRING_MEASURES if c in cells]
    cells[keep].to_csv(out / "firing_cells.csv", index=False)
    stats.to_csv(out / "firing_stats.csv", index=False)
    block_counts, block_p = block_proportions(cells, groups)
    plot_block_examples(cells, groups, out / "block_examples.png")
    rheo_corr = plot_rheobase_vs_rmp(cells, groups, out / "rheobase_vs_rmp.png")
    rheo_models = rheobase_models(cells, groups)
    rheo_corr.to_csv(out / "rheobase_correlations.csv", index=False)
    rheo_models.to_csv(out / "rheobase_models.csv", index=False)
    block_txt = ("cells entering block (≤ "
                 f"{BLOCK_FRACTION:.0%} of peak APs at a higher current): "
                 + ", ".join(f"{g} {k}/{n}" for g, (k, n) in block_counts.items())
                 + f", Fisher {p_text(block_p)}")
    for cat, _measures, fname, title, ncol in MEASURE_SETS:
        sub = stats[stats["category"] == cat]
        if len(sub):
            note = (block_txt[0].upper() + block_txt[1:]
                    if cat == "Depolarization block" else "")
            plot_firing(cells, sub, groups, out / fname, title, ncol=ncol, note=note)
    stale = out / "firing_comparison.png"       # the single figure these replace
    if stale.is_file():
        stale.unlink()

    for g in groups:
        sub = cells[cells["group"] == g]
        print(f"{g}: {len(sub)} cells from {sub['animal'].nunique()} animals")
    print()
    print("Mean ± SEM, n cells; Mann-Whitney / Kruskal-Wallis, uncorrected")
    head = f"{'':<22}" + "".join(f"{g:>24}" for g in groups) + f"{'p':>9}"
    for cat, _measures, _fname, title, _ncol in MEASURE_SETS:
        sub = stats[stats["category"] == cat]
        if not len(sub):
            continue
        print(f"\n{title}")
        print(head)
        for _, st in sub.iterrows():
            cells_txt = "".join(f"{st[f'{g}_mean']:>12.3g} ± {st[f'{g}_sem']:<6.2g}"
                                f"(n={st[f'{g}_n']:<2d})" for g in groups)
            print(f"{st['label']:<22}{cells_txt}{st['p']:>9.3g}")
    print(f"\nDepolarization block: {block_txt}")
    if len(rheo_models):
        print("\nDoes RMP explain the rheobase difference?  (linear models; "
              f"n = {int(rheo_models['n'].iloc[0])} cells)")
        for model, f in rheo_models.groupby("model", sort=False):
            print(f"  {model}   (R² = {f['r2'].iloc[0]:.2f})")
            for _, t in f[f["term"] != "intercept"].iterrows():
                print(f"      {t['term']:<28} {t['coef']:+9.2f} ± {t['se']:<7.2f} "
                      f"{p_text(t['p'])}")
        print("  Spearman correlation with rheobase:")
        for _, r in rheo_corr.iterrows():
            print(f"      {r['x']:<18} {r['group']:<5} r = {r['spearman_r']:+.2f}  "
                  f"{p_text(r['p'])}  (n={r['n']})")
    if not fi.empty and len(av):
        lo, hi, k = av.attrs["currents"]
        ns = ", ".join(f"{g} {n}" for g, n in av.attrs["n_cells"].items())
        print(f"\nF-I curves, mixed ANOVA: genotype (between) x current (within), "
              f"{k} steps {lo:g}-{hi:g} pA; cells: {ns}")
        print(f"  {'effect':<42}{'df':>6}{'F':>9}{'p':>11}{'GG eps':>8}"
              f"{'p (GG)':>11}{'p (perm)':>11}")
        for _, r in av.iterrows():
            if not np.isfinite(r["F"]):
                print(f"  {r['effect']:<42}{r['df']:>6.0f}")
                continue
            eps_txt = f"{r['gg_epsilon']:.3f}" if np.isfinite(r["gg_epsilon"]) else ""
            print(f"  {r['effect']:<42}{r['df']:>6.0f}{r['F']:>9.2f}"
                  f"{p_text(r['p']):>11}{eps_txt:>8}{p_text(r['p_gg']):>11}"
                  f"{(p_text(r['p_perm']) if np.isfinite(r['p_perm']) else ''):>11}")
    if not fi.empty:
        sig = means[means["p_holm"] < 0.05]["stim_pA"].tolist()
        print(f"\nF-I curve: currents with p < 0.05 after Holm correction: "
              f"{', '.join(f'{s:g}' for s in sig) if sig else 'none'}")
    print()

    if not args.no_sepsc:
        run_sepsc(summaries, mapping, groups, out, args.min_amp, args.max_limit,
                  args.events_per_cell, args.hist_bin)

    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
