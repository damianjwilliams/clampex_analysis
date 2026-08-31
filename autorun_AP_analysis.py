#!/usr/bin/env python3
"""
autorun_AP_analysis.py
=======================

Pick a current-clamp .abf file via a file-open dialog and measure action
potential (AP) properties for every spike in every sweep, using the Allen
Institute's ipfx spike-feature-extraction pipeline
(https://github.com/AllenInstitute/ipfx).

For each sweep:
  * the current-injection window is auto-detected from the ABF epoch table
    (the "Step" epoch whose command level differs from the sweep's baseline
    level); if no such epoch exists (e.g. gap-free data) the whole sweep is
    used instead;
  * ipfx.feature_extractor.SpikeFeatureExtractor finds every spike in that
    window and measures its threshold, peak, trough, half-height width, max
    rise/fall rate (upstroke/downstroke) and afterhyperpolarization (fast
    trough / ADP / slow trough);
  * ipfx.feature_extractor.SpikeTrainFeatureExtractor summarises the train
    (firing rate, adaptation index, ISI CV, latency to first spike, sag,
    baseline Vm, ...).

Across sweeps, a cell-level summary is built: resting Vm, rheobase, and the
threshold / amplitude / half-width / AHP of the first spike at rheobase.

Results are written next to the source .abf file:
    <name>_AP_spikes.csv      one row per detected action potential
    <name>_AP_sweeps.csv      one row per analysed sweep (train features)
    <name>_AP_qc.png          diagnostic figure (unless --no-plot)

Usage
-----
    python autorun_AP_analysis.py                  # pops a file-open dialog
    python autorun_AP_analysis.py recording.abf     # skip the dialog
    python autorun_AP_analysis.py recording.abf --channel 1 --no-plot

Requires: numpy, pandas, matplotlib, pyabf, ipfx.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# File selection
# ----------------------------------------------------------------------------
def pick_abf_file() -> Optional[str]:
    """Show a native file-open dialog restricted to .abf files."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select an ABF file (current-clamp recording)",
        filetypes=[("Axon Binary Format", "*.abf"), ("All files", "*.*")],
    )
    root.destroy()
    return path or None


# ----------------------------------------------------------------------------
# Stimulus-window detection (per sweep, from the ABF epoch table)
# ----------------------------------------------------------------------------
def _stim_window(abf, sweep: int, channel: int):
    """
    Return (start_s, end_s, amplitude_pA) for the current-injection epoch of
    one sweep, auto-detected from the epoch table.

    Standard Clampfit step protocols bracket the injected step with baseline
    epochs (baseline - step - baseline [- ...]), so the stimulus is taken to
    be the longest *interior* epoch (neither first nor last) -- this also
    correctly identifies an intentional 0 pA step sweep, which a level-based
    comparison against baseline would miss.

    Falls back to the whole sweep (amplitude = NaN) if there are fewer than
    3 epochs, e.g. gap-free / non-stepped recordings.
    """
    abf.setSweep(sweep, channel=channel)
    ep = abf.sweepEpochs
    fs = abf.dataRate

    if len(ep.p1s) < 3:
        return float(abf.sweepX[0]), float(abf.sweepX[-1]), float("nan")

    baseline_level = ep.levels[0]
    interior = list(zip(ep.p1s[1:-1], ep.p2s[1:-1], ep.levels[1:-1]))
    p1, p2, lvl = max(interior, key=lambda c: c[1] - c[0])
    return p1 / fs, p2 / fs, float(lvl - baseline_level)


def _current_clamp_channels(abf) -> list[int]:
    """Channels whose recorded units are millivolts (i.e. membrane potential)."""
    return [ch for ch in abf.channelList if abf.adcUnits[ch].strip().lower() == "mv"]


def _safe_filter_khz(sample_rate_hz: float, desired_khz: float = 10.0) -> float:
    """
    ipfx's dV/dt filter must sit strictly below the Nyquist frequency. Low
    sample-rate recordings (e.g. 20 kHz) would otherwise make the default
    10 kHz filter invalid, so cap it below Nyquist with headroom.
    """
    nyquist_khz = (sample_rate_hz / 1e3) / 2.0
    return min(desired_khz, nyquist_khz * 0.9)


# ----------------------------------------------------------------------------
# Per-file analysis
# ----------------------------------------------------------------------------
def analyze_abf(
    path: str,
    channel: Optional[int] = None,
    dv_cutoff: float = 20.0,
    min_peak_mV: float = -30.0,
    min_height_mV: float = 2.0,
    max_interval_s: float = 0.005,
) -> dict:
    """
    Run ipfx spike-feature extraction on every sweep of a current-clamp ABF.

    Returns a dict with 'spikes' (DataFrame, one row per AP), 'sweeps'
    (DataFrame, one row per sweep) and 'meta' (dict).
    """
    import pyabf
    from ipfx.feature_extractor import SpikeFeatureExtractor, SpikeTrainFeatureExtractor

    abf = pyabf.ABF(path)

    if channel is None:
        cc_channels = _current_clamp_channels(abf)
        if not cc_channels:
            raise SystemExit(
                f"No current-clamp (mV) channel found in {path!r} "
                f"(channel units: {[abf.adcUnits[c] for c in abf.channelList]}). "
                f"AP analysis needs a current-clamp recording, not voltage-clamp."
            )
        channel = cc_channels[0]
    elif abf.adcUnits[channel].strip().lower() != "mv":
        warnings.warn(
            f"Channel {channel} is recorded in {abf.adcUnits[channel]!r}, not mV. "
            f"Spike detection assumes a membrane-potential trace."
        )

    filter_khz = _safe_filter_khz(abf.dataRate)
    all_spikes = []
    sweep_rows = []

    for sw in range(abf.sweepCount):
        abf.setSweep(sw, channel=channel)
        t = np.asarray(abf.sweepX, float)
        v = np.asarray(abf.sweepY, float)
        i = np.asarray(abf.sweepC, float)

        start, end, stim_amp = _stim_window(abf, sw, channel)

        sfx = SpikeFeatureExtractor(
            start=start, end=end, filter=filter_khz, dv_cutoff=dv_cutoff,
            min_peak=min_peak_mV, min_height=min_height_mV,
            max_interval=max_interval_s,
        )
        spikes_df = sfx.process(t, v, i)

        baseline_interval = min(0.1, max(start - float(t[0]), 0.001))
        stf = SpikeTrainFeatureExtractor(
            start=start, end=end, stim_amp_fn=lambda *_: stim_amp,
            baseline_interval=baseline_interval,
        )
        train = stf.process(
            t, v, i, spikes_df,
            extra_features=["stim_amp", "v_baseline", "sag", "peak_deflect"],
            exclude_clipped=True,
        )

        if not spikes_df.empty:
            spikes_df = spikes_df.copy()
            spikes_df.insert(0, "sweep", sw)
            spikes_df.insert(1, "stim_amp_pA", stim_amp)
            all_spikes.append(spikes_df)

        row = {"sweep": sw, "stim_amp_pA": stim_amp,
               "stim_start_s": start, "stim_end_s": end,
               "n_spikes": int(len(spikes_df))}
        row.update(train)
        sweep_rows.append(row)

    spikes = (pd.concat(all_spikes, ignore_index=True) if all_spikes
              else pd.DataFrame())
    sweeps = pd.DataFrame(sweep_rows)

    return dict(
        spikes=spikes, sweeps=sweeps,
        meta=dict(path=path, channel=channel, sample_rate_hz=abf.dataRate,
                  protocol=abf.protocol, sweep_count=abf.sweepCount),
    )


# ----------------------------------------------------------------------------
# Cell-level summary
# ----------------------------------------------------------------------------
def half_max_sweep(sweeps: pd.DataFrame) -> Optional[pd.Series]:
    """
    Return the row of `sweeps` whose spike count is closest to half of the
    maximum spike count across all sweeps (the "half-maximal" firing sweep).

    Candidates are restricted to sweeps that come *before* the sweep with
    the maximum firing rate (i.e. the ascending limb of the F-I curve), so a
    later sweep with a coincidentally similar spike count (e.g. after
    depolarization block reduces firing) is never picked instead. Ties are
    broken by lowest injected current. Returns None if no sweep fired at
    all.
    """
    firing = sweeps[sweeps["n_spikes"] > 0]
    if firing.empty:
        return None
    max_freq_sweep = firing.loc[firing["avg_rate"].idxmax(), "sweep"]
    candidates = firing[firing["sweep"] < max_freq_sweep]
    if candidates.empty:
        candidates = firing  # max-firing sweep is the first (or only) one firing
    target = firing["n_spikes"].max() / 2.0
    candidates = candidates.copy()
    candidates["_dist"] = (candidates["n_spikes"] - target).abs()
    candidates = candidates.sort_values(["_dist", "stim_amp_pA"])
    return candidates.iloc[0]


def cell_summary(result: dict) -> dict:
    spikes, sweeps = result["spikes"], result["sweeps"]
    out: dict = {}

    hm = half_max_sweep(sweeps)
    if hm is not None:
        out["half_max_sweep"] = int(hm["sweep"])
        out["half_max_n_spikes"] = int(hm["n_spikes"])
        out["half_max_stim_amp_pA"] = float(hm["stim_amp_pA"])

    quiet = sweeps[(sweeps["n_spikes"] == 0) & (sweeps["stim_amp_pA"] <= 0)]
    if not quiet.empty and "v_baseline" in quiet:
        out["resting_Vm_mV"] = float(quiet["v_baseline"].dropna().mean())
    elif "v_baseline" in sweeps:
        out["resting_Vm_mV"] = float(sweeps["v_baseline"].dropna().mean()) \
            if sweeps["v_baseline"].notna().any() else float("nan")

    firing = sweeps[(sweeps["n_spikes"] > 0) & (sweeps["stim_amp_pA"] > 0)]
    if firing.empty:
        out["rheobase_pA"] = float("nan")
        return out

    rheo_row = firing.loc[firing["stim_amp_pA"].idxmin()]
    out["rheobase_pA"] = float(rheo_row["stim_amp_pA"])
    out["rheobase_sweep"] = int(rheo_row["sweep"])

    rheo_spikes = spikes[spikes["sweep"] == out["rheobase_sweep"]].sort_values("threshold_t")
    if not rheo_spikes.empty:
        s0 = rheo_spikes.iloc[0]
        out["AP_threshold_mV"] = float(s0["threshold_v"])
        out["AP_peak_mV"] = float(s0["peak_v"])
        out["AP_amplitude_mV"] = float(s0["peak_v"] - s0["threshold_v"])
        out["AP_halfwidth_ms"] = float(s0["width"] * 1e3)
        out["AP_max_upstroke_mV_per_ms"] = float(s0["upstroke"] / 1e3)
        out["AP_max_downstroke_mV_per_ms"] = float(s0["downstroke"] / 1e3)
        fast_trough_v = s0.get("fast_trough_v", np.nan)
        out["AP_AHP_mV"] = float(fast_trough_v - s0["threshold_v"]) \
            if np.isfinite(fast_trough_v) else float("nan")
        out["AP_latency_ms"] = float(rheo_row["latency"] * 1e3) \
            if pd.notna(rheo_row.get("latency")) else float("nan")

    out["max_firing_rate_Hz"] = float(sweeps["avg_rate"].max())
    max_row = sweeps.loc[sweeps["avg_rate"].idxmax()]
    out["max_firing_rate_at_pA"] = float(max_row["stim_amp_pA"])
    out["max_firing_sweep"] = int(max_row["sweep"])
    if pd.notna(max_row.get("adapt")):
        out["adaptation_index_at_max_rate"] = float(max_row["adapt"])

    return out


def print_summary(result: dict, summary: dict) -> None:
    meta = result["meta"]
    print(f"AP analysis: {os.path.basename(meta['path'])}  "
          f"(protocol '{meta['protocol']}', channel {meta['channel']}, "
          f"{meta['sweep_count']} sweeps)")
    print("-" * 60)
    n_ap = len(result["spikes"])
    n_firing_sweeps = int((result["sweeps"]["n_spikes"] > 0).sum())
    print(f"  Total spikes detected : {n_ap}  across {n_firing_sweeps} sweep(s)")
    fields = [
        ("resting_Vm_mV", "Resting Vm", "mV"),
        ("rheobase_pA", "Rheobase", "pA"),
        ("AP_threshold_mV", "AP threshold (at rheobase)", "mV"),
        ("AP_peak_mV", "AP peak (at rheobase)", "mV"),
        ("AP_amplitude_mV", "AP amplitude (at rheobase)", "mV"),
        ("AP_halfwidth_ms", "AP half-width (at rheobase)", "ms"),
        ("AP_max_upstroke_mV_per_ms", "Max upstroke (dV/dt)", "mV/ms"),
        ("AP_max_downstroke_mV_per_ms", "Max downstroke (dV/dt)", "mV/ms"),
        ("AP_AHP_mV", "Fast AHP (at rheobase)", "mV"),
        ("AP_latency_ms", "Latency to 1st spike (at rheobase)", "ms"),
        ("max_firing_rate_Hz", "Max firing rate", "Hz"),
        ("max_firing_rate_at_pA", "  ...at current", "pA"),
        ("adaptation_index_at_max_rate", "Adaptation index (at max rate)", ""),
        ("half_max_sweep", "Half-max sweep", ""),
        ("half_max_n_spikes", "  ...spike count", ""),
        ("half_max_stim_amp_pA", "  ...at current", "pA"),
    ]
    for key, label, unit in fields:
        if key in summary and np.isfinite(summary[key]):
            print(f"  {label:<38} = {summary[key]:9.3f} {unit}")
    print()


# ----------------------------------------------------------------------------
# Diagnostic plot
# ----------------------------------------------------------------------------
def plot_ap_qc(result: dict, summary: dict, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyabf

    meta = result["meta"]
    abf = pyabf.ABF(meta["path"])
    sweeps, spikes = result["sweeps"], result["spikes"]

    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    fig.suptitle(f"{os.path.basename(meta['path'])}  -  AP analysis QC", fontsize=12)

    # (0,0) whole rheobase sweep
    ax = axes[0, 0]
    rheo_sw = summary.get("rheobase_sweep")
    if rheo_sw is not None:
        sp = spikes[spikes["sweep"] == rheo_sw].sort_values("threshold_t").iloc[0]
        abf.setSweep(int(rheo_sw), channel=meta["channel"])
        t, v = abf.sweepX, abf.sweepY
        dt = t[1] - t[0]
        i0 = int(sp["threshold_index"]) - int(0.002 / dt)
        i1 = int(sp["threshold_index"]) + int(0.008 / dt)
        i0, i1 = max(0, i0), min(len(t), i1)

        ax.plot(t * 1e3, v, color="#1f77b4", lw=0.6)
        ax.set_title(f"rheobase sweep {int(rheo_sw)} ({summary['rheobase_pA']:.0f} pA)")
    else:
        ax.set_title("no suprathreshold sweep found")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("Vm (mV)")

    # (0,1) half-maximal spike train: full sweep with every spike's
    # threshold and peak marked
    ax = axes[0, 1]
    hm_sw = summary.get("half_max_sweep")
    sp_hm = pd.DataFrame()
    if hm_sw is not None:
        abf.setSweep(int(hm_sw), channel=meta["channel"])
        t_hm, v_hm = abf.sweepX, abf.sweepY
        ax.plot(t_hm * 1e3, v_hm, color="#333333", lw=0.7)
        sp_hm = spikes[spikes["sweep"] == hm_sw].sort_values("threshold_t")
        if not sp_hm.empty:
            ax.plot(sp_hm["threshold_t"] * 1e3, sp_hm["threshold_v"], "v",
                    color="green", ms=6, ls="None", label="threshold")
            ax.plot(sp_hm["peak_t"] * 1e3, sp_hm["peak_v"], "^",
                    color="crimson", ms=6, ls="None", label="peak")
            ax.legend(fontsize=7.5, loc="upper right")
        ax.set_title(f"half-max spike train (sweep {int(hm_sw)}, "
                     f"{summary.get('half_max_stim_amp_pA', float('nan')):.0f} pA, "
                     f"{summary.get('half_max_n_spikes', 0)} APs)")
    else:
        ax.set_title("no half-max sweep found")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("Vm (mV)")

    # (0,2) max-firing-rate spike train: full sweep with every spike's
    # threshold and peak marked
    ax = axes[0, 2]
    max_sw = summary.get("max_firing_sweep")
    if max_sw is not None:
        abf.setSweep(int(max_sw), channel=meta["channel"])
        t_max, v_max = abf.sweepX, abf.sweepY
        ax.plot(t_max * 1e3, v_max, color="#333333", lw=0.7)
        sp_max = spikes[spikes["sweep"] == max_sw].sort_values("threshold_t")
        if not sp_max.empty:
            ax.plot(sp_max["threshold_t"] * 1e3, sp_max["threshold_v"], "v",
                    color="green", ms=6, ls="None", label="threshold")
            ax.plot(sp_max["peak_t"] * 1e3, sp_max["peak_v"], "^",
                    color="crimson", ms=6, ls="None", label="peak")
            ax.legend(fontsize=7.5, loc="upper right")
        ax.set_title(f"max-firing spike train (sweep {int(max_sw)}, "
                     f"{summary.get('max_firing_rate_at_pA', float('nan')):.0f} pA, "
                     f"{summary.get('max_firing_rate_Hz', float('nan')):.1f} Hz)")
    else:
        ax.set_title("no max-firing sweep found")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("Vm (mV)")

    # (1,0) phase-plane plot (dV/dt vs V) for the rheobase spike
    ax = axes[1, 0]
    if rheo_sw is not None:
        dvdt = np.gradient(v[i0:i1], t[i0:i1]) / 1e3  # mV/ms
        ax.plot(v[i0:i1], dvdt, color="#1f77b4", lw=1.2)
        ax.plot(sp["threshold_v"], sp["threshold_v"] * 0 +
                (sp["threshold_v"] - v[i0]) * 0 + dvdt[np.argmin(np.abs(v[i0:i1] - sp["threshold_v"]))],
                "v", color="green", ms=8)
        ax.set_title("phase plane (dV/dt vs V)")
    ax.set_xlabel("Vm (mV)"); ax.set_ylabel("dV/dt (mV/ms)")

    # (1,1) F-I curve
    ax = axes[1, 1]
    fi = sweeps.dropna(subset=["stim_amp_pA"]).sort_values("stim_amp_pA")
    ax.plot(fi["stim_amp_pA"], fi["avg_rate"], "o-", color="#1f77b4")
    if np.isfinite(summary.get("rheobase_pA", np.nan)):
        ax.axvline(summary["rheobase_pA"], color="crimson", ls="--", lw=1,
                    label=f"rheobase = {summary['rheobase_pA']:.0f} pA")
        ax.legend(fontsize=8)
    ax.set_xlabel("injected current (pA)"); ax.set_ylabel("firing rate (Hz)")
    ax.set_title("F-I curve")

    # (1,2) AHP/ADP detail on the first spike of the half-max train: zoom
    # from threshold out to the next spike's threshold (or a fixed window),
    # marking threshold, peak, fast trough (AHP), afterdepolarization (ADP)
    # and slow trough, if ipfx detected them.
    ax = axes[1, 2]
    if hm_sw is not None and not sp_hm.empty:
        abf.setSweep(int(hm_sw), channel=meta["channel"])
        t_hm, v_hm = abf.sweepX, abf.sweepY
        dt_hm = t_hm[1] - t_hm[0]
        sp0 = sp_hm.iloc[0]
        j0 = int(sp0["threshold_index"]) - int(0.002 / dt_hm)
        if len(sp_hm) > 1:
            j1 = int(sp_hm.iloc[1]["threshold_index"])
        else:
            j1 = int(sp0["threshold_index"]) + int(0.04 / dt_hm)
        j0, j1 = max(0, j0), min(len(t_hm), j1)
        tt = (t_hm[j0:j1] - sp0["threshold_t"]) * 1e3
        ax.plot(tt, v_hm[j0:j1], color="#1f77b4", lw=1.2)
        ax.plot(0.0, sp0["threshold_v"], "v", color="green", ms=8,
                label=f"threshold {sp0['threshold_v']:.1f} mV")
        ax.plot((sp0["peak_t"] - sp0["threshold_t"]) * 1e3, sp0["peak_v"],
                "^", color="crimson", ms=8, label=f"peak {sp0['peak_v']:.1f} mV")
        if pd.notna(sp0.get("fast_trough_t")):
            ahp = sp0["fast_trough_v"] - sp0["threshold_v"]
            ax.plot((sp0["fast_trough_t"] - sp0["threshold_t"]) * 1e3, sp0["fast_trough_v"],
                    "o", color="purple", ms=7, label=f"AHP {ahp:+.1f} mV")
        if pd.notna(sp0.get("adp_t")) and pd.notna(sp0.get("adp_v")):
            base = sp0.get("fast_trough_v")
            adp = sp0["adp_v"] - base if pd.notna(base) else float("nan")
            adp_x = (sp0["adp_t"] - sp0["threshold_t"]) * 1e3
            ax.plot(adp_x, sp0["adp_v"],
                    "s", color="orange", ms=7, label=f"ADP {adp:+.1f} mV")
            if pd.notna(base):
                # labelled line spanning the two values the ADP amplitude is
                # computed from: fast_trough_v (bottom) to adp_v (top)
                ax.annotate(
                    "", xy=(adp_x, sp0["adp_v"]), xytext=(adp_x, base),
                    arrowprops=dict(arrowstyle="<->", color="orange", lw=1.2),
                )
                ax.text(adp_x, (sp0["adp_v"] + base) / 2,
                        f"  ADP = adp_v − fast_trough_v\n"
                        f"  = {sp0['adp_v']:.1f} − ({base:.1f}) = {adp:+.1f} mV",
                        color="darkorange", fontsize=6.5, va="center")
        if pd.notna(sp0.get("slow_trough_t")):
            ax.plot((sp0["slow_trough_t"] - sp0["threshold_t"]) * 1e3, sp0["slow_trough_v"],
                    "d", color="teal", ms=6, label="slow trough")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"AHP/ADP detail (sweep {int(hm_sw)}, spike 1)")
    else:
        ax.set_title("no spike available for AHP/ADP detail")
    ax.set_xlabel("time from threshold (ms)"); ax.set_ylabel("Vm (mV)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)
    return out_path


def plot_rheobase_detail(result: dict, summary: dict, out_path: str) -> str:
    """
    Standalone, larger-format figure of the rheobase sweep's first spike,
    with dashed lines marking the exact voltages (threshold, peak, and
    half-height) and times (start/end of the half-width window) used to
    compute AP duration.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyabf

    meta = result["meta"]
    spikes = result["spikes"]
    rheo_sw = summary.get("rheobase_sweep")
    if rheo_sw is None:
        raise ValueError("no rheobase sweep found; cannot build rheobase detail figure")

    abf = pyabf.ABF(meta["path"])
    abf.setSweep(int(rheo_sw), channel=meta["channel"])
    t, v = abf.sweepX, abf.sweepY
    dt = t[1] - t[0]

    sp = spikes[spikes["sweep"] == rheo_sw].sort_values("threshold_t").iloc[0]
    i0 = int(sp["threshold_index"]) - int(0.002 / dt)
    i1 = int(sp["threshold_index"]) + int(0.008 / dt)
    i0, i1 = max(0, i0), min(len(t), i1)
    tt, vv = t[i0:i1] * 1e3, v[i0:i1]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(tt, vv, color="#1f77b4", lw=1.6, zorder=3)
    ax.set_xlim(tt[0], tt[-1])
    ymin, ymax = vv.min(), vv.max()
    pad = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    x0, x1 = ax.get_xlim()

    ax.plot(sp["threshold_t"] * 1e3, sp["threshold_v"], "v", color="green", ms=9, zorder=5,
            label=f"threshold  {sp['threshold_v']:.1f} mV @ {sp['threshold_t'] * 1e3:.2f} ms")
    ax.plot(sp["peak_t"] * 1e3, sp["peak_v"], "^", color="crimson", ms=9, zorder=5,
            label=f"peak  {sp['peak_v']:.1f} mV @ {sp['peak_t'] * 1e3:.2f} ms")
    if pd.notna(sp.get("fast_trough_t")):
        ax.plot(sp["fast_trough_t"] * 1e3, sp["fast_trough_v"], "o", color="purple", ms=8,
                zorder=5, label=f"AHP (fast trough)  {sp['fast_trough_v']:.1f} mV")

    # Reference lines for the two voltages AP duration is measured relative
    # to: threshold and peak ("AP height").
    ax.hlines(sp["threshold_v"], x0, x1, color="green", linestyle="--", lw=1.0,
              alpha=0.6, zorder=2)
    ax.hlines(sp["peak_v"], x0, x1, color="crimson", linestyle="--", lw=1.0,
              alpha=0.6, zorder=2)
    ax.text(x1, sp["threshold_v"], f"  threshold {sp['threshold_v']:.1f} mV",
            va="center", ha="left", fontsize=7.5, color="green")
    ax.text(x1, sp["peak_v"], f"  AP height (peak) {sp['peak_v']:.1f} mV",
            va="center", ha="left", fontsize=7.5, color="crimson")

    if pd.notna(sp.get("trough_index")) and pd.notna(sp.get("peak_index")):
        # Replicate ipfx's own half-height crossing search (ipfx.spike_features
        # .find_widths) on the real trace, instead of assuming symmetry about
        # the peak -- this guarantees the dashed lines land exactly where the
        # waveform crosses the width level, and that the printed duration
        # matches sp["width"] exactly.
        peak_idx = int(sp["peak_index"])
        thresh_idx = int(sp["threshold_index"])
        trough_idx = int(sp["trough_index"])

        height = v[peak_idx] - v[trough_idx]
        width_level = v[trough_idx] + height / 2.0
        if width_level < v[thresh_idx]:
            width_level = v[thresh_idx] + (v[peak_idx] - v[thresh_idx]) / 2.0

        back_hits = np.flatnonzero(v[peak_idx:thresh_idx:-1] <= width_level)
        fwd_hits = np.flatnonzero(v[peak_idx:trough_idx] <= width_level)

        if back_hits.size and fwd_hits.size:
            start_idx = peak_idx - back_hits[0]
            end_idx = peak_idx + fwd_hits[0]
            tL, tR = t[start_idx] * 1e3, t[end_idx] * 1e3
            hw = t[end_idx] - t[start_idx]
            y0 = ax.get_ylim()[0]

            ax.hlines(width_level, tL, tR, color="black", linestyle="--", lw=1.3, zorder=4)
            ax.vlines([tL, tR], y0, width_level, color="black", linestyle="--", lw=1.1, zorder=4)
            ax.plot([tL, tR], [v[start_idx], v[end_idx]], "x", color="black", ms=7, zorder=5)
            ax.annotate("", xy=(tR, width_level), xytext=(tL, width_level),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
            ax.text((tL + tR) / 2, width_level, f"  {hw * 1e3:.3f} ms",
                    va="bottom", ha="center", fontsize=9, fontweight="bold")
            ax.text(tL, y0, f" {tL:.2f} ms", va="bottom", ha="right", fontsize=7.5, rotation=90)
            ax.text(tR, y0, f" {tR:.2f} ms", va="bottom", ha="left", fontsize=7.5, rotation=90)
            ax.text(tL, width_level, f"{width_level:.1f} mV  ", va="top", ha="right", fontsize=8)

    ax.set_xlabel("time (ms)")
    ax.set_ylabel("Vm (mV)")
    ax.set_title(f"{os.path.basename(meta['path'])}  -  rheobase sweep {int(rheo_sw)} "
                 f"({summary.get('rheobase_pA', float('nan')):.0f} pA), "
                 f"half-width {summary.get('AP_halfwidth_ms', float('nan')):.2f} ms")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", nargs="?", help="Path to a current-clamp .abf file "
                                          "(omit to pick one via a file dialog)")
    p.add_argument("--channel", type=int, default=None,
                   help="ADC channel to analyse (default: auto-detect the mV channel)")
    p.add_argument("--dv-cutoff", type=float, default=20.0,
                   help="dV/dt threshold (mV/ms) for putative spike detection (default 20)")
    p.add_argument("--min-peak", type=float, default=-30.0,
                   help="minimum peak voltage (mV) to accept a spike (default -30)")
    p.add_argument("--no-plot", action="store_true", help="skip the QC figure")
    p.add_argument("--per-spike", action="store_true",
                   help="also print every detected spike's features")
    args = p.parse_args(argv)

    path = args.abf or pick_abf_file()
    if not path:
        p.error("no file selected")

    result = analyze_abf(path, channel=args.channel, dv_cutoff=args.dv_cutoff,
                          min_peak_mV=args.min_peak)
    summary = cell_summary(result)
    print_summary(result, summary)

    if args.per_spike and not result["spikes"].empty:
        cols = ["sweep", "stim_amp_pA", "threshold_t", "threshold_v", "peak_v",
                "width", "upstroke", "downstroke", "fast_trough_v"]
        cols = [c for c in cols if c in result["spikes"].columns]
        print(result["spikes"][cols].to_string(index=False))
        print()

    stem = os.path.splitext(path)[0]
    spikes_csv, sweeps_csv = f"{stem}_AP_spikes.csv", f"{stem}_AP_sweeps.csv"
    result["spikes"].to_csv(spikes_csv, index=False)
    result["sweeps"].to_csv(sweeps_csv, index=False)
    print(f"Per-spike features   -> {spikes_csv}")
    print(f"Per-sweep features   -> {sweeps_csv}")

    if not args.no_plot:
        qc_png = f"{stem}_AP_qc.png"
        try:
            plot_ap_qc(result, summary, qc_png)
            print(f"QC figure            -> {qc_png}")
        except Exception as exc:
            print(f"WARNING: could not build QC figure ({exc})", file=sys.stderr)

        rheo_png = f"{stem}_AP_rheobase.png"
        try:
            plot_rheobase_detail(result, summary, rheo_png)
            print(f"Rheobase detail       -> {rheo_png}")
        except Exception as exc:
            print(f"WARNING: could not build rheobase detail figure ({exc})", file=sys.stderr)


if __name__ == "__main__":
    main()
