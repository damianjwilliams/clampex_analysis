#!/usr/bin/env python3
"""
membrane_test.py
================

Compute access resistance (Ra), membrane resistance (Rm), membrane capacitance
(Cm), the transient time constant (Tau) and steady-state current (Iss) from a
voltage-clamp "Membrane Test" recording, following the algorithm described in
the Molecular Devices / Clampfit documentation.

Algorithm (as implemented, mirroring the Molecular Devices description)
----------------------------------------------------------------------
A continuous square-wave voltage command oscillates about the holding level.
Both edges of every pulse (both capacitive transients) are used. For each edge:

  * The steady-state current of the segment *before*  the edge is I2 (baseline).
  * The steady-state current of the segment *after*   the edge is I1.
    Each is the mean over the last portion (T1 = 20% of the step) of its segment.
  * dI  = I1 - I2                      (steady-state current step)
  * dV  = V1 - V2                      (command voltage step)
  * Rt  = dV / dI                      = Ra + Rm      (total resistance)
  * Iss = (I1 + I2) / 2                (steady-state current)
  * Tau is found from a fast log-linear fit of the decaying transient,
    fit between the "proportion of peak" ordinates (default 10-80%).
  * Q1  = integral of (I(t) - I1) over the transient          (charge above I1)
  * Q2  = dI * Tau                     (settling-time correction)
  * Qt  = Q1 + Q2
  * Cm  = Qt / dV
  * Cm_charge = Qt / dV        (capacitance as the MD note defines it)
  * Ra  = tau / Cm_charge      (see "Solving for Ra" below)
  * Rm  = Rt - Ra
  * Cm  = Cm_charge * Rt / Rm  (bias-corrected true membrane capacitance)

Solving for Ra -- a deliberate deviation from the published note
----------------------------------------------------------------
The MD note solves  Ra^2 - Rt*Ra + Rt*(tau/Cm) = 0  by Newton-Raphson. That
quadratic follows from  tau/Cm = Ra*Rm/(Ra+Rm), which requires the TRUE Cm.
But the charge integral the note prescribes does not return the true Cm. For
the series model Ra + (Rm||Cm):

    Q1 = dV*Cm*Rm^2/Rt^2 ,   Q2 = dI*tau = dV*Cm*Ra*Rm/Rt^2
    Qt = Q1 + Q2 = dV*Cm*Rm/Rt
    => Cm_charge = Qt/dV = Cm_true * Rm/Rt          (biased LOW by Rm/Rt)

Substituting into tau = Cm_true*Ra*Rm/Rt gives the exact, non-iterative result

    tau = Ra * Cm_charge        =>      Ra = tau / Cm_charge

Feeding the biased Cm_charge into the published quadratic makes it (a) biased
high and (b) have NO REAL ROOT whenever Ra > Rt/4, since the discriminant
Rt^2 - 4*Rt*(tau/Cm_charge) turns negative. Real cells with high Ra or a leaky
seal land in that regime routinely. This module therefore uses the exact
relation by default; `method="quadratic"` reproduces the published behaviour
and returns NaN (never a fabricated Rt/2 vertex) where it has no solution.

The downward pulse needs no special handling here: keeping the natural signs of
dV, dI, Q1 and Q2 makes the down-step calculation identical to the up-step
(equivalent to the manual's "invert the pulse" instruction).

All maths is done in SI units internally (V, A, s, C, F, Ohm); results are then
reported in convenient units (mV, pA, ms, MOhm, pF).

Usage
-----
    python membrane_test.py recording.abf
    python membrane_test.py recording.abf --channel 0 --sweeps 0-4 --fit 10 80
    python membrane_test.py --selftest        # run synthetic RC validation

Requires: numpy, scipy, and (for reading ABFs) pyabf.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# NumPy >=2.0 renamed trapz -> trapezoid; support both.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ----------------------------------------------------------------------------
# Unit handling
# ----------------------------------------------------------------------------
_UNIT_SCALE = {  # multiply a value in <unit> by this to get SI base units
    "V": 1.0, "mV": 1e-3, "uV": 1e-6, "\u00b5V": 1e-6,
    "A": 1.0, "mA": 1e-3, "uA": 1e-6, "\u00b5A": 1e-6, "nA": 1e-9, "pA": 1e-12,
    "s": 1.0, "ms": 1e-3,
}


def _to_si(value: np.ndarray | float, unit: str) -> np.ndarray | float:
    scale = _UNIT_SCALE.get(unit.strip(), None)
    if scale is None:
        raise ValueError(f"Unrecognised unit {unit!r}; add it to _UNIT_SCALE.")
    return value * scale


# ----------------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------------
@dataclass
class TransientResult:
    """One capacitive transient (one command edge)."""
    edge_index: int
    direction: int          # +1 up-step, -1 down-step
    dV: float               # V
    dI: float               # A
    Rt: float               # Ohm  (Ra + Rm)
    Ra: float               # Ohm
    Rm: float               # Ohm
    Cm: float               # F  (bias-corrected, = Cm_charge * Rt/Rm)
    Cm_charge: float        # F  (raw Qt/dV, as the MD note defines it)
    tau: float              # s
    Iss: float              # A
    Q1: float               # C
    Q2: float               # C
    Qt: float               # C
    r2_fit: float           # goodness of the log-linear tau fit
    valid: bool = True
    note: str = ""
    # --- geometry, for the diagnostic plot (absolute sample indices) ---
    I1: float = np.nan      # A, steady state AFTER the edge
    I2: float = np.nan      # A, steady state BEFORE the edge (baseline)
    t1_win_after: tuple = (0, 0)    # samples averaged to get I1
    t1_win_before: tuple = (0, 0)   # samples averaged to get I2
    seg: tuple = (0, 0)             # transient region integrated for Q1
    peak_idx: int = 0               # absolute index of the transient peak
    fit_win: tuple = (0, 0)         # absolute indices actually fitted for tau
    fit_slope: float = np.nan
    fit_intercept: float = np.nan
    fit_sign: float = 1.0
    fit_t0: float = np.nan          # time origin of the fit (s)


@dataclass
class MembraneTestResult:
    transients: list[TransientResult] = field(default_factory=list)
    first_sweep: dict = field(default_factory=dict)  # arrays kept for plotting

    def _agg(self, attr, direction=None):
        vals = [getattr(t, attr) for t in self.transients
                if t.valid and (direction is None or t.direction == direction)]
        vals = [v for v in vals if np.isfinite(v)]
        return np.array(vals, float)

    def summary(self) -> dict:
        out = {}
        for name, attr, scale, unit in [
            ("Ra",  "Ra",  1e-6,  "MOhm"),
            ("Rm",  "Rm",  1e-6,  "MOhm"),
            ("Rt",  "Rt",  1e-6,  "MOhm"),
            ("Cm",  "Cm",  1e12,  "pF"),
            ("Cm_charge", "Cm_charge", 1e12, "pF"),
            ("Tau", "tau", 1e3,   "ms"),
            ("Iss", "Iss", 1e12,  "pA"),
        ]:
            v = self._agg(attr)
            out[name] = {
                "mean": float(np.mean(v)) * scale if v.size else float("nan"),
                "sd":   float(np.std(v)) * scale if v.size else float("nan"),
                "n":    int(v.size),
                "unit": unit,
            }
        return out

    def __str__(self) -> str:
        s = self.summary()
        n = self.transients and sum(t.valid for t in self.transients) or 0
        lines = [f"Membrane Test summary  ({n} valid transients)"]
        lines.append("-" * 46)
        for k in ("Ra", "Rm", "Rt", "Cm", "Cm_charge", "Tau", "Iss"):
            d = s[k]
            lines.append(f"  {k:<4} = {d['mean']:10.3f} \u00b1 {d['sd']:8.3f} "
                         f"{d['unit']:<5} (n={d['n']})")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Core numerics
# ----------------------------------------------------------------------------
def _find_edges(command: np.ndarray, min_step: float) -> np.ndarray:
    """Indices (in `command`) of the sample *after* each command step."""
    dC = np.diff(command)
    raw = np.where(np.abs(dC) > min_step)[0] + 1
    if raw.size == 0:
        return raw
    # collapse runs of adjacent indices (a step spread over a couple samples)
    grouped, run = [], [raw[0]]
    for idx in raw[1:]:
        if idx - run[-1] <= 2:
            run.append(idx)
        else:
            grouped.append(run[0])
            run = [idx]
    grouped.append(run[0])
    return np.array(grouped, int)


def _fit_tau_log(t: np.ndarray, y: np.ndarray, lo_frac: float, hi_frac: float):
    """
    Fast log-linear single-exponential fit.

    `y` is the transient amplitude relative to its own steady state (so it
    decays toward 0). Returns (tau, r2, info) where `info` carries the fit
    geometry (peak index, indices actually fitted, slope/intercept, sign) so a
    diagnostic plot can draw exactly what was fitted rather than re-deriving it.
    Points are selected where |y| lies between lo_frac and hi_frac of the peak,
    on the decaying side only.
    """
    nofit = (np.nan, np.nan, None)
    if y.size < 4:
        return nofit
    peak_i = int(np.argmax(np.abs(y[:max(3, y.size // 4)])))  # peak near the edge
    sign = np.sign(y[peak_i]) or 1.0
    yy = y * sign                       # make the transient decay from +peak to 0
    peak = yy[peak_i]
    if peak <= 0:
        return nofit

    hi, lo = hi_frac * peak, lo_frac * peak
    # restrict to the monotonic decay region after the peak
    seg = np.arange(peak_i, yy.size)
    mask = (yy[seg] <= hi) & (yy[seg] >= lo) & (yy[seg] > 0)
    idx = seg[mask]
    if idx.size < 3:
        return nofit

    tt = t[idx] - t[idx[0]]
    ly = np.log(yy[idx])
    slope, intercept = np.polyfit(tt, ly, 1)
    if slope >= 0:
        return nofit
    tau = -1.0 / slope
    # r^2 of the log-linear fit
    pred = slope * tt + intercept
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    info = dict(peak_i=peak_i, peak=peak, sign=float(sign),
                fit_i0=int(idx[0]), fit_i1=int(idx[-1]),
                t0=float(t[idx[0]]), slope=float(slope),
                intercept=float(intercept),
                lo=float(lo), hi=float(hi))
    return tau, r2, info


def _solve_ra_quadratic(Rt: float, tau: float, Cm: float,
                        iters: int = 50, tol: float = 1e-3):
    """
    Solve Ra^2 - Rt*Ra + Rt*(tau/Cm) = 0 for the smaller (physical) root by
    Newton-Raphson, exactly as the Molecular Devices note describes.

    NOTE: this is provided for fidelity to the published algorithm, but it is
    biased. See `_solve_ra_exact` and the module docstring. Returns
    (Ra, ok) where ok=False means the quadratic has no real root, which happens
    whenever Ra > Rt/4. In that case Ra is NOT recoverable by this method and
    NaN is returned rather than a misleading Rt/2 vertex value.
    """
    k = tau / Cm
    disc = Rt * Rt - 4.0 * Rt * k
    if disc < 0:
        return float("nan"), False
    Ra = min(k, Rt / 2.0)
    for _ in range(iters):
        f = Ra * Ra - Rt * Ra + Rt * k
        fp = 2.0 * Ra - Rt
        if fp == 0:
            break
        step = f / fp
        Ra -= step
        if abs(step) < tol:
            break
    small_root = (Rt - np.sqrt(disc)) / 2.0
    if not (0 <= Ra <= Rt) or abs(Ra - small_root) > abs(Ra) * 0.5:
        Ra = small_root
    return Ra, True


def _solve_ra_exact(Rt: float, tau: float, Cm_charge: float):
    """
    Exact access resistance for the series model Ra + (Rm || Cm).

    Derivation (see module docstring): the charge-integral capacitance is
        Cm_charge = Qt/dV = Cm_true * Rm/Rt
    and the decay time constant is
        tau = Cm_true * Ra*Rm/Rt = Ra * Cm_charge
    hence
        Ra = tau / Cm_charge          (exact, no iteration required)

    Returns (Ra, ok). ok=False if the result is non-physical (Ra >= Rt).
    """
    if not (np.isfinite(tau) and np.isfinite(Cm_charge)) or Cm_charge == 0:
        return float("nan"), False
    Ra = tau / Cm_charge
    return Ra, bool(0.0 < Ra < Rt)


def membrane_test_from_arrays(
    time_s: np.ndarray,
    current_A: np.ndarray,
    command_V: np.ndarray,
    t1_fraction: float = 0.20,
    fit_lo: float = 0.10,
    fit_hi: float = 0.80,
    min_step_V: Optional[float] = None,
    min_r2: float = 0.80,
    method: str = "exact",
) -> MembraneTestResult:
    """
    Run the Membrane Test analysis on a single sweep's SI-unit arrays.

    Parameters
    ----------
    time_s, current_A, command_V : 1-D arrays of equal length (SI units).
    t1_fraction : fraction of each step, at its end, used to measure steady state.
    fit_lo, fit_hi : proportion-of-peak ordinates for the tau fit (e.g. 0.10, 0.80).
    min_step_V : command-step threshold for edge detection; auto if None.
    min_r2 : minimum log-fit r^2 to accept a transient.
    method : 'exact' (default, Ra = tau/Cm_charge) or 'quadratic' (the
        Newton-Raphson root of Ra^2 - Rt*Ra + Rt*(tau/Cm) exactly as published;
        biased, and undefined when Ra > Rt/4).
    """
    time_s = np.asarray(time_s, float)
    current_A = np.asarray(current_A, float)
    command_V = np.asarray(command_V, float)

    pp = float(np.max(command_V) - np.min(command_V))
    if min_step_V is None:
        min_step_V = 0.4 * pp if pp > 0 else 0.0
    if min_step_V <= 0:
        return MembraneTestResult()

    edges = _find_edges(command_V, min_step_V)
    result = MembraneTestResult()
    if edges.size < 2:
        return result

    # Segment boundaries: [start_of_data, edge0, edge1, ..., end_of_data]
    bounds = np.concatenate(([0], edges, [len(command_V)]))

    def seg_steady(seg_start, seg_end):
        """Mean current & command over the last t1_fraction of a segment.

        Returns (I, V, window) where `window` is the (start, end) sample range
        averaged, kept so the diagnostic plot can show where I1/I2 came from.
        """
        n = seg_end - seg_start
        if n <= 1:
            return np.nan, np.nan, (seg_start, seg_end)
        w = max(1, int(round(n * t1_fraction)))
        sl = slice(seg_end - w, seg_end)
        return (float(np.mean(current_A[sl])), float(np.mean(command_V[sl])),
                (seg_end - w, seg_end))

    for e in range(len(edges)):
        edge_idx = edges[e]
        seg_before = (bounds[e], bounds[e + 1])       # ends at this edge
        seg_after = (bounds[e + 1], bounds[e + 2])     # starts at this edge

        I2, V2, w2 = seg_steady(*seg_before)           # baseline (pre-step SS)
        I1, V1, w1 = seg_steady(*seg_after)            # new steady state
        if not all(np.isfinite([I1, I2, V1, V2])):
            continue

        dV = V1 - V2
        dI = I1 - I2
        if dV == 0 or dI == 0:
            continue
        direction = 1 if dV > 0 else -1

        # Transient region: from the edge to the end of the new segment
        ts, te = edge_idx, bounds[e + 2]
        t_seg = time_s[ts:te]
        i_seg = current_A[ts:te]
        if t_seg.size < 5:
            continue

        # amplitude relative to the segment's own steady state (I1); decays to 0
        amp = i_seg - I1
        tau, r2, finfo = _fit_tau_log(t_seg, amp, fit_lo, fit_hi)

        # Q1 = integral of (I(t) - I1) over the transient  (natural sign = sign(dV))
        Q1 = float(_trapz(amp, t_seg))
        Q2 = dI * tau if np.isfinite(tau) else np.nan
        Qt = Q1 + Q2

        Cm_charge = Qt / dV          # as defined in the MD note (biased low)
        Rt = dV / dI

        base_ok = (np.isfinite(tau) and np.isfinite(r2) and r2 >= min_r2
                   and np.isfinite(Cm_charge) and Cm_charge > 0 and Rt > 0)

        Ra = Rm = Cm = float("nan")
        note = ""
        valid = False
        if not base_ok:
            note = "rejected (bad tau fit or non-physical Cm/Rt)"
        else:
            if method == "quadratic":
                Ra, ok = _solve_ra_quadratic(Rt, tau, Cm_charge)
                if not ok:
                    note = ("quadratic has no real root (Ra > Rt/4); "
                            "documented method cannot solve this cell")
            else:
                Ra, ok = _solve_ra_exact(Rt, tau, Cm_charge)
                if not ok:
                    note = "non-physical Ra (>= Rt)"
            if ok:
                Rm = Rt - Ra
                # undo the Rm/Rt bias in the charge-integral capacitance
                Cm = Cm_charge * Rt / Rm if Rm > 0 else float("nan")
                valid = bool(np.isfinite(Cm) and Cm > 0)

        result.transients.append(TransientResult(
            edge_index=int(edge_idx), direction=direction,
            dV=dV, dI=dI, Rt=Rt, Ra=Ra, Rm=Rm,
            Cm=Cm, Cm_charge=Cm_charge, tau=tau,
            Iss=(I1 + I2) / 2.0, Q1=Q1, Q2=Q2, Qt=Qt,
            r2_fit=float(r2) if np.isfinite(r2) else np.nan,
            valid=bool(valid), note=note,
            I1=I1, I2=I2, t1_win_after=w1, t1_win_before=w2,
            seg=(int(ts), int(te)),
            peak_idx=int(ts + finfo["peak_i"]) if finfo else int(ts),
            fit_win=(int(ts + finfo["fit_i0"]), int(ts + finfo["fit_i1"]))
                    if finfo else (0, 0),
            fit_slope=finfo["slope"] if finfo else np.nan,
            fit_intercept=finfo["intercept"] if finfo else np.nan,
            fit_sign=finfo["sign"] if finfo else 1.0,
            fit_t0=finfo["t0"] if finfo else np.nan,
        ))

    return result


# ----------------------------------------------------------------------------
# ABF loading
# ----------------------------------------------------------------------------
def _warn_acquisition_settings(abf, channel: int) -> None:
    """Flag telegraphed settings that undermine a Membrane Test measurement."""
    try:
        filt = abf._adcSection.fTelegraphFilter[channel]
    except Exception:
        return
    if filt and filt < 10_000:
        print(f"NOTE: channel {channel} was recorded through a {filt:.0f} Hz "
              f"lowpass filter. The capacitive peak is therefore smoothed, so "
              f"any peak-based Ra estimate will be too high. Charge (Cm) and a "
              f"slow tau survive filtering, so the charge-based Ra used here is "
              f"largely unaffected.", file=sys.stderr)


def membrane_test_from_abf(
    path: str,
    channel: int = 0,
    command_channel: Optional[int] = None,
    sweeps: Optional[list[int]] = None,
    **kwargs,
) -> MembraneTestResult:
    """
    Analyse a Membrane-Test ABF. Pools transients across the chosen sweeps.

    The command square wave is taken from the reconstructed command waveform
    (`sweepC`) by default. If your rig instead records the command on a separate
    ADC channel, pass `command_channel` and it will be used as the command.
    """
    try:
        import pyabf
    except ImportError as exc:
        raise SystemExit("pyabf is required to read ABF files: pip install pyabf") from exc

    abf = pyabf.ABF(path)
    if channel not in abf.channelList:
        raise SystemExit(f"Channel {channel} not in {abf.channelList}")
    _warn_acquisition_settings(abf, channel)
    sweep_list = list(range(abf.sweepCount)) if sweeps is None else sweeps

    pooled = MembraneTestResult()
    first = {}
    for sw in sweep_list:
        abf.setSweep(sw, channel=channel)
        t = _to_si(np.asarray(abf.sweepX, float), "s")
        i = _to_si(np.asarray(abf.sweepY, float), abf.sweepUnitsY)
        if command_channel is not None:
            abf.setSweep(sw, channel=command_channel)
            c = _to_si(np.asarray(abf.sweepY, float), abf.sweepUnitsY)
            abf.setSweep(sw, channel=channel)
        else:
            # sweepC is the command waveform reconstructed from the epoch table
            c = _to_si(np.asarray(abf.sweepC, float), abf.sweepUnitsC)
        res = membrane_test_from_arrays(t, i, c, **kwargs)
        if not first:
            first = dict(time_s=t, current_A=i, command_V=c, result=res)
        pooled.transients.extend(res.transients)
    pooled.first_sweep = first

    if not pooled.transients:
        print("WARNING: no command steps detected. Is this a Membrane-Test "
              "protocol, and is the square wave present in the command "
              "channel (sweepC)?", file=sys.stderr)
    return pooled



# ----------------------------------------------------------------------------
# Diagnostic plot
# ----------------------------------------------------------------------------
def plot_membrane_test(time_s, current_A, command_V, result,
                       out_path="membrane_test_qc.png", title="",
                       max_edges=2, fit_lo=0.10, fit_hi=0.80):
    """
    Draw a QC figure so every measurement can be checked by eye.

    Row 0  whole sweep, full scale: the I2 and I1 averaging windows are shaded
           where they actually sit (I1 is at the END of the step, far from the
           transient), plus the command trace beneath.
    Row 1  per transient, zoomed to the decay: I1/Iss levels, the shaded Q1
           integration area, the 10-80%-of-peak ordinates, the peak, and the
           fitted exponential drawn in red over the raw data.
    Row 2  the same transient on log axes. A single exponential is a straight
           line here, so any bow reveals a multi-exponential decay -- the
           assumption the whole Membrane Test model rests on.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tms, ipA, vmV = time_s * 1e3, current_A * 1e12, command_V * 1e3
    trs = list(result.transients)[:max_edges]
    n = max(1, len(trs))
    fig = plt.figure(figsize=(7.2 * n, 10.5))
    gs = fig.add_gridspec(4, n, height_ratios=[1.15, 0.42, 1.35, 1.15],
                          hspace=0.55, wspace=0.22)

    # ---- row 0: whole sweep, showing WHERE each measurement is taken --------
    axv = fig.add_subplot(gs[0, :])
    axv.plot(tms, ipA, lw=0.6, color="#1f77b4", zorder=2)
    axv.set_ylabel("I (pA)")
    axv.set_title(title or "Membrane Test - whole sweep", fontsize=11)
    seen = set()
    for k, tr in enumerate(trs):
        axv.axvline(tms[tr.edge_index], color="crimson", lw=0.9, ls="--", zorder=3)
        axv.annotate(f"edge {k+1}", (tms[tr.edge_index], 1.0),
                     xycoords=("data", "axes fraction"), xytext=(3, -11),
                     textcoords="offset points", color="crimson", fontsize=8)
        for (w, col, lab) in [(tr.t1_win_before, "#2ca02c", "I2 window (baseline)"),
                              (tr.t1_win_after, "#ff7f0e", "I1 window (steady state)")]:
            e = min(w[1], len(tms) - 1)
            axv.axvspan(tms[w[0]], tms[e], color=col, alpha=0.20, zorder=0,
                        label=lab if lab not in seen else None)
            seen.add(lab)
    axv.legend(fontsize=7, loc="lower right", framealpha=0.9, ncol=2)
    axc = fig.add_subplot(gs[1, :], sharex=axv)
    axc.plot(tms, vmV, lw=0.9, color="#444444")
    axc.set_ylabel("V cmd\n(mV)"); axc.set_xlabel("time (ms)"); axc.margins(y=0.45)

    for k, tr in enumerate(trs):
        ts, te = tr.seg
        tau_ms = tr.tau * 1e3 if np.isfinite(tr.tau) else 1.0
        # zoom: 0.5 ms before the edge to ~7 tau after (or the segment end)
        dt = tms[1] - tms[0]
        lo_i = max(0, ts - int(0.5 / dt))
        hi_i = min(te, ts + int(min(7 * tau_ms, (te - ts) * dt) / dt))
        I1p, I2p = tr.I1 * 1e12, tr.I2 * 1e12

        ax = fig.add_subplot(gs[2, k])
        # Q1 shaded area
        ax.fill_between(tms[ts:hi_i], I1p, ipA[ts:hi_i], color="#9467bd",
                        alpha=0.22, zorder=1,
                        label="Q1 = $\\int$(I - I1)dt")
        ax.plot(tms[lo_i:hi_i], ipA[lo_i:hi_i], lw=0.8, color="#1f77b4",
                zorder=2, label="data")
        ax.axvline(tms[ts], color="crimson", lw=0.9, ls="--", zorder=3)
        ax.axhline(I2p, color="#2ca02c", lw=1.4, ls="-", zorder=3,
                   label=f"I2 = {I2p:.0f} pA")
        ax.axhline(I1p, color="#ff7f0e", lw=1.4, ls="-", zorder=3,
                   label=f"I1 = {I1p:.0f} pA")
        ax.axhline(tr.Iss * 1e12, color="k", lw=0.8, ls="-.", zorder=3,
                   label=f"Iss = {tr.Iss*1e12:.0f} pA")

        if np.isfinite(tr.tau) and tr.fit_win != (0, 0):
            f0, f1 = tr.fit_win
            yfit = (tr.fit_sign * np.exp(tr.fit_slope * (time_s - tr.fit_t0)
                                         + tr.fit_intercept)) * 1e12 + I1p
            sp = slice(max(ts, f0 - int(0.15 * tau_ms / dt)),
                       min(hi_i, f1 + int(2 * tau_ms / dt)))
            ax.plot(tms[sp], yfit[sp], color="red", lw=1.8, zorder=5,
                    label=f"fit: $\\tau$ = {tau_ms:.3f} ms")
            pk = tr.fit_sign * (ipA[tr.peak_idx] - I1p)
            for frac, ls, lab in ((fit_hi, "--", f"{fit_hi*100:.0f}% of peak"),
                                  (fit_lo, ":", f"{fit_lo*100:.0f}% of peak")):
                ax.axhline(I1p + tr.fit_sign * frac * pk, color="red", lw=0.8,
                           ls=ls, alpha=0.8, zorder=4, label=lab)
            ax.axvspan(tms[f0], tms[f1], color="red", alpha=0.07, zorder=0,
                       label="fit window")
            ax.plot(tms[tr.peak_idx], ipA[tr.peak_idx], "v", color="red", ms=7,
                    zorder=6, label=f"peak ({(tr.peak_idx-ts)*dt*1e3:.0f} $\\mu$s)")

        ax.set_xlim(tms[lo_i], tms[hi_i - 1])
        ax.set_xlabel("time (ms)"); ax.set_ylabel("I (pA)")
        ax.set_title(f"edge {k+1} ({'up' if tr.direction > 0 else 'down'}):  "
                     f"$\\Delta$V = {tr.dV*1e3:+.1f} mV,  "
                     f"$\\Delta$I = {tr.dI*1e12:+.0f} pA,  "
                     f"Rt = $\\Delta$V/$\\Delta$I = {tr.Rt/1e6:.1f} M$\\Omega$",
                     fontsize=9)
        ax.legend(fontsize=6.2, loc="best", framealpha=0.92, ncol=2)

        # ---- log view --------------------------------------------------------
        axl = fig.add_subplot(gs[3, k])
        amp = tr.fit_sign * (current_A[ts:te] - tr.I1) * 1e12
        tt = tms[ts:te] - tms[ts]
        m = amp > 0
        axl.semilogy(tt[m], amp[m], lw=0.7, color="#1f77b4", label="|I - I1|")
        if np.isfinite(tr.tau) and tr.fit_win != (0, 0):
            f0, f1 = tr.fit_win
            yl = np.exp(tr.fit_slope * (time_s[ts:te] - tr.fit_t0)
                        + tr.fit_intercept) * 1e12
            sp = (time_s[ts:te] >= time_s[f0] - 0.2 * tr.tau) & \
                 (time_s[ts:te] <= time_s[f1] + 1.5 * tr.tau)
            axl.semilogy(tt[sp], yl[sp], color="red", lw=1.6,
                         label=f"single-exp fit ($r^2$={tr.r2_fit:.4f})")
            axl.axvspan(tt[f0 - ts], tt[f1 - ts], color="red", alpha=0.07,
                        label="fit window")
        pos = amp[m]
        if pos.size:
            axl.set_ylim(max(pos.max() * 2e-3, 0.3), pos.max() * 2.2)
        axl.set_xlim(0, min(tt[-1], 7 * tau_ms))
        axl.set_xlabel("time from edge (ms)"); axl.set_ylabel("|I - I1| (pA)")
        axl.set_title("log scale: a single exponential is a STRAIGHT line",
                      fontsize=9)
        axl.legend(fontsize=6.2, loc="upper right", framealpha=0.92)

        txt = (f"Ra = {tr.Ra/1e6:.1f} M$\\Omega$\nRm = {tr.Rm/1e6:.1f} M$\\Omega$\n"
               f"Cm = {tr.Cm*1e12:.0f} pF\nCm(charge) = {tr.Cm_charge*1e12:.0f} pF"
               ) if tr.valid else f"REJECTED:\n{tr.note}"
        axl.annotate(txt, (0.02, 0.03), xycoords="axes fraction", ha="left",
                     va="bottom", fontsize=7.5,
                     bbox=dict(fc="lightyellow", ec="grey", alpha=0.95))

    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# Synthetic self-test (validates the maths without needing an ABF)
# ----------------------------------------------------------------------------
def _synthetic_sweep(Ra, Rm, Cm, dV, fs=50_000, period_s=0.02, cycles=6,
                     Vhold=-0.070, noise=0.0, seed=0):
    """Ideal series-RC voltage-clamp response to a square command."""
    rng = np.random.default_rng(seed)
    Rt = Ra + Rm
    tau = Cm * Ra * Rm / Rt
    n_half = int(round(fs * period_s / 2))
    t = np.arange(cycles * 2 * n_half) / fs
    cmd = np.full(t.size, Vhold)
    cur = np.zeros(t.size)
    high = True
    for k in range(cycles * 2):
        sl = slice(k * n_half, (k + 1) * n_half)
        seg_t = (np.arange(n_half)) / fs
        V = Vhold + (dV if high else 0.0)
        cmd[sl] = V
        step = dV if high else -dV      # step taken entering this segment
        Iss = (V - Vhold) / Rt          # steady current relative to hold
        peak = step / Ra - step / Rt    # capacitive peak above the new SS
        cur[sl] = Iss + peak * np.exp(-seg_t / tau)
        high = not high
    if noise > 0:
        cur = cur + rng.normal(0, noise, cur.size)
    return t, cur, cmd, dict(Ra=Ra, Rm=Rm, Cm=Cm, Rt=Rt, tau=tau)


def _selftest():
    print("Synthetic RC validation (ideal series-resistance model)\n")
    cases = [
        dict(Ra=10e6,  Rm=500e6, Cm=20e-12, dV=-0.010),
        dict(Ra=15e6,  Rm=800e6, Cm=8e-12,  dV=-0.010),
        dict(Ra=5e6,   Rm=1.2e9, Cm=35e-12, dV=-0.005),
        # high-Ra / leaky cell: Ra > Rt/4, where the published quadratic dies
        dict(Ra=20e6,  Rm=37.7e6, Cm=174e-12, dV=0.010),
    ]
    hdr = f"{'truth Ra/Rm/Cm/tau':>34} | {'measured Ra/Rm/Cm/tau':>34}"
    print(hdr); print("-" * len(hdr))
    for c in cases:
        t, i, v, gt = _synthetic_sweep(noise=2e-12, **c)
        r = membrane_test_from_arrays(t, i, v)
        s = r.summary()
        truth = (f"{gt['Ra']/1e6:6.2f}M {gt['Rm']/1e6:6.1f}M "
                 f"{gt['Cm']*1e12:5.1f}p {gt['tau']*1e3:5.3f}ms")
        meas = (f"{s['Ra']['mean']:6.2f}M {s['Rm']['mean']:6.1f}M "
                f"{s['Cm']['mean']:5.1f}p {s['Tau']['mean']:5.3f}ms")
        print(f"{truth:>34} | {meas:>34}")
    print()
    print("Same data, published quadratic method (--method quadratic):")
    for c in cases:
        t, i, v, gt = _synthetic_sweep(noise=2e-12, **c)
        r = membrane_test_from_arrays(t, i, v, method="quadratic")
        s = r.summary()
        ok = sum(x.valid for x in r.transients)
        ra = f"{s['Ra']['mean']:6.2f}M" if ok else "  NO REAL ROOT"
        print(f"   true Ra {gt['Ra']/1e6:5.1f}M (Rt/4={gt['Rt']/4e6:5.1f}M) -> "
              f"quadratic {ra}   [{ok}/{len(r.transients)} solved]")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _parse_sweeps(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out



def _make_plot(args, res):
    """Build the QC figure for the CLI, optionally on the sweep-average trace."""
    import pyabf
    kw = dict(t1_fraction=args.t1, fit_lo=args.fit[0] / 100.0,
              fit_hi=args.fit[1] / 100.0, min_r2=args.min_r2,
              method=args.method)
    abf = pyabf.ABF(args.abf)
    sweeps = _parse_sweeps(args.sweeps) if args.sweeps else list(range(abf.sweepCount))

    if args.plot_average:
        ys = []
        for sw in sweeps:
            abf.setSweep(sw, channel=args.channel)
            ys.append(np.asarray(abf.sweepY, float))
        y = np.mean(ys, axis=0)
        abf.setSweep(sweeps[0], channel=args.channel)
        t = _to_si(np.asarray(abf.sweepX, float), "s")
        i = _to_si(y, abf.sweepUnitsY)
        if args.command_channel is not None:
            abf.setSweep(sweeps[0], channel=args.command_channel)
            c = _to_si(np.asarray(abf.sweepY, float), abf.sweepUnitsY)
        else:
            c = _to_si(np.asarray(abf.sweepC, float), abf.sweepUnitsC)
        r = membrane_test_from_arrays(t, i, c, **kw)
        title = f"{args.abf}  -  average of {len(sweeps)} sweeps"
    else:
        sw = args.plot_sweep if args.plot_sweep is not None else sweeps[0]
        abf.setSweep(sw, channel=args.channel)
        t = _to_si(np.asarray(abf.sweepX, float), "s")
        i = _to_si(np.asarray(abf.sweepY, float), abf.sweepUnitsY)
        if args.command_channel is not None:
            abf.setSweep(sw, channel=args.command_channel)
            c = _to_si(np.asarray(abf.sweepY, float), abf.sweepUnitsY)
        else:
            c = _to_si(np.asarray(abf.sweepC, float), abf.sweepUnitsC)
        r = membrane_test_from_arrays(t, i, c, **kw)
        title = f"{args.abf}  -  sweep {sw}"

    out = plot_membrane_test(t, i, c, r, out_path=args.plot, title=title,
                             fit_lo=kw["fit_lo"], fit_hi=kw["fit_hi"])
    print(f"QC figure written to {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", nargs="?", help="Path to a Membrane-Test .abf file")
    p.add_argument("--channel", type=int, default=0, help="current ADC channel (default 0)")
    p.add_argument("--command-channel", type=int, default=None,
                   help="ADC channel carrying the command waveform, if it is "
                        "recorded rather than reconstructed from the epoch table")
    p.add_argument("--sweeps", type=str, default=None,
                   help="Sweeps to pool, e.g. '0-4' or '0,2,5' (default: all)")
    p.add_argument("--t1", type=float, default=0.20,
                   help="Steady-state window as fraction of a step (default 0.20)")
    p.add_argument("--fit", type=float, nargs=2, metavar=("LO", "HI"),
                   default=(10.0, 80.0),
                   help="Proportion-of-peak %% ordinates for the tau fit "
                        "(default 10 80)")
    p.add_argument("--method", choices=("exact", "quadratic"), default="exact",
                   help="'exact' = Ra=tau/Cm_charge (default); 'quadratic' = the "
                        "published Newton-Raphson root (biased, fails if Ra>Rt/4)")
    p.add_argument("--min-r2", type=float, default=0.80,
                   help="Reject transients whose log-fit r^2 is below this")
    p.add_argument("--plot", nargs="?", const="membrane_test_qc.png",
                   default=None, metavar="PNG",
                   help="write a QC figure showing the trace, the I1/I2 windows, "
                        "the Q1 integration area and the tau fit (default: "
                        "membrane_test_qc.png)")
    p.add_argument("--plot-sweep", type=int, default=None,
                   help="which sweep to plot (default: first analysed sweep)")
    p.add_argument("--plot-average", action="store_true",
                   help="plot the across-sweep average trace instead of one sweep "
                        "(cleaner for checking the fit by eye)")
    p.add_argument("--per-transient", action="store_true",
                   help="Also print every accepted transient")
    p.add_argument("--selftest", action="store_true",
                   help="Run the synthetic validation and exit")
    args = p.parse_args(argv)

    if args.selftest:
        _selftest()
        return
    if not args.abf:
        p.error("provide an .abf file, or use --selftest")

    res = membrane_test_from_abf(
        args.abf,
        channel=args.channel,
        command_channel=args.command_channel,
        sweeps=_parse_sweeps(args.sweeps) if args.sweeps else None,
        t1_fraction=args.t1,
        fit_lo=args.fit[0] / 100.0,
        fit_hi=args.fit[1] / 100.0,
        min_r2=args.min_r2,
        method=args.method,
    )

    if args.plot:
        _make_plot(args, res)

    if args.per_transient:
        print(f"{'edge':>7} {'dir':>3} {'Ra(M)':>8} {'Rm(M)':>9} "
              f"{'Cm(pF)':>8} {'Cmq(pF)':>8} {'Tau(ms)':>8} {'r2':>6}")
        for tr in res.transients:
            d = "up" if tr.direction > 0 else "dn"
            if tr.valid:
                print(f"{tr.edge_index:>7} {d:>3} {tr.Ra/1e6:8.2f} "
                      f"{tr.Rm/1e6:9.1f} {tr.Cm*1e12:8.1f} "
                      f"{tr.Cm_charge*1e12:8.1f} "
                      f"{tr.tau*1e3:8.3f} {tr.r2_fit:6.3f}")
            else:
                print(f"{tr.edge_index:>7} {d:>3}  {tr.note}")
        print()
    print(res)


if __name__ == "__main__":
    main()
