"""
Third sEPSC detection method: a per-recording-trained scikit-learn MLP
classifier + iterative peel-off, adapted from Reagan Wang's Mini_Scripts
(https://github.com/mrreganwang/Mini_Scripts, no LICENSE file -- used here
by importing the user's own local clone at runtime, exactly like detect.py
imports the separately-installed miniml package, never by copying its code
into this project).

How it differs from miniML/minianalysis: a small feedforward network
(200-100-100, logistic activation) is trained from scratch on THIS
recording, classifying short (SL=300 sample, 30ms @ 10kHz) windows as
"signal" (a synthetic canonical mEPSC shape, jittered in timescale/
amplitude and added to this recording's own noise) vs. "noise" (a quiet
stretch of this recording's own raw trace). Detection then slides that
window across the trace, and iteratively erases each round's detections
(replaced with a noise snippet) before re-running the classifier, so
overlapping/closely-spaced events aren't hidden from later iterations by
earlier ones still sitting in the window.

Why per-recording training, not the repo's bundled pretrained network: its
canonical template (FastMini.npy) is amplitude-normalized (peak=1.0), and
its training pipeline's amplitude convention (K * 10, K~2 by their own
example) was calibrated against ITS OWN noise file's specific scale/units
-- a decision boundary tuned to that specific SNR, not to absolute pA. That
has no reason to match a different recording's amplitude scale (confirmed:
their example noise file's std is ~5x smaller than that same example's own
raw trace std, so even their own convention isn't simply "raw pA"). Training
fresh from THIS recording's own noise and a target amplitude derived from
THIS recording (via --target-snr x its own noise SD) keeps the calibration
meaningful regardless of scale.

Fixed at the method's own native convention (10 kHz, SL=300 samples = 30ms,
mpd=4, mpw=3, prominence=0.95, peak sits ~69 samples into its window) --
these came from the repo's own example (2022_02_28_0000.abf, confirmed
10 kHz) and its FastMini/SlowMini templates' own argmax -- so every
recording is run through sepsc.preprocess (3 kHz Bessel + 10 kHz downsample
by default) first, the same way this method's author's own example data
already was, rather than rescaling these constants per-recording.

Output (next to the source .abf):
    <name>_fastmini_model.pkl     the trained classifier (reused on rerun
                                    unless --retrain)
    <name>_fastmini_events.csv    location/location_s/baseline/amplitude/
                                    rise_time_ms/decay_time_ms/area -- same
                                    schema as minianalysis.py's Event, so
                                    review.py/compare.py/view.py can load it
                                    via their existing --minianalysis-csv
                                    override with no extra wiring
    <name>_fastmini_trace.png

Usage
-----
    python -m sepsc.fastmini path\\to\\recording.abf
    python -m sepsc.fastmini recording.abf --repo-path C:\\path\\to\\Mini_Scripts
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

from .features import locate_peak_and_baseline
from .preprocess import load_filtered_trace
from .style import SURFACE, TRACE, INK, MUTED

SL = 300                  # analysis window, samples -- 30ms @ this method's native 10 kHz
NATIVE_RATE_HZ = 10_000.0
PEAK_OFFSET = 69          # sample offset from a window's start to its peak (= argmax of the templates)
DETREND_WINDOW = 10_000   # samples (~1s @ native rate) -- matches detect_peaks.py's own slow-drift
                          # removal; without it, a trace's DC baseline offset saturates the
                          # classifier's logistic-activation hidden units regardless of local shape


def detrend(v: np.ndarray) -> np.ndarray:
    """Remove slow (~1s) baseline drift before it reaches the classifier --
    ports detect_peaks.py's own `trace - moving_average(trace)` step, which
    is easy to miss since it's uncommented but easy to lose sight of amid
    that script's interactive-plotting code."""
    return v - uniform_filter1d(v, size=DETREND_WINDOW, mode="nearest")
DEFAULT_REPO_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "_external", "Mini_Scripts", "python"))


def _load_repo_functions(repo_path: str):
    """Import the repo's pure helper functions (no side effects on import,
    unlike its train_net.py/detect_peaks.py scripts) from the user's own
    local clone. Raises a clear error with clone instructions if missing,
    rather than a bare ModuleNotFoundError."""
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"Mini_Scripts not found at {repo_path!r}. Clone it first:\n"
            f"    git clone https://github.com/mrreganwang/Mini_Scripts.git "
            f"{os.path.dirname(repo_path)!r}\n"
            f"or pass --repo-path pointing at your own clone's 'python' subfolder.")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from helper_functions.MakeCircMatData import make_circ_mat_data
    from helper_functions.MakeMiniMatFS import make_mini_mat_fs
    from helper_functions.MakeTrainMat import make_train_mat
    fast_template = np.load(os.path.join(repo_path, "helper_functions", "FastMini.npy"))
    slow_template = np.load(os.path.join(repo_path, "helper_functions", "SlowMini.npy"))
    return make_circ_mat_data, make_mini_mat_fs, make_train_mat, fast_template, slow_template


def find_quiet_noise_reference(v: np.ndarray, fs: float, target_seconds: float = 5.0,
                                window_ms: float = 50.0, quiet_percentile: float = 15.0) -> np.ndarray:
    """Automatically pick quiet stretches of THIS trace as a noise
    reference, replacing the repo's original interactive click-to-select
    (make_noise_from_file) -- ranks samples by local rolling SD and keeps
    the quietest contiguous runs (longest first) up to target_seconds."""
    win = max(3, int(round(window_ms / 1000.0 * fs)))
    rolling_std = pd.Series(v).rolling(win, center=True, min_periods=win).std().to_numpy()
    threshold = np.nanpercentile(rolling_std, quiet_percentile)
    quiet = np.nan_to_num(rolling_std, nan=np.inf) <= threshold
    quiet[:win] = False
    quiet[-win:] = False

    idx = np.where(quiet)[0]
    if len(idx) == 0:
        raise ValueError("No sufficiently quiet stretch found for a noise reference -- "
                          "try a higher --noise-quiet-percentile")
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    runs.sort(key=len, reverse=True)

    target_n = int(round(target_seconds * fs))
    pieces, total = [], 0
    for run in runs:
        pieces.append(v[run[0]:run[-1] + 1])
        total += len(run)
        if total >= target_n:
            break
    return np.concatenate(pieces)


def train_classifier(noise_ref: np.ndarray, target_amplitude: float, fast_template, slow_template,
                      make_mini_mat_fs, make_train_mat, n_events: int = 6000, mode: int = 1,
                      hidden_layer_sizes=(200, 100, 100), max_iter: int = 2000, random_state: int = 0):
    """Train a fresh MLP classifier for THIS recording. target_amplitude is
    the desired synthetic peak amplitude in the trace's own units (e.g.
    pA) -- divided by 10 here to invert make_train_mat's own *10
    multiplier, so the caller can reason in real trace units instead of
    the repo's internal K/amp convention."""
    from sklearn.neural_network import MLPClassifier

    K = abs(target_amplitude) / 10.0
    synthetic_minis, _, _ = make_mini_mat_fs(n_events, SL, K, mode, fast_template, slow_template)
    X, y = make_train_mat(-synthetic_minis, noise_ref, n_events, SL)
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation="logistic",
                         max_iter=max_iter, random_state=random_state)
    mlp.fit(X, y)
    return mlp


def detect_fastmini(v: np.ndarray, model, noise_ref: np.ndarray, make_circ_mat_data,
                     mpd: int = 4, mpw: int = 3, prominence: float = 0.95,
                     cv_smooth: int = 10, max_iterations: int = 20, verbose: bool = True):
    """Iterative peel-off detection (detect_peaks.py's core loop, minus its
    per-iteration interactive matplotlib pan/zoom window). Returns raw
    window-start indices (NOT yet trace peak locations -- add PEAK_OFFSET
    and refine, see main())."""
    smoothed = v.astype(float).copy()
    pk_list: list[int] = []
    for iteration in range(1, max_iterations + 1):
        proba = model.predict_proba(make_circ_mat_data(SL, smoothed))[:, 1]
        proba = np.clip(proba, 0, None)
        cv = np.convolve(proba, np.ones(cv_smooth) / cv_smooth, mode="same")
        pks, _ = find_peaks(cv, distance=mpd, prominence=prominence, width=mpw)
        if len(pks) == 0:
            if verbose:
                print(f"  iteration {iteration}: 0 new peaks -- done")
            break
        pk_list.extend(int(p) for p in pks)
        for p in pks:
            i0, i1 = max(0, p + 20), min(len(smoothed), p + 170)
            n = i1 - i0
            if n > 0:
                smoothed[i0:i1] = noise_ref[:n]
        if verbose:
            print(f"  iteration {iteration}: {len(pks)} new peaks (total {len(pk_list)})")
    return np.array(sorted(set(pk_list)), dtype=np.int64)


def _measure_event(v: np.ndarray, dt: float, center: int, pre_ms: float, post_ms: float,
                    direction: str, smooth_samples: int, onset_fraction: float, decay_fraction: float):
    """Peak/baseline/rise/decay/area for one event, centered on `center` --
    same measurement conventions as minianalysis.py's steps 4-6, reused
    here so all three detectors' output CSVs are comparable, not just
    schema-compatible."""
    pre_n = int(round(pre_ms / 1000.0 / dt))
    post_n = int(round(post_ms / 1000.0 / dt))
    i0, i1 = center - pre_n, center + post_n
    if i0 < 0 or i1 > len(v):
        return None
    window = v[i0:i1]
    baseline, peak_idx, peak_v, _ = locate_peak_and_baseline(window, dt, pre_ms, direction, smooth_samples)
    amplitude = peak_v - baseline
    if amplitude == 0:
        return None

    pol = -1.0 if direction == "negative" else 1.0
    sw = pol * window
    s_baseline, s_peak = pol * baseline, pol * peak_v
    span = s_peak - s_baseline
    if span <= 0:
        return None

    onset_level = s_baseline + onset_fraction * span
    onset_idx = 0
    for k in range(peak_idx, -1, -1):
        if sw[k] < onset_level:
            onset_idx = k
            break
    rise_time_ms = (peak_idx - onset_idx) * dt * 1e3

    decay_level = s_baseline + (1.0 - decay_fraction) * span
    decay_idx = len(window) - 1
    for k in range(peak_idx, len(window)):
        if sw[k] <= decay_level:
            decay_idx = k
            break
    decay_time_ms = (decay_idx - peak_idx) * dt * 1e3

    area = float(_trapz(window[onset_idx:decay_idx + 1] - baseline, dx=dt * 1e3))
    peak_location = i0 + peak_idx
    return dict(location=peak_location, peak_time_s=peak_location * dt, baseline=baseline,
                amplitude=amplitude, rise_time_ms=rise_time_ms, decay_time_ms=decay_time_ms, area=area)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", help="Path to the gap-free voltage-clamp .abf file")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--repo-path", default=DEFAULT_REPO_PATH,
                   help="path to Mini_Scripts' 'python' subfolder (default: "
                        "_external/Mini_Scripts/python next to this project)")
    p.add_argument("--direction", choices=["negative", "positive"], default="negative")
    p.add_argument("--target-snr", type=float, default=8.0,
                   help="synthetic training peak amplitude, as a multiple of this recording's "
                        "own noise-reference SD (default 8, matching the repo's own example ratio)")
    p.add_argument("--n-training-events", type=int, default=6000,
                   help="synthetic training events, half fast-shaped half slow-shaped (default 6000)")
    p.add_argument("--max-iter", type=int, default=2000, help="MLP training max_iter (default 2000)")
    p.add_argument("--noise-seconds", type=float, default=5.0,
                   help="target length of the auto-extracted quiet-stretch noise reference, "
                        "seconds (default 5)")
    p.add_argument("--noise-quiet-percentile", type=float, default=15.0,
                   help="rolling-SD percentile used to call a stretch 'quiet' (default 15)")
    p.add_argument("--mpd", type=int, default=4, help="min peak distance in the confidence curve (default 4)")
    p.add_argument("--mpw", type=int, default=3, help="min peak width in the confidence curve (default 3)")
    p.add_argument("--prominence", type=float, default=0.95,
                   help="min confidence-curve peak prominence, 0-1 (default 0.95)")
    p.add_argument("--max-detect-iterations", type=int, default=20,
                   help="cap on peel-off iterations (default 20)")
    p.add_argument("--onset-fraction", type=float, default=0.1)
    p.add_argument("--decay-fraction", type=float, default=0.5)
    p.add_argument("--retrain", action="store_true",
                   help="retrain even if a cached <name>_fastmini_model.pkl already exists")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    make_circ_mat_data, make_mini_mat_fs, make_train_mat, fast_template, slow_template = \
        _load_repo_functions(args.repo_path)

    stem = os.path.splitext(args.abf)[0]
    t, v, fs, hardware_filter_hz = load_filtered_trace(
        args.abf, channel=args.channel, cutoff_hz=3000.0, target_hz=NATIVE_RATE_HZ)
    dt = 1.0 / fs
    print(f"Preprocessed to {fs:.0f} Hz (this method's native SL={SL} convention) -- "
          f"hardware filter was {hardware_filter_hz} Hz", flush=True)

    # Classifier operates on the DETRENDED trace (see detrend()'s docstring
    # -- a DC baseline offset otherwise saturates the logistic hidden
    # units); final amplitude/kinetics are still measured on the original
    # v further down, so output units match minianalysis/miniML.
    v_dt = detrend(v)

    noise_ref = find_quiet_noise_reference(
        v_dt, fs, args.noise_seconds, quiet_percentile=args.noise_quiet_percentile)
    noise_sd = float(noise_ref.std())
    target_amplitude = args.target_snr * noise_sd
    print(f"Noise reference: {len(noise_ref) / fs:.1f}s, SD={noise_sd:.2f} -- "
          f"training target amplitude {target_amplitude:.1f} (={args.target_snr}x SD)", flush=True)

    model_path = f"{stem}_fastmini_model.pkl"
    if os.path.exists(model_path) and not args.retrain:
        import joblib
        model = joblib.load(model_path)
        print(f"Loaded cached model -> {model_path} (pass --retrain to rebuild)", flush=True)
    else:
        print(f"Training MLP on {args.n_training_events} synthetic events "
              f"(max_iter={args.max_iter})...", flush=True)
        model = train_classifier(noise_ref, target_amplitude, fast_template, slow_template,
                                  make_mini_mat_fs, make_train_mat,
                                  n_events=args.n_training_events, max_iter=args.max_iter)
        import joblib
        joblib.dump(model, model_path)
        print(f"Saved -> {model_path}", flush=True)

    print("Detecting (iterative peel-off)...", flush=True)
    window_starts = detect_fastmini(v_dt, model, noise_ref, make_circ_mat_data,
                                     mpd=args.mpd, mpw=args.mpw, prominence=args.prominence,
                                     max_iterations=args.max_detect_iterations)

    events = []
    for w in window_starts:
        center = int(w) + PEAK_OFFSET
        result = _measure_event(v, dt, center, pre_ms=5.0, post_ms=20.0, direction=args.direction,
                                 smooth_samples=15, onset_fraction=args.onset_fraction,
                                 decay_fraction=args.decay_fraction)
        if result is not None:
            events.append(result)

    df = pd.DataFrame(events)
    if not df.empty:
        df = df.rename(columns={"peak_time_s": "location_s"})
        df = df[["location", "location_s", "baseline", "amplitude", "rise_time_ms", "decay_time_ms", "area"]]
        df = df.sort_values("location").drop_duplicates(subset="location").reset_index(drop=True)

    out_csv = f"{stem}_fastmini_events.csv"
    df.to_csv(out_csv, index=False)

    if len(df):
        rate_hz = len(df) / (len(v) * dt)
        print(f"Detected {len(df)} events ({rate_hz:.3f} Hz), "
              f"mean amplitude {df['amplitude'].mean():.2f}, median {df['amplitude'].median():.2f}")
    else:
        print("Detected 0 events -- check --target-snr/thresholds.")
    print(f"Saved -> {out_csv}")

    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(16, 4.5))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        ax.plot(t, v, color=TRACE, lw=0.3, zorder=1)
        if len(df):
            ax.plot(df["location_s"], v[df["location"].to_numpy()], "x", color="crimson",
                     ms=6, mew=1.2, zorder=3, label=f"{len(df)} detected events")
            ax.legend(loc="upper right", fontsize=9, labelcolor=INK)
        ax.set_xlabel("time (s)", color=INK)
        ax.set_ylabel("current (filtered/downsampled)", color=INK)
        ax.set_title(f"{os.path.basename(args.abf)} -- fastmini (MLP peel-off) detection", color=INK)
        ax.tick_params(colors=MUTED)
        fig.tight_layout()
        out_png = f"{stem}_fastmini_trace.png"
        fig.savefig(out_png, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
