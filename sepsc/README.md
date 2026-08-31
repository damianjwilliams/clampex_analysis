# sepsc — spontaneous EPSC detection, curation & classification

A pipeline of small, single-purpose command-line/GUI tools for finding, verifying, curating,
and classifying spontaneous excitatory postsynaptic currents (sEPSCs) in gap-free
voltage-clamp `.abf` recordings (pCLAMP/Clampex, read via [pyabf](https://pypi.org/project/pyabf/)).

Every tool shares one raw-trace convention (gap-free voltage-clamp `.abf`, events
identified by a sample index into that trace) and one output-file convention
(`<recording_stem>_<tool>_<thing>.<ext>`, always written next to the source `.abf`), so
outputs from one tool are the inputs to the next without any extra glue.

Run any tool two ways:

```
python -m sepsc <command> [args...]        # dispatcher, see `python -m sepsc --help`
python -m sepsc.<module> [args...]         # equivalent, direct module invocation
```

`python -m sepsc <command> --help` always shows that command's own full flag list.

## Contents

- [Environment](#environment)
- [Two detection methods, and how to tune them](#two-detection-methods-and-how-to-tune-them)
  - [`minianalysis` — classical threshold detector](#minianalysis--classical-threshold-detector)
  - [`optimize` — live parameter tuning](#optimize--live-parameter-tuning)
  - [`inspect` — verify a finished detection run](#inspect--verify-a-finished-detection-run)
  - [`detect` — miniML (CNN-LSTM) detector](#detect--minicml-cnn-lstm-detector)
  - [`fastmini` — third method, per-recording MLP](#fastmini--third-method-per-recording-mlp)
- [Curation](#curation)
  - [`review` — accept/reject candidates by eye](#review--acceptreject-candidates-by-eye)
  - [`view` — browse a whole trace with every detector's events marked](#view--browse-a-whole-trace-with-every-detectors-events-marked)
- [Fast/slow kinetic classification](#fastslow-kinetic-classification)
  - [`label` → `train` → `classify` → `overlay`](#label--train--classify--overlay)
- [Cross-method comparison](#cross-method-comparison)
  - [`compare`](#compare)
- [Shared preprocessing](#shared-preprocessing)
  - [`preprocess`](#preprocess)
- [Shared infrastructure (not run directly)](#shared-infrastructure-not-run-directly)
- [Output file naming reference](#output-file-naming-reference)
- [Typical workflows](#typical-workflows)

## Environment

- The project's main virtualenv (`.venv`) has everything except TensorFlow.
- **`detect.py` (miniML) needs TensorFlow**, which lives in a **separate `clampex_miniml`
  conda environment**, not the main venv. Run it with that interpreter explicitly:
  ```
  <path-to-anaconda3>\envs\clampex_miniml\python.exe -m sepsc detect recording.abf
  ```
  The `sepsc` dispatcher (`python -m sepsc detect`) prints this hint automatically if you
  run it from the wrong interpreter.
- The interactive tools (`optimize`, `inspect`'s window, `view`, `review`) need **PyQt5**
  (already in `.venv`) and a working Qt backend; `label` is the one exception, still built
  on Tk — none of them work over a headless/no-display session.
- `fastmini.py` needs a local clone of
  [mrreganwang/Mini_Scripts](https://github.com/mrreganwang/Mini_Scripts) at
  `_external/Mini_Scripts` (or pass `--repo-path`). Every other tool has no such
  dependency.

## Two detection methods, and how to tune them

### `minianalysis` — classical threshold detector

Reimplements the Synaptosoft *Mini Analysis Program*'s method: find a local maximum, find
its baseline, compare amplitude/area to thresholds, and search fixed windows before/after
the peak for the onset and decay crossings — with a candidate rejected outright (not just
truncated) if the onset or decay point is never reached within its search window.

```
python -m sepsc minianalysis recording.abf
```

By default this **opens a parameters dialog first** (all 9 detection parameters, plus
filter/resample settings, pre-filled with defaults or whatever `--flags` you also passed) —
edit values and click OK to run, or Cancel to abort without detecting. Pass `--no-gui` to
skip the dialog and run immediately from the command line (for scripts/automation).

Key flags (all also settable in the GUI dialog):

| Flag | Meaning |
|---|---|
| `--direction {negative,positive}` | peak polarity (inward sEPSCs are negative, the default) |
| `--amplitude-threshold` (a) | minimum \|amplitude\| to accept |
| `--area-threshold` (b) | minimum \|area\| to accept |
| `--n-avg-peak` | points averaged to read the peak value |
| `--search-local-max-ms` (c) | min. separation between candidate peaks |
| `--baseline-before-ms` (d) / `--baseline-avg-ms` (e) | baseline window: ends `d` before the peak, spans `e` |
| `--decay-search-ms` (f) / `--decay-fraction` (g) | how far past the peak to search for `g`-fraction decay; reject if never reached |
| `--onset-fraction` / `--onset-search-ms` | same idea on the rise side (this reimplementation's own addition, not one of the original 9) |
| `--adjust-overlapping-baseline` | opt-in: extrapolate the previous event's decay for a closely-spaced event's baseline instead of the plain window average (off by default) |
| `--filter` / `--cutoff-hz` / `--target-rate-hz` / `--filter-order` | Bessel-filter + resample before detecting (see `preprocess`) |

Analysis add-ons (all optional, all `--no-gui`-compatible):
`--stats`, `--histogram-column <col>` (+`--hist-bin-size`), `--autocorr`, `--cross-corr-csv <other_events.csv>`,
`--group-analysis` (+`--group-pre-ms`/`--group-post-ms`), `--fit-decay {peak_to_end,decay_10_90,decay_20_80,custom}`
(+`--fit-n-exp {1,2}`) — a Simplex single/double-exponential fit to the averaged trace.

**Output** (stem gets a `_filt<cutoff>Hz<rate>Hz` suffix if `--filter` was used):
- `<stem>_minianalysis_events.csv` — one row per event: `peak_idx`, `peak_time_s`, `baseline`,
  `amplitude`, `rise_time_ms`, `decay_time_ms`, `area`, `inter_event_interval_ms`
- `<stem>_minianalysis_params.json` — the exact parameters used (read automatically by
  `inspect`/`optimize`)
- `<stem>_minianalysis_trace.png` (unless `--no-plot`)
- Analysis add-ons write their own `<stem>_minianalysis_{stats,hist_<col>,autocorr,crosscorr,group,decayfit}.{csv,png,txt}`

### `optimize` — live parameter tuning

```
python -m sepsc optimize recording.abf
```

A 3-panel window for dialing in parameters *before* committing to a full run — no
pre-existing detection output needed:

- **Left**: every detection parameter (live-edited, read fresh on each click) plus a
  Filter/resample panel (its own explicit Apply button, since re-filtering a whole trace
  is too expensive to redo on every click) and a **"Run full detection with these
  parameters"** button that writes the same output `minianalysis` would.
- **Top-right**: the full trace, pan/zoom in both X and Y (pyqtgraph).
- **Bottom-right**: click any peak above and it's measured immediately with the current
  left-panel values — same annotated view as `inspect`, but it also shows **why** a
  candidate would be *rejected* (which threshold or search window it failed), which
  `inspect` can't since it only ever shows already-accepted events. Change a parameter and
  click the same peak again to see the new result.

`--filter` starts with filtering on (still live-editable afterward); all detection-param
flags are the same as `minianalysis`, used as the dialog's starting values.

### `inspect` — verify a finished detection run

```
python -m sepsc inspect recording.abf
```

Click a detected event (in a zoomable overview) to see every window/threshold the
detector actually used to accept it, annotated on the real trace — baseline window, decay
search window, amplitude threshold, decay-fraction level, etc. — plus an annotation-key
legend and the exact parameters (read from `<stem>_minianalysis_params.json`, so this is
always what *actually* produced the events, not just today's defaults). `Right`/`N` and
`Left`/`P` step through events. `--filter` must match whatever `minianalysis --filter`
settings produced the CSV you're inspecting (peak indices are into the filtered trace).
Also doubles as a QC tool (Accept/Reject each event, same output as `review`).

Same window/controls work on `detect`'s (miniML) output too, with `--source miniml`:

```
python -m sepsc inspect recording.abf --source miniml
```

miniML has no windowed detection-parameters to annotate (the model outputs a peak
location directly, with no persisted baseline/onset/decay search window), so the detail
view is simpler — the raw snippet around the model's peak, a derived baseline line, and
whichever of score/amplitude/charge/risetime/decaytime/halfwidth/interval are present in
that event's row — but it's the same overview-click-to-inspect window, QC workflow, and
output files (`<stem>_miniML_reviewed.csv` / `_review_progress.csv`) as the default
`--source minianalysis`.

`fastmini`'s output works the same way, with `--source fastmini` — also no baseline-window/
local-max-search/amplitude-area-threshold concept (its "threshold" is the peel-off
confidence curve's own prominence cutoff, not a per-event window), but unlike miniML its
`*_fastmini_events.csv` *does* persist a real baseline plus rise/decay times for every
event, so the onset/decay points and shaded area **are** reconstructed exactly (not just a
derived baseline line). `fastmini` always preprocesses internally at a fixed 3 kHz/10 kHz
filter with no raw-trace mode of its own, so `--filter`/`--cutoff-hz`/`--target-rate-hz` are
ignored for this source (a NOTE prints if you pass them anyway).

### `detect` — miniML (CNN-LSTM) detector

```
<clampex_miniml env>\python.exe -m sepsc detect recording.abf --threshold 0.5
```

Runs the [miniML](https://github.com/delvendahl/miniML) pretrained model, auto-rescaling
its window/smoothing sizes for the recording's actual sample rate vs. the model's native
50 kHz training config. Needs the `clampex_miniml` conda env (see
[Environment](#environment)).

**Output**: `<stem>_miniML_individual.csv` (+ miniML's own other CSV outputs),
`<stem>_miniML_trace.png` if `--plot`.

### `fastmini` — third method, per-recording MLP

```
python -m sepsc fastmini recording.abf
```

A different approach entirely: trains a small feedforward classifier **from scratch on
this recording's own noise**, synthetic template events, and an iterative peel-off
detection loop (adapted from
[mrreganwang/Mini_Scripts](https://github.com/mrreganwang/Mini_Scripts) — needs that repo
cloned locally, see [Environment](#environment)). Fixed at the method's native convention
(10 kHz, 30 ms analysis windows) — recordings are auto-preprocessed to match.

**Output**: `<stem>_fastmini_model.pkl` (cached — pass `--retrain` to rebuild),
`<stem>_fastmini_events.csv` (same `location`/`location_s`/`baseline`/`amplitude`/
`rise_time_ms`/`decay_time_ms`/`area` schema as `minianalysis`'s, so `review`/`compare`/
`view` all load it with no extra wiring), `<stem>_fastmini_trace.png`.

## Curation

### `review` — accept/reject candidates by eye

```
python -m sepsc review recording.abf --source minianalysis   # or --source miniml/fastmini
```

Same window as `inspect` (built directly on its `EventInspector`, not a separate window) —
a zoomable overview of the whole trace with every candidate marked, click one (or Right/N,
Left/P) to see it in the detail panel below, Accept/Reject by button or arrow keys (Right/A
= accept, Left/R = reject) or "Accept All Remaining". The one thing `inspect` doesn't have:
for `--source minianalysis`/`miniml`, if the *other* one's output also exists next to the
`.abf`, its candidates are marked on the overview too (a distinct triangle), and each
event's detail title states whether it was also flagged by that other method — `--source
fastmini` has no such automatic comparison (no single natural "other" detector to diff a
third method against), so it reviews on its own. `--filter` works the same as `inspect`'s
(and is ignored for `--source fastmini`, same as `inspect`'s). Progress
autosaves after every decision (identical output schema to `inspect`'s own QC output — the
two are fully interchangeable, and reviewing/inspecting the same source resumes the other's
progress), so re-running resumes where you left off; unlike a plain queue, already-decided
events stay visible (with a colored halo) and clickable, so revisiting one is just clicking
it again. `--shuffle`/`--sample` restrict the overview/navigation to a randomized/subsampled
subset for that session.

**Output**: `<stem>_<source>_reviewed.csv` (accepted events only),
`<stem>_<source>_review_progress.csv` (every decision, for resuming).

### `view` — browse a whole trace with every detector's events marked

```
python -m sepsc view recording.abf
```

Read-only PyQt/pyqtgraph viewer: the whole recording with every detector's events marked
(mouse-wheel zoom, click-drag pan) — for eyeballing detection coverage across a full
recording at a glance, with no per-event detail panel or QC controls (unlike
`review`/`inspect`). Prefers each source's reviewed/accepted CSV if one exists, falling
back to the raw detector output otherwise.

## Fast/slow kinetic classification

A separate sub-pipeline: hand-label some example fast- and slow-decay events, train a
classifier on them, then apply it to every detected event in a recording.

### `label` → `train` → `classify` → `overlay`

**`label`** — click the raw trace to hand-label training events (left-click = fast,
right-click = slow); a click snaps to the true local peak/baseline/onset/decay via
threshold-crossing heuristics.
```
python -m sepsc label recording.abf
```
Output: `<stem>_training_events.csv`, `<stem>_training_windows.npz`.

**`train`** — trains a logistic-regression fast/slow classifier from one or more
recordings' labeled events (5 features per event: amplitude, 10-90% rise time, half-decay
time, decay tau, charge — see `features.py`), with stratified cross-validation.
```
python -m sepsc train recording1.abf recording2.abf ...
```
Output: `sEPSC_fast_slow_classifier.joblib` + `..._meta.json`.

**`classify`** — applies the trained classifier to every miniML-detected event in a
recording.
```
python -m sepsc classify recording.abf
```
Output: `<stem>_miniML_classified.csv` (+ `predicted_label`/`prob_fast`/`prob_slow`
columns), `<stem>_miniML_classified.png`.

**`overlay`** — peak-aligned, amplitude-normalized overlay plot comparing the hand-labeled
fast vs. slow populations' kinetics (individual traces + bold class means).
```
python -m sepsc overlay recording1.abf recording2.abf ...
```
Output: `sEPSC_fast_slow_overlay.png`.

## Cross-method comparison

### `compare`

```
python -m sepsc compare recording.abf
```

Compares `miniML` vs. `minianalysis` output on the same recording (both must already be
run): amplitude CDF/histogram/box-and-dot panel, plus peak-aligned average waveforms
(unscaled and peak-normalized).

**Output**: `<stem>_method_comparison.png`, `<stem>_method_avg_trace.png`.

## Shared preprocessing

### `preprocess`

The Bessel low-pass + downsample step every other tool's `--filter` flag uses internally
(`sepsc.preprocess.load_filtered_trace` etc.) — also usable standalone:

```
python -m sepsc preprocess recording.abf --cutoff-hz 3000 --target-rate-hz 10000
```

Checks the ABF header's own telegraphed hardware filter setting first, so it can flag
already-redundant filtering. Filters at the native rate before decimating (correct
anti-aliasing order).

**Output**: `<stem>_filtered_<cutoff>Hz_<rate>Hz.npz`, `..._preview.png`.

## Shared infrastructure (not run directly)

- **`features.py`** — the 5-feature set (`amplitude_pA`, `rise_10_90_ms`, `half_decay_ms`,
  `tau_decay_ms`, `charge_pA_ms`) and peak/baseline-location logic shared by `train`,
  `classify`, and `overlay`, so they always agree on what "amplitude"/"rise time"/etc. mean
  for a given event window.
- **`style.py`** — shared colors (`COLORS["fast"]`/`["slow"]`) and chart-chrome tokens used
  by every plot in the package, so fast/slow and light/dark styling stay consistent
  everywhere.
- **`gui_utils.py`** — `safe_callback`, a decorator used by the Tk/matplotlib GUIs
  (`review`, `label`) so an exception inside a button/key callback gets printed instead of
  silently freezing the window.
- **`cli.py`** — the `python -m sepsc <command>` dispatcher (this is what `__main__.py`
  calls); forwards all remaining args to that command's own `argparse` parser untouched.

## Output file naming reference

All outputs are written next to the source `.abf`, stem = `<recording>` (or
`<recording>_filt<cutoff>Hz<rate>Hz` if detected on a filtered trace).

| Pattern | Written by |
|---|---|
| `<stem>_minianalysis_events.csv` / `_params.json` / `_trace.png` | `minianalysis`, `optimize` ("Run full detection") |
| `<stem>_minianalysis_{stats,hist_<col>,autocorr,crosscorr,group,decayfit}.*` | `minianalysis` analysis flags |
| `<stem>_miniML_individual.csv` / `_trace.png` | `detect` |
| `<stem>_miniML_classified.csv` / `.png` | `classify` |
| `<stem>_fastmini_model.pkl` / `_events.csv` / `_trace.png` | `fastmini` |
| `<stem>_<source>_reviewed.csv` / `_review_progress.csv` | `review`, `inspect` (`source` = `miniml` or `minianalysis`) |
| `<stem>_training_events.csv` / `_training_windows.npz` | `label` |
| `sEPSC_fast_slow_classifier.joblib` / `_meta.json` | `train` |
| `sEPSC_fast_slow_overlay.png` | `overlay` |
| `<stem>_method_comparison.png` / `_avg_trace.png` | `compare` |
| `<stem>_filtered_<cutoff>Hz_<rate>Hz.npz` / `_preview.png` | `preprocess` |

## Typical workflows

**Just want events from one recording, classical method:**
```
python -m sepsc minianalysis recording.abf        # tune params in the dialog, or --no-gui
python -m sepsc inspect recording.abf              # verify what it actually did
```

**Not sure what parameters to use yet:**
```
python -m sepsc optimize recording.abf             # click around, tune live, then "Run full detection"
```

**Full pipeline with miniML + fast/slow classification (needs `clampex_miniml` env for `detect`):**
```
<clampex_miniml env>\python.exe -m sepsc detect recording.abf
python -m sepsc review recording.abf --source miniml
python -m sepsc label training_recording.abf       # once, to build a training set
python -m sepsc train training_recording.abf
python -m sepsc classify recording.abf
python -m sepsc overlay training_recording.abf
```

**Sanity-check two methods against each other:**
```
python -m sepsc minianalysis recording.abf --no-gui
<clampex_miniml env>\python.exe -m sepsc detect recording.abf
python -m sepsc compare recording.abf
```
