# clampex_analysis

Analysis tools for pCLAMP/Clampex `.abf` electrophysiology recordings (read via
[pyabf](https://pypi.org/project/pyabf/)): action-potential feature extraction,
membrane-test (Ra/Rm/Cm) calculation, and the [`sepsc`](sepsc/README.md) package
for spontaneous EPSC detection, curation, and classification.

## Setup

Requires Python 3.11+ on Windows.

1. **Create the main virtual environment** (everything except TensorFlow):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   This covers every tool here except `sepsc detect` (see below).

2. **`sepsc detect` (miniML) needs a separate conda environment with TensorFlow**,
   since TensorFlow's Windows build conflicts with the rest of this stack:
   ```
   conda create -n clampex_miniml python=3.11
   conda activate clampex_miniml
   pip install tensorflow pyabf numpy pandas matplotlib scipy scikit-learn
   ```
   Run `detect` with that interpreter explicitly:
   ```
   <path-to-anaconda3>\envs\clampex_miniml\python.exe -m sepsc detect recording.abf
   ```

3. **`sepsc fastmini` needs a local clone of
   [mrreganwang/Mini_Scripts](https://github.com/mrreganwang/Mini_Scripts)** at
   `_external/Mini_Scripts` (or pass `--repo-path`):
   ```
   git clone https://github.com/mrreganwang/Mini_Scripts.git _external/Mini_Scripts
   ```

4. The interactive tools (`sepsc optimize`, `inspect`, `view`, `review`) need
   **PyQt5** (already in `requirements.txt`) and a working Qt backend/display —
   none of them run over a headless session.

## Contents

| Path | Purpose |
|---|---|
| [`sepsc/`](sepsc/README.md) | sEPSC detection, curation, and fast/slow classification pipeline — see its own README for full usage |
| `autorun_AP_analysis.py` | Action-potential feature extraction from current-clamp recordings, via the Allen Institute's [ipfx](https://github.com/AllenInstitute/ipfx) |
| `batch_cell_analysis.py` | Batch driver: one folder per cell (`YYMMDD_s<slice>_c<cell>_<RMP>`), one summary row per cell — membrane test, sag / input resistance, and AP features |
| `batch_run_dialog.py` | PyQt5 dialog the batch driver opens when given no folder: picks the experiment folder and records the animal (DOB, sex, genotype, strain) |
| `batch_qc_window.py` | QC window shown when a batch finishes: F-I curves overlaid, cell-health measurements against QC guides, and the traces behind them |
| `calculate_Ra.py` | Access resistance / membrane resistance / capacitance / tau from a voltage-clamp "Membrane Test" recording |
| `check_sta_data.py` | Quick averaging utility for `.sta` spike-triggered-average files |
| `membrane_test.ipynb`, `membrane_test_whole_trace.ipynb` | Notebook versions of the membrane-test calculation, for interactive exploration |
| `_archive/` | Earlier, superseded versions of the sEPSC scripts, kept for reference |

## Usage

Each top-level script pops a file-open dialog if run with no arguments, or
accepts a path directly:
```
python autorun_AP_analysis.py recording.abf
python calculate_Ra.py recording.abf
```

To analyse a whole experiment at once, put each cell's recordings in a subfolder
named `YYMMDD_s<slice>_c<cell>_<RMP>` (e.g. `260817_s1_c1_-67`; bare numbers and
a trailing label also parse) and point the batch driver at the folder above them:
```
python batch_cell_analysis.py DATA_ROOT
python batch_cell_analysis.py               # dialog: pick the folder + the animal
python batch_cell_analysis.py --selftest    # validate the sag/Rin maths
```
Run with no folder and a dialog asks for the experiment folder together with the
animal it came from — date of birth, sex, genotype, strain — and reports how many
cell folders it found before anything runs. Those go into every summary row, and
the DOB gives each cell an `age_days` from its folder's recording date. The same
fields are flags (`--dob`, `--sex`, `--genotype`, `--strain`) for scripted runs.

When the run finishes a QC window opens, with every cell's F-I curve overlaid on
one axes, the cell-health measurements (Ra, Cm, resting Vm, rheobase, AP
amplitude and half-width) plotted against QC guides, and each cell's rheobase and
max-firing traces. `--no-window` skips it, and `--view` reopens it for an
already-analysed folder:
```
python batch_cell_analysis.py DATA_ROOT --view
```
It picks each cell's recordings out of the folder from their .abf headers — the
first `DW_Ra` membrane test for Ra/Rm/Cm/tau, and the current-clamp step file
with the widest injected-current range for the AP features. Hyperpolarizing
sweeps, from that file or from a separate hyperpolarizing protocol, also give
sag at a fixed membrane potential (-85 mV by default) and an input resistance
fitted only over the steps too small to activate Ih. Per-cell outputs land next
to the recordings; `batch_cell_summary.csv` at the root has one row per cell.
See each script's module docstring for full details, and [`sepsc/README.md`](sepsc/README.md)
for the sEPSC pipeline's commands and output-file conventions.
