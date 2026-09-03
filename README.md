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
   pip install git+https://github.com/delvendahl/miniML.git
   pip install pyabf
   ```
   **Install miniML first, before TensorFlow.** miniML pins `tensorflow==2.15.1` and
   pulls in numpy/pandas/scipy/matplotlib/scikit-learn itself, so this one command gets
   the whole stack at versions that agree. Don't fight the 2.15 pin: `detect` loads the
   bundled legacy-format `GC_lstm_model.h5`, and TensorFlow 2.16+ switches to Keras 3
   (upstream ships a separate `GC_model_keras3.keras` for that case).

   Two gotchas on Windows:
   - `pip install miniml` (plain, from PyPI) installs an **unrelated placeholder
     package** — you must install from the GitHub URL above.
   - pip shells out to `git` for a `git+https://` URL, so `git` has to be on the
     Windows `PATH` (e.g. `C:\Program Files\Git\cmd`), not just inside Git Bash.

   If you install TensorFlow *first* and let miniML downgrade it, pip leaves an
   orphaned `site-packages\tensorflow\` directory with no `__init__.py`, and every
   import then fails with `No module named 'tensorflow.compat'`. To recover:
   ```
   pip uninstall -y tensorflow tensorflow-intel
   rmdir /s /q <env>\Lib\site-packages\tensorflow    # if still present
   pip install tensorflow==2.15.1
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
See each script's module docstring for full details, and [`sepsc/README.md`](sepsc/README.md)
for the sEPSC pipeline's commands and output-file conventions.
