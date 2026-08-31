"""
Startup picker: one small window to choose which detection method to run
(minianalysis / fastmini / detect) and, for the methods that support it,
whether to Bessel-filter + downsample the trace first (sepsc.preprocess) --
then launches that method exactly as its own CLI would, with those choices
passed through as its normal flags. This is a convenience front door, not a
fourth argument set: each method's own dialog/flags still apply after this
(e.g. minianalysis shows its own 9-detection-parameter dialog next, unless
that method is invoked with --no-gui).

`detect` (miniML) needs the separate 'clampex_miniml' conda environment
(TensorFlow isn't in this project's main venv, see sepsc.detect) -- this
picker auto-detects that environment as a sibling of the current conda
root's envs/ folder, and falls back to asking you to locate its python.exe
if that guess is wrong.

`fastmini` always applies its own fixed 3 kHz Bessel + 10 kHz resample
first (that 10 kHz convention is baked into its fixed-sample-count analysis
window/templates, see fastmini.py) -- the filter/resample fields below are
disabled for it, since there's nothing to pass through.

"Open inspect after analysis finishes" and "Open review after analysis
finishes" appear whenever the chosen method's output is something
sepsc.inspect/sepsc.review can actually open (`--source minianalysis`/
`miniml`/`fastmini` -- see both tools' own --help): `minianalysis` offers
inspect only, `detect` offers review only, `fastmini` offers BOTH (it can
be checked/reviewed either way) and its own inspect/review commands are
built with `--source fastmini`. Either one waits for the primary process to
exit, then opens that tool on the same abf/channel/filter settings (both
checked together, for fastmini, wait for the SAME primary process once and
then open both windows) -- if no events CSV exists yet (e.g. minianalysis's
optimize window was closed without ever clicking "Run full detection"),
the opened tool just reports that and exits, same as running it manually.

For `minianalysis`, one more control appears: "Tune parameters first with
optimize" -- opens sepsc.optimize on this trace instead of minianalysis
itself; optimize's own "Run full detection with these parameters" button is
what actually runs minianalysis once you're happy with a click-tested peak.

For `detect`, one more control appears: a detection-threshold field
(defaults to detect.py's own default, 0.5) -- detect.py has no PyQt5 dialog
of its own (the clampex_miniml env it runs in doesn't have PyQt5 installed,
unlike this project's main venv), so this threshold field IS the "parameter
window with defaults" for miniML, not a follow-up dialog detect.py shows
itself.

Usage
-----
    python -m sepsc.launch
    python -m sepsc.launch recording.abf --channel 0   # pre-filled, still editable
    python -m sepsc launch                             # equivalent, via the dispatcher
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from .gui_utils import FILTER_FIELD_SPECS, make_field_widget, read_field_widget
from .preprocess import DEFAULT_CUTOFF_HZ, DEFAULT_ORDER, DEFAULT_TARGET_RATE_HZ

METHOD_CHOICES = [
    ("minianalysis", "Classical (minianalysis) — local-max + baseline + amplitude/area threshold",
     "Runs in this environment. Shows its own detection-parameter dialog next "
     "(9 detection parameters, pre-filled with the filter/resample choice below) -- or, with "
     "'Tune parameters first' checked below, opens sepsc.optimize on this trace instead, whose "
     "own 'Run full detection' button starts minianalysis once you're happy with a click-tested peak."),
    ("fastmini", "MLP peel-off (fastmini) — per-recording-trained classifier",
     "Runs in this environment. Always applies its own fixed 3 kHz Bessel + 10 kHz "
     "resample first (that convention is baked into its analysis window) — the "
     "filter/resample fields below don't apply and are disabled."),
    ("detect", "miniML CNN-LSTM (detect) — deep-learning detector",
     "Runs in the separate 'clampex_miniml' conda environment (TensorFlow). Set its detection "
     "threshold below (default 0.5) -- detect.py has no parameter dialog of its own, so this is "
     "it -- and optionally open sepsc.review afterward to accept/reject the detected events."),
]


def _find_clampex_miniml_python() -> Optional[str]:
    """Sibling-env guess: this project's main interpreter is normally either
    a conda base env or itself an env under <conda_root>/envs/, so
    clampex_miniml should be a sibling under the same envs/ folder either
    way. Returns None if neither guess exists (caller should ask the user)."""
    candidates = [
        os.path.join(sys.prefix, "envs", "clampex_miniml", "python.exe"),
        os.path.join(os.path.dirname(os.path.dirname(sys.prefix)), "envs", "clampex_miniml", "python.exe"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


@dataclass
class LaunchChoice:
    abf_path: str
    channel: int
    method: str
    filter_enabled: bool
    cutoff_hz: float
    target_rate_hz: float
    filter_order: int
    tune_first: bool = False
    inspect_after: bool = False
    threshold: float = 0.5
    review_after: bool = False


def _prompt_for_launch(abf_path: str, channel: int) -> Optional[LaunchChoice]:
    from PyQt5 import QtWidgets

    # Must keep a live reference -- see sepsc.minianalysis._prompt_for_settings
    # for why an unassigned QApplication([]) here can be garbage-collected
    # before the QDialog below is constructed.
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("sepsc — Choose Detection Method")
    layout = QtWidgets.QVBoxLayout(dialog)

    file_row = QtWidgets.QHBoxLayout()
    abf_edit = QtWidgets.QLineEdit(abf_path or "")
    browse_btn = QtWidgets.QPushButton("Browse...")

    def on_browse():
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            dialog, "Select recording", abf_edit.text() or "", "Axon ABF (*.abf)")
        if path:
            abf_edit.setText(path)

    browse_btn.clicked.connect(on_browse)
    file_row.addWidget(QtWidgets.QLabel("Recording (.abf):"))
    file_row.addWidget(abf_edit)
    file_row.addWidget(browse_btn)
    layout.addLayout(file_row)

    channel_row = QtWidgets.QHBoxLayout()
    channel_spin = QtWidgets.QSpinBox()
    channel_spin.setRange(0, 15)
    channel_spin.setValue(channel)
    channel_row.addWidget(QtWidgets.QLabel("Channel:"))
    channel_row.addWidget(channel_spin)
    channel_row.addStretch(1)
    layout.addLayout(channel_row)

    method_group = QtWidgets.QGroupBox("Detection method")
    method_layout = QtWidgets.QVBoxLayout(method_group)
    method_combo = QtWidgets.QComboBox()
    method_combo.addItems([label for _, label, _ in METHOD_CHOICES])
    method_note = QtWidgets.QLabel()
    method_note.setWordWrap(True)
    method_layout.addWidget(method_combo)
    method_layout.addWidget(method_note)
    tune_first_w = QtWidgets.QCheckBox("Tune parameters first with optimize (minianalysis only)")
    method_layout.addWidget(tune_first_w)

    threshold_row = QtWidgets.QHBoxLayout()
    threshold_label = QtWidgets.QLabel("Detection threshold (miniML only):")
    threshold_w = QtWidgets.QDoubleSpinBox()
    threshold_w.setRange(0.0, 1.0)
    threshold_w.setDecimals(2)
    threshold_w.setSingleStep(0.05)
    threshold_w.setValue(0.5)
    threshold_row.addWidget(threshold_label)
    threshold_row.addWidget(threshold_w)
    threshold_row.addStretch(1)
    method_layout.addLayout(threshold_row)

    # Visibility (see on_method_changed) rather than separate per-method
    # checkboxes: minianalysis offers inspect only, detect offers review
    # only, fastmini offers both (its output works with either tool, see
    # _build_inspect_argv/_build_review_argv's --source fastmini).
    inspect_after_w = QtWidgets.QCheckBox("Open inspect after analysis finishes")
    method_layout.addWidget(inspect_after_w)
    review_after_w = QtWidgets.QCheckBox("Open review after analysis finishes")
    method_layout.addWidget(review_after_w)
    layout.addWidget(method_group)

    filter_group = QtWidgets.QGroupBox("Filter / resample (sepsc.preprocess)")
    filter_form = QtWidgets.QFormLayout(filter_group)
    enabled_w = QtWidgets.QCheckBox()
    filter_form.addRow("Apply Bessel low-pass + resample:", enabled_w)
    filter_widgets = {}
    filter_defaults = dict(cutoff_hz=DEFAULT_CUTOFF_HZ, target_rate_hz=DEFAULT_TARGET_RATE_HZ,
                            filter_order=DEFAULT_ORDER)
    for attr, label, kind, kwargs in FILTER_FIELD_SPECS:
        w = make_field_widget(kind, kwargs, filter_defaults[attr])
        w.setEnabled(False)
        enabled_w.toggled.connect(w.setEnabled)
        filter_widgets[attr] = w
        filter_form.addRow(label + ":", w)
    layout.addWidget(filter_group)

    def on_method_changed(index):
        method, _, note = METHOD_CHOICES[index]
        method_note.setText(note)
        is_fastmini = method == "fastmini"
        filter_group.setEnabled(not is_fastmini)
        if is_fastmini:
            enabled_w.setChecked(False)
        is_minianalysis = method == "minianalysis"
        tune_first_w.setVisible(is_minianalysis)
        if not is_minianalysis:
            tune_first_w.setChecked(False)
        is_detect = method == "detect"
        threshold_label.setVisible(is_detect)
        threshold_w.setVisible(is_detect)

        can_inspect = method in ("minianalysis", "fastmini")
        inspect_after_w.setVisible(can_inspect)
        if not can_inspect:
            inspect_after_w.setChecked(False)
        can_review = method in ("detect", "fastmini")
        review_after_w.setVisible(can_review)
        if not can_review:
            review_after_w.setChecked(False)

    method_combo.currentIndexChanged.connect(on_method_changed)
    on_method_changed(0)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)

    def on_accept():
        if not abf_edit.text().strip():
            QtWidgets.QMessageBox.warning(dialog, "Missing recording", "Choose an .abf file first.")
            return
        dialog.accept()

    buttons.accepted.connect(on_accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(480)

    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None

    method = METHOD_CHOICES[method_combo.currentIndex()][0]
    filter_values = {attr: read_field_widget(filter_widgets[attr], kind) for attr, _, kind, _ in FILTER_FIELD_SPECS}
    return LaunchChoice(abf_path=abf_edit.text().strip(), channel=channel_spin.value(), method=method,
                         filter_enabled=enabled_w.isChecked(), tune_first=tune_first_w.isChecked(),
                         inspect_after=inspect_after_w.isChecked(), threshold=threshold_w.value(),
                         review_after=review_after_w.isChecked(), **filter_values)


def _build_argv(choice: LaunchChoice) -> list[str]:
    # optimize takes the same abf/channel/filter flags as minianalysis (see optimize.py's own
    # --help) and its "Run full detection with these parameters" button is what actually runs
    # minianalysis, so tune_first just swaps which command this launches, nothing else.
    command = "optimize" if (choice.method == "minianalysis" and choice.tune_first) else choice.method
    argv = [command, choice.abf_path, "--channel", str(choice.channel)]
    if choice.filter_enabled and choice.method != "fastmini":
        argv += ["--filter", "--cutoff-hz", str(choice.cutoff_hz),
                 "--target-rate-hz", str(choice.target_rate_hz), "--filter-order", str(choice.filter_order)]
    if choice.method == "detect":
        argv += ["--threshold", str(choice.threshold)]
    return argv


# choice.method -> the --source inspect.py/review.py need to find that
# method's own output (see both tools' own SOURCE_CFG/SOURCES) -- "detect"
# is the one mismatch (its --source key is "miniml", not "detect").
SOURCE_FOR_METHOD = {"minianalysis": "minianalysis", "detect": "miniml", "fastmini": "fastmini"}


def _build_inspect_argv(choice: LaunchChoice) -> list[str]:
    # inspect.py auto-discovers <abf>[_filtXHzYHz]_<source>_events/individual.csv
    # from the same abf/channel/filter/--source flags the primary command
    # was run with, so this is exactly the primary command's argv with the
    # method name swapped -- inspect has no idea tune_first/inspect_after
    # exist, it just needs to see the same trace/filter/source settings.
    # --filter is passed through as-is even for --source fastmini (inspect.py
    # itself ignores it for that source, printing a NOTE) -- moot in
    # practice here anyway, since the filter checkbox is disabled/unchecked
    # for fastmini in this dialog (see on_method_changed).
    argv = ["inspect", choice.abf_path, "--channel", str(choice.channel),
            "--source", SOURCE_FOR_METHOD[choice.method]]
    if choice.filter_enabled:
        argv += ["--filter", "--cutoff-hz", str(choice.cutoff_hz),
                 "--target-rate-hz", str(choice.target_rate_hz), "--filter-order", str(choice.filter_order)]
    return argv


def _build_review_argv(choice: LaunchChoice) -> list[str]:
    # review.py now takes the same --filter/--cutoff-hz/--target-rate-hz/
    # --filter-order flags minianalysis.py/detect.py/inspect.py do (see its
    # own --help) -- MUST match the primary command's own --filter choice
    # here, since its default (unfiltered) CSV/output paths won't exist if
    # that run actually used --filter, and vice versa (review.py now errors
    # clearly on that mismatch rather than silently misaligning). --source
    # fastmini has no automatic comparison overlay (review.py's own
    # docstring) -- moot here since this dialog never offers --compare-csv.
    argv = ["review", choice.abf_path, "--channel", str(choice.channel),
            "--source", SOURCE_FOR_METHOD[choice.method]]
    if choice.filter_enabled:
        argv += ["--filter", "--cutoff-hz", str(choice.cutoff_hz),
                 "--target-rate-hz", str(choice.target_rate_hz), "--filter-order", str(choice.filter_order)]
    return argv


def _wait_then_launch(proc: subprocess.Popen, cmds: list[list[str]]) -> None:
    """Blocks until `proc` exits, then launches each of `cmds` -- e.g.
    inspect after minianalysis/optimize, review after detect, or BOTH after
    fastmini (which offers both checkboxes at once, see on_method_changed --
    both just wait for this same primary process, not for each other, since
    neither QC window depends on the other being open). Each target tool
    reports and exits cleanly on its own if the run never actually produced
    output (e.g. optimize closed without clicking "Run full detection"), so
    no separate check is needed here. Deliberately blocking, not a
    background thread: a daemon thread here would be killed the moment
    main() returns, since nothing else keeps this process alive while the
    first command runs."""
    proc.wait()
    for cmd in cmds:
        print("Launching:", " ".join(cmd))
        subprocess.Popen(cmd)


def _python_for(method: str) -> Optional[str]:
    if method != "detect":
        return sys.executable

    python_exe = _find_clampex_miniml_python()
    if python_exe is not None:
        return python_exe

    from PyQt5 import QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Could not auto-detect the clampex_miniml environment — locate its python.exe",
        "", "python.exe (python.exe)")
    return path or None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("abf", nargs="?", default="", help="Path to the .abf file (pre-fills the dialog)")
    parser.add_argument("--channel", type=int, default=0)
    args = parser.parse_args(argv)

    choice = _prompt_for_launch(args.abf, args.channel)
    if choice is None:
        print("Cancelled — no detection method launched.")
        return

    python_exe = _python_for(choice.method)
    if python_exe is None:
        print(f"Could not locate a Python interpreter for {choice.method!r} — aborting.", file=sys.stderr)
        return

    cmd = [python_exe, "-m", "sepsc"] + _build_argv(choice)
    print("Launching:", " ".join(cmd))
    proc = subprocess.Popen(cmd)

    # Both can be checked at once (fastmini offers both, see
    # on_method_changed) -- both just wait for this same primary process.
    followups, names = [], []
    if choice.method in ("minianalysis", "fastmini") and choice.inspect_after:
        followups.append([sys.executable, "-m", "sepsc"] + _build_inspect_argv(choice))
        names.append("inspect")
    if choice.method in ("detect", "fastmini") and choice.review_after:
        followups.append([sys.executable, "-m", "sepsc"] + _build_review_argv(choice))
        names.append("review")
    if followups:
        print(f"Waiting for {cmd[3]!r} to finish before opening {' and '.join(names)}...")
        _wait_then_launch(proc, followups)


if __name__ == "__main__":
    main()
