"""PyQt5 dialog that starts a batch run: pick the experiment folder, say what
was recorded -- brain region and cell type -- and record the animal it came
from: which mouse it was, and its date of birth, sex, genotype and strain.

Region and cell type are kept apart from the animal because they describe
what the electrode was on rather than what it came from. One of each is
applied to the whole run, which is a real limitation where the animal's
genotype is not: a session that patched two regions, or two cell types in one
region, needs a run per group or the column edited afterwards. The dialog
says so.

Which mouse is three fields rather than one, because no single one of them is
reliable on its own: the mouse number is only unique within a cage, the cage
number changes when animals are moved or a litter is weaned, and the ID
markers (ear punch, tail marks, tag) are what is actually readable on the
animal at the rig. Recording all three means a cell can still be traced back
to an animal when one of them turns out to be wrong or missing.

The animal details are applied to every cell analysed in the run, so they end
up as columns in ``batch_cell_summary.csv`` alongside each cell's own
measurements, and the date of birth also gives each cell an age in days
(folder date minus DOB).

Choices are remembered between runs via QSettings -- the free-text fields keep
a history, so the second experiment on a line is a dropdown pick rather than
retyping. Each run also saves its details to ``run_metadata.txt`` in the
experiment folder, and choosing a folder that has one fills the fields in from
it, so re-analysing a folder starts from what was entered for that folder
rather than for whichever experiment was run last. Nothing is remembered about
the recordings themselves; the folder is re-scanned every time it is chosen,
and the dialog says how many cell folders it found before anything is
analysed.

Used by ``batch_cell_analysis.py`` when it is run without a folder argument.
It is a separate module so the analysis itself stays headless: PyQt5 is only
imported when the dialog is actually shown.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

# age_days lives with the analysis, not here, so a headless run can compute an
# age without importing PyQt5. Safe to import at module level: the analysis
# module only imports this one from inside a function.
from batch_cell_analysis import RUN_METADATA_FILE, age_days, read_run_metadata

SEXES = ["unknown", "male", "female"]
SETTINGS_ORG, SETTINGS_APP = "clampex_analysis", "batch_cell_analysis"

# Genotype/strain histories are stored as one delimited string rather than a
# QSettings list: a single-entry list round-trips through the Windows registry
# as a bare string, which would silently turn a one-item history into a list of
# characters.
_SEP = "\x1f"


def _history(settings: QtCore.QSettings, key: str) -> list[str]:
    raw = settings.value(key, "", type=str) or ""
    return [s for s in raw.split(_SEP) if s.strip()]


def _remember(settings: QtCore.QSettings, key: str, value: str, limit: int = 20) -> None:
    value = (value or "").strip()
    if not value:
        return
    hist = [value] + [h for h in _history(settings, key) if h != value]
    settings.setValue(key, _SEP.join(hist[:limit]))


class RunDialog(QtWidgets.QDialog):
    def __init__(self, initial_root: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.setWindowTitle("Batch cell analysis")

        remembered = self.settings.value("root", "", type=str)
        self.root: Optional[Path] = (Path(initial_root) if initial_root
                                     else (Path(remembered) if remembered else None))

        self._loaded_details = False
        self._build_ui()
        self._load_saved_details()
        self._rescan()

    # -- layout --------------------------------------------------------
    def _build_ui(self) -> None:
        self.root_edit = QtWidgets.QLineEdit(str(self.root) if self.root else "")
        self.root_edit.setReadOnly(True)
        self.root_edit.setPlaceholderText("folder holding the cell subfolders")
        self.root_edit.setMinimumWidth(420)
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self.choose_root)

        root_row = QtWidgets.QHBoxLayout()
        root_row.addWidget(self.root_edit, 1)
        root_row.addWidget(browse)

        self.scan_label = QtWidgets.QLabel()
        self.scan_label.setWordWrap(True)

        folder_box = QtWidgets.QGroupBox("Experiment folder")
        folder_layout = QtWidgets.QVBoxLayout(folder_box)
        folder_layout.addLayout(root_row)
        folder_layout.addWidget(self.scan_label)

        # The folders themselves, not just how many: a cell folder with a
        # mistyped name is otherwise only noticeable as a count one short.
        self.cell_list = QtWidgets.QListWidget()
        self.cell_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.cell_list.setFocusPolicy(QtCore.Qt.NoFocus)
        self.cell_list.setMaximumHeight(150)
        self.cell_list.setWordWrap(True)
        folder_layout.addWidget(self.cell_list)

        # -- animal ----------------------------------------------------
        self.dob_edit = QtWidgets.QDateEdit()
        self.dob_edit.setCalendarPopup(True)
        self.dob_edit.setDisplayFormat("yyyy-MM-dd")
        self.dob_edit.setMaximumDate(QtCore.QDate.currentDate())
        remembered_dob = self.settings.value("dob", "", type=str)
        self.dob_edit.setDate(QtCore.QDate.fromString(remembered_dob, "yyyy-MM-dd")
                              if remembered_dob else QtCore.QDate.currentDate().addDays(-30))

        self.dob_unknown = QtWidgets.QCheckBox("unknown")
        self.dob_unknown.setToolTip("Leave the date of birth blank; no age is calculated")
        self.dob_unknown.toggled.connect(self.dob_edit.setDisabled)
        self.dob_unknown.toggled.connect(lambda _c: self._rescan())
        self.dob_edit.dateChanged.connect(lambda _d: self._rescan())
        self.dob_unknown.setChecked(self.settings.value("dob_unknown", False, type=bool))

        dob_row = QtWidgets.QHBoxLayout()
        dob_row.addWidget(self.dob_edit, 1)
        dob_row.addWidget(self.dob_unknown)

        self.sex_combo = QtWidgets.QComboBox()
        self.sex_combo.addItems(SEXES)
        self.sex_combo.setCurrentText(self.settings.value("sex", "unknown", type=str))

        self.mouse_combo = self._history_combo("mouse_number", "e.g. 3, M3")
        self.cage_combo = self._history_combo("cage_number", "e.g. 1234567")
        self.markers_combo = self._history_combo(
            "id_markers", "e.g. ear punch R, tail 2 stripes, tag 417")
        self.mouse_combo.setToolTip("The animal's number. Usually only unique "
                                    "within its cage, so record the cage too.")
        self.cage_combo.setToolTip("The cage the animal was housed in at the "
                                   "time of the recording.")
        self.markers_combo.setToolTip("Whatever physically identifies this animal: "
                                      "ear punch, tail marks, ear tag.")

        self.genotype_combo = self._history_combo("genotype", "e.g. Sst-Cre;Ai14, WT")
        self.strain_combo = self._history_combo("strain", "e.g. C57BL/6J")

        form = QtWidgets.QFormLayout()
        form.addRow("Mouse number", self.mouse_combo)
        form.addRow("Cage number", self.cage_combo)
        form.addRow("ID markers", self.markers_combo)
        form.addRow("Date of birth", dob_row)
        form.addRow("Sex", self.sex_combo)
        form.addRow("Genotype", self.genotype_combo)
        form.addRow("Strain", self.strain_combo)

        animal_box = QtWidgets.QGroupBox("Animal")
        animal_layout = QtWidgets.QVBoxLayout(animal_box)
        animal_layout.addLayout(form)
        note = QtWidgets.QLabel("These details are recorded for every cell in this run.")
        note.setStyleSheet("color: #898781;")
        note.setWordWrap(True)
        animal_layout.addWidget(note)

        # -- recording -------------------------------------------------
        # Separate from the animal because these describe what the electrode
        # was on, not what it came from -- the same animal yields cells in
        # more than one region, and more than one cell type in each.
        self.region_combo = self._history_combo(
            "brain_region", "e.g. CA1, mPFC, dorsal striatum")
        self.cell_type_combo = self._history_combo(
            "cell_type", "e.g. pyramidal, SST interneuron, MSN")
        self.region_combo.setToolTip("The region the recorded cells were in.")
        self.cell_type_combo.setToolTip("What the recorded cells were, as far as "
                                        "it is known at the rig.")

        rec_form = QtWidgets.QFormLayout()
        rec_form.addRow("Brain region", self.region_combo)
        rec_form.addRow("Cell type", self.cell_type_combo)

        rec_box = QtWidgets.QGroupBox("Recording")
        rec_layout = QtWidgets.QVBoxLayout(rec_box)
        rec_layout.addLayout(rec_form)
        # Worth saying explicitly here and not under Animal: an animal's
        # genotype really is one value for the run, but region and cell type
        # are per cell, and this dialog can only apply one of each.
        rec_note = QtWidgets.QLabel(
            "Applied to every cell in this run. Analyse a run per region or "
            "cell type if they differ, or edit the column afterwards.")
        rec_note.setStyleSheet("color: #898781;")
        rec_note.setWordWrap(True)
        rec_layout.addWidget(rec_note)

        buttons = QtWidgets.QDialogButtonBox()
        self.run_btn = buttons.addButton("Run analysis", QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(folder_box)
        layout.addWidget(rec_box)
        layout.addWidget(animal_box)
        layout.addWidget(buttons)

    def _history_combo(self, key: str, placeholder: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.setEditable(True)          # free text, past entries as suggestions
        combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        combo.addItems(_history(self.settings, key))
        combo.setCurrentText(self.settings.value(key + "_last", "", type=str))
        combo.lineEdit().setPlaceholderText(placeholder)
        return combo

    # -- folder --------------------------------------------------------
    def choose_root(self) -> None:
        start = str(self.root) if self.root and self.root.is_dir() else ""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select the folder holding the cell subfolders", start)
        if not folder:
            return
        self.root = Path(folder)
        self.root_edit.setText(folder)
        self._load_saved_details()
        self._rescan()

    def _load_saved_details(self) -> None:
        """Fill the fields in from the folder's run_metadata.txt, if it has one.
        A folder without one leaves the fields as they are."""
        saved = read_run_metadata(self.root) if self.root and self.root.is_dir() else None
        self._loaded_details = saved is not None
        if saved is None:
            return
        for key, combo in (("brain_region", self.region_combo),
                           ("cell_type", self.cell_type_combo),
                           ("mouse_number", self.mouse_combo),
                           ("cage_number", self.cage_combo),
                           ("id_markers", self.markers_combo),
                           ("genotype", self.genotype_combo),
                           ("strain", self.strain_combo)):
            combo.setCurrentText(saved[key])
        self.sex_combo.setCurrentText(saved["animal_sex"] if saved["animal_sex"] in SEXES
                                      else "unknown")
        dob = QtCore.QDate.fromString(saved["animal_dob"], "yyyy-MM-dd")
        if dob.isValid():
            self.dob_edit.setDate(dob)
        self.dob_unknown.setChecked(not dob.isValid())

    def _rescan(self) -> None:
        """Report what the chosen folder actually contains, before anything runs."""
        if not hasattr(self, "scan_label"):
            return
        self.cell_list.clear()
        if not self.root or not self.root.is_dir():
            self.scan_label.setText("Choose a folder.")
            self.scan_label.setStyleSheet("color: #b35c00;")
            self.run_btn.setEnabled(False)
            return

        try:
            import batch_cell_analysis as B
            cells = B.find_cell_folders(self.root)
            skipped = B.skipped_cell_folders(self.root)
        except Exception as exc:                      # pragma: no cover - defensive
            self.scan_label.setText(f"Could not scan the folder: {exc}")
            self.scan_label.setStyleSheet("color: #b00020;")
            self.run_btn.setEnabled(True)
            return

        self._list_folders(cells, skipped)
        self.run_btn.setEnabled(True)
        if not cells:
            self.scan_label.setText(
                "No cell folders found. Expected subfolders named "
                "YYMMDD_<slice>_<cell>_<RMP>, e.g. 260817_s1_c1_-67.")
            self.scan_label.setStyleSheet("color: #b00020;")
            return

        dates = sorted({m["date"] for _f, m in cells if m.get("date")})
        msg = f"{len(cells)} cell folder(s) found."
        colour = "#2e7d32"
        if skipped:
            msg += (f" {len(skipped)} other folder(s) skipped and will not be "
                    f"analysed; see the list.")
            colour = "#b35c00"
        if len(dates) > 1:
            msg += (f" They span {len(dates)} recording dates ({dates[0]}-{dates[-1]}); "
                    f"the animal details below will be applied to all of them.")
            colour = "#b35c00"
        age = self._age_range(cells)
        if age:
            msg += f" {age}"
        if self._loaded_details:
            msg += f" Details below filled in from this folder's {RUN_METADATA_FILE}."
        self.scan_label.setText(msg)
        self.scan_label.setStyleSheet(f"color: {colour};")

    def _list_folders(self, cells, skipped) -> None:
        """Every cell folder that will be analysed, then any that will not."""
        def rel(folder: Path) -> str:
            try:
                return str(folder.relative_to(self.root))
            except ValueError:
                return str(folder)

        for folder, _meta in cells:
            self.cell_list.addItem(rel(folder))
        for folder, reason in skipped:
            item = QtWidgets.QListWidgetItem(f"{rel(folder)}   -- skipped: {reason}")
            item.setForeground(QtGui.QColor("#b35c00"))
            self.cell_list.addItem(item)

    def _age_range(self, cells) -> str:
        """Ages the chosen DOB implies, as a sanity check before running."""
        dob = self.dob()
        if not dob:
            return ""
        ages = []
        for _folder, meta in cells:
            days = age_days(dob, meta.get("date", ""))
            if days is not None:
                ages.append(days)
        if not ages:
            return ""
        if min(ages) < 0:
            return (f"WARNING: the recordings predate this date of birth "
                    f"(by up to {-min(ages)} days).")
        return (f"Age at recording: {min(ages)} days." if min(ages) == max(ages)
                else f"Age at recording: {min(ages)}-{max(ages)} days.")

    # -- values --------------------------------------------------------
    def dob(self) -> str:
        if self.dob_unknown.isChecked():
            return ""
        return self.dob_edit.date().toString("yyyy-MM-dd")

    def values(self) -> dict:
        return {
            "root": self.root,
            "brain_region": self.region_combo.currentText().strip(),
            "cell_type": self.cell_type_combo.currentText().strip(),
            "mouse_number": self.mouse_combo.currentText().strip(),
            "cage_number": self.cage_combo.currentText().strip(),
            "id_markers": self.markers_combo.currentText().strip(),
            "animal_dob": self.dob(),
            "animal_sex": self.sex_combo.currentText().strip(),
            "genotype": self.genotype_combo.currentText().strip(),
            "strain": self.strain_combo.currentText().strip(),
        }

    def accept(self) -> None:
        if not self.root or not self.root.is_dir():
            QtWidgets.QMessageBox.warning(self, "Choose a folder",
                                          "Select the folder holding the cell subfolders.")
            return
        v = self.values()
        self.settings.setValue("root", str(self.root))
        self.settings.setValue("dob", self.dob_edit.date().toString("yyyy-MM-dd"))
        self.settings.setValue("dob_unknown", self.dob_unknown.isChecked())
        self.settings.setValue("sex", v["animal_sex"])
        for key in ("brain_region", "cell_type", "mouse_number", "cage_number",
                    "id_markers", "genotype", "strain"):
            self.settings.setValue(key + "_last", v[key])
            _remember(self.settings, key, v[key])
        super().accept()


def prompt_for_run(initial_root: Optional[Path] = None) -> Optional[dict]:
    """Show the dialog. Returns the chosen folder and animal details, or None
    if it was cancelled."""
    # Shared with the QC window so both honour Windows display scaling; the
    # attributes it sets only take effect before a QApplication exists.
    from batch_qc_window import qt_app
    app = qt_app()
    owns_app = False
    dlg = RunDialog(initial_root)
    accepted = dlg.exec_() == QtWidgets.QDialog.Accepted
    values = dlg.values() if accepted else None
    if owns_app:
        app.processEvents()
    return values


if __name__ == "__main__":
    print(prompt_for_run())
