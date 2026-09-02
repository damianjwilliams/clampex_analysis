"""PyQt5 form for logging per-cell recording metadata at the rig -- date
(YYMMDD), slice number, cell number, RMP, genotype -- and then pairing that
cell with the five .abf files of its protocol run.

Rows are appended to a CSV (default ``cell_metadata.csv``), and that CSV is
also the memory: on startup the last row prefills the form, so date/slice/
genotype carry over from the previous entry and the cell number continues
from where you left off. Everything stays editable -- the defaults are a
starting point, not a constraint.

Adding a cell reveals the protocol panel, which scans the ABF folder,
takes the most recent recordings, reads each one's Clampex protocol name
out of its header (via pyabf) and pre-pairs them with the five slots:

    Initial membrane test    DW_Ra                 (first DW_Ra of the run)
    Hyperpolarizing steps    DW_hyperpolarizing
    Depolarizing steps       DW_IC
    sEPSCs                   DW_Gap_free_-70_mV
    Final membrane test      DW_Ra                 (last DW_Ra of the run)

Every slot is a dropdown over the candidate files, so a mismatch is one
click to fix, and "Browse" reaches any file outside the candidate window.
Saved file names go in one column per protocol, with their folder in the
row's ``abf_dir`` column (a file browsed from elsewhere keeps its full path).

    python cell_metadata_entry.py [--csv path\\to\\cell_metadata.csv] [--abf-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

DEFAULT_CSV = "cell_metadata.csv"

# (csv/dict key, panel label, expected Clampex protocol, which match to take).
# "first"/"last" only matters where one protocol fills two slots: the two
# membrane tests bracket the run, so the earliest DW_Ra is the initial test
# and the latest is the final one.
PROTOCOL_SLOTS = [
    ("initial_ra", "Initial membrane test", "DW_Ra", "first"),
    ("hyperpol", "Hyperpolarizing steps", "DW_hyperpolarizing", "last"),
    ("depol", "Depolarizing steps", "DW_IC", "last"),
    ("sepsc", "sEPSCs", "DW_Gap_free_-70_mV", "last"),
    ("final_ra", "Final membrane test", "DW_Ra", "last"),
]
FILE_KEYS = [f"file_{key}" for key, _l, _p, _w in PROTOCOL_SLOTS]
FIELDNAMES = ["date", "slice", "cell", "rmp_mv", "genotype", "abf_dir"] + FILE_KEYS + ["entered_at"]

# How many of the most recent .abf files to offer. A full run is 5 files;
# the extra headroom covers re-runs of a protocol and aborted sweeps, and
# they all stay visible in the dropdowns.
CANDIDATE_COUNT = 10


# ---------------------------------------------------------------------------
# CSV I/O. Dict-based, so the file stays hand-editable in Excel between
# sessions. Adding a cell appends; assigning its files rewrites that one row
# in place (preserving any extra columns the file already had).
# ---------------------------------------------------------------------------

def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return [row for row in csv.DictReader(fh)
                if any((v or "").strip() for v in row.values())]


def _fieldnames_for(rows: list[dict]) -> list[str]:
    extra = [k for row in rows for k in row
             if k and k not in FIELDNAMES]
    return FIELDNAMES + list(dict.fromkeys(extra))


def append_row(path: Path, row: dict) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def rewrite_rows(path: Path, rows: list[dict]) -> None:
    """Rewrite the whole CSV via a temp file + replace, so an interrupted
    write can't truncate a session's worth of entries."""
    fieldnames = _fieldnames_for(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def valid_yymmdd(text: str) -> bool:
    try:
        dt.datetime.strptime(text, "%y%m%d")
    except ValueError:
        return False
    return True


def as_int(value, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# ABF candidate scanning + protocol matching (pure functions, so the pairing
# rules can be exercised without a display or real recordings).
# ---------------------------------------------------------------------------

def read_abf_protocol(path: Path) -> tuple[str | None, dt.datetime | None]:
    """(protocol name, recording start time) from an .abf header.

    Header-only read (``loadData=False``) -- opening the sweep data of five
    gap-free recordings just to learn their protocol names would make the
    scan feel sluggish. Returns (None, mtime) if the file can't be parsed,
    so an unreadable file still shows up as a manually assignable candidate.
    """
    try:
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        mtime = None
    try:
        import pyabf
        abf = pyabf.ABF(str(path), loadData=False)
        return (abf.protocol or None), (abf.abfDateTime or mtime)
    except Exception:
        return None, mtime


def scan_candidates(folder: Path, limit: int = CANDIDATE_COUNT) -> list[dict]:
    """The `limit` most recent .abf files in `folder`, oldest first.

    Ordered by the recording timestamp in the header where available (file
    mtimes get rewritten by copies and by OneDrive sync, the header doesn't).
    """
    try:
        files = [p for p in folder.iterdir() if p.suffix.lower() == ".abf"]
    except OSError:
        return []
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    scanned = []
    for path in files[:limit]:
        protocol, when = read_abf_protocol(path)
        scanned.append({"path": path, "protocol": protocol, "when": when})
    scanned.sort(key=lambda c: (c["when"] or dt.datetime.min, c["path"].name))
    return scanned


def auto_assign(candidates: list[dict]) -> dict[str, Path]:
    """Pair protocol slots with candidate files by header protocol name.

    `candidates` is oldest-first, as returned by scan_candidates. Slots with
    no matching recording are left unassigned rather than filled with a
    guess -- a wrong-but-plausible pairing is worse than a blank one.
    """
    assigned: dict[str, Path] = {}
    for key, _label, protocol, which in PROTOCOL_SLOTS:
        matches = [c for c in candidates
                   if (c["protocol"] or "").strip().lower() == protocol.lower()]
        if not matches:
            continue
        assigned[key] = (matches[0] if which == "first" else matches[-1])["path"]
    # A run with only one DW_Ra file has an initial membrane test and no
    # final one -- don't pair the same file into both slots.
    if "final_ra" in assigned and assigned.get("initial_ra") == assigned["final_ra"]:
        del assigned["final_ra"]
    return assigned


class MetadataWindow(QtWidgets.QWidget):
    def __init__(self, csv_path: Path, abf_dir: Path | None = None):
        super().__init__()
        self.csv_path = csv_path
        self.rows = read_rows(csv_path)
        self.settings = QtCore.QSettings("clampex_analysis", "cell_metadata_entry")
        self.candidates: list[dict] = []
        self.target_row: int | None = None   # row the protocol panel is editing
        self._files_dirty = False

        remembered = self.settings.value("abf_dir", "", type=str)
        self.abf_dir = Path(abf_dir) if abf_dir else (Path(remembered) if remembered else None)

        self.setWindowTitle(f"Cell metadata -- {csv_path.name}")
        self._build_ui()
        self._prefill_from_last_row()

    # -- layout --------------------------------------------------------
    def _build_ui(self) -> None:
        self.date_edit = QtWidgets.QLineEdit()
        self.date_edit.setPlaceholderText("YYMMDD")
        self.date_edit.setMaxLength(6)
        # Digits only; the calendar-validity check happens on save, so a
        # half-typed date isn't rejected keystroke by keystroke.
        self.date_edit.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp(r"\d{0,6}")))
        self.date_edit.returnPressed.connect(self.save_entry)

        self.slice_spin = QtWidgets.QSpinBox()
        self.slice_spin.setRange(1, 999)
        self.slice_spin.valueChanged.connect(self._on_slice_changed)

        self.cell_spin = QtWidgets.QSpinBox()
        self.cell_spin.setRange(1, 999)

        self.rmp_spin = QtWidgets.QDoubleSpinBox()
        self.rmp_spin.setRange(-150.0, 50.0)
        self.rmp_spin.setDecimals(1)
        self.rmp_spin.setSingleStep(1.0)
        self.rmp_spin.setSuffix(" mV")
        self.rmp_spin.setValue(-70.0)

        self.genotype_combo = QtWidgets.QComboBox()
        self.genotype_combo.setEditable(True)  # free text, past values as suggestions
        self.genotype_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.genotype_combo.addItems(self._known_genotypes())
        self.genotype_combo.setCurrentText("")
        self.genotype_combo.lineEdit().returnPressed.connect(self.save_entry)

        form = QtWidgets.QFormLayout()
        form.addRow("Date (YYMMDD)", self.date_edit)
        form.addRow("Slice #", self.slice_spin)
        form.addRow("Cell #", self.cell_spin)
        form.addRow("RMP", self.rmp_spin)
        form.addRow("Genotype", self.genotype_combo)

        self.save_btn = QtWidgets.QPushButton("Add cell  (Ctrl+Enter)")
        self.save_btn.clicked.connect(self.save_entry)

        self.next_slice_btn = QtWidgets.QPushButton("Next slice")
        self.next_slice_btn.setToolTip("Advance to the next slice and reset the cell number to 1")
        self.next_slice_btn.clicked.connect(
            lambda: self.slice_spin.setValue(self.slice_spin.value() + 1))

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.next_slice_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.save_btn)

        cell_box = QtWidgets.QGroupBox("Cell")
        cell_layout = QtWidgets.QVBoxLayout(cell_box)
        cell_layout.addLayout(form)
        cell_layout.addLayout(buttons)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Date", "Slice", "Cell", "RMP (mV)", "Genotype", "Files"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QTableWidget.NoSelection)
        self._fill_table()

        self.status = QtWidgets.QLabel(str(self.csv_path.resolve()))
        self.status.setToolTip(str(self.csv_path.resolve()))
        self.status.setStyleSheet("color: #898781;")
        self.status.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(cell_box)
        layout.addWidget(self._build_file_panel())
        layout.addWidget(QtWidgets.QLabel("Logged cells"))
        layout.addWidget(self.table)
        layout.addWidget(self.status)
        self.resize(720, 720)

        # This is a plain QWidget, not a QDialog, so QPushButton.setDefault
        # wouldn't bind Return to anything -- the shortcuts are explicit, and
        # work from the spin boxes too (where Return has no other meaning).
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, self.save_entry)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, self.save_files)

    def _build_file_panel(self) -> QtWidgets.QWidget:
        self.file_panel = QtWidgets.QGroupBox("Protocol files")
        self.file_panel.setVisible(False)  # revealed once a cell is added

        self.file_panel_header = QtWidgets.QLabel()
        self.file_panel_header.setStyleSheet("font-weight: bold;")

        self.abf_dir_edit = QtWidgets.QLineEdit(str(self.abf_dir) if self.abf_dir else "")
        self.abf_dir_edit.setReadOnly(True)
        self.abf_dir_edit.setPlaceholderText("choose the folder Clampex is saving into")
        choose_btn = QtWidgets.QPushButton("Choose folder...")
        choose_btn.clicked.connect(self.choose_abf_dir)
        rescan_btn = QtWidgets.QPushButton("Rescan")
        rescan_btn.setToolTip(f"Re-read the {CANDIDATE_COUNT} most recent .abf files and re-match them")
        rescan_btn.clicked.connect(lambda: self.rescan(reassign=True))

        dir_row = QtWidgets.QHBoxLayout()
        dir_row.addWidget(QtWidgets.QLabel("ABF folder"))
        dir_row.addWidget(self.abf_dir_edit, 1)
        dir_row.addWidget(choose_btn)
        dir_row.addWidget(rescan_btn)

        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(2, 1)
        self.slot_combos: dict[str, QtWidgets.QComboBox] = {}
        for r, (key, label, protocol, _which) in enumerate(PROTOCOL_SLOTS):
            grid.addWidget(QtWidgets.QLabel(label), r, 0)
            pro_label = QtWidgets.QLabel(f"{protocol}.pro")
            pro_label.setStyleSheet("color: #898781;")
            grid.addWidget(pro_label, r, 1)

            combo = QtWidgets.QComboBox()
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.currentIndexChanged.connect(self._mark_files_dirty)
            self.slot_combos[key] = combo
            grid.addWidget(combo, r, 2)

            browse = QtWidgets.QPushButton("Browse...")
            browse.clicked.connect(lambda _checked, k=key: self.browse_for_slot(k))
            grid.addWidget(browse, r, 3)

        self.save_files_btn = QtWidgets.QPushButton("Save file assignments  (Ctrl+S)")
        self.save_files_btn.clicked.connect(self.save_files)
        self.clear_files_btn = QtWidgets.QPushButton("Clear all")
        self.clear_files_btn.clicked.connect(self.clear_slots)

        file_buttons = QtWidgets.QHBoxLayout()
        file_buttons.addWidget(self.clear_files_btn)
        file_buttons.addStretch(1)
        file_buttons.addWidget(self.save_files_btn)

        panel = QtWidgets.QVBoxLayout(self.file_panel)
        panel.addWidget(self.file_panel_header)
        panel.addLayout(dir_row)
        panel.addLayout(grid)
        panel.addLayout(file_buttons)
        return self.file_panel

    # -- defaults ------------------------------------------------------
    def _known_genotypes(self) -> list[str]:
        seen: list[str] = []
        for row in self.rows:
            g = (row.get("genotype") or "").strip()
            if g and g not in seen:
                seen.append(g)
        return seen

    def _prefill_from_last_row(self) -> None:
        """Date/slice/genotype carry over from the last logged cell; the cell
        number continues from it (+1). With no CSV yet: today / slice 1 / cell 1."""
        if not self.rows:
            self.date_edit.setText(dt.date.today().strftime("%y%m%d"))
            self.slice_spin.setValue(1)
            self.cell_spin.setValue(1)
            return
        last = self.rows[-1]
        date = (last.get("date") or "").strip()
        self.date_edit.setText(date if valid_yymmdd(date) else dt.date.today().strftime("%y%m%d"))
        # setValue on the slice spin fires _on_slice_changed, which resets the
        # cell number -- so the cell number has to be set after the slice.
        self.slice_spin.setValue(min(999, max(1, as_int(last.get("slice"), 1))))
        self.cell_spin.setValue(min(999, max(1, as_int(last.get("cell"), 0) + 1)))
        self.genotype_combo.setCurrentText((last.get("genotype") or "").strip())
        self.date_edit.setFocus()
        self.date_edit.selectAll()

    def _on_slice_changed(self, _value: int) -> None:
        # A new slice starts a new cell series; the field stays editable.
        self.cell_spin.setValue(1)

    # -- table ---------------------------------------------------------
    def _fill_table(self) -> None:
        self.table.setRowCount(0)
        for row in self.rows:
            self._add_table_row(row)

    def _row_cells(self, row: dict) -> list[str]:
        n_files = sum(1 for k in FILE_KEYS if (row.get(k) or "").strip())
        return [str(row.get(k, "")) for k in ("date", "slice", "cell", "rmp_mv", "genotype")] \
            + [f"{n_files}/{len(FILE_KEYS)}"]

    def _add_table_row(self, row: dict) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self._set_table_row(r, row)
        self.table.scrollToBottom()

    def _set_table_row(self, r: int, row: dict) -> None:
        for col, text in enumerate(self._row_cells(row)):
            self.table.setItem(r, col, QtWidgets.QTableWidgetItem(text))

    # -- save cell -----------------------------------------------------
    def save_entry(self) -> None:
        date = self.date_edit.text().strip()
        if not valid_yymmdd(date):
            self._warn("Date must be a real calendar date in YYMMDD form, e.g. 260901.")
            self.date_edit.setFocus()
            self.date_edit.selectAll()
            return

        genotype = self.genotype_combo.currentText().strip()
        if not genotype:
            self._warn("Enter a genotype.")
            self.genotype_combo.setFocus()
            return

        slice_no, cell_no = self.slice_spin.value(), self.cell_spin.value()
        if self._is_duplicate(date, slice_no, cell_no):
            keep = QtWidgets.QMessageBox.question(
                self, "Already logged",
                f"{date} slice {slice_no} cell {cell_no} is already in the CSV.\nAdd it again?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if keep != QtWidgets.QMessageBox.Yes:
                return

        # The panel is about to be re-pointed at the new cell -- don't let
        # unsaved pairings for the previous one vanish silently.
        if self._files_dirty and not self._offer_to_save_pending_files():
            return

        row = {
            "date": date,
            "slice": slice_no,
            "cell": cell_no,
            "rmp_mv": f"{self.rmp_spin.value():.1f}",
            "genotype": genotype,
            "entered_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        row.update({k: "" for k in FILE_KEYS})
        try:
            append_row(self.csv_path, row)
        except OSError as exc:
            self._warn(f"Could not write {self.csv_path}:\n{exc}")
            return

        self.rows.append({k: str(v) for k, v in row.items()})
        self._add_table_row(row)
        if self.genotype_combo.findText(genotype) < 0:
            self.genotype_combo.addItem(genotype)
        self.status.setText(f"Added {date} slice {slice_no} cell {cell_no}  ->  {self.csv_path.name}")

        # Advance the cell number for the next recording; date/slice/genotype
        # stay as they are. Guarded against the 999 ceiling.
        if cell_no < self.cell_spin.maximum():
            self.cell_spin.setValue(cell_no + 1)

        self._reveal_file_panel(len(self.rows) - 1)

    def _is_duplicate(self, date: str, slice_no: int, cell_no: int) -> bool:
        return any(
            (r.get("date") or "").strip() == date
            and as_int(r.get("slice"), -1) == slice_no
            and as_int(r.get("cell"), -1) == cell_no
            for r in self.rows
        )

    def _offer_to_save_pending_files(self) -> bool:
        """Ask about unsaved pairings. False means "cancel, stay put"."""
        label = self._target_label() or "the previous cell"
        answer = QtWidgets.QMessageBox.question(
            self, "Unsaved protocol files",
            f"The protocol files for {label} haven't been saved.\nSave them first?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save,
        )
        if answer == QtWidgets.QMessageBox.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.Save:
            self.save_files()
        self._files_dirty = False
        return True

    # -- protocol panel ------------------------------------------------
    def _target_label(self) -> str | None:
        if self.target_row is None or self.target_row >= len(self.rows):
            return None
        row = self.rows[self.target_row]
        return f"{row.get('date')} slice {row.get('slice')} cell {row.get('cell')}"

    def _reveal_file_panel(self, row_index: int) -> None:
        self.target_row = row_index
        self.file_panel.setVisible(True)
        self.file_panel_header.setText(f"Files for {self._target_label()}")
        self.rescan(reassign=True)
        first = self.slot_combos[PROTOCOL_SLOTS[0][0]]
        first.setFocus()

    def choose_abf_dir(self) -> None:
        start = str(self.abf_dir) if self.abf_dir and self.abf_dir.exists() else ""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "ABF folder", start)
        if not folder:
            return
        self.abf_dir = Path(folder)
        self.abf_dir_edit.setText(folder)
        self.settings.setValue("abf_dir", folder)
        self.rescan(reassign=True)

    def rescan(self, reassign: bool = True) -> None:
        """Re-read the candidate .abf files and (re)fill the slot dropdowns."""
        if not self.abf_dir or not self.abf_dir.is_dir():
            self.candidates = []
            self._fill_combos({})
            self.status.setText("Choose the ABF folder to match protocol files.")
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.candidates = scan_candidates(self.abf_dir)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        assigned = auto_assign(self.candidates) if reassign else self._current_assignment()
        self._fill_combos(assigned)
        matched = len(assigned)
        self.status.setText(
            f"{len(self.candidates)} recent .abf file(s) in {self.abf_dir.name}; "
            f"{matched}/{len(PROTOCOL_SLOTS)} protocol(s) matched automatically."
        )
        self._files_dirty = False

    def _candidate_label(self, cand: dict) -> str:
        when = cand["when"].strftime("%H:%M:%S") if cand["when"] else "--:--:--"
        protocol = cand["protocol"] or "protocol unreadable"
        return f"{cand['path'].name}   {when}   [{protocol}]"

    def _fill_combos(self, assigned: dict[str, Path]) -> None:
        for key, combo in self.slot_combos.items():
            combo.blockSignals(True)  # a programmatic refill isn't a user edit
            combo.clear()
            combo.addItem("(none)", "")
            for cand in self.candidates:
                combo.addItem(self._candidate_label(cand), str(cand["path"]))
            want = assigned.get(key)
            if want is not None:
                idx = combo.findData(str(want))
                if idx < 0:  # a Browse pick from outside the candidate window
                    combo.addItem(Path(want).name, str(want))
                    idx = combo.count() - 1
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _current_assignment(self) -> dict[str, Path]:
        out = {}
        for key, combo in self.slot_combos.items():
            data = combo.currentData()
            if data:
                out[key] = Path(data)
        return out

    def browse_for_slot(self, key: str) -> None:
        start = str(self.abf_dir) if self.abf_dir and self.abf_dir.is_dir() else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select .abf file", start, "ABF recordings (*.abf);;All files (*)")
        if not path:
            return
        combo = self.slot_combos[key]
        idx = combo.findData(path)
        if idx < 0:
            combo.addItem(Path(path).name, path)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)

    def clear_slots(self) -> None:
        for combo in self.slot_combos.values():
            combo.setCurrentIndex(0)

    def _mark_files_dirty(self, _index: int) -> None:
        self._files_dirty = True

    def save_files(self) -> None:
        if self.target_row is None or self.target_row >= len(self.rows):
            return
        assignment = self._current_assignment()

        # Two slots pointing at the same recording is nearly always a
        # mis-click, but the run may legitimately have only one membrane test.
        labels = {k: l for k, l, _p, _w in PROTOCOL_SLOTS}
        used: dict[str, str] = {}
        for key, path in assignment.items():
            label = labels[key]
            if str(path) in used:
                keep = QtWidgets.QMessageBox.question(
                    self, "Same file twice",
                    f"{Path(path).name} is assigned to both "
                    f"'{used[str(path)]}' and '{label}'.\nSave anyway?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
                if keep != QtWidgets.QMessageBox.Yes:
                    return
                break
            used[str(path)] = label

        # File names are stored bare, with the folder in its own abf_dir
        # column -- five absolute paths per row makes the CSV unreadable in
        # Excel. A Browse pick from some other folder keeps its full path, so
        # every entry still resolves without guessing.
        row = self.rows[self.target_row]
        row["abf_dir"] = str(self.abf_dir) if self.abf_dir else ""
        for key, _l, _p, _w in PROTOCOL_SLOTS:
            path = assignment.get(key)
            if not path:
                row[f"file_{key}"] = ""
            elif self.abf_dir and path.parent == self.abf_dir:
                row[f"file_{key}"] = path.name
            else:
                row[f"file_{key}"] = str(path)
        try:
            rewrite_rows(self.csv_path, self.rows)
        except OSError as exc:
            self._warn(f"Could not write {self.csv_path}:\n{exc}")
            return

        self._set_table_row(self.target_row, row)
        self._files_dirty = False
        self.status.setText(
            f"Saved {len(assignment)}/{len(PROTOCOL_SLOTS)} protocol file(s) for "
            f"{self._target_label()}  ->  {self.csv_path.name}"
        )

    # -- shutdown ------------------------------------------------------
    def closeEvent(self, event) -> None:
        if self._files_dirty and not self._offer_to_save_pending_files():
            event.ignore()
            return
        event.accept()

    def _warn(self, message: str) -> None:
        QtWidgets.QMessageBox.warning(self, "Check the entry", message)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path(DEFAULT_CSV),
                        help=f"CSV to append to and prefill from (default: {DEFAULT_CSV})")
    parser.add_argument("--abf-dir", type=Path, default=None,
                        help="folder of .abf recordings to match (default: last folder used)")
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    win = MetadataWindow(args.csv, args.abf_dir)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
