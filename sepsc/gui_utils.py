"""safe_callback: shared exception-safety wrapper for GUI callbacks, used by
both the Tk/matplotlib GUIs (label.py) and the PyQt5/matplotlib ones
(review.py, inspect.py, optimize.py) -- and shared PyQt5 dialog-field
helpers (minianalysis.py, optimize.py, launcher.py)."""

from __future__ import annotations

import functools
import sys
import traceback


# ---------------------------------------------------------------------------
# Generic PyQt5 dialog-field widgets, keyed the same way across every
# sepsc GUI dialog that edits a small set of named numeric/choice/bool
# settings: (kind, kwargs-for-the-setters, current-value) in, a configured
# widget out (make_field_widget), and the reverse for reading it back
# (read_field_widget). One shared pair so every settings dialog (detection
# parameters, filter/resample, the method launcher) renders/reads fields
# identically instead of re-implementing this per module.
# ---------------------------------------------------------------------------

def make_field_widget(kind: str, kwargs: dict, current):
    from PyQt5 import QtWidgets
    if kind == "double":
        w = QtWidgets.QDoubleSpinBox()
        for k, val in kwargs.items():
            getattr(w, f"set{k[0].upper()}{k[1:]}")(val)
        w.setValue(current)
    elif kind == "int":
        w = QtWidgets.QSpinBox()
        for k, val in kwargs.items():
            getattr(w, f"set{k[0].upper()}{k[1:]}")(val)
        w.setValue(current)
    elif kind == "choice":
        w = QtWidgets.QComboBox()
        w.addItems(kwargs["choices"])
        w.setCurrentText(current)
    elif kind == "bool":
        w = QtWidgets.QCheckBox()
        w.setChecked(bool(current))
    else:
        raise ValueError(kind)
    return w


def read_field_widget(w, kind: str):
    if kind in ("double", "int"):
        return w.value()
    if kind == "choice":
        return w.currentText()
    if kind == "bool":
        return w.isChecked()
    raise ValueError(kind)


# Shared field specs for sepsc.preprocess.load_filtered_trace's own settings
# (cutoff_hz, target_rate_hz, filter_order) -- one definition so every
# dialog offering filter/downsample (minianalysis.py, launcher.py) presents
# the same fields with the same widget ranges/steps.
FILTER_FIELD_SPECS = [
    ("cutoff_hz", "Bessel low-pass cutoff, Hz", "double", dict(minimum=1.0, maximum=1e6, decimals=1, singleStep=100.0)),
    ("target_rate_hz", "Resample to, Hz", "double", dict(minimum=1.0, maximum=1e7, decimals=1, singleStep=1000.0)),
    ("filter_order", "Bessel filter order", "int", dict(minimum=1, maximum=16)),
]


def safe_callback(fn):
    """Never let an exception escape a Tk/matplotlib callback silently.

    An uncaught exception inside a button/key callback can leave the Tk
    mainloop in a state where the window stops updating but still answers
    the OS "are you responding" ping -- it looks frozen forever with no
    error printed anywhere. Catch, print (flushed immediately, since stdout
    is fully buffered once redirected to a file/pipe), and re-raise nothing.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            print(f"ERROR in {fn.__name__}:", file=sys.stderr, flush=True)
            traceback.print_exc()
            sys.stderr.flush()
    return wrapper
