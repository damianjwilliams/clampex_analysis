"""
Shared visual constants for every sepsc plot/GUI -- one source of truth so
"fast" and "slow" are drawn identically everywhere (the labeling GUI, the
classifier's trace plot, and the overlay plot used to disagree slightly on
the exact hex values; they're unified here).

COLORS/MARKERS come from the dataviz skill's reference categorical palette
(slots 1-2: blue, orange), validated colorblind-safe as an adjacent pair.
"""

COLORS = {"fast": "#2a78d6", "slow": "#eb6834"}
MARKERS = {"fast": "^", "slow": "s"}

# chart chrome tokens (dataviz skill reference palette, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
TRACE = "#333333"  # raw-trace line color, distinct from ink/muted so it recedes behind markers
