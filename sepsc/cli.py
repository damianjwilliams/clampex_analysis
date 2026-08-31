"""
Unified command-line entry point for the sepsc pipeline: python -m sepsc
<command> [args...]. Commands run in typical pipeline order:
detect -> review -> (label + train, once) -> classify -> overlay.
See COMMANDS below for what each one does.

Each command forwards its remaining arguments verbatim to that submodule's
own argparse parser (so `python -m sepsc detect --help` shows detect's own
options, not a generic summary) -- this dispatcher only picks which module
to run; it doesn't redeclare any of their flags.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

COMMANDS = {
    "launch": "Pop up a window to choose a detection method (minianalysis/fastmini/detect) plus "
              "optional filter/downsample, then run it",
    "detect": "Detect sEPSCs in a gap-free .abf using miniML (needs the clampex_miniml conda env)",
    "minianalysis": "Detect events with the classical Synaptosoft Mini-Analysis-style algorithm",
    "inspect": "Click a minianalysis-detected event to verify the detection-parameter windows used on it",
    "optimize": "Interactive 3-panel window to tune detection parameters: edit params, click a peak "
                "in the zoomable full trace, see it measured live",
    "review": "Manually accept/reject miniML-detected events",
    "label": "Hand-label fast/slow training events by clicking the trace",
    "train": "Train the fast/slow classifier from labeled events",
    "classify": "Apply the trained classifier to detected events",
    "overlay": "Plot peak-aligned, amplitude-normalized fast vs slow overlay",
    "compare": "Compare miniML vs minianalysis amplitude distributions on one recording",
    "view": "Interactive PyQt/pyqtgraph viewer: full trace with every detector's events marked",
    "preprocess": "Bessel low-pass filter + downsample a trace, checking the ABF header's own "
                   "hardware filter setting first",
    "fastmini": "Third detection method: per-recording-trained MLP classifier + iterative "
                "peel-off (needs a local clone of github.com/mrreganwang/Mini_Scripts)",
}

_MINIML_ENV_HINT = (
    "\n'detect' needs miniML/TensorFlow, which live in the separate 'clampex_miniml' conda\n"
    "env, not this project's main venv. Run it with that interpreter instead, e.g.:\n"
    "    <path-to-anaconda3>\\envs\\clampex_miniml\\python.exe -m sepsc detect ...\n"
)


def _print_top_help():
    width = max(len(n) for n in COMMANDS)
    lines = [f"usage: python -m sepsc <command> [args...]", "", textwrap.dedent(__doc__).strip(), "",
             "commands:"]
    lines += [f"    {name:<{width}}  {help_text}" for name, help_text in COMMANDS.items()]
    lines += ["", "Run `python -m sepsc <command> --help` for that command's own options."]
    print("\n".join(lines))


def main(argv=None):
    # Deliberately NOT argparse subparsers + REMAINDER here: that
    # combination intercepts "-h"/"--help" at the top-level parser instead
    # of forwarding it to the chosen subcommand's own parser (a known
    # argparse gotcha), which would break `sepsc <command> --help`. Plain
    # manual dispatch sidesteps it and is simpler besides.
    argv = sys.argv[1:] if argv is None else list(argv)

    if not argv or argv[0] in ("-h", "--help"):
        _print_top_help()
        sys.exit(0 if argv else 2)

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        _print_top_help()
        print(f"\npython -m sepsc: error: unknown command {command!r}", file=sys.stderr)
        sys.exit(2)

    try:
        module = importlib.import_module(f".{command}", package="sepsc")
    except ImportError as exc:
        if command == "detect":
            print(f"Could not import sepsc.detect: {exc}", file=sys.stderr)
            print(_MINIML_ENV_HINT, file=sys.stderr)
            sys.exit(1)
        raise

    module.main(rest)


if __name__ == "__main__":
    main()
