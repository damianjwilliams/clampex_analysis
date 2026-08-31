"""
sepsc -- spontaneous EPSC detection, curation, and fast/slow classification.

A pipeline of small, single-purpose tools sharing one raw-trace convention
(gap-free voltage-clamp .abf, event windows peak-centered at a known sample
offset) and one feature set (sepsc.features), run either as
``python -m sepsc <command> ...`` (see sepsc.cli) or as
``python -m sepsc.<module> ...`` directly:

    launch        pop up a window to choose a detection method (minianalysis/
                  fastmini/detect) plus optional filter/downsample, then run it
    detect        miniML CNN-LSTM event detection (needs the clampex_miniml
                  conda env -- TensorFlow isn't in the main project venv)
    minianalysis  classical local-max + baseline + amplitude/area-threshold
                  detection (Synaptosoft Mini Analysis Program method)
    inspect       click a minianalysis-detected event to verify the exact
                  detection-parameter windows used on it
    optimize      interactive 3-panel window to tune detection parameters
                  against the zoomable full trace, live
    review     manually accept/reject miniML's detections
    label      hand-label fast/slow training events by clicking the trace
    train      train the fast/slow classifier from labeled events
    classify   apply the trained classifier to detected events
    overlay    plot peak-aligned, amplitude-normalized fast vs slow overlay
    view       interactive PyQt viewer: full trace with events marked

Typical pipeline: detect -> review -> (label + train, once) -> classify.
"""

__all__ = ["cli"]
