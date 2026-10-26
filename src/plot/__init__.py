"""Functionality for plotting forcing data and simulation output

The classes and functions below are imported when first used, not when the package loads. Thus plot.metrics and
plot.style can be imported without config.yaml (scripts/compare_cfm_runs.py in its folder mode, FEAT-007).
"""

import importlib

__version__ = "0.1.0"

_EXPORTS = {
    "ResultsPlotter": "results",
    "plot_larsen_c_melt": "melt",
    "SpatialPlotter": "spatial",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
