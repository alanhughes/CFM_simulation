"""Test for the default scheme list of scripts/plot_cfm.py (BUG-201)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "plot_cfm", Path(__file__).resolve().parents[1] / "scripts" / "plot_cfm.py"
)
plot_cfm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_cfm)


def test_default_phys_rho_is_one_name_in_a_list():
    assert plot_cfm.default_phys_rho({"physRho": "GSFC2020"}) == ["GSFC2020"]
