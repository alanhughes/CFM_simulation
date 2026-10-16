"""Loads config.yaml, expanding "{cpom_mount}" in top level path values so the luna mount point is set in one place.

Also loads the site table, sites.yaml (FEAT-002), with sites(), get_site() and batches().
"""

import calendar
import datetime
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SITES_FILE = REPO_ROOT / "sites.yaml"
SITES_LOCK_FILE = REPO_ROOT / "sites.lock"

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# only expand {cpom_mount} - other placeholders (e.g. in force_data_save_path_pattern) are filled in where they're used
for key, value in config.items():
    if isinstance(value, str):
        config[key] = value.replace("{cpom_mount}", config["cpom_mount"])

# keys that moved from config.yaml to sites.yaml (FEAT-002) - stop, so that an old config.yaml is not used by mistake
_MOVED_TO_SITES_YAML = ["borehole_sites", "sites_batch_1", "sites_batch_2", "sites_batch_3"]
_moved = [key for key in _MOVED_TO_SITES_YAML if key in config]
if _moved:
    raise KeyError(
        f"config.yaml contains {', '.join(_moved)}. The sites and batches are now in sites.yaml (FEAT-002). "
        "Delete these keys from config.yaml."
    )


def format_coord(value: float) -> str:
    """Format a coordinate to 3 dp, rounding half up on its decimal value (-64.1485 -> "-64.149", not "-64.148")."""
    return str(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def run_id(lat: float, lon: float, phys_rho: str, liquid: str, rcm_name: str) -> str:
    """Identifier shared by the CFM output folder and the CFM config JSON for one run."""
    return (
        f"{format_coord(lat)}_{format_coord(lon)}_{config['start_year']}_{config['end_year']}"
        f"_{phys_rho}_{liquid}_{rcm_name}"
    )


def cfm_output_folder(lat: float, lon: float, phys_rho: str, liquid: str, rcm_name: str) -> str:
    """Full path of the CFM output folder for one run."""
    return f"{config['CFM_data_path']}/cfm_output/CFMoutput_{run_id(lat, lon, phys_rho, liquid, rcm_name)}"


@dataclass(frozen=True)
class Site:
    """One entry of sites.yaml. The name, lat and lon of a site do not change (see sites.lock)."""

    name: str
    lat: float
    lon: float
    windows: Tuple[Tuple[datetime.date, datetime.date], ...] = ()
    profile: Optional[str] = None
    active: bool = True
    superseded_by: Optional[str] = None
    extra: Dict = field(default_factory=dict, compare=False)

    @property
    def profile_file(self) -> str:
        """The depth-density CSV file name in in_situ_data_path."""
        return self.profile or f"{self.name}_depth-density.csv"

    def decimal_year_windows(self) -> List[Tuple[float, float]]:
        """The windows as [start, end) in decimal years, as the CFM time vector uses. See decimal_year()."""
        if not self.windows:
            raise ValueError(f"Site {self.name} has no windows in sites.yaml. Add its measurement windows first.")
        return [(decimal_year(start), decimal_year(end + datetime.timedelta(days=1))) for start, end in self.windows]


def decimal_year(date: datetime.date) -> float:
    """Decimal year with 12 equal months: 2014-12-01 -> 2014 + 11/12. This gives the windows that
    scripts/plot_density_error.py used before FEAT-002 (for example [2014 + 11/12, 2015 + 2/12])."""
    days_in_month = calendar.monthrange(date.year, date.month)[1]
    return date.year + (date.month - 1) / 12 + (date.day - 1) / (12 * days_in_month)


def _as_date(value, site: str) -> datetime.date:
    if isinstance(value, datetime.date):
        return value
    try:
        return datetime.date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"Site {site}: window date {value!r} is not YYYY-MM-DD") from None


def _load_sites_yaml(path: Path = SITES_FILE) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def _parse_site(name: str, entry: dict) -> Site:
    known = {"lat", "lon", "windows", "profile", "active", "superseded_by"}
    windows = []
    for window in entry.get("windows") or []:
        start, end = (_as_date(value, name) for value in window)
        if end < start:
            raise ValueError(f"Site {name}: window {start} to {end} ends before it starts")
        windows.append((start, end))
    return Site(
        name=name,
        lat=float(entry["lat"]),
        lon=float(entry["lon"]),
        windows=tuple(windows),
        profile=entry.get("profile"),
        active=bool(entry.get("active", True)),
        superseded_by=entry.get("superseded_by"),
        extra={key: value for key, value in entry.items() if key not in known},
    )


def sites(active_only: bool = True, path: Path = SITES_FILE) -> Dict[str, Site]:
    """The sites in sites.yaml, in file order. By default only the sites without "active: false"."""
    table = _load_sites_yaml(path)["sites"]
    all_sites = {name: _parse_site(name, entry) for name, entry in table.items()}
    return {name: site for name, site in all_sites.items() if site.active or not active_only}


def get_site(name: str, path: Path = SITES_FILE) -> Site:
    """One site of sites.yaml by name, active or not."""
    all_sites = sites(active_only=False, path=path)
    if name not in all_sites:
        raise KeyError(f"Site {name!r} is not in {path}. Known sites: {', '.join(all_sites)}")
    return all_sites[name]


def batches(path: Path = SITES_FILE) -> Dict[str, List[str]]:
    """The named lists of sites in sites.yaml."""
    return dict(_load_sites_yaml(path).get("batches") or {})
