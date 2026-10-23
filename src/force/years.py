"""The years of a run for one RCM (DISC-001, option D).

start_year and end_year in config.yaml are the default years of all the RCMs. The data of an RCM can stop before
end_year: the ANT-12 (RACMO) data stops at 2023-12-31, MAR and XPEN055 (RACMO23) at 2024-12-31. run_years() gives the
years for one RCM:

- an end year from config.yaml is clipped to the last complete year of the data, with a warning;
- an end year from the command line (--end-year) is used as it is, and it is an error if the data does not cover it;
- the start year is never changed: data that starts after it is an error, as it would shorten the spin-up;
- a clipped end year before spin_end_year is an error.

The data years come from the file names only (extract.file_days); no data file is opened. A year is complete if the
data of every variable reaches 31 December of it. A missing year inside the range is not checked here: the forcing
code stops on it (BUG-003).

Each script resolves the years once, at its start, and then sets them in config for the rest of the process
(set_config_years), as the forcing and run code reads config["start_year"] and config["end_year"]. scripts/run_cfm.py
gives its years to sim.worker on the command line, so the two processes of a run use the same years.
"""

import argparse
import datetime
import glob
import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional

from cfm_config import config

from .cache import ForcingCoverageError
from .extract import DATASETS, ExtractSettings, file_days
from .model_point import rcm_release

CONFIG = "config"
CLI = "cli"


@dataclass(frozen=True)
class DataDays:
    """The first and last day of the data of an RCM, for which every variable has data (from the file names)."""

    first_day: datetime.date
    last_day: datetime.date

    @property
    def first_year(self) -> int:
        """The first year whose data starts on 1 January."""
        return self.first_day.year if (self.first_day.month, self.first_day.day) == (1, 1) else self.first_day.year + 1

    @property
    def last_year(self) -> int:
        """The last year whose data reaches 31 December."""
        return self.last_day.year if (self.last_day.month, self.last_day.day) == (12, 31) else self.last_day.year - 1


@dataclass(frozen=True)
class RunYears:
    """The years of a run, and the end year that was asked for (for the run manifest)."""

    start_year: int
    end_year: int
    requested_end_year: int
    end_year_source: str  # CONFIG (config.yaml) or CLI (--end-year)

    @property
    def clipped(self) -> bool:
        return self.end_year != self.requested_end_year


def data_days(rcm_name: str, cfg: Optional[Mapping] = None) -> DataDays:
    """The days of the data of rcm_name in its configured data path, from the file names of its source variables."""
    settings = ExtractSettings.from_config(rcm_name, cfg)
    spec = settings.spec
    # MAR has all the variables in each file, so its variables share one pattern
    patterns = dict.fromkeys(
        os.path.join(settings.data_path, spec.file_glob.format(var=var)) for var in settings.source_vars
    )
    firsts, lasts = [], []
    for pattern in patterns:
        paths = glob.glob(pattern)
        if not paths:
            raise FileNotFoundError(f"No {rcm_name} files that match {pattern}")
        days = [file_days(spec, path) for path in paths]
        firsts.append(min(first for first, _ in days))
        lasts.append(max(last for _, last in days))
    return DataDays(first_day=max(firsts), last_day=min(lasts))


def run_years(
    rcm_name: str, start_year: Optional[int] = None, end_year: Optional[int] = None, cfg: Optional[Mapping] = None
) -> RunYears:
    """The years of a run of rcm_name. start_year and end_year are the command line values (None: config.yaml).

    Stops with ForcingCoverageError if the data starts after the start year, if an end year from the command line is
    after the data, or if the clipped end year is before spin_end_year.
    """
    cfg = config if cfg is None else cfg
    if rcm_name not in DATASETS:
        raise ValueError(f"Unknown rcm_name {rcm_name!r}. Choose one of {sorted(DATASETS)}.")
    start = int(cfg["start_year"]) if start_year is None else int(start_year)
    requested_end = int(cfg["end_year"]) if end_year is None else int(end_year)
    source = CONFIG if end_year is None else CLI
    if start > requested_end:
        raise ValueError(f"start year {start} is after end year {requested_end}")

    days = data_days(rcm_name, cfg)
    dataset = f"{rcm_name} ({rcm_release(rcm_name, cfg[DATASETS[rcm_name].data_path_key])})"
    if days.first_year > start:
        raise ForcingCoverageError(
            f"{dataset} data starts at {days.first_day}, after January {start}. The start year is not changed, as "
            "that would shorten the spin-up. Change start_year (and spin_end_year), or use --start-year."
        )
    end = requested_end
    if days.last_year < requested_end:
        if source == CLI:
            raise ForcingCoverageError(
                f"{dataset} data stops at {days.last_day}, before December {requested_end} (--end-year)."
            )
        end = days.last_year
        if end < int(cfg["spin_end_year"]):
            raise ForcingCoverageError(
                f"{dataset} data stops at {days.last_day}: end_year {end} would be before spin_end_year "
                f"{cfg['spin_end_year']}."
            )
        logging.warning(
            f"{dataset} data stops at {days.last_day}: end_year {end}, not {requested_end} from config.yaml"
        )
    if start > end:
        raise ValueError(f"start year {start} is after end year {end}")
    return RunYears(start, end, requested_end, source)


def set_config_years(years: RunYears) -> None:
    """Set start_year and end_year in config for this process, for the code that reads them from config."""
    config["start_year"] = years.start_year
    config["end_year"] = years.end_year


def add_year_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --start-year and --end-year, the command line overrides of run_years()."""
    parser.add_argument("--start-year", type=int, help="first year (default: start_year in config.yaml)")
    parser.add_argument(
        "--end-year",
        type=int,
        help="last year, used as it is (default: end_year in config.yaml, clipped to the last complete year of the "
        "RCM data)",
    )
