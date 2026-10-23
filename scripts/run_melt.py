"""Script which gets maps of average yearly melt across AIS (or Larsen C), for both MAR and RACMO.

Each RCM has its own years, as in scripts/run_cfm.py (force.years.run_years, DISC-001): start_year and end_year of
config.yaml, with the end year clipped to the last complete year of the RCM data (RACMO: 1979-2023), or
--start-year/--end-year as given. The years are in the names of the melt map files.

Usage:
    - python run_melt.py [--start-year <year>] [--end-year <year>]

"""

import argparse

from force.melt import MeltMAR, MeltRACMO
from force.years import add_year_arguments, run_years, set_config_years

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make the average yearly melt maps of MAR and RACMO")
    add_year_arguments(parser)
    args = parser.parse_args()

    # resolve the years of both RCMs before the long reads; MeltMAR and MeltRACMO read them from config
    years = {rcm_name: run_years(rcm_name, args.start_year, args.end_year) for rcm_name in ("MAR", "RACMO")}
    set_config_years(years["MAR"])
    MeltMAR().get_melt_map()
    set_config_years(years["RACMO"])
    MeltRACMO().get_melt_map()
