"""Loads config.yaml, expanding "{cpom_mount}" in top level path values so the luna mount point is set in one place."""

from decimal import ROUND_HALF_UP, Decimal

import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# only expand {cpom_mount} - other placeholders (e.g. in force_data_save_path_pattern) are filled in where they're used
for key, value in config.items():
    if isinstance(value, str):
        config[key] = value.replace("{cpom_mount}", config["cpom_mount"])


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
