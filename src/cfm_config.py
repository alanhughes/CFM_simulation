"""Loads config.yaml, expanding "{cpom_mount}" in top level path values so the luna mount point is set in one place."""

import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# only expand {cpom_mount} - other placeholders (e.g. in force_data_save_path_pattern) are filled in where they're used
for key, value in config.items():
    if isinstance(value, str):
        config[key] = value.replace("{cpom_mount}", config["cpom_mount"])
