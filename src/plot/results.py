import logging
import os
from typing import Dict, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cfm_config import cfm_version, config, sites
from force.model_point import model_point_label, shared_model_points
from force.years import run_years
from sim.runs import find_run

logging.basicConfig(level=logging.INFO)


class ResultsPlotter:
    def __init__(
        self,
        site: str,
        phys_rho: List[str],
        rcm_name: Literal["RACMO", "MAR"],
        liquid: Literal["bucket"],
        cfm: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> None:
        """Initialize the ResultsPlotter with a site and physical densification schemes.
        Args:
            site (str): Site name in sites.yaml. The runs are those of its model point (FEAT-001, FEAT-002)
            phys_rho (List[str]): List of physical densification schemes to plot
            rcm_name (Literal["RACMO", "MAR"]): RCM name to use for forcing data (e.g., MAR or RACMO)
            liquid (Literal["bucket"]): Meltwater scheme to use (currently only bucket)
            cfm (str | None): Label of the CFM version (default_cfm_version if None)
            start_year, end_year (int | None): the years of the runs, used as they are. None: the years of
                force.years.run_years for rcm_name, as scripts/run_cfm.py uses (DISC-001)
        """

        self.site: str = site
        self.cfm: str = cfm_version(cfm)["label"]
        self.phys_rho: List[str] = phys_rho
        self.rcm_name: Literal["RACMO", "MAR"] = rcm_name
        self.liquid: Literal["bucket"] = liquid
        years = run_years(rcm_name, start_year, end_year)
        self.start_year: int = years.start_year
        self.end_year: int = years.end_year

        phys_rho_str = "_".join(self.phys_rho)
        # the CFMresults.hdf5 of each scheme, found from the run manifests
        self._output_paths: List[str] = [
            find_run(
                site=site,
                rcm_name=rcm_name,
                physRho=scheme,
                liquid=liquid,
                start_year=self.start_year,
                end_year=self.end_year,
                cfm=self.cfm,
            ).results_file
            for scheme in self.phys_rho
        ]
        self._figure_path: str = (
            f"{config['CFM_data_path']}/cfm_figures/figures_{site}_{self.start_year}_{self.end_year}_{phys_rho_str}_"
            f"{self.liquid}_{self.rcm_name}_{self.cfm}"
        )
        os.makedirs(self._figure_path, exist_ok=True)

        # model point of the site and the other sites in its cell; also the inactive sites, as in the other plots (FEAT-002)
        self._model_point_label: str = model_point_label(site, shared_model_points(sites(active_only=False), rcm_name))

        self._thickness_change_range: List[float] = config[
            "thickness_change_range"
        ]  # Range of model time to plot for change in firn thickness (in years)
        self._results_dicts: Dict[str, Dict[str, xr.DataArray]] = {}

    def plot(self) -> None:
        """Main function to load the data and create the plots for the specified physical densification schemes.
        This function will call the internal functions to load the data and create the plots, and can be called from an external script or notebook to generate the figures.
        """

        logging.info(f"Plotting results for site={self.site}, phys_rho={self.phys_rho}, cfm={self.cfm}")
        logging.info(f"Figures will be saved to: {self._figure_path}")

        logging.info("Loading data...")
        self._load_data()

        logging.info("Data loaded, now plotting DIP time series...")
        self._plot_DIP_time_series()

        logging.info("DIP time series plotted, now plotting change in firn thickness...")
        self._plot_firn_thickness_change()

        logging.info("DIP time series plotted, now plotting density profiles...")
        self._plot_density_profiles()

    def _load_data(self) -> None:
        """Load the CFM output data from the specified output paths and store it in a dictionary for plotting.
        The data is stored in a dictionary where the keys are the output paths and the values are dictionaries containing the model time, depth, density, and temperature data as xarray DataArrays.
        """

        for phys_rho, output_path in zip(self.phys_rho, self._output_paths):
            with xr.open_dataset(output_path, engine="h5netcdf", phony_dims="sort") as ds:
                self._results_dicts[phys_rho] = {
                    "model_time_matrix": ds["density"][1:, 0],
                    "model_time_vector": ds["DIP"][1:, 0],
                    "depth": ds["depth"][1:],
                    "density": ds["density"][1:, 1:],
                    "temperature": ds["temperature"][1:, 1:],
                    "DIP": ds["DIP"][1:, 1:],
                }

    def _plot_density_profiles(self) -> None:
        """Plot the density profiles for each physical densification scheme and save the figures to the specified figure paths."""

        for phys_rho, results_dict in self._results_dicts.items():
            # Create a plot of density vs depth using matplotlib
            plt.figure(figsize=(6, 8))
            n_lines = results_dict["density"].shape[0]
            colors = plt.cm.viridis(np.linspace(0, 1, n_lines))
            for i in range(n_lines):
                plt.plot(
                    results_dict["density"][i, :],
                    results_dict["depth"],
                    linewidth=0.5,
                    alpha=0.7,
                    color=colors[i],
                )
            plt.xlabel("Density (kg/m^3)")
            plt.ylabel("Depth (m)")
            plt.title(f"{phys_rho} density profiles\n{self._model_point_label}")
            plt.grid()
            plt.gca().invert_yaxis()  # Invert y-axis to have depth increasing downwards
            plt.savefig(f"{self._figure_path}/{phys_rho}_density_profile.png")
            plt.close()

    def _plot_DIP_time_series(self) -> None:
        """Plot the DIP time series for each physical densification scheme, all on the same plot, and save the figure to the specified figure path."""

        plt.figure(figsize=(10, 6))
        for phys_rho, results_dict in self._results_dicts.items():
            plt.plot(
                results_dict["model_time_vector"].values,
                results_dict["DIP"][:, 0],
                label=f"{phys_rho}",
            )
        plt.xlabel("Model Time (years)")
        plt.ylabel("DIP (m)")
        plt.title(f"DIP Time Series\n{self._model_point_label}")
        plt.legend()
        plt.grid()
        plt.savefig(f"{self._figure_path}/DIP_time_series.png")
        plt.close()

    def _plot_firn_thickness_change(self) -> None:
        """Plot the change in firn thickness over time for each physical densification scheme and save the figures to the specified figure paths.
        This function will create a plot of change in firn thickness vs model time for each physical densification scheme and save the figure to the corresponding figure path.
        """

        plt.figure(figsize=(10, 6))
        for phys_rho, results_dict in self._results_dicts.items():
            dz = results_dict["depth"][:-1] - results_dict["depth"][1:]
            dz = np.mean(dz)
            firn_mask = results_dict["density"] <= config["cfm_config"]["RhoImp"]
            firn_thickness = np.sum(firn_mask * dz, axis=1)
            # firn_delta[i] represents the change from time t_i to t_{i+1}, so we
            # associate it with the starting time t_i (model_time_vector[:-1]).
            firn_delta = firn_thickness[1:] - firn_thickness[:-1]
            plt.plot(
                results_dict["model_time_vector"].values[:-1],
                firn_delta,
                label=f"{phys_rho}",
            )

        plt.xlim(self._thickness_change_range)  # Set x-axis limits to the specified range
        plt.xlabel("Model Time (years)")
        plt.ylabel("Change in Firn Thickness (m)")
        plt.title(f"Change in Firn Thickness over Time\n{self._model_point_label}")
        plt.grid()
        plt.legend()
        plt.savefig(f"{self._figure_path}/firn_thickness_change.png")
        plt.close()
