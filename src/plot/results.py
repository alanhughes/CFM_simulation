import logging
import os
from typing import Dict, List, Literal

import matplotlib.pyplot as plt
import numpy as np

from cfm_config import cfm_version, config, sites
from force.model_point import model_point_label, shared_model_points
from force.years import run_years
from plot import style
from plot.output import RunOutput
from sim.runs import find_run, spin_end_year_filter

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
        spin_end_year: int | str | None = None,
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
            spin_end_year (int | str | None): spin_end_year of the runs; None: spin_end_year of config.yaml; "any": the
                runs of any spin_end_year (FEAT-004)
        """

        self.site: str = site
        self.cfm: str = cfm_version(cfm)["label"]
        self.phys_rho: List[str] = phys_rho
        self.rcm_name: Literal["RACMO", "MAR"] = rcm_name
        self.liquid: Literal["bucket"] = liquid
        years = run_years(rcm_name, start_year, end_year)
        self.start_year: int = years.start_year
        self.end_year: int = years.end_year
        self.spin_end_year: int | None = spin_end_year_filter(spin_end_year)

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
                spin_end_year=self.spin_end_year,
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
        self._outputs: Dict[str, RunOutput] = {}

    def plot(self) -> None:
        """Main function to load the data and create the plots for the specified physical densification schemes.
        This function will call the internal functions to load the data and create the plots, and can be called from an external script or notebook to generate the figures.
        """

        logging.info(f"Plotting results for site={self.site}, phys_rho={self.phys_rho}, cfm={self.cfm}")
        logging.info(f"Figures will be saved to: {self._figure_path}")

        logging.info("Loading data...")
        style.apply()
        self._load_data()

        logging.info("Data loaded, now plotting DIP time series...")
        self._plot_DIP_time_series()

        logging.info("DIP time series plotted, now plotting change in firn thickness...")
        self._plot_firn_thickness_change()

        logging.info("DIP time series plotted, now plotting density profiles...")
        self._plot_density_profiles()

    def _load_data(self) -> None:
        """The output of each scheme, as a plot.output.RunOutput (FEAT-007). The arrays are read when first used."""

        for phys_rho, output_path in zip(self.phys_rho, self._output_paths):
            self._outputs[phys_rho] = RunOutput.from_folder(os.path.dirname(output_path), label=phys_rho)

    def _plot_density_profiles(self) -> None:
        """Plot the density profiles for each physical densification scheme and save the figures to the specified figure paths."""

        for phys_rho, output in self._outputs.items():
            # one line per output step; the colour is the time (a sequential scale), not the scheme
            fig, ax = plt.subplots(figsize=(6, 8))
            n_lines = output.density.shape[0]
            colors = plt.cm.viridis(np.linspace(0, 1, n_lines))
            for i in range(n_lines):
                ax.plot(output.density[i, :], output.depth, linewidth=0.5, alpha=0.7, color=colors[i])
            ax.set_xlabel("Density (kg/m³)")
            years = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(output.time[0], output.time[-1]))
            fig.colorbar(years, ax=ax, label="Year", shrink=0.6)
            ax.set_title(f"{phys_rho} density profiles, {int(output.time[0])}-{int(output.time[-1])}\n"
                         f"{self._model_point_label}", fontsize=8)
            style.depth_axis(ax, output.depth.max())
            style.save_figure(fig, f"{self._figure_path}/{phys_rho}_density_profile.png")

    def _plot_DIP_time_series(self) -> None:
        """Plot the DIP time series for each physical densification scheme, all on the same plot, and save the figure to the specified figure path."""

        fig, ax = plt.subplots(figsize=(10, 6))
        for phys_rho, output in self._outputs.items():
            ax.plot(output.time, output.fac, color=style.scheme_colour(phys_rho), label=phys_rho)
        ax.set_xlabel("Model time (years)")
        ax.set_ylabel("Firn air content, DIP (m)")
        ax.set_title(f"Firn air content\n{self._model_point_label}", fontsize=8)
        ax.legend()
        style.save_figure(fig, f"{self._figure_path}/DIP_time_series.png")

    def _plot_firn_thickness_change(self) -> None:
        """Plot the change in firn thickness over time for each physical densification scheme and save the figures to the specified figure paths.
        This function will create a plot of change in firn thickness vs model time for each physical densification scheme and save the figure to the corresponding figure path.
        """

        fig, ax = plt.subplots(figsize=(10, 6))
        for phys_rho, output in self._outputs.items():
            dz = np.mean(output.depth[:-1] - output.depth[1:])
            firn_mask = output.density <= config["cfm_config"]["RhoImp"]
            firn_thickness = np.sum(firn_mask * dz, axis=1)
            # firn_delta[i] represents the change from time t_i to t_{i+1}, so we
            # associate it with the starting time t_i (time[:-1]).
            firn_delta = firn_thickness[1:] - firn_thickness[:-1]
            ax.plot(output.time[:-1], firn_delta, color=style.scheme_colour(phys_rho), label=phys_rho)

        ax.set_xlim(self._thickness_change_range)  # Set x-axis limits to the specified range
        ax.set_xlabel("Model time (years)")
        ax.set_ylabel("Change in firn thickness (m)")
        ax.set_title(f"Change in firn thickness over time\n{self._model_point_label}", fontsize=8)
        ax.legend()
        style.save_figure(fig, f"{self._figure_path}/firn_thickness_change.png")
