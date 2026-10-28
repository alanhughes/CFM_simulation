import matplotlib.pyplot as plt
import xarray as xr
import logging
from typing import Dict, Literal, Optional, Tuple

from cfm_config import config, find_on_share, share_data_path

logging.basicConfig(level=logging.INFO)


def melt_map_path(rcm_name: str) -> str:
    """Path of the average yearly melt map for a read: the local file if it exists, else the copy in
    {CFM_share_data_path}/melt/ if it exists, else the local path (FEAT-008)."""
    name = f"{rcm_name}_avg_yearly_melt_map_{config['start_year']}-{config['end_year']}.nc"
    local = f"{config['CFM_data_path']}/melt/{name}"
    share = share_data_path()
    return local if share is None else find_on_share(local, f"{share}/melt/{name}")


def plot_larsen_c_melt(
    rcm_name: Literal["mar", "racmo"],
    lat_name: str,
    lon_name: str,
    borehole_sites: Dict[str, Tuple[float, float]],
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Loads and plots the average yearly melt map for the Larsen C area for a given RCM.

    Args:
        rcm_name (str): name of the RCM to plot melt for - either 'mar' or 'racmo'
        lat_name (str): name of latitude variable in the melt dataset
        lon_name (str): name of longitude variable in the melt dataset
        borehole_sites (Dict[str, Tuple[float, float]]): dictionary of borehole site names and their lat/lon coordinates to plot on the map
        labels (Dict[str, str], optional): legend label for each site, for example its model point. Default: the name.
    """

    melt_ds = xr.open_dataarray(melt_map_path(rcm_name))

    # restrict to larsenC_box
    melt_ds = melt_ds.where(
        (melt_ds[lat_name] >= config["larsenC_box"]["lat_min"])
        & (melt_ds[lat_name] <= config["larsenC_box"]["lat_max"])
        & (melt_ds[lon_name] >= config["larsenC_box"]["lon_min"])
        & (melt_ds[lon_name] <= config["larsenC_box"]["lon_max"]),
        drop=True,
    )

    # plot the melt map
    plt.figure(figsize=(10, 8))
    melt_ds.plot(x=lon_name, y=lat_name, cmap="Blues", vmax=1200)
    for site, (lat, lon) in borehole_sites.items():
        plt.plot(lon, lat, marker="o", markersize=5, label=(labels or {}).get(site, site))
    plt.legend(fontsize=7)
    plt.title(f"Average Yearly Melt Map for Larsen C ({rcm_name.upper()})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.xlim(config["larsenC_box"]["lon_min"], config["larsenC_box"]["lon_max"])
    plt.ylim(config["larsenC_box"]["lat_min"], config["larsenC_box"]["lat_max"])
    plt.savefig(
        f"{config['CFM_data_path']}/cfm_figures/melt_maps/melt_avg_{rcm_name}_larsenC_{config['start_year']}_{config['end_year']}.png"
    )
    plt.close()
