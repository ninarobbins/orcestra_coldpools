import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import seaborn as sns

import sys

sys.path.append("./")
sys.path.append("../")

from droputils.physics_utils import (
    add_mr,
    add_theta,
    add_theta_v,
    add_density,
    add_iwv,
    find_ml_height_from_gradient,
)
import droputils.rough_segments as segments  # noqa: E402
import droputils.circle_products as circle_products  # noqa: E402


def get_circle_data(ds, flight_id="20240811"):
    """
    get a dictionary of circle data for one flight
    """

    flight_date = date.fromisoformat(flight_id)
    
    circles = {
        circle: {
            "start_time": np.datetime64(
                datetime.combine(
                    flight_date, time.fromisoformat(segments.starts[flight_id][circle])
                )
            ),
            "end_time": np.datetime64(
                datetime.combine(
                    flight_date, time.fromisoformat(segments.ends[flight_id][circle])
                )
            ),
        }
        for circle in segments.starts[flight_id].keys()
    }
    ds_c = {}
    for circle in list(circles.keys()):
        try:
            ds_c[circle] = ds.where(
                ds["launch_time"].astype("datetime64")
                > circles[circle]["start_time"],
                drop=True,
            ).where(
                ds["launch_time"].astype("datetime64")
                < circles[circle]["end_time"],
                drop=True,
            )
        except ValueError:
            print(f"No sondes for circle {circle}. It is omitted")

    return ds_c


fs = 14

# Dropsonde data
level_3_path = "/Users/ninarobbins/Desktop/PhD/ORCESTRA/-DATA/Level_3/PERCUSION_Level_3.nc"
dropsonde_ds = (
    xr.open_dataset(level_3_path)
    .swap_dims({"sonde_id": "launch_time"})
)

# Add variables
dropsonde_ds = add_mr(dropsonde_ds)
dropsonde_ds = add_theta(dropsonde_ds)
dropsonde_ds = add_theta_v(dropsonde_ds)
dropsonde_ds = add_density(dropsonde_ds)

launch_time_strings = dropsonde_ds.launch_time.values
launch_time_datetimes = np.array([np.datetime64(date) for date in launch_time_strings])
dropsonde_ds = dropsonde_ds.assign_coords(
    launch_time=("launch_time", launch_time_datetimes)
)

# Compute mixed layer height
low_lim = 100
high_lim = 5000
dropsonde_ds = find_ml_height_from_gradient(
    dropsonde_ds.sel(alt=slice(low_lim, high_lim)),
    var="theta_v",
    threshold=0.2,
    lower_lim_m=low_lim,
)

# Split data into south / middle / north circles

flight_ids = list(segments.starts.keys())

# gives reasonable results
all_data = []
for flight_id in flight_ids:
    print(flight_id)
    dict_ds_c = get_circle_data(dropsonde_ds, flight_id)
    c_names = list(dict_ds_c.keys())
    flight_c = []
    for c_name in c_names:
        circle = dict_ds_c[c_name]
        circle = circle.expand_dims({"position": [c_name]})
        circle = circle.expand_dims({"flight_id": [flight_id]})
        flight_c.append(circle.copy())
    try:
        all_data.append(xr.concat(flight_c, dim="position"))
    except ValueError:
        pass
dropsonde_ds = xr.concat(all_data, dim="flight_id")

# Plot
row, col = 3, 5

fig, axes = plt.subplots(row, col, sharey=True, figsize=(15, 15))

variables = ["ta", "theta", "theta_v", "q", "rh"]
variable_titles = [
    "T / $\degree$C",
    "$\\theta$ / K",
    "$\\theta_v$ / K",
    "q / kg kg${-1}$",
    "RH / %",
]

# Iterate over each circle type (south, center, north)
for c_type, j in zip(["north", "center", "south"], range(row)):

    # Select data for the circle category across all flights
    ds_type = dropsonde_ds.sel(position=c_type)

    # Classify into cold pool / environment / none
    cp_soundings = ds_type.where(ds_type.hmix_grad_theta_v < 400, drop=True)
    env_soundings = ds_type.where(ds_type.hmix_grad_theta_v > 500, drop=True)
    none_soundings = ds_type.where(
        ds_type.hmix_grad_theta_v > 400, drop=True
    ).where(ds_type.hmix_grad_theta_v < 500, drop=True)

    print(cp_soundings.q.values)


    axes[j,0].set_ylabel(r"Altitude / m", fontsize=fs - 2)
    # Adding bold text in the upper left corner of each subplot
    axes[j,0].text(0.07, 0.05, f"{c_type.capitalize()} circles", transform=axes[j,0].transAxes,
            fontsize=fs-2, fontweight='bold', va='bottom', ha='left')

    for i in range(col):
        # Cold Pool anomalies
        cp_mean = cp_soundings[variables[i]].mean(dim="launch_time", skipna=True)[0]

        cp_min = cp_soundings[variables[i]].min(dim="launch_time", skipna=True)[0]
        cp_max = cp_soundings[variables[i]].max(dim="launch_time", skipna=True)[0]
        
        axes[j,i].plot(cp_mean, cp_soundings.alt, c="dodgerblue", label="cp")
        axes[j,i].fill_betweenx(cp_soundings.alt, cp_min, cp_max, color="dodgerblue", alpha=0.2)
        
        # Environment
        env_mean = env_soundings[variables[i]].mean(dim="launch_time", skipna=True)[0]
        env_min = env_soundings[variables[i]].min(dim="launch_time", skipna=True)[0]
        env_max = env_soundings[variables[i]].max(dim="launch_time", skipna=True)[0]

        axes[j,i].plot(env_mean, env_soundings.alt, c="red", label="env")
        axes[j,i].fill_betweenx(env_soundings.alt, env_min, env_max, color="red", alpha=0.2)

        # Undefined
        none_mean = none_soundings[variables[i]].mean(dim="launch_time", skipna=True)[0]
        none_min = none_soundings[variables[i]].min(dim="launch_time", skipna=True)[0]
        none_max = none_soundings[variables[i]].max(dim="launch_time", skipna=True)[0]

        axes[j,i].plot(none_mean, none_soundings.alt, c="grey", label="und")
        axes[j,i].fill_betweenx(none_soundings.alt, none_min, none_max, color="grey", alpha=0.2)


        axes[-1,i].set_xlabel(variable_titles[i], fontsize=fs)
        axes[j,i].tick_params(axis="both", labelsize=fs - 2)
        axes[j,i].spines["right"].set_visible(False)
        axes[j,i].spines["top"].set_visible(False)
        axes[j,i].set_ylim(0,6000)

axes[0,0].legend(fontsize=fs, bbox_to_anchor=(0.47, 0.99))

# Save
plt.savefig(
    "/Users/ninarobbins/Desktop/PhD/ORCESTRA/Figures/dropsondes/cold_pool_profiles_circles_dropsondes_PERCUSION_HALO.png",
    bbox_inches="tight",
)

plt.show()
