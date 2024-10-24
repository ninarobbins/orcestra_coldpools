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
print(flight_ids)

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
ds = xr.concat(all_data, dim="flight_id")

# %%
# Check distribution to find cold pool Hmix threshold
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
bins = range(0, 1200, 10) # altitude bins
sns.set_palette("turbo", n_colors=7)

ds.sel(position="south")
for c_type, ax in zip(["north", "center", "south"], axes):
    ds_type = ds.sel(position=c_type)
    for flight_id in ds_type.flight_id.values:
        ds_type.sel(flight_id=flight_id).hmix_grad_theta_v.plot.hist(ax=ax, bins=bins, edgecolor='k', density=True, label=flight_id)
    ax.tick_params(axis="both", labelsize=fs - 2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Occurrence", fontsize=fs)
    ax.set_xlabel("")
    ax.set_title("")
    ax.text(0.01, 0.95, f"{c_type.capitalize()} circles", transform=ax.transAxes,
            fontsize=fs-2, fontweight='bold', va='top', ha='left')

    
axes[-1].set_xlabel(r"$H_{mix}$ from $\theta_v$ / m", fontsize=fs)
axes[0].legend()

plt.show()



# Without distinguishing by flight

# Check distribution to find cold pool Hmix threshold (for all data in each circle category)
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
sns.set_palette("turbo", n_colors=7)

# Iterate over each circle type (south, center, north)
for c_type, ax in zip(["north", "center", "south"], axes):
    # Select data for the circle category across all flights
    ds_type = ds.sel(position=c_type)
    
    # Aggregate data across all flight_ids (ignore flight_id, sum the data together)
    hmix_data = ds_type.hmix_grad_theta_v.values.flatten()  # Flatten the data to 1D array
    
    # Plot the histogram for the aggregated data
    ax.hist(hmix_data, bins=bins, edgecolor='k', density=True)
    
    # Styling and labels
    ax.tick_params(axis="both", labelsize=fs - 2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Occurrence", fontsize=fs)
    ax.set_xlabel("")
    ax.set_title("")
    
    # Adding bold text in the upper left corner of each subplot
    ax.text(0.01, 0.95, f"{c_type.capitalize()} circles", transform=ax.transAxes,
            fontsize=fs-2, fontweight='bold', va='top', ha='left')

# Final adjustment for the x-axis label on the last plot
axes[-1].set_xlabel(r"$H_{mix}$ from $\theta_v$ / m", fontsize=fs)

plt.show()



# For all data

# Check distribution to find cold pool Hmix threshold (for all data in each circle category)
fig, ax = plt.subplots(1, 1, figsize=(10, 3))

# Plot the histogram for the aggregated data
ax.hist(dropsonde_ds.hmix_grad_theta_v, bins=bins, color="grey", edgecolor='k', density=True)

# Styling and labels
ax.tick_params(axis="both", labelsize=fs - 2)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_ylabel("Occurrence", fontsize=fs)
ax.set_title("All data", fontsize=fs)
ax.set_xlabel(r"$H_{mix}$ from $\theta_v$ / m", fontsize=fs)

plt.show()
