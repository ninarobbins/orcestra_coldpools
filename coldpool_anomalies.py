import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

sys.path.append("./")
sys.path.append("../")
from droputils.physics_utils import (
    add_mr,
    add_theta,
    add_theta_v,
    add_density,
    find_ml_height_from_gradient,
)

fs = 14


# Dropsonde data
level_3_path = "/Users/ninarobbins/Desktop/PhD/Dropsondes/data/Level_3/Level_3.nc"
dropsonde_ds = (
    xr.open_dataset(level_3_path)
    .rename({"launch_time_(UTC)": "launch_time"})
    .rename({"gpsalt": "alt"})
    .swap_dims({"sonde_id": "launch_time"})
)

launch_time_strings = dropsonde_ds.coords["launch_time"].values
launch_time_datetimes = np.array([np.datetime64(date) for date in launch_time_strings])
dropsonde_ds = dropsonde_ds.assign_coords(
    launch_time=("launch_time", launch_time_datetimes)
)

# Add variables
dropsonde_ds = add_mr(dropsonde_ds)
dropsonde_ds = add_theta(dropsonde_ds)
dropsonde_ds = add_theta_v(dropsonde_ds)
dropsonde_ds = add_density(dropsonde_ds)

dropsonde_ds = dropsonde_ds.sel(alt=slice(0,6000))

# Compute mixed layer height
low_lim = 100
dropsonde_ds = find_ml_height_from_gradient(
    dropsonde_ds.sel(alt=slice(low_lim, None)),
    var="theta_v",
    threshold=0.2,
    lower_lim_m=low_lim,
)

# Variables for anomalies
variables = ["ta", "theta", "theta_v", "q", "rh"]
variable_titles = [
    "T' / $\degree$C",
    "$\\theta'$ / K",
    "$\\theta_v'$ / K",
    "q' / kg kg${-1}$",
    "RH' / %",
]

anomalies = defaultdict(dict)

# Split data into south / middle / north circles

flight_ids = list(segments.starts.keys())

all_data = []
count = 0
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
        
        # Separate into cold pool / env / undefined
        cp = circle.where(circle.hmix_grad_theta_v < 400, drop=True)
        env = circle.where(circle.hmix_grad_theta_v > 500, drop=True)
        none = circle.where(
            circle.hmix_grad_theta_v > 400, drop=True
        ).where(circle.hmix_grad_theta_v < 500, drop=True)

        count += 1

        # Compute anomalies
        for var in variables:
            if len(cp.launch_time) is not 0:
                anomaly_var = env[var].mean(dim="launch_time", skipna=True) - cp[var].mean(dim="launch_time", skipna=True)
            else:
                anomaly_var = np.nan
            
            anomalies[c_name][var] = 
    try:
        all_data.append(xr.concat(flight_c, dim="position"))
    except ValueError:
        pass
ds = xr.concat(all_data, dim="flight_id")
print(anomalies)

########################## Anomaly profiles ##########################
# Anomalies are computed from the difference in cold pool profiles 
# and environmental profiles within one circle
row, col = 3, 5

fig, axes = plt.subplots(row, col, sharey=True, figsize=(15, 15))

# Iterate over each circle type (south, center, north)
for c_type, j in zip(["north", "center", "south"], range(row)):
    # Select data for the circle category across all flights
    ds_type = ds.sel(position=c_type)
    # Classify into cold pool / environment / none
    cp_soundings = ds_type.where(ds_type.hmix_grad_theta_v < 400, drop=True)
    env_soundings = ds_type.where(ds_type.hmix_grad_theta_v > 500, drop=True)
    none_soundings = ds_type.where(
        ds_type.hmix_grad_theta_v > 400, drop=True
    ).where(ds_type.hmix_grad_theta_v < 500, drop=True)

    # For each circle, calculate anomalies




#     axes[j,0].set_ylabel(r"Altitude / m", fontsize=fs - 2)
#     # Adding bold text in the upper left corner of each subplot
#     axes[j,0].text(0.07, 0.05, f"{c_type.capitalize()} circles", transform=axes[j,0].transAxes,
#             fontsize=fs-2, fontweight='bold', va='bottom', ha='left')

#     for i in range(col):
#         # Cold Pool anomalies
#         print(env_soundings[variables[i]])
#         anomaly = np.abs(env_soundings[variables[i]][0] - cp_soundings[variables[i]][0])
       
       
       
#        print((env_soundings[variables[i]] - cp_soundings[variables[i]]).values)
#         mean_anomaly = anomaly.mean(dim="launch_time", skipna=True)[0]
#         min_anomaly = anomaly.min(dim="launch_time", skipna=True)[0]
#         max_anomaly = anomaly.max(dim="launch_time", skipna=True)[0]
        
#         axes[j,i].plot(mean_anomaly, anomaly.alt, c="k", label="cp")
#         axes[j,i].fill_betweenx(anomaly.alt, min_anomaly, max_anomaly, color="k", alpha=0.2)
        
#         axes[-1,i].set_xlabel(variable_titles[i], fontsize=fs)
#         axes[j,i].tick_params(axis="both", labelsize=fs - 2)
#         axes[j,i].spines["right"].set_visible(False)
#         axes[j,i].spines["top"].set_visible(False)
#         axes[j,i].set_ylim(0,6000)

# axes[0,0].legend(fontsize=fs, bbox_to_anchor=(0.47, 0.99))

# # Save
# plt.savefig(
#     "/Users/ninarobbins/Desktop/PhD/ORCESTRA/Figures/dropsondes/cold_pool_profiles_circles_dropsondes_PERCUSION_HALO.png",
#     bbox_inches="tight",
# )

# plt.show()