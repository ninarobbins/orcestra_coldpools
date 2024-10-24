import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.append("./")
sys.path.append("../")
from droputils.physics_utils import (
    add_mr,
    add_theta,
    add_theta_v,
    add_density,
    find_ml_height_from_gradient,
    add_cloud_flags
)

fs = 14

# Dropsonde data
level_3_path = "/Volumes/Upload/HALO/Dropsonde/dropsondes/Level_3/PERCUSION_Level_3.nc"
dropsonde_ds = (
    xr.open_dataset(level_3_path)
    .rename({"launch_time_(UTC)": "launch_time"})
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
dropsonde_ds = add_cloud_flags(dropsonde_ds)

# Compute mixed layer height
low_lim = 100
high_lim = 5000
dropsonde_ds = find_ml_height_from_gradient(
    dropsonde_ds.sel(alt=slice(low_lim, high_lim)),
    var="theta_v",
    threshold=0.2,
    lower_lim_m=low_lim,
)

# Classify into cold pool / environment / none
cp_soundings = dropsonde_ds.where(dropsonde_ds.hmix_grad_theta_v < 400, drop=True)
env_soundings = dropsonde_ds.where(dropsonde_ds.hmix_grad_theta_v > 500, drop=True)
none_soundings = dropsonde_ds.where(
    dropsonde_ds.hmix_grad_theta_v > 400, drop=True
).where(dropsonde_ds.hmix_grad_theta_v < 500, drop=True)


grouped_data = ds.groupby("cloud_flag")

fig, ax = plt.subplots(1,1,figsize=(10,6))
colors = ["purple", "turquoise", "orange"]
cloud_type = ["no cloud", "low cloud", "high cloud"]
i = 0

for name, group in grouped_data:

    if var == "theta_v":
        sample = group.hmix_grad_theta_v.where(~np.isnan(group.hmix_grad_theta_v), other=group.hmix_grad_theta_v.mean(skipna=True)).values
        #plt.title(r"from ${\theta_v}$", fontsize=fs+2)

    if var == "q":
        sample = group.hmix_grad_q.where(~np.isnan(group.hmix_grad_q), other=group.hmix_grad_q.mean(skipna=True)).values

        #plt.title(r"from $q$", fontsize=fs+2)

    model = KernelDensity(bandwidth=30, kernel='gaussian')
    sample = sample.reshape((len(sample), 1))
    model.fit(sample)

    # sample probabilities for a range of outcomes
    values = np.asarray([value for value in range(1, 1300)])
    values = values.reshape((len(values), 1))
    probabilities = model.score_samples(values)
    probabilities = np.exp(probabilities)

    # plot
    ax.plot(values[:], probabilities, label=cloud_type[i], c=colors[i])

    i += 1

# Customize the plot
ax.axvline(x=400, linestyle=":", c="k")#, label="cold pool")
ax.axvline(x=600, linestyle="--", c="k")#, label="environment")

ax.set_xlabel(r'$H_{mix}^{\theta_v}$ [m]', fontsize=fs)
ax.set_ylabel('Probability Density', fontsize=fs)
ax.set_ylim(0,0.0038)
ax.legend(fontsize=fs)
ax.tick_params(axis="both", labelsize=fs - 2)


ax.annotate("Cold Pool", xy=(0.15, 0.5), xycoords="axes fraction", fontsize=fs, ha="center", va="center")
ax.annotate("Environment", xy=(0.82, 0.5), xycoords="axes fraction", fontsize=fs, ha="center")

plt.show()