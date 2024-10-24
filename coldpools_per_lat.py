import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


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

# Add variables
dropsonde_ds = add_mr(dropsonde_ds)
dropsonde_ds = add_theta(dropsonde_ds)
dropsonde_ds = add_theta_v(dropsonde_ds)
dropsonde_ds = add_density(dropsonde_ds)

# Adjust IWV
dropsonde_ds["iwv"] = dropsonde_ds.iwv * (-1)

# Compute mixed layer height
low_lim = 100
high_lim = 5000
dropsonde_ds = find_ml_height_from_gradient(
    dropsonde_ds.sel(alt=slice(low_lim, high_lim)),
    var="theta_v",
    threshold=0.2,
    lower_lim_m=low_lim,
)

# Make new CP flag variable
cp_flag = np.full(len(dropsonde_ds.launch_time), 0)

for i in range(len(dropsonde_ds.launch_time)):
    if dropsonde_ds.hmix_grad_theta_v.isel(launch_time=i) < 400:
        cp_flag[i] = 1

# alt = dropsonde_ds.alt.values
# lats = dropsonde_ds.lat.values

# # Mask NaNs
# mask = ~np.isnan(lats)

# # Create interpolator
# f = interp1d(alt[mask], lats[mask], bounds_error=False, fill_value="extrapolate")

# # Apply interpolator to the full range of altitudes
# lats_interpolated = f(alt)

# lats = dropsonde_ds.lat.interpolate_na(dim="alt", fill_value="extrapolate").fillna(
#     method="pad"
# )
# lats = dropsonde_ds.lat.isel(alt=200)
# # Make PDF of number of cold pools per latitude
# plt.hist2d(lats, cp_flag, bins=5, cmap="Blues")

iwv = dropsonde_ds.iwv.values

iwv = np.array([np.nan if x is None else x for x in iwv])
cp_flag = np.array([np.nan if x is None else x for x in cp_flag])

# Create a mask to filter out rows where either iwv or cp_flag is NaN
valid_mask = ~np.isnan(iwv) & ~np.isnan(cp_flag)

# Apply the mask to both iwv and cp_flag to remove invalid entries
iwv_cleaned = iwv[valid_mask]
cp_flag_cleaned = cp_flag[valid_mask]

# Assuming iwv_cleaned and cp_flag_cleaned are the cleaned arrays from previous steps
# Define the bins for IWV (adjust the bin edges as needed)
iwv_bins = np.arange(45, 75, 2)  # Create bins with intervals of 5 from 45 to 70

# Compute histogram of IWV counts for cold pool flags == 1
cold_pool_hist, _ = np.histogram(iwv_cleaned[cp_flag_cleaned == 1], bins=iwv_bins)

# Compute total number of observations in each IWV bin (for all values)
total_hist, _ = np.histogram(iwv_cleaned, bins=iwv_bins)

# Avoid division by zero (if any bin has no total counts)
with np.errstate(divide='ignore', invalid='ignore'):
    pdf = cold_pool_hist / total_hist  # This will give you the fraction of cold pools per IWV bin

# Plot the PDF
bin_centers = (iwv_bins[:-1] + iwv_bins[1:]) / 2  # Calculate bin centers for plotting
plt.figure(figsize=(8, 6))
plt.bar(bin_centers, pdf, width=iwv_bins[1] - iwv_bins[0], align='center', alpha=0.7, color='dodgerblue', edgecolor="k")
plt.xlabel(r"IWV / kg m$^{-2}$", fontsize=14)
plt.ylabel("Probability of Cold Pools", fontsize=14)
plt.title("PDF of Cold Pools per IWV Bin", fontsize=16)

plt.show()