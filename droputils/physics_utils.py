import numpy as np
import metpy.calc as mpcalc
from metpy.units import units


def get_rh_max_circle(ds, hmin=8000, alt_var="alt"):
    """
    Get maximum RH above a certain height
    """

    ds_cut = ds.where(ds[alt_var] >= hmin, drop=True).mean("sonde_id")
    max_rh_sonde = ds_cut["rh"].max(dim=alt_var).values
    rh_idx_sonde = np.abs(ds_cut["rh"] - max_rh_sonde).argmin(dim=alt_var)
    rh_h_sonde = ds_cut[alt_var].isel({alt_var: rh_idx_sonde})
    return dict(rh=max_rh_sonde, height=rh_h_sonde.values, idx=rh_idx_sonde.values)


def get_levels_circle(ds, variable="ta", value=273.15, alt_var="alt"):
    """
    Get values closest to a certain Temperature

    Return:
        value of that level
        index of that level
    """
    freezing_index = np.abs(ds[variable] - value).argmin(dim=alt_var)
    return (
        ds[alt_var].isel({alt_var: freezing_index}).values.mean(),
        ds[alt_var].isel({alt_var: freezing_index}).values,
    )


def get_heights_from_array(ds, values, alt_var="alt"):
    """
    get closest index to a list of values in the mean over sondes
    """
    indices = [
        np.abs(ds - val).mean("sonde_id").argmin(dim=alt_var).values for val in values
    ]
    return indices


def get_lcl_circle(ds, min_h=0, max_h=200, alt_var="alt"):
    """
    get lcl from a ds with temperature (ta), pressure (p) and relative humidity (rh)
    """

    base_values = ds.sel({alt_var: slice(min_h, max_h)}).mean(alt_var)

    temperature = base_values["ta"].values * units.kelvin
    pressure = base_values["p"].values * units.Pa
    rh = (base_values["rh"] * 100).values * units.percent
    mask_t = ~np.isnan(temperature)
    mask_p = ~np.isnan(pressure)
    mask = mask_t & mask_p
    dewpoint = mpcalc.dewpoint_from_relative_humidity(temperature[mask], rh[mask])
    print(pressure[mask])
    lcl_pressure, lcl_temperature = mpcalc.lcl(
        pressure[mask], temperature[mask], dewpoint
    )

    return lcl_pressure, lcl_temperature


def add_iwv(ds):
    """
    Compute IWV and add it to the dataset.
    """

    iwv = [None] * len(ds["sonde_id"])
        
    for i in range(len(ds["sonde_id"])):
    
        try:
            dp = mpcalc.dewpoint_from_relative_humidity(
                    ds.ta.isel(launch_time=i)* units.K, ds.rh.isel(launch_time=i).values
                ).interpolate_na(dim="alt", fill_value="extrapolate")
            pres = ds.p.isel(launch_time=i).interpolate_na(dim="alt", fill_value="extrapolate")
            iwv[i] = mpcalc.precipitable_water(
                pres.values * units.Pa,
                dp.values * units.degC,
            ).magnitude
        except (ValueError, IndexError):
            continue

    ds["iwv"] = (["launch_time"], iwv)


def add_mr(ds):
    """
    Function to estimate mixing ratio from specific humidity in the given dataset.
    """
    mr = mpcalc.mixing_ratio_from_specific_humidity(ds.q)

    ds = ds.assign(mr=(ds.ta.dims, mr.data))
    ds["mr"].attrs = dict(
        standard_name="mixing ratio",
        long_name="mixing ratio",
        units=r"kg kg$^{-1}$",
    )

    return ds


def add_theta(ds):
    """
    Function to estimate potential temperature from the temperature and pressure in the given dataset.
    """
    theta = mpcalc.potential_temperature(
        ds.p.values * units.Pa, ds.ta.values * units.kelvin
    )
    ds = ds.assign(theta=(ds.ta.dims, theta.magnitude))
    ds["theta"].attrs = dict(
        standard_name="potential temperature",
        long_name="potential temperature",
        units=str(theta.units),
    )

    return ds


def add_theta_v(ds):
    """
    Function to estimate virtual potential temperature in the given dataset.
    """
    mr = mpcalc.mixing_ratio_from_specific_humidity(ds.q)
    theta_v = mpcalc.virtual_potential_temperature(
        ds.p.values * units.Pa, ds.ta.values * units.kelvin, mr
    )

    ds = ds.assign(theta_v=(ds.ta.dims, theta_v.magnitude))
    ds["theta_v"].attrs = dict(
        standard_name="virtual potential temperature",
        long_name="virtual potential temperature",
        units=str(theta_v.units),
    )

    return ds


def add_density(ds):
    """
    Function to estimate density in the given dataset.
    """
    mr = mpcalc.mixing_ratio_from_specific_humidity(ds.q)

    density = (0.622 * ds.p * (1 + mr)) / (287.053 * ds.ta * (mr + 0.622))

    ds = ds.assign(density=(ds.ta.dims, density.data))
    ds["density"].attrs = dict(
        standard_name="density",
        long_name="density",
        units=r"kg m${-1}$",
    )

    return ds


# Calculate Hmix
def find_ml_height_from_gradient(
    ds,
    var,
    threshold,
    time="launch_time",
    altitude="alt",
    lower_lim_m=100.0,
):
    def _calculateHmix_var(density_profile, var_profile, threshold, lower_lim_m):
        # size = len(sonde.alt)
        var_diff = 0
        numer = 0
        denom = 0
        var_mix = 0

        k = int(lower_lim_m / 10)  # lower limit index (height = index * 10m)

        while abs(var_diff) < threshold:
            numer += 0.5 * (
                density_profile[k + 1].values * var_profile[k + 1].values
                + density_profile[k].values * var_profile[k].values
            )
            denom += 0.5 * (density_profile[k + 1].values + density_profile[k].values)
            var_mix = numer / denom
            k += 1
            var_diff = var_profile[k].values - var_mix

        hmix = var_profile[altitude].values[k]

        return hmix

    # call inner function
    hmix_vec = np.zeros(len(ds[time]))
    for i in range(len(ds[time])):
        density_profile = (
            ds["density"]
            .isel({time: i})
            .interpolate_na(dim=altitude)
            .rolling({altitude: 4}, center=True)
            .mean()
        )
        var_profile = (
            ds[var].isel({time: i}).interpolate_na(dim=altitude)
        )  # .rolling({altitude: 4}, center=True).mean()
        hmix_vec[i] = _calculateHmix_var(
            density_profile, var_profile, threshold, lower_lim_m
        )
    k = int(lower_lim_m / 10) + 2
    hmix_vec[hmix_vec == k * 10] = np.nan

    # assign variable to dataset
    ds = ds.assign({f"hmix_grad_{var}": ((time), hmix_vec)})
    ds = ds.assign_attrs({f"hmix_threshold_{var}": threshold})

    return ds


def add_cloud_flags(ds):
    
    min_rh_low = 0.92
    max_rh_low = 0.95
    max_rh_middle = 0.93

    min_moistlayer_thickness = 400
    min_moistlayer_base = 120

    min_cloudlayer_base = 280

    min_cloud_thickness_low = 30
    min_cloud_thickness_middle = 60
    
    cloud_flag_vec = np.full(len(ds.sonde_id), np.nan)
    
    for i, sonde in enumerate(ds.sonde_id):
        
        #print(sonde.values)
        
        rh = ds.rh[i]
        
        rh_ml = rh.where(rh > min_rh_low)
        
        # Check if all values are nan, then there is no moist layer
        if rh_ml.isnull().all().item():
            cloud_flag = 0
            cloud_flag_vec[i] = cloud_flag

            continue

        else:
            
            # Keep only moist layer for this analysis
            rh_ml = rh_ml.where(~np.isnan(rh_ml), drop=True)

            base_ml = rh_ml.alt[0]
            top_ml = rh_ml.alt[-1]
            thickness_ml = top_ml - base_ml

            min_rh_low = 0.92
            max_rh_low = 0.95
            max_rh_middle = 0.93

            min_moistlayer_thickness = 400
            min_moistlayer_base = 120

            min_cloudlayer_base = 280

            min_cloud_thickness_low = 30
            min_cloud_thickness_middle = 60

            if (thickness_ml > min_moistlayer_thickness) and (base_ml > min_moistlayer_base):

                # Check if max RH threshold for moist layer is exceeded --> cloud
                max_rh_ml = rh_ml.where(rh_ml == rh_ml.max(), drop=True)
                
                # Choose altitude range and max RH based on height of moist layer base
                if base_ml < 1300:
                    max_rh = max_rh_low
                    min_cloud_thickness = min_cloud_thickness_low
                    flag = 1

                if base_ml > 1300:
                    max_rh = max_rh_middle
                    min_cloud_thickness = min_cloud_thickness_middle
                    flag = 2

                # Check if the layer is cloudy
                if max_rh_ml > max_rh:
                    rh_cl = rh_ml.where(rh_ml > max_rh, drop=True)

                    # Check if cloud is thicker than threshold in total (includes separate cloud layers)
                    if len(rh_cl) > (min_cloud_thickness/10):
                        
                        # Check if cloud top is lower than threshold
                        if top_ml > min_cloudlayer_base:
                            cloud_flag = flag
                        else:
                            cloud_flag = 0
                    else:
                        cloud_flag = 0
                else:
                    cloud_flag = 0
            else: 
                cloud_flag = 0
            
        # Add cloud flag to array
        cloud_flag_vec[i] = cloud_flag
        
    # Add cloud flags to dataset
    ds = ds.assign({"cloud_flag" : (("sonde_id"), cloud_flag_vec)})
    ds["cloud_flag"] = ds.cloud_flag.assign_attrs(standard_name="cloud_flag", 
                                                  flag_meanings=["no_cloud", "low_cloud", "high_cloud"],
                                                  flag_values=[0,1,2])
    return(ds)