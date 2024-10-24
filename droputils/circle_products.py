# %%

"""
This is NOT a file that will stay. it is just to get the circle products
that are in level 4 of pydropsonde
"""
# %%
import numpy as np
import xarray as xr
import circle_fit as cf
import metpy.calc as mpcalc
from metpy.units import units
import metpy.constants as mpconst
from tqdm import tqdm


def get_div_and_vor(circle):
    D = circle.dudx + circle.dvdy
    vor = circle.dvdx - circle.dudy

    circle = circle.assign(
        dict(div=(["gpsalt"], D.values), vor=(["gpsalt"], vor.values))
    )
    return circle


def get_density(circle, sonde_dim="sonde_id"):
    mr = mpcalc.mixing_ratio_from_specific_humidity(
        circle.q.values,
    )
    density = mpcalc.density(
        circle.p.values * units.Pa, circle.ta.values * units.kelvin, mr
    )
    circle = circle.assign(dict(density=(circle.ta.dims, density.magnitude)))
    circle["density"].attrs = {
        "standard_name": "density",
        "units": str(density.units),
    }
    circle = circle.assign(dict(mean_density=circle["density"].mean(sonde_dim)))
    circle["mean_density"].attrs = {
        "standard_name": "mean density",
        "units": str(density.units),
    }
    return circle


def get_vertical_velocity(circle, sonde_dim="sonde_id", alt_dim="gpsalt"):
    div = circle.div.where(~np.isnan(circle.div), drop=True).sortby(alt_dim)
    zero_vel = xr.DataArray(data=[0], dims=alt_dim, coords={alt_dim: [0]})

    height = xr.concat([zero_vel, div[alt_dim]], dim=alt_dim)
    height_diff = height.diff(dim=alt_dim)

    del_w = -div * height_diff.values

    w_vel = del_w.cumsum(dim=alt_dim)
    circle = circle.assign(dict(w_vel=w_vel))
    circle["w_vel"].attrs = {
        "standard name": "vertical velocity",
        "units": str(units.meter / units.second),
    }
    return circle


def get_omega(circle, sonde_dim="sonde_id", alt_dim="gpsalt"):
    p_vel = (
        -circle.mean_density.values
        * units(circle.mean_density.attrs["units"])
        * circle.w_vel.values
        * units(circle.w_vel.attrs["units"])
        * mpconst.earth_gravity
    )
    circle = circle.assign(dict(omega=(circle.w_vel.dims, p_vel.magnitude)))
    return circle


def get_vertical_velocity_geet(circle, alt_dim="gpsalt"):
    circle = circle.sortby(alt_dim)

    w_vel = np.empty(len(circle.gpsalt))
    w_vel[:] = np.nan
    w_vel[0] = 0
    # last = 0

    last = 0
    differences = np.empty(len(circle.gpsalt))
    differences[:] = np.nan

    for m in range(1, len(circle.gpsalt)):
        if not np.isnan(circle.div.isel(gpsalt=m)):
            w_vel[m] = w_vel[last] - circle.div.isel(gpsalt=m).values * 10 * (m - last)
            last = m
            differences[m] = circle.div.isel(gpsalt=m).values

    circle = circle.assign(dict(w_vel=(["gpsalt"], w_vel)))

    return circle


def get_omega_geet(circle):
    p_vel = np.empty(len(circle.gpsalt))
    w_vel = circle.w_vel.values
    p_vel[:] = np.nan

    for n in range(1, len(circle.gpsalt)):
        p_vel[n] = (
            -circle.mean_density.isel(gpsalt=n) * 9.81 * w_vel[n]
            # * 60
            # * 60
            # / 100
        )
    circle = circle.assign(dict(omega=(["gpsalt"], p_vel)))
    return circle


def fit2d(x, y, u):
    """
    estimate a 2D linear model to calculate u-values from x-y coordinates

    :param x: x coordinates of data points. shape: (...,M)
    :param y: y coordinates of data points. shape: (...,M)
    :param u: data values. shape: (...,M)

    all points along the M dimension are expected to belong to the same model
    all other dimensions are for different models

    :returns: intercept, dudx, dudy. all shapes: (...)
    """
    # to fix nans, do a copy
    u_cal = np.array(u, copy=True)
    # a does not need to be copied as this creates a copy already
    a = np.stack([np.ones_like(x), x, y], axis=-1)

    # for handling missing values, both u and a are set to 0, that way
    # these items don't influence the fit
    invalid = np.isnan(u_cal) | np.isnan(x) | np.isnan(y)
    under_constraint = np.sum(~invalid, axis=-1) < 6
    u_cal[invalid] = 0
    a[invalid] = 0

    a_inv = np.linalg.pinv(a)

    intercept, dudx, dudy = np.einsum("...rm,...m->r...", a_inv, u_cal)

    intercept[under_constraint] = np.nan
    dudx[under_constraint] = np.nan
    dudy[under_constraint] = np.nan

    return intercept, dudx, dudy


def fit2d_xr(x, y, u, sonde_dim="sonde_id"):
    # input and output dims must be a list
    return xr.apply_ufunc(
        fit2d,
        x,
        y,
        u,
        input_core_dims=[[sonde_dim], [sonde_dim], [sonde_dim]],
        output_core_dims=[(), (), ()],
    )


def apply_fit2d(circle):  # can be all circles concatenated together
    for par in tqdm(["u", "v", "q", "ta", "p"]):
        varnames = [par + "0", "d" + par + "dx", "d" + par + "dy"]

        circle = circle.assign(
            dict(
                zip(
                    varnames,
                    fit2d_xr(
                        x=circle.x,
                        y=circle.y,
                        u=circle[par],
                        sonde_dim="sonde_id",
                    ),
                )
            )
        )
    return circle


def get_xy_coords_for_circles(circle):
    x_coor = circle["lon"] * 111.320 * np.cos(np.radians(circle["lat"])) * 1000
    y_coor = circle["lat"] * 110.54 * 1000
    # converting from lat, lon to coordinates in metre from (0,0).

    c_xc = np.full(np.size(x_coor, 1), np.nan)
    c_yc = np.full(np.size(x_coor, 1), np.nan)
    c_r = np.full(np.size(x_coor, 1), np.nan)

    for j in range(np.size(x_coor, 1)):
        a = ~np.isnan(x_coor.values[:, j])
        if a.sum() > 4:
            c_xc[j], c_yc[j], c_r[j], _ = cf.least_squares_circle(
                [
                    (k, h)
                    for k, h in zip(x_coor.values[:, j], y_coor.values[:, j])
                    if ~np.isnan(k)
                ]
            )

    circle_y = np.nanmean(c_yc) / (110.54 * 1000)
    circle_x = np.nanmean(c_xc) / (111.320 * np.cos(np.radians(circle_y)) * 1000)

    circle_diameter = np.nanmean(c_r) * 2

    xc = [None] * len(x_coor.T)
    yc = [None] * len(y_coor.T)

    xc = np.mean(x_coor, axis=0)
    yc = np.mean(y_coor, axis=0)

    x = x_coor - xc  # *111*1000 # difference of sonde long from mean long
    y = y_coor - yc  # *111*1000 # difference of sonde lat from mean lat

    new_vars = dict(
        flight_altitude=circle["aircraft_geopotential_altitude_(m)"].mean().values,
        circle_time=circle["launch_time_(UTC)"].astype("datetime64").mean().values,
        circle_lon=circle_x,
        circle_lat=circle_y,
        circle_diameter=circle_diameter,
        x=(["sonde_id", "gpsalt"], x.values),
        y=(["sonde_id", "gpsalt"], y.values),
    )
    circle = circle.assign(new_vars)
    return circle
