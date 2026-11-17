import numpy as np
import xarray as xr

import grid_toolbox.spherical_derivatives_healpix as hp_sphere
import grid_toolbox.vertical_interpolation as vinterp
import grid_toolbox.vertical_integration as vintegr

from atm_physics.physics import calc_evap_from_lhf

REQUIRED_VARS = [
    'pr', 'hfls', 'ts', 'hus', 'clw', 'cli', 'qg', 'qr', 'qs', 'ua', 'va', 'rho'
    ]

TARGET_PCOORDS = np.array([
    100000, 97500, 95000, 92500, 90000, 87500, 85000, 80000, 75000, 70000,
     60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000,
     ])


def calc_moisture_budget(data, vcoord_system, t_granularity):
    # check availability of all required variables
    for var in REQUIRED_VARS:
        if var not in data:
            raise ValueError(f"Missing required variable: {var}.")

    # Calculate evaporation and split into 2d and 3d fields
    data['evap'] = calc_evap_from_lhf(data['hfls'], data['ts'])
    data_2d = data[['pr', 'evap']]
    data_3d = data.drop_vars(
        [v for v in data.data_vars if 'level_full' not in data[v].dims]
        )

    # Calculate total specific humidity and its time derivative
    data_3d['q'] = _calc_total_q(data_3d)
    if vcoord_system == 'z':
        data_3d['q'] = data_3d['q'] * data_3d['rho']
    data_3d['dq_dt'] = calc_dq_dt(data_3d, t_granularity)

    # Interpolate to target vertical coordinate system
    if vcoord_system == 'p':
        data_3d = vinterp.interp_vertical_p(data_3d, TARGET_PCOORDS)
    elif vcoord_system == 'z':
        target_z = vinterp.get_zg_over_ocean(data['zg'])
        data_3d = vinterp.interp_vertical_z(data_3d, target_z)
    else:
        raise ValueError(
            f"Unsupported vertical coordinate system: {vcoord_system}.",
            )

    # Calculate horizontal moisture convergence
    data_3d['uv_conv'] = hp_sphere.compute_hor_wind_conv_on_hp(
        data_3d['ua'], data_3d['va'],
        )
    data_3d['q_conv'] = data_3d['q'] * data_3d['uv_conv']

    # Calculate horizontal moisture advection
    q_grad = hp_sphere.compute_gradient_on_hp(data_3d['q'])
    q_adv = (q_grad[0] * data_3d['ua'], q_grad[1] * data_3d['va'])
    data_3d['q_grad'] = _calc_abs_hor_vector(q_grad[0], q_grad[1])
    data_3d['q_adv'] = _calc_abs_hor_vector(q_adv[0], q_adv[1])

    # Integrate vertically
    data_3d = data_3d[['q', 'dq_dt', 'uv_conv', 'q_conv', 'q_grad', 'q_adv']]
    if vcoord_system == 'p':
        data_vint = vintegr.intv_p(data_3d, vcoord_system)
        #data_3d = calc_layer_contribution(data_3d, 'p')
    elif vcoord_system == 'z':
        data_vint = vintegr.intv_z(data_3d, 1, vcoord_system)
        #data_3d = calc_layer_contribution(data_3d, 'z')
        # Input "rho" is set to 1 because it is already included in the "q"
        # variable for z-coordinate system.

    # Transform units to mm/day, build column moisture budget and calculate residual
    data_2d = xr.merge([data_2d, data_vint])
    for var in ['pr', 'evap', 'q_conv', 'q_adv']:
        data_2d[var] = data_2d[var] * 86400
    data_2d['q_C'] = data_2d['pr'] - data_2d['evap'] - data_2d['dq_dt']
    data_2d['residual'] = data_2d['q_C'] - data_2d['q_conv'] - data_2d['q_adv']
    data_2d = _add_attrs(data_2d)

    
    

    return (data_2d.astype(np.float32), data_3d.astype(np.float32))


# Auxiliary functions
def _calc_abs_hor_vector(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    return np.sqrt(u**2 + v**2)


def _calc_total_q(data: xr.Dataset) -> xr.DataArray:
    return data['hus'] + data['clw'] + data['cli'] + data['qg'] + \
        data['qr'] + data['qs']


def calc_dq_dt(data: xr.Dataset, t_granularity: str) -> xr.DataArray:
    if t_granularity == 'P1D':
        dt_unit = 'D'
    else:
        raise ValueError(f"Unsupported time granularity: {t_granularity}.")
    
    return data['q'].chunk(time=-1).differentiate('time', datetime_unit=dt_unit)


def calc_layer_contribution(
        da: xr.Dataset | xr.DataArray, dim: str,
        ) -> xr.Dataset | xr.DataArray:
    f_lower = da.isel({dim: slice(None, -1)})
    f_upper = da.isel({dim: slice(1, None)})
    
    dim_lower = f_lower[dim].values
    dim_upper = f_upper[dim].values
    
    dx = da[dim].diff(dim)
    out = 0.5 * (f_lower.assign_coords({dim: f_upper[dim]}) + f_upper) * dx
    out[dim] = 0.5 * (dim_upper + dim_lower)
    out = out.assign_coords(
        p_lower=(dim, dim_lower),
        p_upper=(dim, dim_upper),
    )
    return out


def _add_attrs(data_2d: xr.Dataset) -> xr.Dataset:
    data_2d['pr'].attrs['units'] = 'mm day-1'
    data_2d['pr'].attrs['param'] = None

    data_2d['evap'].attrs['units'] = 'mm day-1'
    data_2d['evap'].attrs['standard_name'] = 'evap'
    data_2d['evap'].attrs['long_name'] = 'evaporation flux'
    data_2d['evap'].attrs['param'] = None

    data_2d['dq_dt'].attrs['units'] = 'mm day-1'
    data_2d['dq_dt'].attrs['standard_name'] = 'dq_dt'
    data_2d['dq_dt'].attrs['long_name'] = 'column-integrated moisture tendency'
    data_2d['dq_dt'].attrs['param'] = None

    data_2d['q_C'].attrs['units'] = 'mm day-1'
    data_2d['q_C'].attrs['standard_name'] = 'q_C'
    data_2d['q_C'].attrs['long_name'] = 'total column-integrated moisture convergence'
    data_2d['q_C'].attrs['param'] = None

    data_2d['residual'].attrs['units'] = 'mm day-1'
    data_2d['residual'].attrs['standard_name'] = 'residual'
    data_2d['residual'].attrs['long_name'] = 'column-integrated moisture residual'
    data_2d['residual'].attrs['param'] = None

    data_2d['q_conv'].attrs['units'] = 'mm day-1'
    data_2d['q_conv'].attrs['standard_name'] = 'q_conv'
    data_2d['q_conv'].attrs['long_name'] = 'column-integrated moisture convergence'
    data_2d['q_conv'].attrs['param'] = None

    data_2d['q_adv'].attrs['units'] = 'mm day-1'
    data_2d['q_adv'].attrs['standard_name'] = 'q_adv'
    data_2d['q_adv'].attrs['long_name'] = 'column-integrated moisture advection'
    data_2d['q_adv'].attrs['param'] = None

    data_2d['q'].attrs['units'] = 'kg m-2'
    data_2d['q'].attrs['standard_name'] = 'q'
    data_2d['q'].attrs['long_name'] = 'column-integrated specific humidity'
    data_2d['q'].attrs['param'] = None

    data_2d['uv_conv'].attrs['units'] = 's-1'
    data_2d['uv_conv'].attrs['standard_name'] = 'uv_conv'
    data_2d['uv_conv'].attrs['long_name'] = 'column-integrated horizontal wind convergence'
    data_2d['uv_conv'].attrs['param'] = None

    data_2d['q_grad'].attrs['units'] = 'kg m-3'
    data_2d['q_grad'].attrs['standard_name'] = 'q_grad'
    data_2d['q_grad'].attrs['long_name'] = 'column-integrated horizontal specific humidity gradient'
    data_2d['q_grad'].attrs['param'] = None

    return data_2d