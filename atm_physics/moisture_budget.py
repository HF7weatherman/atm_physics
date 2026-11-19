import numpy as np
import xarray as xr
import warnings

import grid_toolbox.spherical_derivatives_healpix as hp_sphere
import grid_toolbox.vertical_interpolation as vinterp

from atm_physics.physics import calc_evap_from_lhf, G


# ------------------------------------------------------------------------------
def calc_moisture_budget(data, vcoord_system, target_coords, t_granularity):
    warnings.filterwarnings(action="ignore")
    
    # Calculate evaporation and split into 2d and 3d fields
    data['evap'] = calc_evap_from_lhf(data['hfls'], data['ts'])
    data_2d = data[['pr', 'evap']] * 86400.    # mm day-1
    data_3d = data.drop_vars(
        [v for v in data.data_vars if 'level_full' not in data[v].dims]
        )

    # Interpolate to target vertical coordinate system
    if vcoord_system == 'p':
        data_3d['q'] = data_3d['q'] / G
        data_3d = vinterp.interp_vertical_p(data_3d, target_coords)
    elif vcoord_system == 'z':
        data_3d['q'] = data_3d['q'] * data_3d['rho']
        data_3d = vinterp.interp_vertical_z(data_3d, target_coords)
    else:
        raise ValueError(
            f"Unsupported vertical coordinate system: {vcoord_system}.",
            )

    # Calculate horizontal moisture convergence in mm day-1
    data_3d['uv_conv'] = hp_sphere.compute_hor_wind_conv_on_hp(
        data_3d['ua'], data_3d['va'],
        )
    data_3d['q_conv'] = data_3d['q'] * data_3d['uv_conv'] * 86400.

    # Calculate horizontal moisture advection in mm day-1
    data_3d['dq_dx'], data_3d['dq_dy'] = hp_sphere.compute_gradient_on_hp(
        data_3d['q'],
        )
    data_3d['q_adv'] = -86400. * \
        (data_3d['dq_dx'] * data_3d['ua'] + data_3d['dq_dy'] * data_3d['va'])

    data_3d = data_3d[
        ['ua', 'va', 'q', 'dq_dt', 'uv_conv', 'q_conv', 'dq_dx', 'dq_dy',
         'q_adv',
         ],
        ]

    # Integrate vertically
    data_3d = calc_layer_contribution(data_3d, vcoord_system)
    data_3d_vint = data_3d.sum(dim=vcoord_system, skipna=True)

    # Transform units to mm/day, build column moisture budget and calculate residual
    data_2d = xr.merge([data_2d, data_3d_vint])
    data_2d['q_C'] = data_2d['pr'] - data_2d['evap'] + data_2d['dq_dt']
    data_2d['residual'] = data_2d['q_C'] - data_2d['q_conv'] - data_2d['q_adv']
    
    return (
        _add_attrs(xr.merge([data_2d, data[['ts', 'ofs']]]), dim='2d').\
            astype(np.float32),
        _add_attrs(xr.merge([data_3d, data[['ts', 'ofs']]]), dim='3d').\
            astype(np.float32),
        )


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
    if dim == 'p':
        dx = -dx  # pressure decreases with height
    out = 0.5 * (f_lower.assign_coords({dim: f_upper[dim]}) + f_upper) * dx
    out[dim] = 0.5 * (dim_upper + dim_lower)
    out = out.assign_coords(
        p_lower=(dim, dim_lower),
        p_upper=(dim, dim_upper),
    )
    return out


def _add_attrs(data: xr.Dataset, dim: str='3d') -> xr.Dataset:
    if dim == '2d':
        add_str = 'column-integrated '
    elif dim == '3d':
        add_str = ''

    if 'pr' in data.data_vars:
        data['pr'].attrs['units'] = 'mm day-1'
        data['pr'].attrs['param'] = 'XXXXX'

    if 'evap' in data.data_vars:
        data['evap'].attrs['units'] = 'mm day-1'
        data['evap'].attrs['standard_name'] = 'evap'
        data['evap'].attrs['long_name'] = 'evaporation flux'
        data['evap'].attrs['param'] = 'XXXXX'

    if 'dq_dt' in data.data_vars:
        data['dq_dt'].attrs['units'] = 'mm day-1'
        data['dq_dt'].attrs['standard_name'] = 'dq_dt'
        data['dq_dt'].attrs['long_name'] = f'{add_str}moisture tendency'
        data['dq_dt'].attrs['param'] = 'XXXXX'

    if 'q_C' in data.data_vars:
        data['q_C'].attrs['units'] = 'mm day-1'
        data['q_C'].attrs['standard_name'] = 'q_C'
        data['q_C'].attrs['long_name'] = f'total {add_str}moisture convergence'
        data['q_C'].attrs['param'] = 'XXXXX'

    if 'residual' in data.data_vars:
        data['residual'].attrs['units'] = 'mm day-1'
        data['residual'].attrs['standard_name'] = 'residual'
        data['residual'].attrs['long_name'] = f'{add_str}moisture residual'
        data['residual'].attrs['param'] = 'XXXXX'

    if 'q_conv' in data.data_vars:
        data['q_conv'].attrs['units'] = 'mm day-1'
        data['q_conv'].attrs['standard_name'] = 'q_conv'
        data['q_conv'].attrs['long_name'] = f'{add_str}moisture convergence'
        data['q_conv'].attrs['param'] = 'XXXXX'

    if 'q_adv' in data.data_vars:
        data['q_adv'].attrs['units'] = 'mm day-1'
        data['q_adv'].attrs['standard_name'] = 'q_adv'
        data['q_adv'].attrs['long_name'] = f'{add_str}moisture advection'
        data['q_adv'].attrs['param'] = 'XXXXX'

    if 'q' in data.data_vars:
        data['q'].attrs['units'] = 'kg m-2'
        data['q'].attrs['standard_name'] = 'q'
        data['q'].attrs['long_name'] = f'{add_str} moisture'
        data['q'].attrs['param'] = 'XXXXX'

    if 'uv_conv' in data.data_vars:
        data['uv_conv'].attrs['units'] = 's-1'
        data['uv_conv'].attrs['standard_name'] = 'uv_conv'
        data['uv_conv'].attrs['long_name'] = f'{add_str}horizontal wind convergence'
        data['uv_conv'].attrs['param'] = 'XXXXX'

    if 'q_grad' in data.data_vars:
        data['q_grad'].attrs['units'] = 'kg m-3'
        data['q_grad'].attrs['standard_name'] = 'q_grad'
        data['q_grad'].attrs['long_name'] = f'{add_str}horizontal moisture gradient'
        data['q_grad'].attrs['param'] = 'XXXXX'

    if 'p' in data.coords:
        data['p'].attrs['units'] = 'Pa'
        data['p'].attrs['standard_name'] = 'air_pressure'
        data['p'].attrs['long_name'] = 'pressure'
        data['p'].attrs['param'] = 'XXXXX'

    if 'p_upper' in data.coords:
        data['p_upper'].attrs['units'] = 'Pa'
        data['p_upper'].attrs['standard_name'] = 'air_pressure'
        data['p_upper'].attrs['long_name'] = 'pressure'
        data['p_upper'].attrs['param'] = 'XXXXX'

    if 'p_lower' in data.coords:
        data['p_lower'].attrs['units'] = 'Pa'
        data['p_lower'].attrs['standard_name'] = 'air_pressure'
        data['p_lower'].attrs['long_name'] = 'pressure'
        data['p_lower'].attrs['param'] = 'XXXXX'

    return data
