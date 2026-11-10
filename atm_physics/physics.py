"""
Atmospheric Physics Functions

This module provides a comprehensive collection of thermodynamic and dynamic
functions for atmospheric science calculations. Functions are organized into
three main categories:

1. Moist Thermodynamics: Functions dealing with moisture, humidity, and
   saturation calculations.
2. Dry Thermodynamics: Functions related to temperature, pressure, and
   energy transformations in dry air.
3. Dynamics: Functions for atmospheric dynamics and stability calculations.

Constants:
    G: Acceleration due to gravity (m/s^2)
    R_dry: Specific gas constant for dry air (J/kg·K)
    LV: Latent heat of vaporization (J/kg)
    LS: Latent heat of sublimation (J/kg)
    LF: Latent heat of fusion (J/kg)
    CPD_BF: Specific heat at constant pressure for dry air at ~20°C (J/kg/K)
    CPD: Specific heat at constant pressure for dry air at ~5°C (J/kg/K)
    CL: Specific heat of liquid water (J/kg/K)
    BETA: Mixing ratio factor for virtual temperature correction
    R_D: Specific gas constant for dry air (J/kg/K)
"""

import numpy as np
import xarray as xr
import scipy.interpolate as ipl

# Physical Constants
G = 9.80665             # m/s^2, acceleration due to gravity
R_DRY = 287.085         # specific gas constant for dry air J/(kg·K)
LV = 2.5008e6           # latent heat vaporization J/kg
LS = 2.8346e6           # latent heat sublimation J/kg
LF = LS - LV            # latent heat fusion J/kg
CPD_BF = 1005.7         # sp. heat const. pressure (dry air) J/kg/K, at about 20°C
CPD = 1004.64           # sp. heat const. pressure (dry air) J/kg/K, at about 5°C
CL = 4184               # sp. heat constant (liquid) J/kg/K
BETA = 0.6078           # mixing ratio factor for virtual temperature
R_D = 287.085           # specific gas constant for dry air (J/kg/K)
eps = 0.622             # ratio of molecular weights (dry air / water vapor)

K2C = 273.15            # conversion from Kelvin to Celsius

# ==============================================================================
# MOIST THERMODYNAMICS
# ==============================================================================
def calc_q_sat(T, p, use_ice=True):
    """
    Saturation specific humidity q_s (kg/kg) from temperature (K) and pressure (Pa).

    Uses Bolton/Tetens formula over liquid water. If use_ice=True, uses the 
    Murphy & Koop (2005) fit over ice for T < 273.15 K.
    
    Parameters
    ----------
    T : float or array
        Temperature (K).
    p : float or array
        Pressure (Pa).
    use_ice : bool, optional
        If True, uses Murphy & Koop (2005) fit over ice for T < 273.15 K.
        Default is True.
    
    Returns
    -------
    xarray.DataArray
        Saturation specific humidity (kg/kg) aligned with input ds.
    
    Notes
    -----
    - Saturated vapor pressure over liquid: Bolton/Tetens formula
    - Saturated vapor pressure over ice: Murphy & Koop (2005) fit
    - Includes safeguards against unphysical values
    """

    # Saturation vapor pressure over liquid (Bolton/Tetens), result in Pa
    e_sw = magnus_formula_in_hPa(T, state='liquid') * 100

    if use_ice:
        e_si = magnus_formula_in_hPa(T, state='frozen') * 100
        e_s = xr.where(T >= K2C, e_sw, e_si)
    else:
        e_s = e_sw

    # Avoid unphysical/nearly singular situations where e_s approaches p/(1-eps)
    e_max = 0.999 * (p / (1.0 - eps))
    e_s_safe = xr.where(e_s >= e_max, e_max, e_s)

    q_sat = (eps * e_s_safe) / (p - (1.0 - eps) * e_s_safe)
    return q_sat


def magnus_formula_in_hPa(T, state='liquid'):
    """
    Saturation vapor pressure using Magnus formula.
    
    Calculates saturation water vapor partial pressure for a given temperature.
    Works for both liquid and solid (ice) states.
    
    Parameters
    ----------
    T : float or array
        Temperature (K).
    state : str, optional
        Aggregate state: 'liquid' or 'frozen'/'solid'. Default is 'liquid'.
    
    Returns
    -------
    float or array
        Saturation vapor pressure (hPa).
    
    Raises
    ------
    ValueError
        If state is not 'liquid' or 'frozen'/'solid'.
    """
    if state == 'liquid':
        # Bolton/Tetens formula for saturation over liquid water (T in °C),
        # e_s in hPa
        T_c = T - K2C
        e_s = 6.112 * np.exp((17.67 * T_c) / (T_c + 243.5))
    
    elif state == 'frozen':
        # Murphy & Koop (2005) fit for saturation over ice (T in K), e_s in hPa
        ln_e_si = 9.550426 - 5723.265/T + 3.53068*np.log(T) - 0.00728332*T
        e_s = np.exp(ln_e_si) / 100

    else:
        raise ValueError('No valid aggregate state!')

    return e_s

    
def int_q(q):
    """
    Vertically integrated specific humidity (column water vapor).
    
    Parameters
    ----------
    q : xarray.DataArray
        Specific humidity (kg/kg).

    Returns
    -------
    xarray.DataArray
        Vertically integrated specific humidity (kg/m^2).
    """
    return (q / G).integrate(coord='plev')


def calc_crh(ds):
    """
    Column Relative Humidity (CRH).
    
    Calculates the column-integrated relative humidity as the ratio of
    vertically integrated specific humidity to vertically integrated
    saturation specific humidity.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'hus' (specific humidity), 'ta' (temperature),
        and 'plev' (pressure levels).
    
    Returns
    -------
    xarray.DataArray
        Column relative humidity (dimensionless, 0-1).
    """
    q_sat = calc_q_sat(ds.T, ds.plev)
    qsat_int = int_q(q_sat)
    q_int = int_q(ds.hus)
    return (q_int / qsat_int)


def calc_T_v(T: xr.DataArray, q: xr.DataArray) -> xr.DataArray:
    """
    Virtual temperature.
    
    Calculates the virtual temperature using the relationship:
    T_v = T * (1 + BETA * q), where BETA ≈ 0.6078.
    
    Parameters
    ----------
    T : xarray.DataArray
        Temperature (K).
    hus : xarray.DataArray
        Specific humidity (kg/kg).
    
    Returns
    -------
    xarray.DataArray
        Virtual temperature (K).
    """
    return T * (1. + BETA * q)


def calc_density(
        T: xr.DataArray, q: xr.DataArray, p: xr.DataArray,
        ) -> xr.DataArray:
    """
    Air density accounting for moisture.
    
    Calculates air density using the ideal gas law with virtual temperature:
    ρ = p / (R_D * T_v)
    
    Parameters
    ----------
    T : xarray.DataArray
        Temperature (K).
    hus : xarray.DataArray
        Specific humidity (kg/kg).
    pres : xarray.DataArray
        Pressure (Pa).
    
    Returns
    -------
    xarray.DataArray
        Air density (kg/m^3).
    """
    T_v = calc_T_v(T, q)
    return p / (R_D * T_v)


def calc_rh(T, T_d):
    """
    Relative humidity from temperature and dew point.
    
    Calculates relative humidity as the ratio of saturation vapor pressure
    at dew point to saturation vapor pressure at actual temperature.
    
    Parameters
    ----------
    T : float or array
        Temperature (K).
    T_d : float or array
        Dew point temperature (K).
    
    Returns
    -------
    float or array
        Relative humidity (%).
    """
    return (magnus_formula_in_hPa(T_d) / magnus_formula_in_hPa(T)) * 100


def calc_q_from_Td(T_d: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    """
    Specific humidity at 2m from dew point temperature and surface pressure.
    
    Calculates specific humidity at 2m above the surface using dew point
    temperature and surface pressure.
    
    Parameters
    ----------
    T_d : xarray.DataArray or float
        Dew point temperature (K).
    p : xarray.DataArray or float
        Pressure (Pa).
    
    Returns
    -------
    xarray.DataArray or float
        Specific humidity at 2m (kg/kg).
    """
    e_s = magnus_formula_in_hPa(T_d) * 100 # in Pa
    q = eps * e_s / (p - 0.25 - 0.37 * e_s)
    return q 


def calc_clwvi(cllvi: xr.DataArray, clivi: xr.DataArray) -> xr.DataArray:
    """
    Total cloud liquid water vertical integral.
    
    Combines liquid and ice cloud water vertical integrals.
    
    Parameters
    ----------
    cllvi : xarray.DataArray
        Cloud liquid water vertical integral (kg/m^2).
    clivi : xarray.DataArray
        Cloud ice water vertical integral (kg/m^2).
    
    Returns
    -------
    xarray.DataArray
        Total cloud water vertical integral (kg/m^2).
    """
    return cllvi + clivi


def calc_mse_e(
        T: xr.DataArray,
        q: xr.DataArray,
        ql: xr.DataArray,
        qi: xr.DataArray,
        z: xr.DataArray,
        ) -> xr.DataArray:
    """
    Moist static energy (environmental).
    
    Calculates the moist static energy as the sum of thermal, latent, and
    potential energy components:
    MSE = cp*T + L_v*q - L_f*q_i + G*z
    
    Parameters
    ----------
    T : xarray.DataArray
        Temperature (K).
    q : xarray.DataArray
        Specific humidity (kg/kg).
    ql : xarray.DataArray
        Cloud liquid water (kg/kg).
    qi : xarray.DataArray
        Cloud ice (kg/kg).
    z : xarray.DataArray
        Geopotential height (m).

    Returns
    -------
    xarray.DataArray
        Moist static energy (J/kg).

    Notes
    -----
    MSE accounts for the energy in water vapor and ice phases, as well as
    the potential energy at a given height.
    """
    qt = q + ql + qi

    thermal_energy = (CPD + qt * (CL - CPD)) * T
    latent_energy = LV * q - LF * qi
    potential_energy = G * z

    return thermal_energy + latent_energy + potential_energy


def calc_mse_sfc(T: xr.Dataset, q: xr.Dataset) -> xr.Dataset:
    """
    Moist static energy at the surface (2m).
    
    Calculates the moist static energy at the surface using 2m temperature,
    2m specific humidity, and a reference height of 2m.
    
    Parameters
    ----------
    T : xarray.Dataset
        Surface temperature (K).
    q : xarray.Dataset
        Specific humidity at 2m (kg/kg).

    Returns
    -------
    xarray.Dataset
        Surface moist static energy (J/kg).
    """
    Z = 2.  # Reference height at 2m
    return (CPD_BF * T + LV * q + G * Z)


def calc_sfc_bf(
        p: xr.DataArray,
        shf: xr.DataArray,
        lhf: xr.DataArray,
        T: xr.DataArray,
        q: xr.DataArray,
        ) -> xr.DataArray:
    """
    Surface buoyancy flux.
    
    Calculates the surface buoyancy flux from sensible and latent heat fluxes.
    Based on de Szoeke formulation as presented in Chen et al. (2025).
    
    The buoyancy flux represents the kinetic energy flux at the surface:
    B_flux = (G / (rho * T_v)) * (SHF / (cp * (1 + BETA*q)) + LHF / (L_v) * BETA * T)
    
    Parameters
    ----------
    p : xarray.DataArray
        Surface pressure (Pa).
    shf : xarray.DataArray
        Sensible heat flux (W/m^2).
    lhf : xarray.DataArray
        Latent heat flux (W/m^2).
    T : xarray.DataArray
        Temperature (K).
    q : xarray.DataArray
        Specific humidity (kg/kg).
    
    Returns
    -------
    xarray.DataArray
        Surface buoyancy flux (W/m^2).
    
    References
    ----------
    Chen et al. (2025) - following de Szoeke formulation.
    """
    Tv = calc_T_v(T, q)
    rho = calc_density(T, q, p)

    # Units: m^2/s^3 (flux of kinetic energy)
    return G / (rho * Tv) * (shf / CPD_BF * (1 + BETA * q) + lhf / LV * BETA * T)


def calc_evap_from_lhf(lhf: xr.DataArray, SST: xr.DataArray) -> xr.DataArray:
    """
    Calculate evaporation from latent heat flux and SST.

    Parameters
    ----------
    lhf : xr.DataArray or array-like
        Latent heat flux [W m-2].
    SST : xr.DataArray or array-like
        Sea surface temperature [K].

    Returns
    -------
    xr.DataArray or array-like
        Evaporation [kg m-2 s-1].

    Notes
    -----
    - The latent heat of vaporization is temperature-dependent and calculated
    following Zhao et al. (2024):
    https://lpdaac.usgs.gov/documents/2254/VNP16_User_Guide_V1.pdf?utm_source=chatgpt.com
    """
    Lv = LV - 0.002361 * (SST - K2C) * 1e6  # J/kg
    evap = -lhf / Lv
    return evap


# ==============================================================================
# DRY THERMODYNAMICS
# ==============================================================================
def t_pot(T, p, p_norm=1000.00):
    """
    Potential temperature.
    
    Calculates the potential temperature at a given pressure level for a given
    temperature. The potential temperature is the temperature an air parcel
    would have if adiabatically brought to a reference pressure.
    
    Parameters
    ----------
    T : float or array
        Temperature at pressure level (K).
    p : float or array
        Pressure level (hPa).
    p_norm : float, optional
        Reference (normalization) pressure (hPa). Default is 1000 hPa.
    
    Returns
    -------
    float or array
        Potential temperature (K).
    """
    return T * (p_norm / p) ** (R_DRY / CPD)


def pres2height(p, scale=7, norm=1013.25):
    """
    Convert pressure to height using barometric formula.
    
    Converts pressure to height levels using a simple exponential scale height
    model: z = -H * ln(p/p0), where H is the scale height.
    
    Parameters
    ----------
    pres : float or array
        Pressure (hPa).
    scale : float, optional
        Scale height (km). Default is 7 km.
    norm : float, optional
        Reference pressure (hPa). Default is 1013.25 hPa.
    
    Returns
    -------
    float or array
        Height (km).
    """
    return -scale * (np.log(p) - np.log(norm))


def height2pres(height, scale=7, norm=1013.25):
    """
    Convert height to pressure using barometric formula.
    
    Converts height to pressure using a simple exponential scale height model:
    p = p0 * exp(-z/H), where H is the scale height.
    
    Parameters
    ----------
    height : float or array
        Height (km).
    scale : float, optional
        Scale height (km). Default is 7 km.
    norm : float, optional
        Reference pressure (hPa). Default is 1013.25 hPa.
    
    Returns
    -------
    float or array
        Pressure (hPa).
    """
    return norm * np.exp(height / -scale)


# ==============================================================================
# DYNAMICS
# ==============================================================================
def sfcwind(u, v):
    """
    Surface wind speed.
    
    Calculates the surface wind velocity magnitude from u and v components
    using the Pythagorean theorem: V = sqrt(u^2 + v^2).
    
    Parameters
    ----------
    u : float or array
        Zonal (u) surface wind component (m/s).
    v : float or array
        Meridional (v) surface wind component (m/s).
    
    Returns
    -------
    float or array
        Surface wind speed (m/s).
    """
    return np.sqrt(u**2 + v**2)


def N2_from_ϴ(ϴ, height_unit='km'):
    """
    Brunt-Väisälä frequency from potential temperature.
    
    Calculates the Brunt-Väisälä frequency (squared), which measures static
    stability, from a potential temperature field.
    N^2 = (G / ϴ) * (dϴ / dz)
    
    Parameters
    ----------
    ϴ : xarray.DataArray
        Potential temperature (K) with 'height' dimension.
    height_unit : str, optional
        Height coordinate unit: 'km' or 'm'. Default is 'km'.
    
    Returns
    -------
    xarray.DataArray
        Brunt-Väisälä frequency squared (1/s^2).
    
    Notes
    -----
    - Positive N^2 indicates stable stratification
    - N^2 = 0 indicates neutral stratification
    - Negative N^2 indicates unstable stratification
    """
    if height_unit == 'km':
        return G / ϴ * ϴ.differentiate('height') * 0.001
    elif height_unit == 'm':
        return G / ϴ * ϴ.differentiate('height')
    else:
        raise ValueError(f"height_unit must be 'km' or 'm', got '{height_unit}'")


def N2_from_T(T, p):
    """
    Brunt-Väisälä frequency from temperature.
    
    Calculates the Brunt-Väisälä frequency (squared) from a temperature field
    by first converting to potential temperature, then computing N^2.
    
    Parameters
    ----------
    T : xarray.DataArray
        Temperature (K) with dimensions including height.
    pres : xarray.DataArray or float
        Pressure (hPa).
    
    Returns
    -------
    xarray.DataArray
        Brunt-Väisälä frequency squared (1/s^2).
    """
    ϴ = t_pot(T, p)
    return N2_from_ϴ(ϴ)


def tropopause_value(dset, method=None, thresh=None):
    """
    Tropopause value from Brunt-Väisälä frequency.
    
    Returns the data at the tropopause height level based on user-specified
    criteria. The tropopause can be identified using a Brunt-Väisälä
    frequency threshold.
    
    Parameters
    ----------
    dset : xarray.DataArray
        Data field as a function of height (e.g., N^2 or temperature).
    method : str, optional
        Method for tropopause detection:
        - 'bvf2_thresh': Simple threshold crossing
        - 'bvf2_thresh_spline': Spline interpolation for precise height
    thresh : float, optional
        Threshold value for tropopause detection.
    
    Returns
    -------
    xarray.DataArray or float
        Tropopause height or interpolated value, depending on method.
    
    Raises
    ------
    ValueError
        If method is not specified or is invalid.
    """
    if method == 'bvf2_thresh':
        return abs(dset - thresh).argmin(dim='height')
    elif method == 'bvf2_thresh_spline':
        return ipl.InterpolatedUnivariateSpline(
            dset['height'][::-1], dset[::-1] - thresh, k=3,
            ).roots()
    else:
        raise ValueError(f'No valid method specified! Got: {method}')
