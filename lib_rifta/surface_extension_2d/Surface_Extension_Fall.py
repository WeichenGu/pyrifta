# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 00:33:16 2023

@author: frw78547
"""

import numpy as np
from scipy.integrate import quad

def Surface_Extension_Fall(X_ext, Y_ext, Z_ext, ca_range, brf_params, Z_tif):
    """
    Apply the fall profile to the extended part of the surface

    :param X_ext: extended surface X coordinates
    :param Y_ext: extended surface Y coordinates
    :param Z_ext: extended surface Z values
    :param ca_range: clear aperture range in pixels
    :param brf_params: BRF parameters
    :param Z_tif: TIF profile
    :return: Z_fall: extended surface with fall profile applied
    """
    from .Surface_Extension_EdgeExtraction import Surface_Extension_EdgeExtraction
    # Obtain parameters
    r = (max(Z_tif.shape) - 1) * brf_params['lat_res_brf'] * 0.5

    Z_fall = np.full_like(Z_ext, np.nan)
    Z_fall[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] = 1

    # Find edge points
    id_edg = Surface_Extension_EdgeExtraction(Z_fall)
    u_edg = X_ext[id_edg]
    v_edg = Y_ext[id_edg]

    # Obtain the filled & original ids
    id_fil = np.isnan(Z_fall)
    u_fil = X_ext[id_fil]
    v_fil = Y_ext[id_fil]

    # Calculate fall profiles
    fun = lambda x, A, sigma: A * np.exp(-(x)**2 / (2 * sigma**2))
    B = 1 / quad(lambda x: fun(x, brf_params['A'], brf_params['sigma_xy'][0]), -r, r)[0]

    fall_profiles = np.zeros_like(u_fil)
    for k in range(len(u_fil)):
        min_distance = np.min(np.sqrt((u_fil[k] - u_edg)**2 + (v_fil[k] - v_edg)**2))
        fall_profiles[k] = B * quad(lambda x: fun(x, brf_params['A'], brf_params['sigma_xy'][0]), -(r - min_distance), r)[0]

    Z_fall[id_fil] = fall_profiles
    Z_fall = Z_ext * Z_fall
    return Z_fall
