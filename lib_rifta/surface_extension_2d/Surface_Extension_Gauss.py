# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:36:20 2023

@author: frw78547
"""

import numpy as np

def Surface_Extension_Gauss(X, Y, Z, brf_params, Z_tif):

    from .Surface_Extension_EdgeExtraction import Surface_Extension_EdgeExtraction

    # Sampling intervals
    surf_mpp = np.median(np.diff(X[0,:]))

    sigma = brf_params['sigma_xy'][0]

    m = Z.shape[0]
    n = Z.shape[1]

    m_ext = int(np.floor(brf_params['lat_res_brf'] * (Z_tif.shape[0]) * 0.5 / surf_mpp))
    n_ext = int(np.floor(brf_params['lat_res_brf'] * (Z_tif.shape[1]) * 0.5 / surf_mpp))

    ca_range = {'v_s': m_ext+1, 'v_e': m_ext+m, 'u_s': n_ext+1, 'u_e': n_ext+n}

    # Initial extension matrices
    X_ext, Y_ext = np.meshgrid(np.arange(-n_ext, n-1+n_ext+1), np.arange(-m_ext, m-1+m_ext+1))
    X_ext = X_ext * surf_mpp + X[0,0]
    Y_ext = Y_ext * surf_mpp + Y[-1,-1]
    Z_ext = np.empty_like(X_ext, dtype=float)
    Z_ext[:] = np.nan
    Z_ext[ca_range['v_s']-1:ca_range['v_e'], ca_range['u_s']-1:ca_range['u_e']] = Z

    # Finding edge points
    id_edg = Surface_Extension_EdgeExtraction(Z_ext)
    u_edg = X_ext[id_edg]
    v_edg = Y_ext[id_edg]
    z_edg = Z_ext[id_edg]

    id_fil = np.isnan(Z_ext)
    u_fil = X_ext[id_fil]
    v_fil = Y_ext[id_fil]

    # Calculate the gaussian profile
    gauss_profiles = np.zeros_like(u_fil)
    for k in range(len(u_fil)):
        # min distances from filled points to edge points
        min_dist, i = np.min(np.sqrt((u_fil[k] - u_edg)**2 + (v_fil[k] - v_edg)**2)), np.argmin(np.sqrt((u_fil[k] - u_edg)**2 + (v_fil[k] - v_edg)**2))

        # calculate the fall profile
        gauss_profiles[k] = z_edg[i] * np.exp(-min_dist**2 / (2*sigma**2))

    Z_ext[id_fil] = gauss_profiles

    return X_ext, Y_ext, Z_ext, ca_range
