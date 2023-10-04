# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:50:11 2023

@author: frw78547
"""

import numpy as np

def Surface_Extension_GP(X, Y, Z, tif_mpp, Z_tif, fu_range, fv_range):
    from .Surface_Extension_GerchbergPapoulis import Surface_Extension_GerchbergPapoulis
    # 0. Obtain required parameters
    surf_mpp = np.median(np.diff(X[0, :]))

    m = Z.shape[0]
    n = Z.shape[1]

    m_ext = int(np.floor(tif_mpp * Z_tif.shape[0] * 0.5 / surf_mpp))
    n_ext = int(np.floor(tif_mpp * Z_tif.shape[1] * 0.5 / surf_mpp))

    ca_range = {
        'v_s': m_ext,
        'v_e': m_ext + m,
        'u_s': n_ext,
        'u_e': n_ext + n
    }

    # 1. Initial extension matrices
    X_ext, Y_ext = np.meshgrid(np.arange(-n_ext, n + n_ext), np.arange(-m_ext, m + m_ext))
    X_ext = X_ext * surf_mpp + X[0, 0]
    Y_ext = Y_ext * surf_mpp + Y[-1, -1]

    Z_ini = np.empty_like(X_ext)
    Z_ini[:] = np.NaN
    Z_ini[ca_range['v_s']:ca_range['v_e'], ca_range['u_s']:ca_range['u_e']] = Z

    Z_ext = np.zeros_like(X_ext)
    Z_ext[ca_range['v_s']:ca_range['v_e'], ca_range['u_s']:ca_range['u_e']] = Z
    Z_ext[np.isnan(Z_ext)] = 0

    G = np.zeros_like(Z_ext)
    G[~np.isnan(Z_ini)] = 1

    Gy = np.zeros_like(G)
    Gy[ca_range['v_s']:ca_range['v_e'], :] = 1

    Gox = np.zeros_like(Z_ext)
    Goy = np.zeros_like(Z_ext)
    Gox[:, int(Z_ext.shape[1] / 2) + 1 + fu_range] = 1
    Goy[int(Z_ext.shape[0] / 2) + 1 + fv_range, :] = 1


    Z_ext = Surface_Extension_GerchbergPapoulis(Z_ext, G, Gy, Gox, Goy)

    return X_ext, Y_ext, Z_ext, ca_range
