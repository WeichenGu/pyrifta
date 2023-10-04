# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:40:07 2023

@author: Etrr
"""

import numpy as np
from lib_rifta.surface_extension_2d.Chebyshev_XYnm import Chebyshev_XYnm
from lib_rifta.surface_extension_2d.Legendre_XYnm import Legendre_XYnm


def Surface_Extension_Polyfit(X, Y, Z, tif_mpp, Z_tif, order_m, order_n, poly_type):
    """
    Extend the surface error map using polynomial fitting.

    Parameters:
        X, Y, Z : np.ndarray
            Unextended surface error map.
        tif_mpp : float
            TIF sampling interval [m/pxl].
        Z_tif : np.ndarray
            TIF profile.
        order_m, order_n : int
            Polynomial orders in y, x.
        poly_type : str
            Type of polynomial ('Chebyshev' or 'Legendre').

    Returns:
        X_ext, Y_ext, Z_ext : np.ndarray
            Extended surface error map.
        ca_range : dict
            Dictionary containing y and x start & end ids of CA in FA [pixel].
    """
    # 0. Obtain required parameters
    surf_mpp = np.median(np.diff(X[0, :]))
    m, n = Z.shape
    m_ext = int(np.floor(tif_mpp * Z_tif.shape[0] * 0.5 / surf_mpp))
    n_ext = int(np.floor(tif_mpp * Z_tif.shape[1] * 0.5 / surf_mpp))

    ca_range = {
        'y_s': m_ext,
        'y_e': m_ext + m,
        'x_s': n_ext,
        'x_e': n_ext + n
    }

    # 1. Initial extension matrices
    X_ext, Y_ext = np.meshgrid(np.arange(-n_ext, n + n_ext), np.arange(-m_ext, m + m_ext))
    X_ext = X_ext * surf_mpp + X[0, 0]
    Y_ext = Y_ext * surf_mpp + Y[-1, -1]
    Z_ext = np.full(X_ext.shape, np.nan)
    Z_ext[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] = Z

    # Fit the edge values
    w = 100
    Z_ext[:, 0] = 0
    Z_ext[0, :] = 0
    Z_ext[:, -1] = 0
    Z_ext[-1, :] = 0

    W = np.ones(Z_ext.shape)
    W[:, 0] = w
    W[0, :] = w
    W[:, -1] = w
    W[-1, :] = w

    # 2. Poly fit
    p, q = np.meshgrid(np.arange(order_n + 1), np.arange(order_m + 1))
    X_nor = -1 + 2 * (X_ext - X_ext.min()) / (X_ext.max() - X_ext.min())
    Y_nor = -1 + 2 * (Y_ext - Y_ext.min()) / (Y_ext.max() - Y_ext.min())

    if poly_type == 'Chebyshev':

        z3, _, _ = Chebyshev_XYnm(X_nor, Y_nor, p.ravel(), q.ravel())
    elif poly_type == 'Legendre':
        print('not finished legendre yet')
        # z3, _, _ = Legendre_XYnm(X_nor, Y_nor, p.ravel(), q.ravel())
    else:
        raise ValueError('Unknown polynomial type.')

    z3_res = z3.reshape((-1, z3.shape[-1]))

    A = z3_res[~np.isnan(Z_ext.ravel()), :]
    b = Z_ext[~np.isnan(Z_ext)]
    c = np.linalg.lstsq(A, b, rcond=None)[0]

    for i in range(len(c)):
        z3[:, :, i] = z3[:, :, i] * c[i]

    Z_ext = z3.sum(axis=2)

    return X_ext, Y_ext, Z_ext, ca_range
