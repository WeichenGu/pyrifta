# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 03:31:51 2024

@author: Etrr
"""
import numpy as np
from scipy.interpolate import interp2d
from lib_rifta import BRFGaussian2D


def DwellTime2D_Assemble_C_d(Nr, Nt, brf_params, z_to_remove, X, Y, P, X_brf, Y_brf, Z_avg, ca_range, resampling_method):
    # Assemble the BRF matrix C, vector d, and pseudo-transpose matrix C_T, similar to the MATLAB version
    A = brf_params['A']                   # Peak removal rate [m/s]
    sigma_xy = brf_params['sigma_xy']     # Standard deviation [m]
    mu_xy = [0, 0]                        # Center is 0 [m]

    # 2. Get the clear aperture size
    ca_m = ca_range['y_e'] - ca_range['y_s'] + 1
    ca_n = ca_range['x_e'] - ca_range['x_s'] + 1

    row_C = ca_m * ca_n

    # 3. Assemble the matrix C
    C = np.zeros((row_C, Nt))
    C_T = np.zeros((Nr, Nt))
    for i in range(Nt):
        Yk = Y - P[i, 1]   # yk - vi
        Xk = X - P[i, 0]   # xk - ui

        if resampling_method.lower() == 'avg':
            interpolating_function = interp2d(X_brf, Y_brf, Z_avg, kind='linear')
            z_brf = interpolating_function(Xk, Yk)
            z_brf[~np.isfinite(z_brf)] = 0
        elif resampling_method.lower() == 'model':
            z_brf = BRFGaussian2D(Xk, Yk, 1, [A, sigma_xy, mu_xy])

        z_brf = np.array(z_brf)
        C_T[:, i] = z_brf.flatten()
        z_brf_ca = z_brf[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
        C[:, i] = z_brf_ca.flatten()

    # 4. Assemble the vector d
    d = z_to_remove[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    d = d - np.nanmin(d)
    d = d.flatten()

    return C, d, C_T