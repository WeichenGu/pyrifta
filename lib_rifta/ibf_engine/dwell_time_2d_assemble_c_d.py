# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 01:00:10 2023

@author: frw78547
"""

import numpy as np
from scipy.interpolate import interp2d
from lib_rifta import BRFGaussian2D

def dwell_time_2d_assemble_c_d(Nr, Nt, BRF_params, Z_to_remove, X, Y, P, X_brf, Y_brf, Z_avg, ca_range, resampling_method):
    """
    Assemble the matrix C and vector d
    
    Parameters:
    Nr (int): number of elements in Z_to_remove
    Nt (int): number of dwell positions
    BRF_params (dict): BRF parameters
    Z_to_remove (array): desired height to remove
    X (array): X coordinates of Z_to_remove
    Y (array): Y coordinates of Z_to_remove
    P (array): Dwell time positions
    X_brf (array): BRF X coordinates
    Y_brf (array): BRF Y coordinates
    Z_avg (array): Averaged BRF
    ca_range (dict): range of the clear aperture
    resampling_method (str): use 'model' or 'avg'
    """

    # 1. Release the BRF parameters
    A = BRF_params['A']                   # peak removal rate [m/s]
    sigma_xy = BRF_params['sigma_xy']     # standard deviation [m]
    mu_xy = [0, 0]                        # center is 0 [m]

    # 2. Get the clear aperture size
    ca_m = ca_range['y_e'] - ca_range['y_s'] + 1
    ca_n = ca_range['x_e'] - ca_range['x_s'] + 1

    row_C = ca_m * ca_n

    # 3. Assemble the matirx C
    C = np.zeros((row_C, Nt))
    C_T = np.zeros((Nr, Nt))
    for i in range(Nt):
        Yk = Y - P[i, 1]  # yk - vi
        Xk = X - P[i, 0]  # xk - ui

        if resampling_method.lower() == 'avg':
            z_brf = interp2d(X_brf, Y_brf, Z_avg, Xk, Yk)
            z_brf[np.logical_not(np.isfinite(z_brf))] = 0
        elif resampling_method.lower() == 'model':
            z_brf = BRFGaussian2D(Xk, Yk, 1, [A, sigma_xy, mu_xy])

        C_T[:, i] = z_brf.flatten()
        z_brf = z_brf[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
        C[:, i] = z_brf.flatten()

    # 4. Assemble the vector d
    z = Z_to_remove[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    z = z - np.nanmin(z)
    z = z.flatten()

    return C, z, C_T
