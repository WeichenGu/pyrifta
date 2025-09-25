# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 03:21:12 2024

@author: Etrr
"""

import numpy as np
from scipy.sparse.linalg import lsqr
from lib_rifta import DwellTime2D_Assemble_C_d
from lib_rifta.ibf_engine.remove_surface1 import remove_surface1

def DwellTime2D_LSQR(z_to_remove, X, Y, brf_params, X_brf, Y_brf, Z_avg, X_P, Y_P, ca_range, rms_d, resampling_method):
    """
    Implements the LSQR dwell time algorithm to predict removal and residual error.
    
    Parameters:
    - z_to_remove: Desired removal map
    - X, Y: X and Y grid points of the z_to_remove map
    - brf_params: BRF parameters
    - X_brf, Y_brf, Z_avg: Data for BRF
    - X_P, Y_P: Coordinates of the dwell points
    - ca_range: Clear aperture range
    - rms_d: Desired RMS value of the final residual
    - resampling_method: Method to use for resampling
    
    Returns:
    - T: Dwell time map
    - z_removal: Prediction of the removed height
    - z_residual: Prediction of the residual height
    - z_to_remove_ca: Desired removal for the clear aperture
    - z_removal_ca: Removal amount predicted using T for the clear aperture
    - z_residual_ca: Prediction of the residual after removal using T for the clear aperture
    """
    # 1. Dump X_P, Y_P dwell point positions into a 2D array
    P = np.column_stack((X_P.flatten(), Y_P.flatten()))

    # 2. Get the number of IBF machining points and sampling points of the surface error map R
    Nt = P.shape[0]
    Nr = z_to_remove.size

    # 3. Assemble the BRF matrix C, vector d, and pseudo-transpose matrix C_T
    C, d, C_T = DwellTime2D_Assemble_C_d(Nr, Nt, brf_params, z_to_remove, X, Y, P, X_brf, Y_brf, Z_avg, ca_range, resampling_method)

    # 4. Extract clear aperture region
    z_to_remove_ca = z_to_remove[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    m, n = C.shape

    # 5. Piston adjustment and damp optimization
    gamma = 0
    damp = 1e-9
    d_piston = d + gamma

    T = lsqr(C, d_piston, damp=damp, iter_lim=100)[0]

    while np.min(T) < 0:
        # Piston adjustment
        gamma += 0.2e-9
        d_piston = d + gamma

        # Optimize damp factor
        T = lsqr(C, d_piston, damp=damp, iter_lim=100)[0]
        z_removal_ca = C @ T
        z_residual_ca = z_to_remove_ca.flatten() - z_removal_ca

        while np.std(z_residual_ca, ddof=1) > rms_d:
            damp -= 0.02e-9
            T = lsqr(C, d_piston, damp=damp, iter_lim=100)[0]
            z_removal_ca = C @ T
            z_residual_ca = z_to_remove_ca.flatten() - z_removal_ca

        print(f'Optimized damp factor = {damp:.2e}')

    print(f'Piston adjustment done. The piston added is {gamma * 1e9:.2e} [nm]')

    # 6. Add path and surface error weights
    # TODO: Implementation of weighting

    # 7. Results for clear aperture
    z_removal_ca = C @ T
    z_residual_ca = d_piston - z_removal_ca
    z_removal_ca = z_removal_ca.reshape(z_to_remove_ca.shape)
    z_residual_ca = z_residual_ca.reshape(z_to_remove_ca.shape)
    z_to_remove_ca = d_piston.reshape(z_to_remove_ca.shape)

    # Detilt and ensure non-negative values
    X_ca = X[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Y_ca = Y[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    z_to_remove_ca = remove_surface1(X_ca, Y_ca, z_to_remove_ca)
    z_to_remove_ca -= np.nanmin(z_to_remove_ca)

    z_removal_ca = remove_surface1(X_ca, Y_ca, z_removal_ca)
    z_removal_ca -= np.nanmin(z_removal_ca)

    z_residual_ca = remove_surface1(X_ca, Y_ca, z_residual_ca)

    # Full results
    z_removal = C_T @ T
    z_residual = z_to_remove.flatten() - z_removal
    z_removal = z_removal.reshape(z_to_remove.shape)
    z_residual = z_residual.reshape(z_to_remove.shape)

    # Reshape T to match the original grid
    T = T.reshape(X_P.shape)

    return T, z_removal, z_residual, z_to_remove_ca, z_removal_ca, z_residual_ca
'''
def assemble_C_d(Nr, Nt, brf_params, z_to_remove, X, Y, P, X_brf, Y_brf, Z_avg, ca_range, resampling_method):
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

        C_T[:, i] = z_brf.flatten()
        z_brf = z_brf[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
        C[:, i] = z_brf.flatten()

    # 4. Assemble the vector d
    d = z_to_remove[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    d = d - np.nanmin(d)
    d = d.flatten()

    return C, d, C_T
'''

