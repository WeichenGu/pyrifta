# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 02:41:37 2025

@author: Etrr
"""

import numpy as np
from scipy.linalg import svd

# assumes assemble_C_d and remove_surface1 are available Python functions
# from dwell_time_2d_assemble_C_d import assemble_C_d
from lib_rifta.ibf_engine.remove_surface1 import remove_surface1

def DwellTime2D_TSVD(
    Z_to_remove, X, Y,
    brf_params,
    X_brf, Y_brf, Z_avg,
    X_P, Y_P,
    ca_range,
    rms_d,
    resampling_method
):
    """
    Compute dwell time map using truncated SVD for 2D IBF.

    Parameters
    ----------
    Z_to_remove : 2D ndarray
        Desired removal map.
    X, Y : 2D ndarray
        Grid points of Z_to_remove.
    brf_params : dict or object
        Parameters for the beam removal function (BRF).
    X_brf, Y_brf, Z_avg : ndarray
        Coordinates and average height for BRF sampling.
    X_P, Y_P : 2D ndarray
        Coordinates of dwell points (shape m x n).
    ca_range : object
        Contains y_s, y_e, x_s, x_e for clear aperture indices.
    rms_d : float
        Desired RMS of the final residual.
    resampling_method : str
        Method for resampling BRF data.

    Returns
    -------
    T : 2D ndarray
        Dwell time map (m x n).
    z_removal : 2D ndarray
        Predicted removed height over full map.
    z_residual : 2D ndarray
        Predicted residual height over full map.
    z_to_remove_ca : 2D ndarray
        Desired removal on clear aperture.
    z_removal_ca : 2D ndarray
        Predicted removal on clear aperture.
    z_residual_ca : 2D ndarray
        Predicted residual on clear aperture.
    C : 2D ndarray
        BRF matrix (Nr x Nt).
    """
    # 1. Flatten dwell points into P array (Nt x 2)
    P = np.column_stack((X_P.ravel(), Y_P.ravel()))
    Nt = P.shape[0]
    Nr = Z_to_remove.size

    # 2. Assemble matrix C, vector d, and transpose matrix
    C, d, C_T = assemble_C_d(
        Nr, Nt, brf_params,
        Z_to_remove, X, Y, P,
        X_brf, Y_brf, Z_avg,
        ca_range, resampling_method
    )

    # 3. SVD of C
    U, sigma_vals, Vt = svd(C, full_matrices=False)
    V = Vt.T

    # 4. Truncated SVD to meet rms_d
    Z_to_remove_ca = Z_to_remove[
        ca_range.y_s:ca_range.y_e,
        ca_range.x_s:ca_range.x_e
    ]

    tmp = U.T @ d
    k_opt = len(sigma_vals)
    for k in range(1, len(sigma_vals) + 1):
        # compute T using first k singular values
        Tk = V[:, :k] @ (tmp[:k] / sigma_vals[:k])
        z_removal_ca_test = C.dot(Tk)
        res = Z_to_remove_ca.ravel() - z_removal_ca_test
        if np.std(res, ddof=0) < rms_d:
            k_opt = k
            break

    # 5. Compute dwell time with piston adjustment
    d_piston = d.copy()
    while True:
        tmp_p = U.T @ d_piston
        T_full = V[:, :k_opt] @ (tmp_p[:k_opt] / sigma_vals[:k_opt])
        if T_full.min() >= 0:
            break
        d_piston += 0.1e-9
        if T_full.min() < 1e-14:
            T_full = T_full - T_full.min()
            break

    # 6. Clear aperture results
    z_removal_ca = C.dot(T_full).reshape(Z_to_remove_ca.shape)
    z_residual_ca = (d_piston - C.dot(T_full)).reshape(Z_to_remove_ca.shape)
    z_to_remove_ca = d_piston.reshape(Z_to_remove_ca.shape)

    # 7. Detilt clear aperture
    X_ca = X[ca_range.y_s:ca_range.y_e, ca_range.x_s:ca_range.x_e]
    Y_ca = Y[ca_range.y_s:ca_range.y_e, ca_range.x_s:ca_range.x_e]

    z_to_remove_ca = remove_surface1(X_ca, Y_ca, z_to_remove_ca)
    z_to_remove_ca -= np.nanmin(z_to_remove_ca)

    z_removal_ca = remove_surface1(X_ca, Y_ca, z_removal_ca)
    z_removal_ca -= np.nanmin(z_removal_ca)

    z_residual_ca = remove_surface1(X_ca, Y_ca, z_residual_ca)

    # 8. Full map results
    z_removal = C_T.dot(T_full).reshape(Z_to_remove.shape)
    z_residual = (Z_to_remove.ravel() - C_T.dot(T_full)).reshape(Z_to_remove.shape)

    # 9. Reshape T into grid shape
    T = T_full.reshape(X_P.shape)

    return T, z_removal, z_residual, z_to_remove_ca, z_removal_ca, z_residual_ca, C
