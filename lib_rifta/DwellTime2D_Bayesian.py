# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 02:43:23 2025

@author: Etrr
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import convolve, gaussian_filter
from scipy.ndimage import zoom
from lib_rifta import BRFGaussian2D, BRFSuperGaussian2D
from lib_rifta.ibf_engine.remove_surface1 import remove_surface1
from lib_rifta.ibf_engine.conv_fft2 import conv_fft2

# Assumed available helpers:
#   BRFGaussian2D(x, y, scale, params)
#   conv_fft2(image, kernel)
#   divergence(gx, gy)
#   gradient(arr)
#   remove_surface1(x, y, z)


def DwellTime2D_Bayesian(
    Z_to_remove,
    BRF_mode,
    BRF_params,
    X_BRF, Y_BRF, Z_BRF,
    ca_range,
    pixel_m,
    options=None,
    ratio=1.0
):
    """
    Bayesian-based dwell time computation for 2D IBF.

    Parameters
    ----------
    Z_to_remove : 2D ndarray
        Desired height removal map [m].
    BRF_mode : str
        'avg' to interpolate average BRF, else use Gaussian model.
    BRF_params : object
        Contains A, sigma_xy, d_pix attributes.
    X_BRF, Y_BRF : 1D or 2D coords for Z_BRF sampling.
    Z_BRF : 2D ndarray
        Averaged BRF heights.
    ca_range : object
        Attributes x_s, x_e, y_s, y_e, and optional mask.
    pixel_m : float
        Pixel size [m/pixel].
    options : dict, optional
        maxIters, RMS_dif, dwellTime_dif, lambda, isDownSampling.
    ratio : float, optional
        Scaling factor for dwell map.

    Returns
    -------
    B : 2D ndarray
        Beam removal footprint [m/s].
    X_B, Y_B : 2D coords for B.
    X, Y : full-aperture coords.
    Z_removal, Z_residual : full-ap results.
    T_P : dwell time grid [s].
    X_P, Y_P : dwell grid coords.
    X_dw, Y_dw : dwell-grid coords.
    Z_to_remove_dw, Z_removal_dw, Z_residual_dw : dwell results.
    X_ca, Y_ca : clear-ap coords.
    Z_to_remove_ca, Z_removal_ca, Z_residual_ca : clear-ap results.
    
    """
    def divergence(gx, gy, dx=1.0, dy=1.0):
        dgx_dx = np.gradient(gx, dx, axis=1, edge_order=2)
        dgy_dy = np.gradient(gy, dy, axis=0, edge_order=2)
        return dgx_dx + dgy_dy

    # 0. default options
    defaults = dict(
        maxIters=10,
        RMS_dif=0.04e-9,
        dwellTime_dif=20.0,
        lambda_tv=0.1e1,
        isDownSampling=False
    )
    if options is None:
        options = defaults
    else:
        for k, v in defaults.items():
            options.setdefault(k, v)

    A = BRF_params.A
    sigma_xy = BRF_params.sigma_xy
    d_pix = BRF_params.d_pix
    mu_xy = (0.0, 0.0)
    l = options['lambda_tv']

    # 2. build BRF footprint B
    r_p = int(np.floor(0.5 * d_pix))
    x = np.arange(-r_p, r_p) * pixel_m
    y = np.arange(-r_p, r_p) * pixel_m
    X_B, Y_B = np.meshgrid(x, y)
    if BRF_mode.lower() == 'avg':
        interp = RectBivariateSpline(
            X_BRF[:,0], Y_BRF[0,:], Z_BRF.T, kx=3, ky=3
        )
        B = interp(x, y)
    else:
        B = BRFGaussian2D(X_B, Y_B, 1.0, (A, sigma_xy, mu_xy))
    # 2. 定义 grids   
    mM, nM = Z_to_remove.shape
    dw_x_s = ca_range.x_s - r_p
    dw_x_e = ca_range.x_e + r_p
    dw_y_s = ca_range.y_s - r_p
    dw_y_e = ca_range.y_e + r_p
    x_full = np.arange(nM) * pixel_m
    y_full = np.arange(mM) * pixel_m
    X, Y = np.meshgrid(x_full, y_full)
    X_dw = X[dw_y_s:dw_y_e, dw_x_s:dw_x_e]
    Y_dw = Y[dw_y_s:dw_y_e, dw_x_s:dw_x_e]
    X_ca = X[ca_range.y_s:ca_range.y_e, ca_range.x_s:ca_range.x_e]
    Y_ca = Y[ca_range.y_s:ca_range.y_e, ca_range.x_s:ca_range.x_e]
    
    

    # 3. prepare removal arrays
    Z = Z_to_remove.copy()

    if hasattr(ca_range, 'dw_mask'):
        Z = np.where(ca_range.dw_mask, Z, np.nan)
    
    Z_to_remove_dw = Z[dw_y_s:dw_y_e, dw_x_s:dw_x_e]
    Z_to_remove_dw -= np.nanmin(Z_to_remove_dw)
    
    Z_to_remove_ca = Z[ca_range.y_s:ca_range.y_e, ca_range.x_s:ca_range.x_e]
    
    if hasattr(ca_range, 'mask'):
        Z_to_remove_ca[~ca_range.mask] = np.nan
    Z_to_remove_ca = remove_surface1(X_ca, Y_ca, Z_to_remove_ca)
    Z_to_remove_ca -= np.nanmin(Z_to_remove_ca)
    


    # init
    Tk = Z_to_remove_dw.copy()
    if hasattr(ca_range, 'mask'):
        dw_mask = ca_range.dw_mask
        Tk[~dw_mask] = 0.0
    Bhat = B[::-1, ::-1]
    Bn = np.sum(B)
    Tk0 = Tk.copy()

    # iterations
    Z_res_ca_prev = None
    for it in range(1, options['maxIters']+1):
        # 梯度和归一化
        gx, gy = np.gradient(Tk, pixel_m, pixel_m)
        normG = np.sqrt(np.nansum(gx**2 + gy**2))
        gx /= normG; gy /= normG
    
        # 计算 divergence
        div = divergence(gx, gy, pixel_m, pixel_m)
    
        # 更新公式
        denom = (1.0 - options['lambda_tv'] * div)
        conv1 = conv_fft2(
            Tk0 / (conv_fft2(Tk, B) + np.finfo(float).eps),
            Bhat / Bn
        )
        T_P = (Tk / denom) * conv1
    
        if hasattr(ca_range, 'dw_mask'):
            T_P[~ca_range.dw_mask] = 0.0

        # 全孔径预测
        T_full = np.zeros_like(Z)
        T_full[dw_y_s:dw_y_e, dw_x_s:dw_x_e] = T_P
        Z_removal = conv_fft2(T_full, B)
        Z_residual = Z - Z_removal

        Z_res_ca = Z_residual[
            ca_range.y_s:ca_range.y_e,
            ca_range.x_s:ca_range.x_e
        ]
        if hasattr(ca_range, 'mask'):
            Z_res_ca[~ca_range.mask] = np.nan
            
    
    
    
        # 收敛判断
        if Z_res_ca_prev is not None:
            diff = np.nanstd(Z_res_ca - Z_res_ca_prev)
            if diff < opts['RMS_dif']:
                break
            if np.nanstd(Z_res_ca) > np.nanstd(Z_res_ca_prev):
                T_P = Tk.copy()
                break
            if abs(np.sum(T_P - Tk)) < opts['dwellTime_dif']:
                break
        Z_residual_ca_prev = Z_residual_ca.copy()
        Tk = T_P.copy()

    # Post-process clear aperture residual
    Z_removal_ca = remove_surface1(X_ca, Y_ca, Z_removal_ca)
    Z_removal_ca -= np.nanmin(Z_removal_ca)
    
    # 对 residual 做去斜面（不归一，直接取原值减去最小值也可以）
    Z_residual_ca = remove_surface1(X_ca, Y_ca, Z_residual_ca)

    # Dwell-grid results
    Z_removal_dw = Z_removal[dw_y_s:dw_y_e, dw_x_s:dw_x_e]
    Z_residual_dw = Z_residual[dw_y_s:dw_y_e, dw_x_s:dw_x_e]

    # 5. Downsampling
    '''    '''
    if options['isDownSampling']:
        print("未完成")
        pixel_P_m = options['samplingInterval']
        interval = pixel_P_m / pixel_m
        X_P = zoom(X_dw, 1/interval)
        Y_P = zoom(Y_dw, 1/interval)
        P = np.column_stack((X_P.ravel(), Y_P.ravel()))
        Nt = P.shape[0]; Nr = Z_to_remove.size
        C, d, C_T = DwellTime2D_Assemble_C_d(Nr, Nt, BRF_params,
                                            Z_to_remove, X, Y, P,
                                            X_BRF, Y_BRF, Z_BRF,
                                            ca_range, BRF_mode)
        
        T_P = zoom(T_P, 1/interval, order=3) * interval**2
        T_P_v = np.clip(T_P.ravel(), 0, None)
        
        # Clear aperture results
        Z_removal_ca = C.dot(T_P_v)
        Z_residual_ca = d - Z_removal_ca
        Z_removal_ca = Z_removal_ca.reshape(Z_to_remove_ca.shape)
        Z_residual_ca = Z_residual_ca.reshape(Z_to_remove_ca.shape)
        # Z_to_remove_ca corresponds to d
        Z_to_remove_ca = d.reshape(Z_to_remove_ca.shape)

        # Detilt clear aperture
        Z_to_remove_ca = remove_surface1(X_ca, Y_ca, Z_to_remove_ca)
        Z_to_remove_ca -= np.nanmin(Z_to_remove_ca)
        Z_removal_ca = remove_surface1(X_ca, Y_ca, Z_removal_ca)
        Z_removal_ca -= np.nanmin(Z_removal_ca)
        Z_residual_ca = remove_surface1(X_ca, Y_ca, Z_residual_ca)

        # Full aperture results
        Z_removal = C_T.dot(T_P_v)
        Z_residual = Z_to_remove.ravel() - Z_removal
        Z_residual -= np.nanmean(Z_residual)
        Z_removal = Z_removal.reshape(Z_to_remove.shape)
        Z_residual = Z_residual.reshape(Z_to_remove.shape)

        # Dwell grid results
        Z_to_remove_dw = Z_to_remove[dw_y_s:dw_y_e, dw_x_s:dw_x_e]
        Z_removal_dw = Z_removal[dw_y_s:dw_y_e, dw_x_s:dw_x_e]
        Z_residual_dw = Z_residual[dw_y_s:dw_y_e, dw_x_s:dw_x_e]
        
    else:
        X_P, Y_P = X_dw, Y_dw

    X_P, Y_P = X_dw, Y_dw
    # Scale
    T_P *= ratio

    return (
        B, X_B, Y_B,
        X, Y,
        Z_removal, Z_residual,
        T_P,
        X_P, Y_P,
        X_dw, Y_dw,
        Z_to_remove_dw, Z_removal_dw, Z_residual_dw,
        X_ca, Y_ca,
        Z_to_remove_ca, Z_removal_ca, Z_residual_ca
    )

