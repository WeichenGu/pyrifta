# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 19:52:35 2025

@author: Etrr
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from scipy.linalg import lstsq

from lib_rifta import Surface_Extension
from lib_rifta.DwellTime2D_FFT_Full_Test import DwellTime2D_FFT_Full_Test
from lib_rifta.ibf_engine.remove_surface1 import remove_surface1
from lib_rifta.ibf_engine.conv_fft2 import conv_fft2
from lib_rifta.surface_extension_2d import Chebyshev_XYnm
from lib_rifta.surface_extension_2d import Legendre_XYnm

from lib_rifta.surface_extension_2d.Surface_Extension_Smooth import Surface_Extension_Smooth
from lib_rifta.surface_extension_2d.Surface_Extension_GP import Surface_Extension_GP
from lib_rifta.surface_extension_2d.Chebyshev_XYnm import Chebyshev_XYnm
from lib_rifta.surface_extension_2d.Legendre_XYnm import Legendre_XYnm
# from lib_rifta.surface_extension_2d.Hermite_XYnm import Hermite_XYnm
# from lib_rifta.surface_extension_2d.Laguerre_XYnm import Laguerre_XYnm



def Surface_Extension_RISE(
    X, Y, Z,               # 2D numpy arrays: unextended surface error map
    tif_mpp,               # TIF sampling interval [m/pxl]
    Z_tif,                 # TIF profile (2D numpy array)
    order_m, order_n,      # polynomial orders in y, x
    poly_type,              # 'Chebyshev' or 'Legendre'
    ext_rx = 0.5, ext_ry = 0.5
):
    """
    Robust Iterative Surface Extension (Python port of MATLAB Surface_Extension_RISE)
    """
    # --- 1. CA 多项式拟合以得到 e_CA|est_fit ---
    w = 100.0
    W = np.ones_like(Z)
    W[~np.isnan(Z)] = w

    # 构造 (m+1)*(n+1) 个多项式基的指数对
    p_est_fit, q_est_fit = np.meshgrid(np.arange(order_n+1),
                               np.arange(order_m+1),
                               indexing='xy')

    # 归一化坐标到 [-1,1]
    X_nor = -1 + 2 * (X - X.min()) / (X.max() - X.min())
    Y_nor = -1 + 2 * (Y - Y.min()) / (Y.max() - Y.min())

    # 计算基函数值 (shape: m×n×Ncoeff)
    if poly_type == 'Chebyshev':
        Z_basis_3, _, _ = Chebyshev_XYnm(X_nor, Y_nor,
                                      p_est_fit.ravel(), q_est_fit.ravel())
    elif poly_type == 'Legendre':
        Z_basis_3, _, _ = Legendre_XYnm(X_nor, Y_nor,
                                     p_est_fit.ravel(), q_est_fit.ravel())
    else:
        raise ValueError("Unknown polynomial type.")

    # reshape basis to 2D: (m*n) × Ncoeff
    m_dim, n_dim = Z.shape
    Ncoef = Z_basis_3.shape[2]
    Zb = Z_basis_3.reshape(m_dim*n_dim, Ncoef)

    # 构造最小二乘：只取 CA 区非 nan 值
    mask_CA = ~np.isnan(Z.ravel())
    A = Zb[mask_CA, :]
    b_vec = Z.ravel()[mask_CA]
    W_vec = W.ravel()[mask_CA]

    # 带权最小二乘求解系数 c
    # solve min ||W_vec*(A c - b_vec)||_2
    # 可以使用 lstsq 先做 sqrt 权重
    Aw = A * W_vec[:,None]
    bw = b_vec * W_vec
    c, *_ = lstsq(Aw, bw)

    # 重构 CA 拟合面
    Z_CA_est_fit = (Zb @ c).reshape(m_dim, n_dim)
    residual_fit = Z_CA_est_fit - Z
    print(f"CA polynomial fit residual: {np.nanstd(residual_fit)*1e9:.2f} nm")







    # --- 2. 计算扩展大小和 CA 在扩展网格中的索引 ---
    surf_mpp = np.median(np.diff(X[0,:]))  # [m/pxl]
    m, n = Z.shape
    # 扩展像素数
    m_ext = int(np.ceil(np.round(tif_mpp*(Z_tif.shape[0])*ext_ry/surf_mpp,decimals=4)))  # extension size in y [pixels]
    n_ext = int(np.ceil(np.round(tif_mpp*(Z_tif.shape[1])*ext_rx/surf_mpp,decimals=4)))  # extension size in x [pixels]

    ca_range = {
        'y_s': m_ext,
        'y_e': m_ext + m,
        'x_s': n_ext,
        'x_e': n_ext + n
    }

    # --- 3. 构造扩展网格并填充 CA 区 ---，由于是e_CA|est_fit, X_ext应该和X暂时一样

    ys = np.arange(-m_ext, m + m_ext)* surf_mpp + Y[0,0]
    xs = np.arange(-n_ext, n + n_ext)* surf_mpp + X[0,0]
    X_ext, Y_ext = np.meshgrid(xs, ys)
    Z_ext = np.full_like(X_ext, np.nan)
    
    # Z_ext[ca_range['y_s']:ca_range['y_e']+1, ca_range['x_s']:ca_range['x_e']+1] = Z

    # --- 4. 用平滑扩展拟合边界条件 ---
    # 对 residual_fit 做扩展来生成边界 Z_ext_boundary
    X_ext, Y_ext, Z_ext_boundry, _ = Surface_Extension_Smooth(X, Y, Z_CA_est_fit, tif_mpp, Z_tif, ext_rx, ext_ry)
    
    Z_ext[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] = Z

    # 将四边界赋值
    Z_ext[:, 0]    = Z_ext_boundry[:, 0]
    Z_ext[0, :]    = Z_ext_boundry[0, :]
    Z_ext[:, -1]   = Z_ext_boundry[:, -1]
    Z_ext[-1, :]   = Z_ext_boundry[-1, :]

    # CA 区恢复原始 Z
    # Z_ext[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] = Z

    # 构造边界权重矩阵
    W_ext = np.zeros_like(Z_ext)
    W_ext[:, 0]    = w
    W_ext[0, :]    = w
    W_ext[:, -1]   = w
    W_ext[-1, :]   = w
    mask_ext = ~np.isnan(Z_ext)
    W_ext[mask_ext] = w

    # --- 5. 对整个扩展图做多项式拟合 ---
    # 重新归一化到 [-1,1]
    p, q = np.meshgrid(np.arange(order_m + 1), np.arange(order_n + 1))  #  20240108 m n reverse,fixed it
    X_nor = -1 + 2 * (X_ext - X_ext.min()) / (X_ext.max() - X_ext.min())
    Y_nor = -1 + 2 * (Y_ext - Y_ext.min()) / (Y_ext.max() - Y_ext.min())
    
    if poly_type == 'Chebyshev':
        z3, _, _ = Chebyshev_XYnm(X_nor, Y_nor, p.ravel(), q.ravel())
    elif poly_type == 'Legendre':
        z3, _, _ = Legendre_XYnm(X_nor, Y_nor, p.ravel(), q.ravel())
    else:
        raise ValueError("Unknown polynomial type.")
    # Zb_ext = z3.reshape(-1, Ncoef)

    z3_res = z3.reshape((-1, z3.shape[-1]))
    W_flat = W_ext.ravel()
    weights = np.sqrt(W_flat[~np.isnan(Z_ext.ravel())]**2)
    A = z3_res[~np.isnan(Z_ext.ravel()), :]
    b = Z_ext[~np.isnan(Z_ext)]
    
    A = A * weights[:, np.newaxis]
    b = b * weights  # Z_ext 展开一维后的个数，也是目标
    
    # # m1.使用最小二乘拟合，考虑权重
    c, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    mse = residuals
    # print(f"Mean Squared Error (MSE): {mse:.4f}")
    print("Mean Squared Error (MSE): ",mse)

    '''
    mask_ext2 = ~np.isnan(Z_ext.ravel())
    A_ext = Zb_ext[mask_ext2, :]
    b_ext = Z_ext.ravel()[mask_ext2]
    Wext_vec = W_ext.ravel()[mask_ext2]

    # 加权最小二乘
    Aew = A_ext * Wext_vec[:,None]
    bew = b_ext * Wext_vec
    c_ext, *_ = lstsq(Aew, bew)
    '''
    # 重建扩展表面
    Z_ext = (z3_res @ c).reshape(Z_ext.shape)

    # 最后报告拟合残差
    resid2 = Z_ext[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] - Z
    print(f"Extension polynomial fit residual: {np.nanstd(resid2)*1e9:.2f} nm")

    # 返回扩展后网格和 CA 范围
    # ca_range = {'y_s': y_s, 'y_e': y_e, 'x_s': x_s, 'x_e': x_e}
    return X_ext, Y_ext, Z_ext, ca_range
