# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:32:00 2023

@author: frw78547
"""

import numpy as np
from scipy.ndimage import binary_dilation
from math import floor
from numpy import isnan


def Surface_Extension_8NN(X, Y, Z, tif_mpp, Z_tif):
    
    # Obtain required parameters
    surf_mpp = np.median(np.diff(X[0, :]))
    m = Z.shape[0]
    n = Z.shape[1]
    m_ext = floor(tif_mpp * Z_tif.shape[0] * 0.5 / surf_mpp)
    n_ext = floor(tif_mpp * Z_tif.shape[1] * 0.5 / surf_mpp)
    ca_range = {'y_s': m_ext + 1, 'y_e': m_ext + m, 'x_s': n_ext + 1, 'x_e': n_ext + n}
    
    # Initial extension matrices
    X_ext, Y_ext = np.meshgrid(np.arange(-n_ext, n+n_ext), np.arange(-m_ext, m+m_ext))
    X_ext = X_ext * surf_mpp + X[0, 0]
    Y_ext = Y_ext * surf_mpp + Y[-1, -1]
    Z_ext = np.full(X_ext.shape, np.nan)
    Z_ext[ca_range['y_s']-1:ca_range['y_e'], ca_range['x_s']-1:ca_range['x_e']] = Z
    BW_ini = ~isnan(Z_ext)
    BW_prev = BW_ini
    h = Z_ext.shape[0]
    w = Z_ext.shape[1]
    
    # Filling the invalid points
    r = 1
    while r <= max(m_ext, n_ext):
        u, v = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
        rr = np.sqrt(u**2 + v**2)
        se = rr <= r
        BW_curr = binary_dilation(BW_ini, structure=se)
        BW_fill = BW_curr ^ BW_prev
        idy, idx = np.nonzero(BW_fill == 1)
        # idx , idy = np.nonzero(BW_fill == 1) 
        while idy.size != 0:
            for k in range(idy.size):
                count = 0
                nn_sum = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i != 0 or j != 0:
                            idi = idy[k] + i
                            idj = idx[k] + j
                            if (0 < idi <= h-1 and 0 < idj <= w-1 and not isnan(Z_ext[idi, idj])):  #Z_ext[idi-1, idj-1]
                                count += 1
                                # nn_sum += Z_ext[idi-1, idj-1]
                                nn_sum += Z_ext[idi, idj]
                if count >= 3:
                    Z_ext[idy[k], idx[k]] = nn_sum / count
                    BW_fill[idy[k], idx[k]] = 0
            # print(k)
            idy, idx = np.where(BW_fill == True)
            # idx, idy = np.where(BW_fill == 1)
        BW_prev = BW_curr
        r += 1
    
    Z_ext[isnan(Z_ext)] = 0
    
    return X_ext, Y_ext, Z_ext, ca_range
