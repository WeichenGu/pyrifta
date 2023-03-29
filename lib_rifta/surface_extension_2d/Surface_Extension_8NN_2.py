# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 00:16:58 2023

@author: frw78547
"""
import numpy as np
from scipy import ndimage

def surface_extension_8NN(X, Y, Z, brf_params, tif_mpp, Z_avg):
    # Sampling intervals
    surf_mpp = np.median(np.diff(X[0,:]))   # surface sampling interval [m/pxl]

    m = Z.shape[0]   # CA height [pixel]
    n = Z.shape[1]   # CA width [pixel]
    m_ext = int(np.floor(tif_mpp*(Z_avg.shape[0])*0.5/surf_mpp))   # extension size in y [pixel]
    n_ext = int(np.floor(tif_mpp*(Z_avg.shape[1])*0.5/surf_mpp))   # extension size in x [pixel]
    ca_range = {}
    ca_range['v_s'] = m_ext + 1   # y start id of CA in FA [pixel]
    ca_range['v_e'] = ca_range['v_s'] + m - 1   # y end id of CA in FA [pixel]
    ca_range['u_s'] = n_ext + 1   # x start id of CA in FA [pixel]
    ca_range['u_e'] = ca_range['u_s'] + n - 1   # x end id of CA in FA [pixel]

    # Initial extension matrices
    X_ext, Y_ext = np.meshgrid(np.arange(-n_ext,n+n_ext), np.arange(-m_ext,m+m_ext))
    X_ext = X_ext * surf_mpp + X[0,0]   # adjust X grid add X(1,1)
    Y_ext = Y_ext * surf_mpp + Y[-1,-1]   # adjust Y grid add Y(1,1)

    Z_ext = np.full(X_ext.shape, np.nan)   # mark the Z_ext to NaN
    Z_ext[ca_range['v_s']-1:ca_range['v_e'], ca_range['u_s']-1:ca_range['u_e']] = Z   # fill in the valid data points
    BW_ini = ~np.isnan(Z_ext)   # obtain the black&white map
    BW_prev = BW_ini
    h = Z_ext.shape[0]
    w = Z_ext.shape[1]
    
    
    # Filling the invalid points
    r = 1   # extension radius (1 ~ max(m_ext, n_ext))
    while(r <= max(m_ext, n_ext)):
        u,v = np.meshgrid(np.arange(-r,r+1), np.arange(-r,r+1))
        rr, _ = np.meshgrid(np.sqrt(u**2 + v**2), np.zeros(u.shape))
        se = rr<=r
        BW_curr = ndimage.binary_dilation(BW_ini, se)
        BW_fill = BW_curr.astype(int) - BW_prev.astype(int)
        idy, idx = np.where(BW_fill==1)

        while len(idy) > 0:
            # 8-neighbor averaging
            for k in range(len(idy)):
                count = 0
                nn_sum = 0

                for i in range(-1,2):
                    for j in range(-1,2):
                        if i != 0 or j != 0:
                            idi = idy[k]+i    # neighbor y id
                            idj = idx[k]+j    # neighbor x id
                            
                            if (0<idi and idi<=h and 0<idj and idj<=w and not np.isnan(Z_ext[idi-1, idj-1])):
                                count = count+1
                                nn_sum = nn_sum + Z_ext[idi-1, idj-1]
            
                if (count >=3):
                    Z_ext[idy[k]-1, idx[k]-1] = nn_sum/count
                    BW_fill[idy[k]-1, idx[k]-1] = 0
            
            idy, idx = np.where(BW_fill==1)
            
            BW_prev = BW_curr.copy()
            r = r + 1
            
            
    return X_ext, Y_ext, Z_ext, ca_range