# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:09:51 2023

@author: frw78547
"""

import numpy as np

def Surface_Extension_Zeros(X, Y, Z, tif_mpp, Z_tif):
    # Sampling intervals
    surf_mpp = np.median(np.diff(X[0,:]))  # surface sampling interval [m/pxl]
    m, n = Z.shape  # CA height and width [pixels]

    m_ext = int(np.floor(tif_mpp*(Z_tif.shape[0])*0.5/surf_mpp))  # extension size in y [pixels]
    n_ext = int(np.floor(tif_mpp*(Z_tif.shape[1])*0.5/surf_mpp))  # extension size in x [pixels]

    ca_range = {}  # create a dictionary to store the CA range
    # ca_range['y_s'] = m_ext + 1  # y start id of CA in FA [pixels]
    # ca_range['y_e'] = ca_range['y_s'] + m - 1  # y end id of CA in FA [pixels]
    # ca_range['x_s'] = n_ext + 1  # x start id of CA in FA [pixels]
    # ca_range['x_e'] = ca_range['x_s'] + n - 1  # x end id of CA in FA [pixels]

    ca_range['y_s'] = m_ext  # y start id of CA in FA [pixels]
    ca_range['y_e'] = ca_range['y_s'] + m   # y end id of CA in FA [pixels]
    ca_range['x_s'] = n_ext  # x start id of CA in FA [pixels]
    ca_range['x_e'] = ca_range['x_s'] + n   # x end id of CA in FA [pixels]



    # Initial extension matrices
    X_ext, Y_ext = np.meshgrid(np.arange(-n_ext, n-1+n_ext+1), np.arange(-m_ext, m-1+m_ext+1))
    X_ext = X_ext * surf_mpp + X[0,0]  # adjust X grid add X[0,0]
    Y_ext = Y_ext * surf_mpp + Y[0,0]  # adjust Y grid add Y[0,0]

    # Extend the surface with 0
    Z_ext = np.zeros_like(X_ext)  # mark the Z_ext to 0
    Z_ext[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] = Z  # fill in the valid data points

    return X_ext, Y_ext, Z_ext, ca_range
