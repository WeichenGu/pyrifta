# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:06:15 2023

@author: frw78547
"""
# import sys

# sys.path.append('../lib/surface_extension_2d/')
import numpy as np
from scipy import ndimage, optimize
# from surface_extension_2d import Surface_Extension_8NN
# from lib.surface_extension_2d import Surface_Extension_8NN
# main.py


def Surface_Extension(X, Y, Z, brf_params, Z_tif, method='smooth', isFall=False,
                      fu_range=None, fv_range=None, order_m=None, order_n=None, poly_type=None):
    
    
    from .surface_extension_2d.Surface_Extension_Zero import Surface_Extension_Zeros
    from .surface_extension_2d.Surface_Extension_Gauss import Surface_Extension_Gauss
    from .surface_extension_2d.Surface_Extension_8NN import Surface_Extension_8NN
    from .surface_extension_2d.Surface_Extension_EdgeExtraction import Surface_Extension_EdgeExtraction
    from .surface_extension_2d.Surface_Extension_Fall import Surface_Extension_Fall
 
    
    
    # Default parameters
    if method is None:
        method = 'smooth'
        isFall = False
    if isFall is None:
        isFall = False
    
    # Different extension algorithms
    if method == 'zero':
        X_ext, Y_ext, Z_ext, ca_range = Surface_Extension_Zeros(X, Y, Z, brf_params['lat_res_brf'], Z_tif)
    elif method == '8nn':
        X_ext, Y_ext, Z_ext, ca_range = Surface_Extension_8NN(X, Y, Z, brf_params['lat_res_brf'], Z_tif)
    # elif method == 'smooth':
    #     X_ext, Y_ext, Z_ext, ca_range = Surface_Extension_Smooth(X, Y, Z, brf_params.lat_res_brf, Z_tif)
    elif method == 'gauss':
        X_ext, Y_ext, Z_ext, ca_range = Surface_Extension_Gauss(X, Y, Z, brf_params, Z_tif)
    # elif method == 'gerchberg':
    #     if fu_range is None or fv_range is None:
    #         raise ValueError("Not enough parameters for Gerchberg algorithm: fu_range and fv_range should be fed.")
    #     else:
    #         X_ext, Y_ext, Z_ext, ca_range = Surface_Extension_GP(X, Y, Z, brf_params.lat_res_brf, Z_tif, fu_range, fv_range)
    # elif method == 'poly':
    #     if order_m is None or order_n is None or poly_type is None:
    #         raise ValueError("Not enough parameters for polynomial fitting algorithm: order_m, order_n, and type should be fed.")
    #     else:
    #         X_ext, Y_ext, Z_ext, ca_range = Surface_Extension_Polyfit(X, Y, Z, brf_params.lat_res_brf, Z_tif, order_m, order_n, poly_type)
    else:
        raise ValueError("Invalid algorithm selected.")
    
    # Fall or not
    if isFall:
        if method not in ['zeros', 'gauss', 'poly']:
            Z_ext = Surface_Extension_Fall(X_ext, Y_ext, Z_ext, ca_range, brf_params, Z_tif)
        else:
            print('Fall profile is automatically disabled for {} algorithm'.format(method))
    
    return X_ext, Y_ext, Z_ext, ca_range



# def Surface_Extension_Zeros(X, Y, Z, tif_mpp, Z_tif):
#     # Sampling intervals
#     surf_mpp = np.median(np.diff(X[0,:]))  # surface sampling interval [m/pxl]
#     m, n = Z.shape  # CA height and width [pixels]

#     m_ext = int(np.floor(tif_mpp*(Z_tif.shape[0])*0.5/surf_mpp))  # extension size in y [pixels]
#     n_ext = int(np.floor(tif_mpp*(Z_tif.shape[1])*0.5/surf_mpp))  # extension size in x [pixels]

#     ca_range = {}  # create a dictionary to store the CA range
#     ca_range['y_s'] = m_ext + 1  # y start id of CA in FA [pixels]
#     ca_range['y_e'] = ca_range['y_s'] + m - 1  # y end id of CA in FA [pixels]
#     ca_range['x_s'] = n_ext + 1  # x start id of CA in FA [pixels]
#     ca_range['x_e'] = ca_range['x_s'] + n - 1  # x end id of CA in FA [pixels]

#     # Initial extension matrices
#     X_ext, Y_ext = np.meshgrid(np.arange(-n_ext, n-1+n_ext+1), np.arange(-m_ext, m-1+m_ext+1))
#     X_ext = X_ext * surf_mpp + X[0,0]  # adjust X grid add X[0,0]
#     Y_ext = Y_ext * surf_mpp + Y[0,0]  # adjust Y grid add Y[0,0]

#     # Extend the surface with 0
#     Z_ext = np.zeros_like(X_ext)  # mark the Z_ext to 0
#     Z_ext[ca_range['y_s']-1:ca_range['y_e'], ca_range['x_s']-1:ca_range['x_e']] = Z  # fill in the valid data points

#     return X_ext, Y_ext, Z_ext, ca_range



