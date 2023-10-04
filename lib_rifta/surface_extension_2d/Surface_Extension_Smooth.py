# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 23:45:02 2023

@author: Etrr
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation
# from scipy.spatial.transform import Rotation as R
# from scipy.interpolate import RectBivariateSpline
# from scipy.interpolate import CloughTocher2DInterpolator as CT
# from scipy.ndimage import binary_dilation


def Surface_Extension_Smooth(X, Y, Z, tif_mpp, Z_tif):
    # 0. Obtain required parameters
    surf_mpp = np.median(np.diff(X[0, :]))

    m = Z.shape[0]  # CA height [pixel]
    n = Z.shape[1]  # CA width [pixel]

    m_ext = int(np.floor(tif_mpp * Z_tif.shape[0] * 0.5 / surf_mpp))
    n_ext = int(np.floor(tif_mpp * Z_tif.shape[1] * 0.5 / surf_mpp))
    r = max(m_ext, n_ext)

    ca_range = {
        'y_s': m_ext,
        'y_e': m_ext + m,
        'x_s': n_ext,
        'x_e': n_ext + n
    }

    # 1. Initial extension matrices
    X_ext, Y_ext = np.meshgrid(np.arange(-n_ext, n+n_ext), np.arange(-m_ext, m+m_ext))
    X_ext = X_ext * surf_mpp + X[0, 0]
    Y_ext = Y_ext * surf_mpp + Y[0, 0]

    Z_ext = np.full_like(X_ext, np.nan)
    Z_ext[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] = Z

    id_valid = ~np.isnan(Z)

    r = max(np.array(Z_ext.shape) - np.array(Z.shape)) // 2  # obtain the area of extension
    u, v = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
    
    coors = np.vstack((u.flatten(), v.flatten())).T
    rr = np.linalg.norm(coors,axis=1).reshape(u.shape)
    se = rr <= r
    BW_Z = binary_dilation(~np.isnan(Z_ext), structure=se)
    id_ext = BW_Z == 1


    # Interpolation
    # points = np.column_stack((X[id_valid].flatten(), Y[id_valid].flatten()))
    # values = Z[id_valid].flatten()
    # Z_ext = griddata(points, values, (X_ext, Y_ext), method='cubic', fill_value=0)



    # Estimate the values over the extended grid
    # Z_ext = spline(X_ext, Y_ext, grid=True)
    # spline = RectBivariateSpline(X.flatten(), Y.flatten(), Z.flatten(), bbox=[min(np.unique(Y_ext)), max(np.unique(Y_ext)), min(np.unique(X_ext)), max(np.unique(X_ext))], kx=8, ky=8)
    # Z_ext = spline.ev(Y_ext, X_ext).reshape(X_ext.shape)
    
    def extrapolated_spline_2D_new(x0,y0,z2d0):    
        from scipy.interpolate import RectBivariateSpline
        assert z2d0.shape == (y0.shape[0],x0.shape[0])
        spline = RectBivariateSpline(y0,x0,z2d0,kx=3,ky=3)


        def f(x,y,spline=spline):
    
            x = np.array(x,dtype='f4')
            y = np.array(y,dtype='f4') 
            assert x.shape == y.shape 
            ndim = x.ndim   
            # We want the output to have the same dimension as the input, 
            # and when ndim == 0 or 1, spline(x,y) is always 2D. 
            if   ndim == 0: result = spline.ev(x,y)[0][0]
            elif ndim == 1: 
                result = np.array([spline.ev(x[i],y[i])[0][0] for i in range(len(x))])
            else:           
                result = np.array([spline(x.flatten()[i],y.flatten()[i])[0][0] for i in range(len(x.flatten()))]).reshape(x.shape)         
            return result
        return f


    spline_2d = extrapolated_spline_2D_new(np.unique(X),np.unique(Y),Z)
    # Z_ext = spline_2d(np.unique(X_ext),np.unique(Y_ext))
    Z_ext = spline_2d(X_ext,Y_ext)
    



    return X_ext, Y_ext, Z_ext, ca_range
    print('5')
    m = Z.shape[0] 
    return X_ext, Y_ext, Z_ext, ca_range