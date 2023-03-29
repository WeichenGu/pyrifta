# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 00:51:29 2023

@author: frw78547
"""

import numpy as np

def BRFGaussian2D(X, Y, t, params):
    """
    2D Gaussian Beam removal function model:
    Z(X, Y) = tt*A*exp(-((X-u_x)^2/2*sigma_x^2+(Y-u_y)^2/2*sigma_y^2))
    """
    # Get the parameters
    A = params[0]
    sigmax = params[1][0]
    sigmay = params[1][1]
    # ux = params[3::2]
    # uy = params[4::2]
    ux = np.array(params[2::2])
    uy = np.array(params[3::2])
    # if np.isscalar(t):
    # t = np.array([t])

    # Ensure that X, Y are 3D arrays
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    # Feed the result
    Z_fitted = np.zeros_like(X)
    for i in range(len(str(t))):
        # Z_fitted[:, :, i] = A*t*np.exp(-((X[:, :, i]-ux[i])**2/(2*sigmax**2) + (Y[:, :, i]-uy[i])**2/(2*sigmay**2)))
        # print(i)
        Z_fitted += A*t*np.exp(-((X[:,:]-ux)**2/(2*sigmax**2) + (Y[:,:]-uy)**2/(2*sigmay**2)))

    return Z_fitted

