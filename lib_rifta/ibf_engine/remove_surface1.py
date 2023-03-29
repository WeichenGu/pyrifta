# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 00:29:13 2023

@author: frw78547
"""

import numpy as np

def remove_surface1(X, Y, Z):
    idx = np.isfinite(Z)

    z = Z[idx]
    x = X[idx]
    y = Y[idx]

    H = np.vstack((np.ones_like(x), x, y)).T

    f = np.linalg.lstsq(H, z, rcond=None)[0]

    Zf = f[0] + f[1] * X + f[2] * Y

    Zres = Z - Zf

    return Zres
