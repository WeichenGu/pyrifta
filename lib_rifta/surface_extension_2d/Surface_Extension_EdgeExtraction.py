# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 00:50:53 2023

@author: frw78547
"""

import numpy as np

def Surface_Extension_EdgeExtraction(Z_ext):
    """
    Function:
        id_edge = Surface_Extension_EdgeExtraction(Z_ext)
    Purpose:
        Find edge points in the Z_ext, which contains NaN as the extension area
    Inputs:
        Z_ext: initial extended surface with NaNs
    Outputs:
        id_edge: ids with the same size as Z_ext, with edge points = 1 and the
        other positions = 0
    """
    id_edge = ~np.isnan(Z_ext)
    idy, idx = np.where(id_edge==1)
    h, w = Z_ext.shape

    for k in range(len(idy)):
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i!=0 or j!=0:
                    idi = idy[k]+i    # neighbor y id
                    idj = idx[k]+j    # neighbor x id
                    if (0<idi<=h and 0<idj<=w):
                        if np.isnan(Z_ext[idi, idj]):
                            count += 1
        if count==0:
            id_edge[idy[k], idx[k]] = 0
    return id_edge
