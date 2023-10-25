# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 07:30:26 2023

@author: Etrr
"""

import numpy as np
from numpy.polynomial.hermite import hermval, hermder

def Hermite_XYnm(X, Y, n, m):
    """
    Hermite polynomials in normalized X and Y coordinates with order vectors (n, m).

    Parameters:
        X, Y : np.ndarray
            Normalized X and Y coordinates.
        n, m : np.ndarray
            Order vectors.

    Returns:
        z3, zx3, zy3 : np.ndarray
            Hermite polynomials and their derivatives.
    """
    def Hermite_T1(X, n):
        """
        Get the 1D Hermite polynomial and its derivative.

        Parameters:
            X : np.ndarray
                Input coordinate.
            n : np.ndarray
                Order vector.

        Returns:
            T, Tx : np.ndarray
                1D Hermite polynomial and its derivative.
        """
        NUM = len(n)
        T = np.zeros((*X.shape, NUM))
        Tx = np.zeros_like(T)

        for num in range(NUM):
            c = [0]*n[num] + [1]
            T[:, :, num] = hermval(X, c)
            Tx[:, :, num] = hermval(X, hermder(c))

        return T, Tx

    # Get the 1D Hermite polynomial and its derivative.
    Tm, Tmy = Hermite_T1(Y, m)
    Tn, Tnx = Hermite_T1(X, n)

    # Convert 1D to 2D polynomials with considering the derivatives.
    z3 = Tn * Tm
    zx3 = Tnx * Tm
    zy3 = Tn * Tmy
   
    return z3, zx3, zy3
#%%
'''

# 2. Poly fit
p, q = np.meshgrid(np.arange(order_n + 1), np.arange(order_m + 1))
# X_nor = -1 + 2 * (X_ext - X_ext.min()) / (X_ext.max() - X_ext.min())
# Y_nor = -1 + 2 * (Y_ext - Y_ext.min()) / (Y_ext.max() - Y_ext.min())



z3, _, _ = Hermite_XYnm(X_ext, Y_ext, p.ravel(), q.ravel())

'''


