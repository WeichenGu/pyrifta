# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 08:14:42 2023

@author: Etrr
"""

import numpy as np
from numpy.polynomial.laguerre import lagval, lagder

def Laguerre_XYnm(X, Y, n, m):
    """
    Laguerre polynomials in normalized X and Y coordinates with order vectors (n, m).

    Parameters:
        X, Y : np.ndarray
            Normalized X and Y coordinates.
        n, m : np.ndarray
            Order vectors.

    Returns:
        z3, zx3, zy3 : np.ndarray
            Laguerre polynomials and their derivatives.
    """
    def Laguerre_T1(X, n):
        """
        Get the 1D Laguerre polynomial and its derivative.

        Parameters:
            X : np.ndarray
                Input coordinate.
            n : np.ndarray
                Order vector.

        Returns:
            T, Tx : np.ndarray
                1D Laguerre polynomial and its derivative.
        """
        NUM = len(n)
        T = np.zeros((*X.shape, NUM))
        Tx = np.zeros_like(T)

        for num in range(NUM):
            c = [0]*n[num] + [1]
            T[:, :, num] = lagval(X, c)
            Tx[:, :, num] = lagval(X, lagder(c))

        return T, Tx

    # Get the 1D Laguerre polynomial and its derivative.
    Tm, Tmy = Laguerre_T1(Y, m)
    Tn, Tnx = Laguerre_T1(X, n)

    # Convert 1D to 2D polynomials with considering the derivatives.
    z3 = Tn * Tm
    zx3 = Tnx * Tm
    zy3 = Tn * Tmy
   
    return z3, zx3, zy3
