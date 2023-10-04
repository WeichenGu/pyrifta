# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 22:42:49 2023

@author: Etrr
"""

import numpy as np

def Chebyshev_XYnm(X, Y, n, m):
    """
    Chebyshev polynominals in normalized X and Y coordinates with order vectors (n, m).

    Parameters:
        X, Y : np.ndarray
            Normalized X and Y coordinates.
        n, m : np.ndarray
            Order vectors.

    Returns:
        z3, zx3, zy3 : np.ndarray
            Chebyshev polynominals and their derivatives.
    """
    def Chebyshev_T1(X, n):
        """
        Get the T type 1D Chebyshev polynominal and its derivative.

        Parameters:
            X : np.ndarray
                Input coordinate.
            n : np.ndarray
                Order vector.

        Returns:
            T, Tx : np.ndarray
                T type 1D Chebyshev polynominal and its derivative.
        """
        NUM = len(n)
        T = np.zeros((*X.shape, NUM))
        Tx = np.zeros_like(T)

        for num in range(NUM):
            T[:, :, num] = np.cos(n[num] * np.arccos(X))

            Tx_temp = n[num] * np.sin(n[num] * np.arccos(X)) / np.sin(np.arccos(X))
            Tx_temp[np.sin(np.arccos(X)) == 0] = n[num] ** 2
            Tx[:, :, num] = Tx_temp

        return T, Tx

    # Get the T type 1D Chebyshev polynominal and its derivative.
    Tm, Tmy = Chebyshev_T1(Y, m)
    Tn, Tnx = Chebyshev_T1(X, n)

    # Convert 1D to 2D polynominals with considering the derivatives.
    z3 = Tn * Tm
    zx3 = Tnx * Tm
    zy3 = Tn * Tmy

    return z3, zx3, zy3