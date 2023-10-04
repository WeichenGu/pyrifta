# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 22:49:16 2023

@author: Etrr
"""

import numpy as np
from scipy.special import eval_legendre

def Legendre_XYnm(X, Y, n, m):
    """
    Compute the 2D Orthogonal Legendre polynomials and its derivatives
    with order vector (m, n).
    
    Parameters:
        X, Y : np.ndarray
            Normalized X and Y coordinates.
        n, m : np.ndarray
            Order vectors.
    
    Returns:
        Q, Qx, Qy : np.ndarray
            Legendre polynomials and their derivatives.
    """

    def Legendre_T1(X, n):
        """
        Compute the Legendre polynomial and its derivative.

        Parameters:
            X : np.ndarray
                Input coordinate.
            n : np.ndarray
                Order vector.

        Returns:
            L, Lx : np.ndarray
                Legendre polynomial and its derivative.
        """
        NUM = len(n)
        L = np.zeros((*X.shape, NUM))
        Lx = np.zeros_like(L)

        for num in range(NUM):
            L[:, :, num] = eval_legendre(n[num], X)

            if n[num] == 0:
                Lx[:, :, num] = 0
            else:
                L_previous = eval_legendre(n[num]-1, X)
                Lx[:, :, num] = n[num] / (1 - X**2) * (L_previous - X * L[:, :, num])

        return L, Lx
    
    # Compute the 1D Legendre polynomial and its derivative for X and Y
    Lm, Lmx = Legendre_T1(X, m)
    Ln, Lny = Legendre_T1(Y, n)

    # Convert 1D to 2D polynomials with considering the derivatives
    Q = Lm * Ln
    Qx = Lmx * Ln
    Qy = Lm * Lny

    return Q, Qx, Qy
