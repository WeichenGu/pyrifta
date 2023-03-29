# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 00:47:16 2023

@author: frw78547
"""

import numpy as np
from lib_rifta import ibf_engine
from lib_rifta.ibf_engine import *
from lib_rifta.ibf_engine.dwell_time_2d_fft_iterative_fft_one_iter import dwell_time_2d_fft_iterative_fft_one_iter
# from lib_rifta.ibf_engine import dwell_time_2d_fft_iterative_fft_one_iter

def DwellTime2D_FFT_IterativeFFT(Z_to_remove, B, dw_range, ca_range, maxIters, PV_dif, RMS_dif, dwellTime_dif):
    # This function implement the Iterative DCT algorithm for dwell time calculation
    # print(Z_to_remove[1,1])
    # Iteration 0
    T, Z_removal, Z_residual, T_dw, Z_to_remove_dw, Z_removal_dw, Z_residual_dw, Z_to_remove_ca, Z_removal_ca, Z_residual_ca, Z_residual_ca_woDetilt = dwell_time_2d_fft_iterative_fft_one_iter(Z_to_remove, B, 0, dw_range, ca_range)

    Z_residual_ca_pre = Z_residual_ca
    PV_pre = smart_ptv(Z_residual_ca.ravel())
    dwellTime_pre = np.sum(T_dw)
    num_iter = 1

    # Iterations 1 to maxIters
    while True:
        # Calculation of new dwell time
        T, Z_removal, Z_residual, T_dw, Z_to_remove_dw, Z_removal_dw, Z_residual_dw, Z_to_remove_ca, Z_removal_ca, Z_residual_ca, Z_residual_ca_woDetilt = dwell_time_2d_fft_iterative_fft_one_iter(Z_to_remove, B, -np.min(Z_residual_ca_woDetilt), dw_range, ca_range)
        print(np.size(T_dw,1))
        # Calculate stopping threshold
        PV_cur = smart_ptv(Z_residual_ca.ravel())
        dwellTime_cur = np.sum(T_dw)

        if num_iter > maxIters:
            print('Maximum number of iterations reached.')
            break
        elif np.nanstd(Z_residual_ca - Z_residual_ca_pre) < RMS_dif:
            print('RMS difference limit reached.')
            break
        elif abs(dwellTime_pre - dwellTime_cur) < dwellTime_dif:
            print('Dwell time difference limit reached.')
            break
        else:
            Z_residual_ca_pre = Z_residual_ca
            PV_pre = PV_cur
            dwellTime_pre = dwellTime_cur
            num_iter += 1
            print(num_iter)

    return T, Z_removal, Z_residual, T_dw, Z_to_remove_dw, Z_removal_dw, Z_residual_dw, Z_to_remove_ca, Z_removal_ca, Z_residual_ca



def smart_ptv(x):
    """
    Compute the PV value of an input array without considering any NaNs.
    
    Parameters:
        x (array_like): Input array.
    
    Returns:
        smartPTV (float): PV value of x.
    """
    smartPTV = np.ptp(x[np.isfinite(x)])
    return smartPTV
