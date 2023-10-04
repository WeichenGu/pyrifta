# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:48:58 2023

@author: frw78547
"""

import numpy as np
# import scipy.optimize
# import time
# from lib_rifta.ibf_engine import DwellTime2D_FFT_IterativeFFT
from lib_rifta.ibf_engine.remove_surface1 import remove_surface1
from lib_rifta.ibf_engine.conv_fft2 import conv_fft2
from lib_rifta.ibf_engine.dwell_time_2d_fft_inverse_filter_test import dwell_time_2d_fft_inverse_filter_test
from lib_rifta.ibf_engine.dwell_time_2d_fft_optimize_gamma import dwell_time_2d_fft_optimize_gamma


def DwellTime2D_FFT_Test(Z_to_remove, Z_last_removal_dw, B, dw_range, ca_range):


    Z_to_remove_dw = Z_to_remove[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]
    Z_to_remove_dw = Z_to_remove_dw - np.nanmin(Z_to_remove_dw + Z_last_removal_dw)

    T_dw = dwell_time_2d_fft_inverse_filter_test(Z_to_remove_dw, B, 1, False)

    # Initialize T
    T = np.zeros_like(Z_to_remove)
    T[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']] = T_dw

    # Calculate the height removal in the entire aperture
    Z_removal = conv_fft2(T, B)

    Z_to_remove_ca = Z_to_remove[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_removal_ca = Z_removal[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    
    # Get gamma0
    gamma0 = np.nanstd(Z_to_remove_ca) / np.nanstd(Z_removal_ca)
    
    # Optimize gamma
    # start_time = time.time()
    
    gamma = dwell_time_2d_fft_optimize_gamma(gamma0, Z_to_remove, Z_to_remove_dw, B, dw_range, ca_range, 'dwell', True)
    # Define the optimization function separately.
    
    # end_time = time.time()
    # print("Time taken for optimization:", end_time - start_time)
    
    print([gamma0, gamma])
     
    # 2. Use the optimal gamma to do the computation again
    T_dw = dwell_time_2d_fft_inverse_filter_test(Z_to_remove_dw, B, gamma, False)

    # Initialize T with zeros and set the computed T_dw in the dwell grid region
    T = np.zeros_like(Z_to_remove)
    T[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']] = T_dw
    
    # Calculate the height removal in the entire aperture

    Z_removal = conv_fft2(T, B)
    
    # Obtain the height to remove and height removal in the clear aperture
    Z_to_remove_ca = Z_to_remove[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_to_remove_dw = Z_to_remove[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]
    
    T_dw = T[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]


    Z_residual = Z_to_remove - Z_removal

    # Obtain the dwell grid result

    Z_removal_dw = Z_removal[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]
    Z_residual_dw = Z_residual[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]

    # Obtain the clear aperture results

    Z_removal_ca = Z_removal[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca = Z_residual[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    
    # De-tilt

    u_ca, v_ca = np.meshgrid(np.arange(Z_to_remove_ca.shape[1]), np.arange(Z_to_remove_ca.shape[0]))
    Z_to_remove_ca = remove_surface1(u_ca, v_ca, Z_to_remove_ca) - np.nanmin(Z_to_remove_ca)
    Z_removal_ca = remove_surface1(u_ca, v_ca, Z_removal_ca) - np.nanmin(Z_removal_ca)
    Z_residual_ca = remove_surface1(u_ca, v_ca, Z_residual_ca)


    return T, Z_removal, Z_residual, T_dw, Z_to_remove_dw, Z_removal_dw, Z_residual_dw, Z_to_remove_ca, Z_removal_ca, Z_residual_ca

