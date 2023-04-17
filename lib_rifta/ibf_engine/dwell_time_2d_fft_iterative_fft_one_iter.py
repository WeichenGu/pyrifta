# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 00:00:56 2023

@author: frw78547
"""

import numpy as np
import scipy
from scipy.signal import convolve2d
from lib_rifta.ibf_engine.dwell_time_2d_fft_inverse_filter import dwell_time_2d_fft_inverse_filter
# import dwell_time_2d_fft_inverse_filter_test
from lib_rifta.ibf_engine.dwell_time_2d_fft_optimize_gamma import dwell_time_2d_fft_optimize_gamma
from lib_rifta.ibf_engine.remove_surface1 import remove_surface1
from lib_rifta.ibf_engine.conv_fft2 import conv_fft2


def dwell_time_2d_fft_iterative_fft_one_iter(Z_to_remove, B, hBias, dw_range, ca_range):

    ca_in_dw_y_s = ca_range['y_s'] - dw_range['y_s']
    ca_in_dw_x_s = ca_range['x_s'] - dw_range['x_s']
    ca_in_dw_y_e = ca_in_dw_y_s + ca_range['y_e'] - ca_range['y_s']
    ca_in_dw_x_e = ca_in_dw_x_s + ca_range['x_e'] - ca_range['x_s']

    Z_to_remove_ca = Z_to_remove[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]

    Z_to_remove = Z_to_remove - np.nanmin(Z_to_remove_ca)
    Z_to_remove_dw = Z_to_remove[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']] + hBias

    # You'll need to implement 'dwell_time_2d_fft_inverse_filter' function here
    T_dw = dwell_time_2d_fft_inverse_filter(Z_to_remove_dw, B, 1, False)

    T = np.zeros_like(Z_to_remove)
    T[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']] = T_dw
    
    # if the spacing of surface and BRF is different, try convolve2d.
    # Z_removal = convolve2d(T, B, mode='same')
    # Z_removal = scipy.signal.fftconvolve(T, B, mode='same')
    
    Z_removal = conv_fft2(T,B)
    
    Z_removal_ca = Z_removal[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    # bcs the difference between index of python and matlab
    Z_to_remove_ca = Z_to_remove_dw[ca_in_dw_y_s-1:ca_in_dw_y_e-1, ca_in_dw_x_s-1:ca_in_dw_x_e-1]

    #Get gamma0
    gamma0 = np.nanstd(Z_to_remove_ca) / np.nanstd(Z_removal_ca)

    #Get optimized gamma
    gamma = dwell_time_2d_fft_optimize_gamma(gamma0, Z_to_remove, Z_to_remove_dw, B, dw_range, ca_range, 'dwell', False)

    T_dw = dwell_time_2d_fft_inverse_filter(Z_to_remove_dw, B, gamma, False)

    T = np.zeros_like(Z_to_remove)
    T[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']] = T_dw
    
    # if the spacing of surface and BRF is different, try convolve2d.
    # Z_removal = convolve2d(T, B, mode='same')
    # Z_removal = scipy.signal.fftconvolve(T, B, mode='same')
    Z_removal = conv_fft2(T,B)
    Z_to_remove_ca = Z_to_remove_dw[ca_in_dw_y_s-1:ca_in_dw_y_e-1, ca_in_dw_x_s-1:ca_in_dw_x_e-1]

    # Obtain the entire aperture result
    Z_residual = Z_to_remove - Z_removal
    # Obtain the dwell grid result
    Z_removal_dw = Z_removal[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]
    Z_residual_dw = Z_residual[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]
    # Obtain the clear aperture results
    Z_removal_ca = Z_removal[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca_wo_detilt = Z_residual[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]


    # De-tilt
    x_ca, y_ca = np.meshgrid(range(Z_to_remove_ca.shape[1]), range(Z_to_remove_ca.shape[0]))
    Z_to_remove_ca = remove_surface1(x_ca, y_ca, Z_to_remove_ca)
    # print(Z_to_remove_ca)
    Z_to_remove_ca = Z_to_remove_ca - np.nanmin(Z_to_remove_ca)
    Z_removal_ca = remove_surface1(x_ca, y_ca, Z_removal_ca)
    Z_removal_ca = Z_removal_ca - np.nanmin(Z_removal_ca)
    Z_residual_ca = remove_surface1(x_ca, y_ca, Z_residual_ca_wo_detilt)


    return T,Z_removal,Z_residual,T_dw,Z_to_remove_dw,Z_removal_dw,Z_residual_dw,Z_to_remove_ca,Z_removal_ca,Z_residual_ca,Z_residual_ca_wo_detilt





    # return {
    #     'T': T,
    #     'Z_removal': Z_removal,
    #     'Z_residual': Z_residual,
    #     'T_dw': T_dw,
    #     'Z_to_remove_dw': Z_to_remove_dw,
    #     'Z_removal_dw': Z_removal_dw,
    #     'Z_residual_dw': Z_residual_dw,
    #     'Z_to_remove_ca': Z_to_remove_ca,
    #     'Z_removal_ca': Z_removal_ca,
    #     'Z_residual_ca': Z_residual_ca,
    #     'Z_residual_ca_wo_detilt': Z_residual_ca_wo_detilt
    # }
                                                                                    

                                                                                                   

                                                                                                   
