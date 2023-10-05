# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 00:16:04 2023

@author: frw78547
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
# from scipy.optimize import basinhopping
# from functools import partial
# from scipy.signal import fftconvolve
from lib_rifta.ibf_engine.dwell_time_2d_fft_inverse_filter import dwell_time_2d_fft_inverse_filter
from lib_rifta.ibf_engine.conv_fft2 import conv_fft2

from lib_rifta.ibf_engine.pymoo_patternsearch import pymoo_minimize_entire
from lib_rifta.ibf_engine.pymoo_patternsearch import pymoo_minimize_dwell_grid

def dwell_time_2d_fft_optimize_gamma(gamma0, Z_to_remove, Z_to_remove_dw, B, dw_range, ca_range, flag, use_DCT):
    if flag == 'entire':
        # result = minimize_scalar(objective_func_entire, args=(Z_to_remove, B, dw_range, ca_range, use_DCT), method='bounded', bounds=(0, 2*gamma0))
        # result = minimize(objective_func_entire, gamma0, args=(Z_to_remove, B, dw_range, ca_range, use_DCT), method='L-BFGS-B',bounds=[(0, 2*gamma0)], options={'disp':False, 'maxiter': 500})
        # objfun = partial(objective_func_entire,
        #                  Z_to_remove = Z_to_remove, 
        #                  B = B, 
        #                  dw_range = dw_range,
        #                  ca_range = ca_range,
        #                  use_DCT = use_DCT)
        # result = basinhopping(objfun, gamma0, disp = False)
        result = pymoo_minimize_entire(gamma0, Z_to_remove, B, dw_range, ca_range, use_DCT, iter_show = False)
        gamma =result.x[0]
    elif flag == 'dwell':
        '''
        result = minimize_scalar(objective_func_dwell_grid, args=(Z_to_remove, Z_to_remove_dw, B, dw_range, ca_range, use_DCT), method='bounded', bounds=(0, 2*gamma0))
        # result = minimize(objective_func_dwell_grid, gamma0, args=(Z_to_remove, Z_to_remove_dw, B, dw_range, ca_range, use_DCT), method='L-BFGS-B',bounds=[(0, 2*gamma0)], options={'disp':False, 'maxiter': 500})
        # objfun = partial(objective_func_dwell_grid,
        #                  Z_to_remove = Z_to_remove, 
        #                  Z_to_remove_dw = Z_to_remove_dw,
        #                  B = B, 
        #                  dw_range = dw_range,
        #                  ca_range = ca_range,
        #                  use_DCT = use_DCT)
        # result = basinhopping(objfun, gamma0, disp = False)
        gamma = result.x[0]'''
        result = pymoo_minimize_dwell_grid(gamma0, Z_to_remove, Z_to_remove_dw, B, dw_range, ca_range, use_DCT, iter_show = False)
        gamma = result.X
    else:
        gamma = gamma0

    return gamma





def objective_func_entire(gamma, Z_to_remove, B, dw_range, ca_range, use_DCT):
    Tms = dwell_time_2d_fft_inverse_filter(Z_to_remove, B, gamma, use_DCT)

    T = np.zeros(Z_to_remove.shape)
    # T[dw_range['y_s']:dw_range['y_e']+1, dw_range['x_s']:dw_range['x_e']+1] = Tms[dw_range['y_s']:dw_range['y_e']+1, dw_range['x_s']:dw_range['x_e']+1]
    T[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']] = Tms[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]
    
    print('no')
    Z_removal = conv_fft2(T, B)
    # Z_removal = fftconvolve(T, B, mode='same')
    Z_residual_ca = Z_to_remove[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] - Z_removal[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]

    fGamma = np.nanstd(Z_residual_ca)

    return fGamma

def objective_func_dwell_grid(gamma, Z_to_remove, Z_to_remove_dw, B, dw_range, ca_range, use_DCT):
    ca_in_dw_y_s = ca_range['y_s'] - dw_range['y_s']
    ca_in_dw_x_s = ca_range['x_s'] - dw_range['x_s'] 
    ca_in_dw_y_e = ca_in_dw_y_s + ca_range['y_e'] - ca_range['y_s']
    ca_in_dw_x_e = ca_in_dw_x_s + ca_range['x_e'] - ca_range['x_s']

    T_dw = dwell_time_2d_fft_inverse_filter(Z_to_remove_dw, B, gamma, use_DCT)

    T = np.zeros(Z_to_remove.shape)
    # T[dw_range['y_s']:dw_range['y_e']+1, dw_range['x_s']:dw_range['x_e']+1] = T_dw
    T[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']] = T_dw

    Z_removal = conv_fft2(T, B)
    # Z_removal = fftconvolve(T, B, mode='same')
    Z_residual_ca = Z_to_remove_dw[ca_in_dw_y_s:ca_in_dw_y_e, ca_in_dw_x_s:ca_in_dw_x_e] - Z_removal[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]

    fGamma = np.nanstd(Z_residual_ca)

    return fGamma
