# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:36:13 2023

@author: frw78547
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem


from lib_rifta.ibf_engine.dwell_time_2d_fft_inverse_filter import dwell_time_2d_fft_inverse_filter
from lib_rifta.ibf_engine.conv_fft2 import conv_fft2

class Problem_func_dwell_grid(ElementwiseProblem):
    def __init__(self, gamma0, Z_to_remove, Z_to_remove_dw, B, dw_range, ca_range, use_DCT):
        super().__init__(n_var=1, n_obj=1, n_constr=0)
        #constraint xl=0, xu=2*gamma0
        self.gamma0 = gamma0
        self.Z_to_remove = Z_to_remove
        self.Z_to_remove_dw = Z_to_remove_dw
        self.B = B
        self.dw_range = dw_range
        self.ca_range = ca_range
        self.use_DCT = use_DCT

    def _evaluate(self, gamma0, out, *args, **kwargs):
        # gamma0 = x
        Z_to_remove = self.Z_to_remove
        Z_to_remove_dw = self.Z_to_remove_dw
        B = self.B
        dw_range = self.dw_range
        ca_range = self.ca_range
        use_DCT = self.use_DCT
        # print(gamma0)
        # The ca in dw range
        ca_in_dw_y_s = ca_range['y_s'] - dw_range['y_s']
        ca_in_dw_x_s = ca_range['x_s'] - dw_range['x_s']
        ca_in_dw_y_e = ca_in_dw_y_s + ca_range['y_e'] - ca_range['y_s']
        ca_in_dw_x_e = ca_in_dw_x_s + ca_range['x_e'] - ca_range['x_s']
        
        # Calculate T_dw for the dwell positions
        T_dw = dwell_time_2d_fft_inverse_filter(Z_to_remove_dw, B, gamma0, use_DCT)

        # Only keep T in the dwell grid and let others to be 0
        T = np.zeros(Z_to_remove.shape)
        T[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']] = T_dw
        # Calculate the height removal in the entire aperture
        Z_removal = conv_fft2(T, B)

        # Calculate the residual
        Z_residual_ca = Z_to_remove_dw[ca_in_dw_y_s:ca_in_dw_y_e, ca_in_dw_x_s:ca_in_dw_x_e] - Z_removal[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]

        # fGamma = np.nanstd(Z_residual_ca.ravel(), ddof=1)
        fGamma = np.nanstd(Z_residual_ca)
        # print(fGamma)
        out["F"] = fGamma


class Problem_func_entire(ElementwiseProblem):

    def __init__(self, gamma0, Z_to_remove, B, dw_range, ca_range, use_DCT):
        super().__init__(n_var=1, n_obj=1, n_constr=0)
        #constraint xl=0, xu=2*gamma0
        self.gamma0 = gamma0
        self.Z_to_remove = Z_to_remove
        self.B = B
        self.dw_range = dw_range
        self.ca_range = ca_range
        self.use_DCT = use_DCT
    def _evaluate(self, gamma0, out, *args, **kwargs):
        # gamma0 = x
        Z_to_remove = self.Z_to_remove
        Z_to_remove_dw = self.Z_to_remove_dw
        B = self.B
        dw_range = self.dw_range
        ca_range = self.ca_range
        use_DCT = self.use_DCT
    
        Tms = dwell_time_2d_fft_inverse_filter(Z_to_remove, B, gamma0, use_DCT)
    
        T = np.zeros(Z_to_remove.shape)
        T[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']] = Tms[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]
        # print('entire')
        Z_removal = conv_fft2(T, B)
        # Z_removal = fftconvolve(T, B, mode='same')
        Z_residual_ca = Z_to_remove[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] - Z_removal[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    
        fGamma = np.nanstd(Z_residual_ca)
    
        return fGamma