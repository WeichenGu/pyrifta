# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:36:13 2023

@author: frw78547
"""

import numpy as np
import lib_rifta
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem

from lib_rifta import Surface_Extension
from lib_rifta.ibf_engine.dwell_time_2d_fft_inverse_filter import dwell_time_2d_fft_inverse_filter
from lib_rifta.ibf_engine.conv_fft2 import conv_fft2
from lib_rifta.surface_extension_2d.frequency_separate_dct import frequency_separate_dct
# from lib_rifta.DwellTime2D_FFT_Full_Test import DwellTime2D_FFT_Full_Test

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
    
    
    

class Problem_func_cutoff_Freq(ElementwiseProblem):

    def __init__(self, cutoff_freq, X, Y, Z, brf_params, Z_tif, init_ext_method, ini_is_fall, ini_fu_range, ini_fv_range, ini_order_m, ini_order_n, ini_type, du_ibf, brf_mode, X_tif, Y_tif):
        

        super().__init__(n_var=1, n_obj=1, n_constr=2)  # Assuming there are no constraints
        self.cutoff_freq = cutoff_freq
        self.X = X
        self.Y = Y
        self.Z = Z
        self.brf_params = brf_params
        self.Z_tif = Z_tif
        self.init_ext_method = init_ext_method
        self.ini_is_fall = ini_is_fall
        self.ini_fu_range = ini_fu_range
        self.ini_fv_range = ini_fv_range
        self.ini_order_m = ini_order_m
        self.ini_order_n = ini_order_n
        self.ini_type = ini_type
        self.du_ibf = du_ibf
        self.brf_mode = brf_mode
        self.X_tif = X_tif
        self.Y_tif = Y_tif

    def _evaluate(self, cutoff_freq, out, *args, **kwargs):
        
        constraint1 = 0.5 - cutoff_freq 
        constraint2 = cutoff_freq - 3

        
        Z_ini = self.Z.copy()

        _, Z_low_freq_data, Z_high_freq_data = frequency_separate_dct(
            self.X*1e3, self.Y*1e3, self.Z*1e9, 1e3*(self.X[1,2]-self.X[1,1]), cutoff_freq,'no')

        Z_low_freq_data = Z_low_freq_data / 1e9
        Z_high_freq_data = Z_high_freq_data / 1e9

        # self.Z = Z_low_freq_data.copy()

        self.X_ext, self.Y_ext, Z_ext, self.ca_range = Surface_Extension(
            self.X, self.Y, Z_low_freq_data.copy(),
            self.brf_params.copy(),
            self.Z_tif,
            self.init_ext_method,
            self.ini_is_fall,
            self.ini_fu_range, self.ini_fv_range,
            self.ini_order_m, self.ini_order_n, self.ini_type)

        self.ratio = 1
        self.tmin = 0
        self.tmax = 1
        self.pixel_m = np.median(np.diff(self.X[0, :]))
        self.options = {
            'Algorithm': 'Iterative-FFT',
            'maxIters': 50,
            'PV_dif': 0.01e-9,
            'RMS_dif': 0.02e-9,
            'dwellTime_dif': 60,
            'isDownSampling': False,
            'samplingInterval': self.du_ibf
        }
        
        _, _, _, _, _, T_Per, _, _, _, _, _, _, _, Z_removal_dw, _, _, _, _, _, Z_residual_ca = \
            lib_rifta.DwellTime2D_FFT_Full_Test(
            self.X_ext, self.Y_ext, Z_ext, 0,
            self.brf_params.copy(), self.brf_mode,
            self.X_tif, self.Y_tif, self.Z_tif,
            self.ca_range, self.pixel_m,
            self.tmin, self.tmax,
            self.options, self.ratio, False
        )
            
        self.Z_low_freq_data = Z_low_freq_data

        f_value = np.std(Z_ini - Z_removal_dw[self.ca_range['y_s']:self.ca_range['y_e'], self.ca_range['x_s']:self.ca_range['x_e']])  # Objective function value
        
        # out["X"] = cutoff_freq
        out["F"] = f_value
        out["G"] = np.array([constraint1, constraint2])
        
        return f_value



    
    