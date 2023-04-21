# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:18:40 2023

@author: frw78547
"""

import numpy as np
from lib_rifta import ibf_engine
from scipy.ndimage import zoom
from scipy.interpolate import interp2d
from scipy.signal import fftconvolve
from lib_rifta import BRFGaussian2D
# from lib_rifta import ibf_engine
from lib_rifta.ibf_engine import DwellTime2D_FFT_IterativeFFT
from lib_rifta.ibf_engine import remove_surface1
from lib_rifta.ibf_engine import dwell_time_2d_assemble_c_d

def DwellTime2D_FFT_Full_Test(X, Y, Z_to_remove, Z_last_removal_dw, BRF_params, BRF_mode,
                               X_BRF, Y_BRF, Z_BRF, ca_range, pixel_m, tmin, tmax, options=None,
                               ratio=1, viewFullAperture=True):
    # 0. Set the default options for the function
    defaultOptions = {
        'Algorithm': 'Iterative-FFT',
        'maxIters': 10,
        'PV_dif': 0.001e-9,  # [m]
        'RMS_dif': 0.001e-9,  # [m]
        'dwellTime_dif': 60,  # [s]
        'isDownSampling': False
    }

    # 1. Deal with input arguments
    if options is None:
        options = defaultOptions
    if ratio is None:
        ratio = 1
    if viewFullAperture is None:
        viewFullAperture = True
        
    # 2. Construct the BRF using the BRF parameters
    # Release the BRF parameters
    A = BRF_params['A']  # peak removal rate [m/s]
    sigma_xy = BRF_params['sigma_xy']  # standard deviation [m]
    mu_xy = [0, 0] # center is 0 [m]
    brf_res = BRF_params['lat_res_brf']
    brf_pix = BRF_params['d_pix']
    
    brf_d = brf_res * brf_pix  # [m] range of Measurement_Area
    brf_r = brf_d * 0.5  # radius of the brf
    
    # X_B, Y_B = np.meshgrid(np.arange(-brf_r, brf_r + pixel_m, pixel_m),
    #                        np.arange(-brf_r, brf_r + pixel_m, pixel_m))
    # for brf_r = 0.002079050155, brf_r-pixel_m = 0.0020333567450000084>0.0020333567449992546 (Y_B[-1,0])

    # for matlab: brf_r = 0.002079050155000 Y_B(end) = 0.002033356745001, brf_r-pixel_m = 0.002033356745000<0.002033356745001 (Y_B(end))
    X_B, Y_B = np.meshgrid(np.arange(-brf_r, np.round(brf_r,12), pixel_m),
                           np.arange(-brf_r, np.round(brf_r,12), pixel_m))
    # Get B
    if BRF_mode.lower() == 'avg':
        B = interp2d(X_BRF, Y_BRF, Z_BRF, kind='cubic')(X_B, Y_B)
    else:
        B = BRFGaussian2D(X_B, Y_B, 1, [A,sigma_xy,mu_xy[0],mu_xy[1]])
    
    d_p = B.shape[0]  # diameter [pixel]
    r_p = int(np.floor(0.5 * d_p))  # radius [pixel]
    
    # reset the BRF params
    BRF_params['lat_res_brf'] = pixel_m
    BRF_params['d_pix'] = d_p
    
    # 3. Define the dwell grid
    # Get the size of the full aperture
    mM, nM = Z_to_remove.shape
    
    # Get the dwell grid pixel range
    dw_range = {'x_s': ca_range['x_s'] - r_p, 'x_e': ca_range['x_e'] + r_p,
                'y_s': ca_range['y_s'] - r_p, 'y_e': ca_range['y_e'] + r_p}
    
    # Determine if the range is valid
    if dw_range['x_s'] < 0 or dw_range['x_e'] > nM or dw_range['y_s'] < 0 or dw_range['y_e'] > mM:  # different to matlab, bcs python index from 0 and not include the end 
        raise ValueError('Invalid clear aperture range with [{}, {}] and [{}, {}]'.format(dw_range['x_s'], dw_range['x_e'], dw_range['y_s'], dw_range['y_e']))

    else:
        X_dw = X[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]
        Y_dw = Y[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]
        
        #% Clear aperture coordinates
        X_ca = X[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
        Y_ca = Y[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
        
    # % 4. Real FFT algorithm
    # % Iterative FFT on dwell grid        
    if options["Algorithm"] == "Iterative-FFT":
        maxIters = options["maxIters"]
        PV_dif = options["PV_dif"]
        RMS_dif = options["RMS_dif"]
        dwellTime_dif = options["dwellTime_dif"]
        
        _, Z_removal, Z_residual, T_P, Z_to_remove_dw, Z_removal_dw,Z_residual_dw,Z_to_remove_ca,Z_removal_ca, Z_residual_ca = DwellTime2D_FFT_IterativeFFT(Z_to_remove, B, dw_range, ca_range, maxIters, PV_dif, RMS_dif,dwellTime_dif)
    '''      
    elif options["Algorithm"] == "Iterative-FFT-Optimal-DwellTime":
        maxIters = options["maxIters"]
        PV_dif = options["PV_dif"]
        RMS_dif = options["RMS_dif"]
        dwellTime_dif = options["dwellTime_dif"]
        
        (_, Z_removal, Z_residual, dw_range, X_dw, Y_dw, T_P, Z_to_remove_dw,
         Z_removal_dw, Z_residual_dw,
         Z_to_remove_ca, Z_removal_ca,
         Z_residual_ca)= DwellTime2D_FFT_IterativeFFT_Optimal_DwellTime(Z_to_remove, X, Y, B, BRF_params, ca_range, maxIters, PV_dif, RMS_dif, dwellTime_dif)
        
    elif options["Algorithm"] == "FFT":
        _, Z_removal, Z_residual, T_P, Z_to_remove_dw, Z_removal_dw, Z_residual_dw, Z_to_remove_ca, Z_removal_ca, Z_residual_ca = DwellTime2D_FFT_Test(Z_to_remove, Z_last_removal_dw, B, dw_range, ca_range)
    else:
        raise ValueError("Invalid FFT algorithm chosen. Should be either Iterative-FFT or FFT")
    '''
    T_P = T_P * ratio
    
    # output
    # X_ext, Y_ext, Z_ext = np.meshgrid(ca_range, dw_range, indexing='ij')
    # options = {"Algorithm": options["Algorithm"], "maxIters": options["maxIters"], "PV_dif": PV_dif, "RMS_dif": RMS_dif, "dwellTime_dif": dwellTime_dif}
 
    
    
    # 5. Downsampling if used
    # Use a sparser dwell grid
    if options['isDownSampling']:
        # Obtain the sampling interval
        pixel_P_m = options['samplingInterval']
        interval_P_m = pixel_P_m / pixel_m
    
        # Down sample the dwell grid
        X_P = zoom(X_dw, 1 / interval_P_m)
        Y_P = zoom(Y_dw, 1 / interval_P_m)
    
        # Dump X_P, Y_P & X_P_F, Y_P_F dwell point positions into a 2D array as
        P = np.column_stack((X_P.ravel(), Y_P.ravel()))
    
        # Get the numbers of IBF machining points and sampling points of the surface error map R
        Nt = P.shape[0]
        Nr = Z_to_remove.size
    
        # Assemble the BRF matrix C, size(C) = Nr x Nt and vector d
        C, d, C_T = dwell_time_2d_assemble_c_d(Nr, Nt, BRF_params, Z_to_remove, X, Y, P, X_BRF, Y_BRF, Z_BRF, ca_range, BRF_mode)
    
        # Down sample T_dw
        T_P = zoom(T_P, 1 / interval_P_m, order=3) * interval_P_m**2
        T_P_Real = T_P + tmin * np.ceil(np.max(T_P) / tmax)
        T_P_v = T_P_Real.ravel()
    
        # Clear aperture results
        Z_removal_ca = C @ T_P_v
        Z_residual_ca = d - Z_removal_ca
        Z_removal_ca = Z_removal_ca.reshape(Z_to_remove_ca.shape)
        Z_residual_ca = Z_residual_ca.reshape(Z_to_remove_ca.shape)
        Z_to_remove_ca = d.reshape(Z_to_remove_ca.shape)
    
        # Detilt
        Z_to_remove_ca = remove_surface1(X_ca, Y_ca, Z_to_remove_ca)
        Z_to_remove_ca = Z_to_remove_ca - np.nanmin(Z_to_remove_ca)
        Z_removal_ca = remove_surface1(X_ca, Y_ca, Z_removal_ca)
        Z_removal_ca = Z_removal_ca - np.nanmin(Z_removal_ca)
        Z_residual_ca = remove_surface1(X_ca, Y_ca, Z_residual_ca)
    
        if viewFullAperture:
            # Full aperture results
            Z_removal = C_T @ T_P_v
            Z_residual = Z_to_remove.ravel() - Z_removal
            Z_residual = Z_residual - np.nanmean(Z_residual)
            Z_removal = Z_removal.reshape(Z_to_remove.shape)
            Z_residual = Z_residual.reshape(Z_to_remove.shape)
    
            # Dwell grid results
            Z_to_remove_dw = Z_to_remove[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]
            Z_removal_dw = Z_removal[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]
            Z_residual_dw = Z_residual[dw_range['y_s']:dw_range['y_e'], dw_range['x_s']:dw_range['x_e']]
        else:
            Z_removal = 0
            Z_residual = 0
    
            # Dwell grid
    else:
            X_P = X_dw;
            Y_P = Y_dw;   
            T_P_Real = T_P;
            return B, X_B, Y_B, Z_removal, Z_residual, T_P, T_P_Real, X_P, Y_P, X_dw, Y_dw, dw_range, Z_to_remove_dw, Z_removal_dw, Z_residual_dw, X_ca, Y_ca, Z_to_remove_ca, Z_removal_ca, Z_residual_ca        
         
    return B, X_B, Y_B, Z_removal, Z_residual, T_P, _, _, _, X_dw, Y_dw, dw_range, Z_to_remove_dw, Z_removal_dw, Z_residual_dw, X_ca, Y_ca, Z_to_remove_ca, Z_removal_ca, Z_residual_ca        

    
        
        
        
        