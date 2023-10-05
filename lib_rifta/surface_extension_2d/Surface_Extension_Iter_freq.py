# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:06:30 2023

@author: Etrr
"""

import numpy as np

from scipy.ndimage import binary_dilation

from lib_rifta import Surface_Extension
from lib_rifta.DwellTime2D_FFT_Full_Test import DwellTime2D_FFT_Full_Test
from lib_rifta.ibf_engine.remove_surface1 import remove_surface1
from lib_rifta.ibf_engine.conv_fft2 import conv_fft2
# from lib_rifta.surface_extension_2d import Chebyshev_XYnm
# from lib_rifta.surface_extension_2d import Legendre_XYnm
from lib_rifta.surface_extension_2d.frequency_separate_dct import frequency_separate_dct


def Surface_Extension_Iter_freq(
    X, Y, Z,  # unextended surface error map
    brf_params,  # TIF sampling interval [m/pxl]
    brf_mode,
    du_ibf,
    rms_thrd,
    X_tif, Y_tif, Z_tif,  # TIF profile
    init_ext_method,  # initial extention method
    ini_is_fall,
    ini_fu_range, ini_fv_range,  # for gerchberg initial extention method
    ini_order_m, ini_order_n, ini_type,  # for poly initial extension method
    iter_ext_method,  # has initial extension or not
    iter_is_fall,
    iter_fu_range, iter_fv_range,  # for gerchberg initial extention method
    iter_order_m, iter_order_n, iter_type,  # for poly initial extension method
    cutoff_freq
    ):

    Z_ini = Z.copy()

    _, Z_low_freq_data, Z_high_freq_data = frequency_separate_dct(
        X*1e3, Y*1e3, Z*1e9, 1e3*(X[1,2]-X[1,1]),cutoff_freq,'yes');
    
    Z_low_freq_data = Z_low_freq_data / 1e9
    Z_high_freq_data = Z_high_freq_data / 1e9
    

    Z = Z_low_freq_data.copy()
    
    
    # Initial extension
    X_ext, Y_ext, Z_ext, ca_range = Surface_Extension(
        X, Y, Z,
        brf_params.copy(),
        Z_tif,
        init_ext_method,
        ini_is_fall,
        ini_fu_range, ini_fv_range,
        ini_order_m, ini_order_n, ini_type)

    # Calculate dwell time
    TT = 0
    pixel_m = np.median(np.diff(X[0, :]))

    # options structure can be converted to a dictionary in Python
    options = {
        'Algorithm': 'Iterative-FFT',
        'maxIters': 50,
        'PV_dif': 0.01e-9,
        'RMS_dif': 0.02e-9,
        'dwellTime_dif': 60,
        'isDownSampling': False,
        'samplingInterval': du_ibf
    }

    ratio = 1
    tmin = 0
    tmax = 1

 
    _, _, _, _, _, T_Per, _, _, _, _, _, _, _, Z_removal_dw, _, _, _, _, _, Z_residual_ca =\
        DwellTime2D_FFT_Full_Test(
        X_ext, Y_ext, Z_ext, 0,
        brf_params.copy(), brf_mode,
        X_tif, Y_tif, Z_tif,
        ca_range, pixel_m,
        tmin, tmax,
        options, ratio, False
    )

    TT += T_Per

    # Iterative refinement
    Z_residual_ca_prev = Z_residual_ca.copy()
    mau_iter = 20
    iter_time = 1
    loop_over = True
    while loop_over:

        if np.std(Z_residual_ca) < 3e-10:
            X_ext, Y_ext, Z_ext, ca_range = Surface_Extension(
                X, Y, Z_residual_ca,
                brf_params,
                Z_tif,
                '8nn',
                False)
        else:
            X_ext, Y_ext, Z_ext, ca_range = Surface_Extension(
                X, Y, Z_residual_ca,
                brf_params,
                Z_tif,
                iter_ext_method,
                iter_is_fall,
                iter_fu_range, iter_fv_range,
                iter_order_m, iter_order_n, iter_type)


        du_ibf = 1e-3  # [m] ibf dwell grid sampling interval
        
        # dwell time calculation
        options = {
            'Algorithm': 'FFT',  # or 'Iterative-FFT'
            'maxIters': 10,
            'PV_dif': 0.001e-9,  # [m]
            'RMS_dif': 0.02e-9,  # [m]
            'dwellTime_dif': 60,  # [s]
            'isDownSampling': False,
            'samplingInterval': du_ibf  # [m]
        }
        
        
        ratio = 0.75
        tmin = 0
        tmax = 1
        
        (B,_, _,  # BRF Coordinates
         _, _,  # full aperture results [m]
         T_Per,  # dwell time on the dwell grid [s]
         _,
         _,_,  # dwell grid 
         _, _,_,  # dwell grid coordinates [m]
         _, Z_removal_dw, _,  # dwell grid results [m]
         X_ca, Y_ca,  # clear aperture coordinates [m]
         _, _, Z_residual_ca  # [m]
        ) = DwellTime2D_FFT_Full_Test(
            X_ext, Y_ext, Z_ext,  # height to remove [m]
            Z_removal_dw,
            brf_params.copy(),  # BRF parameters
            brf_mode,
            X_tif, Y_tif, Z_tif,
            ca_range,  # Clear aperture range [pixel] 
            pixel_m, # pixel size [m/pixel]
            tmin, tmax,
            options,
            ratio,
            False
        )
        
        std_curr = np.std(Z_residual_ca)  # using numpy for standard deviation
        
        if std_curr > np.std(Z_residual_ca_prev):
            loop_over = False
            break
        elif std_curr < rms_thrd:
            TT += T_Per
            loop_over = False
            break
        elif iter_time > mau_iter:
            loop_over = False
            break
        else:
            # changed 20231004
            # Z_residual_ca_loop = Z_residual_ca + Z_high_freq_data
            Z_residual_ca_prev = Z_residual_ca
            TT += T_Per
            iter_time += 1


    Z_ext = np.nan * np.ones_like(X_ext)  # Create an array of NaNs with the same shape as X_ext
    
    Z_ext[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] = Z  # Fill in the valid data points
    
    r = np.max(np.array(Z_ext.shape) - np.array(Z.shape)) // 2  
    
    


 # any-nn down
    h, w = TT.shape
    u, v = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
    coors = np.vstack((u.flatten(), v.flatten())).T
    rr = np.linalg.norm(coors,axis=1).reshape(u.shape)
    se = rr <= r
    BW_Z = binary_dilation(~np.isnan(Z_ext), structure=se)
    id_ext = BW_Z == 0
    
    TT[id_ext] = np.nan  # Reset the ext area (T)
    TT = TT - np.nanmin(TT)
    
    BW_ini = ~np.isnan(TT)  # START POINT
    BW_prev = BW_ini.copy()
    
    while np.any(np.isnan(TT)):
        
        BW_curr = binary_dilation(BW_ini, structure=se)
        BW_fill = BW_curr.astype(int) - BW_prev.astype(int)
        idy, idx = np.where(BW_fill == 1)
        
        while len(idy) > 0:
            for k in range(len(idy)):
                count = 0
                nn_sum = 0
                
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        if not (i == 0 and j == 0):
                            idi = idy[k] + i  # neighbor y id
                            idj = idx[k] + j  # neighbor x id
                            
                            if 0 < idi <= h-1 and 0 < idj <= w-1 and ~np.isnan(TT[idi, idj]):
                                count += 1
                                nn_sum += TT[idi, idj]
                                
                if count >= 1:
                    TT[idy[k], idx[k]] = nn_sum / count
                    BW_fill[idy[k], idx[k]] = 0
    
            idy, idx = np.where(BW_fill == 1)
    
        BW_prev = BW_curr.copy()
        r = r + 1
    
    TT[np.isnan(TT)] = 0
    TT = TT - np.min(TT)
    
    Z_removal_dw = conv_fft2(TT, B)
    

    Z_residual_ca = Z_ini - Z_removal_dw[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca = remove_surface1(X_ca, Y_ca, Z_residual_ca) 
    
    return TT, B, Z_ini, Z_residual_ca, Z_removal_dw
    







