# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:02:38 2023

@author: frw78547
"""

# import json
# import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import metrology as mt
import numpy as np
import pandas as pd
import pymurilo.File_Functions as pymf
import os

# from ibfanalysis import dtc

# from ibfanalysis import pvt

# from scipy.signal import convolve2d

import ibfanalysis.processing as process
import ibfanalysis.utilities as utils


def Reference_Point(include, base_path, x_min,x_max,y_min,y_max):

    # include = ["Proc_IBF_HDX_data_Gordo-B_230308_AB_clamped_0_data_stitched_formrmv.datx"]
    # base_path = r"C:/Users/frw78547/OneDrive - Diamond Light Source Ltd/Documents/IBF DATA/20230306_2D_2nd_iteration_result"
    # --------------------------
    
    hdx_path = pymf.List_Files(base_path, file_type=".datx", include=include, level=3)
    
    df = mt.read_mx(hdx_path[0], replace_na=True, convert_z=True, apply_coord=True, apply_lat_cal=True)
    
    # x = df.columns.to_numpy() * 1e3
    # y = df.index.to_numpy() * 1e3
    # z = df.to_numpy() * 1e9
    
    # --------------------------
    # BRF center or any special point
    # x_min = 103
    # x_max = 108
    # y_min = 6
    # y_max = 11
    # 
    # ------------------------------
    
    
    df = df.multiply(1e9)
    df.index = df.index * 1e3
    df.columns = df.columns * 1e3
    df.fillna(method="backfill")
    
    region = ((y_min, y_max), (x_min, x_max))
    df_crater = utils.get_df_region_2d(df, region)
    
    process.plot_fiducial_map(df, region)
    
    t = 60
    
    # --------------- 
    
    fit_result = process.fit_fiducial_2d(df_crater, t=t, show_report=True)
    vals = fit_result.get_dict()
    
    amp = np.abs(vals["amp"])
    sigma_x = np.abs(vals["sig_x"])
    sigma_y = np.abs(vals["sig_y"])
    brf = process.Brf2D(process.brf_gaussian_2d, amp, sigma_x, sigma_y)
    
    process.plot_fiducial_3d(df_crater, vals)
    
    return  fit_result

