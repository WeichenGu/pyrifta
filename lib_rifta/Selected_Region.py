# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:49:34 2023

@author: frw78547
"""

import numpy as np
import metrology as mt
import pymurilo.File_Functions as pymf

def Selected_Region(include, base_path,fitresult, x_offset_distance, y_offset_distance, x_ibf, y_ibf, selected_region_length,selected_region_width ):
    
    hdx_path = pymf.List_Files(base_path, file_type=".datx", include=include, level=3)
    
    df = mt.read_mx(hdx_path[0], replace_na=True, convert_z=True, apply_coord=True, apply_lat_cal=True)
    
    x = df.columns.to_numpy() * 1e3
    y = df.index.to_numpy() * 1e3
    z = df.to_numpy() * 1e9
     
    '''

    #############################################################################
    #                                  'ru'       x_offset_distance             #
    #                         ----------@         y_offset_distance        *    #
    #                         |        |                                        #
    #                         |        |                                        #
    #                  'ld'   ----------                                        #
    #                       selected_region_length                              #
    #                                                                           #
    #                                                                           #
    #############################################################################

    * : reference point
    @ : Pb
    x_offset_distance,y_offset_distance mean distance between * and @


    '''  
    Pb={}
    
    Pb['x_ru']= fitresult.cen_x - x_offset_distance 
    Pb['y_ru']= fitresult.cen_y - y_offset_distance 
    
    Pb['x_ld']= fitresult.cen_x - x_offset_distance - selected_region_length
    Pb['y_ld']= fitresult.cen_y - y_offset_distance - selected_region_width

    # find indices of Xs and Ys that satisfy the conditions
    acol = np.where((x >= Pb['x_ld']) & (x <= Pb['x_ru']))
    arow = np.where((y >= Pb['y_ld']) & (y <= Pb['y_ru']))
    # extract Z values for selected region
    z_selected_region = z[min(arow[0]):max(arow[0])+1, min(acol[0]):max(acol[0])+1]
    
    # shift X_selected_region and Y_selected_region by (-111.465-fitresult.xc) and (33.336-fitresult.yc)
    # X_selected_region = x[min(arow[0]):max(arow[0])+1, min(acol[0]):max(acol[0])+1] + (x_ibf - fitresult.cen_x)
    # Y_selected_region = y[min(arow[0]):max(arow[0])+1, min(acol[0]):max(acol[0])+1] + (y_ibf - fitresult.cen_y)

    x_selected_region,y_selected_region = np.meshgrid(
        x[min(acol[0]):max(acol[0])+1] + (x_ibf - fitresult.cen_x),
        y[min(arow[0]):max(arow[0])+1] + (y_ibf - fitresult.cen_y)
        )
    # x_selected_region_coors = x_ibf+(fitresult.cen_x - x_offset_distance)
    # y_selected_region_coors = y_ibf+(fitresult.cen_y - y_offset_distance)
    
    
    
    
    return x_selected_region/1e3,y_selected_region/1e3,z_selected_region/1e9