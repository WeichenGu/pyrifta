# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:10:23 2023

@author: frw78547
"""

import os
import json
import numpy as np

def Scanfile_Savejson(filename, Z, Y, dwell_time,
                      testname, data_from, calculation,
                      grid, dwell_time_unit, height_error_filter,
                      brf_params, x_start,y_start,
                      selected_region_length,selected_region_width,
                      fitresult):

    scan = {
        "name": testname,
        "original_height_error": data_from,
        "calculation": calculation,
        "grid": grid,
        # "grid_unit": grid_unit,
        "dwell_time_unit": dwell_time_unit,
        "filter": height_error_filter,
        "BRF": brf_params,
        "x_etching_region_startpoint": x_start,
        "y_etching_region_startpoint": y_start,
        "etching_region_length": selected_region_length,
        "etching_region_width": selected_region_width
        
    }
    if fitresult is not None:
        scan["HD_X_BRF_centre_point"]=[fitresult.cen_x, fitresult.cen_y]

    if np.max(np.unique(Z)) < 1:
        scan["Z"] = np.round((Z[0, :] * 1e3), 4).tolist()
        scan["Y"] = np.round((Y[:, 0] * 1e3), 4).tolist()
    else:
        scan["Z"] = np.round(Z[0, :], 4).tolist()
        scan["Y"] = np.round(Y[:, 0], 4).tolist()

    scan["dwell_time"] = dwell_time.tolist()

    scan_json = json.dumps(scan, indent=4)

    foldername = os.path.dirname(filename)
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    with open(f"{filename}.json", 'w+') as fid:
        fid.write(scan_json)
