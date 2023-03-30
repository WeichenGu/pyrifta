# -*- coding: utf-8 -*-
"""
generate_pvt_from_json
loaded json file only data time

"""
import os
import json
import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from ibfanalysis import pvt
from ibfanalysis import utilities as utils

def Generate_pvt_from_json(json_path,data_filename,run_in,n_points):
# json_path = r"//dc/dls/science/groups/optics_and_metrology/ibf/pvt/Gordo-B/gordo-B_P003/2023-03-30/Gordo-B_y-spacingx0"

    
    path = os.path.join(json_path, data_filename)
    
    f = open(path, "r")
    data = json.load(f)
    
    dwell_z = np.asarray(data["Z"])
    dwell_y = np.asarray(data["Y"])
    dwell_time = np.asarray(data["dwell_time"])
    t_dwell = np.sum(dwell_time)
    
    print("total time = {:.3f} s".format(t_dwell))
    print("total time = {:d} m {:.2f} s".format(int(t_dwell/60), 60 * t_dwell % 60))
    print("total time = {:d} h {:.2f} m".format(int(t_dwell/3600), (t_dwell % 3600) / 60))
    
    extent = [dwell_z[0], dwell_z[-1], dwell_y[-1], dwell_y[0]]
    
    plt.figure(figsize=(10, 10), dpi=150)
    plt.imshow(dwell_time, extent=extent)
    plt.show()
    
    df_dwell = pd.DataFrame(dwell_time, columns=dwell_z, index=dwell_y)
    xyz = utils.df_to_3col(df_dwell)
    xx, yy = np.meshgrid(dwell_z, dwell_y)
    
    fig = plt.figure(figsize=(10,10), dpi=150)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(xx, yy, dwell_time, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()
    
    
    # run_in = 60
    # n_points = 101
    
    flip_y = True
    flip_z = True
    
    if flip_y and not flip_z:
        pvt2d = pvt.PVT2D(dwell_z, np.flip(dwell_y), np.flip(dwell_time, axis=0))
    elif not flip_y and flip_z:
        pvt2d = pvt.PVT2D(np.flip(dwell_z), dwell_y, np.flip(dwell_time, axis=1))
    elif flip_y and flip_z:
        pvt2d = pvt.PVT2D(np.flip(dwell_z), np.flip(dwell_y), np.flip(dwell_time))
    else: # not flip_y and not flip_z
        pvt2d = pvt.PVT2D(dwell_z, dwell_y, dwell_time)
    
    pvt2d.n_points = n_points
    pvt2d.run_in = run_in
    
    pvt.plot_pvt_path_2d(pvt2d)
    
    return dwell_z, dwell_y, pvt2d