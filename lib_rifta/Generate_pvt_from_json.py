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

def Generate_pvt_from_json(
        json_path,data_filename,run_in = 10,n_points = 200,
        flip_y = True, flip_z = False,
        plot2d = True, plot1d = True
        ):
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
    
    max_v_list = [max(pvt1d.velocity) for pvt1d in pvt2d.generate_pvts()]
    max(max_v_list)
    
    # In[5]:
    pvt_filename = "v10test_10runin"+data_filename +"_pvt_" +str(pvt2d.total_points)+".json"
    
    # pvt_filename = r"Non_inputIBF_trapz_c20_l95_8NN_extension_dwell time_pvt_1396062.json"
    pvt2d.write_profile(json_path, pvt_filename)
    
    
    # In[6]:
    
    # base_path = r"C:/Users/frw78547/OneDrive - Diamond Light Source Ltd/Documents/IBF DATA/230619_trapz_P001/"
    
    path = os.path.join(json_path, pvt_filename)
    
    with open(path, 'r') as f:
        data_pvt = json.load(f)
    
    # Extract y and z data
    y = np.array(data_pvt['y'])
    z = np.array(data_pvt['z'])
    delta_t = np.array(data_pvt['delta_t'])
    # Flatten z and repeat y
    z_trajectory = np.zeros(z.shape)
    for i in range(z.shape[0]):
        if  i % 2 == 0:
            z_trajectory[i,:] =  z[i,:]
        else :
            z_trajectory[i,:] = np.flip(z[i,:])
    
    v = np.zeros_like(z_trajectory)
    a = np.zeros_like(z_trajectory)
    
    for i in range(z_trajectory.shape[0]):
        df_z = abs(np.diff(z_trajectory[i,:]))
        v[i,:] = np.concatenate(([df_z[0]/delta_t[i]], df_z/delta_t[i])) 
        a[i,:] = np.concatenate(([0], np.diff(v[i,:])/delta_t[i]))
        
    
    z_flat = z_trajectory.flatten()
    y_flat = np.repeat(y, z.shape[1])
    delta_t_flat =np.repeat(delta_t, z.shape[1])
    
    
    t = delta_t_flat.cumsum()-delta_t_flat[0]
    
    # Calculate velocity
    # v = np.sqrt((np.diff(y_flat))**2 + (np.diff(z_flat))**2)/delta_t_flat[:-1]
    v_flat = v.flatten()
    
    # Calculate acceleration
    # a = np.pad(np.diff(v), (1,0), 'constant', constant_values=0)/delta_t_flat[:-1]
    a_flat = a.flatten()
    # Create figure and subplots for velocity and acceleration
    
    
    if plot2d == True:
        fig, axs2d = plt.subplots(1, 2, figsize=(12, 4))
        ## In[7.1]:
        # Velocity subplot
        sc = axs2d[0].scatter(z_flat, y_flat, c=v, s=0.004, cmap='jet',vmin=0,vmax=10*np.std(v))
        axs2d[0].set_title('Velocity')
        axs2d[0].set_xlabel('z Axis (mm)')
        axs2d[0].set_ylabel('y Axis (mm)')
        
        fig.colorbar(sc, ax=axs2d[0], label='mm/s')
        fig.set_dpi(600)
        
        # Acceleration subplot
        sc = axs2d[1].scatter(z_flat, y_flat, c=a, s=0.004, cmap='jet',vmin=-10*np.std(a),vmax=10*np.std(a))
        axs2d[1].set_title('Acceleration')
        axs2d[1].set_xlabel('z Axis (mm)')
        axs2d[1].set_ylabel('y Axis (mm)')
        
        fig.colorbar(sc, ax=axs2d[1], label='mm/s^2')
        fig.set_dpi(600)
        
        # Show the plots
        plt.show()

# In[2]:  linear plot
    if plot1d == True:
        # Z vs Time
        plt.figure(figsize=(12, 6), dpi=800)  # Create a new figure with specific size
        plt.plot(t, z_flat, linewidth=1)
        plt.title('Input Z vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Z (mm)')
        # plt.xlim([1, 11]) # set x lim
        plt.ylim([np.nanmin(z_flat), np.nanmax(z_flat)])
        plt.show()
        
        # V vs Time
        plt.figure(figsize=(12, 6), dpi=800)  # Create a new figure with specific size
        plt.plot(t, v_flat, linewidth=1)
        plt.title('Input V vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('V (mm/s)')
        # plt.xlim([1, 11]) # set x lim
        # plt.xlim([1000, 1120]) # set x lim
        # plt.ylim([np.nanmin(v), np.nanmax(v)])
        plt.ylim([np.nanmin(v)/1.2, np.nanmax(v)*1.2])
        plt.show()
    
        # a vs Time
        plt.figure(figsize=(12, 6), dpi=800)  # Create a new figure with specific size
        plt.plot(t, a_flat, linewidth=1)
        plt.title('Input a vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('a (mm/s^2)')
        # plt.xlim([1, 11]) # set x lim
        # plt.xlim([1000, 1120]) # set x lim
        # plt.ylim([np.nanmin(a), np.nanmax(a)])
        # plt.ylim([-10*np.std(a), 10*np.std(a)])
        plt.ylim([-650, 650])
        plt.show()
    
    
    
    return dwell_z, dwell_y, pvt2d, path