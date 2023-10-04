# -*- coding: utf-8 -*-

"""
20230418
RIFTA with 4 extension

set brf
set etching region 
extension
fft
calculate the simulation result
draw

"""
import sys
sys.path.append('../lib/')
# import os

import scipy
import datetime
import numpy as np
import metrology as mt
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import pymurilo.File_Functions as pymf
# from scipy.io import loadmat

# from scipy import interpolate

# import lib_rifta.surface_extension_2d
from lib_rifta import BRFGaussian2D
from lib_rifta import Surface_Extension
from lib_rifta import DwellTime2D_FFT_Full_Test
from lib_rifta import Reference_Point
from lib_rifta import Selected_Region
from lib_rifta import Scanfile_Savejson
from lib_rifta import Generate_pvt_from_json
from lib_rifta.ibf_engine.remove_surface1 import remove_surface1
from lib_rifta.ibf_engine.conv_fft2 import conv_fft2
from lib_rifta.draw import draw_init, draw_simulation 

# from mpl_toolkits.mplot3d import Axes3D


#%% BRF params
brf_params = {}
brf_params['A'] = 5.01e-10
brf_params['sigma_xy'] = [4.96e-4, 4.96e-4]
m_per_pixel = 4.569341e-05
brf_params['d_pix'] = 90
brf_params['d'] = brf_params['d_pix'] * m_per_pixel
brf_params['lat_res_brf'] = m_per_pixel

brf_r = brf_params['d'] * 0.5

X_brf = np.arange(-brf_r, brf_r + m_per_pixel, m_per_pixel)
Y_brf = np.arange(-brf_r, brf_r + m_per_pixel, m_per_pixel)
xx, yy = np.meshgrid(X_brf, Y_brf)

Z_avg = BRFGaussian2D(xx, yy, 1, [brf_params['A'], brf_params['sigma_xy'], [0],[0]])
# brf_params
# fig = plt.figure(dpi=1800)
# ax = fig.add_subplot(111, projection='3d')


# ax.plot_surface(xx, yy, Z_avg , cmap='coolwarm')
# ax.set_xlabel('x (mm)')
# ax.set_ylabel('y (mm)')
# ax.set_zlabel('z (nm)')
# plt.show()

# load datx path
include = ["Proc_IBF_HDX_data_Gordo-B_230406_AB_clamped_5_data_stitched_mask_rmv.datx"]
base_path = r"C:/Users/frw78547/OneDrive - Diamond Light Source Ltd/Documents/IBF DATA/20230418_2D_4th_iter"
# base_path = r"./"

# save simulation result path
testname = 'Gordo-B_2023406_test'  #json file name saved 
folderjson = '../simu_data/'+datetime.datetime.now().strftime("%Y-%m-%d")+'/'+'Gordo-B_y-spacingx'+'{:.0f}'.format(1)
# load json dwell time and convert to pvt
# json_path = r"../simu_data/2023-04-18/Gordo-B_y-spacingx"+'{:.0f}'.format(1)+'/'

json_path = folderjson + '/'
# pvt load file
# pvt_gen = True
pvt_gen = False
json_data_filename = testname+"_8nn_extension_dwell_time.json"


#%% load datx - measurement file

'''
# BRFfitting = Reference_Point(include,base_path, x_min=103,x_max=108,y_min=6,y_max=11)

BRFfitting.cen_x = 105.3221
BRFfitting.cen_y = 8.5752
'''
BRFfitting = None
selected_region_length,selected_region_width = 12,5
X,Y,Z = Selected_Region(include,base_path,
                        fitresult = None,
                        x_offset_distance = 40, 
                        y_offset_distance = -10, 
                        x_ibf = -111.465,
                        y_ibf = 33.336,
                        selected_region_length = selected_region_length,
                        selected_region_width = selected_region_width,
                        mask = True, # no BRF x_shift,y_shift are from matlab calculation, or python
                        x_shift=-221.3591,
                        y_shift=24.6847
                        )
# Z = Z-np.nanmin(Z)

# m output-X,Y
# m output-Z
# selected_region_length,selected_region_width = 12,5
# hdx_path = pymf.List_Files(base_path, file_type=".datx", include=include, level=3)

# df = mt.read_mx(hdx_path[0], replace_na=True, convert_z=True, apply_coord=True, apply_lat_cal=True)

# X = df.columns.to_numpy() * 1e3 -24.6847
# Y = df.index.to_numpy() * 1e3 -221.3591
# Z = df.to_numpy() * 1e9
#%%

# data_dir = r'C:\Users\frw78547\OneDrive - Diamond Light Source Ltd\Documents\IBF DATA\20230314_2D_3rd_iter\\'
# mat_file = "Gordo_B_2D_3rd_etching_region_12x5_new.mat"

# mat_contents = loadmat(data_dir + mat_file)
# # Z_region_b = mat_contents['Z_region_b']/1e9
# Z = mat_contents['Z_region_b']/1e9

# X = mat_contents['X_selected_region']/1e3
# Y = mat_contents['Y_selected_region']/1e3
m_per_pixel = 4.569341e-05
pixel_m = np.median(np.diff(X[0,:]))
coors = [X[0,0]*1e3, X[0,-1]*1e3, Y[-1,0]*1e3, Y[0,0]*1e3]

fig = plt.figure(dpi=1800)
plt.imshow(Z, extent=coors)
plt.colorbar()  
plt.xlabel('x')
plt.ylabel('y')
# plt.gca().invert_yaxis()
plt.show()


#%% meter unit

# Dwell time calculation parameters
dx_ibf = 1e-4
brf_mode = 'model'
ratio = 1
tmin = 0
tmax = 1
run_iter = True
TT = 0

options = {
    'Algorithm': 'Iterative-FFT',
    'maxIters': 50, #50
    'PV_dif': 0.01e-9,
    'RMS_dif': 0.02e-9,
    'dwellTime_dif': 60,
    'isDownSampling': False,
    'samplingInterval': dx_ibf
}

selection = [True, True, True, True, False, False, False, False]
selection_savejson = [False, False, False, False, False, False, False, False]



#%%

# 1. Zero extension
X_ext, Y_ext, Z_0, ca_range = Surface_Extension(X, Y, Z, brf_params, Z_avg, 'zero', False)
# X_ext started from X[0,0], keep in normal order(small to large)
Z_ext = np.empty(X_ext.shape) * np.nan  # mark the Z_ext to NaN
Z_ext[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] = Z  # fill points
r = max(np.array(Z_ext.shape) - np.array(Z.shape)) // 2  # obtain the area of extension
u, v = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))

coors = np.vstack((u.flatten(), v.flatten())).T
rr = np.linalg.norm(coors,axis=1).reshape(u.shape)
se = rr <= r
BW_Z = binary_dilation(~np.isnan(Z_0), structure=se)
id_ext = BW_Z == 0


if selection[0]:
    B, X_B, Y_B, _, _, T_0, _, X_P, Y_P, _, _, _, _, _, _, X_ca, Y_ca, _, _, _ = \
        DwellTime2D_FFT_Full_Test(
            X_ext, Y_ext, Z_0, 0, brf_params.copy(), brf_mode, X_brf, Y_brf, Z_avg, ca_range,
            pixel_m, tmin, tmax, options, ratio, False
        )
        
    T_0[id_ext] = 0

    # Z_removal_dw_0 = scipy.signal.fftconvolve(T_0, B,mode='same')
    Z_removal_dw_0 = conv_fft2(T_0, B)
    Z_0 = remove_surface1(X_ext, Y_ext, Z_0)
    Z_0 = Z_0 - np.nanmin(Z_0)
    Z_residual_dw = Z_0 - np.nanmin(Z_0) - Z_removal_dw_0
    Z_residual_ca_0 = Z_residual_dw[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca_0 = remove_surface1(X_ca, Y_ca, Z_residual_ca_0)
#%
#%
# 2. Gaussian 
if selection[1]:
    _, _, Z_gauss, _ = Surface_Extension(X, Y, Z, brf_params, Z_avg, 'gauss', False)

    _, _, _, _, _, T_gauss, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
        DwellTime2D_FFT_Full_Test(
            X_ext, Y_ext, Z_gauss, 0, brf_params.copy(), brf_mode, X_brf, Y_brf, Z_avg, ca_range,
            pixel_m, tmin, tmax, options, ratio, False
        )

    T_gauss[id_ext] = 0
    # Z_removal_dw_gauss = scipy.signal.fftconvolve(T_gauss, B, mode='same')
    Z_removal_dw_gauss = conv_fft2(T_gauss, B)
    Z_gauss = remove_surface1(X_ext, Y_ext, Z_gauss)
    Z_gauss = Z_gauss - np.nanmin(Z_gauss)  
    Z_residual_dw = Z_gauss - Z_removal_dw_gauss
    Z_residual_ca_gauss = Z_residual_dw[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca_gauss = remove_surface1(X_ca, Y_ca, Z_residual_ca_gauss)
#%
#%%
# 3. 8nn
if selection[2]:
    _, _, Z_8nn, _ = Surface_Extension(X, Y, Z, brf_params, Z_avg, '8nn', False)

    _, _, _, _, _, T_8nn, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
        DwellTime2D_FFT_Full_Test(
            X_ext, Y_ext, Z_8nn, 0, brf_params.copy(), brf_mode, X_brf, Y_brf, Z_avg, ca_range,
            pixel_m, tmin, tmax, options, ratio, False
        )#

    T_8nn[id_ext] = 0
    # Z_removal_dw_8nn = scipy.signal.fftconvolve(T_8nn, B, mode='same')
    Z_removal_dw_8nn = conv_fft2(T_8nn, B)
    Z_8nn = remove_surface1(X_ext, Y_ext, Z_8nn)
    Z_8nn = Z_8nn - np.nanmin(Z_8nn)
    Z_residual_dw = Z_8nn - Z_removal_dw_8nn
    Z_residual_ca_8nn = Z_residual_dw[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca_8nn = remove_surface1(X_ca, Y_ca, Z_residual_ca_8nn)

#%%

# 4. 8nn_fall

if selection[3]:
    _, _, Z_8nn_fall, _ = Surface_Extension(X, Y, Z, brf_params, Z_avg, '8nn', True)

    _, _, _, _, _, T_8nn_fall, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
        DwellTime2D_FFT_Full_Test(
            X_ext, Y_ext, Z_8nn_fall, 0, brf_params.copy(), brf_mode, X_brf, Y_brf, Z_avg, ca_range,
            pixel_m, tmin, tmax, options, ratio, False
        )

    T_8nn_fall[id_ext] = 0
    # Z_removal_dw_8nn_fall = scipy.signal.fftconvolve(T_8nn_fall, B, mode='same')
    Z_removal_dw_8nn_fall = conv_fft2(T_8nn_fall, B)
    Z_8nn_fall = remove_surface1(X_ext, Y_ext, Z_8nn_fall)
    Z_8nn_fall = Z_8nn_fall - np.nanmin(Z_8nn_fall)
    Z_residual_dw = Z_8nn_fall - Z_removal_dw_8nn_fall
    Z_residual_ca_8nn_fall = Z_residual_dw[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca_8nn_fall = remove_surface1(X_ca, Y_ca, Z_residual_ca_8nn_fall)


#1
# fig_Z_0 = plt.figure(dpi=1800)
# plt.pcolormesh(X_ext*1e3, Y_ext*1e3, Z_0*1e9)


#%%
# fig, axs = plt.subplots(ncols=4, figsize=(15, 6))
# Assuming X, Y, Z, X_ext, Y_ext, Z_0, T_0, Z_residual_ca_0, Z_removal_dw_0 are given numpy arrays

map_height = 1 + np.count_nonzero(selection)
map_name = ['height_error', 'dwell_time', 'residual', 'removal']
ext_name = ['Zero extension','Gaussian extension','8nn extension','8nn fall extension']

fig, grid, ax0, mesh0 = draw_init(X, Y, Z, selection, xx, yy, Z_avg, brf_params, ext_name, map_name)



fig.tight_layout()
fig.subplots_adjust(top=0.99, bottom=0.1, left=0.1, right=0.8, hspace=0.5, wspace=0.4)

if selection[0]:
    draw_simulation(fig, X, Y, X_ext, Y_ext, Z_0, T_0,
                    Z_residual_ca_0, Z_removal_dw_0, grid, ext_name, map_name, map_row = 1)

if selection_savejson[0]:
    simulated_result = {map_name[0]:Z_0* 1e9,map_name[1]:T_0,map_name[2]:Z_residual_ca_0* 1e9,map_name[3]:Z_removal_dw_0* 1e9}

    for i_name in range(0, len(map_name)):
        # create a new folder simu_data
        Scanfile_Savejson(folderjson+'/'+testname+'_test_Zero_extension_'+map_name[i_name],
                          X_ext, Y_ext, simulated_result[map_name[i_name]],
                          testname=testname, data_from='HD-X', calculation='RIFTA',
                          grid = m_per_pixel*1E3 ,dwell_time_unit = 's',
                          height_error_filter = 0, brf_params=brf_params,
                          x_start = np.min(X)*1e3,y_start = np.min(Y)*1e3,
                          selected_region_length = selected_region_length,
                          selected_region_width = selected_region_width,
                          fitresult=BRFfitting)
    
        # scanfile_savejson([folderjson,'/Gordo-B_202303021_test_8NN_fall_extension_',map_name{1}],X_ext,Y_ext,Z_8nn_fall*1e9, ...
        # 'Gordo-B_20230228_test','HD-X','RIFTA', ...
        # m_per_pixel*1E3,'mm','s',0,brf_params,fitresult)

if selection[1]:
    draw_simulation(fig, X, Y, X_ext, Y_ext, Z_gauss, T_gauss,
                    Z_residual_ca_gauss, Z_removal_dw_gauss, grid, ext_name, map_name, map_row = 2)

if selection_savejson[1]:
    simulated_result = {map_name[0]:Z_gauss* 1e9,map_name[1]:T_gauss,map_name[2]:Z_residual_ca_gauss* 1e9,map_name[3]:Z_removal_dw_gauss* 1e9}

    for i_name in range(0, len(map_name)):
        Scanfile_Savejson(folderjson+'/'+testname+'_Gaussian_extension_'+map_name[i_name],
                          X_ext, Y_ext, simulated_result[map_name[i_name]],
                          testname=testname, data_from='HD-X', calculation='RIFTA',
                          grid = m_per_pixel*1E3 ,dwell_time_unit = 's',
                          height_error_filter = 0, brf_params=brf_params,
                          x_start = np.min(X)*1e3,y_start = np.min(Y)*1e3,
                          selected_region_length = selected_region_length,
                          selected_region_width = selected_region_width,
                          fitresult=BRFfitting)
    
    
if selection[2]:
    draw_simulation(fig, X, Y, X_ext, Y_ext, Z_8nn, T_8nn,
                    Z_residual_ca_8nn, Z_removal_dw_8nn, grid, ext_name, map_name, map_row = 3)

if selection_savejson[2]:
    simulated_result = {map_name[0]:Z_8nn* 1e9,map_name[1]:T_8nn,map_name[2]:Z_residual_ca_8nn* 1e9,map_name[3]:Z_removal_dw_8nn* 1e9}
    for i_name in range(0, len(map_name)):
        Scanfile_Savejson(folderjson+'/'+testname+'_8nn_extension_'+map_name[i_name],
                          X_ext, Y_ext, simulated_result[map_name[i_name]],
                          testname=testname, data_from='HD-X', calculation='RIFTA',
                          grid = m_per_pixel*1E3 ,dwell_time_unit = 's',
                          height_error_filter = 0, brf_params=brf_params,
                          x_start = np.min(X)*1e3,y_start = np.min(Y)*1e3,
                          selected_region_length = selected_region_length,
                          selected_region_width = selected_region_width,
                          fitresult=BRFfitting)

if selection[3]:
    draw_simulation(fig, X, Y, X_ext, Y_ext, Z_8nn_fall, T_8nn_fall,
                    Z_residual_ca_8nn_fall, Z_removal_dw_8nn_fall, grid, ext_name, map_name, map_row = 4)


if selection_savejson[3]:
    simulated_result = {map_name[0]:Z_8nn_fall* 1e9,map_name[1]:T_8nn_fall,map_name[2]:Z_residual_ca_8nn_fall* 1e9,map_name[3]:Z_removal_dw_8nn_fall* 1e9}
    for i_name in range(0, len(map_name)):
        Scanfile_Savejson(folderjson+'/'+testname+'_8nn_fall_extension_'+map_name[i_name],
                          X_ext, Y_ext, simulated_result[map_name[i_name]],
                          testname=testname, data_from='HD-X', calculation='RIFTA',
                          grid = m_per_pixel*1E3 ,dwell_time_unit = 's',
                          height_error_filter = 0, brf_params=brf_params,
                          x_start = np.min(X)*1e3,y_start = np.min(Y)*1e3,
                          selected_region_length = selected_region_length,
                          selected_region_width = selected_region_width,
                          fitresult=BRFfitting)


plt.show()

#%% generate_pvt_from_json
#pvt2d_set comtains dwell_z, dwell_y, pvt2d
if pvt_gen:
    pvt2d_set = Generate_pvt_from_json(json_path = json_path,
                               data_filename = json_data_filename,
                               run_in =60,
                               n_points =101)