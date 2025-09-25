# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 02:49:26 2025

@author: Etrr
"""


import os
import sys
os.chdir(r"C:/Users/Etrr/OneDrive - Diamond Light Source Ltd/Documents/Python Scripts/pyrifta")

import h5py
import time

import scipy
import datetime

import numpy as np

from lib_rifta.surface_extension_2d import Surface_Extension_Iter_freq
from lib_rifta.surface_extension_2d import Surface_Extension_Iter
from lib_rifta.ibf_engine.conv_fft2 import conv_fft2
from lib_rifta.ibf_engine.remove_surface1 import remove_surface1
from lib_rifta import Generate_pvt_from_json
from lib_rifta import Scanfile_Savejson
from lib_rifta import Selected_Region
from lib_rifta import Reference_Point
from lib_rifta import BRFGaussian2D, BRFSuperGaussian2D, Surface_Extension, DwellTime2D_FFT_Full_Test
from lib_rifta import BRFGaussian2D, BRFSuperGaussian2D, Surface_Extension, DwellTime2D_FFT_Full_Test
from lib_rifta import DwellTime2D_TSVD, DwellTime2D_LSQR, DwellTime2D_Bayesian
import pymurilo.File_Functions as pymf
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import metrology as mt




sys.path.append('./pyrifta/lib_rifta/')
# import os
# from scipy.io import loadmat

# from scipy import interpolate

# import lib_rifta.surface_extension_2d
# from mpl_toolkits.mplot3d import Axes3D


# %% BRF params

tic = time.time()


# load datx path
# include = ["Proc_IBF_HDX_data_Gordo-B_230406_AB_clamped_5_data_stitched_mask_rmv.datx"]
# base_path = r"C:/Users/Etrr/OneDrive - Diamond Light Source Ltd/Documents/IBF DATA/20230418_2D_4th_iter"

# include = ["DUNiP-S4-BRF_20230919_clamped_7_data_stitched_pss.datx"]
# base_path = r"C:/Users/Etrr/OneDrive - Diamond Light Source Ltd/Documents/IBF DATA/230921_DUNiP_S4_7_P001"

include = ["Thesis_temp_figure_error.mat"]
base_path = r"C:\Users\Etrr\Desktop\IBF_Code\IBF_data\Template\\"

# base_path = r"./"

# save simulation result path
testname = 'Test'  # json file name saved
folderjson = '../simu_data/'+datetime.datetime.now().strftime("%Y-%m-%d")+'/' + \
    'y-spacingx'+'{:.0f}'.format(1)
# load json dwell time and convert to pvt
# json_path = r"../simu_data/2023-04-18/Gordo-B_y-spacingx"+'{:.0f}'.format(1)+'/'

json_path = folderjson + '/'
# pvt load file

pvt_gen = False
# json_data_filename = testname+"_8nn_extension_dwell_time.json"


# json_data_filename = testname+"_8nn_fall_extension_dwell_time"   #better look through the savejson part to keep the name consistency
# better look through the savejson part to keep the name consistency
json_data_filename = testname+"_iter_extension_dwell_time"

# %% load datx - measurement file

'''
# BRFfitting = Reference_Point(include,base_path, x_min=103,x_max=108,y_min=6,y_max=11)

BRFfitting.cen_x = 105.3221
BRFfitting.cen_y = 8.5752
'''

BRFfitting = None
if include[0].endswith('datx'):

    BRFfitting = None
    selected_region_length, selected_region_width = 12, 5
    X, Y, Z = Selected_Region(include, base_path,
                              fitresult=None,
                              x_offset_distance=40,
                              y_offset_distance=-10,
                              x_ibf=-111.465,
                              y_ibf=33.336,
                              selected_region_length=selected_region_length,
                              selected_region_width=selected_region_width,
                              mask=True,  # no BRF x_shift,y_shift are from matlab calculation, or python
                              x_shift=-221.3591,
                              y_shift=24.6847
                              )
    X,Y = Y.T,X.T
    Z = Z.T
    # Z = np.flipud(Z).T

elif include[0].endswith('.mat'):
    # base_path = r'C:\Users\frw78547\OneDrive - Diamond Light Source Ltd\Documents\IBF DATA\20230314_2D_3rd_iter\\'
    # include = "Gordo_B_2D_3rd_etching_region_12x5_new.mat"
    # mat_contents = scipy.io.loadmat(base_path + include[0])

    mat_contents = {}
    with h5py.File(base_path + include[0], 'r') as file:
        # 注意数据翻转问题，python(行优先)与matlab(列优先)数据读取顺序不一样
        for key in file.keys():
            data = file[key][()]
            if isinstance(data, np.ndarray) and data.ndim == 2:
                mat_contents[key] = np.flipud(data.T)
            else:
                mat_contents[key] = data

    # Z_region_b = mat_contents['Z_region_b']/1e9
    Z = np.flipud(mat_contents['Z_region_b']/1e9)

    X = mat_contents['X_selected_region']/1e3
    Y = np.flipud(mat_contents['Y_selected_region'])/1e3
    # Z = Z-np.nanmin(Z)


# m output-X,Y
# m output-Z
# selected_region_length,selected_region_width = 12,5
# hdx_path = pymf.List_Files(base_path, file_type=".datx", include=include, level=3)

# df = mt.read_mx(hdx_path[0], replace_na=True, convert_z=True, apply_coord=True, apply_lat_cal=True)

# X = df.columns.to_numpy() * 1e3 -24.6847
# Y = df.index.to_numpy() * 1e3 -221.3591
# Z = df.to_numpy() * 1e9
# %

# data_dir = r'C:\Users\frw78547\OneDrive - Diamond Light Source Ltd\Documents\IBF DATA\20230314_2D_3rd_iter\\'
# mat_file = "Gordo_B_2D_3rd_etching_region_12x5_new.mat"

# mat_contents = loadmat(data_dir + mat_file)
# # Z_region_b = mat_contents['Z_region_b']/1e9
# Z = mat_contents['Z_region_b']/1e9

# X = mat_contents['X_selected_region']/1e3
# Y = mat_contents['Y_selected_region']/1e3
# m_per_pixel = 4.569341e-05
pixel_m = np.median(np.diff(X[0, :]))
m_per_pixel = pixel_m
coors = [X[0, 0]*1e3, X[0, -1]*1e3, Y[0, 0]*1e3, Y[-1, 0]*1e3]

selected_region_length, selected_region_width = Z.shape[1] * \
    pixel_m*1e3, Z.shape[0]*pixel_m*1e3

fig = plt.figure(dpi=800)
plt.imshow(Z, extent=coors)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')


rms_Z = np.std(Z) * 1e9
# plt.tight_layout()
plt.title(f"PV = {np.round((np.max(Z) - np.min(Z)) * 1e9, 2)} nm, RMS = {np.round(rms_Z, 2)} nm", y=1.05)

# plt.gca().invert_yaxis()
plt.show()

# %%
brf_type = "gaussian"
brf_params = {}

# input_BRF = {}
# with h5py.File(base_path + "MRF_sp_1300_diamond_interp3.mat", 'r') as file:

#     for key in file.keys():
#         data = file[key][()]
#         if isinstance(data, np.ndarray) and data.ndim == 2:
#             input_BRF[key] = np.flipud(data.T)
#         else:
#             input_BRF[key] = data



if 'input_BRF' in locals():

    brf_params = {}
    brf_params['A'] = np.max(input_BRF['Z_BRF_sample']) * 1e-9
    brf_params['sigma_xy'] = [8.358e-4, 8.343e-4]
    brf_params['dx_pix'] = input_BRF['Z_BRF_sample'].shape[1]
    brf_params['dy_pix'] = input_BRF['Z_BRF_sample'].shape[0]
    if brf_params['dx_pix'] == brf_params['dy_pix']:
        brf_params['d_pix'] = brf_params['dx_pix']

    brf_params['dx'] = brf_params['dx_pix'] * m_per_pixel
    brf_params['dy'] = brf_params['dy_pix'] * m_per_pixel
    brf_params['lat_res_brf'] = m_per_pixel

    brf_rx = brf_params['dx'] * 0.5
    brf_ry = brf_params['dy'] * 0.5

    # X_brf = np.arange(-brf_rx, brf_rx+m_per_pixel*1e-3, m_per_pixel)
    # Y_brf = np.arange(-brf_ry, brf_ry+m_per_pixel*1e-3, m_per_pixel)

    X_brf = np.arange(-brf_rx, brf_rx, m_per_pixel)
    # 如果X_brf尺寸小于input_BRF['Z_BRF_sample'].shape, 请加一个微小量m_per_pixel*1e-3在结尾的brf_rx后
    Y_brf = np.arange(-brf_ry, brf_ry, m_per_pixel)

    xx, yy = np.meshgrid(X_brf, Y_brf)
    Z_avg = input_BRF['Z_BRF_sample'] * 1e-9
    # Z_avg = BRFGaussian2D(xx, yy, 1, [brf_params['A'], brf_params['sigma_xy'], [0],[0]])
elif 'brf_type' in locals():
    brf_params['A'] = 1.289e-9
    brf_params['sigma_xy'] = [x * 1 for x in [0.558e-3, 0.543e-3]]

    brf_params['d_pix'] = 41
    brf_params['d'] = brf_params['d_pix'] * m_per_pixel
    brf_params['lat_res_brf'] = m_per_pixel

    brf_r = brf_params['d'] * 0.5
    X_brf = np.arange(-brf_r, brf_r+m_per_pixel*1e-3,
                      m_per_pixel)  # digit accuracy problem
    Y_brf = np.arange(-brf_r, brf_r+m_per_pixel*1e-3, m_per_pixel)
    xx, yy = np.meshgrid(X_brf, Y_brf)

    X_brf_fullres = np.arange(-brf_r, brf_r + m_per_pixel, m_per_pixel)
    Y_brf_fullres = np.arange(-brf_r, brf_r + m_per_pixel, m_per_pixel)

    if brf_type == "gaussian":
        Z_avg = BRFGaussian2D(
            xx, yy, 1, [brf_params['A'], brf_params['sigma_xy'], [0, 0]])
    elif brf_type == "supergaussian":
        brf_params['p'] = 1.644
        Z_avg = BRFSuperGaussian2D(
            xx, yy, 1, [brf_params['A'], brf_params['sigma_xy'], [0, 0], brf_params['p']])


# brf_params
fig = plt.figure(dpi=1800)
ax = fig.add_subplot(111, projection='3d')


ax.plot_surface(xx*1000, yy*1000, Z_avg , cmap='coolwarm')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (nm)')
plt.show()


# %% meter unit

# Dwell time calculation parameters
dx_ibf = 1e-4
brf_mode = 'avg'
ratio = 1
tmin = 0
tmax = 1
run_iter = True
TT = 0

# T_min = 0.008


options = {
    'Algorithm': 'Iterative-FFT',
    'maxIters': 50,  # 50
    'PV_dif': 0.01e-9,
    'RMS_dif': 0.001e-9,
    'dwellTime_dif': 30,
    'isDownSampling': False,
    'samplingInterval': dx_ibf
}
selection = [True, True, True, True, True, True, True, True, True]
# selection = [True, False, False, False, False, False, False, True]
selection_savejson = [False, False, False, False, False, False, False, False, False]


# %%
Ext_rx, Ext_ry = 0.5, 0.5
# 1. Zero extension
X_ext, Y_ext, Z_0, ca_range = Surface_Extension(
    X, Y, Z, brf_params, Z_avg, 'zero', False, ext_r_x = Ext_rx, ext_r_y = Ext_ry)
# X_ext started from X[0,0], keep in normal order(small to large)
Z_ext = np.empty(X_ext.shape) * np.nan  # mark the Z_ext to NaN
Z_ext[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] = Z  # fill points

'''
r = max(np.array(Z_ext.shape) - np.array(Z.shape)
        ) // 2  # obtain the area of extension
u, v = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))

coors = np.vstack((u.flatten(), v.flatten())).T
rr = np.linalg.norm(coors, axis=1).reshape(u.shape)
se = rr <= r
BW_Z = binary_dilation(~np.isnan(Z_0), structure=se)
id_ext = BW_Z == 0
'''

# 分别计算 X 和 Y 方向上的扩展区域大小
r_x = max(X_ext.shape[1] - Z.shape[1], 0) // 2
r_y = max(Y_ext.shape[0] - Z.shape[0], 0) // 2
u, v = np.meshgrid(np.arange(-r_x, r_x+1), np.arange(-r_y, r_y+1))

# 计算每个点到中心的距离
coors = np.vstack((u.flatten(), v.flatten())).T
rr = np.linalg.norm(coors, axis=1).reshape(u.shape)

# 创建结构元素
if r_x != r_y:
    se = (abs(u) <= r_x) & (abs(v) <= r_y)
else:
    se = rr <= np.maximum(r_x,r_y)

Z_b = Z_0.copy()
Z_b[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] = np.nan

# 二进制膨胀操作，考虑不同方向上的扩展
BW_Z = binary_dilation(np.isnan(Z_b), structure=se)

# 确定需要扩展的区域
id_ext = BW_Z == 0


if selection[0]:
    B, X_B, Y_B, _, _, T_0, _, X_P, Y_P, _, _, _, _, _, _, X_ca, Y_ca, _, _, _ = \
        DwellTime2D_FFT_Full_Test(
            X_ext, Y_ext, Z_0, 0, brf_params.copy(), brf_mode, X_brf, Y_brf, Z_avg, ca_range,
            pixel_m, tmin, tmax, options, ratio, False, ext_r_x = Ext_rx, ext_r_y = Ext_ry
        )

    T_0[id_ext] = 0

    # Z_removal_dw_0 = scipy.signal.fftconvolve(T_0, B,mode='same')
    if 'T_min' in locals():
        T_0 = T_0 - np.min(T_0) + T_min
    else:
        T_0 = T_0 - np.min(T_0)

    Z_removal_dw_0 = conv_fft2(T_0, B)
    Z_0 = remove_surface1(X_ext, Y_ext, Z_0)
    Z_0 = Z_0 - np.nanmin(Z_0)
    Z_residual_dw = Z_0 - np.nanmin(Z_0) - Z_removal_dw_0
    Z_residual_ca_0 = Z_residual_dw[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca_0 = remove_surface1(X_ca, Y_ca, Z_residual_ca_0)
    print(f"Zero_CA残差: {np.std(Z_residual_ca_0)*1e9:.2f}")
# %
# %%
# 2. Gaussian
if selection[1]:
    _, _, Z_gauss, _ = Surface_Extension(
        X, Y, Z, brf_params, Z_avg, 'gauss', False, ext_r_x = Ext_rx, ext_r_y = Ext_ry)

    _, _, _, _, _, T_gauss, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
        DwellTime2D_FFT_Full_Test(
            X_ext, Y_ext, Z_gauss, 0, brf_params.copy(
            ), brf_mode, X_brf, Y_brf, Z_avg, ca_range,
            pixel_m, tmin, tmax, options, ratio, False, ext_r_x = Ext_rx, ext_r_y = Ext_ry
        )

    T_gauss[id_ext] = 0
    # Z_removal_dw_gauss = scipy.signal.fftconvolve(T_gauss, B, mode='same')
    if 'T_min' in locals():
        T_gauss = T_gauss - np.min(T_gauss) + T_min
    else:
        T_gauss = T_gauss - np.min(T_gauss)

    Z_removal_dw_gauss = conv_fft2(T_gauss, B)
    Z_gauss = remove_surface1(X_ext, Y_ext, Z_gauss)
    Z_gauss = Z_gauss - np.nanmin(Z_gauss)
    Z_residual_dw = Z_gauss - Z_removal_dw_gauss
    Z_residual_ca_gauss = Z_residual_dw[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca_gauss = remove_surface1(X_ca, Y_ca, Z_residual_ca_gauss)
    print(f"Gauss_CA残差: {np.std(Z_residual_ca_gauss)*1e9:.2f}")
# %
# %%
# 3. 8nn
if selection[2]:
    _, _, Z_8nn, _ = Surface_Extension(
        X, Y, Z, brf_params, Z_avg, '8nn', False, ext_r_x = Ext_rx, ext_r_y = Ext_ry)

    _, _, _, _, _, T_8nn, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
        DwellTime2D_FFT_Full_Test(
            X_ext, Y_ext, Z_8nn, 0, brf_params.copy(), brf_mode, X_brf, Y_brf, Z_avg, ca_range,
            pixel_m, tmin, tmax, options, ratio, False, ext_r_x = Ext_rx, ext_r_y = Ext_ry
        )
    T_8nn[id_ext] = np.nan
    # T_8nn[id_ext] = 0
    # Z_removal_dw_8nn = scipy.signal.fftconvolve(T_8nn, B, mode='same')
    if 'T_min' in locals():
        T_8nn = T_8nn - np.nanmin(T_8nn) + T_min
    else:
        T_8nn = T_8nn - np.nanmin(T_8nn)
        
    T_8nn[id_ext] = 0
    
    Z_removal_dw_8nn = conv_fft2(T_8nn, B)
    # Z_8nn = remove_surface1(X_ext, Y_ext, Z_8nn)
    Z_8nn = Z_8nn - np.nanmin(Z_8nn)
    Z_residual_dw = Z_8nn - Z_removal_dw_8nn
    Z_residual_ca_8nn = Z_residual_dw[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca_8nn = remove_surface1(X_ca, Y_ca, Z_residual_ca_8nn)
    print(f"8nn_CA残差: {np.std(Z_residual_ca_8nn)*1e9:.2f}")
# %%
# 4. 8nn Bayesian
if selection[2]:
    _, _, Z_8nn, _ = Surface_Extension(
        X, Y, Z, brf_params, Z_avg, '8nn', False, ext_r_x = Ext_rx, ext_r_y = Ext_ry)


    _, _, _, _, _, T_8nn, _, _, _, _, _, _, _, _, _, _, _, _ = \
        DwellTime_2D_Bayesian(
            Z_8nn, brf_params.copy(), brf_mode, X_brf, Y_brf, Z_avg, ca_range,
            pixel_m, options, ratio
        )
    T_8nn[id_ext] = np.nan
    # T_8nn[id_ext] = 0
    # Z_removal_dw_8nn = scipy.signal.fftconvolve(T_8nn, B, mode='same')
    if 'T_min' in locals():
        T_8nn = T_8nn - np.nanmin(T_8nn) + T_min
    else:
        T_8nn = T_8nn - np.nanmin(T_8nn)
        
    T_8nn[id_ext] = 0
    
    Z_removal_dw_8nn = conv_fft2(T_8nn, B)
    # Z_8nn = remove_surface1(X_ext, Y_ext, Z_8nn)
    Z_8nn = Z_8nn - np.nanmin(Z_8nn)
    Z_residual_dw = Z_8nn - Z_removal_dw_8nn
    Z_residual_ca_8nn = Z_residual_dw[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca_8nn = remove_surface1(X_ca, Y_ca, Z_residual_ca_8nn)
    print(f"8nn_CA残差: {np.std(Z_residual_ca_8nn)*1e9:.2f}")
    
    
    