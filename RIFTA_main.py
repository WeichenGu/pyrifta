# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 00:46:49 2023

@author: frw78547
"""
import sys

sys.path.append('../lib/')
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat
from scipy import interpolate

from scipy.ndimage import binary_dilation
from scipy.ndimage import generate_binary_structure

from lib_rifta import BRFGaussian2D
import lib_rifta.surface_extension_2d
from lib_rifta import Surface_Extension
from lib_rifta import DwellTime2D_FFT_Full_Test
from lib_rifta.ibf_engine.remove_surface1 import remove_surface1

from mpl_toolkits.mplot3d import Axes3D

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


# plt.imshow(np.flipud(Zf/632.8*1e9), extent=[np.min(Xp), np.max(Xp), np.min(Yp), np.max(Yp)], cmap='gray')

# fig = plt.figure(dpi=1800)
# ax = fig.add_subplot(111, projection='3d')


# ax.plot_surface(xx, yy, Z_avg , cmap='coolwarm')
# ax.set_xlabel('x (mm)')
# ax.set_ylabel('y (mm)')
# ax.set_zlabel('z (nm)')
# plt.show()
#%%


m_per_pixel = 4.569341e-05
data_dir = r'C:\Users\frw78547\OneDrive - Diamond Light Source Ltd\Documents\IBF DATA\20230314_2D_3rd_iter\\'
mat_file = "Gordo_B_2D_3rd_etching_region_12x5_new.mat"

mat_contents = loadmat(data_dir + mat_file)
# Z_region_b = mat_contents['Z_region_b']/1e9
Z = mat_contents['Z_region_b']/1e9

X = mat_contents['X_selected_region']/1e3
Y = mat_contents['Y_selected_region']/1e3
Y = Y[::-1, :]
# X = np.arange(0, m_per_pixel*Z_region_f.shape[1], m_per_pixel)
# Y = np.arange(0, m_per_pixel*Z_region_f.shape[0], m_per_pixel)
# X, Y = np.meshgrid(X, Y)
# Z = Z_region_b;
# Z = Z_region_b - np.min(Z_region_b)
# arr = arr[::-1, :]
# Xp = np.tile(np.arange(np.min(X), np.max(X)+m_per_pixel, m_per_pixel), (len(np.arange(np.min(Y), np.max(Y), m_per_pixel)), 1))
# Yp = np.tile(np.arange(np.min(Y), np.max(Y), m_per_pixel), (len(np.arange(np.min(X), np.max(X)+m_per_pixel, m_per_pixel)), 1)).T
# f = scipy.interpolate.RectBivariateSpline(Y[:,0].ravel(),X[0,:].ravel(),  Z)
# Zf = f(Yp[:,0].ravel(),Xp[0,:].ravel()).reshape(Xp.shape) / 1E9
# pixel_m = np.median(np.diff(Xp[0]))

# # Xp += (-111.465-12.6791-2.5-15)/1e3  #coordinate transfer

# X = Xp
# Y = Yp
# Z = Zf
pixel_m =  np.median(np.diff(X[0,:]))
# (Xp.ravel(), Yp.ravel(), Z_region_f)
fig = plt.figure(dpi=1800)
# ax = fig.add_subplot(121, projection='3d')

# ax.plot_surface(Xp, Yp, Zf , cmap = 'coolwarm')
# ax.view_init(80, -45)
# ax.view_init(elev=90, azim=-90)  # flat view
plt.pcolormesh(X*1e3, mat_contents['Y_selected_region'], Z*1e9)
plt.colorbar()  
plt.xlabel('x')
plt.ylabel('y')
plt.gca().invert_yaxis()
plt.show()

#%%

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

Z_ext = np.empty(X_ext.shape) * np.nan  # mark the Z_ext to NaN
Z_ext[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] = Z  # fill points
r = max(np.array(Z_ext.shape) - np.array(Z.shape)) // 2  # obtain the area of extension
u, v = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
# u, v = u.flatten(), v.flatten()
# _, rr = np.cartesian_to_polar(u, v)
# rr = scipy.spatial.distance.cdist(u.reshape(-1,1),v.shape(-1,1),'euclidean').reshape(u.shape)
coors = np.vstack((u.flatten(), v.flatten())).T
rr = np.linalg.norm(coors,axis=1).reshape(u.shape)
se = rr <= r
BW_Z = binary_dilation(~np.isnan(Z_0), structure=se)
id_ext = BW_Z == 0


if selection[0]:
    B, X_B, Y_B, _, _, T_0, _, X_P, Y_P, _, _, _, X_ca, Y_ca, _, _, _ = \
        DwellTime2D_FFT_Full_Test(
            X_ext, Y_ext, Z_0, 0, brf_params, brf_mode, X_brf, Y_brf, Z_avg, ca_range,
            pixel_m, tmin, tmax, options, ratio, False
        )
        
    T_0[id_ext] = 0
    # Z_removal_dw_0 = ConvFFT2(T_0, B)
    Z_removal_dw_0 = scipy.signal.fftconvolve(T_0, B,mode='same')
    Z_0 = remove_surface1(X_ext, Y_ext, Z_0)
    Z_0 = Z_0 - np.nanmin(Z_0)
    Z_residual_dw = Z_0 - np.nanmin(Z_0) - Z_removal_dw_0
    Z_residual_ca_0 = Z_residual_dw[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca_0 = remove_surface1(X_ca, Y_ca, Z_residual_ca_0)
#%%

# 2. Gaussian 
if selection[1]:
    _, _, Z_gauss, _ = Surface_Extension(X, Y, Z, brf_params, Z_avg, 'gauss', False)

    _, _, _, _, _, T_gauss, _, _, _, _, _, _, _, _, _, _, _ = \
        DwellTime2D_FFT_Full_Test(
            X_ext, Y_ext, Z_gauss, 0, brf_params, brf_mode, X_brf, Y_brf, Z_avg, ca_range,
            pixel_m, tmin, tmax, options, ratio, False
        )

    T_gauss[id_ext] = 0
    Z_removal_dw_gauss = scipy.signal.fftconvolve(T_gauss, B, mode='same')
    Z_gauss = remove_surface1(X_ext, Y_ext, Z_gauss)
    Z_gauss = Z_gauss - np.nanmin(Z_gauss)
    Z_residual_dw = Z_gauss - Z_removal_dw_gauss
    Z_residual_ca_gauss = Z_residual_dw[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca_gauss = remove_surface1(X_ca, Y_ca, Z_residual_ca_gauss)
#%%

# 3. 8nn
if selection[2]:
    _, _, Z_8nn, _ = Surface_Extension(X, Y, Z, brf_params, Z_avg, '8nn', False)

    _, _, _, _, _, T_8nn, _, _, _, _, _, _, _, _, _, _, _ = \
        DwellTime2D_FFT_Full_Test(
            X_ext, Y_ext, Z_8nn, 0, brf_params, brf_mode, X_brf, Y_brf, Z_avg, ca_range,
            pixel_m, tmin, tmax, options, ratio, False
        )

    T_8nn[id_ext] = 0
    Z_removal_dw_8nn = scipy.signal.fftconvolve(T_8nn, B, mode='same')
    Z_8nn = remove_surface1(X_ext, Y_ext, Z_8nn)
    Z_8nn = Z_8nn - np.nanmin(Z_8nn)
    Z_residual_dw = Z_8nn - Z_removal_dw_8nn
    Z_residual_ca_8nn = Z_residual_dw[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca_8nn = remove_surface1(X_ca, Y_ca, Z_residual_ca_8nn)

#%%

# 4. 8nn_fall

if selection[3]:
    _, _, Z_8nn_fall, _ = Surface_Extension(X, Y, Z, brf_params, Z_avg, '8nn', True)

    _, _, _, _, _, T_8nn_fall, _, _, _, _, _, _, _, _, _, _, _ = \
        DwellTime2D_FFT_Full_Test(
            X_ext, Y_ext, Z_8nn_fall, 0, brf_params, brf_mode, X_brf, Y_brf, Z_avg, ca_range,
            pixel_m, tmin, tmax, options, ratio, False
        )

    T_8nn_fall[id_ext] = 0
    Z_removal_dw_8nn_fall = scipy.signal.fftconvolve(T_8nn_fall, B, mode='same')
    Z_8nn_fall = remove_surface1(X_ext, Y_ext, Z_8nn_fall)
    Z_8nn_fall = Z_8nn_fall - np.nanmin(Z_8nn_fall)
    Z_residual_dw = Z_8nn_fall - Z_removal_dw_8nn_fall
    Z_residual_ca_8nn_fall = Z_residual_dw[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']]
    Z_residual_ca_8nn_fall = remove_surface1(X_ca, Y_ca, Z_residual_ca_8nn_fall)


#1
# fig_Z_0 = plt.figure(dpi=1800)
# plt.pcolormesh(X_ext*1e3, Y_ext*1e3, Z_0*1e9)

""" #1
fig_height_error = axs[0].pcolormesh(X_ext*1e3, Y_ext*1e3, Z_0*1e9)
axs[0].set_title('Extension height error')
fig.colorbar(fig_height_error,ax=axs[0])
# 2
fig_height_error = axs[1].pcolormesh(X_ext*1e3, Y_ext*1e3, T_0)
axs[1].set_title('dwell time')
axs[1].set_xlim(axs[0].get_xlim())
axs[1].set_ylim(axs[0].get_ylim())
fig.colorbar(fig_height_error,ax=axs[1])
# 3
axs[2].pcolormesh(X*1e3, Y*1e3, Z_residual_ca_0)
axs[2].set_title('residual')
axs[2].set_xlim(axs[0].get_xlim())
axs[2].set_ylim(axs[0].get_ylim())
fig.colorbar(fig_height_error,ax=axs[2])
# 4
axs[3].pcolormesh(X_ext*1e3, Y_ext*1e3, Z_removal_dw_0)
axs[3].set_title('removal')
axs[3].set_xlim(axs[0].get_xlim())
axs[3].set_ylim(axs[0].get_ylim())
fig.colorbar(fig_height_error,ax=axs[3])
# spacing
plt.tight_layout()

# show all subplots
plt.show() """
#%%
# fig, axs = plt.subplots(ncols=4, figsize=(15, 6))
# Assuming X, Y, Z, X_ext, Y_ext, Z_0, T_0, Z_residual_ca_0, Z_removal_dw_0 are given numpy arrays
# selection = np.array([1, 1, 1, 0])
# selection_savejson = np.array([0, 0, 0, 0])


map_height = 1 + np.count_nonzero(selection)
map_name = ['height error', 'dwell time', 'residual', 'removal']
grid = plt.GridSpec(map_height, 4)
fig = plt.figure("One-step surface extension dwell time results",figsize=(15, 10),dpi=800)

ax0 = fig.add_subplot(grid[0,1:3])
mesh0 = ax0.pcolormesh(X * 1e3, Y * 1e3, Z * 1e9, cmap='viridis')
ax0.set_aspect('equal')
c0 = plt.colorbar(mesh0, ax=ax0, pad=0.05)
c0.set_label('[nm]')
rms_Z = np.std(Z) * 1e9
ax0.set_title(f"Original Surface: PV = {np.round((np.max(Z) - np.min(Z)) * 1e9, 2)} nm, RMS = {np.round(rms_Z, 2)} nm")

fig.tight_layout()
fig.subplots_adjust(top=0.99, bottom=0.1, left=0.1, right=0.9, hspace=0.5, wspace=0.5)

if selection[0]:
    ax1 = fig.add_subplot(grid[1,0])
    mesh1 = ax1.pcolormesh(X_ext * 1e3, Y_ext * 1e3, Z_0 * 1e9, cmap='viridis')
    ax1.set_aspect('equal')
    c1 = plt.colorbar(mesh1, ax=ax1, pad=0.05)
    c1.set_label('[nm]')
    ax1.set_title(f"Zero extension: \nPV = {np.round((np.max(Z_0) - np.min(Z_0)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_0) * 1e9, 2)} nm")
    fig.gca().invert_yaxis()
    # fig.subplots_adjust(top=1, bottom=0.1, left=0.1, right=0.9, hspace=0.5, wspace=0.5)

    # Other subplots and operations...
    # ...
    ax2 = fig.add_subplot(grid[1,1])
    mesh2 = ax2.pcolormesh(X_ext * 1e3, Y_ext * 1e3, T_0 * 1e9, cmap='viridis')
    ax2.set_aspect('equal')
    c2 = plt.colorbar(mesh2, ax=ax2, pad=0.05)
    c2.set_label('[s]')
    ax2.set_title(f"Zero extension: \ndwell time =  {np.round((np.sum(T_0)), 2)} s")


    ax3 = fig.add_subplot(grid[1,2])
    mesh3 = ax3.pcolormesh(X * 1e3, Y * 1e3, Z_residual_ca_0 * 1e9, cmap='viridis')
    ax3.set_aspect('equal')
    c3 = plt.colorbar(mesh3, ax=ax3, pad=0.05)
    c3.set_label('[nm]')
    ax3.set_title(f"Zero extension: residual \nPV =  {np.round((np.max(Z_residual_ca_0) - np.min(Z_residual_ca_0)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_0) * 1e9, 2)} nm")
    ax3.set_xlim(ax2.get_xlim())
    ax3.set_ylim(ax2.get_ylim())
    
    ax4 = fig.add_subplot(grid[1,3])
    mesh4 = ax4.pcolormesh(X_ext * 1e3, Y_ext * 1e3, Z_removal_dw_0 * 1e9, cmap='viridis')
    ax4.set_aspect('equal')
    c4 = plt.colorbar(mesh4, ax=ax4, pad=0.05)
    c4.set_label('[nm]')
    ax4.set_title(f"Zero extension: removal \nPV =  {np.round((np.max(Z_removal_dw_0) - np.min(Z_removal_dw_0)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_0) * 1e9, 2)} nm")
fig.tight_layout()

if selection[1]:
    ax1 = fig.add_subplot(grid[2,0])
    mesh1 = ax1.pcolormesh(X_ext * 1e3, Y_ext * 1e3, Z_gauss * 1e9, cmap='viridis')
    ax1.set_aspect('equal')
    c1 = plt.colorbar(mesh1, ax=ax1, pad=0.05)
    c1.set_label('[nm]')
    ax1.set_title(f"Gaussian extension: \nPV = {np.round((np.max(Z_gauss) - np.min(Z_gauss)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_gauss) * 1e9, 2)} nm")
    # fig.subplots_adjust(top=1, bottom=0.1, left=0.1, right=0.9, hspace=0.5, wspace=0.5)

    # Other subplots and operations...
    # ...
    ax2 = fig.add_subplot(grid[2,1])
    mesh2 = ax2.pcolormesh(X_ext * 1e3, Y_ext * 1e3, T_gauss * 1e9, cmap='viridis')
    ax2.set_aspect('equal')
    c2 = plt.colorbar(mesh2, ax=ax2, pad=0.05)
    c2.set_label('[s]')
    ax2.set_title(f"Gaussian extension: \ndwell time =  {np.round((np.sum(T_gauss)), 2)} s")


    ax3 = fig.add_subplot(grid[2,2])
    mesh3 = ax3.pcolormesh(X * 1e3, Y * 1e3, Z_residual_ca_gauss * 1e9, cmap='viridis')
    ax3.set_aspect('equal')
    c3 = plt.colorbar(mesh3, ax=ax3, pad=0.05)
    c3.set_label('[nm]')
    ax3.set_title(f"Gaussian extension: residual \nPV =  {np.round((np.max(Z_residual_ca_gauss) - np.min(Z_residual_ca_gauss)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_gauss) * 1e9, 2)} nm")
    ax3.set_xlim(ax2.get_xlim())
    ax3.set_ylim(ax2.get_ylim())
    
    ax4 = fig.add_subplot(grid[2,3])
    mesh4 = ax4.pcolormesh(X_ext * 1e3, Y_ext * 1e3, Z_removal_dw_gauss * 1e9, cmap='viridis')
    ax4.set_aspect('equal')
    c4 = plt.colorbar(mesh4, ax=ax4, pad=0.05)
    c4.set_label('[nm]')
    ax4.set_title(f"Gaussian extension: removal \nPV =  {np.round((np.max(Z_removal_dw_gauss) - np.min(Z_removal_dw_gauss)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_gauss) * 1e9, 2)} nm")

fig.tight_layout()

if selection[2]:
    ax1 = fig.add_subplot(grid[3,0])
    mesh1 = ax1.pcolormesh(X_ext * 1e3, Y_ext * 1e3, Z_8nn * 1e9, cmap='viridis')
    ax1.set_aspect('equal')
    c1 = plt.colorbar(mesh1, ax=ax1, pad=0.05)
    c1.set_label('[nm]')
    ax1.set_title(f"8nn extension: \nPV = {np.round((np.max(Z_8nn) - np.min(Z_8nn)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_8nn) * 1e9, 2)} nm")
    # fig.subplots_adjust(top=1, bottom=0.1, left=0.1, right=0.9, hspace=0.5, wspace=0.5)

    # Other subplots and operations...
    # ...
    ax2 = fig.add_subplot(grid[3,1])
    mesh2 = ax2.pcolormesh(X_ext * 1e3, Y_ext * 1e3, T_8nn * 1e9, cmap='viridis')
    ax2.set_aspect('equal')
    c2 = plt.colorbar(mesh2, ax=ax2, pad=0.05)
    c2.set_label('[s]')
    ax2.set_title(f"8nn extension: \ndwell time =  {np.round((np.sum(T_8nn)), 2)} s")


    ax3 = fig.add_subplot(grid[3,2])
    mesh3 = ax3.pcolormesh(X * 1e3, Y * 1e3, Z_residual_ca_8nn * 1e9, cmap='viridis')
    ax3.set_aspect('equal')
    c3 = plt.colorbar(mesh3, ax=ax3, pad=0.05)
    c3.set_label('[nm]')
    ax3.set_title(f"8nn extension: residual \nPV =  {np.round((np.max(Z_residual_ca_8nn) - np.min(Z_residual_ca_8nn)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_8nn) * 1e9, 2)} nm")
    ax3.set_xlim(ax2.get_xlim())
    ax3.set_ylim(ax2.get_ylim())
    
    ax4 = fig.add_subplot(grid[3,3])
    mesh4 = ax4.pcolormesh(X_ext * 1e3, Y_ext * 1e3, Z_removal_dw_8nn * 1e9, cmap='viridis')
    ax4.set_aspect('equal')
    c4 = plt.colorbar(mesh4, ax=ax4, pad=0.05)
    c4.set_label('[nm]')
    ax4.set_title(f"8nn extension: removal \nPV =  {np.round((np.max(Z_removal_dw_8nn) - np.min(Z_removal_dw_8nn)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_8nn) * 1e9, 2)} nm")

fig.tight_layout()

if selection[3]:
    ax1 = fig.add_subplot(grid[4,0])
    mesh1 = ax1.pcolormesh(X_ext * 1e3, Y_ext * 1e3, Z_8nn_fall * 1e9, cmap='viridis')
    ax1.set_aspect('equal')
    c1 = plt.colorbar(mesh1, ax=ax1, pad=0.05)
    c1.set_label('[nm]')
    ax1.set_title(f"8nn_fall extension: \nPV = {np.round((np.max(Z_8nn_fall) - np.min(Z_8nn_fall)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_8nn_fall) * 1e9, 2)} nm")
    # fig.subplots_adjust(top=1, bottom=0.1, left=0.1, right=0.9, hspace=0.5, wspace=0.5)

    # Other subplots and operations...
    # ...
    ax2 = fig.add_subplot(grid[4,1])
    mesh2 = ax2.pcolormesh(X_ext * 1e3, Y_ext * 1e3, T_8nn_fall * 1e9, cmap='viridis')
    ax2.set_aspect('equal')
    c2 = plt.colorbar(mesh2, ax=ax2, pad=0.05)
    c2.set_label('[s]')
    ax2.set_title(f"8nn_fall extension: \ndwell time =  {np.round((np.sum(T_8nn_fall)), 2)} s")


    ax3 = fig.add_subplot(grid[4,2])
    mesh3 = ax3.pcolormesh(X * 1e3, Y * 1e3, Z_residual_ca_8nn_fall * 1e9, cmap='viridis')
    ax3.set_aspect('equal')
    c3 = plt.colorbar(mesh3, ax=ax3, pad=0.05)
    c3.set_label('[nm]')
    ax3.set_title(f"8nn_fall extension: residual \nPV =  {np.round((np.max(Z_residual_ca_8nn_fall) - np.min(Z_residual_ca_8nn_fall)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_8nn_fall) * 1e9, 2)} nm")
    ax3.set_xlim(ax2.get_xlim())
    ax3.set_ylim(ax2.get_ylim())
    
    ax4 = fig.add_subplot(grid[4,3])
    mesh4 = ax4.pcolormesh(X_ext * 1e3, Y_ext * 1e3, Z_removal_dw_8nn_fall * 1e9, cmap='viridis')
    ax4.set_aspect('equal')
    c4 = plt.colorbar(mesh4, ax=ax4, pad=0.05)
    c4.set_label('[nm]')
    ax4.set_title(f"8nn_fall extension: removal \nPV =  {np.round((np.max(Z_removal_dw_8nn_fall) - np.min(Z_removal_dw_8nn_fall)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_8nn_fall) * 1e9, 2)} nm")

fig.tight_layout()










plt.show()

#%%
# fig_Z_0 = plt.figure(dpi=1800)
# plt.pcolormesh(X_ext*1e3, Y_ext*1e3, Z_removal_dw_0*1e9)
# plt.colorbar()  
# plt.xlabel('x')
# plt.ylabel('y')
