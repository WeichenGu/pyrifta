# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 23:13:31 2023

@author: frw78547
"""

import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

def frequency_separate_dct(X_selected_region, Y_selected_region, Z_region_b, latRes, cutoff_freq, plot_para):
    # Fill NaN values using interpolation
    Z_filled = np.nan_to_num(Z_region_b, nan=np.nanmean(Z_region_b))
    
    # Perform 2D DCT
    Z_dct_filled = dct(dct(Z_filled.T, norm='ortho').T, norm='ortho')
    
    # Compute frequency ranges
    num_rows, num_cols = Z_filled.shape
    delta_freq = 1 / (num_rows * latRes)
    freq_x = np.arange(0, num_rows) * delta_freq
    delta_freq = 1 / (num_cols * latRes)
    freq_y = np.arange(0, num_cols) * delta_freq
    
    # Calculate the Nyquist frequency or choose a cutoff frequency
    nyquist_freq = 1 / (2 * latRes)
    if cutoff_freq > nyquist_freq:
        print('cutoff frequency is too high')
        cutoff_freq = nyquist_freq
    
    # Create a mask for low frequency components
    Y, X = np.meshgrid(freq_y, freq_x)
    low_freq_mask = (X**2 + Y**2) <= cutoff_freq**2
    
    # Apply the mask to get the low frequency and high frequency components
    low_freq_dct = Z_dct_filled * low_freq_mask
    high_freq_dct = Z_dct_filled * (~low_freq_mask)
    
    # Transform back to spatial domain
    low_freq_data = idct(idct(low_freq_dct.T, norm='ortho').T, norm='ortho')
    high_freq_data = idct(idct(high_freq_dct.T, norm='ortho').T, norm='ortho')
    
    # Plot the results
    if plot_para.lower() == 'yes':
        fig = plt.figure("separated", figsize=(16,10), dpi=600)

        # Original Filled Data
        ax1 = plt.subplot(2,2,(1,2))
        plt.pcolor(X_selected_region, Y_selected_region, Z_filled, shading='auto')
        
        plt.title(f'Original Filled Data: RMS = {np.std(Z_filled):.2f} nm')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.colorbar()
        # plt.axis('equal')
        # plt.tight_layout()
        ax1.set_aspect('equal')
        
        
        # Low Frequency Data
        ax2 = plt.subplot(2,2,3)
        plt.pcolor(X_selected_region, Y_selected_region, low_freq_data, shading='auto')
        plt.title(f'Low Frequency Data: RMS = {np.std(low_freq_data):.2f} nm')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.colorbar()
        # plt.axis('equal')
        plt.tight_layout()
        ax2.set_aspect('equal')

        # High Frequency Data
        ax3 = plt.subplot(2,2,4)
        plt.pcolor(X_selected_region, Y_selected_region, high_freq_data, shading='auto')
        plt.title(f'High Frequency Data: RMS = {np.std(high_freq_data):.2f} nm')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.colorbar()
        # plt.axis('equal')
        ax3.set_aspect('equal')
        plt.tight_layout()
        # fig.tight_layout()
        plt.show()
        # fig.subplots_adjust(top=0.99, bottom=0.1, left=0.1, right=0.8, hspace=0.5, wspace=0.4)

    return Z_filled, low_freq_data, high_freq_data

