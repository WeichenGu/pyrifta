# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:22:06 2023

@author: frw78547
"""

import numpy as np

def conv_fft2(S, K):
    # Compute the FFT padding size
    mS, nS = S.shape
    mK, nK = K.shape

    m = mS + mK - 1
    n = nS + nK - 1

    # Padding S and K
    S_padded = np.pad(S, ((0, m - mS), (0, n - nS)), mode='constant')
    K_padded = np.pad(K, ((0, m - mK), (0, n - nK)), mode='constant')

    # Perform FFT & convolution
    R = np.real(np.fft.ifft2(np.fft.fft2(S_padded) * np.fft.fft2(K_padded)))

    # Crop the correct portion to recover the same size of S
    R_cropped = conv_fft2_crop(R, mS, nS, mK, nK)

    return R_cropped

def conv_fft2_crop(R_crop, mS, nS, mK, nK):
    if mK % 2 == 1 and nK % 2 == 1:
        hmK = (mK - 1) // 2
        hnK = (nK - 1) // 2
        R_crop = R_crop[hmK:(mS + hmK), hnK:(nS + hnK)]

    elif mK % 2 == 0 and nK % 2 == 1:
        hmK = mK // 2
        hnK = (nK - 1) // 2
        R_crop = R_crop[hmK:(mS + hmK), hnK:(nS + hnK)]

    elif mK % 2 == 1 and nK % 2 == 0:
        hmK = (mK - 1) // 2
        hnK = nK // 2
        R_crop = R_crop[hmK:(mS + hmK), hnK:(nS + hnK)]

    else:
        hmK = mK // 2
        hnK = nK // 2
        R_crop = R_crop[hmK:(mS + hmK), hnK:(nS + hnK)]

    return R_crop
