# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 00:25:57 2023

@author: frw78547
"""



import numpy as np

def dwell_time_2d_fft_inverse_filter_test(R, B, gamma, use_DCT):
    """
    2D Inverse filtering for deconvolution.
    Inputs:
        R: the filtered signal
        B: the kernel
        gamma: the filtering threshold
        use_DCT: whether to use DCT or not
    Output:
        T: the recovered original signal
    """

    mR, nR = R.shape

    if use_DCT:
        R_top = np.hstack((R, np.fliplr(R)))
        R_T = np.vstack((R_top, np.flipud(R_top)))

        mRt, nRt = R_T.shape

        FR = np.fft.fft2(R_T)
        FB = np.fft.fft2(B, s=(mRt, nRt))
    else:
        FR = np.fft.fft2(R)
        FB = np.fft.fft2(B, s=(mR, nR))

    sFB = np.where(np.abs(FB) > 0, FB, 1 / gamma)
    iFB = np.where(np.abs(sFB) * gamma > 1, 1 / sFB, gamma)

    T = np.real(np.fft.ifft2(iFB * FR))

    if use_DCT:
        T = T[:mR, :nR]
    
    # T[T < 0] = 0
    return T
