# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:51:46 2023

@author: frw78547
"""

import numpy as np

def Surface_Extension_GerchbergPapoulis(u0, G, Gy, Gox, Goy, rms_thrd=1e-9, max_iter=500):
    """
    Function to perform the improved 2D Gerchberg-Papoulis bandlimited
    surface extrapolation algorithm. 

    Reference:
    Marks, R. J. (1981). Gerchberg’s extrapolation algorithm in two 
    dimensions. Applied optics, 20(10), 1815-1820. 
    """

    u_pre = u0 * G
    v_pre = u_pre
    w_pre = u_pre

    i = 1
    while i <= max_iter:
        wk = (1 - G) * np.fft.ifft(np.fft.ifftshift(Gox * np.fft.fftshift(np.fft.fft(w_pre, axis=1), axes=1), axes=1), axis=1)
        vk = (1 - Gy) * np.fft.ifft(np.fft.ifftshift(Goy * np.fft.fftshift(np.fft.fft(v_pre, axis=0), axes=0), axes=0), axis=0) + wk
        uk = np.real(u_pre + vk)

        if np.nanstd(uk - u_pre) <= rms_thrd:
            break

        u_pre = uk
        v_pre = vk
        w_pre = wk
        i += 1

    return uk


'''

    # Iterative update
    i = 1
    while i <= max_iter:
        wk = (1 - G) * np.fft.ifft(np.fft.ifftshift(Gox * np.fft.fftshift(np.fft.fft(w_pre, axis=2), axes=2), axes=2), axis=2)
        vk = (1 - Gy) * np.fft.ifft(np.fft.ifftshift(Goy * np.fft.fftshift(np.fft.fft(v_pre, axis=1), axes=1), axes=1), axis=1) + wk
        uk = u_pre + vk

        # Early stop when the rms difference is satisfied 
        if np.nanstd(uk - u_pre, ddof=1) <= rms_thrd:
            break

        u_pre = uk
        v_pre = vk
        w_pre = wk
        i += 1

    return uk
'''