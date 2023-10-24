import numpy as np

def BRFSuperGaussian2D(X, Y, t, params):
    """
    2D Super Gaussian Beam removal function model:
    Z(X, Y) = z0 + A*exp(-0.5*(((X-xc)/wx)^2 + ((Y-yc)/wy)^2)^p)

    Parameters:
    - X, Y: 2D x, y coordinate grids
    - t: 1D array of scalar of dwell time in seconds [s]
    - params: 1D array of the BRF parameters, e.g 
      [z0, A, wx, wy, xc1, yc1, p, xc2, yc2, p,...]

    Returns:
    - Z_fitted: 2D matrix or 1D array of the calculated 2D Super Gaussian function map [m]
    """
    if isinstance(t,list):
        pass
    else:
        t=[t]
    # Get the parameters
    A = params[0]
    sigmax = params[1][0]
    sigmay = params[1][1]
    ux = params[2][0::2]
    uy = params[2][1::2]
    p = params[-1]  # assume the same p for all instances

    # Initialize result
    Z_fitted = np.zeros(X.shape)

    for i in range(len(t)):
        Z_fitted = A * t[i] * np.exp(-0.5 * (((X[:, :] - ux[i]) / sigmax) ** 2 + ((Y[:, :] - uy[i]) / sigmay) ** 2) ** p)

    return Z_fitted
