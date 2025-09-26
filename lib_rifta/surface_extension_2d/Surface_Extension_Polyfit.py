# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:40:07 2023

@author: Etrr
"""

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import lsq_linear
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import CubicSpline
from scipy.interpolate import RegularGridInterpolator

from lib_rifta.surface_extension_2d.Surface_Extension_Smooth import Surface_Extension_Smooth
from lib_rifta.surface_extension_2d.Surface_Extension_SmoothBSpline import Surface_Extension_SmoothBSpline
from lib_rifta.surface_extension_2d.Surface_Extension_GP import Surface_Extension_GP
from lib_rifta.surface_extension_2d.Chebyshev_XYnm import Chebyshev_XYnm
from lib_rifta.surface_extension_2d.Legendre_XYnm import Legendre_XYnm
from lib_rifta.surface_extension_2d.Hermite_XYnm import Hermite_XYnm
from lib_rifta.surface_extension_2d.Laguerre_XYnm import Laguerre_XYnm

# from pymoo.core.problem import ElementwiseProblem
# from pymoo.optimize import minimize
# from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
# from pymoo.core.population import Population
# from pymoo.core.individual import Individual

from numpy.polynomial import chebyshev as cheb

def Surface_Extension_Polyfit(
        X, Y, Z,               # 2D numpy arrays: unextended surface error map
        tif_mpp,               # TIF sampling interval [m/pxl]
        Z_tif,                 # TIF profile (2D numpy array)
        order_m, order_n,      # polynomial orders in y, x
        poly_type,              # 'Chebyshev' or 'Legendre'
        ext_rx = 0.5, ext_ry = 0.5
    ):
    """
    Extend the surface error map using polynomial fitting.

    Parameters:
        X, Y, Z : np.ndarray
            Unextended surface error map.
        tif_mpp : float
            TIF sampling interval [m/pxl].
        Z_tif : np.ndarray
            TIF profile.
        order_m, order_n : int
            Polynomial orders in y, x.
        poly_type : str
            Type of polynomial ('Chebyshev' or 'Legendre').

    Returns:
        X_ext, Y_ext, Z_ext : np.ndarray
            Extended surface error map.
        ca_range : dict
            Dictionary containing y and x start & end ids of CA in FA [pixel].
    """
    # 0. Obtain required parameters
    surf_mpp = np.median(np.diff(X[0, :]))
    m, n = Z.shape
    # 20240110 floor -- ceil ? at least X_ext, Y_ext >= X, Y + Z_tif.shape[?])*0.5
    # 20241123 np.round, Preventing data accuracy problems from "tif_mpp*(Z_tif.shape[0])*ext_ry/surf_mpp,decimals=4)"
    m_ext = int(np.ceil(np.round(tif_mpp*(Z_tif.shape[0])*ext_ry/surf_mpp,decimals=4)))  # extension size in y [pixels]
    n_ext = int(np.ceil(np.round(tif_mpp*(Z_tif.shape[1])*ext_rx/surf_mpp,decimals=4)))  # extension size in x [pixels]

    ca_range = {
        'y_s': m_ext,
        'y_e': m_ext + m,
        'x_s': n_ext,
        'x_e': n_ext + n
    }

    # 1. Initial extension matrices
    X_ext, Y_ext = np.meshgrid(np.arange(-n_ext, n + n_ext), np.arange(-m_ext, m + m_ext))
    X_ext = X_ext * surf_mpp + X[0, 0]
    Y_ext = Y_ext * surf_mpp + Y[-1, -1]
    Z_ext = np.full(X_ext.shape, np.nan)
    Z_ext[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] = Z

    # Fit the edge values
    w = 100
    # Z_ext[:, 0] = 0
    # Z_ext[0, :] = 0
    # Z_ext[:, -1] = 0
    # Z_ext[-1, :] = 0

    # _, _, Z_ext_boundry, _ = Surface_Extension_GP(X, Y, Z, tif_mpp, Z_tif, np.arange(-18, 19), np.arange(-3, 4))
    _, _, Z_ext_boundry, _ = Surface_Extension_Smooth(X, Y, Z, tif_mpp, Z_tif, ext_rx, ext_ry)
    
    # _, _, Z_ext_boundry, _ = Surface_Extension_SmoothBSpline(X, Y, Z, tif_mpp, Z_tif, ext_rx, ext_ry)
    if poly_type != 'Interp2':
        
        Z_ext[:, 0] = Z_ext_boundry[:, 0]
        Z_ext[0, :] = Z_ext_boundry[0, :]
        Z_ext[:, -1] = Z_ext_boundry[:, -1]
        Z_ext[-1, :] = Z_ext_boundry[-1, :]
        
        # Z_ext[0, :] = interpolate_nan_values(Z_ext_boundry[0, :], kind='cubic')
        # Z_ext[-1, :] = interpolate_nan_values(Z_ext_boundry[-1, :], kind='cubic')
        # Z_ext[:, 0] = interpolate_nan_values(Z_ext_boundry[:, 0], kind='cubic')
        # Z_ext[:, -1] = interpolate_nan_values(Z_ext_boundry[:, -1], kind='cubic')
    
    else:
        # 20250319 change
        # 左上角
        Z_ext_boundry[0, 0] = (Z_ext_boundry[0, :].mean() + Z_ext_boundry[:, 0].mean()) / 2
    
        # 右上角
        Z_ext_boundry[0, -1] = (Z_ext_boundry[0, :].mean() + Z_ext_boundry[:, -1].mean()) / 2
    
        # 左下角
        Z_ext_boundry[-1, 0] = (Z_ext_boundry[-1, :].mean() + Z_ext_boundry[:, 0].mean()) / 2
    
        # 右下角
        Z_ext_boundry[-1, -1] = (Z_ext_boundry[-1, :].mean() + Z_ext_boundry[:, -1].mean()) / 2
        
        Z_ext_boundry[1:ca_range['y_s'],0:ca_range['x_s']] = np.nan
        Z_ext_boundry[ca_range['y_e']+1:-1,0:ca_range['x_s']] = np.nan
        
        Z_ext_boundry[1:ca_range['y_s'],ca_range['x_e']+1:-1] = np.nan
        Z_ext_boundry[ca_range['y_e']+1:-1,ca_range['x_e']+1:-1] = np.nan
        
        Z_ext_boundry[0:ca_range['y_s'],1:ca_range['x_s']] = np.nan
        Z_ext_boundry[0:ca_range['y_s'], ca_range['x_e']+1:-1] = np.nan
        
        Z_ext_boundry[ca_range['y_e']+1:-1,1:ca_range['x_s']] = np.nan
        Z_ext_boundry[ca_range['y_e']+1:-1,ca_range['x_e']+1:-1] = np.nan
        
        # 20250319 change
        
        weights = np.zeros_like(Z_ext_boundry[:, 0])
        weights[ca_range['y_s']:ca_range['y_e']+1] = 10.0  # 中间区域权重为 1
        weights[0] = 10.0
        weights[-1] = 10.0
        weights[weights == np.nan] = 0.1  # 其他区域权重为 0.1
        Z_ext[:, 0] = interpolate_nan_values(Z_ext_boundry[:, 0], kind='cubic')
        Z_ext[:, -1] = interpolate_nan_values(Z_ext_boundry[:, -1], kind='cubic')
        # nan_mask = np.isfinite(Z_ext_boundry[:, 0])
        # Z_ext[:, 0] = np.polyval(polynomial_fit(Z_ext_boundry[:, 0][nan_mask], weights[nan_mask], degree=order_n), np.linspace(-1, 1, len(Z_ext_boundry[:, 0])))
        # Z_ext[:, -1] = np.polyval(polynomial_fit(Z_ext_boundry[:, -1][nan_mask], weights[nan_mask], degree=order_n), np.linspace(-1, 1, len(Z_ext_boundry[:, -1])))
    
        
        # Z_ext[:, 0] = cheb.chebval(np.linspace(-1, 1, len(Z_ext_boundry[:, 0])), chebyshev_fit(Z_ext_boundry[:, 0], weights, degree=order_n))
        # Z_ext[:, -1] = cheb.chebval(np.linspace(-1, 1, len(Z_ext_boundry[:, -1])), chebyshev_fit(Z_ext_boundry[:, -1], weights, degree=order_n))
        
        weights = np.zeros_like(Z_ext_boundry[0, :])
        weights[ca_range['x_s']:ca_range['x_e']+1] = 10.0  # 中间区域权重为 1
        weights[0] = 10.0
        weights[-1] = 10.0
        weights[weights == np.nan] = 0.1  # 其他区域权重为 0.1
        
        Z_ext[0, :] = interpolate_nan_values(Z_ext_boundry[0, :], kind='cubic')
        Z_ext[-1, :] = interpolate_nan_values(Z_ext_boundry[-1, :], kind='cubic')

    
    # nan_mask = np.isfinite(Z_ext_boundry[0, :])
    # Z_ext[0, :] = np.polyval(polynomial_fit(Z_ext_boundry[0, :][nan_mask], weights[nan_mask], degree=order_m), np.linspace(-1, 1, len(Z_ext_boundry[0, :])))
    # Z_ext[-1, :] = np.polyval(polynomial_fit(Z_ext_boundry[-1, :][nan_mask], weights[nan_mask], degree=order_m), np.linspace(-1, 1, len(Z_ext_boundry[-1, :])))

    # Z_ext[0, :] = cheb.chebval(np.linspace(-1, 1, len(Z_ext_boundry[0, :])), chebyshev_fit(Z_ext_boundry[0, :], weights, degree=order_m))
    # Z_ext[-1, :] = cheb.chebval(np.linspace(-1, 1, len(Z_ext_boundry[-1, :])), chebyshev_fit(Z_ext_boundry[-1, :], weights, degree=order_m))

    
    # # 20240119 test
    # # 创建一个全是 True 的掩码，尺寸与 Z_ext 相同
    # mask = np.ones_like(Z_ext, dtype=bool)
    
    # # 将中心区域设置为 False
    # mask[ca_range['y_s']:ca_range['y_e'], ca_range['x_s']:ca_range['x_e']] = False
    
    # # 现在，mask 中 True 的位置对应于需要从 Z_ext_boundry 替换的部分
    # Z_ext[mask] = Z_ext_boundry[mask]
    # # 20240119 test
    
    # 权重矩阵 W
    # W = np.ones(Z_ext.shape)  # 空的地方权重为1
    '''  20250322 更改权重方式    '''
    W = np.zeros(Z_ext.shape)  # 空的地方权重为0
    W[:, 0] = w
    W[0, :] = w
    W[:, -1] = w
    W[-1, :] = w

    
    W[ca_range['y_s']:ca_range['y_e']+1, ca_range['x_s']:ca_range['x_e']+1] = w;
    # 2. Poly fit
    p, q = np.meshgrid(np.arange(order_m + 1), np.arange(order_n + 1))  #  20240108 m n reverse,fixed it
    X_nor = -1 + 2 * (X_ext - X_ext.min()) / (X_ext.max() - X_ext.min())
    Y_nor = -1 + 2 * (Y_ext - Y_ext.min()) / (Y_ext.max() - Y_ext.min())

    if poly_type == 'Chebyshev':

        z3, _, _ = Chebyshev_XYnm(X_nor, Y_nor, p.ravel(), q.ravel())
    elif poly_type == 'Legendre':
        # print('not finished legendre yet')
        z3, _, _ = Legendre_XYnm(X_nor, Y_nor, p.ravel(), q.ravel())
        
    elif poly_type == 'Hermite':
        # 标准化
        X_nor = (X_ext - np.mean(X_ext)) / np.std(X_ext)
        Y_nor = (Y_ext - np.mean(Y_ext)) / np.std(Y_ext)
        z3, _, _ = Hermite_XYnm(X_ext.copy(), Y_ext.copy(), p.ravel(), q.ravel())
    elif poly_type == 'Laguerre':
        z3, _, _ = Laguerre_XYnm(X_ext+np.min(X_ext), Y_ext-np.min(Y_ext), p.ravel(), q.ravel())
    elif poly_type == 'Interp2':
            # 获取非 NaN 值的索引
        valid_mask = ~np.isnan(Z_ext)
        
        # 提取有效的坐标和值
        X_valid = X_ext[valid_mask]
        Y_valid = Y_ext[valid_mask]
        Z_valid = Z_ext[valid_mask]
        
        # 创建插值函数
        Z_ext = griddata(
            (X_valid, Y_valid),  # 有效数据点的坐标
            Z_valid,             # 有效数据点的值
            (X_ext, Y_ext),      # 需要插值的坐标
            method='cubic'       # 插值方法，可选 'linear', 'nearest', 'cubic'
        )
        
        
        
    else:
        raise ValueError('Unknown polynomial type.')
        
        
    '''
# 20231011 added nonlinear ls
    # Reshape z3 for matrix operations
    z3_res = z3.reshape((-1, z3.shape[-1]))
    
    # Define A and b using the reshaped z3 and Z_ext
    A = z3_res[~np.isnan(Z_ext.ravel()), :]
    b = Z_ext[~np.isnan(Z_ext)]
    
    # Initial guess for parameters
    c_init = np.ones(A.shape[1])
    
    # Use least_squares to find the best-fitting parameters
    result = least_squares(lambda c: np.dot(A, c) - b, c_init, method='lm')
    c_opt = result.x
    
    # Apply the optimized coefficients to each term in z3 and sum them up
    Z_ext = np.sum(z3 * c_opt.reshape(1, 1, -1), axis=2)

    
    
     '''
    
    '''
    #20240225 added
    lambda_reg = 5
    # Reshape z3 for matrix operations
    z3_res = z3.reshape((-1, z3.shape[-1]))
   
    # Define A and b using the reshaped z3 and Z_ext
    A = z3_res[~np.isnan(Z_ext.ravel()), :]
   
    # 应用权重到设计矩阵A和目标向量b
    A_nonan = A[~np.isnan(Z_ext.ravel()), :]
    b_nonan = Z_ext[~np.isnan(Z_ext)].ravel()
    W_nonan = W[~np.isnan(Z_ext)].flatten()
    A_weighted = A_nonan * np.sqrt(W_nonan)[:, np.newaxis]
    b_weighted = b_nonan * np.sqrt(W_nonan)

    # 引入正则化项
    n_coeffs = A_nonan.shape[1]
    A_reg = np.vstack([A_weighted, np.sqrt(lambda_reg) * np.eye(n_coeffs)])
    b_reg = np.concatenate([b_weighted, np.zeros(n_coeffs)])

    # 使用scipy.linalg.lstsq进行拟合，求解系数c
    c, _, _, _ = lstsq(A_reg, b_reg)

    for i in range(len(c)):
        z3[:, :, i] = z3[:, :, i] * c[i]

    Z_ext = z3.sum(axis=2)
    '''


    
    '''
    # Initial guess for c
    c_initial = np.zeros(A.shape[1])
    
    # Bounds for c
    bounds = [(-2.5e-8, None) for _ in range(A.shape[1])]
    
    # Solve the constrained optimization problem
    result = minimize(objective, c_initial, args=(A, b),method = 'Newton-CG', bounds=bounds)
    
    # Extract the optimized c values
    c_optimized = result.x
    '''

    '''
    z3_res = z3.reshape((-1, z3.shape[-1]))

    A = z3_res[~np.isnan(Z_ext.ravel()), :]
    b = Z_ext[~np.isnan(Z_ext)]
    
    
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    '''
    if poly_type != 'Interp2':
        z3_res = z3.reshape((-1, z3.shape[-1]))
        W_flat = W.ravel()
        weights = W_flat[~np.isnan(Z_ext.ravel())]
        A = z3_res[~np.isnan(Z_ext.ravel()), :]
        b = Z_ext[~np.isnan(Z_ext)]
        
        A = A * weights[:, np.newaxis]
        b = b * weights  # Z_ext 展开一维后的个数，也是目标
        
        # # m1.使用最小二乘拟合，考虑权重
        c, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        mse = residuals
        # print(f"Mean Squared Error (MSE): {mse:.4f}")
        print("Mean Squared Error (MSE): ",mse)

        
    
    # # m2.带约束的拟合，考虑权重，设0回直接返回0，陷入局部最优？
    # # 
    # def objective(c, A, b):
    #     residuals = A @ c - b
    #     return np.sum(residuals ** 2)


    # # constraints = [
    # #     {'type': 'ineq', 'fun': lambda c: 1.5*np.max(Z) - np.dot(A, c)},  # 确保 A @ c <= z_max
    # #     {'type': 'ineq', 'fun': lambda c: np.dot(A, c) - 1.5*np.min(Z)}   # 确保 A @ c >= z_min
    # # ]

    
    # initial_guess = np.random.uniform(-1e-9, 1e-9, size=A.shape[1])

    # # result = minimize(objective, initial_guess, args=(A, b), constraints=constraints)
    # result = minimize(objective, initial_guess, args=(A, b), bounds=[(1.5*np.min(Z), 1.5*np.max(Z))] * A.shape[1], method='L-BFGS-B')

    
    # # 获取拟合后的系数 c
    # c = result.x
    
    
    # m3. lsq  容易被中频干扰
    # result = lsq_linear(A, 
    #                     b, 
    #                     bounds=(np.min(Z)-0.2*np.abs(np.min(Z)), np.max(Z)+0.2*np.abs(np.max(Z))), 
    #                     method='bvls')
    # # 获取拟合后的系数 c
    # c = result.x
    
        '''
        def objective(c, A, b, z_min, z_max):
            # 目标函数是最小化残差平方和
            residuals = A @ c - b
            residual_rms = np.sqrt(np.mean(residuals ** 2))
            # return np.sum(residuals ** 2)
            return residual_rms
        
        # 定义加权目标函数并加入罚函数
        sum_constraint = {
            'type': 'ineq', 
            'fun': lambda c: z_max - np.sum(c[c > 0])  # 确保 Σc ≤ c_max
        }
        
        sum_constraint_lower = {
            'type': 'ineq', 
            'fun': lambda c: np.sum(c[c < 0]) - z_min  # 确保 Σc ≥ c_min
        }
        constraints = [sum_constraint, sum_constraint_lower]
    
    
        def callback(c):
            print(f"Current coefficients sum: {np.sum(c[c < 0])} to {np.sum(c[c > 0])}")
        
        # 定义较为宽松的边界条件
        z_min = 1.1 * np.min(Z)  # 放宽最小值约束
        z_max = 1.1 * np.max(Z)  # 放宽最大值约束
        
        # 设置初始猜测
        initial_guess = np.linalg.lstsq(A, b, rcond=None)[0]
    
        
        
        # 调用 minimize 函数，使用罚函数作为目标函数
        result = minimize(
            objective,
            initial_guess, 
            args=(A, b, z_min, z_max),
            constraints=constraints,
            method='SLSQP', 
            callback=callback, 
            options={
                'ftol': 1e-10,
                'disp': True
                })
        
        # 获取拟合后的系数 c
        c = result.x
        '''
        
        
        '''
        # m.4. Define optimization problem using pymoo
        class MyOptimizationProblem(ElementwiseProblem):
        
            def __init__(self, A, b, z3, z_min, z_max):
                self.A = A
                self.b = b
                self.z3 = z3
                self.z_min = z_min
                self.z_max = z_max
                super().__init__(n_var=A.shape[1],
                                 n_obj=1,
                                 n_ieq_constr=2,  # 两个不等式约束
                                 xl=z_min,
                                 xu=z_max)
        
            def _evaluate(self, c, out, *args, **kwargs):
                # 计算 Z_ext
                Z_ext = np.sum(self.z3 * c[np.newaxis, np.newaxis, :], axis=2)
        
                # 计算残差 RMS
                residual = self.A @ c - self.b
                # residual_rms = np.sqrt(np.mean(residual ** 2))
        
                # 目标函数为残差的 RMS
                out["F"] = np.sum(residual ** 2)
        
                # 定义约束条件
                # 确保正值和小于等于 z_max
                sum_pos = np.sum(c[c > 0])
                # 确保负值和大于等于 z_min
                sum_neg = np.sum(c[c < 0])
        
                # 不等式约束，确保 sum_pos ≤ z_max, sum_neg ≥ z_min
                out["G"] = [sum_pos - z_max, z_min - sum_neg]
        
        # 初始化问题参数
        z_min = 1.1 * np.min(Z)  # 例如，目标的最小值
        z_max = 1.1 * np.max(Z)  # 例如，目标的最大值
        
        # 定义优化问题
        problem = MyOptimizationProblem(A=A, b=b, z3=z3, z_min=z_min, z_max=z_max)
        
        # 使用模式搜索（Pattern Search）算法进行优化
        algorithm = PatternSearch()
        
        # 设置一个初始解
        initial_guess = np.linalg.lstsq(A, b, rcond=None)[0]  # 这里可以根据你的需求设置，比如使用零向量作为初始解，或者是其他你认为合理的初始值
        
        # 创建初始种群
        initial_population = Population().create(Individual(X=initial_guess))
        # 设置算法的初始种群
        algorithm.initial_population = initial_population
    
        # 调用 minimize 函数进行优化
        res = minimize(problem,
                       algorithm,
                       seed=1,
                       verbose=True,
                       save_history=True)
        
        # 获取拟合后的系数 c
        c = res.X
        '''
        
    
        for i in range(len(c)):
            z3[:, :, i] = z3[:, :, i] * c[i]
    
        Z_ext = z3.sum(axis=2)
        
    else:
        print("Interp2d")

    
    return X_ext, Y_ext, Z_ext, ca_range


def chebyshev_fit(data, weights=None, degree=3):
    """
    对输入数据进行切比雪夫多项式拟合
    :param data: 输入数据（一维数组）
    :param degree: 多项式阶数
    :return: 拟合后的多项式函数
    """
    if weights is None:
        weights = np.ones_like(data)
    
    
    x = np.linspace(-1, 1, len(data))  # 归一化到 [-1, 1] 区间
    coeffs = cheb.chebfit(x, data, deg=degree,w=weights)  # 切比雪夫拟合
    return coeffs

def polynomial_fit(data, weights=None, degree=3):
    """
    对输入数据进行普通多项式拟合
    :param data: 输入数据（一维数组）
    :param weights: 权重（一维数组，与 data 形状相同，默认为全 1）
    :param degree: 多项式阶数
    :return: 拟合后的多项式系数
    """
    if weights is None:
        weights = np.ones_like(data)
    
    x = np.linspace(-1, 1, len(data))  # 归一化到 [-1, 1] 区间
    coeffs = np.polyfit(x, data, deg=degree, w=weights)  # 普通多项式拟合
    return coeffs

def interpolate_nan_values(data, kind='linear'):
    """
    使用插值填充一维数据中的 NaN 值。

    参数:
        data (np.ndarray): 输入的一维数据，可能包含 NaN 值。
        kind (str): 插值方法，默认为 'linear'，可选 'cubic' 等。

    返回:
        np.ndarray: 填充 NaN 值后的一维数据。
    """
    # 找到非 NaN 的索引
    valid_mask = ~np.isnan(data)
    x_valid = np.arange(len(data))[valid_mask]
    y_valid = data[valid_mask]
    
    # 如果所有值都是 NaN，直接返回原数据
    if len(x_valid) == 0:
        return data

    # 创建插值函数
    # f = interp1d(x_valid, y_valid, kind=kind, fill_value="extrapolate")
    # f = Akima1DInterpolator(x_valid, y_valid)
    f = CubicSpline(x_valid, y_valid)
    # 填充 NaN 值
    return f(np.arange(len(data)))



