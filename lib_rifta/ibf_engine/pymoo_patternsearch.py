# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:51:59 2023

@author: frw78547
"""
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
# from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize

# from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
# from pymoo.operators.sampling.rnd import FloatRandomSampling
from lib_rifta.ibf_engine.problem_func import Problem_func_entire
from lib_rifta.ibf_engine.problem_func import Problem_func_dwell_grid
from lib_rifta.ibf_engine.problem_func import Problem_func_cutoff_Freq


# class MyCallback(Callback):

#     def __init__(self) -> None:
#         super().__init__()
#         self.data["best"] = []

#     def notify(self, algorithm):
#         self.data["best"].append(algorithm.pop.get("F").min())


def pymoo_minimize_entire(gamma0, Z_to_remove, B, dw_range, ca_range, use_DCT, iter_show = False):
    myproblem = Problem_func_entire(gamma0, Z_to_remove, B, dw_range, ca_range, use_DCT)
    # algorithm = PatternSearch(sampling = FloatRandomSampling())
    algorithm = PatternSearch(x0 = np.array([gamma0]))
    # init_delta = 1
    # termination = ("n_gen", 100)  # Run the algorithm for 100 generations
    # callback = AlgorithmCallback()
    res = minimize(myproblem,
                   algorithm,
                   # callback = MyCallback,
                   verbose = False,
                   save_history=iter_show,
                   seed = 1
                   )
        
    
    val = [e.opt.get("F")[0] for e in res.history]
    if iter_show:
        print("Best solution found: \nX = {} \nF = {}".format(res.X, res.F))
        plt.plot(np.arange(len(val)), val)
    
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.title('Function Value vs Iteration')
        plt.show()   
    return res


def pymoo_minimize_dwell_grid(gamma0, Z_to_remove, Z_to_remove_dw, B, dw_range, ca_range, use_DCT, iter_show = False):
    myproblem = Problem_func_dwell_grid(gamma0, Z_to_remove, Z_to_remove_dw, B, dw_range, ca_range, use_DCT)

    # algorithm = PatternSearch(sampling = FloatRandomSampling())
    algorithm = PatternSearch(x0 = np.array([gamma0]) )
    # init_delta = 1
    # termination = ("n_gen", 100)  # Run the algorithm for 100 generations
    # callback = AlgorithmCallback()
    res = minimize(myproblem,
                   algorithm,
                   # callback = MyCallback,
                   verbose = False,
                   save_history=iter_show,
                   seed = 1
                   )
    

    
    # val = res.algorithm.callback.data["best"]
    # plt.plot(np.arange(len(val)), val)
    val = [e.opt.get("F")[0] for e in res.history]
    
    
    
    if iter_show:
        print("Best solution found: \nX = {} \nF = {}".format(res.X, res.F))
        plt.plot(np.arange(len(val)), val)
    
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.title('Function Value vs Iteration')
        plt.show()    
        
        
    return res



def pymoo_minimize_cutoff_freq(cutoff_freq, X, Y, Z, brf_params, Z_tif, init_ext_method, ini_is_fall, ini_fu_range, ini_fv_range, ini_order_m, ini_order_n, ini_type, du_ibf, brf_mode, X_tif, Y_tif, iter_show=False):

    class cf_Callback(Callback):

        def __init__(self) -> None:
            super().__init__()
            self.data = {"Z_low_freq_data": []}

        def notify(self, algorithm):
            self.data["Z_low_freq_data"].append(algorithm.problem.Z_low_freq_data)
    
    
    cf_cb_instance = cf_Callback()
    myproblem = Problem_func_cutoff_Freq(cutoff_freq, X, Y, Z, brf_params, Z_tif, init_ext_method, ini_is_fall, ini_fu_range, ini_fv_range, ini_order_m, ini_order_n, ini_type, du_ibf, brf_mode, X_tif, Y_tif)
    
    algorithm = PatternSearch(x0 = np.array([cutoff_freq]),init_delta=0.25, init_rho=0.5, step_size=0.1)
    
    res = minimize(myproblem,
                   algorithm,
                   verbose=False,
                   save_history=iter_show,
                   seed=1,
                   callback=cf_cb_instance
                   )
    

    val = [e.opt.get("F")[0] for e in res.history]
        
    x_vals = [e.opt.get("X")[0] for e in res.history]
    
    print("Best solution found: \nX = {} \nF = {}".format(res.X, res.F))

    if iter_show:
        fig, ax1 = plt.subplots()  
    
        color = 'tab:red'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Function Value', color=color)
        ax1.plot(np.arange(len(val)), val, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
    
        ax2 = ax1.twinx()  
    
        color = 'tab:blue'
        ax2.set_ylabel('Variable X', color=color)
        ax2.plot(np.arange(len(x_vals)), x_vals, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
    
        plt.title('Function Value and Variable X vs Iteration')
        plt.show()  
        
    # ts = np.std(cf_cb_instance.data["Z_low_freq_data"])

    return res
# result = pymoo_minimize_cutoff_freq(...)
