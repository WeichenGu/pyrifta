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
from lib_rifta.ibf_engine.problem_func_dwell_grid import Problem_func_dwell_grid


# class MyCallback(Callback):

#     def __init__(self) -> None:
#         super().__init__()
#         self.data["best"] = []

#     def notify(self, algorithm):
#         self.data["best"].append(algorithm.pop.get("F").min())


def pymoo_minimize(gamma0, Z_to_remove, Z_to_remove_dw, B, dw_range, ca_range, use_DCT):
    myproblem = Problem_func_dwell_grid(gamma0, Z_to_remove, Z_to_remove_dw, B, dw_range, ca_range, use_DCT)
    
    # algorithm = PatternSearch(sampling = FloatRandomSampling())
    algorithm = PatternSearch(x0 = np.array([gamma0]))
    # termination = ("n_gen", 100)  # Run the algorithm for 100 generations
    # callback = AlgorithmCallback()
    res = minimize(myproblem,
                   algorithm,
                   # callback = MyCallback,
                   verbose = False,
                   save_history=True,
                   seed = 2
                   )
    
    print("Best solution found: \nX = {} \nF = {}".format(res.X, res.F))
    
    # val = res.algorithm.callback.data["best"]
    # plt.plot(np.arange(len(val)), val)
    val = [e.opt.get("F")[0] for e in res.history]
    plt.plot(np.arange(len(val)), val)

    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Function Value vs Iteration')
    plt.show()

    return res

