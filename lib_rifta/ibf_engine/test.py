# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:02:12 2023

@author: frw78547
"""
import numpy as np
from pymoo.core.problem import ElementwiseProblem


class ElementwiseSphereWithConstraint(ElementwiseProblem):

    def __init__(self):
        xl = np.zeros(10)
        xl[0] = -5.0

        xu = np.ones(10)
        xu[0] = 5.0

        super().__init__(n_var=10, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum((x - 0.5) ** 2)
        out["G"] = np.column_stack([0.1 - out["F"], out["F"] - 0.5])
        
       
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from lib_rifta.ibf_engine.problem_func_dwell_grid import Problem_func_dwell_grid

problem = ElementwiseSphereWithConstraint()

algorithm = PatternSearch()

termination = ("n_gen", 100)  # Run the algorithm for 100 generations

res = minimize(problem,
               algorithm,
               verbose = False,
               seed=1)

print("Best solution found: \nX = {} \nF = {}".format(res.X, res.F))


#%%

from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.problems.single import Himmelblau
from pymoo.optimize import minimize


problem = Himmelblau()

algorithm = PatternSearch()

res = minimize(problem,
               algorithm,
               verbose=False,
               seed=1)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
