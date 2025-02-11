# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 22:14
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _test_pymoo3.py
# @Software: PyCharm
import enum
import time

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.core .callback import CallbackCollection,Callback
from _logging import  logger
from pymoo.core.algorithm import Algorithm
import os.path
import numpy
import pandas
import pymoo.util.display.display
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem

from pymoo.optimize import minimize

def f(x, y):
    # print("hello")
    sigmax, sigmay = 1, 2

    return numpy.exp(-(x ** 2 / (2 * sigmax ** 2) + (y - 0.5) ** 2 / (2 * sigmay ** 2)))
class MyProblem(Problem):
    def __init__(self, ):
        super().__init__(n_var=2, xl=[-10, -10], xu=[10, 10], )

    def _evaluate(self, x, out, *args, **kwargs):
        # z = anp.power(x, 2) - self.A * anp.cos(2 * anp.pi * x)
        kwargs['algorithm'].callback.data["test1"] = x - 2.

        out["F"] = -f(*x.T)


problem = MyProblem()

algorithm = NelderMead()
res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True,
               save_history =True,
               # display = MyDisplay()
               # output = MyOutput()
               # callback =  Recorder(Parameters(ParameterName),r"F:\Tecent\MyData\WeChat Files\wxid_7252352519612\FileStorage\File\2024-10\log.csv"),
               )
# res.algorithm.callback.to_disk()
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))



from scipy.optimize import minimize, OptimizeResult

i = 0
def my_callback (intermediate_result: OptimizeResult):
    global i
    i = i +1
    logger.info("i = %d, res.x = %s"%(i,intermediate_result))
res2 = minimize(lambda x:-f(x[0],x[1]),numpy.array((5,1)),method =  "Nelder-Mead",
                bounds=((-10, 10),(-10, 10)),callback=my_callback)
