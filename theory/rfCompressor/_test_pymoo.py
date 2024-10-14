# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 17:40
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _test_pymoo.py
# @Software: PyCharm
import enum
import time
import matplotlib
import numpy
import pandas
import pyomo.environ as pyo
import sys
sys.path.append(r"D:\ipopt\Ipopt-3.11.1-win64-intel13.1\bin")
def ext_fcn(a, b):
    return pyo.sin(a - b)
def grad_ext_fcn(args, fixed):
    a, b = args[:2]
    return [ pyo.cos(a - b), -pyo.cos(a - b) ]
def create_model():
    m = pyo.ConcreteModel()
    m.name = 'Example 1: Eason'
    m.z = pyo.Var(range(3), domain=pyo.Reals, initialize=2.)
    m.x = pyo.Var(range(2), initialize=2.)
    m.x[1] = 1.0

    m.ext_fcn = pyo.ExternalFunction(ext_fcn, grad_ext_fcn)

    m.obj = pyo.Objective(
        expr=(m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 \
           + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6
    )

    m.c1 = pyo.Constraint(
        expr=m.x[0] * m.z[0]**2 + m.ext_fcn(m.x[0], m.x[1]) == 2*pyo.sqrt(2.0)
        )
    m.c2 = pyo.Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+pyo.sqrt(2.0))
    return m
model = create_model()
# trf_solver = pyo.SolverFactory('trustregion',# executable=r"D:\ipopt\Ipopt-3.11.1-win64-intel13.1\bin\ipopt.exe"
#                                )
trf_solver = pyo.SolverFactory('ipopt',# executable=r"D:\ipopt\Ipopt-3.11.1-win64-intel13.1\bin\ipopt.exe"
                               )
# === Solve with TRF ===
result = trf_solver.solve(model, [model.z[0], model.z[1], model.z[2]])
