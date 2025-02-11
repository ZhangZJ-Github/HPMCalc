# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 17:14
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _test_pymoo2.py
# @Software: PyCharm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.algorithms .soo .nonconvex.nelder import NelderMead
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt5")

algorithm = NSGA2(pop_size=100,
                  sampling=BinaryRandomSampling(),
                  crossover=TwoPointCrossover(),
                  mutation=BitflipMutation(),
                  eliminate_duplicates=True)
# algorithm =NelderMead
res = minimize(problem,
               algorithm,
               ('n_gen', 500),
               seed=1,
               verbose=True,
               save_history=  True)

Scatter().add(res.F).show()