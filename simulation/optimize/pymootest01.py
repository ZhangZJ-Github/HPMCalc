# -*- coding: utf-8 -*-
# @Time    : 2023/6/25 23:17
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : pymootest01.py
# @Software: PyCharm
import numpy
import pandas
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.lhs import LHS


def obj_func(x: numpy.ndarray):
    if x [5]> 0.5:return 11111111111111111111111111
    return (numpy.sin(x[0]) ** 2+ numpy.exp(x[1]*x[2]) )


n_var = 21
ccsv_name= "temp.csv"
pandas.DataFrame(columns=list(range
                              (n_var))+['score']).to_csv(ccsv_name, index=False)
from threading import Lock
lock = Lock()
constraint_ueq= [
    lambda x:    x[0]+x[1],
    lambda x:    x[1]-x[2],

]*10
class MyProblem(ElementwiseProblem):
    BIG_NUM = 1

    def __init__(self, *args, **kwargs):
        super(MyProblem, self, ).__init__(*args, n_var=n_var,
                                          n_obj=1,
                                          n_ieq_constr=len(constraint_ueq),
                                          xl=[-1] * n_var,
                                          xu=[1] * n_var, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out['G'] = [g(x) for g in constraint_ueq]
        if numpy.any(numpy.array(out['G']) > 0):
            # logger.warning("并非全部约束条件都满足：%s" % (out['G']))
            out['F'] =[111111111111]

            return

        out['F'] =[ obj_func(x)]
        lock.acquire()
        df = pandas.read_csv("temp.csv")
        pandas.concat([df, pandas.DataFrame(numpy.hstack((x,numpy.array(out['F']))).reshape((-1,  n_var+1)),  columns=  df.columns)],axis =0).to_csv(ccsv_name,
            index=False)
        lock.release()


algorithm = PSO(
    pop_size=20,
    sampling=LHS(),
    # ref_dirs=ref_dirs
)
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from multiprocessing.pool import ThreadPool

n_threads = 7
pool = ThreadPool(n_threads)

runner = StarmapParallelization(pool.starmap)

res = minimize(
    MyProblem(elementwise_runner=runner
              ),
    algorithm,
    seed=1,
    termination=('n_gen', 100),
    verbose=True,
    # callback =log_iter,#save_history=True
)

# plt.ion()
Scatter().add(res.F).show()
