# -*- coding: utf-8 -*-
# @Time    : 2024/10/20 19:16
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _test_pymoo.py
# @Software: PyCharm
import enum
import time

import matplotlib

matplotlib.use("tkagg")
import matplotlib.pyplot as plt

from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS
from pymoo.util import plotting
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.core.callback import CallbackCollection, Callback
from _logging import logger
from pymoo.core.algorithm import Algorithm
import os.path
import numpy
import pandas
import pymoo.util.display.display
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from simulation.task_manager.initialize import Initializer


from pymoo.optimize import minimize
plt.ion()

def f(x, y):
    # print("hello")
    sigmax, sigmay = 1, 2
    return numpy.exp(-(x ** 2 / (2 * sigmax ** 2) + (y - 0.5) ** 2 / (2 * sigmay ** 2)))







class ParameterName(enum.Enum):
    x = 0
    y = 1


class SharedMediumResult(enum.Enum):
    S11 = 0


class Parameters:
    def __init__(self, parameter_names: enum.EnumMeta):
        self.parameter_names = parameter_names
        # for parameter_name in (parameter_names):
        self.parameter_name_to_index = {parameter_name.name: i for i, parameter_name in enumerate(parameter_names)}
        self.index_to_parameter_name = list(self.parameter_name_to_index.keys())
        self._vec_index_to_parameter_name = numpy.vectorize(lambda index: self.index_to_parameter_name[index])

    def to_df(self, arr: numpy.ndarray):
        return pandas.DataFrame(data=arr, columns=list(self.parameter_name_to_index.keys()))


class Recorder:
    def __init__(self, parameters: Parameters, log_csv_path: str = None, ) -> None:
        super().__init__()
        self.parameters = parameters
        self.temp_result = {"F":None,}

        self.columns = ["timestamp", "algorithm.seed", "algorithm.n_gen", "algorithm.w", "algorithm.c1",
                        "algorithm.c2", ] + list(self.parameters.parameter_name_to_index.keys()) + list(self.temp_result.keys())

        logger.info("I am a new one")
        self.t_last_to_disk = time.time()
        self.i_last_to_disk = -1
        self.log_df = pandas.DataFrame(columns=self.columns)
        if not log_csv_path:
            prefix_of_log_csv = os.path.split(os.path.abspath(__file__))[0]
            log_csv_path = os.path.join(prefix_of_log_csv, "log.csv")
        self.log_csv_path = log_csv_path
        # assert not       os.path.exists(self.log_csv_path)

        self.to_disk()

    def check_csv_in_disk(self):
        if os.path.exists(self.log_csv_path):
            df = pandas.read_csv(self.log_csv_path)
            assert list(df.columns) == self.columns

    def update(self, algorithm: PSO):
        super().update(algorithm)
        new_df = self.parameters.to_df(algorithm.pop.get("X"))
        for key in self.temp_result:
            new_df[key] = self.temp_result[key]  # .pop.get("test2")

        d = {
            "algorithm.n_gen": algorithm.n_gen,
            "algorithm.w": algorithm.w,
            "algorithm.c1": algorithm.c1,
            "algorithm.c2": algorithm.c2,
            "algorithm.seed": algorithm.seed,
            "timestamp": time.time()
        }
        for key in d:
            new_df[key] = d[key]

        self.log_df = pandas.concat([self.log_df, new_df], ignore_index=True)

        # if( t - self.t_last_to_disk > 10.0 ):

        self.to_disk()

    def to_disk(self,  # log_df:pandas.DataFrame
                # mode = "a"
                ):
        logger.info('To csv file "%s"' % self.log_csv_path)
        if os.path.exists(self.log_csv_path):
            self.log_df.iloc[self.i_last_to_disk + 1:].to_csv(self.log_csv_path, index=False,
                                                              mode="a", header=False,
                                                              )
        else:
            self.log_df.to_csv(self.log_csv_path, index=False, )

        self.t_last_to_disk = time.time()
        self.i_last_to_disk = len(self.log_df) - 1
class RecorderBase:
    d = {}
class AlgorithmRecorder(RecorderBase):
    d = {
                "algorithm.n_gen":lambda algorithm: algorithm.n_gen,
                "algorithm.w":lambda algorithm: algorithm.w,
                "algorithm.c1":lambda algorithm: algorithm.c1,
                "algorithm.c2": lambda algorithm:algorithm.c2,
                "algorithm.seed": lambda algorithm:algorithm.seed,
            }
class IntermediateResultRecorder(RecorderBase):

    @staticmethod
    def get_f(params:dict):
        return f(params["x"],params['y'])

    d = {
        "result.f":get_f
        }

class MyProblem(Problem):
    def __init__(self,initializer:Initializer,log_csv_path :str = None ,IntermediateResultRecorder = IntermediateResultRecorder,AlgorithmRecorder = AlgorithmRecorder):
        self.initializer = initializer
        super().__init__(n_var=len(self.initializer.initial_df.columns), xl=self.initializer.lower_bound, xu=self.initializer.upper_bound, )
        logger.info("I am a new one")
        self.t_last_to_disk = time.time()
        self.i_last_to_disk = -1
        # self.IntermediateResultRecorder=IntermediateResultRecorder

        self.__key_timestamp = "timestamp"
        # self.AlgorithmRecorder = AlgorithmRecorder
        # self.columns = [self.__key_timestamp,]+                     list(   self.AlgorithmRecorder.d.keys())+ list(self.initializer.initial_df.columns) + list(self.IntermediateResultRecorder.d.keys())
        self.log_df = pandas.DataFrame(#columns=self.columns
                                       )
        if not log_csv_path:
            prefix_of_log_csv = os.path.split(os.path.abspath(__file__))[0]
            log_csv_path = os.path.join(prefix_of_log_csv, "log.csv")
        self.log_csv_path = log_csv_path
        # assert not       os.path.exists(self.log_csv_path)

        # self.to_disk()


    def to_disk(self,  # log_df:pandas.DataFrame
                # mode = "a"
                ):
        logger.info('To csv file "%s"' % self.log_csv_path)
        if os.path.exists(self.log_csv_path):
            self.log_df.iloc[self.i_last_to_disk + 1:].to_csv(self.log_csv_path, index=False,
                                                              mode="a", header=False,
                                                              )
        else:
            self.log_df.to_csv(self.log_csv_path, index=False, )

        self.t_last_to_disk = time.time()
        self.i_last_to_disk = len(self.log_df) - 1


    def _evaluate(self, x, out, *args, **kwargs):
        # callback:Recorder =kwargs['algorithm'].callback
        x =numpy.around( x// self.initializer.precision )* self.initializer.precision
        # temp_result = pandas.DataFrame(x, columns=self.initializer.initial_df.columns)
        temp_result = {}



        # 记录中间结果
        for each_line in x:
            params= {}
            for paramname in self.initializer.initial_df.columns:
                temp_result[paramname] =params[paramname] = each_line[self.initializer.param_name_to_index[paramname]]
            for k in self.IntermediateResultRecorder.d.keys():
                temp_result[k]= self.IntermediateResultRecorder.d[k](  params )





        out["F"] =-temp_result['result.f'].values
        self.log_df = pandas.concat([self.log_df, temp_result ],ignore_index=True)
        self.to_disk()




# from
# class MySampling():

sampling = LHS()
popsize = 20
problem = MyProblem(Initializer(r"initialize.csv"))
X= sampling(problem,popsize).get("X")
X[0] = (0,0.5)

pass


# plt.figure()
# plt.scatter(*X.T)

algorithm = PSO(popsize = popsize,sampling= X )



res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False,
               save_history=True,
               # display = MyDisplay()
               # output = MyOutput()
               # callback=Recorder(Parameters(ParameterName), ),
               )
# problem.to_disk()
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

