# -*- coding: utf-8 -*-
# @Time    : 2024/10/12 16:23
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : rf_compressor.py
# @Software: PyCharm
from threading import Lock

import numpy
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LHS

from pymoo.optimize import minimize

import simulation
from simulation.task_manager.initialize import Initializer
from simulation.task_manager.simulator import InputFileTemplateBase, SimulationExecutor
from simulation.task_manager.task import CachedTask,LoggedTask
from  simulation.optimize.hpm.hpm import SamplingWithGoodEnoughValues

initialize_csv = r'initialize.csv'
get_initializer = lambda: Initializer(initialize_csv)  # 动态调用，每次生成新个体时都会重新读一遍优化配置，从而支持在运行时临时修改优化配置

class MyObjectives:
    def __init__(self,cst_proj_path:str,run_id :int):
        self.cst_proj_path = cst_proj_path
        self.TMPG = self.get_TMPG(self.cst_proj_path,run_id)
        self.EmaxGDT = self.get_EmaxGDT(self.cst_proj_path,run_id)
        self.S23_abs = self.get_S23_abs(self.cst_proj_path,run_id)
        self.F = self.get_F()

    def to_dict(self):
        return {
            "TMPG": self.TMPG,
            "EmaxGDT":self.EmaxGDT,
            "S23_abs":self.S23_abs
        }
    @staticmethod
    def get_TMPG(    cst_proj_path:str   ,run_id :int):

        return 1.0
    @staticmethod
    def get_EmaxGDT(    cst_proj_path:str,  run_id :int  ):

        return 1.0
    @staticmethod
    def get_S23_abs(   cst_proj_path:str,run_id :int      ):
        return 1.0
    def get_F(self):

        return 1.0
from _logging import logger
class RFCompressorOptimiazationProblem(LoggedTask,#Problem
                                       ):
    class AddressTranslator:
        @staticmethod
        def to_str(cst_proj_path, run_id ):
            return "%s\n%d"%(cst_proj_path, (run_id))
        @staticmethod
        def parse(cst_proj_path_and_run_id:str):
            cst_proj_path, run_id = cst_proj_path_and_run_id.split("\n")
            return cst_proj_path,int(run_id)
    def __init__(self, lock: Lock = Lock(),
                 initializer: Initializer = None,
                 log_file_name='OptimizingRfCompressor.log.csv'):
        LoggedTask.__init__(self, lock, initializer, log_file_name)

    def run(self, param_set: dict) -> str:
        logger.info("Dummy run")
        return ""#super().run(param_set)





    def evaluate(self, medium_res: dict) -> float:
        return medium_res["F"]

    def get_res(self, cst_proj_path_and_run_id: str) -> dict:
        return MyObjectives(*self.AddressTranslator.parse(cst_proj_path_and_run_id)).to_dict()





def update_cst(params  :dict)->MyObjectives:
    """
    :param params: 形如 {"GDT_z": 285.0, "GDT_y": 30.0,}
    :return: 执行成功则返回True
    """
    return  MyObjectives("",0)

def dump_history():
    """

    :return:
    """





lock = Lock()


def get_task():
    return RFCompressorOptimiazationProblem()
if __name__ == '__main__':
    problem = RFCompressorOptimiazationProblem()
    optjob = simulation.optimize.hpm.hpm.OptimizeJob(get_initializer(), get_task)

    algorithm  = PSO(
        pop_size=14,
        sampling= SamplingWithGoodEnoughValues(optjob.initializer)  if optjob.initializer.N_initial else LHS()  # LHS(),
        # ref_dirs=ref_dirs
    )
