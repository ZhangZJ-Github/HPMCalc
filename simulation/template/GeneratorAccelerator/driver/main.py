# 2023年11月2日19:22:59
# 张子靖
# -*- coding: utf-8 -*-
# @Time    : 2023/10/22 21:35
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : main.py
# @Software: PyCharm

import matplotlib

import simulation.optimize.hpm
from simulation.optimize.hpm import HPMSimWithInitializer
from simulation.optimize.initialize import Initializer

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

_ = plt

initialize_csv = r'initialize.csv'
initializer = Initializer(initialize_csv)


class Driver(HPMSimWithInitializer):
    def get_res(self, m2d_path: str, ) -> dict:
        return {}

    def evaluate(self, res: dict):
        return 1.


from threading import Lock

lock = Lock()


def get_hpm():
    return Driver(initializer, r'driver.m2d',
                  r'E:\GeneratorAccelerator\Genac\optmz\Genac20G100keV\粗网格\单独处理\driver.rundir', 25e9, lock=lock)


if __name__ == '__main__':
    # Initializer.make_new_initial_csv(initialize_csv,
    #     Genac(None, r'Genac25G-template.m2d', r'D:\MagicFiles\Genac\optmz', 25e9, ))
    # hpm = get_hpm()

    # res
    # aaa
    manjob = simulation.optimize.hpm.MaunualJOb(initializer, get_hpm, )

    # optjob.algorithm = PSO(
    #     pop_size=14,
    #     sampling=simulation.optimize.hpm.SamplingWithGoodEnoughValues(optjob.initializer),  # LHS(),
    #     # ref_dirs=ref_dirs
    # )
    manjob.run(maxworkers=12)
