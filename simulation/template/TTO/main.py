# -*- coding: utf-8 -*-
# @Time    : 2023/9/14 23:56
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : main.py
# @Software: PyCharm
from simulation.optimize.hpm import HPMSimWithInitializer, OptimizeJob, lock
import simulation.optimize.initialize

initial_csv = r"F:\changeworld\HPMCalc\simulation\template\TTO\Initialize.csv"
initializer = simulation.optimize.initialize.Initializer(initial_csv)


def get_HPMSimWithInitializer():
    return HPMSimWithInitializer(initializer, r"F:\changeworld\HPMCalc\simulation\template\TTO\TTO-template.m2d",
                                 r'E:\HPM\11.7GHz\optimize\TTO\from_good_5', 11.7e9, 3e9, lock=lock)


if __name__ == '__main__':
    job = OptimizeJob(initializer, get_HPMSimWithInitializer)
    job.run()
