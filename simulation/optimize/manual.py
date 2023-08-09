# -*- coding: utf-8 -*-
# @Time    : 2023/7/19 13:12
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : initial_test.py
# @Software: PyCharm
# 手动设置参数，开始模拟
import matplotlib

matplotlib.use('tkagg')
from simulation.optimize.optimize_HPM import *
from simulation.optimize.initialize import Initializer
from concurrent.futures import ThreadPoolExecutor

if __name__ == '__main__':
    def get_hpmsim(lock=Lock()):
        return HPMSim(r"F:\changeworld\HPMCalc\simulation\template\TTO\TTO-template.m2d",
                      r'E:\HPM\11.7GHz\optimize\TTO.manual', 11.7e9, 1e9, lock=lock)


    initial_csv_path = r"F:\changeworld\HPMCalc\simulation\template\TTO\Initialize.manual.csv"

    # Initializer.make_new_initial_csv(initial_csv_path, get_hpmsim())
    initializer = Initializer(initial_csv_path)

    with ThreadPoolExecutor(max_workers=6) as pool:
        run = lambda i: get_hpmsim().update(initializer.init_params[i])
        for i in range(initializer.N_initial):
            pool.submit(run, i)
