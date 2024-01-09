# -*- coding: utf-8 -*-
# @Time    : 2023/7/19 13:12
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : initial_test.py
# @Software: PyCharm
# 手动设置参数，开始模拟
import matplotlib

matplotlib.use('tkagg')
from simulation.task_manager.initialize import Initializer
from concurrent.futures import ThreadPoolExecutor

lock = Lock()
if __name__ == '__main__':
    def get_hpmsim(lock=lock):
        return HPMSim(r"E:\RBWO1\一段结构模板\40ghz_2.m2d",
                      r"E:\ref_test", 40e9, 3e9, lock=lock)


    initial_csv_path = r"E:\ref_test\921.csv"

    # Initializer.make_new_initial_csv(initial_csv_path, get_hpmsim())
    initializer = Initializer(initial_csv_path)

    with ThreadPoolExecutor(max_workers=6) as pool:
        run = lambda i: get_hpmsim().update(initializer.init_params[i])
        for i in range(initializer.N_initial):
            pool.submit(run, i)
