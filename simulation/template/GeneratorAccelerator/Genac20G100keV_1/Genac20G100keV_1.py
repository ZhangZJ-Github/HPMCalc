# -*- coding: utf-8 -*-
# @Time    : 2023/11/3 16:37
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : Genac20G100keV_1.py
# @Software: PyCharm
import simulation.optimize.hpm
from simulation.template.GeneratorAccelerator.main import *

initialize_csv = r'initialize.csv'
get_initializer = lambda: Initializer(initialize_csv)  # 动态调用，每次生成新个体时都会重新读一遍优化配置，从而支持在运行时临时修改优化配置


def get_genac():
    return Genac(get_initializer(), r'Genac20G100keV_1.m2d',
                 r'E:\GeneratorAccelerator\Genac\optmz\Genac20G100keV_1\粗网格',
                 25e9,
                 lock=lock)


if __name__ == '__main__':
    # Initializer.make_new_initial_csv(initialize_csv,
    #     Genac(None, r'Genac25G-template.m2d', r'D:\MagicFiles\Genac\optmz', 25e9, ))
    # genac = get_genac()
    # res = genac.get_res(r"E:\GeneratorAccelerator\Genac\optmz\另行处理\CouplerGap3mm.grd")
    # genac.evaluate(res)
    # res
    # aaa
    optjob = simulation.optimize.hpm.OptimizeJob(get_initializer(), get_genac)
    optjob.algorithm = PSO(
        pop_size=24,
        sampling=simulation.optimize.hpm.SamplingWithGoodEnoughValues(optjob.initializer),  # LHS(),
        # ref_dirs=ref_dirs
    )
    optjob.run(n_threads=8)
