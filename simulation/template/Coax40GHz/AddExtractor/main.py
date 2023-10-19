# -*- coding: utf-8 -*-
# @Time    : 2023年9月20日23:50:50
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : main.py
# @Software: PyCharm

import simulation.optimize.hpm as hpm
import simulation.optimize.initialize

from simulation.template.Coax40GHz.main import Coax40GHzSim

initial_csv = r"F:\changeworld\HPMCalc\simulation\template\Coax40GHz\AddExtractor\Initialize2.csv"
initializer = simulation.optimize.initialize.Initializer(initial_csv)


def get_coax40GHzsim():
    return Coax40GHzSim(initializer,
                        r"F:\changeworld\HPMCalc\simulation\template\Coax40GHz\AddExtractor\40GHzTemp.m2d",
                        r'E:\HPM\40GHz\optimize\AddExtractor2', 40e9, 3e9, lock=hpm.lock)


if __name__ == '__main__':
    # initializer.make_new_initial_csv(initial_csv,
    #                                  get_coax40GHzsim())
    # coax40GHz_sim = get_coax40GHzsim()
    # res = coax40GHz_sim.get_res(r"E:\HPM\40GHz\optimize\40_ghz-2.m2d")
    # logger.info(coax40GHz_sim.evaluate(res))
    manualjob = hpm.MaunualJOb(initializer, get_coax40GHzsim)
    manualjob.run()
