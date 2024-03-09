# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/6 19:48
@Auth ： Zi-Jing Zhang (张子靖)
@File ：GPTpp01.py
@IDE ：PyCharm
"""
import re

import numpy
import pygpt
import matplotlib.pyplot as plt
from _logging import logger

from simulation.post_processing.GPT_trajectory import GPTTraj


def Pout(Poutref, Eref, E):
    return Poutref * (E / Eref) ** 2


from simulation.optimize.hpm.hpm import HPMSim

if __name__ == '__main__':
    from simulation.task_manager.simulator import GeneralParticleTracerSim, InputFileTemplateBase

    template = InputFileTemplateBase(r"E:\changeworld\hpmcalc\fastdesign\microwave_source\.GPT\test_acc.template.in",
                                     r'E:\changeworld\hpmcalc\fastdesign\microwave_source\.GPT', )
    template.new_file_name = lambda: r"E:\changeworld\hpmcalc\fastdesign\microwave_source\.GPT\test_acc.in"
    gpt = GeneralParticleTracerSim()
    Efactors = numpy.linspace(0.1, 20, 10)
    Pparticle = []
    Pfield = []
    import grd_parser

    grd = grd_parser.GRD(
        r"E:\BigFiles\GENAC\GENACX50kV\手动\GenacX50kV_tmplt_20240210_051249_02\在线终止\GenacX50kV_tmplt_20240210_051249_02 周期内密集采样4.grd")  # 94e-3  # 3.65 / 2#20e-3/2#69.28e-3 / 2
    f = 9.4667e9
    Pref_MW = HPMSim._get_mean(grd.obs[r' FIELD_POWER S.DA @COUPLER.PORT,FFT-#5.1']['data'], 1 / f) / -1e6
    Eref = 1  # 5.9169e3#441#811.5
    for Efactor in Efactors:
        template.generate_and_to_disk({"%Efactor%": Efactor})
        gpt.run_bat(r"E:\changeworld\hpmcalc\fastdesign\microwave_source\.GPT\test_acc.bat")
        traj_gdf = pygpt.gdftomemory(r"E:\changeworld\hpmcalc\fastdesign\microwave_source\.GPT\traj.gdf")
        traj = GPTTraj(traj_gdf, 9.4667e9)

        screen_data = traj.interpolate_at_screen(80e-3)
        plt.ion()
        # plt.hist(((screen_data['G'] - 1) * 511), bins=numpy.linspace(0, 200.0, 100))
        Eavg = ((screen_data['G'] - 1) * 511).mean()  # keV

        E0 = 50  # keV
        I0 = 120

        Pfield.append(Pout(Pref_MW, Eref, Eref * Efactor))
        Pparticle.append(((E0 - Eavg) * 1e3 * I0) / 1e6)
        logger.info("\n输出的微波功率：%.2f MW\n粒子损失的功率：%.2f MW" % (Pfield[-1], Pparticle[-1]))
    plt.figure()
    plt.plot(Efactors, Pfield, label='Pfield')
    plt.plot(Efactors, Pparticle, label='Pparticle')
    plt.legend()
    plt.xlabel("Efactor")
    plt.ylabel('power / MW')
