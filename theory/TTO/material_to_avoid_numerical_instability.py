# -*- coding: utf-8 -*-
# @Time    : 2025/3/19 16:34
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : vacuum_to_avoid_numerical_instability.py
# @Software: PyCharm



import matplotlib
import pandas

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()
import common
import numpy
import scipy.constants as C

ln = numpy.log

import cst.results
import cst.results
import matplotlib

matplotlib.use('tkagg')

from _logging import logger

import numpy
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

plt.ion()
class SpecialDispersiveMaterialToDampNonPhysicalPoorlyResolvedWavesAtHighFrequencies:
    @staticmethod
    def epsilon_r(f,epsilon_inf,f_R, f_0,delata, r):


        """
        Ref:
        https://space.mit.edu/RADIO/CST_online/mergedProjects/3D/special_overview/numerical_cerenkov_instability.htm
        查询时间：2025年3月19日16:40:25


        :param epsilon_inf:

        下面几个频率是可比的，即，要么全部用角频率，要么全部用频率
        :param f:
        :param f_R: Relaxation frequency
        :param f_0: resonance frequency
        :param delata: resonance width



        :param r: resonance weight
        :return:
        """
        return epsilon_inf + (1- epsilon_inf) * ((1-r ) * f_R /(f_R + 1j * f) + r * f_0**2 / (f_0**2 + 1j * f * delata - f**2))
    @staticmethod
    def relative_attenuation(epsilon_r:complex,omega,L):
        return numpy.exp(+epsilon_r.imag * omega*L / C.c)

if __name__ == '__main__':


    # CST help example (https://space.mit.edu/RADIO/CST_online/mergedProjects/3D/special_overview/numerical_cerenkov_instability.htm)
    # f = numpy.linspace(0, 500e9, 5000)
    # eps = SpecialDispersiveMaterialToDampNonPhysicalPoorlyResolvedWavesAtHighFrequencies.epsilon_r(
    #     f,0.98,
    #     1400e9,140e9,30e9,1)

    # For my structure
    f = numpy.linspace(0, 50e9, 5000)
    eps = SpecialDispersiveMaterialToDampNonPhysicalPoorlyResolvedWavesAtHighFrequencies.epsilon_r(
        f,0.98,
        700e9,32e9,2e9,1)
    GHz = 1e9
    plt.figure()

    # plt.figure()
    plt.plot(f /GHz,eps.real,)
    plt.plot(f / GHz,-eps.imag,)
    plt.plot(f / GHz,SpecialDispersiveMaterialToDampNonPhysicalPoorlyResolvedWavesAtHighFrequencies.relative_attenuation(eps,2* numpy.pi * f, C.c / f))
    plt.grid()

    df = pandas.DataFrame([f/GHz, eps.real,-eps.imag]).T
    df.to_csv("FakeVacuumToMitigateNCI.csv",index = False,header=False)

