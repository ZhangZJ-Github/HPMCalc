# -*- coding: utf-8 -*-
# @Time    : 2025/1/26 21:59
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : thermal.py
# @Software: PyCharm

import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import scipy.constants as C
import numpy
from _logging import logger
def J_thermal(T, Phi_M, ):
    """
    热发射电流密度
    检索用关键字：热阴极、电子枪
    Ref:
    刘军乐, 邵文生和于志强. 《基于热阴极的相对论返波管电子枪设计》. 太赫兹科学与电子信息学报 16, 期 1 (2018年): 131–34.
    Eq (1)

    :return:
    """
    A0 = 1.202e6
    return A0*T**2 *numpy.exp(-Phi_M / (C.k *T))
def J_thermal_modified_Richardson_equation(W,T,E =0.,
        Ag:float = None,):
    """
    考虑电场导致的热发射度下降

    Ref:
    Toonen, W.F., A. Rajabi, R.G.W. Van Den Berg, 等. 《Development of a Low-Emittance High-Current Continuous Electron Source》. Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment 1013 (2021年10月): 165678. https://doi.org/10.1016/j.nima.2021.165678.

    As the beam is generated through Schottky-assisted thermionic emission, the current density follows the modified Richardson equation ...



    :return:
    """
    if Ag is None:
        Ag = 1.20173e6 # A m-2 K-2
    DeltaW = (C.e **3 *E / (4 * numpy.pi *C.epsilon_0))** 0.5

    return Ag * T**2 * numpy.exp(-(W - DeltaW)/(C.k *T))

if __name__ == '__main__':
    cm = 1e-2
    logger.info(J_thermal(1300, 1.5*C.eV) *cm**2)

    T = numpy.linspace(10,2000, 1000)
    plt.figure()
    E_cathode = 20e6
    # LaB6
    plt.plot(T, J_thermal_modified_Richardson_equation(2.5 * C.eV ,T, E_cathode,29 )# A/cm2
             )
    plt.xlabel("T (K)")
    plt.ylabel("J ($A/cm^2$)")