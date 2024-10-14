# -*- coding: utf-8 -*-
# @Time    : 2024/5/11 13:13
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : transition.py
# @Software: PyCharm
import typing

import numpy
import scipy.constants as C
mu_B = C.physical_constants['Bohr magneton'][0]

def g_J_weak_mag_field(J, S, L):
    """
    按照L-S耦合
    :return:
    """
    return 1 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))


def g_F_approximately(F, J, I, g_J):
    """
    Lande g-factor
    :param F:
    :param J:
    :param I:
    :param g_J:
    :return:
    """
    return (F * (F + 1) + J * (J + 1) - I * (I + 1)) / (2 * F * (F + 1)) * g_J

class Rb87_D2:
    mass = 87 * C.m_p
    J = 1 / 2
    I = 3 / 2  # Ref: 印建平《原子光学》2012年第一版 Page 8
    F = I + 1 / 2
    S = 1 / 2
    L = 0
    gF = g_F_approximately(F, J, I, g_J_weak_mag_field(J, S, L))
    f_0 = 384.2304844685e12
    omega_0 = 2 * numpy.pi * f_0
    Gamma = 38.117e6  # Hz
    mu_F = gF * mu_B

    I_sat =4.6e-3/1e-2**2  #W/m^2, see ref: 2021-Maximized atom number for a grating magneto-optical trap via machine-learning assisted parameter optimization 贝叶斯优化-光栅磁光阱俘获原子数最大化-Optics Express.pdf
