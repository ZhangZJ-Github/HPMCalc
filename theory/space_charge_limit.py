# -*- coding: utf-8 -*-
# @Time    : 2023/6/6 16:00
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : space_charge_limit.py
# @Software: PyCharm
import common
import numpy
import scipy.constants as C


def space_charge_limit_current(Ek_eV, r1, r2, a):
    """
    Ref 2011_朱_低磁场准单模Cerenkov型高功率毫米波器件研究 Eq 3.3
    :return:
    """
    gamma = common.Ek_to_gamma(Ek_eV)
    return 4 * numpy.pi * C.epsilon_0 * C.m_e * C.c ** 3 / C.e * ((gamma ** (2 / 3) - 1) ** (3 / 2) / (
                1 - 2 * r1 ** 2 / (r2 ** 2 - r1 ** 2) * numpy.log(r2 / r1) + 2 * numpy.log(a / r2)))


print(space_charge_limit_current(500e3, 15e-3, 16e-3, 18e-3))
