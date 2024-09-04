# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 20:31
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : common.py
# @Software: PyCharm
from scipy.special import jn_zeros

import common


def wavelength(f):
    return C.c / f


def oversize_ratio(r, f):
    return r * 2 / wavelength(f)


def frequency_of_resonator_TMnmp_cylindial(a, l, n=0, m=1, p=0):
    """
    圆柱型谐振腔中TM_{nmp}模式的频率
    a：圆柱谐振腔的半径
    l：圆柱谐振腔的轴向长度
    n: 角向周期数
    m：径向周期数
    p: 轴（zs）向周期数
    :return:
    """
    Tnm = jn_zeros(n, m)[-1] / a
    betap = p * numpy.pi / l
    return (Tnm ** 2 + betap ** 2) ** .5 * C.c / (2 * numpy.pi)


def radii_of_waveguide_for_TMnm(f, n=0, m=1):
    return jn_zeros(n, m)[-1] * C.c / (2 * numpy.pi * f)


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


class CherenkovDevice:

    @staticmethod
    def B_cyclotron_resonance_absorption(Ek_eV: float, p: float, kz: float) -> float:
        """
        切伦科夫设备的回旋共振磁场，设计时应避开此值
        Ref: Eq (5) of

        2023 Numerical Studies on an Efficient Coaxial Superradiant Relativistic Backward Wave Oscillator With Low Magnetic Field IEEE TRANSACTIONS ON PLASMA SCIENCE, VOL. 51, NO. 10, OCTOBER 2023

        :param Ek_eV:
        :param p:
        :return:
        """
        return C.m_e * common.Ek_to_beta(Ek_eV) * C.c / C.e * common.Ek_to_gamma(Ek_eV) * (numpy.pi / p - kz)

    @staticmethod
    def B2(Ek_eV: float, p: float) -> float:
        """
        切伦科夫设备的回旋共振磁场，设计时应避开此值
        Ref: Eq (5) of

        2023 Numerical Studies on an Efficient Coaxial Superradiant Relativistic Backward Wave Oscillator With Low Magnetic Field IEEE TRANSACTIONS ON PLASMA SCIENCE, VOL. 51, NO. 10, OCTOBER 2023

        :param Ek_eV:
        :param p:
        :return:
        """
        return CherenkovDevice.B_cyclotron_resonance_absorption(Ek_eV, p, 0)
