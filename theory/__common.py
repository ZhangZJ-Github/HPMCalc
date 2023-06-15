# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 20:31
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : common.py
# @Software: PyCharm
import numpy
import scipy.constants as C
from scipy.special import jn_zeros


def wavelength(f):
    return C.c / f


def oversize_ratio(r, f):
    return r * 2 / wavelength(f)


def frequency_of_resonator_TMnmp_cylindial(a, l, n=0, m=1, p=0):
    """
    圆柱型谐振腔中TM_{nmp}模式的频率
    :return:
    """
    Tnm = jn_zeros(n, m)[-1] / a
    betap = p * numpy.pi / l
    return (Tnm ** 2 + betap ** 2) ** .5 * C.c / (2 * numpy.pi)


def radii_of_waveguide_for_TMnm(f,n=0,m=1):
    return jn_zeros(n,m)[-1] * C.c / (2 * numpy.pi * f)
