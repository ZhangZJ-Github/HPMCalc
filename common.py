# -*- coding: utf-8 -*-
# @Time    : 2023/4/1 13:35
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : common.py
# @Software: PyCharm
# import matplotlib
# matplotlib.use('TkAgg')

import numpy
import scipy.constants as C
from scipy.stats import maxwell

def Ek_to_beta(Ek_eV, mass=C.m_e):
    return (1 - 1 / (1 + Ek_eV / (mass * C.c ** 2 / C.eV)) ** 2) ** .5


def Ek_to_gamma(Ek_eV, mass=C.m_e):
    return (1 + Ek_eV / ((mass * C.c ** 2 / C.eV)))


def Ek_to_gamma_beta(Ek_eV, mass=C.m_e):
    return Ek_to_gamma(Ek_eV, mass) * Ek_to_beta(Ek_eV, mass)

def beta_to_gamma(beta):
    return 1/(1-beta**2)**0.5
def beta_to_betagamma(beta):
    return  beta*beta_to_gamma(beta)
def Cherenkov_angles(Ek_eV, n, mass=C.m_e):
    """

    :param Ek_eV:
    :param n:
    :param mass:
    :return:
    """
    theta = numpy.arccos(1 / (n * Ek_to_beta(Ek_eV, mass)))
    return theta, numpy.pi / 2 - theta

def thermal_velocity_3D(T, m, nps):
    """

    :param T:
    :param m:
    :param nps:
    :return: shape (3, nps)
    """
    v_thermal = (maxwell.rvs(size=nps) * C.k * T / m) ** 0.5  # 热速度绝对值
    theta, phi = numpy.random.random((2, nps)) * 2 * numpy.pi
    return v_thermal * numpy.array((
        numpy.sin(theta) * numpy.cos(phi),
        numpy.sin(theta) * numpy.sin(phi),
        numpy.cos(phi)
    ))
