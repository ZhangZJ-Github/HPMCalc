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
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

def gammabeta_to_gamma(gammabeta, ):
    return (1 + gammabeta ** 2) ** 0.5
def gammabeta_to_beta(gammabeta, ):
    return gammabeta / gammabeta_to_gamma(gammabeta)

def gamma_to_beta(gamma, ):
    return (1 - 1 / gamma ** 2) ** 0.5


def Ek_to_beta(Ek_eV, mass=C.m_e):
    return (1 - 1 / (1 + Ek_eV / (mass * C.c ** 2 / C.eV)) ** 2) ** .5


def p_to_v(p, mass_kg=C.m_e):
    """
    Ref:
    1.7 Equations of motion
    in
    S. B. van der Geer和M. J. de Loos. 《General Particle Tracer User Manual (Version 3.38)》. 2008年.

    :param p:
    :param mass_kg:
    :return:
    """
    return p * C.c / (p ** 2 + (mass_kg * C.c) ** 2) ** 0.5


def Ek_to_gamma(Ek_eV, mass=C.m_e):
    return (1 + Ek_eV / ((mass * C.c ** 2 / C.eV)))


def Ek_to_gamma_beta(Ek_eV, mass=C.m_e):
    return Ek_to_gamma(Ek_eV, mass) * Ek_to_beta(Ek_eV, mass)


def beta_to_gamma(beta):
    return 1 / (1 - beta ** 2) ** 0.5


def beta_to_betagamma(beta):
    return beta * beta_to_gamma(beta)


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


def Gaussian(z, z0, sigma):
    return 1 / ((2 * numpy.pi) ** 0.5 * sigma) * numpy.exp(-(z - z0) ** 2 / (2 * sigma ** 2))

class ShapeFunctions:
    @staticmethod
    def delta_shape(z,Dz):
        absz= numpy.abs(z)
        return numpy.piecewise(absz,  [absz<Dz],[lambda absz:
            (-1/(2*Dz**2 ) *absz  + 1/(2*Dz)),
                                       0.])
def skin_depth(f, sigma, mu=C.mu_0, ):
    """
    趋肤深度
    :param f:
    :param mu: 磁导率
    :param sigma: 电导率
    :return:
    """
    return (2 / (2 * numpy.pi * f * mu * sigma)) ** 0.5


def complex_from_amp_and_phase(amp, phase_in_degree):
    return amp * numpy.exp(1j * numpy.deg2rad(phase_in_degree))


if __name__ == '__main__':
    beta = 0.8
    from _logging import logger

    logger.info(p_to_v(beta_to_betagamma(0.8) * C.m_e * C.c, C.m_e) / C.c)
    plt.figure()
    z = numpy.linspace(-10, 10,1000)
    plt.plot(z,ShapeFunctions.delta_shape(z, 1))