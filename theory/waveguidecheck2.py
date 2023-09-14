# -*- coding: utf-8 -*-
# @Time    : 2023/9/11 about 23:00
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : waveguidecheck.py
# @Software: PyCharm
"""
测试《微波与光电子学中的电磁理论（第二版）》P408的公式（7-100）：盘荷波导作为周期系统
"""

import matplotlib

matplotlib.use('tkagg')
import numpy
import scipy.constants as C
from mpmath import besseli, besselj, bessely
from scipy.special import iv, jv, yv

import common

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()

nmax = 3  # 3
ns = numpy.arange(-nmax, nmax + 1, 1).astype(int)
beta = common.Ek_to_beta(250e3)
# ~12.5 GHz
p = 5.2e-3
d = 3.5e-3
b = 6.4e-3
a = 12e-3

# My 11.7 GHz optimized (Maxwell centered)
p = 9.3e-3  # 周期长度
d = 3.6e-3  # 盘片距离
a = 21.5e-3  # 外径
b = 18e-3  # 内径


# 清华大学 王荟达 2021 博士论文 Fig. 3.6
# p = 32e-3
# d = 16e-3
# a = 57e-3
# b = 47e-3


# p = 11.3e-3
# d = 6.1e-3
# b = 7.4e-3
# a = 11.5e-3
# # ~4 GHz
# p=15e-3
# d=10e-3
# b=18e-3
# a=35e-3


# 5 * numpy.pi / (12.5e9 * 2 * 3.14 / (common.Ek_to_beta(500e3) * C.c) * 2)  # 预调制腔之间的距离


def F__(kz0, k):
    kzns = numpy.array([kz0 + 2 * numpy.pi * n / p for n in ns])
    # numpy.array([beta0 * numpy.zeros(freqs.shape) + 2 * numpy.pi * n / p for n in ns])
    # ks = omegas / C.c
    taun = (kzns ** 2 - k ** 2) ** .5
    return float(d / p * numpy.nansum(numpy.array([
        besseli(1, taun[i] * b) / (taun[i] * b * besseli(0, taun[i] * b)) * numpy.sinc(kzns[i] * d / 2) ** 2 for i in
        range(len(kzns))]),
        axis=0) - 1 / (k * b) * (
                         bessely(0, k * a) * besselj(1, k * b) - besselj(0, k * a) * bessely(1, k * b)) / (
                         bessely(0, k * a) * besselj(0, k * b) - besselj(0, k * a) * bessely(0, k * b)))


# F__(1e3,2e3)
def eigen_eq_disk_loaded_waveguide_as_periodic_system(kz0, k):
    """
    盘荷波导本征方程（作为周期系统），P406 Eq.7-100
    :param kz0:
    :param k: omega/c
    :return:
    """
    kzns = numpy.array([kz0 + 2 * numpy.pi * n / p for n in ns]).astype(complex)
    # numpy.array([beta0 * numpy.zeros(freqs.shape) + 2 * numpy.pi * n / p for n in ns])
    # ks = omegas / C.c
    taun = (kzns ** 2 - k ** 2) ** .5
    # if numpy.any(
    #     numpy.isnan(iv(1, taun * b) / (taun * b * iv(0, taun * b)) * numpy.sinc(kzns * d / 2) ** 2)): raise RuntimeError
    return (d / p * numpy.nansum(
        iv(1, taun * b) / (taun * b * iv(0, taun * b))
        * jv(0, kzns * d / 2) * numpy.sinc(kzns * d / 2),
        axis=0) - 1 / (k * b) * (
                    yv(0, k * a) * jv(1, k * b) - jv(0, k * a) * yv(1, k * b)) / (
                    yv(0, k * a) * jv(0, k * b) - jv(0, k * a) * yv(0, k * b))).real


def eigen_eq_disk_loaded_waveguide_as_uniform_system(kz, k):
    """
    盘荷波导本征方程（作为均匀系统），P406 Eq.7-23
    :param kz:
    :param k:
    :return:
    """
    tau = (kz ** 2 - k ** 2) ** .5
    return (d / p * numpy.nansum(
        iv(1, tau * b) / (tau * b * iv(0, tau * b)),
        axis=0) - 1 / (k * b) * (
                    yv(0, k * a) * jv(1, k * b) - jv(0, k * a) * yv(1, k * b)) / (
                    yv(0, k * a) * jv(0, k * b) - jv(0, k * a) * yv(0, k * b))).real


if __name__ == '__main__':

    # F = numpy.vectorize(F__)
    F = numpy.vectorize(eigen_eq_disk_loaded_waveguide_as_periodic_system)

    kz0s = 2 * numpy.pi * numpy.hstack(
        (numpy.linspace(0.1, 0.4, 6), numpy.linspace(0.41, 0.59, 13), numpy.linspace(0.6, 0.9, 6),)).astype(complex) / p
    mmax = 3  # TM01~TM03

    from scipy.optimize import root

    freqs = numpy.zeros((len(kz0s), mmax))
    freq_tol = 0.1e9

    for i in range(len(kz0s)):
        # for j in range(freqs.shape[1]):
        # logger.info(j)
        # TODO

        # res = root(lambda f: F(kz0s[i], 2 * numpy.pi * f / C.c), numpy.arange(.1e9, 100e9, 2e9), )
        res = root(lambda f: F(kz0s[i], 2 * numpy.pi * f / C.c), numpy.array([1]), )
        freqs[i][0] = res.x

        # logger.info(res.x)
        # freqs[i] = numpy.sort(numpy.unique((res.x % freq_tol) * freq_tol, ))[:mmax]
    for j in range(1  # freqs.shape[1]
                   ):
        plt.plot(kz0s * p / (2 * numpy.pi), freqs[:, j] / 1e9, label='$TM_{01}$')
    plt.plot(kz0s * p / (2 * numpy.pi), kz0s * beta * C.c / (2 * numpy.pi) / 1e9, label='electron')
    plt.xlabel('$k_z p / (2 \pi)$')
    plt.ylabel('f / GHz')
    plt.legend()
