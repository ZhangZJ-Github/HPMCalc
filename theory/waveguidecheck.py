# -*- coding: utf-8 -*-
# @Time    : 2023/5/25 10:59
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : waveguidecheck.py
# @Software: PyCharm
"""
测试《微波与光电子学中的电磁理论（第二版）》P408的公式（7-100）：盘荷波导作为周期系统
"""
import typing

import matplotlib
import numpy as np

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy
import scipy.constants as C
from mpmath import besseli, besselj, bessely
from scipy.special import iv, jv, yv, jn_zeros

import common

matplotlib.use('tkagg')

nmax = 5  # 3
ns = numpy.arange(-nmax, nmax + 1, 1).astype(int)
beta = common.Ek_to_beta(250e3)
# ~12.5 GHz
p = 5.2e-3
d = 3.5e-3
b = 6.4e-3
a = 12e-3


# p = 11.3e-3
# d = 6.1e-3
# b = 7.4e-3
# a = 11.5e-3
# # ~4 GHz
# p=15e-3
# d=10e-3
# b=18e-3
# a=35e-3



5 * numpy.pi / (12.5e9 * 2 * 3.14 / (common.Ek_to_beta(500e3) * C.c) * 2)  # 预调制腔之间的距离


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
    :param k:
    :return:
    """
    kzns = numpy.array([kz0 + 2 * numpy.pi * n / p for n in ns]).astype(complex)
    # numpy.array([beta0 * numpy.zeros(freqs.shape) + 2 * numpy.pi * n / p for n in ns])
    # ks = omegas / C.c
    taun = (kzns ** 2 - k ** 2) ** .5
    # if numpy.any(
    #     numpy.isnan(iv(1, taun * b) / (taun * b * iv(0, taun * b)) * numpy.sinc(kzns * d / 2) ** 2)): raise RuntimeError
    return (d / p * numpy.nansum(
        iv(1, taun * b) / (taun * b * iv(0, taun * b)) * jv(0, kzns * d / 2) * numpy.sinc(kzns * d / 2),
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


# F = numpy.vectorize(F__)
F = numpy.vectorize(eigen_eq_disk_loaded_waveguide_as_periodic_system)

kz0s = 2 * numpy.pi * numpy.hstack(
    (numpy.linspace(0.1, 0.4, 6), numpy.linspace(0.41, 0.59, 13), numpy.linspace(0.6, 0.9, 6),)).astype(complex) / p
N_roots = 3


def check_ans(x, f):
    pass


from scipy.optimize import root


def find_root_in_range(f: typing.Callable, xrange):
    """
    在给定范围内查找f的零点
    :param f:
    :param xrange:
    :return: 经检测有bug
    """
    f_ = lambda x_and_fake_var: numpy.array((f(x_and_fake_var[0]), f(x_and_fake_var[0]) if (
            x_and_fake_var[0] > xrange[0] and x_and_fake_var[0] < xrange[1]) else (x_and_fake_var[0] - numpy.array(
        xrange)).min() ** 2)).astype(float)
    return root(f_, numpy.array((xrange[0], 1111)))


find_root_in_range(lambda x: numpy.sin(x), [1, 10])


def find_N_root_from(f: typing.Callable, xmin=0, Nroots=3):
    """
    从xmin开始，查找方程的根，直到找满Nroots个未知
    :param f:
    :param xmin:
    :param Nroots:
    :return:
    """
    roots = []
    while Nroots > 0:
        Nroots -= 1
    return roots


def _grid_search_filter(possible_roots, values_abs, dx):
    """
    从几组根中选出代表。网格搜索会产生一些接近（间距为网格搜索的步长dx）的根，在其中
    :param possible_roots:
    :param values_abs: >0, real is OK
    :param dx:
    :return:
    """
    roots = [possible_roots[0]]
    value_square_of_roots = [values_abs[0]]

    dx_ = 1.5 * dx
    for i in range(1, len(possible_roots)):
        if possible_roots[i] - possible_roots[i - 1] < dx_:
            if values_abs[i] <= value_square_of_roots[-1]:
                roots[-1] = possible_roots[i]
                value_square_of_roots[-1] = values_abs[i]
        else:
            roots.append(possible_roots[i])
            value_square_of_roots.append(values_abs[i])
    return roots


def grid_search(f: typing.Callable, x_range, dx=1e7, xtol=1.):
    """
    查找x_range范围内的所有x，使f(x)=0
    :param f: 函数，调用形式为f(x)
    :param dx:
    :param x_range: [xmin, xmax)
    :return:
    """
    xs = numpy.arange(*x_range, dx)
    values = numpy.abs((f(xs)))
    #     找出接近0的数
    indexes_of_zero = values < xtol ** 2

    return _grid_search_filter(xs[indexes_of_zero], values[indexes_of_zero], dx)


from scipy.signal import argrelextrema


def grid_search_minimal(f: typing.Callable, x_range, dx=1e7, xtol=1.):
    """
    查找x_range范围内的所有x，使f(x)=0
    :param f: 函数，调用形式为f(x)
    :param dx:
    :param x_range: [xmin, xmax)
    :return:
    """
    xs = numpy.arange(*x_range, dx)
    values = numpy.abs((f(xs)))
    #     找出接近0的数
    indexes_of_minimal = argrelextrema(values, np.less)

    return xs[indexes_of_minimal]


# grid_search(lambda x: numpy.sin(x), [1, 10], 1e-3, 1e-1)

# for i in range(len(kz0s)):
#     sol = scipy.optimize.root(lambda k: F(kz0s[i], k), 0.1)
#     if sol.status and numpy.all(sol.fun < 1e-3):
#         res.append([kz0s[i], sol.x[0]])

from _logging import logger
import shapely

res = []

for i in range(len(kz0s)):
    logger.info(i)
    sols = grid_search_minimal(lambda freq: F(kz0s[i], 2 * numpy.pi / C.c * freq), [1e-12, 50e9], 2e7, 10)
    res.append([kz0s[i], sols])

res_ = numpy.array(res)


def _get_jth_root(res_, j):
    return numpy.array([(res_[i][1][j] if len(res_[i][1]) >= j + 1 else numpy.nan) for i in range(len(res_))])


KZ0S, FS = numpy.meshgrid(kz0s, numpy.arange(1e-12, 50e9, 2e7))

VS = (F(KZ0S, 2 * numpy.pi / C.c * FS))
plt.ion()
plt.figure()
plt.contourf(KZ0S * p / (2 * numpy.pi), FS / 1e9, (VS), numpy.linspace(-5, 5, 10))
plt.colorbar()
line2Ds_driver = plt.plot(kz0s * p / (2 * numpy.pi), beta * C.c / (2 * numpy.pi) * kz0s / 1e9, label='driver')
driver_data = numpy.array(numpy.array(line2Ds_driver[0].get_data()).T.astype(float))

for j in range(3):
    line2Ds_curve = plt.plot(res_[:, 0] * p / (2 * numpy.pi), _get_jth_root(res, j) / 1e9, '.-',
                             label='dispersion curve, %d' % j)

    curve_data = numpy.array(line2Ds_curve[0].get_data()).T.astype(float)

    cross_pts = shapely.LineString(curve_data[~numpy.isnan(curve_data).any(axis=1)]).intersection(
        shapely.LineString(driver_data[~numpy.isnan(driver_data).any(axis=1)]))
    print(cross_pts)
    plt.scatter(*cross_pts.xy, label='intersection: (%.2f, %.2f)' % (cross_pts.x, cross_pts.y))
plt.xlabel(r'$k_{z0} p/(2 \pi)$')
plt.ylabel(r'f / GHz')
plt.legend()
