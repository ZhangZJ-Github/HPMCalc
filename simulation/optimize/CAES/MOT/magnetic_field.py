# -*- coding: utf-8 -*-
# @Time    : 2024/5/11 15:04
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : magnetic_field.py
# @Software: PyCharm
import typing

import matplotlib
import numpy
import scipy.constants as C
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ellipe, ellipk

matplotlib.use('tkagg')
import matplotlib.pyplot as plt


class CoilMagField:
    """
    中心位于z=z0，轴线沿着z轴放置的理想线圈
    Ref: 2011 Atom trapping in non-trivial geometries for micro-fabrication applications MVAngeleyn_Thesis_final Page 136
    """

    def __init__(self, r_ring, I, z0):
        self.r_ring = r_ring
        self.I = I
        self.z0 = z0
        self.vectorized_get_B = numpy.vectorize(self.get_B)

    def B_on_axial(self, z, ):
        """
        :param Z:
        :param r_ring:
        :param I:

        以下参数为环形线圈的中心位置

        :param Z0:
        :return:
        """

        return C.mu_0 * self.I * self.r_ring ** 2 / (2 * (self.r_ring ** 2 + (z - self.z0) ** 2) ** (3 / 2))

    def get_B(self, x, y, z) -> typing.Tuple:
        """

        :param r:
        :param z:
        :return: (Bx, By, Bz)
        """
        r = (x ** 2 + y ** 2) ** 0.5

        z__ = z - self.z0
        if r == 0: return (0, 0, self.B_on_axial(z__))
        alpha = r / self.r_ring
        beta = z__ / self.r_ring

        gamma = z__ / r

        Q = (1 + alpha) ** 2 + beta ** 2
        m = 4 * alpha / Q
        _xy = numpy.array([x, y])
        e_r = _xy / numpy.linalg.norm(_xy)
        return (*(self.I * C.mu_0 / (2 * numpy.pi * self.r_ring * Q ** .5) * numpy.array([
            *(gamma * ((1 + alpha ** 2 + beta ** 2) / (Q - 4 * alpha) * ellipe(m) - ellipk(m)) * e_r),
            (1 - alpha ** 2 - beta ** 2) / (Q - 4 * alpha) * ellipe(m) + ellipk(m)
        ])),)


if __name__ == '__main__':
    plt.ion()
    from _grid import X, Y, Z, x, y, z

    r_coil = 11e-2 / 2
    NI_coil = 1000
    cmf1 = CoilMagField(r_coil, NI_coil, r_coil)
    cmf2 = CoilMagField(r_coil, -NI_coil, -r_coil)
    Bx, By, Bz = cmf1.vectorized_get_B(X, Y, Z)
    B = numpy.array((Bx, By, Bz)).transpose([1, 2, 3, 0])
    B += numpy.array(cmf2.vectorized_get_B(X, Y, Z)).transpose([1, 2, 3, 0])

    B_interpolator = RegularGridInterpolator((x, y, z), B)
    _X_for_plot, _Z_for_plot = numpy.meshgrid(x, z, indexing='xy')
    _B = B_interpolator((_X_for_plot, 0, _Z_for_plot))

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(4, 6), constrained_layout=True)
    axs: typing.List[plt.Axes]
    axs[0].streamplot(_X_for_plot, _Z_for_plot, _B[:, :, 0], _B[:, :, 2])
    axs[0].xaxis.set_major_formatter(lambda x, pos: "%.1f" % (x / 1e-2))
    axs[0].yaxis.set_major_formatter(lambda x, pos: "%.1f" % (x / 1e-2))
    axs[0].set_ylabel('zs / cm')
    # axs[0].set_aspect('equal')
    axs[1].plot(x, 1e4 * B_interpolator((x, 0, 0))[:, 0], label='$B_x$ @ zs = 0')
    axs[1].set_ylabel('B / Gs')
    axs[1].legend()
    axs[1].set_xlabel('x / cm')
