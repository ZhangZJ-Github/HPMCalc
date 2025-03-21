# -*- coding: utf-8 -*-
# @Time    : 2025/3/16 17:18
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : beam_conductance.py
# @Software: PyCharm


import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()
import common
import numpy
import scipy.constants as C

ln = numpy.log

import cst.results
import cst.results
import matplotlib

matplotlib.use('tkagg')

from _logging import logger

import numpy
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

plt.ion()



def get_interpolator(E_data):
    return interp1d(E_data[:, 0].real, E_data[:, 1])


def calculate_M(z_data: numpy.ndarray, Ez_data: numpy.ndarray,
                beta_e, n=1, dz=None) -> complex:
    if dz is None: dz = numpy.diff(z_data, append=z_data[-1])
    M = numpy.sum(Ez_data * numpy.exp(1j * n * beta_e * z_data) * dz) / numpy.sum(numpy.abs(Ez_data) * dz)
    return M


vecf_calculate_M = numpy.vectorize(calculate_M, signature="(n),(n),(),(),()->()")


def calculate_Ge_G0(betaes, Ms, ):
    """
    计算归一化电子电导G_e/G_0
    :param betaes:
    :param Ms:
    :return:
    """
    return -1 / 4 * betaes[:-1] * numpy.diff(numpy.abs(Ms) ** 2) / numpy.diff(betaes)


mm = 1e-3


def plot_Ge_G0(Ez_interp, Ek_eV, f0, length_unit=mm):
    """

    绘制归一化电子电导G_e/G_0相关图像
    :return:
    """
    # Ek_eV = numpy.linspace(10e3, 200e3, 100)
    v_e = common.Ek_to_beta(Ek_eV) * C.c
    # f0 = 9.3e9
    omega0 = 2 * numpy.pi * f0
    beta_e = omega0 / v_e
    Ms = vecf_calculate_M(Ez_interp.x * length_unit, Ez_interp.y, beta_e, 1, numpy.diff(Ez_interp.x)[0])
    # plt.figure()
    Ge_G0 = calculate_Ge_G0(beta_e, Ms)
    plt.plot(Ek_eV[:-1] / 1e3, Ge_G0)
    plt.axvline(50)
    plt.grid()
    plt.xlabel("beam voltage / keV")
    plt.ylabel("$G_e / G_0$")
    return Ge_G0


if __name__ == '__main__':
    proj_3D: cst.results.ProjectFile = cst.results.ProjectFile(
        r"E:\CSTprojects\Genac2\CoaxialRFsource.buncher.Eigenmode.cst",
        # r"E:\CSTprojects\Genac2\CoaxialRFsource.buncher.PIC.cold.cst",

        allow_interactive=True)
    run_id = 0

    Ez_data = numpy.array(
        proj_3D.get_3d().get_result_item('Tables\\1D Results\\e_Z (Z)',
                                         # 'Tables\\1D Results\\e-field (f=9.3) (pic)_Z (Z)',
                                         run_id).get_data())
    # Ez_along_r_data = numpy.array(
    #             proj_3D.get_3d().get_result_item( 'Tables\\1D Results\\e_Z (Y)',
    #                                              run_id).get_data())
    # fake_Ez_data = Ez_data.copy()

    # __sigma =5
    # __z0s =[30.06,43.06,56.06]
    # __amps = [1,0,-1]
    __sigma = 4
    __p = 14
    __amps = [1, 0, -1]


    __z0s = [200 + __p * (i - 1) for i in range(len(__amps))]
    __zs = numpy.linspace(min(__z0s)-4 *__sigma,max(__z0s) + 4 *__sigma,2000)
    fake_Ez_data = numpy.zeros((len(__zs,),2))
    fake_Ez_data[:, 0] = __zs
    fake_Ez_data[:, 1] = numpy.sum(
        [common.Gaussian(__zs, __z0s[i], __sigma) * __amps[i] for i in range(len(__z0s))], axis=0)
    plt.figure()
    plt.plot(Ez_data[:, 0], numpy.abs(Ez_data[:, 1]) / numpy.abs(Ez_data[:, 1]).max(), label="CST")
    plt.plot(fake_Ez_data[:, 0], numpy.abs(fake_Ez_data[:, 1]) / numpy.abs(fake_Ez_data[:, 1]).max(), label="ideal")
    plt.legend()

    # Ez_interp = get_interpolator(fake_Ez_data)
    Ez_interp = get_interpolator(Ez_data)
    Ek_eV = numpy.linspace(10e3, 200e3, 100)
    f0 = 9.3e9
    plt.figure()
    Ge_G0_data = plot_Ge_G0(Ez_interp, Ek_eV, f0, mm)
    Ek_eV_ref = 50e3
    I_beam = 300
    # Eigenmode默认储能为1J
    energy_storeed = 1
    R_over_Q = numpy.sum(numpy.abs(Ez_interp.y) * numpy.diff(Ez_interp.x, append=Ez_interp.x[-1], ) * mm) ** 2 / (
                2 * 2 * numpy.pi * f0 * energy_storeed)
    logger.info("R/Q = %.2f"%R_over_Q)
    Ge_to_G0 = interp1d(Ek_eV[1:], Ge_G0_data)(Ek_eV_ref)
    logger.info("Ge/G0 = %.4e" % Ge_to_G0)
    Ge = (Ge_to_G0 * (I_beam / Ek_eV_ref))
    Qb = 1 / (Ge * R_over_Q)
    logger.info("Q_b = %.2f" % Qb)
