# -*- coding: utf-8 -*-
# @Time    : 2025/3/17 10:48
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : CoaxialRFSourceEgun.py
# @Software: PyCharm


import matplotlib
import pandas

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()
import common
import numpy
import scipy
from scipy.integrate import quad
import scipy.constants as C
from scipy.integrate import solve_ivp

ln = numpy.log

import cst.results
import cst.results
import matplotlib

matplotlib.use('tkagg')
import skrf
from skrf.media.rectangularWaveguide import RectangularWaveguide
import re
from _logging import logger

from scipy.optimize import minimize

import numpy
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

plt.ion()

key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
key_interpolated_periodic_avg_square = 'interpolated_periodic_avg_square'
f_target = 9.3e9
f_target_GHz = f_target / 1e9


def get_interpolator(E_data):
    return interp1d(E_data[:, 0].real, E_data[:, 1])

def calculate_M(z_data: numpy.ndarray, Ez_data: numpy.ndarray,
                beta_e,n = 1,dz = None) -> complex:
    if dz is None:dz = numpy.diff(z_data,append=z_data[-1])
    M = numpy.sum ( Ez_data* numpy.exp(1j *n* beta_e * z_data) * dz) / numpy.sum ( numpy.abs(Ez_data) * dz)
    return M

vecf_calculate_M = numpy.vectorize(calculate_M,signature="(n),(n),(),(),()->()")

def calculate_Ge_G0(betaes, Ms,):
    """
    计算归一化电子电导G_e/G_0
    :param betaes:
    :param Ms:
    :return:
    """
    return -1/4 * betaes[:-1] * numpy.diff(numpy.abs(Ms) **2 ) /numpy.diff(betaes)


mm = 1e-3
def plot_Ge_G0(Ez_interp,Ek_eV,f0,length_unit = mm):
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
    plt.plot(Ek_eV[:-1] / 1e3, calculate_Ge_G0(beta_e, Ms))
    plt.axvline(50)
    plt.grid()
    plt.xlabel("beam voltage / keV")
    plt.ylabel("$G_e / G_0$")
if __name__ == '__main__':
    columns =re.split(r'\s+',"posX             posY             posZ             momX             momY             momZ             mass     macro-charge             time       particleID         sourceID          Current    SEEGeneration",)
    from io import StringIO
    lines = []

    with open(r"E:\CSTprojects\Genac2\TRK2DmonitorData.txt",'r')as f:
        lines = f.readlines()
    i_headerline =0
    for i,line in enumerate(lines):
        if line.startswith( "%  Plane id 4 Normal (0 0 1) Point on plane (0 0 0.045)"):
            i_headerline = i
    TRK_2d_monitor_data = pandas.read_csv(StringIO(''.join(lines[i_headerline+1:])),sep= r'\s+',)
    TRK_2d_monitor_data .columns = columns

    tmin= TRK_2d_monitor_data["time"].min()
    tmax= TRK_2d_monitor_data["time"].max()

    total_charge = TRK_2d_monitor_data["macro-charge"].sum()
    I_avg= total_charge / (tmax- tmin)


