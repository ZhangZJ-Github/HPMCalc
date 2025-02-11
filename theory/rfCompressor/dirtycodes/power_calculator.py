# -*- coding: utf-8 -*-
# @Time    : 2024/12/5 22:47
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : power_calculator.py
# @Software: PyCharm

import enum
import typing

import cst.results
import matplotlib
import numpy
import pandas
import scipy
import skrf
from scipy.fft import ifft
from skrf.network import Network

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from _logging import logger
plt.ion()
cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\why_2hole\2hole_sym_modified_min_Y3.noterminal.cst"
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_switch.TapperedSwitchCav.2.paramsweep.cst"
# cst_proj_path =         r"E:\CSTprojects\rfCompressor\cascadedHT\SES_allCST.cst"
proj: cst.results.ProjectFile = cst.results.ProjectFile(cst_proj_path,
                                                                    allow_interactive=True)
S11_data = numpy.array(proj.get_3d().get_result_item( '1D Results\\S-Parameters\\S1(1),1(1)').get_data())

E_data = numpy.array(proj.get_3d().get_result_item('Tables\\1D Results\\Ex coupling hole').get_data())
H_data = numpy.array(proj.get_3d().get_result_item('Tables\\1D Results\\Hz coupling hole').get_data())
def get_interpolator (E_data):
    return interp1d(E_data[:,0],E_data[:,1])
E_interpolator= get_interpolator(E_data)
H_interpolator= get_interpolator(H_data)
z_ =E_data[:,0].real
b_mm= 15
mm = 1e-3

P_out =( numpy.abs (E_interpolator(z_))  * numpy.abs( H_interpolator(z_))   * numpy.array([0,*numpy.diff(z_)]) ).sum()* b_mm *mm**2 /2 # 平均功率，行波
# P_out =( numpy.abs(E_interpolator(z_)) /2  *numpy.abs( H_interpolator(z_)) /2  * numpy.array([0,*numpy.diff(z_)]) ).sum()* b_mm *mm**2 /2 # 平均功率，驻波

power_leakage_to_switch_cav = P_out / 0.5 # 相对于输入功率之比
logger.info(power_leakage_to_switch_cav)
plt.figure()
plt.plot(z_ , E_interpolator(z_) * H_interpolator(z_))


