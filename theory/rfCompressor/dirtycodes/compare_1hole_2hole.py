# -*- coding: utf-8 -*-
# @Time    : 2024/12/11 15:34
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : compare_1hole_2hole.py
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
import scipy.constants as C
from _logging import logger
plt.ion()
color_table = list(matplotlib.colors.TABLEAU_COLORS.keys())
f0_GHz = 9.3

lambda_0 = C.c / (f0_GHz*1e9)
NFI_max_theoretical_TE10 = lambda SwitchCavDz_to_lambda,b: 16    * eta_0/(b*lambda_0*(SwitchCavDz_to_lambda**2-1/4) **0.5 )
NFI_max_theoretical_TE10_ref = NFI_max_theoretical_TE10(1,0.5 * lambda_0,)
def get_interpolator (E_data):
    return interp1d(E_data[:,0],E_data[:,1])
cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\why_2hole\1hole.001.cst"
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_switch.TapperedSwitchCav.2.paramsweep.cst"
# cst_proj_path =         r"E:\CSTprojects\rfCompressor\cascadedHT\SES_allCST.cst"
proj: cst.results.ProjectFile = cst.results.ProjectFile(cst_proj_path,
                                                                    allow_interactive=True)


E_data = numpy.array(proj.get_3d().get_result_item('Tables\\1D Results\\Ex(Y=33)').get_data())
S11_data = numpy.array(proj.get_3d().get_result_item( '1D Results\\S-Parameters\\S1(1),1(1)').get_data())
E_interpolator= get_interpolator(E_data)
S11_interpolator= get_interpolator(S11_data)
phase_input_port_reaches_Emax = -numpy.angle(1 + S11_interpolator(f0_GHz))
# plt.figure()
plt.plot(E_data[:,0], numpy.abs(2*(E_data[:,1]*numpy.exp(1j*phase_input_port_reaches_Emax)).real) ** 2 / (0.5 * (1 - numpy.abs(S11_interpolator(f0_GHz)) ** 2))  / NFI_max_theoretical_TE10_ref,
         label =r"1-hole")

plt.axhline(NFI_max_theoretical_TE10(1.5, 15e-3) /NFI_max_theoretical_TE10_ref,c = color_table[3],ls = ':', label = "$\Delta z = 3\lambda_0/2$")
plt.legend()
plt.xlabel('z (mm)')
# plt.ylabel( r"NFI ($\Omega m^{-2}$)")
plt.ylabel( r"NFI ($NFI_{max}^{ref}$)")

SwitchCavDz_to_lambda = numpy.linspace(1.0001/2,3,1000)
SwitchCavDz_to_lambda2 = numpy.linspace(1.5,3,1000)
eta_0 = 377 # Ohm
b = 15e-3

plt.figure()
plt.plot(SwitchCavDz_to_lambda,
         (NFI_max_theoretical_TE10_ref /NFI_max_theoretical_TE10(numpy.piecewise(SwitchCavDz_to_lambda,[SwitchCavDz_to_lambda<1.5,],[lambda x:x, 1.5]),b)),
         label= "theoretical")
# plt.axhline(NFI_max_theoretical_TE10(1.5)**-1,#ls = ':',
#             c = color_table[1],label= "pure TE10 PG, the largest")

# plt.plot((1.5,3),numpy.ones(2) * ( NFI_max_theoretical_TE10_ESWG / NFI_max_theoretical_TE10(1.5)),#ls = ':',
#             c = color_table[1],label= "pure TE10 PW, the largest")
plt.plot(SwitchCavDz_to_lambda2,
         (NFI_max_theoretical_TE10_ref / NFI_max_theoretical_TE10(SwitchCavDz_to_lambda2, b)),
         label= "extrapolation by assuming a pure TE10 PW",ls = ':')


plt.xlabel('width of the switch cavity ($\lambda_0$)')
# plt.ylabel('$NFI_{max}^{-1}$ ($m^{2} / \Omega $)')
plt.ylabel('$NFI_{max}^{-1}$ [unit in $(NFI_{max}^{ref})^{-1}$]')
plt.legend()