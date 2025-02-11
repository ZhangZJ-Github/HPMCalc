# -*- coding: utf-8 -*-
# @Time    : 2024/12/9 20:39
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : processingEdata.py
# @Software: PyCharm
import pandas

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
from scipy.interpolate import interp1d, RegularGridInterpolator
from _logging import logger
import scipy.constants as C
plt.ion()


def get_e_m(x:numpy.ndarray, z, a, m, f,propagating_direction= +1):
    """

    :param x:
    :param z:
    :param a:
    :param m:
    :param f:
    :param propagating_direction: +1 表示沿+z方向传播的波
    :return:
    """
    kxm =m*numpy.pi / a
    k = 2*numpy .pi *f/ C.c
    kz =propagating_direction* (k**2 - kxm**2          )**0.5
    # logger.info("kxm = %.2e"%(kxm))
    # logger.info("kz = %.2e"%(kz))
    return kxm  * numpy.sin(kxm *(x + a/2 )) * numpy.exp(+1j *kz * z)
def get_interpolator(df_Edata:pandas.DataFrame):
    zs_cst, ys_cst = df_Edata['z [mm]'].unique(), df_Edata['y [mm]'].unique()
    df_Edata_3d = numpy.array(((df_Edata['ExRe [V/m]'] + 1j * df_Edata['ExIm [V/m]']).values,
                               (df_Edata['EyRe [V/m]'] + 1j * df_Edata['EyIm [V/m]']).values,
                               (df_Edata['EzRe [V/m]'] + 1j * df_Edata['EzIm [V/m]']).values,
                               )).transpose((1, 0)).reshape((ys_cst.shape[0], zs_cst.shape[0], 3), order='F')
    df_Edata_interpolator = RegularGridInterpolator((ys_cst, zs_cst), df_Edata_3d)
    return df_Edata_interpolator


df_Edata = pandas.read_csv(r"H:\Users\Zhang\Desktop\test_Edata_with_small_structure.txt", skiprows=[1,],sep = r'\s{2,}')
df_Edata_without_small_structure = pandas.read_csv(r"H:\Users\Zhang\Desktop\test_Edata_without_small_structure.txt", skiprows=[1,],sep = r'\s{2,}')

zs_cst ,ys_cst = df_Edata['z [mm]'].unique(),df_Edata['y [mm]'].unique()

df_Edata_interpolator =get_interpolator(df_Edata)
df_Edata_interpolator_without_small_structure =get_interpolator(df_Edata_without_small_structure)
phase_input_port_reaches_Emax  =-  numpy.angle(df_Edata_interpolator((0, 300))[0])
logger.info(numpy.rad2deg(phase_input_port_reaches_Emax))
Z_cst,Y_cst  = numpy.meshgrid(zs_cst,ys_cst)

plt.figure()
cf = plt.contourf(Z_cst,Y_cst,numpy.abs((df_Edata_interpolator((Y_cst,Z_cst))[...,0] * numpy.exp(1j*phase_input_port_reaches_Emax)).real) ,
                  cmap = plt.get_cmap('jet'),levels = numpy.linspace(0, numpy.abs(df_Edata_interpolator((Y_cst,Z_cst))[...,0]).max() , 50))
plt.gca().set_aspect('equal')
plt.colorbar(cf)



plt.figure()
cf = plt.contourf(Z_cst,Y_cst,
                  ((df_Edata_interpolator_without_small_structure((Y_cst,Z_cst))[...,0] * numpy.exp(1j*phase_input_port_reaches_Emax))) -
                  ((df_Edata_interpolator((Y_cst,Z_cst))[...,0] * numpy.exp(1j*phase_input_port_reaches_Emax))) ,
                  cmap = plt.get_cmap('jet'),levels =50)
plt.gca().set_aspect('equal')
plt.colorbar(cf)





cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\why_2hole\2hole_sym_modified_min_Y3.cst"
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_switch.TapperedSwitchCav.2.paramsweep.cst"
# cst_proj_path =         r"E:\CSTprojects\rfCompressor\cascadedHT\SES_allCST.cst"
proj: cst.results.ProjectFile = cst.results.ProjectFile(
    cst_proj_path,
    allow_interactive=True)

E_data = numpy.array(proj.get_3d().get_result_item( 'Tables\\1D Results\\e-field (f=9.3) (1(1))_X (Z)').get_data())

plt.figure()
plt.plot(E_data[:, 0].real,
         numpy.abs(
             (E_data[:,1] *numpy.exp(1j*phase_input_port_reaches_Emax)).real
                   )
         )

# plt.figure()
plt.plot(zs_cst,numpy.abs((df_Edata_interpolator((32.,zs_cst))[:,0] * numpy.exp(1j*phase_input_port_reaches_Emax) ).real))
