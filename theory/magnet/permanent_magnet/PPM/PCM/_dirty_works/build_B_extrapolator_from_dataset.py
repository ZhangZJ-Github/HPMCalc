# -*- coding: utf-8 -*-
# @Time    : 2025/9/12 10:41
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : build_B_extrapolator_from_dataset.py
# @Software: PyCharm
"""
基于数据集构造磁场插值器
"""

import cst.results
import matplotlib

matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import common
import numpy

key = "Dz"

path = r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Magnet_dataset_outward.cst"
run_id = 3
class         DatasetFromCSTProj:
    def __init__(self,path):
        self.proj =  cst.results.ProjectFile(path,allow_interactive=True)
        self. parameter_combination = proj.get_3d().get_parameter_combination(run_id)


    def Fourier_analysis(self,run_id:int,tree_path):
        Bz_data = numpy.array(
            self.proj.get_3d().get_result_item(tree_path, run_id).get_data())


        Dz = self.parameter_combination[key]
        L_ref = 13 * Dz
        ns = numpy.array([list(range(50))]).T
        an = 2 / L_ref * numpy.nansum(
            Bz_data[:, 1] * numpy.cos(2 * numpy.pi * ns / L_ref * Bz_data[:, 0]) * numpy.diff(Bz_data[:, 0],
                                                                                              append=numpy.nan),
            axis=-1)
        bn = 2 / L_ref * numpy.nansum(
            Bz_data[:, 1] * numpy.sin(2 * numpy.pi * ns / L_ref * Bz_data[:, 0]) * numpy.diff(Bz_data[:, 0],
                                                                                              append=numpy.nan),
            axis=-1)

        an[0] /= 2
        return ns, an, bn


if 0:
    proj = cst.results.ProjectFile(path,allow_interactive=True)
    parameter_combination = proj.get_3d().get_parameter_combination(run_id)
    Bz_data = numpy.array(proj.get_3d().get_result_item('Tables\\1D Results\\B-Field (Ms)_Z (Z)',run_id).get_data())
    Br_data = numpy.array(proj.get_3d().get_result_item('Tables\\1D Results\\B-Field (Ms)_Y (Z)',run_id).get_data())

    Dz =  parameter_combination[key]
    L_ref = 13 *Dz
    ns =numpy.array( [list(range(50))]).T
    an = 2/ L_ref * numpy.nansum(Bz_data[:,1] * numpy.cos(2 * numpy.pi *ns/ L_ref * Bz_data[:,0]) * numpy.diff(Bz_data[:,0],append=numpy.nan),axis= -1 )
    bn = 2/ L_ref * numpy.nansum(Bz_data[:,1] * numpy.sin(2 * numpy.pi *ns/ L_ref * Bz_data[:,0]) * numpy.diff(Bz_data[:,0],append=numpy.nan),axis= -1 )

    an[0]/=2

    get_Bz_series_approx  =lambda __zs_in_one_period: numpy.nansum(
         + an[numpy.newaxis].T * numpy.cos(2 * numpy.pi *ns/ L_ref * __zs_in_one_period)
         + bn[numpy.newaxis].T * numpy.sin(2 * numpy.pi *ns/ L_ref * __zs_in_one_period)
        ,axis=0)


    plt.figure()
    plt.plot(Bz_data[:,0],Bz_data[:,1])
    plt.plot(Bz_data[:,0],get_Bz_series_approx(Bz_data[:,0]))


    plt.figure()
    plt.plot(an)
    plt.plot(bn)
