# -*- coding: utf-8 -*-
# @Time    : 2025/9/5 16:27
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : expand_near_a_line_for_axisymmetric_B_field_without_source.py
# @Software: PyCharm
# 轴对称（且不含Bphi分量)的磁场在无散无旋区的展开
import deprecated
import matplotlib

import common

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import scipy.integrate
import cst.results
import numpy
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import scipy.constants as C

mm =1e-3

def build_interpolator(Bz_data,length_unit = mm):
    return interp1d(Bz_data[:,0].real * length_unit,Bz_data[:,1],fill_value=0.0,bounds_error=False)
class NoDivNoCurlNoAngularComponentAxisSymmetricFieldExtrapolator:
    """
    无散度、无旋度、无角向分量的轴对称场外推插值器


    以下内容用于增加本文件的被检索概率

    轴对称、无角向分量的无散、无旋场（如，螺线管内部的磁场、电容极板间的电场）在某一直线附近（r = r_0）的级数展开


    """
    def __init__(self,Bz_data,Br_data,r_center_unit_in_m ,
                 length_unit_given_B_data = mm):
        """

        :param Bz_data: 沿着某条平行于z轴的直线的磁场数据，shape (N, 2), 两列分别表示 (z, Bz)
        :param Br_data: 沿着某条平行于z轴的直线的磁场数据，shape (N, 2), 两列分别表示 (z, Br)
        :param r_center_unit_in_m: 采样磁场数据的直线所在的径向位置
        :param length_unit_given_B_data: Bz_data和Br_data第一列（长度）采用的单位

        """
        self.r_center = r_center_unit_in_m

        self.Bz_interpolator = build_interpolator(Bz_data, length_unit_given_B_data)
        self.Br_interpolator = build_interpolator(Br_data, length_unit_given_B_data)

        pBr_pz_data = numpy.array(
            [
                (Br_data[1:, 0] + Br_data[:-1, 0]) / 2,
                numpy.diff(Br_data[:, 1]) / (numpy.diff(Br_data[:, 0]) * length_unit_given_B_data)
            ]
        ).T
        pBz_pz_data = numpy.array(
            [
                (Bz_data[1:, 0] + Bz_data[:-1, 0]) / 2,
                numpy.diff(Bz_data[:, 1]) / (numpy.diff(Bz_data[:, 0]) * length_unit_given_B_data)
            ]
        ).T

        p2Bz_pz2_data = numpy.array(
            [
                (pBz_pz_data[1:, 0] +pBz_pz_data[:-1, 0]) / 2,
                numpy.diff(pBz_pz_data[:, 1]) / (numpy.diff(pBz_pz_data[:, 0]) * length_unit_given_B_data)
            ]
        ).T
        p2Br_pz2_data =numpy.array(
            [
        (pBr_pz_data[1:, 0] + pBr_pz_data[:-1,0]) / 2,
        numpy.diff(pBr_pz_data[:, 1]) / (numpy.diff(pBr_pz_data[:,0]) * length_unit_given_B_data)
            ]
        ).T

        self.pBr_pz_interpolator = build_interpolator(pBr_pz_data)
        self.pBz_pz_interpolator = build_interpolator(pBz_pz_data)
        self.p2Bz_pz2_interpolator = build_interpolator(p2Bz_pz2_data)
        self.p2Br_pz2_interpolator = build_interpolator(p2Br_pz2_data)




    def Bz_expand(self,r, z):
        Dr = r -self. r_center
        return self.Bz_interpolator(z) + self. pBr_pz_interpolator(z) * Dr + 1 / 2. * (- self.p2Bz_pz2_interpolator(z) - 2 / r * self. pBr_pz_interpolator(z)) * Dr ** 2

    @deprecated.deprecated
    def Br_expand_old(self,r, z):
        Dr = r -self. r_center
        return (
                self.    Br_interpolator(z)
                + (-self. pBz_pz_interpolator(z) - self. Br_interpolator(z) / r) * Dr
                + 1 / 2. * (-self. p2Bz_pz2_interpolator(z) + 2 / r ** 2 * self.Br_interpolator(z) - 1 / r * self.pBz_pz_interpolator(z)) * Dr ** 2
                )
    def Br_expand(self,r, z):
        Dr = r -self. r_center
        return (
                self.Br_interpolator(z)
                +( - self.pBz_pz_interpolator(z) ) * Dr
                + 1/2. *( -self. p2Br_pz2_interpolator(z) - 1/r *self. pBz_pz_interpolator(z)) * Dr **2
                ) / (1 + Dr/ r - (Dr/r) **2 )

if __name__ == '__main__':
    proj = cst.results.ProjectFile(r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Magnet_PCM.cst",allow_interactive=True)
    Bz_data = numpy.array(proj.get_3d().get_result_item('Tables\\1D Results\\B-Field (Ms)_Z (Z)').get_data())
    Br_data = numpy.array(proj.get_3d().get_result_item('Tables\\1D Results\\B-Field (Ms)_Y (Z)').get_data())
    r_center = 40e-3

    field_extrapolator=NoDivNoCurlNoAngularComponentAxisSymmetricFieldExtrapolator(Bz_data, Br_data, r_center, mm, )
    dr_interested_region = 20e-3



    rs,zs = numpy.linspace(r_center - dr_interested_region/ 2 ,r_center + dr_interested_region/ 2 ,30),Bz_data[:,0] * mm
    R,Z = numpy.meshgrid(rs,zs)
    fig, axs = plt.subplots(2,1 ,sharex=True,sharey=True)
    plt.sca(axs[0])
    cf = plt.contourf(Z / mm,R / mm,  field_extrapolator.   Bz_expand(R,Z),cmap = plt.get_cmap('jet'),levels = 20)
    plt.colorbar(cf,label = "$B_z$")
    plt.xlabel("z (mm)")
    plt.ylabel("r (mm)")
    plt.gca().set_aspect('equal')


    plt.sca(axs[1])
    cf = plt.contourf(Z / mm,R / mm,    field_extrapolator.   Br_expand(R,Z),cmap = plt.get_cmap('jet'),levels = 20)
    plt.colorbar(cf,label = "$B_r$")
    plt.xlabel("z (mm)")
    plt.ylabel("r (mm)")
    plt.gca().set_aspect('equal')
    plt.xlim(0,120)

    plt.figure()
    plt.plot(Bz_data[:,0],Bz_data[:,1],label = "$B_z$")
    plt.plot(Bz_data[:,0],Br_data[:,1],label = "$B_r$")
    plt.legend()
    plt.xlabel("z (mm)")
    plt.ylabel("B-field (T)")