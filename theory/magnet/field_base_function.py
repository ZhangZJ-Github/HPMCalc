# -*- coding: utf-8 -*-
# @Time    : 2025/9/15 19:00
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : field_base_function.py
# @Software: PyCharm
from scipy.special import i1,k1,j1,y1


import matplotlib
import pandas

import common
import re

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import cst.results
import numpy
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import scipy.constants as C





class AxisymmetricNonSourceField2:
    def __init__(self,k,
                 D1,D2,
                 C1,C2
                 ):
        """
        基于 Eq. (3-13) in

        魏元璋. 《强流相对论环形电子束的周期磁场引导技术研究》. 硕士学位论文, 电子科技大学, 2018年. https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201802&filename=1018990986.nh.

        推导的另一组解

        """
        self.k =k
        self.D1 = D1
        self.D2 = D2
        self.C1 = C1
        self.C2 = C2

        self. eps = 2 * numpy.pi / self.k *1e-10
    def A_phi(self,r,z):

        return (self.D1 * j1(self.k * r ) + self.D2 * y1(self.k * r )  ) *(self.C1 * numpy.cosh(self.k *z) + self.C2 * numpy.sinh(self.k * z)  )


    def B_r(self,r,z):
        return -(self.A_phi(r,z+self.eps)-self.A_phi(r,z))/self.eps

    def B_z(self,r,z):
        r_ = r+self. eps
        return 1/r* (r_*self.A_phi(r_,z)-r*self.A_phi(r,z))/self.eps


if __name__ == '__main__':

    import matplotlib
    matplotlib.use('tkagg')
    r_beam_center = 40e-3
    dr_channel = 36e-3
    p = 50e-3

    rs,zs = numpy.linspace(r_beam_center - dr_channel/2,r_beam_center + dr_channel/2 * 4,30,),numpy.linspace(0,150e-3,31,)
    R,Z = numpy.meshgrid(rs,zs)
    k = 2 * numpy.pi / p
    ansf = AxisymmetricNonSourceField2(k,
                                      # k1(k* (r_beam_center-dr_channel/2)) / i1(k* (r_beam_center-dr_channel/2)),
                                      # i1(k*( r_beam_center+dr_channel/2))/k1(k*( r_beam_center+dr_channel/2)),
                                      # i1(k*r_beam_center)**-1,
                                      # k1(k * r_beam_center)**-1,
                                       1,1,
                                       1,-1

                                      )

    mm =1e-3

    plt.figure(figsize=(15,2),constrained_layout = True)
    cf = plt.contourf(Z/mm,R/mm, ansf.B_z(R,Z),cmap = plt.get_cmap('jet'),levels = 20)
    plt.colorbar(cf,label = "$B_z$")
    plt.gca().set_aspect('equal')
    plt.xlabel("z (mm)")
    plt.ylabel("r (mm)")

    plt.figure(figsize=(15,2),constrained_layout = True)
    cf = plt.contourf(Z/mm,R/mm, ansf.B_r(R,Z),cmap = plt.get_cmap('jet'),levels = 20)
    plt.colorbar(cf,label = "$B_r$")
    plt.gca().set_aspect('equal')
    plt.xlabel("z (mm)")
    plt.ylabel("r (mm)")



    plt.figure()
    plt.plot(zs / mm, ansf.B_z(r_beam_center,zs))
    if 0:
        plt.figure()
        _rs = numpy.linspace(0, rs[-1],1000)

        plt.plot(_rs,i1(k*_rs),label = "I1")
        plt.plot(_rs,k1(k*_rs),label = "K1")
        plt.axhspan(rs[0],rs[-1],alpha = 0.1)
        plt.legend()