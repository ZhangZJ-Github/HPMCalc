# -*- coding: utf-8 -*-
# @Time    : 2024/12/3 22:43
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : radiation_from_coupling_hole.py
# @Software: PyCharm
"""
从耦合孔向准自由空间的辐射
"""
import numpy
import scipy.constants as C
import enum
import typing

import cst.results
import matplotlib
import numpy
import pandas
import scipy
import skrf
import scipy.constants as C
from scipy.fft import ifft
from skrf.network import Network
from _logging import  logger
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

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
if __name__ == '__main__':
    plt.ion()
    z1 = 2e-3 # 耦合孔厚度
    f0 = 9.3e9
    a1  = 15e-3 # 耦合孔大小
    a2 = 1000e-3 # 超大波导的宽边
    mm= 1e-3
    lambda_0  = C.c /f0
    calculating_zone_a = a2#100e-3
    min_d = lambda_0 / 15
    xs,zs =numpy.array( [*numpy.arange(-calculating_zone_a/2,-a1/2,min_d),*numpy.linspace(-a1/2,a1/2,30),*numpy.arange(a1/2,calculating_zone_a/2,min_d),
                        ]   ),\
           numpy.array([*numpy.linspace(0,z1, 3), *numpy.arange(z1,z1+5*lambda_0,min_d),])
    X,Z = numpy.meshgrid(xs,zs)
    ey= numpy.zeros(X.shape).astype(complex)
    def lower_bound_zone2(x):
        return numpy.piecewise ( x, [numpy.abs(x)>=a1 / 2],[lambda  x:5*numpy.abs(x-numpy.sign(x)* a1/2)+z1,lambda  x:z1])
        # z2 = 16e-3
        # L = 30e-3
        # return numpy.piecewise ( x, [numpy.abs(x)>=a1 / 2],[lambda  x:(z2-z1)*(1-numpy.exp(-numpy.abs(x-numpy.sign(x)* a1/2) / L))+z1,lambda  x:z1])
    plt.figure()
    plt.plot(xs/mm, lower_bound_zone2(xs) / mm)
    plt.ylim(Z.min() / mm, Z.max() / mm)
    ey_1_zone1_at_lower_bound_zone2 = numpy.zeros(xs.shape)

    ey_1_zone1_at_lower_bound_zone2 [(numpy.abs(xs) < a1 / 2)] =get_e_m(
        xs[(numpy.abs(xs) < a1 / 2)],
        lower_bound_zone2(xs [(numpy.abs(xs) < a1 / 2)]),
        a1, 1, f0)


    plt.figure()
    plt.plot(xs, ey_1_zone1_at_lower_bound_zone2)

    # 需要考虑的模式数
    max_m = xs.shape[0]//3
    M = numpy.matrix([*[get_e_m(xs,lower_bound_zone2(xs),a2,m,f0,+1) for m in range(1, 1+max_m)],
                     -get_e_m(xs ,lower_bound_zone2(xs), a1,1,f0,-1) ]).T

    res = (numpy.linalg.pinv(M) * numpy.matrix([ey_1_zone1_at_lower_bound_zone2]).T).A.ravel()
    Ey_zone2_m, Ey_zone1_reflected = res[:-1],res[-1]






    plt.figure()
    plt.plot(numpy.arange(1,(len(Ey_zone2_m))+1,1), numpy.abs(Ey_zone2_m))
    plt.axhline(numpy.abs(Ey_zone1_reflected))
    plt.ylim(0,2)







    index_of_zone1 = (numpy.abs(X) < a1/ 2) & (Z < lower_bound_zone2(X))
    index_of_zone2 =  (Z >= lower_bound_zone2(X))



    ey [index_of_zone1]= get_e_m(X[index_of_zone1], Z[index_of_zone1], a1, 1, f0,+1) + Ey_zone1_reflected*get_e_m(X[index_of_zone1], Z[index_of_zone1], a1, 1, f0,-1)
    ey [index_of_zone2] =numpy.array([Ey_zone2_m_ * get_e_m(X[index_of_zone2], Z[index_of_zone2], a2, m_minus_1+1, f0, +1) for m_minus_1 ,Ey_zone2_m_ in enumerate(Ey_zone2_m)]).sum(axis= 0)
    emax =  numpy.max(numpy. abs(ey))



    plt.ioff()

    for phi in numpy.linspace(0,360,36):
        plt.figure()
        cf = plt.contourf(X / 1e-3, Z / 1e-3,
                          +ey * numpy.exp(-1j *numpy.deg2rad (phi))
                          ,
                          cmap=plt.get_cmap('jet'), levels = numpy.linspace(-emax, emax , 20), order = -1)
        plt.plot(xs / mm, lower_bound_zone2(xs) / mm)
        plt.ylim(Z.min() / mm,Z.max() / mm)
        plt.colorbar(cf)
        plt.gca().set_aspect('equal')
        plt.savefig('res/phi=%03.1f.png'%(phi))
        plt.close()
    plt.ion()
