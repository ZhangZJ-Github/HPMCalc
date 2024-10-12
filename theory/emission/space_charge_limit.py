# -*- coding: utf-8 -*-
# @Time    : 2023/6/6 16:00
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : space_charge_limit.py
# @Software: PyCharm
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
plt.ion()
import common
import numpy
import scipy.constants as C



def space_charge_limit_current(Ek_eV, r1, r2, a):
    """
    Ref 2011 朱俊 博士 低磁场准单模Cerenkov型高功率毫米波器件研究 Eq 3.3

    薄环形束在漂移管中的空间电荷极限电流

    其中 r1、r2为束的内外半径，a 为漂移管内半径。
    :return:
    """
    gamma = common.Ek_to_gamma(Ek_eV)
    return 4 * numpy.pi * C.epsilon_0 * C.m_e * C.c ** 3 / C.e * ((gamma ** (2 / 3) - 1) ** (3 / 2) / (
                1 - 2 * r1 ** 2 / (r2 ** 2 - r1 ** 2) * numpy.log(r2 / r1) + 2 * numpy.log(a / r2)))
def j_CL_nonrelativistic_1D(V,d):
    """
    Ref 2018 轴对称真空二极管空间电荷限制流 Eq (4)

    非相对论情况下一维空间电荷限制流密度

    原文的介绍：
    Y.Y.Child和J.W.Langmuir假定二极管为两无限大的平行极板,发射电子的初速度为零,电子从阴极表面均匀注入无限大平板电极空间,得到了非相对论情况下一维空间电荷限制流密度
    :param V: 阴阳极之间的电压差
    :param d: 阴阳极间距
    :return:
    """
    return 4 / 9 * C.epsilon_0 * (2*C.e/C.m_e) ** 0.5 * V ** (3/2) / d ** 2
class CoaxialAndHollow:
    """
    Ref 2018 Ka波段高功率同轴渡越时间振荡器的研究_宋莉莉 博士

    Eq (2.1) & Eq (2.2)
    """
    @staticmethod

    def I_space_charge_limit_coaxial(V,r1,r2,rb):
        """
        其中 r1、r2以及 rb 分别为内导体、外导体以及电子束的半径

        假设电子束径向厚度趋于0
        :param V:
        :param r1:
        :param r2:
        :param rb:
        :return:
        """
        m = C.m_e
        gamma = common.Ek_to_gamma(V,m )
        return 4*numpy.pi *C.epsilon_0* m*C.c**3/C.e*numpy.log(r2/r1) / (2*numpy.log(r2/rb)  * numpy.log(rb/r1) ) *(gamma**(2/3)-1)**(3/2 )



    @staticmethod
    def I_space_charge_limit_hollow(V,r2,rb):
        """
        其中r2以及 rb 分别为导体以及电子束的半径

        假设电子束径向厚度趋于0
        """
        m = C.m_e
        gamma = common.Ek_to_gamma(V,m )
        return 4*numpy.pi *C.epsilon_0 * m*C.c**3/C.e*(gamma**(2/3)-1)**(3/2 ) / (2 * numpy.log(r2/rb))

class HPMJamesBenford2009:
    """
    Ref: 《高功率微波（第二版）》 James Benford, John A. Swegle 等
    国防工业出版社 Eq. (4.115)-(4.116)

    假设束流径向厚度为0


    """
    @staticmethod
    def I_SCL_unit_in_kA_ring_shaped_beam_inside_hollow_waveguide(gamma_0 , r0, rb  ):
        """
        空心漂移管内，环形电子束的空间电荷限制流
        :param gamma_0: 电子束的洛伦兹因子
        :param r0: 圆波导（漂移管）半径
        :param rb: 束流发射的径向位置（假设径向厚度为0）
        :return:
        """
        return 8.5 / numpy.log(r0/rb)*(gamma_0**(2/3)-1)**(3/2)

    @staticmethod
    def I_SCL_unit_in_kA_cylindrial_beam_inside_hollow_waveguide(gamma_0 , r0, rb  ):
        """
        空心漂移管内，圆柱型实心电子束的空间电荷限制流
        :param gamma_0: 电子束的洛伦兹因子
        :param r0: 圆波导（漂移管）半径
        :param rb: 束流发射的径向位置（假设径向厚度为0）
        :return:
        """
        return 8.5 / (1+numpy.log(r0/rb))*(gamma_0**(2/3)-1)**(3/2)


if __name__ == '__main__':
    print(space_charge_limit_current(500e3, 15e-3, 16e-3, 18e-3))
    print(space_charge_limit_current(500e3, 11.4e-3, 13e-3, 15e-3))
    print(space_charge_limit_current(500e3, 11.4e-3, 11.4e-3 + 2.6e-3, 15e-3))
    print(CoaxialAndHollow.I_space_charge_limit_hollow(500e3,15e-3,11.4e-3,))
    print(CoaxialAndHollow.I_space_charge_limit_coaxial(500e3,8e-3,15e-3,11.4e-3,))
    print(HPMJamesBenford2009.I_SCL_unit_in_kA_ring_shaped_beam_inside_hollow_waveguide(common.Ek_to_gamma(500e3,),14e-3,11.4e-3))
    print(HPMJamesBenford2009.I_SCL_unit_in_kA_cylindrial_beam_inside_hollow_waveguide(common.Ek_to_gamma(500e3,),14e-3,11.4e-3))
    print(j_CL_nonrelativistic_1D(500e3, 15e-3) * 2 * numpy.pi * 12.2e-3 * 1.6e-3)

    plt.figure()
    delta_r = 1e-3#numpy.linspace(0.5e-3,5e-3,100)
    rin = 11.4e-3
    V= 10**numpy.linspace(1, 10, 100)#numpy.linspace(10, 500e6, 100)
    V=  numpy.linspace(10, 1e6, 100)
    plt.plot(     V /1e3, space_charge_limit_current(V,rin, rin+delta_r, 15e-3) / 1e3)
    plt.figure()
    plt.loglog(    V, space_charge_limit_current(V,rin, rin+delta_r, 15e-3) )
    plt.plot( V ,      V** (3/2),label = '3/2')
    plt.legend()

    plt.figure(figsize=(4,3),constrained_layout=True)
    rb = 11.4e-3
    Dr = 4e-3
    plt.plot(V /1e3 , CoaxialAndHollow.I_space_charge_limit_coaxial(V, rb-Dr, rb+Dr,rb)/1e3,label = "coaxial waveguide")
    plt.plot(V /1e3 , CoaxialAndHollow.I_space_charge_limit_hollow(V,  rb+Dr,rb)/1e3,label = "circular waveguide")
    plt.xlabel("$V$ / kV")
    plt.ylabel("$I_{SC}$ / kA")
    plt.legend()

    plt.figure()
    plt.plot(V /1e3 , CoaxialAndHollow.I_space_charge_limit_coaxial(V, rb-Dr, rb+Dr,rb)/ CoaxialAndHollow.I_space_charge_limit_hollow(V,  rb+Dr,rb) )



    plt.figure(figsize=(4,3),constrained_layout=True)
    rb = numpy.linspace(5e-3, 50e-3,100)
    V= 500e3
    Dr = 2e-3
    delta_r = 1.6e-3
    # plt.plot(rb /1e-3 , CoaxialAndHollow.I_space_charge_limit_coaxial(V, rb-Dr, rb+Dr,rb)/1e3,label = "coaxial waveguide")
    plt.plot(rb /1e-3 ,space_charge_limit_current(V,  rb-delta_r/2,rb+delta_r/2, rb + Dr )/1e3,label = "circular waveguide")
    plt.xlabel("$r_{b}$ / mm")
    plt.ylabel("$I_{SC}$ / kA")
    plt.legend()

    plt.figure(figsize=(4,3),constrained_layout=True)
    rb = 11.4e-3#numpy.linspace(5e-3, 50e-3,100)
    V= 500e3
    Dr = 2e-3
    delta_r =numpy.linspace(0.1e-3,5.6e-3,1010)
    # plt.plot(rb /1e-3 , CoaxialAndHollow.I_space_charge_limit_coaxial(V, rb-Dr, rb+Dr,rb)/1e3,label = "coaxial waveguide")
    plt.plot(delta_r /1e-3 ,space_charge_limit_current(V,  rb-delta_r/2,rb+delta_r/2, rb + Dr )/1e3,label = "circular waveguide")
    plt.xlabel("$\Delta r_{b}$ / mm")
    plt.ylabel("$I_{SC}$ / kA")
    plt.legend()


