# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 22:14
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : ebeam_transverse.py
# @Software: PyCharm

import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()
import common
import numpy
import scipy.constants as C
from scipy.integrate import solve_ivp

ln = numpy.log


class AnnularBeamInsideCoaxialDriftWeiYuanZhang:
    """
    考虑电子束厚度

    主要参考：
    魏元璋. 《强流相对论环形电子束的周期磁场引导技术研究》. 硕士学位论文, 电子科技大学, 2018. https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201802&filename=1018990986.nh.
    """

    def __init__(self, Rout, ro, ri, Rin):
        self.Rout, self.ro, self.ri, self.Rin = Rout, ro, ri, Rin
        self.X = 1 / ln(Rout / Rin) * (
                1 / 2 + (ro ** 2 * ln(Rout / ro) + ri ** 2 * ln(ri / Rin)) / (ro ** 2 - ri ** 2)
        )

    def E_sc_r(self, r,
               Ib, vz,
               ):
        """
        魏元璋 Eqn. (3-19)
        :param r:
        :param Ib:
        :param vz:
        :return:
        """
        return -Ib / (2 * numpy.pi * C.epsilon_0 * vz * r) * (
                self.X - numpy.piecewise(r, [(r < self.ri), (r >= self.ri) & (r < self.ro)], [
            self.ri, lambda r: r, self.ro
        ]) ** 2 / (self.ro ** 2 - self.ri ** 2))


class AnnularBeamInsideCoaxialDrift:
    @staticmethod
    def E_SC(r, I0, rout, rb, rin, vz,
             # Delta_r = 2e-3
             ):
        """
        主要参考：
        令钧溥. 《Ku波段低磁场同轴渡越时间振荡器的研究》. 博士学位论文, 国防科学技术大学, 2014. https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CDFD&dbname=CDFDLAST2017&filename=1016921902.nh.
        Eqn. (2.4), (2.6)

        :param r:
        :param I0:
        :param rout:
        :param rb:
        :param rin:
        :param vz:
        :param Delta_r:
        :return:
        """
        G = ln(rout / rb) / ln(rout / rin)
        return -I0 / (2 * numpy.pi * vz * C.epsilon_0 * r) * numpy.piecewise(r, [r < rb, ], [G, G - 1])


def dY(t, r, theta,
       D_r, r2_D_theta,
       # 以下参数用于空间电荷力计算
       I0, rout, rb, rin, vz,
       q, m_relativistic, beta,
       Bz):
    """
    D 表示对后面的变量求关于时间的导数
    :param t:
    :param r:
    :param D_r:
    :param r2_D_theta:
    :return:
    """
    E_SC = AnnularBeamInsideCoaxialDrift.E_SC(r, I0, rout, rb, rin, vz)
    D_theta = r2_D_theta / r ** 2
    DD_r = r * D_theta ** 2 + q * r * D_theta * Bz / m_relativistic + q * E_SC * (1 - beta ** 2) / m_relativistic
    D__r2_D_theta = - q * r * D_r * Bz / m_relativistic

    return [D_r, D_theta, DD_r, D__r2_D_theta]


def dY_considering_thickness(
        t, r, theta,
        D_r, r2_D_theta,
        # 以下参数用于空间电荷力计算
        ab: AnnularBeamInsideCoaxialDriftWeiYuanZhang,
        I0, vz,
        q, m_relativistic, beta,
        Bz):
    """
    D 表示对后面的变量求关于时间的导数
    :param t:
    :param r:
    :param D_r:
    :param r2_D_theta:
    :return:
    """
    E_SC = ab.E_sc_r(r, I0, vz)
    D_theta = r2_D_theta / r ** 2
    DD_r = r * D_theta ** 2 + q * r * D_theta * Bz / m_relativistic + q * E_SC * (1 - beta ** 2) / m_relativistic
    D__r2_D_theta = - q * r * D_r * Bz / m_relativistic

    return [D_r, D_theta, DD_r, D__r2_D_theta]


if __name__ == '__main__':
    L_drift = 250e-3
    Ek_beam = 50e3
    rb = 40e-3
    drbeam = 1.8e-3
    rbin, rbout = rb - drbeam / 2, rb + drbeam / 2  # 电子束内外半径
    I0 = 300
    q = C.e
    m = C.m_e
    dr_channel = 6e-3
    rin = rb - dr_channel /2
    rout =rb + dr_channel /2
    Bz = 0.1

    vz = common.Ek_to_beta(Ek_beam) * C.c
    gamma = common.Ek_to_gamma(Ek_beam)
    beta = vz / C.c
    ts = numpy.linspace(0, L_drift / vz, 2000)

    __rs_for_plot = numpy.linspace(rin, rout, 300)
    ab_wei = AnnularBeamInsideCoaxialDriftWeiYuanZhang(rout, rbout, rbin, rin)
    ab_ling = AnnularBeamInsideCoaxialDrift
    plt.figure(3)
    mm = 1e-3
    plt.plot(__rs_for_plot / mm, ab_wei.E_sc_r(__rs_for_plot, I0, vz) / 1e6, label="considering thickness")
    plt.plot(__rs_for_plot / mm, ab_ling.E_SC(__rs_for_plot, I0, rout, rb, rin, vz)/ 1e6, label="0-thickness approximation")
    plt.legend()
    plt.xlabel("r (mm)")
    plt.ylabel("$E_{sc,r}$ (MV/m)")

    from theory.ebeam_dynamic_in_HPM_device.confine_e_beam import WeiYuanzhang2018
    from _logging import logger

    logger.info(WeiYuanzhang2018.min_B_to_confine_annular_ebeam_inside_coaxial_drift(I0,  rbout, rbin, vz, q, ))
    logger.info(WeiYuanzhang2018.B_Brillouin_annular_ebeam_inside_coaxial_drift(I0, rout, rbout, rbin, rin, vz, q, ))

    sol = solve_ivp(lambda t, r__theta__D_r__r2_D_theta: dY_considering_thickness(
        t,
        *r__theta__D_r__r2_D_theta,
        ab_wei,
        I0, vz,
        q, m * gamma, beta,
        Bz
    ), [ts[0], ts[-1]],
                    numpy.array([#rb,
                        47e-3,
                                 0, 0, 0]), t_eval=ts,
                    # args=(I0, rout, rb, rin, vz,
                    #       C.e, C.m_e, beta,
                    #       Bz)
                    # method="Radau"
                    )

    r = sol.y[0, :]
    theta = sol.y[1, :]
    mm = 1e-3
    plt.figure(1)
    plt.axhline(48e-3 / mm, ls=":")
    plt.plot(vz * ts / mm, r / mm, label="B = %.2f T" % Bz)
    plt.legend()
    # plt.ylim(-1.5 * rb /mm, 1.5* rb /mm)
    # plt.gca().set_aspect("equal")
    plt.xlabel("z (mm)")

    plt.figure(2)
    plt.plot(r * numpy.cos(theta) / mm, r * numpy.sin(theta) / mm, label="B = %.2f T" % Bz)
    plt.legend()
    plt.xlim(-1.5 * rb / mm, 1.5 * rb / mm)
    plt.ylim(-1.5 * rb / mm, 1.5 * rb / mm)
    plt.gca().set_aspect("equal")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
