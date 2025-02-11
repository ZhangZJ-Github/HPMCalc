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

ln =numpy.log

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


def j_CL_nonrelativistic_1D(V, d):
    """
    Ref 2018 轴对称真空二极管空间电荷限制流 Eq (4)

    非相对论情况下一维空间电荷限制流密度

    原文的介绍：
    Y.Y.Child和J.W.Langmuir假定二极管为两无限大的平行极板,发射电子的初速度为零,电子从阴极表面均匀注入无限大平板电极空间,得到了非相对论情况下一维空间电荷限制流密度

    该公式也即

    Benford, James, John Allan Swegle, Edl Schamiloglu和John Swegle. High Power Microwaves. 2. ed. Series in Plasma Physics. New York: Taylor & Francis, 2007.

    的Eqn. (4.104)

    :param V: 阴阳极之间的电压差
    :param d: 阴阳极间距
    :return:
    """
    return 4 / 9 * C.epsilon_0 * (2 * C.e / C.m_e) ** 0.5 * V ** (3 / 2) / d ** 2


def j_CL_relativistic_1D(V0_MV, d_cm):
    """
    Ref: Benford, James, John Allan Swegle, Edl Schamiloglu和John Swegle. High Power Microwaves. 2. ed. Series in Plasma Physics. New York: Taylor & Francis, 2007.

    Eqn. (4.107)

    :return: unit in A/m**2
    """
    kA_cm_2 = 1e7
    return kA_cm_2 * 2.71 / (d_cm ** 2) * ((1 + V0_MV / 0.511) ** 0.5 - 0.847)


class CoaxialAndHollow:
    """
    Ref 2018 Ka波段高功率同轴渡越时间振荡器的研究_宋莉莉 博士

    Eq (2.1) & Eq (2.2)

    或

    令钧溥. 《Ku波段低磁场同轴渡越时间振荡器的研究》. 博士学位论文, 国防科学技术大学, 2017. https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CDFD&dbname=CDFDLAST2017&filename=1016921902.nh.

    Eqn. (2.13) & (2.15)
    """

    @staticmethod
    def get_IA(m=C.m_e):
        return 4 * numpy.pi * C.epsilon_0 * m * C.c ** 3 / C.e

    @staticmethod
    def I_space_charge_limit_coaxial(V, r1, r2, rb):
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
        gamma = common.Ek_to_gamma(V, m)
        return CoaxialAndHollow.get_IA(m) * numpy.log(r2 / r1) / (2 * numpy.log(r2 / rb) * numpy.log(rb / r1)) * (
                gamma ** (2 / 3) - 1) ** (3 / 2)

    @staticmethod
    def I_space_charge_limit_hollow(V, r2, rb):
        """
        其中r2以及 rb 分别为导体以及电子束的半径

        假设电子束径向厚度趋于0


        此公式实际上与Benford, James, John Allan Swegle, Edl Schamiloglu和John Swegle. High Power Microwaves. 2. ed. Series in Plasma Physics. New York: Taylor & Francis, 2007.

        的Eqn. (4.115)一致

        """
        m = C.m_e
        gamma = common.Ek_to_gamma(V, m)
        return CoaxialAndHollow.get_IA(m) * (gamma ** (2 / 3) - 1) ** (3 / 2) / (
                2 * numpy.log(r2 / rb))


class HPMJamesBenford2009:
    """
    Ref: 《高功率微波（第二版）》 James Benford, John A. Swegle 等
    国防工业出版社 Eq. (4.115)-(4.116)

    假设束流径向厚度为0


    """
    _8_5_kA = 2 * numpy.pi * C.epsilon_0 * C.m_e * C.c ** 3 / C.e

    @staticmethod
    def I_SCL_unit_in_A_ring_shaped_beam_inside_hollow_waveguide(gamma_0, r0, rb):
        """
        空心漂移管内，环形电子束的空间电荷限制流
        :param gamma_0: 电子束的洛伦兹因子
        :param r0: 圆波导（漂移管）半径
        :param rb: 束流发射的径向位置（假设径向厚度为0）
        :return:
        """
        return HPMJamesBenford2009._8_5_kA / numpy.log(r0 / rb) * (gamma_0 ** (2 / 3) - 1) ** (3 / 2)

    @staticmethod
    def I_SCL_unit_in_A_cylindrial_beam_inside_hollow_waveguide(gamma_0, r0, rb):
        """
        空心漂移管内，圆柱型实心电子束的空间电荷限制流
        :param gamma_0: 电子束的洛伦兹因子
        :param r0: 圆波导（漂移管）半径
        :param rb: 束流发射的径向位置（假设径向厚度为0）
        :return:
        """
        return HPMJamesBenford2009._8_5_kA / (1 + numpy.log(r0 / rb)) * (gamma_0 ** (2 / 3) - 1) ** (3 / 2)

    @staticmethod
    def estimate_Z_diode(
            d_cm, A_cm2, V0_MV
    ):
        """
        粗略估计二极管的阻抗

        Ref:

        Benford, James, John Allan Swegle, Edl Schamiloglu和John Swegle. High Power Microwaves. 2. ed. Series in Plasma Physics. New York: Taylor & Francis, 2007.

        Eqn. (4.104)后面的那个公式

        "This is the Child–Langmuir,11 or space-charge-limited, current density in a planar electron diode. Now if we can ignore edge effects, so that the current density is uniform across the surface of a cathode with area A, the current I from such a Child–Langmuir diode is JSCLA and the impedance Z for such a diode would be the ratio of the current to the voltage, or ..."

        参数含义见参考文献原文

        :return: 阻抗， unit in Ohm
        """
        return 429 * d_cm ** 2 / A_cm2 / V0_MV ** 0.5


class NUDT_doctor_ZhangWei_2019:
    """
    Ref: 2019 NUDT 博士 X波段高功率高效率相对论三轴速调管放大器研究_张威
    Eq 2.11
    """
    get_IA = CoaxialAndHollow.get_IA

    @staticmethod
    def get_D(rbout, rbin, rout, rin, ):
        rb0 = (rbout + rbin) / 2
        return (rb0 ** 2 - rbin ** 2) + \
               2 * rbin ** 2 * numpy.log(rbin / rb0) + \
               ((rbout ** 2 - rbin ** 2) * (2 * numpy.log(rout / rbout) + 1) + 2 * rbin ** 2 * numpy.log(
                   rbin / rbout)) / (numpy.log(rout / rin)) * numpy.log(rin / rb0)

    @staticmethod
    def I_SCL_coaxial(rbout, rbin,
                      rout, rin,
                      gamma):
        return - NUDT_doctor_ZhangWei_2019.get_IA() * (rbout ** 2 - rbin ** 2) / NUDT_doctor_ZhangWei_2019.get_D(rbout,
                                                                                                                 rbin,
                                                                                                                 rout,
                                                                                                                 rin, ) * (
                       gamma ** (2 / 3) - 1) ** (3 / 2)

    @staticmethod
    def I_SCL_coaxial2(rb0, delta_rbin,
                       rout, rin,
                       gamma):
        rbout = rb0 + delta_rbin / 2
        rbin = rb0 - delta_rbin / 2
        return - NUDT_doctor_ZhangWei_2019.get_IA() * (rbout ** 2 - rbin ** 2) / NUDT_doctor_ZhangWei_2019.get_D(rbout,
                                                                                                                 rbin,
                                                                                                                 rout,
                                                                                                                 rin, ) * (
                       gamma ** (2 / 3) - 1) ** (3 / 2)


class LiuJing2012Doctor:
    """
    Ref: 刘静. 《同轴波导虚阴极振荡器的研究》. 博士学位论文, 国防科学技术大学, 2012. https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CDFD&dbname=CDFD1214&filename=1012020375.nh.

    """
    IA = CoaxialAndHollow.get_IA(C.m_e)

    @staticmethod
    def gamma___(gamma_0):
        return (gamma_0 ** (2 / 3) - 1) ** (3 / 2)

    @staticmethod
    def I_SCL_solid_beam(gamma_0, Rout, rbeam_out):
        """
        Eqn. (1.3)
        实心电子束在空心漂移管中的空间电荷限制流

        :param Rout: 导体的内半径

        :param rbeam_out: 实心电子束的外半径

        :return:
        """
        return LiuJing2012Doctor.IA * LiuJing2012Doctor.gamma___(gamma_0) / (1 + 2 * numpy.log(Rout / rbeam_out))

    @staticmethod
    def I_SCL_annular_beam_inside_hollow_drift(gamma_0, Rout, rbeam_out, rbeam_in, ):
        """
        Eqn. (1.4)
        环形电子束在空心漂移管中的空间电荷限制流


        :param Rout: 导体的内半径

        :param rbeam_out: 环形电子束的外半径
        :param rbeam_in: 环形电子束的内半径


        :return:
        """
        return LiuJing2012Doctor.IA * LiuJing2012Doctor.gamma___(gamma_0) / (
                1 + 2 * numpy.log(Rout / rbeam_out)
                - 2 * rbeam_in ** 2 * numpy.log(rbeam_out / rbeam_in) / (rbeam_out ** 2 - rbeam_in ** 2))

    @staticmethod
    def I_SCL_annular_beam_inside_coaxial_drift(gamma_0, Rout, rbeam_out, rbeam_in, Rin):
        """

        Eqn. (1.5)
        环形电子束在同轴漂移管中的空间电荷限制流

        此公式准确性存疑，原文中存在变量名前后不一致的情况

        :param Rout: 外导体的内半径
        :param Rin: 内导体的外半径

        :param rbeam_out: 环形电子束的外半径
        :param rbeam_in: 环形电子束的内半径


        :return:
        """
        G = 1+ 2 * ln(Rout / rbeam_out)+ 2 * rbeam_in **2/ (rbeam_out **2 - rbeam_in**2) * ln((rbeam_in / rbeam_out) )


        rmax = rbeam_in * (1+ G * (rbeam_out  **2-rbeam_in **2) /(2 * rbeam_in **2  * ln(Rout / Rin)))**0.5
        return LiuJing2012Doctor.IA * LiuJing2012Doctor.gamma___(gamma_0) * ln(Rout / Rin) / (ln(rmax / Rin) -1/2) * (
            G + 2* rbeam_in **2 /(rbeam_out **2- rbeam_in **2) *
            (ln(Rout / Rin)*ln(rmax/rbeam_in)) / (rbeam_out **2- rbeam_in **2 * ln(rmax / Rin) -1/2)
        )**-1

class ZhangWei2022Doctor(LiuJing2012Doctor):

    @staticmethod
    def I_SCL_annular_beam_inside_coaxial_drift(gamma_in, Rout, rbeam_out, rbeam_in, Rin):
        """
        Ref:
        Eqn. (2.11) of
        张威. 《X波段高功率高效率相对论三轴速调管放大器研究》. 博士学位论文, 国防科技大学, 2022. https://doi.org/10.27052/d.cnki.gzjgu.2019.000388.

        环形电子束在同轴漂移管中的空间电荷限制流


        :param Rout: 外导体的内半径
        :param Rin: 内导体的外半径

        :param rbeam_out: 环形电子束的外半径
        :param rbeam_in: 环形电子束的内半径


        :return:
        """
        rb0 = (rbeam_in+ rbeam_out) / 2
        D= ((rb0**2- rbeam_in **2 )
            + 2 * rbeam_in**2* ln(rbeam_in / rb0)
            + ((rbeam_out **2 - rbeam_in **2 )*(2 * ln(Rout /  rbeam_out) +1)+2 *rbeam_in **2  * ln(rbeam_in / rbeam_out)) / ln(Rout / Rin) * ln(Rin / rb0))


        return - LiuJing2012Doctor.IA * (rbeam_out **2 - rbeam_in **2 ) / D * LiuJing2012Doctor.gamma___(gamma_in)
if __name__ == '__main__':
    print(space_charge_limit_current(500e3, 15e-3, 16e-3, 18e-3))
    print(space_charge_limit_current(500e3, 11.4e-3, 13e-3, 15e-3))
    print(space_charge_limit_current(500e3, 11.4e-3, 11.4e-3 + 2.6e-3, 15e-3))
    print(CoaxialAndHollow.I_space_charge_limit_hollow(500e3, 15e-3, 11.4e-3, ))
    print(CoaxialAndHollow.I_space_charge_limit_coaxial(500e3, 8e-3, 15e-3, 11.4e-3, ))
    print(
        HPMJamesBenford2009.I_SCL_unit_in_A_ring_shaped_beam_inside_hollow_waveguide(common.Ek_to_gamma(500e3, ), 14e-3,
                                                                                     11.4e-3))
    print(
        HPMJamesBenford2009.I_SCL_unit_in_A_cylindrial_beam_inside_hollow_waveguide(common.Ek_to_gamma(500e3, ), 14e-3,
                                                                                    11.4e-3))
    print(j_CL_nonrelativistic_1D(500e3, 15e-3) * 2 * numpy.pi * 12.2e-3 * 1.6e-3)

    plt.figure()
    delta_r = 1e-3  # numpy.linspace(0.5e-3,5e-3,100)
    rin = 11.4e-3
    V = 10 ** numpy.linspace(1, 10, 100)  # numpy.linspace(10, 500e6, 100)
    V = numpy.linspace(10, 1e6, 100)
    plt.plot(V / 1e3, space_charge_limit_current(V, rin, rin + delta_r, 15e-3) / 1e3)
    plt.figure()
    plt.loglog(V, space_charge_limit_current(V, rin, rin + delta_r, 15e-3))
    plt.plot(V, V ** (3 / 2), label='3/2')
    plt.legend()

    plt.figure(figsize=(4, 3), constrained_layout=True)
    rb = 11.4e-3
    Dr = 4e-3
    plt.plot(V / 1e3, CoaxialAndHollow.I_space_charge_limit_coaxial(V, rb - Dr, rb + Dr, rb) / 1e3,
             label="coaxial waveguide")
    plt.plot(V / 1e3, CoaxialAndHollow.I_space_charge_limit_hollow(V, rb + Dr, rb) / 1e3, label="circular waveguide")
    plt.xlabel("$V$ / kV")
    plt.ylabel("$I_{SC}$ / kA")
    plt.legend()

    plt.figure()
    plt.plot(V / 1e3, CoaxialAndHollow.I_space_charge_limit_coaxial(V, rb - Dr, rb + Dr,
                                                                    rb) / CoaxialAndHollow.I_space_charge_limit_hollow(
        V, rb + Dr, rb))

    plt.figure(figsize=(4, 3), constrained_layout=True)
    rb = numpy.linspace(5e-3, 50e-3, 100)
    V = 500e3
    Dr = 2e-3
    delta_r = 1.6e-3
    # plt.plot(rb /1e-3 , CoaxialAndHollow.I_space_charge_limit_coaxial(V, rb-Dr, rb+Dr,rb)/1e3,label = "coaxial waveguide")
    plt.plot(rb / 1e-3, space_charge_limit_current(V, rb - delta_r / 2, rb + delta_r / 2, rb + Dr) / 1e3,
             label="circular waveguide")
    plt.xlabel("$r_{b}$ / mm")
    plt.ylabel("$I_{SC}$ / kA")
    plt.legend()

    plt.figure(figsize=(4, 3), constrained_layout=True)
    rb = 11.4e-3  # numpy.linspace(5e-3, 50e-3,100)
    V = 500e3
    Dr = 2e-3
    delta_r = numpy.linspace(0.1e-3, 5.6e-3, 1010)
    # plt.plot(rb /1e-3 , CoaxialAndHollow.I_space_charge_limit_coaxial(V, rb-Dr, rb+Dr,rb)/1e3,label = "coaxial waveguide")
    plt.plot(delta_r / 1e-3, space_charge_limit_current(V, rb - delta_r / 2, rb + delta_r / 2, rb + Dr) / 1e3,
             label="circular waveguide")
    plt.xlabel("$\Delta r_{b}$ / mm")
    plt.ylabel("$I_{SC}$ / kA")
    plt.legend()
