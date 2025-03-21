# -*- coding: utf-8 -*-
# @Time    : 2025/2/5 22:26
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : confine_e_beam.py
# @Software: PyCharm
"""
计算用于约束电子束的磁场大小
"""

import numpy
import scipy.constants as C

import common
from _logging import logger
ln = numpy.log

class HPMBenford2007:
    @staticmethod
    def Bz_to_confine_pencil_ebeam(I_b_unit_in_kA, gamma_b, r_b_unit_in_cm):
        """
        Ref:

        Benford, James, John Allan Swegle, Edl Schamiloglu和John Swegle. High Power Microwaves. 2. ed. Series in Plasma Physics. New York: Taylor & Francis, 2007.

        Eqn. (9.4)

        The magnetic field must be large enough to confine the beam within the drift tube and to keep the beam relatively stiff by preventing too much radial motion. For the pencil beams in high-impedance klystrons, the axial magnetic field required to confine the beam current is10
        :param I_b_unit_in_kA:
        :param gamma_b:
        :param r_b_unit_in_cm:
        :return:
        """
        beta_b = common.gamma_to_beta(gamma_b)
        return 0.34 / (r_b_unit_in_cm) * (I_b_unit_in_kA / (8.5 * beta_b * gamma_b)) ** 0.5


class WangHuida2021:
    @staticmethod
    def omega_b(Ib, rb, Dr, vz=C.c, q=C.e, m=C.m_e):
        """
        环形电子束的等离子体频率
        :param Ib:
        :param rb:
        :param Dr:
        :param q:
        :param m:
        :return:
        """
        return (Ib * q / (2 * numpy.pi * m * C.epsilon_0 * rb * Dr * vz)) ** 0.5

    @staticmethod
    def Omega(B0, gamma=1, m=C.m_e, q=C.e):
        return q * B0 / (gamma * m)

    @staticmethod
    def B0_min_for_balanced_transmiting_ebeam_in_hollow_drift(Ib, rb,Dr, gammab,
                                                              q=C.e, m=C.m_e
                                                              ):
        """
        Ref: 王荟达. 《C波段低磁场高效率相对论返波管研究》. 博士学位论文, 清华大学, 2021. https://doi.org/10.27266/d.cnki.gqhau.2021.000264.
        Eqn. (2.4)下面的一小段

        :return:
        """
        vz = common.gamma_to_beta(gammab) * C.c
        return (2 * Ib * m / (numpy.pi * C.epsilon_0 * q * gammab * vz * rb * Dr)) ** 0.5

class WeiYuanzhang2018:
    """
    Ref:
    魏元璋. 《强流相对论环形电子束的周期磁场引导技术研究》. 硕士学位论文, 电子科技大学, 2018. https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201802&filename=1018990986.nh.



    """
    @staticmethod
    def min_B_to_confine_annular_ebeam_inside_coaxial_drift(Ib,ro,ri,
                                                            v_z,q =C.e ,m = C.m_e):
        """
        Eqn. (3-36)
        :param Ib:
        :param ro:
        :param ri:
        :param v_z:
        :param q:
        :param m:
        :return:
        """
        gamma = common.beta_to_gamma(v_z / C.c)
        return (2 * Ib * m /(numpy.pi * C.epsilon_0* q * gamma * v_z * (ro **2-  ri **2 ))  )**0.5
    @staticmethod
    def B_Brillouin_annular_ebeam_inside_coaxial_drift(Ib,Ro,ro,ri,Ri,
                                                            v_z,q =C.e ,m = C.m_e,):
        """
        Eqn. (4-15)
        :param Ib:
        :param Ro: 同轴外导体的内径
        :param ro:
        :param ri:
        :param Ri: 同轴内导体的外径
        :param v_z:
        :param q:
        :param m:
        :return:
        """
        eta = q/ m
        omega_p =( eta * numpy.abs(Ib/ (v_z * numpy.pi *(ro**2  -ri**2 ))) / C.epsilon_0) **0.5
        gamma_0 = common.beta_to_gamma(v_z / C.c)

        KB =2  /(gamma_0 +1)**(1/4)

        return 2 **0.5 * omega_p/ eta * ((ro ** 2 -ri**2) / ro**2 * (1-ln(Ro / ro)/ln(Ro /Ri))) **0.5 * KB

class LiYangmei2014:
    """
    李阳梅. 《X波段新型同轴双电子束高功率微波源的研究》. 硕士学位论文, 国防科学技术大学, 2014. https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201701&filename=1016921102.nh.

    """
    @staticmethod
    def B_cyclotron_resonance_absorption(Ek_eV,L,k0=0.):
        """
        回旋共振发生时对应的导引磁场

        Eqn. (2.10)
        :param Ek_eV:
        :param L:
        :param k0:
        :return:
        """
        return 2 * common.Ek_to_beta(Ek_eV)*C.c /(C.e/ C.m_e) * common.Ek_to_gamma(Ek_eV)*(numpy.pi / L-k0)




from theory.ebeam_dynamic_in_HPM_device.ebeam_envelop import min_guiding_B, C, min_guiding_B_pencil_beam

if __name__ == '__main__':
    Vb = 50e3
    Ib = 120
    rb = 45e-3
    dr = 3.5e-3
    logger.info(HPMBenford2007.Bz_to_confine_pencil_ebeam(Ib / 1e3, common.Ek_to_gamma(Vb), rb / 1e-2))
    logger.info(min_guiding_B_pencil_beam(Ib,  # 5e-3,
                                          dr, common.Ek_to_gamma(Vb), C.m_e))
    logger.info(min_guiding_B(Ib, rb, dr, common.Ek_to_gamma(Vb), C.m_e))
    logger.info(min_guiding_B(13.8e3, 43e-3, 2e-3, common.Ek_to_gamma(880e3), C.m_e))
    logger.info(WangHuida2021.omega_b(Ib, rb, dr, common.Ek_to_beta(Vb) * C.c, ) / (2 * numpy.pi * 1e9))
    logger.info(WangHuida2021.Omega(0.2, common.Ek_to_gamma(Vb), ) / (2 * numpy.pi * 1e9))
    logger.info(WangHuida2021.B0_min_for_balanced_transmiting_ebeam_in_hollow_drift(Ib,rb,dr,common.Ek_to_gamma(Vb)))
    logger.info(WangHuida2021.B0_min_for_balanced_transmiting_ebeam_in_hollow_drift(13.8e3,43e-3,2e-3,common.Ek_to_gamma(725e3),))
    logger.info(WeiYuanzhang2018.min_B_to_confine_annular_ebeam_inside_coaxial_drift(13.8e3,
                                                                                     43e-3,41e-3,common.Ek_to_beta(725e3) * C.c,))

    logger.info(LiYangmei2014.B_cyclotron_resonance_absorption(50e3,6.6e-3,0))