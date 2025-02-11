# -*- coding: utf-8 -*-
# @Time    : 2025/2/5 22:26
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : confine_e_beam.py
# @Software: PyCharm
"""
计算用于约束电子束的磁场大小
"""

import common
from _logging import logger


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
from theory.ebeam_dynamic_in_HPM_device.ebeam_envelop import  min_guiding_B,C,min_guiding_B_pencil_beam


if __name__ == '__main__':
    logger.info(HPMBenford2007.Bz_to_confine_pencil_ebeam(0.12, common.Ek_to_gamma(50e3), 0.2))
    logger.info(min_guiding_B_pencil_beam(120, #5e-3,
                                          2e-3,common.Ek_to_gamma(50e3),C.m_e ))
    logger.info(min_guiding_B(120, 90e-3,2e-3,common.Ek_to_gamma(50e3),C.m_e ))