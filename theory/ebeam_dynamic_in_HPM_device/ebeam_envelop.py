# -*- coding: utf-8 -*-
# @Time    : 2025/1/22 14:32
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : ebeam_envelop.py
# @Software: PyCharm
import numpy

import common
from _logging import logger
import scipy.constants as C

class DengRujin2024MasterThesis:
    @staticmethod
    def min_guiding_B(I, rb, Dr, gamma_0,
                      m=C.m_e):
        """
        :param Dr: 电子束厚度
        其余参数同文献
        Ref: 邓如金. 《C波段低磁场高效率同轴高功率微波振荡器研究》. 硕士学位论文, 国防科技大学, 2024. https://doi.org/10.27052/d.cnki.gzjgu.2021.001400.
        用于阻止空心电子束横向发散的最小约束磁感应强度B_min

        Eqn. (4.2)
        :return:
        """
        return (I * m / (C.epsilon_0 * C.e * numpy.pi * Dr * rb * C.c * (1 - 1 / gamma_0 ** 2) ** 0.5)) ** 0.5
    @staticmethod
    def min_guiding_B_pencil_beam(I, #rb,
                                  Dr,gamma_0,
                      m = C.m_e):
        """
        :param Dr: 电子束厚度
        其余参数同文献

        Ref: 邓如金. 《C波段低磁场高效率同轴高功率微波振荡器研究》. 硕士学位论文, 国防科技大学, 2024. https://doi.org/10.27052/d.cnki.gzjgu.2021.001400.
        用于阻止空心电子束横向发散的最小约束磁感应强度B_min

        Eqn. (4.2)


        :return:
        """
        return (2* I * m /(C.epsilon_0 *C.e *numpy.pi*Dr **2  * C.c* (1 -1/gamma_0**2 )**0.5))**0.5


min_guiding_B = DengRujin2024MasterThesis.min_guiding_B
min_guiding_B_pencil_beam = DengRujin2024MasterThesis.min_guiding_B_pencil_beam
if __name__ == '__main__':
    logger.info(min_guiding_B(120, 90e-3,2e-3,common.Ek_to_gamma(50e3),C.m_e ))