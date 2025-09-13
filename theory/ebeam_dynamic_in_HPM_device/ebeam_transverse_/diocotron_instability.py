# -*- coding: utf-8 -*-
# @Time    : 2025/9/13 17:56
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : diocotron_instability.py
# @Software: PyCharm
"""
带状电子束的成丝不稳定性相关问题
"""

import common


class Duguangxing2011:
    """
    Ref
    杜广星. 《强流相对论带状电子束的产生与传输》.博士学位论文, 国防科学技术大学, 2011年.https: // kns.cnki.net / KCMS / detail / detail.aspx?dbcode = CDFD & dbname = CDFD0911 & filename = 1011073921.            nh.

    """
    @staticmethod
    def Ld_cm(Ek_eV, Jb_kA_per_cm2, Bz_T):
        """
        Ref Eqn. (1.1) of

        杜广星. 《强流相对论带状电子束的产生与传输》.博士学位论文, 国防科学技术大学, 2011年.https: // kns.cnki.net / KCMS / detail / detail.aspx?dbcode = CDFD & dbname = CDFD0911 & filename = 1011073921.            nh.


        通常认为,成丝不稳定性是需要空间和时间逐渐发展的,其不稳定性增长需要的  轴向长度 Ld 可进行定量估计,通常参考以下公式


        :param Ek_eV: 电子束能量, unit in eV
        :param Jb_kA_per_cm2: 平均电流密度，unit in kA/cm^2
        :param Bz_T: 轴向磁场，单位为T

        :return 成丝不稳定性增长所需的轴向长度，单位为cm

        """
        return 8 * common.Ek_to_gamma(Ek_eV) ** 3 * common.Ek_to_beta(Ek_eV) ** 2 * Bz_T / Jb_kA_per_cm2
