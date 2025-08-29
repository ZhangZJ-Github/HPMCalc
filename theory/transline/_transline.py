# -*- coding: utf-8 -*-
# @Time    : 2025/6/23 23:28
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _transline.py
# @Software: PyCharm
import numpy
import scipy.constants as C
class CoaxialTEMTransline:
    """
    同轴TEM传输线相关计算

    """
    @staticmethod
    def Z0(mu_r,epsilon_r,D, d):
        """
        特性阻抗

        D是外导体（屏蔽层）的外径。
        d是内导体的内径。

        Ref:
        https://www.bchrt.com/tools/coaxial-line-impedance-calculator/

        :return: 特性阻抗
        """
        return (
                (C.mu_0 /C.epsilon_0 * mu_r/epsilon_r)**0.5/(2* numpy.pi)
                *numpy.log(D/d))
class RadialTEMTransline:
    """
    径向TEM传输线相关计算
    """
    @staticmethod
    def Z0(mu_r,epsilon_r,d,r ):
        """
        特性阻抗

        Ref:
        Eq (2) of

        毛重阳, 邹晓兵和王新新. 《小型整体径向传输线的设计与实验》. 强激光与粒子束 28, 期 4 (2016年): 153–57.


        :param d: 导体板之间的距离
        :param r: 径向位置


        :return:
        """
        return (C.mu_0/C.epsilon_0 * mu_r / epsilon_r) **0.5 * d / (2* numpy.pi * r)

