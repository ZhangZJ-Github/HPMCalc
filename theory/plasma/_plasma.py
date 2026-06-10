# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 22:22
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _plasma.py
# @Software: PyCharm
import numpy
import scipy.constants as C

from _logging import logger


def omega_p_component(n0, m=C.m_e, Z=1):
    """
    等离子体某组分的振荡频率

    :param n0:
    :return: unit in Hz
    """
    return Z * C.e * (n0 / (m * C.epsilon_0)) ** 0.5
def attenuation_distance(f, omega_p ):
    """
    定义能使微波的振幅衰减到 1/ e 的那个距离为等离子体的衰减距离

    Ref: 2005 脉冲压缩系统中气体开关对功率增益的影响
    Eq (2)

    :param f:
    :param omega_p:
    :return:
    """
    return C.c /omega_p /(1- (2*numpy.pi *f / omega_p)**2) ** 0.5
class LaserWakefieldAcceleration:
    @staticmethod
    def Ep(omega_p):
        """
        [1] 马跃. 全光逆康普顿散射源偏振特性与CT成像应用研究[D]. 清华大学, 2020.
        1.3.1 基本原理与发展历程

        根据等离子体波破极限,尾波场中加速梯度约为 Ep = mecωp/e

        :param omega_p:
        :return:
        """
        return  C.m_e *C.c *omega_p/C.e

if __name__ == '__main__':
    ne = 1e13 * 1e6
    omega_p = omega_p_component(ne, )
    logger.info(omega_p/(2*numpy.pi) / 1e9)
    logger.info(attenuation_distance(9.3e9,omega_p))

