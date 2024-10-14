# -*- coding: utf-8 -*-
# @Time    : 2024/5/10 20:59
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : doppler_force_model.py
# @Software: PyCharm
# 主要参考：2021-Maximized atom number for a grating magneto-optical trap via machine-learning assisted parameter optimization 贝叶斯优化-光栅磁光阱俘获原子数最大化-Optics Express
from laser_beam import MOField
from transition import *

UsedTransition = Rb87_D2


def _eta(m_L, costheta_i, epsilon_i):
    if m_L == 0:
        return (1 - costheta_i ** 2) / 2
    else:
        return (1 - m_L * epsilon_i * costheta_i) ** 2 / 4


def _delta(Delta, k_i, v, m_L, Babs):
    return Delta - numpy.dot(k_i, v) - m_L * mu_B * Babs / C.hbar


def Doppler_force(position, v: numpy.ndarray, lasers: typing.List[MOField], I_total_interpolator, Babs_interpolator):
    total_force = numpy.zeros(v.shape)
    Babs = Babs_interpolator(*position)
    I_total = I_total_interpolator(*position)
    for laser in lasers:
        to_be_sumed = 0
        I_i = laser.I_interpolator(*position)
        for m_L in [-1, 0, 1]:
            to_be_sumed += _eta(m_L, laser.costheta_interpolator, laser.epsilon) / (
                    1 + I_total / UsedTransition.I_sat + 4 * (
                    _delta(laser.Delta, laser.k, v, m_L, Babs) / UsedTransition.Gamma) ** 2)
        total_force += laser.k * I_i
    return total_force * C.hbar * UsedTransition.Gamma / (2 * UsedTransition.I_sat)
