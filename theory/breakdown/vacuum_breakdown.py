# -*- coding: utf-8 -*-
# @Time    : 2024/10/21 21:16
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : vacuum_breakdown.py
# @Software: PyCharm
import matplotlib
import numpy
matplotlib.use('tkagg')
import matplotlib .pyplot as plt


def Kilpatrick(E_unit_in_MV_m):
    """
    Ref: 2023 小型化微波超构材料速调管的研究__张宣铭__电子科技大学

    真空中随频率变化的最大微波击穿场强通常采用半经验公式 Kilpatrick 准则来预测

    :param E_unit_in_MV_m: 击穿场强，单位MV/m
    :return: f——击穿场强对应的频率，unit in MHz
    """
    return 1.643*E_unit_in_MV_m**2 / numpy.exp(-8.5 / E_unit_in_MV_m)
if __name__ == '__main__':
    plt.ion()
    plt.figure()
    Es = numpy.linspace(10,200, 200)
    plt.plot(Es, Kilpatrick(Es))

