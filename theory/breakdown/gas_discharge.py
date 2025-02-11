# -*- coding: utf-8 -*-
# @Time    : 2024/6/20 21:45
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : gas_discharge.py
# @Software: PyCharm
# 气体放电相关
def E_breakdown(p, d_eff,f):
    """
    2014 硕士 高功率高重复频率气体开关研究__杨力云__电子科技大学
    Eq 4-5
    静电击穿阈值
    :param f为场增强因子
    :param p为气体压强，unit in bar (0.1 MPa)
    :param d_eff为电极有效间距，单位 cm
    :return: unit in V/m
    """
    return ((24.6*p +6.7*(p/d_eff)**0.5)/f)*1e5
