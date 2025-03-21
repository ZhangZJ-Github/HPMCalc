# -*- coding: utf-8 -*-
# @Time    : 2025/3/3 21:43
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : VSWR.py
# @Software: PyCharm
import numpy

from _logging import logger
def power_transmission_efficiency_to_VSWR(power_transmission_efficiency):
    abs_Gamma = (1-power_transmission_efficiency)**0.5 # 电压反射系数
    VSWR = (1+abs_Gamma) / (1- abs_Gamma)
    return VSWR
def Gamma_to_VSWR(Gamma):
    abs_Gamma = numpy.abs(Gamma) # 电压反射系数
    VSWR = (1+abs_Gamma) / (1- abs_Gamma)
    return VSWR

# logger.info(power_transmission_efficiency_to_VSWR(63.035/63.3))
logger.info(Gamma_to_VSWR((-381e-6 / 83e-3)**0.5))