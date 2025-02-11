# -*- coding: utf-8 -*-
# @Time    : 2025/1/26 21:59
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : thermal.py
# @Software: PyCharm
import scipy.constants as C
import numpy
from _logging import logger
def J_thermal(T, Phi_M, ):
    """
    热发射电流密度
    检索用关键字：热阴极、电子枪
    Ref:
    刘军乐, 邵文生和于志强. 《基于热阴极的相对论返波管电子枪设计》. 太赫兹科学与电子信息学报 16, 期 1 (2018年): 131–34.
    Eq (1)

    :return:
    """
    A0 = 1.202e6
    return A0*T**2 *numpy.exp(-Phi_M / (C.k *T))

if __name__ == '__main__':
    cm = 1e-2
    logger.info(J_thermal(1300, 1.5*C.eV) *cm**2)