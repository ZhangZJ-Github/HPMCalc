# -*- coding: utf-8 -*-
# @Time    : 2025/1/30 17:51
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : spac_charge.py
# @Software: PyCharm
import numpy
import scipy.constants as C
import matplotlib

import common

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from _logging import logger

class HollowAndCoaxial:
    """
    Ref:
    Cao, Yibing, Zhimin Song, Ping Wu, Zhiqiang Fan, Yuchuan Zhang, Yan Teng和Jun Sun. 《Effective Suppression of Pulse Shortening in a Relativistic Backward Wave Oscillator》. Physics of Plasmas 24, 期 3 (2017年3月7日): 033109. https://doi.org/10.1063/1.4977811.
    Eq (1), (2)
    """

    @staticmethod
    def phi_hollow(Ib, vb, rb, r2):
        """
        空心漂移管中+空心电子束构成的系统的电势分布
        参数定义同参考文献
        :param r2: 空心漂移管的内径

        :return:
        """
        return Ib / (2 * numpy.pi * C.epsilon_0 * vb) * numpy.log(r2 / rb)

    @staticmethod
    def phi_coaxial(Ib, vb, rb, r2, r1):
        """
        同轴漂移管中+空心电子束构成的系统的电势分布
        参数定义同参考文献
        :param r1: 漂移管内导体的外径
        :param r2: 漂移管外导体的内径
        即
        r2 > r1

        :return:
        """
        return Ib / (2 * numpy.pi * C.epsilon_0 * vb) * numpy.log(r2 / rb) * numpy.log(rb / r1) / numpy.log(r2 / r1)


if __name__ == '__main__':
    plt.ion()

    # Z, R = numpy.meshgrid(numpy.linspace(0, 1e-3, ))
    r2 = 15e-3
    r1 = 10e-3
    rb = (r1  + r2)/2
    rs = numpy.linspace(0, r2,100)
    Ib = 15e3
    vb = common.Ek_to_beta(50e3) * C.c

    # plt.figure()

    # plt.plot(rs, HollowAndCoaxial.phi_hollow(Ib ,vb, rb, r2),label = 'hollow')
    # plt.plot(rs, HollowAndCoaxial.phi_coaxial(Ib ,vb, rb, r2,r1,),label = 'coaxial')
    logger.info( HollowAndCoaxial.phi_hollow(Ib ,vb, rb, r2))
    logger.info(  HollowAndCoaxial.phi_coaxial(Ib ,vb, rb, r2,r1,))