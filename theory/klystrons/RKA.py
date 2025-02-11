# -*- coding: utf-8 -*-
# @Time    : 2025/2/9 13:59
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : RKA.py
# @Software: PyCharm

import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()
import common
import numpy
import scipy.constants as C
import numpy
import scipy.constants as C
import theory.emission.space_charge_limit as scl
from scipy.optimize import fsolve,minimize
from _logging import logger
ln = numpy.log
class YYLau_1990_RKA_TPS:
    """
    主要参考
    Lau, Y.Y., M. Friedman, J. Krall和V. Serlin. 《Relativistic Klystron Amplifiers Driven by Modulated Intense Relativistic Electron Beams》. IEEE Transactions on Plasma Science 18, 期 3 (1990年6月): 553–69. https://doi.org/10.1109/27.55927.
    空心电子束在空心速调管中的相关问题
    """
    @staticmethod
    def Is(rw,rb):
        """
        Eqn. (2)
        :param rw:
        :param rb:
        :return:
        """
        return scl.ZhangWei2022Doctor.IA / 2  / ln(rw/ rb)

    @staticmethod
    def Ic(rw, rb, gamma_inj):
        """
        空间电荷限制流
        Eqn. (3)
        :param rw:
        :param rb:
        :return:
        """
        return  YYLau_1990_RKA_TPS.Is(rw, rb)  *scl.ZhangWei2022Doctor.gamma___(gamma_inj)
    @staticmethod
    def gamma_inj_Eq1_(gamma_0, I0, Is):
        """
        Eqn. (1)

        :param gamma_0:
        :param I0:
        :param Is:
        :param beta_0:
        :return:
        """
        return  gamma_0 + I0/ (Is * (1- 1/gamma_0 **2 ) **0.5)
    @staticmethod
    def gamma_0_Eqn_1(gamma_inj, I0, Is,):
        """
        From Eqn. (1)
        :param gamma_inj:
        :return:
        """
        # return fsolve(lambda gamma_0: YYLau_1990_RKA_TPS.gamma_inj_Eq1_(gamma_0, I0, Is) - gamma_inj,
        #               x0= numpy.array([gamma_inj]), )[0]

        res =  minimize(lambda gamma_0: (YYLau_1990_RKA_TPS.gamma_inj_Eq1_(gamma_0, I0, Is) - gamma_inj) ** 2,x0=  numpy.array([gamma_inj]),bounds=[[1.,gamma_inj]])
        return res.x[0]


    @staticmethod
    def Vth(gamma_inj, I0, Ic, m = C.m_e):
        """
        Eqn. (5)
        2025年2月9日19:19:48评论：暂不确定此公式的含义
        :param gamma_inj:
        :param I0:
        :param Ic:
        :param m:
        :return:
        """
        return m*C.c**2 / C.e * (gamma_inj -(1+ (I0/Ic) ** (2/3) * (gamma_inj**(2/3) -1))**(3/2))
if __name__ == '__main__':
    Ek_inj = 50e3
    I0 = 120
    rb  = 45e-3
    rw= 48e-3
    # Ek_inj = 425e3
    # I0 = 5e3
    # rb  = 1.9e-2
    # rw= 2.4e-2
    gamma_inj = common.Ek_to_gamma(Ek_inj)
    Is = YYLau_1990_RKA_TPS.Is(rw,rb)
    Ic= YYLau_1990_RKA_TPS.Ic(rw,rb,gamma_inj)
    gamma_0 = YYLau_1990_RKA_TPS.gamma_0_Eqn_1(gamma_inj,I0, Is)
    Vth = YYLau_1990_RKA_TPS.Vth(gamma_inj, I0, Ic)
    logger.info(Is)
    logger.info(Ic)
    logger.info((gamma_0 - 1) * 511)
    logger.info(Vth)
