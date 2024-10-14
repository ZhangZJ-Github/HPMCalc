# -*- coding: utf-8 -*-
# @Time    : 2024/3/19 21:14
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : LINAC.py
# @Software: PyCharm
import numpy
import scipy.constants as C


class BiPeriodicSWLINAC:
    @staticmethod
    def frequency(theta, k, f0):
        """
        Ref: 2013 Development of a C-band 6 MeV standing-wave linear accelerator 清华 Jiahang ShaoPhysRevSTAB.16.090102
        :param theta: 相邻cav之间的平均相移
        :param k: cell间耦合系数
        :param f0: 腔链的本征频率，即pi/2模的频率
        :return:
        """
        return f0 / (1 + k * numpy.cos(theta)) ** 0.5

    @staticmethod
    def cell_to_cell_coupling_coefficient_and_f0(fs:numpy.ndarray,thetas:numpy.ndarray):
        """
        :param fs: shape (N,)
        :param thetas: shape (N,)
        相邻的两个cell之间的耦合系数

        :return: k, f_target
        """
        N = len(fs)
        M =  numpy.matrix([[1 ,- fs[i]**2 * numpy.cos(thetas[i])] for i in range(N)])
        res = ( numpy.linalg.pinv(M) *(fs**2).reshape((N,1)))
        return  res[1,0],res[0,0]**0.5
    @staticmethod
    def beta_optimized_from_shunt_impedance(ZTT, I,L, P_rf):
        return ()



def radial_transline_impedance_TEM(epsilon_r, mu_r, distance_between_plates, r):
    return ((C.mu_0 * mu_r) / (C.epsilon_0 * epsilon_r)) ** 0.5 * distance_between_plates / (2 * numpy.pi * r)
if __name__ == '__main__':
    k, f0 = BiPeriodicSWLINAC.cell_to_cell_coupling_coefficient_and_f0(numpy.array([9.282774 ,9.459932,9.641257  ]),numpy.array([numpy.pi,numpy.pi/2,0]))