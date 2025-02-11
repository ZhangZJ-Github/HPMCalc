# -*- coding: utf-8 -*-
# @Time    : 2024/12/3 18:06
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : mode_mix.py
# @Software: PyCharm


import enum
import typing

import cst.results
import matplotlib
import numpy
import pandas
import scipy
import skrf
import scipy.constants as C
from scipy.fft import ifft
from skrf.network import Network

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from _logging import logger


class TEModeRectangularWaveguide:
    """
    横截面为a, b的矩形波导中的TE_{m,0,p} mode
    其中，m为长边（长度为a）的极大值个数
    n为短边（长度为b）的极大值个数

    """
    def __init__(self, a,b , n=1,
                 m=0,epsilon_r=  1,
                 mu_r = 1,
                 ):
        self.n = n
        self.m = m
        self.a = a
        self.b = b
        self.kx = self.n * numpy.pi / self.a
        self.ky = self.m * numpy.pi / self.b
        self.T =( self.kx**2 + self.ky**2) **0.5
        self.mu_r = mu_r
        self.epsilon_r = epsilon_r


    def get_kz(self,f):
        return ((2 * numpy.pi * f / (C.c/(self.epsilon_r*self.mu_r)**0.5)) ** 2 - self.T ** 2) ** 0.5
    def get_lambda_z (TE10, f0):
        return 2*numpy.pi / TE10.get_kz(f0)
    def get_u(self     ,x,y ,z         ,kz):
        return numpy.zeros(x.shape                           )
    def get_ux(self     ,x,y ,z         ,kz):
        return numpy.zeros(x.shape                           )
    def get_uy(self     ,x,y ,z         ,kz):
        return numpy.zeros(x.shape                           )
    def get_v (self     ,x,y ,z         ,kz):
        """
        :param x: 长边方向的位置（左边界为0）
        :param y: 短边方向的位置（左边界为0）
        :param z: 传播方向的位置
        :param kz:
        :return:
        """
        return numpy.exp(1j *kz*z ) * numpy.cos(self.kx*x ) * numpy.cos(self.ky*y )

    def get_vx (self     ,x,y ,z         ,kz):
        """
        :param x: 长边方向的位置（左边界为0）
        :param y: 短边方向的位置（左边界为0）
        :param z: 传播方向的位置
        :param kz:
        :return:
        """
        return - self.kx*  numpy.exp(1j *kz*z )  *numpy.sin(self.kx*x ) * numpy.cos(self.ky*y )
    def get_vy (self     ,x,y ,z         ,kz):
        """
        :param x: 长边方向的位置（左边界为0）
        :param y: 短边方向的位置（左边界为0）
        :param z: 传播方向的位置
        :param kz:
        :return:
        """
        return - self.ky*  numpy.exp(1j *kz*z )  *numpy.cos(self.kx*x ) * numpy.sin(self.ky*y )
    def get_f(self,kz):
        return C.c * (self.T**2 + kz**2)**0.5


    def get_EH_normalized(self, x,y,z,f,phi_0 = numpy.pi/2 ):
        """

        :param x: 长边方向的位置（左边界为0）
        :param y: 短边方向的位置（左边界为0）
        :param z: 传播方向的位置
        :return: E,H
        """
        kz    = self.get_kz(f)
        u =self.get_u(x,y,z,kz)
        v= self.get_v(x,y,z,kz)
        ux = self.get_ux(x,y,z,kz)
        uy = self.get_uy(x,y,z,kz)
        vx = self.get_vx(x,y,z,kz)
        vy = self.get_vy(x,y,z,kz)
        omega_mu =  2* numpy.pi * f * C.mu_0 * self.mu_r
        omega_epsilon =  2* numpy.pi * f * C.epsilon_0 * self.epsilon_r
        return [
            # (Ex, Ey, Ez)
            numpy.array([
            -1j*kz * ux-1j *omega_mu * vy,
            -1j * kz * uy - 1j * omega_mu * vx,
            self.T**2 * u
        ]),
            # (Hx, Hy, Hz)
            numpy.array(
            [-1j *kz *vx +1j* omega_epsilon * uy,
             -1j * kz * vy+ 1j * omega_epsilon * ux,
             self.T**2 * v,
             ]
        )]

if __name__ == '__main__':
    plt.ion()
    a = 17e-3
    b= 15e-3
    f0 = 9.3e9


    TE10 =  TEModeRectangularWaveguide(a,b ,1,0 )
    lambda_g = 2*numpy.pi / TE10.get_kz(f0)

    xs,zs = numpy.linspace(0,a, 100),numpy.linspace(0,lambda_g *5, 100)
    X,Z = numpy.meshgrid(xs,zs)
    plt.figure()
    cf = plt.contourf(X / 1e-3,Z / 1e-3,
                      +TE10.get_EH_normalized(X,0 , Z,f0)[0][1]
                      # +0.5*TEModeRectangularWaveguide(a,b ,3,0 ).get_EH_normalized(X,0 , Z,f0)[0][1]
                      # -0.2*TEModeRectangularWaveguide(a,b ,5,0 ).get_EH_normalized(X,0 , Z,f0)[0][1]
                      # +0.1*TEModeRectangularWaveguide(a,b ,10,0 ).get_EH_normalized(X,0 , Z,f0)[0][1]
                 ,
                 cmap =plt.get_cmap('jet'))
    plt.colorbar(cf)
    plt.gca().set_aspect('equal')





