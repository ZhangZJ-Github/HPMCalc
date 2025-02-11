# -*- coding: utf-8 -*-
# @Time    : 2024/12/4 19:38
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : dispersion.py
# @Software: PyCharm
import matplotlib
import numpy
import scipy.constants as C
from scipy.optimize import fsolve, minimize

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()
epsilons = [1, 1.5, -10 + 1j]


def get_k_square(beta, omega, epsilon):
    return beta ** 2 - epsilon * omega ** 2 / C.c ** 2 +0j


def get_k_to_epsilon(beta, omega, epsilon):
    return get_k_square(beta, omega, epsilon) ** 0.5 / epsilon


a = 50e-9


def func(beta, omega):
    return numpy.exp(-4 * get_k_square(beta, omega, epsilons[0]) ** 0.5 * a) \
           - (get_k_to_epsilon(beta, omega, epsilons[0]) + get_k_to_epsilon(beta, omega, epsilons[1])) / (
                   get_k_to_epsilon(beta, omega, epsilons[0]) - get_k_to_epsilon(beta, omega, epsilons[1])) * \
           (get_k_to_epsilon(beta, omega, epsilons[0]) + get_k_to_epsilon(beta, omega, epsilons[2])) / (
                   get_k_to_epsilon(beta, omega, epsilons[0]) - get_k_to_epsilon(beta, omega, epsilons[2]))


def func_real(beta_real, omega_real, omega_imag):
    res = func(beta_real, omega_real + omega_imag * 1j)
    return [res.real, res.imag]


beta = numpy.linspace(0, 10e7, 1000).astype(complex)
# root = numpy.array([fsolve((lambda omega_real_and_imag: func_real(beta_.real, omega_real_and_imag[0], omega_real_and_imag[1])),
#                x0=[1e+15, 1e+15], ) for beta_ in beta])
#
#
#
# plt.figure()
# plt.plot(beta, root[:,0]
#          )

root = numpy.array([minimize((lambda omega_real_and_imag: -numpy.abs(func(beta_,omega_real_and_imag + 0j))),
               x0=[1e+15,], ).x for beta_ in beta])
plt.figure()
plt.plot(beta, root[:]
         )
omega = numpy.linspace(0,5e16,2000)

plt.figure()
plt.plot(omega,(lambda omega_real_and_imag: -numpy.abs(func(beta[100],omega_real_and_imag + 0j)))(omega) )
aaa

Beta, Omega = numpy.meshgrid(beta, omega)
plt.figure()
vmax = 100
cf = plt.contourf(Beta, Omega,
               (func(Beta, Omega)),
                  levels = numpy.linspace(-vmax,vmax,20),
                  cmap = plt.get_cmap('jet')
)

plt.colorbar(cf)
