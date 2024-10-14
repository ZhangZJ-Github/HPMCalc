# -*- coding: utf-8 -*-
# @Time    : 2024/9/26 22:57
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _signal.py
# @Software: PyCharm

import matplotlib
import numpy

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()

from scipy.optimize import curve_fit

from _logging import logger

logger.info("I am reloaded")
class ExponentialRising:
    def __str__(self):
        return "v_inf = %.2e, tau = %.2e, t0 = %.2e"%(self.v_inf,self.tau,self.t0)
    @staticmethod
    def exponential_rising(t, tau, t0, v_inf):
        """
        :param t:
        :param tau:
        :param t0:
        :param v_inf:
        :return:
        """
        return numpy.piecewise(t, [t > t0, ], [
            (lambda t: v_inf * (1 - numpy.exp(-(t - t0) / tau))),
            0
        ])

    def __init__(self, tau, t0, v_inf):
        self.tau = tau
        self.t0 = t0
        self.v_inf = v_inf

    def f(self, t):
        return self.exponential_rising(t, self.tau, self.t0, self.v_inf)

    @staticmethod
    def fit(known_ts, known_vs, *args, **kwargs):
        (tau, t0, v_inf), cov = curve_fit(ExponentialRising.exponential_rising, known_ts, known_vs, *args, **kwargs)
        return ExponentialRising(tau, t0, v_inf)


if __name__ == '__main__':
    ts = numpy.linspace(0, 40, 2000)
    plt.figure()
    vs = ExponentialRising(5, 2, 0.5).f(ts)
    plt.plot(ts, vs)
    er = ExponentialRising.fit(ts, vs)
