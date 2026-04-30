# -*- coding: utf-8 -*-
# @Time    : 2026/4/1 15:06
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : dose_rate.py
# @Software: PyCharm
import numpy
from scipy.interpolate import interp1d


class DoseRateCalculator:
    def __init__(self):
        n_data = numpy.array([
            [3, 6],  # Ek in MeV
            [3, 2.7],  # n
        ]
        )
        self.n_interp = interp1d(*n_data, bounds_error=False, fill_value=tuple(n_data[1]))

    # @staticmethod
    def dose_rate_easy(self,
            Ek_MeV, I_avg_uA):
        """
        Refs: Eqn. (3) in
        Lee Y, Kim S, Kim G, Lee J, Kim I S, Kim J, Shin K Y, Seol Y, Oh T, An N, Lee J, Hwang J, Oh Y, Kang Y. Medical X‐band linear accelerator for high‐precision radiotherapy[J]. Medical Physics, 2021, 48(9): 5327-5342.

        :param Ek_MeV:
        :param I_avg_uA:
        :return: unit in cCy/m
        """
        return 0.067 * I_avg_uA * Ek_MeV **self.n_interp(Ek_MeV)
if __name__ == '__main__':
    from _logging import logger
    logger.info(DoseRateCalculator().dose_rate_easy(10, 10e-6 * 100 * 100e-3 /1e-6))