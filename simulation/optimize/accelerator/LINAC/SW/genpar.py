# -*- coding: utf-8 -*-
# @Time    : 2024/1/9 16:38
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : genpar.py
# @Software: PyCharm
import os
from abc import ABC, abstractmethod

import matplotlib
import numpy
import pandas
import scipy.constants as C

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()
import common
from simulation.task_manager.simulator import csv_to_gdf


class Distribution(ABC):
    @abstractmethod
    def generate_gdf(self, nps=int(1e4), filename='electron.gdf', ):
        pass


class RingShapedCathode(Distribution):
    """
    环形阴极
    通过指定rin rout，也可以作为常规的圆形阴极(rin = 0)，点阴极（rin = rout = 0)
    """

    def __init__(self, rin, rout, z_emit, tmin, tmax, I, E0_eV, T, m=C.m_e, q=-C.e):
        self.rin, self.rout, self.z_emit, self.tmin, self.tmax, self.T = rin, rout, z_emit, tmin, tmax, T
        self.I = I
        self.E0_eV = E0_eV
        self.m = m
        self.q = q

    def generate_gdf(self, nps=int(1e4), filename='electron.gdf', ):
        z_emit = self.z_emit
        t0 = self.tmin  # 开始发射粒子的时刻
        dt_emit = self.tmax - self.tmin  # 发射粒子的持续时长

        E0 = self.E0_eV  # 粒子初始动能
        I = self.I  # A
        gamma0 = common.Ek_to_gamma(E0)
        beta0 = common.Ek_to_beta(E0)
        # ts = numpy.linspace(0, tend, nps)
        rmin_square, rmax_square = self.rin, self.rout ** 2
        tmin, tmax = numpy.array((0, dt_emit)) + t0
        rs_square, phis, ts = ((rmin_square, 0, tmin)
                               + numpy.random.random((nps, 3)) *
                               (rmax_square - rmin_square, 2 * numpy.pi, tmax - tmin)).T
        xs, ys = rs_square ** 0.5 * (numpy.cos(phis), numpy.sin(phis))
        particles_df = pandas.DataFrame({'t': ts, 'x': xs, 'y': ys})
        T = self.T  # 粒子温度

        particles_df['m'] = m = C.m_e
        gb_3d = common.beta_to_betagamma(common.thermal_velocity_3D(T, m, nps) / C.c)
        particles_df['GBx'] = 0 + gb_3d[0]
        particles_df['GBy'] = 0 + gb_3d[1]
        particles_df['GBz'] = gamma0 * beta0 + +gb_3d[2]
        particles_df['z'] = z_emit
        particles_df['q'] = self.m
        particles_df['nmacro'] = numpy.abs(I * dt_emit / self.q / nps)
        csv_name = os.path.splitext(filename)[0] + '.txt'
        particles_df.to_csv(csv_name, index=False, sep='\t')
        csv_to_gdf(csv_name)
        return particles_df


if __name__ == '__main__':
    # gptsim = GeneralParticleTracerSim()
    # gptsim.run_bat(r"E:\GeneratorAccelerator\Genac\BiPeriodicSWLINAC\test_acc.bat")
    f = 9.3e9
    RingShapedCathode(0, 0, -10e-3, 0, 4e-9, 100e-3, 50e3, 0, ).generate_gdf()
