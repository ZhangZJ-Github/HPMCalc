# -*- coding: utf-8 -*-
# @Time    : 2024/1/9 21:07
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : applegate.py
# @Software: PyCharm
"""
Applegate图：横坐标为z，纵坐标为粒子相位
用于反映聚束情况，常见于速调管、驻波加速管设计
Ref: 2021 Pushing the capture limit of thermionic gun linacs Sadiq Setiniyaz PhysRevAccelBeams.24.080401
"""
import enum
import typing

import matplotlib
import numpy
import pandas

import pygpt

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()

from enum import auto


class AppleGateData:
    class Colnames_of_AnalysisResult(enum.Enum):
        z = auto()
        phi = auto()

    def __init__(self, traj_gdf, f, phi0=0.0):
        """
        :param traj_gdf: type of pygpt.gdftomemory(r"traj.gdf")
        :param phi0: gdf中记录的首个时刻的射频相位
        """
        self.traj_gdf = traj_gdf
        self.phi0 = phi0
        self.f = f
        self.analysis_result = self.analysis()

    def analysis(self) -> typing.Dict[int, pandas.DataFrame]:
        dfs: typing.Dict[int, pandas.DataFrame] = {}
        for data in self.traj_gdf:
            if 'ID' in data['p']:
                dfs[int(data['p']['ID'])] = pandas.DataFrame({
                    self.Colnames_of_AnalysisResult.z.name: data['d']['z'],
                    self.Colnames_of_AnalysisResult.phi.name:
                        (data['d']['time'] / (1 / self.f))  # % 1
                        * 2*numpy.pi + self.phi0
                })
        return dfs

    def plot(self, ax: plt.Axes):
        for id in self.analysis_result:
            ax.plot(*self.analysis_result[id].values.T, label=id)


if __name__ == '__main__':
    trajgdf = pygpt.gdftomemory(r"E:\GeneratorAccelerator\Genac\BiPeriodicSWLINAC\traj.gdf")
    agdata = AppleGateData(trajgdf, 9.3e9, 0)
    analysis_data = agdata.analysis()
    plt.figure()
    ax: plt.Axes = plt.gca()
    agdata.plot(ax)
    ax.set_xlim(0, 0.15)
    ax.xaxis.set_major_formatter(lambda z, pos: "%d" % (z / 1e-3))
    ax.set_xlabel('z / mm')
    ax.yaxis.set_major_formatter(lambda phi, pos: "%d" % (phi / numpy.pi * 180))
    ax.set_ylabel(r'particle phase / degree')
