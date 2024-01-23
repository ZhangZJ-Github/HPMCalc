# -*- coding: utf-8 -*-
# @Time    : 2024/1/9 21:07
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : applegate.py
# @Software: PyCharm
"""
Applegate图：横坐标为z，纵坐标为粒子感受的电场相位
用于反映聚束情况，常见于速调管、驻波加速管设计
Ref: 2021 Pushing the capture limit of thermionic gun linacs Sadiq Setiniyaz PhysRevAccelBeams.24.080401
"""
import enum
import time
import typing

import matplotlib
import numpy
import pandas
import scipy.constants as C
import shapely
from shapely.geometry import LineString

import pygpt
from enum import auto

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from _logging import logger

plt.ion()



class GPTTraj:
    """
    General Particle Tracer轨迹数据
    """
    class NeededFields_d(enum.Enum):
        """
        traj_gdf['d']中必须存在的字段
        """
        z = auto()
        # phi = auto()
        nmacro = auto()
        time = auto()
        G = auto()
        Bz = auto()  # means beta_z

    class NeededFields_p(enum.Enum):
        """
        traj_gdf['p']中必须存在的字段
        """
        ID = auto()

    def __init__(self, traj_gdf, f, phi0=0.0):
        """
        :param traj_gdf: type of pygpt.gdftomemory(r"traj.gdf")
        :param phi0: gdf中记录的首个时刻的射频相位
        """
        self.traj_gdf = traj_gdf
        self.phi0 = phi0
        self.f = f
        self.dfs = self.analysis()
        self.particle_ids = list(self.dfs.keys())

    def analysis(self) -> typing.Dict[int, pandas.DataFrame]:
        from scipy.interpolate import interp1d
        self._traj_gdf_d_keys = [self.traj_gdf[0]['d'].keys()]
        self._traj_gdf_array = {
            # _id: [self.traj_gdf[_id]['d'][key] for key in self._traj_gdf_d_keys] for _id in
            #                     self.analysis_result
        }

        dfs: typing.Dict[int, pandas.DataFrame] = {}
        self.index_key = 'time'
        self.__column_for_particle_counting = '__column_for_particle_counting'

        self.time_interpolators = {}  # 关于time的插值器

        for data in self.traj_gdf:
            if self.NeededFields_p.ID.name in data['p']:
                _id = int(data['p'][self.NeededFields_p.ID.name])

                _data_dict = {
                    key: data['d'][key] for key in data['d']
                }
                _data_dict.update({
                    'phi':
                        (data['d'][self.NeededFields_d.time.name] / (1 / self.f))  # % 1
                        * 2 * numpy.pi + self.phi0,

                })
                dfs[_id] = pandas.DataFrame(_data_dict)
                _df = dfs[_id]
                _df[self.__column_for_particle_counting] = 1
                self.time_interpolators[_id] = interp1d(_df[self.index_key], _df, axis=0,
                                                        )
        return dfs

    def get_time_interpolator(self, par_id: int, keys: typing.Union[str, typing.List[str]], ):
        return lambda t: pandas.Series(
            self.time_interpolators[par_id](t)
            , index=self.dfs[par_id].columns
        )[keys]

    def plot_AppleGate_diagram(self, ax: plt.Axes):
        for id in self.dfs:
            ax.plot(*self.dfs[id][['z', 'phi']].values.T, label=id)


    def time_across_screen(self, z_screen) -> typing.Dict[int, typing.List[float]]:
        # bounds = []
        across_ts: typing.Dict[int, typing.List[float]] = {}

        for id in self.dfs:
            curve_z_time = LineString(self.dfs[id][[self.NeededFields_d.z.name, self.NeededFields_d.time.name]].values)
            # plt.plot(*curve_phi_z.xy)
            line = LineString(numpy.array(((z_screen,) * 2, curve_z_time.bounds[1::2]), ).T)
            intersections = curve_z_time.intersection(line)
            if isinstance(intersections, shapely.geometry.Point):
                across_ts[id] = [intersections.y]

            elif isinstance(intersections, shapely.geometry.MultiPoint):
                across_ts[id] = [intersection.y for intersection in intersections]

        return across_ts

    def interpolate_at_screen(self, z_screen):
        ts = self.time_across_screen(z_screen)
        columns = self.dfs[list(self.dfs.keys())[0]].columns
        df = pandas.DataFrame(columns=[self.NeededFields_p.ID.name, *self.dfs[list(self.dfs.keys())[0]].columns])
        for parid in ts:
            for t in ts[parid]:
                df.loc[len(df)] = [parid, *self.get_time_interpolator(parid, columns)(t)]
        return df

    def average_at_screen(self, z_screen, key: typing.Union[str, typing.List[str]]):
        interpolated_data_at_screen = self.interpolate_at_screen(z_screen)
        return numpy.average(interpolated_data_at_screen[key],
                             weights=interpolated_data_at_screen[self.NeededFields_d.nmacro.name], axis=0)

    def std_at_screen(self, z_screen, key: typing.Union[str, typing.List[str]]):
        interpolated_data_at_screen = self.interpolate_at_screen(z_screen)

        avg = numpy.average(interpolated_data_at_screen[key],
                            weights=interpolated_data_at_screen[self.NeededFields_d.nmacro.name], axis=0)
        return numpy.average((interpolated_data_at_screen[key] - avg) ** 2,
                             weights=interpolated_data_at_screen[self.NeededFields_d.nmacro.name], axis=0) ** 0.5

    def flux_at_screen(self, z_screen, key: str):
        interpolated_data_at_screen = self.interpolate_at_screen(z_screen)

        pass

        def _flux(_filter):
            return numpy.sum(
                (interpolated_data_at_screen[key]
                 * interpolated_data_at_screen[self.NeededFields_d.nmacro.name]
                 )[_filter])

        flux_positive = _flux(interpolated_data_at_screen[self.NeededFields_d.Bz.name] > 0)
        flux_negative = _flux(interpolated_data_at_screen[self.NeededFields_d.Bz.name] < 0)
        flux_net = flux_positive - flux_negative
        return flux_net, flux_positive, flux_negative

    def captured_cnt(self, z):
        """
        俘获的粒子个数
        :return:
        """
        return self.flux_at_screen(z, self.__column_for_particle_counting)

        # cnt_net = 0
        # cnt_pos = 0  # 沿着+z方向通过探测平面的粒子个数
        # cnt_neg = 0  # 沿着-z方向通过探测平面的粒子个数
        # # bounds = []
        #
        # for id in self.dfs:
        #     curve_phi_z = LineString(self.dfs[id][['z', 'time']].values)
        #
        #     line = LineString(numpy.array(((z,) * 2, curve_phi_z.bounds[1::2]), ).T)
        #     intersections = curve_phi_z.intersection(line)
        #     if isinstance(intersections, shapely.geometry.Point):
        #         cnt_net += 1
        #         cnt_pos += 1
        #     elif isinstance(intersections, shapely.geometry.MultiPoint):
        #         # 偶数个交点：0次
        #         # 奇数个交点：1次
        #         cnt_net += len(intersections) % 2
        #         dcnt_pos = (len(intersections) + 1) // 2
        #         cnt_pos += dcnt_pos
        #         cnt_neg += len(intersections) - dcnt_pos
        # return cnt_net, cnt_pos, cnt_neg

    def capture_efficiency(self,z=0):
        # z = self.dfs[self.particle_ids[0]][self.NeededFields_d.z.name][0]
        captured_net, captured_pos, captured_neg = self.captured_cnt(z)
        return captured_net / captured_pos


if __name__ == '__main__':
    # trajgdf = pygpt.gdftomemory(r"E:\GeneratorAccelerator\Genac\BiPeriodicSWLINAC\traj.gdf")
    trajgdf = pygpt.gdftomemory(r"F:\changeworld\HPMCalc\simulation\optimize\accelerator\LINAC\SW\traj.gdf")
    traj = GPTTraj(trajgdf, 9.3e9, 0)
    # G, = agdata.get_time_interpolator(1, ['G'])(0.1)
    t1 = time.time()
    avgG, avgBz = traj.average_at_screen(0., ['G', 'Bz'])
    stdG, stdBz = traj.std_at_screen(0., ['G', 'Bz'])
    t2 = time.time()

    logger.info("dt = %.2f" % (t2 - t1))

    # agdata.flux_at_screen(0, ['G', 'Bz'])
    E = (avgG - 1) * C.m_e * C.c ** 2 / C.eV

    plt.figure()
    ax: plt.Axes = plt.gca()
    traj.plot_AppleGate_diagram(ax)
    ax.set_xlim(0, 0.15)
    ax.xaxis.set_major_formatter(lambda z, pos: "%d" % (z / 1e-3))
    ax.set_xlabel('z / mm')
    ax.yaxis.set_major_formatter(lambda phi, pos: "%d" % (phi / numpy.pi * 180))
    ax.set_ylabel(r'particle phase / degree')
    z = 0.02  # 0.15
    cnt_net, cnt_pos, cnt_neg = traj.captured_cnt(z)
    capture_eff = traj.capture_efficiency()


    zs = numpy.linspace(0, 0.15, 10)
    t1 = time.time()
    cnt_net_arr, cnt_pos_arr, cnt_neg_arr = numpy.vectorize(traj.captured_cnt)(zs)
    t2 = time.time()

    logger.info("dt = %.2f" % (t2 - t1))
    plt.figure()
    plt.plot(zs, cnt_pos_arr / cnt_pos_arr[0], label="capture eff., positive")
    plt.plot(zs, cnt_net_arr / cnt_pos_arr[0],
             label="capture eff., net (%.2f)" % ((cnt_net_arr / cnt_pos_arr[0])[0]))
    plt.legend()
    plt.gca().xaxis.set_major_formatter(lambda z, pos: "%d" % (z / 1e-3))
    plt.gca().set_xlabel('z / mm')
