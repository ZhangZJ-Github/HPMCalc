# -*- coding: utf-8 -*-
"""
用于从仅包含实部的单频稳态field map（contour）中还原虚部信息

单频场即，假设t时刻电场表示为以下形式：
F0 = R0 + j I0
则经过时间t后，电场可表示为
F0 exp(-j omega t) = E1 + j I1

@Time ： 2024/2/26 16:55
@Auth ： Zi-Jing Zhang (张子靖)
@File ：EMcomplex.py
@IDE ：PyCharm
"""
import json
import os
import typing

import grd_parser
from scipy.fft import fft, fftshift, fftfreq

import numpy
import fld_parser
import matplotlib.pyplot as plt
import pandas
import filenametool
import geom_parser
import scipy.interpolate
from myutils.time_domain_signal import Signal

import myutils.time_domain_signal
from _logging import logger
from scipy.interpolate import griddata, interpn
from scipy.interpolate import interpn
from simulation.task_manager.simulator import df_to_gdf

from scipy.interpolate.interpolate import RegularGridInterpolator


class FieldNFrame:
    """
    通过N帧场实部信息重构时变场 (N >= 4)。
    假设待重构的场仅包含以下成分：
    1. 静场；
    2. 频率为f、幅值随时间平滑变化的时谐场。
    由2可知，对输入数据的要求是，N帧应均处于已经起振的状态，否则包含的时谐场频率不唯一。
    为了避免求解非线性方程组，充分利用已有的数据，需要用户提供时谐场幅值随时间的变化关系。
    """

    @staticmethod
    def from_frames(grid,
                    frames: typing.Dict[float, numpy.ndarray],
                    f: float,
                    get_Cabs: typing.Callable[[float], float]
                    ):
        """
        @param frames: 形如{t0:R0, t1:R1, t2:R2}，其中t为时刻，R为场的实部(numpy.ndarray)。 对时间顺序无要求。
        @param f:时谐场的频率
        @param get_Cabs: 函数，应支持通过get_Cabs(t)得到一个正比于t时刻的时谐场幅值的量，如输出功率周期平均值的平方根。

        """

        self = FieldNFrame(0, 0, None, None,None)
        self.f = f
        ts = list(frames.keys())
        self.t0 = ts[0]
        __Cabs0 = get_Cabs(self.t0)
        self.get_Cabs_normalized = lambda t: get_Cabs(t) / __Cabs0
        shape = frames[ts[0]].shape
        N = len(ts)

        def _zip_complex(c):
            return numpy.array((numpy.real(c), -numpy.imag(c)))

        res = numpy.linalg.pinv(numpy.matrix([
            [
                1,
                *(self.get_Cabs_normalized(ts[i]) * _zip_complex(self.time_harmonic_factor(self.f, ts[i] - ts[0]))),
                *(self.get_Cabs_normalized(ts[i]) * _zip_complex(self.time_harmonic_factor(2*self.f, ts[i] - ts[0])))
             ] for
            i in
            range(N)
        ])) * numpy.array([frames[ts[i]].reshape(-1) for i in range(N)])

        S, C1a, C1b ,C2a,C2b= (res[i].reshape(shape) for i in range(5))
        rs_Ez, zs_Ez = grid[1][:, 0], grid[0][0, :]
        self.S = RegularGridInterpolator((rs_Ez, zs_Ez), S, bounds_error=False, fill_value=0.0)
        self.C1 = RegularGridInterpolator((rs_Ez, zs_Ez), C1a + 1j * C1b, bounds_error=False, fill_value=0.0)
        self.C2 = RegularGridInterpolator((rs_Ez, zs_Ez), C2a + 1j * C2b, bounds_error=False, fill_value=0.0)

        return self

    def __init__(self, f, t0, S: RegularGridInterpolator, C1: RegularGridInterpolator,C2:RegularGridInterpolator):
        self.f = f
        self.t0 = t0
        self.S = S  # 静场
        self.C1 = C1  # 频率为f的复电场在t0时刻的值（复矩阵）
        self.C2 = C2  # 频率为2f的复电场在t0时刻的值（复矩阵）

        self.get_Cabs_normalized = lambda t: 1.0

    # def zip(self, ):
    #     return numpy.array([self.S, self.C1]).transpose((*(numpy.arange(len(self.S.shape)) + 1), 0))
    #
    # def from_ziped(self, ziped_data: numpy.ndarray):
    #     return FieldNFrame(self.f, self.t0,
    #                        *ziped_data.transpose((len(ziped_data.shape) - 1, *numpy.arange(len(ziped_data.shape) - 1))))

    @staticmethod
    def time_harmonic_factor(f, dt):
        return numpy.exp(1j * (2 * numpy.pi * f * dt))

    def field_at_time(self, t):
        return self.S + self.get_Cabs_normalized(t) * self.C1 * self.time_harmonic_factor(self.f, t - self.t0)+self.C2 * self.time_harmonic_factor(2*self.f, t - self.t0)

    def plot(self, fieldname: str, t,
             vmin, vmax,
             axs: typing.List[plt.Axes] = None):
        field_interpolator, f = {'S': (self.S, 0),
                                 'C1': (self.C1, self.f),
                                 'C2': (self.C2, 2*self.f),
                                 }[fieldname]
        x1grid, x2grid = numpy.meshgrid(self.S.grid[1], self.S.grid[0], )
        cf = axs[0].contourf(x1grid, x2grid,
                             field_interpolator((x2grid, x1grid)) * self.time_harmonic_factor(f, t - self.t0),
                             numpy.linspace(vmin, vmax, 15),
                             cmap='jet', zorder=-1,
                             extend='both')
        axs[0].figure.colorbar(cf, ax=axs)


def RZ_to_Cartesian(rs, zs, field_at_RZ_coord_system,
                    XYZ_i: typing.Tuple[numpy.ndarray]):
    """
    @param rs: shape (Nr,), 已知的点的r坐标
    @param zs: shape (Nz,), 已知的点的z坐标
    @param field_at_RZ_coord_system: shape (Nr,Nz, 场量的维度)，已知的场值
    @param     XYZ_i: 生成方式示范： XYZ_i = numpy.meshgrid(x2g[:,0],x2g[:,0],x1g[0,:],indexing='ij')
    @return:
    """

    # xy,z = xyz[:,:,0],xyz[0,0,:]
    Xi, Yi, Zi = XYZ_i
    Ri = (Xi ** 2 + Yi ** 2) ** 0.5
    Nx, Ny, Nz = Xi.shape
    RZi = numpy.array((Ri, Zi)).transpose([1, 2, 3, 0]).reshape((Nx * Ny, Nz, 2))
    interpolated_data = interpn((rs, zs), field_at_RZ_coord_system, RZi, bounds_error=False, fill_value=0.).reshape(
        (Nx, Ny, Nz, -1))
    return interpolated_data


class MagicContourRebuilder:
    def __init__(self, filename):
        self.et = filenametool.ExtTool.from_filename(filename)
        self.fld = fld_parser.FLD(self.et.get_name_with_ext(self.et.FileType.fld))
        self.geom = geom_parser.GEOM(self.fld.filename)

    @staticmethod
    def get_contour_interpolator(fld: fld_parser.FLD,
                                 contour_title: str,
                                 get_Eout: typing.Callable[[float], float],
                                 # XYZ:typing.Tuple[numpy.ndarray,numpy.ndarray,numpy.ndarray] = None
                                 indexes, f
                                 ):

        t_and_Ez = {}
        # t_and_Er = {}
        for i in indexes:
            t_and_Ez[fld.all_generator[contour_title][i]['t']] = \
                fld.all_generator[contour_title][i]['generator'].get_field_values(fld.blocks_groupby_type)[0]

        grid = fld.x1x2grid[contour_title]
        cf_Ez = FieldNFrame.from_frames(grid, t_and_Ez, f, get_Eout)
        return cf_Ez

    def reconstruct_from_contours(self, obs_power_out_title: str, Ez_title: str = None, Er_title: str = None,
                                  interested_frequency: float = 9e9):
        """
        假设
        1. Ez和Er的采样时刻相差不多
        2. 起振后至少含有8帧

        @param Ez_title:
        @param Er_title:
        @param interested_frequency: 感兴趣的频率，只需大致的值即可
        @return:
        """

        def find_key(keys, check: typing.Callable[[str], bool]):
            for key in keys:
                if check(key): return key

        if Ez_title is None: Ez_title = find_key(self.fld.all_generator.keys(),
                                                 lambda key: key.startswith(' FIELD EZ @'))
        if Er_title is None: Er_title = find_key(self.fld.all_generator.keys(),
                                                 lambda key: key.startswith(' FIELD ERHO @'))

        N = 10
        indexes = list(range(-N, -1, 1))

        # 用于确定频率的时间窗口
        __outpower_time_seq = self.geom.grd.obs[obs_power_out_title]['data']

        tmax = __outpower_time_seq[0].max()
        tmin = min(self.fld.all_generator[Ez_title][indexes[0]]['t'], tmax - 10 / interested_frequency)
        outpower_time_seq = __outpower_time_seq[
            (__outpower_time_seq[0] >= tmin) & (__outpower_time_seq[0] <= tmax)].values

        # f = Signal.peaks(Signal.spectrum(outpower_time_seq))[0, 0] / 2
        f = Signal.frequency_of_monochromatic_data(outpower_time_seq) / 2

        logger.info("起振的频率：%.2e Hz" % f)

        P_out_avg = Signal.periodic_mean(__outpower_time_seq, 1 / f)
        get_Eout = scipy.interpolate.interp1d(P_out_avg[0].values, (-P_out_avg[1].values) ** 0.5, assume_sorted=True,
                                              fill_value='extrapolate', bounds_error=False)
        fld = self.fld
        logger.info('get_contour_interpolator')

        cf_Ez = self.get_contour_interpolator(fld, Ez_title, get_Eout, indexes, f)
        cf_Er = self.get_contour_interpolator(fld, Er_title, get_Eout, indexes, f)
        # self.cf_Ez = cf_Ez
        # self.cf_Er= cf_Er

        # x1g_Ez, x2g_Ez = fld.all_generator[Ez_title][0]['generator'].get_x1x2grid(fld.blocks_groupby_type)

        rs_, zs_ = cf_Ez.S.grid
        xs_uniform = numpy.linspace(-max(rs_), max(rs_), 80)  # numpy.hstack((-rs_Ez[::-1], rs_Ez))
        zs_uniform = numpy.linspace(min(zs_), max(zs_), 200)
        X, Y, Z = numpy.meshgrid(xs_uniform, xs_uniform, zs_uniform, indexing='ij')
        R = (X ** 2 + Y ** 2) ** 0.5

        logger.info('from_ziped')
        # cf_Ez_at_cartesian = cf_Ez.from_ziped(ziped_cf_Ez_interpolator((R, Z), ))
        # cf_Er_at_cartesian = cf_Er.from_ziped(ziped_cf_Er_interpolator((R, Z), ))

        Ex_complex1, Ey_complex1 = cf_Er.C1((R, Z)) / R * numpy.array(
            [X, Y])  # cf_Er_at_cartesian.C1 / R * numpy.array([X, Y])
        logger.info('make Ecomplex1')
        # TODO: 优化代码效率
        Ez_complex1 = cf_Ez.C1((R, Z))
        df = pandas.DataFrame({
            'x': X.reshape((-1)), 'y': Y.reshape((-1)), 'z': Z.reshape((-1)),
            'Exre':  numpy.real(Ex_complex1).reshape((-1)),
            'Eyre':  numpy.real(Ey_complex1).reshape((-1)),
            'Ezre': numpy.real(Ez_complex1).reshape((-1)),
            'Exim':  numpy.imag(Ex_complex1).reshape((-1)), 'Eyim': numpy.imag(Ey_complex1).reshape((-1)),
            'Ezim': numpy.imag(Ez_complex1).reshape((-1))
        })
        df_to_gdf(df, ".GPT/Ecomplex1.gdf", False)
        logger.info('make Ecomplex2')
        Ex_complex2, Ey_complex2 = cf_Er.C2((R, Z)) / R * numpy.array(
            [X, Y])
        Ez_complex2 = cf_Ez.C2((R, Z))
        df = pandas.DataFrame({
            'x': X.reshape((-1)), 'y': Y.reshape((-1)), 'z': Z.reshape((-1)),
            'Exre':  numpy.real(Ex_complex2).reshape((-1)),
            'Eyre':  numpy.real(Ey_complex2).reshape((-1)),
            'Ezre': numpy.real(Ez_complex2).reshape((-1)),
            'Exim':  numpy.imag(Ex_complex2).reshape((-1)), 'Eyim': numpy.imag(Ey_complex2).reshape((-1)),
            'Ezim': numpy.imag(Ez_complex2).reshape((-1))
        })
        df_to_gdf(df, ".GPT/Ecomplex2.gdf", False)
        logger.info('make Estatic')

        Ex_static, Ey_static = cf_Er.S((R, Z)) / R * numpy.array([X, Y])
        Ez_static = cf_Ez.S((R, Z))

        df_static = pandas.DataFrame({
            'x': X.reshape((-1)), 'y': Y.reshape((-1)), 'z': Z.reshape((-1)),
            'Ex': numpy.real(Ex_static).reshape((-1)), 'Ey': numpy.real(Ey_static).reshape((-1)),
            'Ez': numpy.real(Ez_static).reshape((-1)),

        })
        df_to_gdf(df_static, ".GPT/Estatic.gdf", False)
        return f, cf_Ez, cf_Er, Ez_title, Er_title


if __name__ == '__main__':
    rebuilder = MagicContourRebuilder(
        r"E:\BigFiles\GENAC\GENACX50kV\手动\GenacX50kV_tmplt_20240210_051249_02\GenacX50kV_tmplt_20240210_051249_02 Er Ez.m2d")
    f, cf_Ez, cf_Er, Ez_title, Er_title = rebuilder.reconstruct_from_contours(
        ' FIELD_POWER S.DA @COUPLER.PORT,FFT-#5.1',
        interested_frequency=9e9)
    rebuilder.geom.export_geometry(True)
    from total_parser_2 import make_default_result_dir, validateTitle, get_default_result_dir_name

    _res_dir = get_default_result_dir_name(rebuilder.fld)
    contour_Ez_dir = validateTitle(Ez_title)
    field_name_to_plot = 'C2'
    plot_dir_name = '%s/%s/%s' % (_res_dir, contour_Ez_dir, field_name_to_plot)
    os.makedirs(plot_dir_name, exist_ok=True)
    plt.ioff()
    vmax = numpy.max(numpy.abs(cf_Ez.C2.values))
    for t in numpy.linspace(cf_Ez.t0, cf_Ez.t0 + 1 / (cf_Ez.f*2), 20):
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5, 5))
        axs: typing.List[plt.Axes]
        axs[0].imshow(plt.imread(rebuilder.geom.filename), extent=(*rebuilder.geom.x1lim, *rebuilder.geom.x2lim))

        cf_Ez.plot(field_name_to_plot, t, -vmax, vmax, axs)
        plt.savefig('%s/t=%.5fns.png' % (plot_dir_name, t / 1e-9))
        plt.close()
    logger.info('See results: "%s"' % (os.path.abspath(plot_dir_name)))
