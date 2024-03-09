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
import timeit
import typing

import numpy
import fld_parser
import matplotlib.pyplot as plt
import pandas


class ComplexField:
    def __init__(self, t_and_R: typing.Dict[float, numpy.ndarray],
                 f: float,
                 ):
        """
        :param t_and_R: 形如{t0:R0, t1:R1}，其中t为时刻，R为场的实部
        """
        self.f = f
        ts = list(t_and_R.keys())
        self.t0 = ts[0]

        def __unzip_complex(c):
            return numpy.real(c), -numpy.imag(c)

        shape = t_and_R[ts[0]].shape
        res = numpy.matrix([
            [1, *__unzip_complex(self.time_harmonic_factor(self.f,ts[i] - ts[0]))] for i in range(3)
        ]) ** (-1) * numpy.array([t_and_R[ts[i]].reshape(-1) for i in range(3)])

        self.S, self.C0a, self.C0b = (res[i].reshape(shape) for i in range(3))
        self.C0 = self.C0a + 1j * self.C0b

    @staticmethod
    def time_harmonic_factor(f, dt):
        return numpy.exp(1j * (2 * numpy.pi * f * dt))

    def field_at_time(self, t):
        return self.S + self.C0 * self.time_harmonic_factor(self.f, t - self.t0)
        # return self.F0 * self.time_harmonic_factor(f, t - self.t0)


def RZ_to_Cartesian(rs, zs, field_at_RZ_coord_system,
                    XYZ_i: typing.Tuple[numpy.ndarray]):
    """

    @param field_at_RZ_coord_system: shape (Nx,Ny, 场量的维度):
    @param     XYZ_i: 生成方式示范： XYZ_i = numpy.meshgrid(x2g[:,0],x2g[:,0],x1g[0,:],indexing='ij')
    @return:
    """
    from scipy.interpolate import griddata, interpn

    # xy,z = xyz[:,:,0],xyz[0,0,:]
    Xi, Yi, Zi = XYZ_i
    Ri = (Xi ** 2 + Yi ** 2) ** 0.5
    Nx, Ny, Nz = Xi.shape
    RZi = numpy.array((Ri, Zi)).transpose([1, 2, 3, 0]).reshape((Nx * Ny, Nz, 2))
    interpolated_data = interpn((rs, zs), field_at_RZ_coord_system, RZi, bounds_error=False, fill_value=0.).reshape(
        (Nx, Ny, Nz, -1))
    return interpolated_data


if __name__ == '__main__':

    f = 9.46E9
    import filenametool

    et = filenametool.ExtTool.from_filename(
        r"E:\BigFiles\GENAC\GENACX50kV\手动\GenacX50kV_tmplt_20240210_051249_02\在线终止\GenacX50kV_tmplt_20240210_051249_02 周期内密集采样4.m2d")
    fld = fld_parser.FLD(et.get_name_with_ext(et.FileType.fld))
    import geom_parser

    geom = geom_parser.GEOM(fld.filename)
    Ez_title = ' FIELD EZ @OSYS$AREA,SHADE-#2'
    Er_title = ' FIELD ERHO @OSYS$AREA,SHADE-#4'
    # _, __, i0 = fld.get_field_value_by_time(149683e-12, Ez_title)
    i0 = -3
    i1 = i0 + 1
    i2 = i0+2
    t_and_Ez = {}
    t_and_Er = {}
    for i in [i0, i1,i2]:
        t_and_Ez[fld.all_generator[Ez_title][i]['t']] = \
            fld.all_generator[Ez_title][i]['generator'].get_field_values(fld.blocks_groupby_type)[0]
        t_and_Er[fld.all_generator[Er_title][i]['t']] = \
            fld.all_generator[Er_title][i]['generator'].get_field_values(fld.blocks_groupby_type)[0]
    f = 9.4667e9
    cf_Ez = ComplexField(t_and_Ez, f)
    cf_Er = ComplexField(t_and_Er, f)
    x1g_Ez, x2g_Ez = fld.all_generator[Ez_title][0]['generator'].get_x1x2grid(fld.blocks_groupby_type)
    x1g_Er, x2g_Er = fld.all_generator[Er_title][0]['generator'].get_x1x2grid(fld.blocks_groupby_type)

    rs_Ez, zs_Ez = x2g_Ez[:, 0], x1g_Ez[0, :]
    rs_Er, zs_Er = x2g_Er[:, 0], x1g_Er[0, :]

    xs = numpy.linspace(-max(rs_Ez), max(rs_Ez), 100)  # numpy.hstack((-rs_Ez[::-1], rs_Ez))
    zs = numpy.linspace(min(zs_Ez), max(zs_Ez), 200)
    XYZ = numpy.meshgrid(
        xs, xs, zs, indexing='ij')
    t0 = cf_Ez.t0
    Ez_at_cartesian = RZ_to_Cartesian(rs_Ez, zs_Ez, cf_Ez.C0, XYZ)[:, :, :, 0]
    Er_at_cartesian = RZ_to_Cartesian(rs_Er, zs_Er, cf_Er.C0, XYZ)[:, :, :, 0]

    Ex = Er_at_cartesian * XYZ[0] / (XYZ[0] ** 2 + XYZ[1] ** 2) ** 0.5
    Ey = Er_at_cartesian * XYZ[1] / (XYZ[0] ** 2 + XYZ[1] ** 2) ** 0.5

    df = pandas.DataFrame({
        'x': XYZ[0].reshape((-1)), 'y': XYZ[1].reshape((-1)), 'z': XYZ[2].reshape((-1)),
        'Exre': numpy.real(Ex).reshape((-1)), 'Eyre': numpy.real(Ey).reshape((-1)),
        'Ezre': numpy.real(Ez_at_cartesian).reshape((-1)),
        'Exim': numpy.imag(Ex).reshape((-1)), 'Eyim': numpy.imag(Ey).reshape((-1)),
        'Ezim': numpy.imag(Ez_at_cartesian).reshape((-1))
    })
    from simulation.task_manager.simulator import df_to_gdf

    df_to_gdf(df, ".GPT/Ecomplex.gdf", False)

    # 输出组图到文件夹
    plt.ioff()
    vmax = numpy.abs(cf_Ez.S + cf_Ez.C0).max() / 2
    vmin = -vmax
    png_path, zlim, rlim = geom_parser.export_geometry(geom, True)
    img = plt.imread(png_path)
    dt = fld.all_generator[Ez_title][i]['t'] - fld.all_generator[Ez_title][i - 1]['t']
    from scipy.interpolate import interpn

    rbeam = 45e-3
    line = numpy.ones((len(zs), 2))
    line[:, 1] = zs
    line[:, 0] = rbeam
    for t in numpy.arange(cf_Ez.t0, cf_Ez.t0 + dt * 30, dt):
        title = "t = %.4f ns" % (t / 1e-9)
        F = cf_Ez.field_at_time(t)
        # plt.figure()
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5, 5)
                                # constrained_layout=True
                                )
        axs[0].imshow(img, extent=[*zlim, *rlim])
        contourf = axs[0].contourf(x1g_Ez, x2g_Ez, F, numpy.linspace(vmin, vmax, 15), cmap='jet', zorder=-1,
                                   extend='both')

        axs[1].plot(zs, interpn((rs_Ez, zs_Ez), F, line,
                                bounds_error=False, fill_value=0.0), )
        axs[1].set_ylim(vmin, vmax)

        fig.suptitle(title)
        pts, labels = axs[1].get_legend_handles_labels()
        # fig.legend(pts, labels, loc='lower right')

        cbar = fig.colorbar(contourf, ax=axs,  # location=""
                            )

        plt.savefig("Ez/%s.png" % title)
        plt.close()
    # df = pandas.DataFrame({
    #
    # })
