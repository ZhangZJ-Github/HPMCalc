# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 15:05
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : generate_Ez_field.py
# @Software: PyCharm
import abc
import inspect
import typing

import matplotlib
import numpy
import pandas
import scipy.constants as C

from _logging import logger
from simulation.task_manager.simulator import df_to_gdf

np = numpy


def _Fourier_series(x, L, *a):
    """
    傅里叶级数
    :param x:
    :return:
    """
    ret = a[0] * np.cos(np.pi / L * x)
    for deg in range(1, len(a)):
        ret += a[deg] * np.cos((deg + 1) * np.pi / L * x)
    return ret


from scipy.optimize import curve_fit

matplotlib.use('tkagg')
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class EzGeneratorBase(abc.ABC):
    def __init__(self):
        self.Ez: typing.Callable[[numpy.ndarray], numpy.ndarray] = numpy.vectorize(self._Ez)

    @abc.abstractmethod
    def _Ez(self, z):
        return 1.0


class CellEzGenerator(EzGeneratorBase):
    """
    非端部(即，不处于首尾两端)的CAC单元的电场
    这类电场有明确的波节、波腹位置，电场几乎对称
    """

    def __init__(self, beta, Ezmax, f, dz_acc, sigmaz_coupling,# n_sigmaz_coupling: float
                 ):
        super(CellEzGenerator, self).__init__()
        self.beta = beta
        assert (dz_acc >= 0 and f > 0 and
                sigmaz_coupling > 0 and 0 < beta <= 1 #and n_sigmaz_coupling > 0
                )

        self.Ezmax = Ezmax
        self.f = f
        self.L = self.beta * C.c / f / 2  # cell长度
        self.dz_acc = dz_acc
        self.sigmaz_coupling = sigmaz_coupling
        # self.dz_coupling = self.L- self.dz_acc
        # self.Ez: typing.Callable[[numpy.ndarray], numpy.ndarray] = numpy.vectorize(self._Ez)
        self.default_zs = numpy.linspace(-self.L / 2, +self.L / 2, 100)
        self.n_sigmaz_coupling = 10#n_sigmaz_coupling

        self.default_Es = self.Ez(self.default_zs)
        # self.Ez_Fourier_approx: typing.Callable[
        #     [float], float] = self._Fourier_series_approximation()

    def _Ez(self, z):
        """
        :param z: z = 0处为电场高原中央
        :return:
        """
        return self.Ezmax * self._Ez_normalized(z)

    def _Ez_normalized(self, z):
        shape = 0
        if numpy.abs(z) < self.dz_acc / 2:
            shape = 1
        elif numpy.abs(z) < self.L / 2:  # 处于电场上升沿
            shape = ((sigmoid((-numpy.abs(z) + self.L / 2) / self.sigmaz_coupling - self.n_sigmaz_coupling) - sigmoid(
                -self.n_sigmaz_coupling))
                     /
                     (sigmoid((self.L / 2 - self.dz_acc / 2) / self.sigmaz_coupling - self.n_sigmaz_coupling) - sigmoid(
                         -self.n_sigmaz_coupling))
                     )
        return shape

    def _Fourier_series_approximation(self, order=5):
        f = lambda z, *a: _Fourier_series(z, 1 * (self.default_zs.max() - self.default_zs.min()), *a)
        popt, pcov = curve_fit(f, self.default_zs, self.default_Es, [1.0] * order)
        return lambda z: f(z, *popt)


class FirstCellEzGenerator(CellEzGenerator):
    def __init__(self, beta, Ezmax, f, dz_acc, sigmaz_coupling_1,# n_sigmaz_coupling_1,
                 sigmaz_coupling_2, #n_sigmaz_coupling_2
                 ):
        self.sigmaz_coupling_1 = sigmaz_coupling_1
        self.n_sigmaz_coupling_1 = 10 #n_sigmaz_coupling_1
        super(FirstCellEzGenerator, self).__init__(beta, Ezmax, f, dz_acc, sigmaz_coupling_2, #n_sigmaz_coupling_2
                                                   )
        # self.n_sigmaz_coupling = n_sigmaz_coupling_1

    def _Fourier_series_approximation(self, ):
        self.default_zs = numpy.linspace(-self.L * 3, +self.L / 2, 100)
        self.default_Es = self.Ez(self.default_zs)
        return super(FirstCellEzGenerator, self)._Fourier_series_approximation(order=10)

    def _Ez_normalized(self, z):
        shape = super(FirstCellEzGenerator, self)._Ez_normalized(z)
        if z < -self.dz_acc / 2:
            shape = ((sigmoid(
                (-numpy.abs(z) + self.L / 2) / self.sigmaz_coupling_1 - self.n_sigmaz_coupling_1) - sigmoid(
                -self.n_sigmaz_coupling_1) * 0)
                     /
                     (sigmoid(
                         (self.L / 2 - self.dz_acc / 2) / self.sigmaz_coupling_1 - self.n_sigmaz_coupling_1) - sigmoid(
                         -self.n_sigmaz_coupling_1) * 0)
                     )
        return shape


class LastCellEzGenerator(FirstCellEzGenerator):
    def __init__(self, beta, Ezmax, f, dz_acc, sigmaz_coupling_1,  # n_sigmaz_coupling_1,
                 sigmaz_coupling_2,  # n_sigmaz_coupling_2
                 ):
        super(LastCellEzGenerator, self).__init__(beta, Ezmax, f, dz_acc, sigmaz_coupling_2,  # n_sigmaz_coupling_2,
                                                  sigmaz_coupling_1,  # n_sigmaz_coupling_1
                                                  )
        # self.sigmaz_coupling_1 = sigmaz_coupling_1

    def _Ez_normalized(self, z):
        return super(LastCellEzGenerator, self)._Ez_normalized(-z)


class CavChainEzGenerator(EzGeneratorBase):
    def __init__(self, cell_chain: typing.List[CellEzGenerator], z_1st_cell_Ez_peak: float = None, ):
        """

        :param cell_chain:
        :param z_1st_cell_Ez_peak: 1st cell Ez峰值的位置(在WCS中)
        """
        super(CavChainEzGenerator, self).__init__()

        self.cell_chain: typing.List[CellEzGenerator] = cell_chain
        self.z_1st_cell_Ez_peak = z_1st_cell_Ez_peak
        if z_1st_cell_Ez_peak is None: self.z_1st_cell_Ez_peak = 0 + self.cell_chain[0].L
        self.zends: typing.List[float] = [self.z_1st_cell_Ez_peak + self.cell_chain[0].L / 2]  # 每个cell的终点（WCS下）
        self.zmids: typing.List[float] = [self.z_1st_cell_Ez_peak]  # 每个cell的中点（WCS下）
        for i in range(1, len(self.cell_chain) - 1):
            _zend = self.zends[-1]
            self.zends.append(_zend + self.cell_chain[i].L)
            self.zmids.append(_zend + self.cell_chain[i].L / 2)

        self.zmids.append(self.zends[-1] + self.cell_chain[-1].L / 2)
        assert len(self.zmids) == len(self.cell_chain)

    def _Ez(self, z):
        i = numpy.digitize(z, self.zends)
        return self.cell_chain[i]._Ez(z - self.zmids[i])
        # return self.cell_chain[i].Ez_Fourier_approx(z - self.zmids[i])

    @staticmethod
    def from_1D_array(args_to_build_cells: numpy.ndarray, z_1st_cell_Ez_peak: float = None):
        N_params_per_first_cell = len(inspect.signature(FirstCellEzGenerator.__init__).parameters) - 1
        N_params_per_last_cell = len(inspect.signature(LastCellEzGenerator.__init__).parameters) - 1
        N_params_per_normal_cell = len(inspect.signature(CellEzGenerator.__init__).parameters) - 1
        # assert (len(args_to_build_cells) - first_cell_params - last_cell_params) % normal_cell_params == 0
        fc = FirstCellEzGenerator(*args_to_build_cells[:N_params_per_first_cell])
        lc = LastCellEzGenerator(*args_to_build_cells[-N_params_per_last_cell:])
        params_all_normal_cell = (len(args_to_build_cells) - N_params_per_first_cell - N_params_per_last_cell)
        N_normal_cells = (params_all_normal_cell // N_params_per_normal_cell)
        assert params_all_normal_cell % N_params_per_normal_cell == 0
        ncs = [CellEzGenerator(
            *args_to_build_cells[
             N_params_per_first_cell + i * N_params_per_normal_cell:
             N_params_per_first_cell + (i + 1) * N_params_per_normal_cell])
            for i in range(N_normal_cells)]
        return CavChainEzGenerator([fc, *ncs, lc], z_1st_cell_Ez_peak)

    def param_names_1D_to_build_cell_chain(self):
        """
                :return: 构建本对象所需的所有基本参数
                """
        param_names = []
        for i, cell in enumerate(self.cell_chain):
            param_names += ['cells[%d].%s' % (i, param_name) for param_name in
                            (inspect.signature(cell.__init__).parameters)]
        return param_names

    def param_names_1D(self):
        """
        :return: 构建本对象所需的所有基本参数
        """
        return self.param_names_1D_to_build_cell_chain() + list(inspect.signature(self.__init__).parameters)[1:]


if __name__ == '__main__':
    plt.ion()
    arr = numpy.array((0.656535985,0.517152954,9299704018,0.001821529,0.001730694,0.001393273,0.758308151,-0.742903312,9299937096,0.002774948,0.001646869,0.900758366,0.86382351,9299903266,0.004591892,0.002049264,0.997980363,-0.971443468,9299348330,0.006425466,0.0018265,1,0.974428682,9299000000,0.006667525,0.001639259,0.999781993,-0.978029256,9299044824,0.006680035,0.001622029,1,0.982519568,9299000000,0.006426291,0.001820985,1,-0.968968812,9299000000,0.006455776,0.001941463,0.001507043
                       ))
    #
    # arr = numpy.array((0.5,113000000.0,9300000000.0,0.001,0.003,0.003,0.75,-165000000.0,9300000000.0,0.003,0.002,0.9,186000000.0,9300000000.0,0.006,0.001,1.0,-210000000.0,9300000000.0,0.006,0.001,1.0,210000000.0,9300000000.0,0.008,0.001,1.0,-210000000.0,9300000000.0,0.008,0.001,1.0,210000000.0,9300000000.0,0.008,0.001,1.0,-210000000.0,9300000000.0,0.008,0.001,0.001
    #     ))
    f = 9.3e9

    cells = CavChainEzGenerator.from_1D_array(
        arr,
        13.6e-3)
    # asaaaa

    # nonend_Ez_generator = LastCellEzGenerator(0.5, 100e6, 9.3e9, 1e-3, 1e-3, 1e-3
    #                                           )
    # zs = numpy.linspace(-nonend_Ez_generator.L / 1.5, nonend_Ez_generator.L / 1.5, 100)
    # plt.figure()
    # plt.plot(zs, nonend_Ez_generator.Ez(zs))
    # cells = CavChainEzGenerator([
    #     FirstCellEzGenerator(0.5, 113e6, 9.3e9, 1e-3, 3e-3,10., 3e-3,10.),
    #     CellEzGenerator(0.75, -165e6, 9.3e9, 3e-3, 2e-3, 10),
    #     CellEzGenerator(0.9, 186e6, 9.3e9, 6e-3, 1e-3, 10),
    #     CellEzGenerator(1, -210e6, 9.3e9, 6e-3, 1e-3, 10),
    #     CellEzGenerator(1, 210e6, 9.3e9, 8e-3, 1e-3, 10),
    #     CellEzGenerator(1, -210e6, 9.3e9, 8e-3, 1e-3, 10),
    #     CellEzGenerator(1, 210e6, 9.3e9, 8e-3, 1e-3, 10),
    #     LastCellEzGenerator(1, -210e6, 9.3e9, 8e-3, 1e-3,10, 1e-3,10)
    #
    # ], 13.6e-3)
    zs = numpy.linspace(cells.zmids[0] - 2 * cells.cell_chain[0].L, cells.zmids[-1] + cells.cell_chain[-1].L, 1000)
    Ez = cells.Ez(zs)
    # Ez /= Ez.max()
    df = pandas.DataFrame({'z': zs, 'Ez': Ez})
    df_to_gdf(df, 'Ez1D.gdf')

    CST_data = pandas.read_csv(r"E:\GeneratorAccelerator\Genac\BiPeriodicSWLINAC\BiPeriodicSWEz.txt", sep=r'\s+',
                               header=None
                               , skiprows=3, )
    CST_data[0] *= 1e-3
    CST_data[1] /= CST_data[1].max()
    paramname_list = cells.param_names_1D_to_build_cell_chain()
    _mask_without_f = numpy.array(
        [(False if paramname.endswith('.f') else True) for i, paramname in enumerate(paramname_list)])
    _mask_without_beta = numpy.array(
        [(False if paramname.endswith('.beta') else True) for i, paramname in enumerate(paramname_list)])
    _mask_without_dz_acc = numpy.array(
        [(False if paramname.endswith('.dz_acc') else True) for i, paramname in enumerate(paramname_list)])
    _mask_without_sigmaz_coupling = numpy.array(
        [(False if '.sigmaz_coupling' in paramname else True) for i, paramname in enumerate(paramname_list)])
    _mask_without_n_sigmaz_coupling = numpy.array(
        [(False if '.n_sigmaz_coupling' in paramname else True) for i, paramname in enumerate(paramname_list)])

    _bounds = numpy.zeros((_mask_without_f.shape[0], 2))
    _bounds[:, 0] = -numpy.inf
    _bounds[:, 1] = numpy.inf
    _bounds[~_mask_without_f] = (9.299e9, 9.301e9)
    _bounds[~_mask_without_beta] = (0.0001, 1)
    _bounds[~_mask_without_dz_acc] = (0.00001, numpy.inf)
    _bounds[~_mask_without_sigmaz_coupling] = (0.00001, numpy.inf)
    _bounds[~_mask_without_n_sigmaz_coupling] = (9.99, 10.01)
    arr[~_mask_without_n_sigmaz_coupling] = 10

    arr, pcov = curve_fit(lambda z, *arr: CavChainEzGenerator.from_1D_array(arr, 13.6e-3).Ez(z), *CST_data.values.T,
                          arr, bounds=_bounds.T)
    logger.info(cells.param_names_1D_to_build_cell_chain())
    pandas.DataFrame(  # data=arr,columns=cells.param_names_1D_to_build_cell_chain()
        {key: [arr[i]] for i, key in enumerate(cells.param_names_1D_to_build_cell_chain())}
    ).to_csv('initial.temp.csv', index=False)
    # aaaaaa
    CST_data.columns = ['z', 'Ez']

    plt.figure()
    plt.plot(zs, Ez, label='Assumed')
    plt.plot(*CST_data.values.T, label='CST')
    plt.legend()



    # plt.figure()
    # i = 1
    # plt.plot(cells.cell_chain[i].default_zs, cells.cell_chain[i].default_Es, label='Original')
    # plt.plot(cells.cell_chain[i].default_zs,
    #          cells.cell_chain[i].Ez_Fourier_approx(cells.cell_chain[i].default_zs),
    #          label='Fourier approximation')
    # plt.legend()
