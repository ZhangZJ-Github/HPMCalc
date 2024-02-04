# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 15:05
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : generate_Ez_field.py
# @Software: PyCharm
import typing

import matplotlib
import pandas

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy
import sympy
from simulation.optimize.hpm.hpm import OptimizeJob
from _logging import logger
from scipy.optimize import curve_fit

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


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def convergent_exp(x: sympy.Expr):
    """
    :param x:
    :return:
    """
    return sympy.Piecewise([sympy.exp(x), x <= 0], [x + 1, True])


class CellBase:

    @staticmethod
    def base_function_with_attenuation_boundary(
            base_func: typing.Callable[[sympy.Symbol], sympy.Expr],
            z_boundary_1,
            z_boundary_2) -> typing.Callable[
        [sympy.Symbol], sympy.Expr]:
        """
        衰减边界：形如exp(-k z)
        :param base_func:

        :param z_boundary_1:
        :param z_boundary_2:
        :return:
        """
        k1, k2 = sympy.symbols('k1,k2', positive=True)
        A1, A2, z = sympy.symbols('A1,A2,z',  # positive = True
                                  )

        exp1 = lambda z: A1 * sympy.exp(+k1 * (z - z_boundary_1))
        exp2 = lambda z: A2 * sympy.exp(-k2 * (z - z_boundary_2))
        var_to_be_solve = [k1, k2, A1, A2]
        sol = sympy.solve([
            *[f(z_boundary_1) for f in [
                lambda z: exp1(z) - base_func(z),
                lambda z_: sympy.diff(exp1(z) - base_func(z), z).subs({z: z_})]],
            *[f(z_boundary_2) for f in [
                lambda z: exp2(z) - base_func(z),
                lambda z_: sympy.diff(exp2(z) - base_func(z), z).subs({z: z_})]], ],
            var_to_be_solve)
        sol_dict = {var: sol[0][i] for i, var in enumerate(var_to_be_solve)}
        # for k_ in [k1, k2]:
        #     sol_dict[k_] = sympy.Abs(sol_dict[k_])

        return lambda z: sympy.Piecewise(
            (A1 * convergent_exp(+k1 * (z - z_boundary_1)), z < z_boundary_1),
            (A2 * convergent_exp(-k2 * (z - z_boundary_2)), z > z_boundary_2),
            (base_func(z), True)).subs(sol_dict)


z, k, z_coupling1, z_coupling2, L = sympy.symbols('z,k,z_coupling1,z_coupling2,L')
cosexp = sympy.lambdify((z, k, z_coupling1, z_coupling2, L),
                        CellBase.base_function_with_attenuation_boundary(lambda z_:
                                                                         sympy.cos(2 * sympy.pi * k * z_ / L),
                                                                         z_coupling1, z_coupling2)(z), numpy)
sinexp = sympy.lambdify((z, k, z_coupling1, z_coupling2, L),
                        CellBase.base_function_with_attenuation_boundary(lambda z_:
                                                                         sympy.sin(2 * sympy.pi * k * z_ / L),
                                                                         z_coupling1, z_coupling2)(z), numpy)


class Cell(CellBase):

    def __init__(self, L, coupling_z1, coupling_z2, A: numpy.ndarray, B: numpy.ndarray):
        self.coupling_z1 = min(coupling_z1, coupling_z2)
        self.coupling_z2 = max(coupling_z1, coupling_z2)
        self.L = max(L, (coupling_z2 - coupling_z1) * 2)

        self.A = numpy.array(A)
        self.B = numpy.array(B)

        self.N = len(self.A)
        # self.F1k = lambda k:self.F1(L,coupling_z1,coupling_z2,k)
        # self.F2k = lambda k:self.F2(L,coupling_z1,coupling_z2,k)
        # self.F1s =[self.EzGenerator.cosexp.subs({})]# self.get_F1s(L, coupling_z1, coupling_z2, self.N)
        # self.F2s = self.get_F2s(L, coupling_z1, coupling_z2, self.N)
        # self.F2s[0] = lambda z: numpy.zeros(z.shape)

    def Ez(self, z: numpy.ndarray, ax: plt.Axes = None):
        ret = 0
        self.B[0] = 0
        I = numpy.ones(z.shape)
        for k in range(self.N):
            # F1k = self.F1k(k)
            cos: numpy.ndarray = cosexp(z, k, self.coupling_z1, self.coupling_z2, self.L)
            sin = (sinexp(z, k, self.coupling_z1, self.coupling_z2, self.L) if k > 0 else 0.)
            if numpy.any(numpy.isnan(cos) | numpy.isinf(cos) | (cos > 1.1) | (cos < -1.1)):
                self.A[k] = 0
                cos = 0

            if numpy.any(numpy.isnan(sin) | numpy.isinf(sin) | (sin > 1.1) | (sin < -1.1)):
                self.B[k] = 0
                sin = 0
            d_ret = self.A[k] * cos + self.B[k] * sin
            ret += d_ret
            if ax:
                # ax.plot(z, d_ret, label="%d" % k)
                ax.plot(z, self.A[k] * cos * I, label="F1_%d" % k)
                ax.plot(z, self.B[k] * sin * I, label="F2_%d" % k)
        return ret

    def __str__(self):
        return "%s(%s,%s,%s,%s,%s)" % (
            self.__class__.__name__, self.L, self.coupling_z1, self.coupling_z2,
            numpy.array2string(self.A, separator=',', prefix='array'),
            numpy.array2string(self.B, separator=',', prefix='array'))

    @staticmethod
    def symmetric_cell_Ez_base_normalized(z, L, half_dz_acc, n=0):
        """
        关于z = 0对称的腔体
        :param z:
        :param L:
        :param half_dz_acc:
        :param n:
        :return:
        """

        L2 = L * 2
        E1 = lambda z: numpy.cos((2 * n + 1) * numpy.pi / L2 * z)
        dE1_dz = lambda z: -(2 * n + 1) * numpy.pi / L2 * numpy.sin((2 * n + 1) * numpy.pi / L2 * z)
        E2 = lambda z: E1(half_dz_acc) * numpy.exp(+ dE1_dz(half_dz_acc) / (E1(half_dz_acc)) * (z - half_dz_acc))
        # if z <0:return 0
        zabs = numpy.abs(z)
        if zabs < half_dz_acc:
            return E1(zabs)
        else:
            return E2(zabs)

    @staticmethod
    def asymmetric_cell_Ez_base_normalized(z, Dz_cos, half_dz_acc1, half_dz_acc2, n=0):
        """
        z = 0处仍未电场幅值最大位置
        :param z:
        :param Dz_cos:

        :param n:
        :return:
        """


class CellChain:
    def __init__(self,
                 cells: typing.Tuple[
                     typing.Tuple[Cell, float], ...
                 ]):
        cells_and_z = numpy.array(cells)
        self.cells: typing.List[Cell] = cells_and_z[:, 0]
        self.Ncells = len(self.cells)
        self.z_cells = cells_and_z[:, 1]  # 每个cell的原点在WCS下的位置

    def Ez(self, z: numpy.ndarray):
        E = 0
        for i, cell in enumerate(self.cells):
            E += cell.Ez(z - self.z_cells[i])
        return E

    def __str__(self):
        cells_str = ""
        for i, cell in enumerate(self.cells):
            cells_str += '(%s,\t%s),\n' % (str(cell), self.z_cells[i])
        return """%s((\n%s\n))
        """ % (self.__class__.__name__, cells_str)

    def __getitem__(self, item):
        return self.cells[item]


from simulation.task_manager.task import LoggedTask, Initializer

initializer = Initializer('curve_fit_initializer.csv')


class CurveFitTask(LoggedTask):
    def __init__(self, xdata, ydata):
        super(CurveFitTask, self).__init__(initializer=initializer)
        self.xdata = xdata
        self.ydata = ydata
        self.temp_file_to_save_score = 'CurveFitTask.temp.csv'

    def run(self, param_set: dict) -> str:
        cols = self.initializer.init_params_df.columns
        args = [param_set[k] for k in cols]
        L, coupling_z1, coupling_z2 = args[:3]
        A, B = numpy.array(args[3:]).reshape((2, -1))

        A[1] = numpy.sign(A[1]) * max(numpy.abs(args[3:]))

        cell = Cell(L, coupling_z1, coupling_z2, A, B)
        Ez_pred = cell.Ez(self.xdata)
        AB = (*cell.A, *cell.B)
        for i, k in enumerate(cols[3:]):
            param_set[k] = AB[i]
        score = -((self.ydata - Ez_pred) ** 2).mean() ** 0.5
        pandas.DataFrame({
            self.Colname.score: score
        }, index=[0]).to_csv(self.temp_file_to_save_score,
                             index=False)
        return self.temp_file_to_save_score

    def get_res(self, res_path: str) -> dict:
        df = pandas.read_csv(self.temp_file_to_save_score).astype(float)
        return {k: df[k][0] for k in df.columns}

    def evaluate(self, res: dict) -> float:
        return res[self.Colname.score]


if __name__ == '__main__':
    plt.ion()
    L = 15e-3
    zs = numpy.linspace(-L / 2, L / 2, 100)
    # Es = numpy.zeros(zs.shape)

    cell = Cell(12e-3, -2.5e-3, 2.8e-3, numpy.array([0, 1, 0]), numpy.array([0, 0, 0]))
    Es = cell.Ez(zs)
    plt.plot(zs, Es, '.-', label='$\sum{E_k}$')
    plt.legend()

    df = pandas.read_csv(r'E:\GeneratorAccelerator\Genac\BiPeriodicSWLINAC\BiPeriodicSWEz.txt', sep=r'\s+', skiprows=3,
                         header=None)

    flt = (df[0] <= 19.3) & (df[0] > 7)
    new_df = df[flt]
    # new_df[0] -= 13.59

    task = CurveFitTask(*new_df.values.T, )

    job = OptimizeJob(initializer, lambda: task, )


    # job.run(1, False)

    # aaaa

    def get_cell_Ez(z, *args, ax=None):
        L, coupling_z1, coupling_z2 = args[:3]
        A, B = numpy.array(args[3:]).reshape((2, -1))
        logger.info("\n%s\n%s\n%s" % ((L, coupling_z1, coupling_z2), A, B))
        if ax:
            z0 = (coupling_z1 + coupling_z2) / 2
            ax.axvspan(z0 - L / 4, z0 + L / 4, alpha=0.1)
            ax.axvline(coupling_z1, )
            ax.axvline(coupling_z2, )
        return Cell(L, coupling_z1, coupling_z2, A, B).Ez(z, ax=ax)


    # zs_ = numpy.linspace(-5, 5, 200)

    # from scipy.interpolate import interp1d

    # Ezs_ = interp1d(*new_df.values.T, )(zs_)
    # plt.plot(zs_, Ezs_, label='interpolated')
    params, cov = curve_fit(get_cell_Ez, *new_df.values.T,
                            p0=[18, 10, 16,
                                *[0, 1.2e8, 0, 0],
                                *[0, 1.2e8, 0, 0]
                                ], )
    log_df = pandas.read_csv(r"F:\changeworld\HPMCalc\simulation\optimize\accelerator\LINAC\SW\default.log.csv")
    # params = log_df.sort_values(LoggedTask.Colname.score).iloc[-1].values[:11]
    plt.plot(new_df[0], get_cell_Ez(new_df[0], *params), label='fitted')
    get_cell_Ez(new_df[0], *params, ax=plt.gca())

    plt.legend()
    plt.grid()

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, tight_layout=True, figsize=(3, 3))
    N = (len(params) - 3) // 2
    A, B = params[3:].reshape((2, -1))

    axs[0].bar(range(N), A)
    axs[1].bar(range(N), B)
    axs[2].bar(range(N), (A ** 2 + B ** 2) ** 0.5)


    def f(z, *args3):
        cellchain = CellChain((
            (Cell(12.08586909640892, -1.9523060192089308, 1.8263431856963257,
                  [9.11741273e+05, 1.10459803e+08, 0.00000000e+00, 9.08860255e+06],
                  [0., 4304029.4, -11610223.7, 7113797.14]), 13.5768109),
            (Cell(15.3311419, -3.10459576, 2.80805095,
                  [-7.50448234e+05, -1.64417513e+08, 0.00000000e+00, -6.65995909e+05], [0., 0., -3704039.15, 0.]),
             24.8868042),
            (Cell(18.4720593, -3.57991678, 3.41642168,
                  [6.60759432e+05, 2.10158722e+08, 0.00000000e+00, -2.40525880e+07],
                  [0., 0., 724220.283, 6118150.22]), 38.3195474),
            (Cell(22.4135098, -4.57685464, 4.66466162,
                  [-7.03753575e+05, -2.43759463e+08, 0.00000000e+00, 3.79578840e+07],
                  [0., 0., 270146.915, -1736066.83]), 53.6723866),
            (Cell(22.4135098, -4.57685464, 4.66466162,
                  [7.03753575e+05, 2.43759463e+08, -0.00000000e+00, -3.79578840e+07],
                  [-0., -0., -270146.915, 1736066.83]), 69.7984713),
            (Cell(22.4135098, -4.57685464, 4.66466162,
                  [-7.03753575e+05, -2.43759463e+08, 0.00000000e+00, 3.79578840e+07],
                  [0., 0., 270146.915, -1736066.83]), 85.9024438),
            (Cell(22.4135098, -4.57685464, 4.66466162,
                  [7.03753575e+05, 2.43759463e+08, -0.00000000e+00, -3.79578840e+07],
                  [-0., -0., -270146.915, 1736066.83]), 102.006876),
            (Cell(22.4135098, -4.57685464, 4.66466162,
                  [-7.03753575e+05, -2.43759463e+08, 0.00000000e+00, 3.79578840e+07],
                  [0., 0., 270146.915, -1736066.83]), 118.119191),
        ))

        logger.info(cellchain)
        logger.info(cellchain[1])
        return cellchain.Ez(z)


    flt = (df[0] > -20)
    # args, cov = curve_fit(f, *(df[flt].values.T),
    #                       p0=[18,-4,4, *[0,1e8,0,0],*([0]*4),38.7],
    #                       # bounds=[[10, -20, 1, 51, 67, *(8 * [-5e8])], [40, 1, 20, 56, 72, *(8 * [+5e8])]]
    #                       )
    args = [1]
    plt.figure()
    plt.plot(*df.values.T, label='original')
    plt.plot(df[0], f(df[0], *args), label='fitted')
    plt.legend()
