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
            (exp1(z), z < z_boundary_1),
            (exp2(z), z > z_boundary_2),
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

        self.A = A
        self.B = B

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
    def __init__(self,cells:typing.List[Cell],z_cells:typing.List[float]):
        self.cells = cells
        self.N  = N
        self.z_cells = z_cells # 每个cell的原点在WCS下的位置

    def Ez(self, z:numpy.ndarray):
        E = 0
        for i,  cell in enumerate(self.cells):
            E+= cell.Ez(z- self.z_cells[i])
        return E



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
    N = 10
    cell = Cell(12e-3, -2.5e-3, 2.8e-3, numpy.array([0, 1, 0]), numpy.array([0, 0, 0]))
    Es = cell.Ez(zs)
    plt.plot(zs, Es, '.-', label='$\sum{E_k}$')
    plt.legend()

    df = pandas.read_csv(r'E:\GeneratorAccelerator\Genac\BiPeriodicSWLINAC\BiPeriodicSWEz.txt', sep=r'\s+', skiprows=3,
                         header=None)
    plt.figure()
    plt.plot(*df.values.T)

    flt = (df[0] <= 19.3) & (df[0] > 7)
    new_df = df[flt]
    new_df[0] -= 13.59

    task = CurveFitTask(*new_df.values.T, )
    from simulation.optimize.hpm.hpm import OptimizeJob

    job = OptimizeJob(initializer, lambda: task, )

    job.run(1, False)

    # aaaa
    from _logging import logger


    def get_cell_Ez(z, *args, ax=None):
        L, coupling_z1, coupling_z2 = args[:3]
        A, B = numpy.array(args[3:]).reshape((2, -1))
        logger.info("\n%s\n%s\n%s" % ((L, coupling_z1, coupling_z2), A, B))
        plt.axvspan(-L / 4, L / 4, alpha=0.1)
        plt.axvline(coupling_z1, )
        plt.axvline(coupling_z2, )
        return Cell(L, coupling_z1, coupling_z2, A, B).Ez(z, ax=ax)


    plt.figure()
    plt.plot(*new_df.values.T, label='original')
    zs_ = numpy.linspace(-5, 5, 200)
    # from scipy.interpolate import interp1d

    # Ezs_ = interp1d(*new_df.values.T, )(zs_)
    # plt.plot(zs_, Ezs_, label='interpolated')
    # params, cov = curve_fit(get_cell_Ez, zs_, Ezs_, p0=[18, -3, 3,
    #                                                     *[0, 1.2e8, 0],
    #                                                     *[0] * 3], )
    # params = [15.90064707,
    #           -2.70444942,
    #           2.893379078,
    #           10147268.75,
    #           85739880.58,
    #           106337860.6,
    #           17521797.48,
    #           -102120654.4,
    #           179699618.0,
    #           -8158391.611,
    #           153447887.2
    #           ]
    # params = [20.99337311, -1.295269417, 1.614343679, 3.795307941, -2627855.744, 134623068.5, -19094584.37, 0, 0, 0, 0]
    log_df = pandas.read_csv(r"F:\changeworld\HPMCalc\simulation\optimize\accelerator\LINAC\SW\default.log.csv")
    params = log_df.sort_values(LoggedTask.Colname.score).iloc[-1].values[:11]
    plt.plot(new_df[0], get_cell_Ez(new_df[0], *params), label='fitted')
    get_cell_Ez(zs_, *params, ax=plt.gca())

    plt.legend()
    plt.grid()
