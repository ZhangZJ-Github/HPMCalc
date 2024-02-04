# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 15:05
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : generate_Ez_field.py
# @Software: PyCharm
"""
采用矩形波近似
"""

import cst.results
import matplotlib

matplotlib.use('tkagg')

from generate_Ez_field3 import *
import scipy.constants as C


class RectangularWaveCell(CellBase):
    def __init__(self, Eeven, dEodd_dz, beta, z0, f):
        self.dEodd_dz = dEodd_dz  # 奇函数成分
        self.Eeven = Eeven  # 偶函数成分
        self.beta = beta
        self.L = self.beta * C.c / f / 2
        self.f = f
        self.z0 = z0

    def Ez(self, z: numpy.ndarray):
        return (lambda z: numpy.piecewise(
            z,
            [numpy.abs(z) <= self.L / 2, numpy.abs(z) > self.L / 2],
            [lambda z: self.dEodd_dz * z + self.Eeven, lambda z: 0])
                )(z - self.z0)

    @staticmethod
    def from_true_Ez(Ezdata: numpy.ndarray, f, Delta_phi_z=numpy.pi):
        """
        :param Ezdata:  shape (N,2)
        :return:
        """
        z = Ezdata[:, 0]
        z0 = (z.max() + z.min()) / 2
        beta = (z.max() - z.min()) / (C.c / f / (2 * numpy.pi / Delta_phi_z))
        # beta = min((z.max() - z.min()) / (C.c / f / 2), 1)
        dz = numpy.diff(z)
        Ez = Ezdata[:, 1]
        Eeven = (lambda z: ((Ez * numpy.cos(2 * numpy.pi * f * z / (beta * C.c)))[1:] * dz).sum() / (
                (numpy.cos(2 * numpy.pi * f * z / (beta * C.c)))[1:] * dz).sum())(z - z0)
        dEodd_dz = (lambda z: ((Ez * numpy.sin(2 * numpy.pi * f * z / (beta * C.c)))[1:] * dz).sum() / (
                (z * numpy.sin(2 * numpy.pi * f * z / (beta * C.c)))[1:] * dz).sum())(z - z0)
        return RectangularWaveCell(Eeven,  dEodd_dz, beta, z0, f)


class RectangularCellChain:
    def __init__(self, cells: typing.Iterable[RectangularWaveCell]):
        self.cells = cells

    def Ez(self, z: numpy.ndarray) -> numpy.ndarray:
        Ez = numpy.zeros(z.shape)
        for cell in self.cells:
            Ez += cell.Ez(z)
        return Ez


if __name__ == '__main__':
    plt.ion()
    cst._check_supported_python_version()
    cst.results.print_version_info()
    proj: cst.results.ProjectFile = cst.results.ProjectFile(
        r'E:\CSTprojects\GeneratorAccelerator\StandingWaveAccelerator_EigenSolver.cst', allow_interactive=True)
    res3d: cst.results.ResultModule = proj.get_3d()
    Ez: cst.results.ResultItem = res3d.get_result_item(r'Tables\1D Results\e_Z (Z)')
    Ezdata = numpy.array(Ez.get_data())
    plt.figure()
    plt.plot(*Ezdata.T, label='CST')
    plt.legend()

    flt = (Ezdata[:, 0] > 6) & (Ezdata[:, 0] < 20)

    newEzdata = Ezdata.copy()
    newEzdata[:, 0] *= 1e-3

    # newEzdata[:,0] -= 13.60
    cell = RectangularWaveCell.from_true_Ez(newEzdata[flt], 9.3e9, numpy.pi)
    cellchain = RectangularCellChain(
        [RectangularWaveCell.from_true_Ez(newEzdata[(Ezdata[:, 0] > 0) & (Ezdata[:, 0] < 20)], 9.3e9, numpy.pi),
         RectangularWaveCell.from_true_Ez(newEzdata[(Ezdata[:, 0] > 20) & (Ezdata[:, 0] < 31)], 9.3e9, numpy.pi),
         RectangularWaveCell.from_true_Ez(newEzdata[(Ezdata[:, 0] > 31) & (Ezdata[:, 0] < 46)], 9.3e9, numpy.pi),
         RectangularWaveCell.from_true_Ez(newEzdata[(Ezdata[:, 0] > 46) & (Ezdata[:, 0] < 62)], 9.3e9, numpy.pi),
         RectangularWaveCell.from_true_Ez(newEzdata[(Ezdata[:, 0] > 62) & (Ezdata[:, 0] < 77)], 9.3e9, numpy.pi),
         RectangularWaveCell.from_true_Ez(newEzdata[(Ezdata[:, 0] > 77) & (Ezdata[:, 0] < 93)], 9.3e9, numpy.pi),
         RectangularWaveCell.from_true_Ez(newEzdata[(Ezdata[:, 0] > 93) & (Ezdata[:, 0] < 109)], 9.3e9, numpy.pi),
         RectangularWaveCell.from_true_Ez(newEzdata[(Ezdata[:, 0] > 109) & (Ezdata[:, 0] < 126)], 9.3e9, numpy.pi),
         ])
    plt.figure()
    plt.plot(*newEzdata.T, label='CST')
    zs = newEzdata[:, 0]
    Ez_equivalent = cellchain.Ez(zs)
    plt.plot(newEzdata[:, 0], Ez_equivalent, label='equivalent')

    plt.legend()
    from simulation.task_manager.simulator import df_to_gdf
    # Ez_equivalent = Ezdata[:,1]
    df_to_gdf(pandas.DataFrame({
        'z': zs,
        'Ez': Ez_equivalent / max(Ez_equivalent.max(), -Ez_equivalent.min())
    }), 'Ez1D.gdf')
    from high_capture_efficiency_Ez import GPTEzCaptureSimulator, HighCaptureEffEzTask

    GPTEzCaptureSimulator().run_bat('test_acc.bat')
    res = HighCaptureEffEzTask().get_res('')
    logger.info('\n%s'%res)
