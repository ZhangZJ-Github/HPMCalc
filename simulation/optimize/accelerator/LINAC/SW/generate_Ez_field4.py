# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 15:05
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : generate_Ez_field.py
# @Software: PyCharm
"""
假设耦合腔中电场分布是固定的
"""

import cst.results
import matplotlib
import numpy

matplotlib.use('tkagg')

from generate_Ez_field3 import *


def get_cell_Ez(z, *args, info=False):
    args = [1.11293430e+08, 0.00000000e+00, 9.08860255e+06,
            4.30402940e+06, 2.46703919e+06, 6.14129388e+06,
            -1.65585753e+08, 0.00000000e+00, -6.88394607e+05,
            0.00000000e+00, -2.39555098e+06, 0.00000000e+00,
            2.07058005e+08, 0.00000000e+00, -2.01528712e+07,
            0.00000000e+00, -8.38582691e+05, 6.11815022e+06,
            -2.43730909e+08, 0.00000000e+00, 3.79018870e+07,
            0.00000000e+00, 4.90472588e+04, -1.73606683e+06,

            6.43860468e+07,
            1.21940308e+01, 1.53274266e+01, 1.92477255e+01, 2.24145041e+01,
            -7.69452913e-02, -1.18405994e-01, 5.16428385e-02, 2.21121438e-02,
            1.32450836e+01,
            6.82870171e+00]
    AB_ = [[[0, *args[6 * i:6 * i + 3]], [0, *args[6 * i + 3:6 * i + 3 + 3]]] for i in range(4)]
    AB_ += [list(numpy.array(AB_[-1]) * (-1) ** i) for i in range(1, 5)]
    AB = numpy.array(AB_)
    # AB = numpy.array([
    #     numpy.array(
    #         [[0, 1.07760157e+08, 0.00000000e+00, 9.08860255e+06],
    #          [0., 4304029.4, 15149124.45472458, -3620861.27077519]]),
    #     numpy.array([[ 0, -1.66565974e+08, 0.00000000e+00, 5.91480536e+05],
    #                  [0., 0., -2114746.85415662, 0.]]),
    #     numpy.array([
    #         [ 0, 2.06270470e+08, 0.00000000e+00, -1.80950890e+07],
    #         [0., 0., -3096882.96679501, 6118150.22]]),
    #     *[((-1) ** i *
    #        numpy.array([[ 0, -2.43732818e+08, 0.00000000e+00, 3.79395934e+07],
    #                     [0., 0., -100533.54918419, -1736066.83]])) for i in
    #       range(5)]
    # ])
    Ezmax_coupling = args[24]  # 60e6
    # L = [12.837850613347923, 15.172960345156897, 19.441775000094744, *([22.42171383466281] * 5)]
    L = numpy.array([*args[25:28], *([args[28]] * 5)])
    L[:2] *=2
    # zc1 = [-1.954252491983956,-2.9264465635777133,-3.835622311999435,*([-4.629696622892495]*5)]
    # zc2 =[ 2.1396514036262646,2.857096300881434,3.797384188092191, 4.662991196222726]
    zc2 = numpy.arccos(Ezmax_coupling / numpy.abs(AB[:, 0, 1])) / (2 * numpy.pi) * numpy.array(L)
    zc1 = -zc2 + (list(args[29:33]) + [args[32]] * 4)


    cells = [
        Cell(L[i] , zc1[i], zc2[i], *AB[i])
        for i in range(len(AB))
    ]

    z_cells = [args[33]]  # [13.026595942982336]
    for i in range(1, len(cells)):
        z_cells.append(z_cells[i - 1] - cells[i].coupling_z1 + cells[i - 1].coupling_z2 + args[34]  # 6.81828404
                       )

    cc = CellChain(tuple((cells[i], z_cells[i]) for i in range(len(cells))))
    if info: logger.info(cc)
    return cc.Ez(z)


if __name__ == '__main__':
    plt.ion()
    cst._check_supported_python_version()
    cst.results.print_version_info()
    proj: cst.results.ProjectFile = cst.results.ProjectFile(
        r'E:\CSTprojects\GeneratorAccelerator\StandingWaveAccelerator_EigenSolver.cst', allow_interactive=True)
    res3d: cst.results.ResultModule = proj.get_3d()
    Ez: cst.results.ResultItem = res3d.get_result_item(r'Tables\1D Results\e_Z (Z)')
    Ezdata = numpy.array(Ez.get_data())
    # zs= numpy.linspace(-10, 200, 1000)
    popt = [1.07760157e+08, 0.00000000e+00, 9.08860255e+06,
            4304029.4, 15149124.45472458, -3620861.27077519,
            -1.66565974e+08, 0.00000000e+00, 5.91480536e+05,
            0., -2114746.85415662, 0.,
            2.06270470e+08, 0.00000000e+00, -1.80950890e+07,
            0., -3096882.96679501, 6118150.22,
            -2.43732818e+08, 0.00000000e+00, 3.79395934e+07,
            0., -100533.54918419, -1736066.83,

            60e6,
            12.837850613347923, 15.172960345156897, 19.441775000094744, 22.42171383466281,
            0, 0, 0, 0,
            13.026595942982336,
            6.81828404
            ]
    # _bounds = numpy.array(p0 * 0.9, p0 * 1.1)
    # bounds = _bounds.max()
    # popt, pcov = curve_fit(get_cell_Ez, Ezdata[:, 0], Ezdata[:, 1],
    #                        p0=popt,  # bounds=numpy.array(p0*0.9,p0*1.1)
    #                        )

    Ez_fitted = get_cell_Ez(Ezdata[:, 0], *popt, info=True)
    plt.figure()
    plt.plot(*Ezdata.T, label='CST')
    plt.plot(Ezdata[:, 0], Ez_fitted, label='fitted')
    plt.legend()
