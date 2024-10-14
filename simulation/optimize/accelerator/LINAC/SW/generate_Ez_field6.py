# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 15:05
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : generate_Ez_field.py
# @Software: PyCharm
"""
采用不对称高斯函数近似近似
"""
import abc
import typing

import cst.results
import matplotlib
import matplotlib.pyplot as plt
import numpy

matplotlib.use('tkagg')
plt.ion()
from _logging import logger


# from generate_Ez_field3 import *

def asymmetrical_Gauss_test(z, sigmaz1, sigmaz2):
    return numpy.exp(- (z / numpy.piecewise(z, [z < 0, z >= 0], [lambda z: sigmaz1, lambda z: sigmaz2])) ** 2 / 2)


class EzBase(abc.ABC):
    @abc.abstractmethod
    def Ez(self, z: numpy.ndarray) -> numpy.ndarray:
        """

        :param z:
        :return:
        """


class Nose(EzBase):
    def __init__(self, sigmaz1, sigmaz2):
        self.sigmaz1, self.sigmaz2 = sigmaz1, sigmaz2

    def Ez(self, z):
        return asymmetrical_Gauss_test(z, self.sigmaz1, self.sigmaz2)

    def __str__(self):
        return "%s(%s,%s)" % (self.__class__.__name__, self.sigmaz1, self.sigmaz2)


class Cell(EzBase):
    def __init__(self, E0, nose1: Nose, nose2: Nose, z_nose1: float, z_nose2: float):
        """
        :param nose1:
        :param nose2:
        :param z_nose1: nose1在Cell坐标系原点的位置
        :param z_nose2:
        """
        self.E0 = E0

        self.nose1 = nose1
        self.nose2 = nose2
        self.z_nose1 = z_nose1
        self.z_nose2 = z_nose2

    def Ez(self, z: numpy.ndarray) -> numpy.ndarray:
        shape = numpy.abs(self.nose1.Ez(z - self.z_nose1) + self.nose2.Ez(z - self.z_nose2))
        mx = shape.max()
        if mx:
            return self.E0 / shape.max() * shape
        return shape

    def __str__(self):
        return "%s(%s,%s,%s,%s,%s)" % (
            self.__class__.__name__, self.E0, self.nose1, self.nose2, self.z_nose1, self.z_nose2)


class CellChain(EzBase):
    @staticmethod
    def remove_bias(cells: typing.List[Cell], z0s: typing.List[float]):
        for i in range(len(cells)):
            z0_ref = (cells[i].z_nose1 + cells[i].z_nose2) / 2
            cells[i].z_nose1 -= z0_ref
            cells[i].z_nose2 -= z0_ref
            z0s[i] -= z0_ref

        return cells, z0s

    def __init__(self, cells: typing.List[typing.Tuple[Cell, float]]):
        # for i in range(len(cells)):
        #     z0_ref = (cells[i][0].z_nose1 +cells[i][0].z_nose2)  / 2
        #     cells[i][0].z_nose1 -= z0_ref
        #     cells[i][0].z_nose2 -= z0_ref
        #     cells[i][1] -= z0_ref
        self.cells = cells

    def Ez(self, z: numpy.ndarray) -> numpy.ndarray:
        ret = numpy.zeros(z.shape)
        for cell in self.cells:
            ret += cell[0].Ez(z - cell[1])
        return ret

    def __str__(self):
        s = ""
        for cell in self.cells:
            s += "(%s,%s),\n" % (cell[0], cell[1])
        s = s[:-2]
        return "%s([\n%s\n])" % (self.__class__.__name__, s)


if __name__ == '__main__':
    proj: cst.results.ProjectFile = cst.results.ProjectFile(
        r'E:\CSTprojects\GeneratorAccelerator\StandingWaveAccelerator_EigenSolver.cst', allow_interactive=True)
    res3d: cst.results.ResultModule = proj.get_3d()
    Ez: cst.results.ResultItem = res3d.get_result_item(r'Tables\1D Results\e_Z (Z)')
    Ezdata = numpy.array(Ez.get_data())
    Ezdata = Ezdata[  # (30.2<=Ezdata[:, 0] ) &
        (Ezdata[:, 0] <= 400)]
    z = Ezdata[:, 0]


    def gen_cellchain(E01,
                      nose_type1_s1, nose_type1_s2,
                      nose_type2_s1, nose_type2_s2,
                      c1_z_nose1, c1_z_nose2,
                      c1_z0):
        # [E01,
        #  nose_type1_s1, nose_type1_s2,
        #  nose_type2_s1, nose_type2_s2,
        #  c1_z_nose1, c1_z_nose2,
        #  c1_z0] =[8.11193900e+07, 2.03377487e+00, 2.77811466e+00, 2.03284611e+00,
        #        1.23273844e+00, 1.82025859e+03, 1.82166142e+03, 1.83504414e+03]
        dz_normal = 16.12
        N_normal = 16
        cells = numpy.array([
            (Cell(-39524049.663693234, Nose(2.1762185438586785, 1.8109482018623393),
                  Nose(1.5217192847042633, 1.3107333260147371), -3.043040147855785e-12, 3.043040147855785e-12),
             13.309777400002096),
            (Cell(88640861.53084418, Nose(1.3794301274516583, 2.113445026634417),
                  Nose(1.9978967002947698, 1.3986733526634776), -1.4991213812449569, 1.4991213812449569),
             22.404430890863264),
            (Cell(-111096163.37660348, Nose(1.62226346714395, 2.886477749368537),
                  Nose(2.88477780604098, 1.6271689656732584), -3.0877333256268624, 3.0877333256268624),
             34.94694711451781),
            *[(Cell(E01 * (-1) ** i, Nose(nose_type1_s1, nose_type1_s2), Nose(nose_type2_s1, nose_type2_s2), c1_z_nose1,
                    c1_z_nose2),
               c1_z0 + dz_normal * i) for i in range(N_normal)],

        ])
        CellChain.remove_bias(cells[:, 0], cells[:, 1])

        return CellChain(cells.tolist())


    popt = [1.1e8,
            1.2, 2.7,
            2.7, 1.2,
            -3, 3,
            50
            ]
    bounds = [
        [-numpy.inf,
         -numpy.inf, -numpy.inf,
         -numpy.inf, -numpy.inf,
         -10, 0,
         0,
         ],
        [numpy.inf,
         numpy.inf, numpy.inf,
         numpy.inf, numpy.inf,
         0, 10,
         200,
         ]
    ]
    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(lambda z, *args: gen_cellchain(*args).Ez(z),
                           Ezdata[:, 0], Ezdata[:, 1],
                           p0=popt,
                           bounds=bounds
                           )

    cellchain = gen_cellchain(*popt)
    cellchain = CellChain([
        (Cell(-39524049.663693234, Nose(2.1762185438586785, 1.8109482018623393),
              Nose(1.5217192847042633, 1.3107333260147371), -3.043040147855785e-12, 3.043040147855785e-12),
         13.309777400002096),
        (Cell(88640861.53084418, Nose(1.3794301274516583, 2.113445026634417),
              Nose(1.9978967002947698, 1.3986733526634776), -1.4991213812449569, 1.4991213812449569),
         22.404430890863264),
        (
        Cell(-111096163.37660348, Nose(1.62226346714395, 2.886477749368537), Nose(2.88477780604098, 1.6271689656732584),
             -3.0877333256268624, 3.0877333256268624), 34.94694711451781),
        (Cell(118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 50.22817883006155),
        (Cell(-118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 66.34817883006156),
        (Cell(118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 82.46817883006156),
        (Cell(-118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 98.58817883006155),
        (Cell(118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 114.70817883006156),
        (Cell(-118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 130.82817883006157),
        (Cell(118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 146.94817883006155),
        (Cell(-118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 163.06817883006156),
        (Cell(118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 179.18817883006156),
        (Cell(-118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 195.30817883006156),
        (Cell(118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 211.42817883006157),
        (Cell(-118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 227.54817883006157),
        (Cell(118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 243.66817883006155),
        (Cell(-118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 259.78817883006155),
        (Cell(118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 275.90817883006156),
        (Cell(-118314535.96548162, Nose(1.6067765569554884, 2.728040289860505),
              Nose(2.730376722127042, 1.624499901646521), -3.017647653092852, 3.017647653092852), 292.02817883006156)
    ])

    plt.figure()
    plt.plot(*Ezdata.T, label='CST')
    plt.plot(z, cellchain.Ez(z), label='fitted')
    plt.legend()
    logger.info('\n%s' % cellchain)
